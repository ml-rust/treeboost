//! AutoBuilder: High-level AutoML interface for TreeBoost
//!
//! Provides a simplified, opinionated API that automatically:
//! 1. Profiles your data (identifies column types, drops useless columns)
//! 2. Selects preprocessing (scalers, encoders, imputers)
//! 3. Engineers features (polynomials, interactions)
//! 4. Tunes hyperparameters (LTT dual-phase, tree params)
//! 5. Trains the optimal model
//!
//! # Design Philosophy
//!
//! AutoBuilder follows the principle of **analyze first, train once**. Unlike
//! AutoML systems that try every combination, TreeBoost analyzes dataset
//! characteristics and makes informed decisions.
//!
//! # Architecture
//!
//! This module is split into three files for maintainability:
//! - `config.rs` - Configuration types (AutoConfig, BuildResult, etc.)
//! - `tuning.rs` - Hyperparameter tuning methods
//! - `builder.rs` - Core AutoBuilder and orchestration logic (this file)
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::{AutoBuilder, AutoConfig};
//! use polars::prelude::*;
//!
//! // Load your data
//! let df = CsvReader::from_path("data.csv")?.finish()?;
//!
//! // Build with defaults (analyze and train)
//! let model = AutoBuilder::new()
//!     .fit(&df, "target_column")?;
//!
//! // Or customize the process
//! let model = AutoBuilder::new()
//!     .with_tuning(TuningLevel::Thorough)
//!     .with_random_validation_split(0.2)
//!     .fit(&df, "target_column")?;
//!
//! // Predict on new data
//! let predictions = model.predict(&test_df)?;
//! ```

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis, PanelDataInfo};
use crate::dataset::feature_extractor::LinearFeatureConfig;
use crate::dataset::{BinnedDataset, DataPipeline};
use crate::defaults::auto as auto_defaults;
use crate::features::{FeaturePlan, SmartFeatureEngine};
use crate::model::config::{
    AutoConfig, BuildPhaseTimes, BuildResult, TargetBoundConfig, TuningLevel,
};
use crate::model::progress::{ProgressCallback, ProgressUpdate, TrainingPhase};
use crate::model::universal::ModeSelection;
use crate::model::{
    BinNumericFeaturesState, CategoryEncoding, CustomFeaturesStep, DropColumnsStep,
    EncodeCategoricalsState, EngineerFeaturesStep, ExtractLinearFeaturesStep, Pipeline,
    PipelineStep, PipelineStepKind, TransformTargetStep,
};
use crate::model::{BoostingMode, UniversalConfig, UniversalModel};
use crate::preprocessing::{ModelType, PreprocessingPlan, SmartPreprocessor};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Import tuning functions
use super::tuning;

/// AutoBuilder: High-level AutoML interface
pub struct AutoBuilder {
    config: AutoConfig,
    /// Optional custom validation DataFrame (for time-series or grouped data)
    validation_df: Option<DataFrame>,
    /// Custom loss function (defaults to MSE)
    loss: Option<Box<dyn crate::loss::LossFunction>>,
}

impl AutoBuilder {
    /// Create a new AutoBuilder with default configuration
    pub fn new() -> Self {
        Self {
            config: AutoConfig::default(),
            validation_df: None,
            loss: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(cfg: AutoConfig) -> Self {
        Self {
            config: cfg,
            loss: None,
            validation_df: None,
        }
    }

    /// Set tuning level
    pub fn with_tuning(mut self, level: TuningLevel) -> Self {
        self.config.tuning_level = level;
        self
    }

    /// Set validation split ratio for **random** train/validation split.
    ///
    /// **WARNING**: Only use this for cross-sectional (i.i.d.) data where rows are independent.
    /// For time-series or panel data, use `Self::with_presplit_validation` instead to avoid data leakage.
    ///
    /// The library will randomly split your data into:
    /// - Training set: (1 - ratio) * num_rows
    /// - Validation set: ratio * num_rows
    ///
    /// # Arguments
    ///
    /// * `ratio` - Fraction of data to use for validation (typically 0.2 for 80/20 split)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Cross-sectional data (rows are independent)
    /// let model = AutoBuilder::new()
    ///     .with_random_validation_split(0.2)  // Random 80/20 split
    ///     .fit(&df, "target")?;
    /// ```
    ///
    /// # See Also
    ///
    /// * `Self::with_presplit_validation` - For time-series or panel data
    pub fn with_random_validation_split(mut self, ratio: f32) -> Self {
        self.config.val_ratio = ratio;
        self
    }

    /// Set feature engineering mode
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoBuilder, FeatureEngineeringMode};
    /// use treeboost::features::SmartFeatureConfig;
    ///
    /// // Disable feature engineering
    /// let builder = AutoBuilder::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::None);
    ///
    /// // Use aggressive features
    /// let builder = AutoBuilder::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::Aggressive);
    ///
    /// // Custom configuration
    /// let builder = AutoBuilder::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::Custom(
    ///         SmartFeatureConfig::default()
    ///             .with_enable_polynomial(false)
    ///             .with_top_n_interactions(10)
    ///     ));
    /// ```
    pub fn with_feature_engineering(
        mut self,
        mode: crate::model::config::FeatureEngineeringMode,
    ) -> Self {
        self.config.feature_engineering = mode;
        self
    }

    /// Set preprocessing mode
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoBuilder, PreprocessingMode};
    ///
    /// // No preprocessing
    /// let builder = AutoBuilder::new()
    ///     .with_preprocessing(PreprocessingMode::None);
    ///
    /// // Strict preprocessing
    /// let builder = AutoBuilder::new()
    ///     .with_preprocessing(PreprocessingMode::Strict);
    /// ```
    pub fn with_preprocessing(mut self, mode: crate::model::config::PreprocessingMode) -> Self {
        self.config.preprocessing = mode;
        self
    }

    /// Use pre-split validation data for time-series, panel, or grouped data.
    ///
    /// **Use this method when** random splits would cause data leakage:
    /// - **Time-series data**: Validate on future dates (date-based split)
    /// - **Panel data**: Validate on held-out groups (no group leakage)
    /// - **Cross-validation**: Custom fold splits
    /// - **Any non-i.i.d. data**: Where rows have dependencies
    ///
    /// When you provide pre-split validation data, the library will:
    /// 1. Disable internal random splitting (sets `val_ratio = 0.0` automatically)
    /// 2. Use your training DataFrame for training
    /// 3. Use your validation DataFrame for hyperparameter tuning and early stopping
    ///
    /// # Arguments
    ///
    /// * `validation_df` - Pre-split validation DataFrame with same schema as training data
    ///
    /// # Example: Time-Series (Date-Based Split)
    ///
    /// ```ignore
    /// // Split by date for time-series forecasting
    /// let train_df = df.filter(col("date").lt(lit("2024-01-01")))?;
    /// let val_df = df.filter(col("date").gte(lit("2024-01-01")))?;
    ///
    /// let model = AutoBuilder::new()
    ///     .with_presplit_validation(val_df)  // Correct: no leakage!
    ///     .fit(&train_df, "target")?;
    /// ```
    ///
    /// # Example: Panel Data (Group-Based Split)
    ///
    /// ```ignore
    /// // Hold out specific stocks for validation
    /// let train_df = df.filter(col("stock_id").is_in(train_stocks))?;
    /// let val_df = df.filter(col("stock_id").is_in(val_stocks))?;
    ///
    /// let model = AutoBuilder::new()
    ///     .with_presplit_validation(val_df)
    ///     .fit(&train_df, "target")?;
    /// ```
    ///
    /// # See Also
    ///
    /// * `Self::with_random_validation_split` - For cross-sectional (i.i.d.) data
    pub fn with_presplit_validation(mut self, validation_df: DataFrame) -> Self {
        // Disable internal validation split when custom data is provided
        self.config.val_ratio = 0.0;
        self.validation_df = Some(validation_df);
        self
    }

    /// Force a specific mode
    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.config.mode_selection = ModeSelection::Fixed(mode);
        self
    }

    /// Set ensemble mode (PureTree only)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoBuilder, EnsembleMode, AutoEnsembleMethod, AutoEnsembleConfig};
    ///
    /// // No ensemble
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::Disabled);
    ///
    /// // Default ensemble
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::Default);
    ///
    /// // With specific method
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::WithMethod(AutoEnsembleMethod::SimpleAverage));
    ///
    /// // Custom configuration
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::Custom(AutoEnsembleConfig::new()));
    /// ```
    pub fn with_ensemble(mut self, mode: crate::model::config::EnsembleMode) -> Self {
        self.config.ensemble = mode;
        self
    }

    /// Set custom loss function (defaults to MSE)
    pub fn with_loss(mut self, loss: impl crate::loss::LossFunction + 'static) -> Self {
        self.loss = Some(Box::new(loss));
        self
    }

    /// Enable verbose output (light mode: everything except preprocessing details)
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose.into();
        self
    }

    /// Set granular verbosity control
    pub fn with_verbosity(mut self, verbosity: crate::model::config::Verbosity) -> Self {
        self.config.verbose = verbosity;
        self
    }

    /// Set target bound configuration for bounded regression.
    ///
    /// Controls how target bounds are determined and what transformation to apply.
    /// See [`TargetBoundConfig`] for available modes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::model::{AutoBuilder, TargetBoundConfig};
    ///
    /// // Exam scores: known bounds [0, 100] with Logit transform
    /// let model = AutoBuilder::new()
    ///     .with_target_bound_config(TargetBoundConfig::Logit { min: 0.0, max: 100.0 })
    ///     .fit(&train_df, "exam_score")?;
    ///
    /// // Simple clamp using empirical min/max from data
    /// let model = AutoBuilder::new()
    ///     .with_target_bound_config(TargetBoundConfig::ClampEmpirical)
    ///     .fit(&train_df, "target")?;
    /// ```
    pub fn with_target_bound_config(mut self, config: TargetBoundConfig) -> Self {
        self.config.target_bound_config = config;
        self
    }

    /// Set explicit target bounds for bounded regression (deprecated)
    ///
    /// **Deprecated:** Use [`with_target_bound_config`] instead for more control.
    #[deprecated(since = "0.2.0", note = "Use with_target_bound_config() instead")]
    #[allow(deprecated)]
    pub fn with_target_bounds(mut self, min: f32, max: f32) -> Self {
        self.config = self.config.with_target_bounds(min, max);
        self
    }

    /// Set time budget for training
    ///
    /// AutoBuilder will adapt its behavior to fit within this time limit:
    /// - Skips feature engineering if time is tight (< 20s remaining)
    /// - Reduces tuning intensity if time is limited (< 30s → Quick tuning)
    /// - Skips tuning entirely if very low on time (< 10s)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    /// let builder = AutoBuilder::new()
    ///     .with_time_budget(Duration::from_secs(60));  // 1 minute max
    /// ```
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.config.time_budget = Some(budget);
        self
    }

    /// Set progress callback for tracking training phases
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::ConsoleProgress;
    /// use std::sync::Arc;
    ///
    /// let builder = AutoBuilder::new()
    ///     .with_progress_callback(Arc::new(ConsoleProgress::detailed()));
    /// ```
    pub fn with_progress_callback(mut self, callback: Arc<dyn ProgressCallback>) -> Self {
        self.config.progress_callback = callback;
        self
    }

    /// Set custom feature extractor configuration for LinearThenTree mode
    ///
    /// This controls which features are extracted for the linear component.
    /// By default, all features are included (no exclusions).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::dataset::feature_extractor::LinearFeatureConfig;
    ///
    /// let feature_config = LinearFeatureConfig::default()
    ///     .with_exclude_categorical(true)  // Exclude categorical features from linear model
    ///     .with_exclude_id(true);          // Exclude ID columns
    ///
    /// let builder = AutoBuilder::new()
    ///     .with_linear_feature_config(feature_config);
    /// ```
    pub fn with_linear_feature_config(mut self, config: LinearFeatureConfig) -> Self {
        self.config.linear_feature_config = config;
        self
    }

    /// Fit the model on a Polars DataFrame
    ///
    /// This is the main entry point. It:
    /// 1. Profiles the DataFrame
    /// 2. Determines preprocessing strategy
    /// 3. Plans feature engineering
    /// 4. Analyzes for mode selection
    /// 5. Tunes hyperparameters
    /// 6. Trains the final model
    pub fn fit(&self, df: &DataFrame, target_col: &str) -> Result<BuildResult> {
        let start = Instant::now();
        let mut phase_times = BuildPhaseTimes::default();

        // Adapt configuration based on time budget
        let mut adapted_config = self.config.clone();
        if let Some(budget) = self.config.time_budget {
            if self.config.verbose.enabled() {
                println!("AutoBuilder: Time budget set to {:?}", budget);
            }
        }

        if self.config.verbose.enabled() {
            println!("AutoBuilder: Starting build process...");
        }

        // === Phase 1: Column Profiling ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Profiling,
            progress_pct: TrainingPhase::Profiling.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("Analyzing {} columns", df.width())),
        });

        let phase_start = Instant::now();
        let profile = self.profile_dataframe(df, target_col)?;
        phase_times.profiling = phase_start.elapsed();

        if self.config.verbose.enabled() {
            println!(
                "  [Profile] {} columns analyzed, {} marked for dropping, task: {:?}",
                profile.columns.len(),
                profile.drop_columns.len(),
                profile.task_type
            );
            if !profile.drop_columns.is_empty() {
                println!("  [Profile] Columns to drop:");
                for col in &profile.drop_columns {
                    println!("    - {} ({})", col.name, col.reason);
                }
            }
        }

        // === Phase 2: Determine Model Type and Preprocessing ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Preprocessing,
            progress_pct: TrainingPhase::Preprocessing.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!(
                "{} columns retained",
                profile
                    .columns
                    .len()
                    .saturating_sub(profile.drop_columns.len())
            )),
        });

        let phase_start = Instant::now();
        let (model_type, preprocessing_plan) = self.plan_preprocessing(&profile)?;
        phase_times.preprocessing = phase_start.elapsed();

        if self.config.verbose.enabled() {
            println!(
                "  [Preprocess] Model type: {:?}, {} steps planned",
                model_type,
                preprocessing_plan.steps.len()
            );

            // WHITEBOX: Show exact preprocessing steps
            if self.config.verbose.preprocessing && !preprocessing_plan.steps.is_empty() {
                println!("  [Preprocess] Steps:");
                for (i, step) in preprocessing_plan.steps.iter().enumerate() {
                    println!("    {}. {:?}", i + 1, step);
                }
            }
        }

        // === Phase 3: Feature Engineering ===
        // Adapt feature engineering based on remaining time
        let mut skip_features = !adapted_config.feature_engineering.is_enabled();
        if let Some(budget) = self.config.time_budget {
            let elapsed = start.elapsed();
            let remaining = budget.saturating_sub(elapsed);

            // Skip feature engineering if time is tight
            if remaining < Duration::from_secs(20) {
                skip_features = true;
                if self.config.verbose.enabled() && adapted_config.feature_engineering.is_enabled()
                {
                    println!("  [Budget] Low time remaining, skipping feature engineering");
                }
            }
        }

        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::FeatureEngineering,
            progress_pct: TrainingPhase::FeatureEngineering.progress_pct(),
            elapsed: start.elapsed(),
            message: if skip_features {
                Some("Skipped".to_string())
            } else {
                Some("Planning features".to_string())
            },
        });

        let phase_start = Instant::now();
        let mut feature_plan = if !skip_features {
            Some(self.plan_features(&profile)?)
        } else {
            None
        };
        phase_times.feature_engineering = phase_start.elapsed();

        if self.config.verbose.enabled() {
            if let Some(ref plan) = feature_plan {
                println!(
                    "  [Features] Planning: {} polynomial, {} ratio, {} interaction features",
                    plan.polynomial_features.len(),
                    plan.ratio_pairs.len(),
                    plan.interaction_pairs.len()
                );

                // WHITEBOX: Show exact features being created
                if !plan.polynomial_features.is_empty() {
                    println!("  [Features] Polynomial features (^2, ^3, sqrt, log1p):");
                    for feat in &plan.polynomial_features {
                        println!("    - {}", feat);
                    }
                }

                if !plan.ratio_pairs.is_empty() {
                    println!("  [Features] Ratio features:");
                    for (num, denom) in &plan.ratio_pairs {
                        println!("    - {} / {}", num, denom);
                    }
                }

                if !plan.interaction_pairs.is_empty() {
                    println!("  [Features] Interaction features (multiplicative):");
                    for (f1, f2) in &plan.interaction_pairs {
                        println!("    - {} × {}", f1, f2);
                    }
                }
            }
        }

        // === Phase 3b: Apply Cross-Sectional Features (polynomial, ratio, interaction) ===
        let mut working_df = df.clone();

        // === Phase 3a: Apply Custom User-Defined Features ===
        // These encode domain-specific knowledge that AutoML cannot discover
        if !self.config.custom_features.is_empty() {
            if self.config.verbose.enabled() {
                println!(
                    "  [CustomFeatures] Applying {} user-defined features",
                    self.config.custom_features.len()
                );
            }
            let custom_step = CustomFeaturesStep::new(self.config.custom_features.clone());
            working_df = custom_step.transform(working_df)?;
            if self.config.verbose.enabled() {
                for feature in &self.config.custom_features {
                    println!("    - {}", feature.name);
                }
            }
        }

        let initial_cols = working_df.width();

        // Apply cross-sectional features (polynomial, ratio, interaction) using canonical helper
        working_df = crate::features::apply_feature_plan(
            working_df,
            feature_plan.as_ref(),
            None, // panel_info will be added for time-series features below
        )?;

        if self.config.verbose.enabled() && feature_plan.as_ref().is_some_and(|p| !p.is_empty()) {
            let added = working_df.width() - initial_cols;
            let num_poly = feature_plan
                .as_ref()
                .map(|p| p.polynomial_features.len())
                .unwrap_or(0);
            let num_interactions = feature_plan
                .as_ref()
                .map(|p| p.interaction_pairs.len())
                .unwrap_or(0);
            let num_ratios = feature_plan
                .as_ref()
                .map(|p| p.ratio_pairs.len())
                .unwrap_or(0);

            println!(
                "  [Features] Applied: {} → {} columns (+{} = {} poly×4 + {} interactions + {} ratios)",
                initial_cols,
                working_df.width(),
                added,
                num_poly,
                num_interactions,
                num_ratios
            );
        }

        // === Phase 3c: Time-Series Feature Engineering ===
        // ALWAYS detect panel data structure (needed for era indices, even if not generating features)
        let panel_info = self.detect_panel_structure(&profile, df);

        if let Some(ref info) = panel_info {
            if self.config.verbose.enabled() {
                println!(
                    "  [Panel] Detected: {} groups by '{}', ordered by '{}'",
                    info.num_groups, info.group_column, info.date_column
                );
            }

            // Only generate time-series features if feature engineering is enabled
            if !skip_features {
                // Plan time-series features using user config (or default if not specified)
                let default_config = crate::features::SmartFeatureConfig::default();
                let config_option = self.config.feature_engineering.get_config();
                let feature_config = config_option.as_ref().unwrap_or(&default_config);
                let ts_plan =
                    SmartFeatureEngine::plan_timeseries_features(&profile, info, feature_config);

                if !ts_plan.is_empty() {
                    if self.config.verbose.enabled() {
                        println!(
                            "  [TimeSeries] Generating {} features (lag, rolling, EWMA)",
                            ts_plan.estimated_features
                        );
                    }

                    // Apply time-series features to the DataFrame
                    working_df = crate::features::apply_timeseries_features(
                        working_df, &ts_plan, info, true,
                    )?;

                    if self.config.verbose.enabled() {
                        println!(
                            "  [TimeSeries] New shape: {} rows × {} columns",
                            working_df.height(),
                            working_df.width()
                        );
                    }

                    // Update feature plan with time-series info
                    if let Some(ref mut plan) = feature_plan {
                        plan.timeseries_features = Some(ts_plan);
                    }
                }
            }
        }

        // === Phase 4: Prepare Dataset ===
        // Apply stateless pre-processing BEFORE creating BinnedDataset
        // (DropColumns and EngineerFeatures transform the schema)

        // Step 1: Drop useless columns (id, constants, etc.) BUT NOT the target
        if !profile.drop_columns.is_empty() {
            let cols_to_drop: Vec<String> = profile
                .drop_columns
                .iter()
                .filter(|c| c.name != target_col) // Never drop target
                .map(|c| c.name.clone())
                .collect();

            if !cols_to_drop.is_empty() {
                working_df = working_df.drop_many(&cols_to_drop);
                if self.config.verbose.enabled() {
                    println!(
                        "  [Prepare] Dropped {} columns: {:?}",
                        cols_to_drop.len(),
                        cols_to_drop
                    );
                }
            }
        }

        // CRITICAL VALIDATION: Ensure we have at least one feature after dropping columns
        let remaining_cols = working_df.get_column_names();
        let num_features = remaining_cols
            .iter()
            .filter(|&name| name.as_str() != target_col)
            .count();

        if num_features == 0 {
            let dropped_summary = if profile.drop_columns.is_empty() {
                String::from("No columns were marked for dropping by profiler.")
            } else {
                let reasons: Vec<String> = profile
                    .drop_columns
                    .iter()
                    .filter(|dc| dc.name != target_col) // Don't show target in dropped list
                    .map(|dc| format!("'{}' ({:?})", dc.name, dc.reason))
                    .collect();
                if reasons.is_empty() {
                    String::from("Only the target column was marked (but not dropped).")
                } else {
                    format!("Profiler recommended dropping: {}", reasons.join(", "))
                }
            };

            // Get mode for error message (fallback to PureTree if not explicitly set)
            let current_mode = adapted_config
                .custom_config
                .as_ref()
                .map(|c| c.mode)
                .unwrap_or(BoostingMode::PureTree);

            let mode_specific_msg = match current_mode {
                BoostingMode::LinearThenTree => {
                    "LinearThenTree requires features for both the linear and tree components."
                }
                BoostingMode::PureTree => "PureTree requires features for the tree model.",
                BoostingMode::RandomForest => {
                    "RandomForest requires features for the tree ensemble."
                }
            };

            return Err(TreeBoostError::Data(format!(
                "No features remaining after column profiling! {}\n\n\
                 Dataset has {} total columns, but all non-target columns were dropped.\n\
                 {}\n\n\
                 Common causes:\n\
                 1. All features are constants (zero variance)\n\
                 2. All features are detected as IDs (sequential integers with ID-like names)\n\
                 3. All features are text columns\n\n\
                 Solutions:\n\
                 - Verify your data has valid predictive features\n\
                 - Check column names (avoid 'id', 'index', 'row_id' for real features)\n\
                 - Disable automatic dropping: AutoConfig::new().with_feature_engineering(FeatureEngineeringMode::None)",
                mode_specific_msg,
                working_df.width(),
                dropped_summary
            )));
        }

        // Step 2: Apply feature engineering (already done above, working_df has engineered features)
        // (No-op here, just documenting the flow)

        // Now create BinnedDataset from preprocessed DataFrame
        // This learns categorical encodings and binning boundaries on the FINAL schema
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::DatasetPreparation,
            progress_pct: TrainingPhase::DatasetPreparation.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!(
                "{} rows × {} cols",
                working_df.height(),
                working_df.width()
            )),
        });

        let (mut dataset, pipeline_state, filtered_df) =
            self.prepare_dataset_and_state(&working_df, target_col, panel_info.as_ref())?;

        // === Phase 4b: Auto-detect bounded targets and recommend transformation ===
        let target_transform = self.detect_and_recommend_transform(&filtered_df, target_col)?;

        // === Phase 4c: Apply target transformation to training targets ===
        // This transforms targets BEFORE training (e.g., logit maps [0,100] → (-∞,+∞))
        // The inverse transform is applied during prediction via Pipeline.target_transform()
        if let Some(ref transform) = target_transform {
            use crate::preprocessing::TargetTransform;
            transform.transform(dataset.targets_mut())?;
            if self.config.verbose.enabled() {
                let targets = dataset.targets();
                let mean = targets.iter().sum::<f32>() / targets.len() as f32;
                let min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                println!(
                    "  [TargetTransform] Applied transform to training targets: mean={:.4}, min={:.4}, max={:.4}",
                    mean, min, max
                );
            }
        }

        // Prepare validation dataset if custom validation data provided
        // CRITICAL: Apply training's learned pipeline state (bin edges, encodings) to validation
        // data. This prevents data leakage - validation distribution must NOT influence training
        // bin boundaries. This matches inference behavior where serialized training state is applied.
        let validation_dataset = if let Some(ref val_df) = self.validation_df {
            // Apply same time-series features to validation data
            let val_working_df = if let (Some(ref info), Some(ref plan)) =
                (&panel_info, &feature_plan)
            {
                if let Some(ref ts_plan) = plan.timeseries_features {
                    crate::features::apply_timeseries_features(val_df.clone(), ts_plan, info, true)?
                } else {
                    val_df.clone()
                }
            } else {
                val_df.clone()
            };

            // Extract validation targets before binning
            let val_target_col = val_working_df.column(target_col).map_err(|e| {
                TreeBoostError::Data(format!(
                    "Target column '{}' not found in validation data: {}",
                    target_col, e
                ))
            })?;
            let val_targets: Vec<f32> = val_target_col
                .as_materialized_series()
                .cast(&polars::prelude::DataType::Float32)
                .map_err(|e| {
                    TreeBoostError::Data(format!("Failed to cast validation targets to f32: {}", e))
                })?
                .f32()
                .map_err(|e| {
                    TreeBoostError::Data(format!(
                        "Failed to extract validation targets as f32: {}",
                        e
                    ))
                })?
                .into_no_null_iter()
                .map(|v| if v.is_nan() { 0.0 } else { v })
                .collect();

            // Apply training's pipeline state to validation data (same bin edges, same encodings)
            let pipeline = crate::dataset::pipeline::DataPipeline::with_defaults();
            let (_val_preprocessed_df, mut val_dataset) =
                pipeline.process_for_inference(val_working_df.clone(), &pipeline_state)?;

            // Replace dummy targets with actual validation targets
            val_dataset.targets_mut().copy_from_slice(&val_targets);

            // Extract era indices for validation (must match training dataset's era structure)
            if dataset.era_indices().is_some() {
                if let Some(ref info) = panel_info {
                    if let Ok(era_col) = val_working_df.column(info.date_column.as_str()) {
                        let era_series = era_col.as_materialized_series();
                        let era_indices = pipeline.extract_era_indices(era_series)?;
                        val_dataset.set_era_indices(era_indices);
                    }
                }
            }

            // Apply same target transformation to validation targets
            if let Some(ref transform) = target_transform {
                use crate::preprocessing::TargetTransform;
                transform.transform(val_dataset.targets_mut())?;
            }

            Some(val_dataset)
        } else {
            None
        };

        // === Phase 5: Dataset Analysis (for mode selection) ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Analysis,
            progress_pct: TrainingPhase::Analysis.progress_pct(),
            elapsed: start.elapsed(),
            message: Some("Selecting optimal mode".to_string()),
        });

        let phase_start = Instant::now();
        let (mode, analysis, mode_confidence) = self.select_mode(&dataset)?;
        phase_times.analysis = phase_start.elapsed();

        if self.config.verbose.enabled() {
            println!(
                "  [Analysis] Selected mode: {:?}, confidence: {:?}",
                mode, mode_confidence
            );
        }

        // === Phase 6: Hyperparameter Tuning ===
        // Adapt tuning level based on remaining time
        if let Some(budget) = self.config.time_budget {
            let elapsed = start.elapsed();
            let remaining = budget.saturating_sub(elapsed);

            // Adjust tuning level based on remaining time
            if remaining < Duration::from_secs(10) {
                // Less than 10 seconds - skip tuning
                adapted_config.tuning_level = TuningLevel::None;
                if self.config.verbose.enabled() {
                    println!("  [Budget] Low time remaining, skipping tuning");
                }
            } else if remaining < Duration::from_secs(30) {
                // Less than 30 seconds - use quick tuning
                if adapted_config.tuning_level != TuningLevel::None {
                    adapted_config.tuning_level = TuningLevel::Quick;
                    if self.config.verbose.enabled() {
                        println!("  [Budget] Limited time, using quick tuning");
                    }
                }
            }
        }

        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Tuning,
            progress_pct: TrainingPhase::Tuning.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("{:?} - {:?}", adapted_config.tuning_level, mode)),
        });

        let phase_start = Instant::now();

        let (universal_config, ltt_tuning, tree_tuning) =
            // CRITICAL: If user provided custom_config, use it directly (bypasses all tuning)
            if let Some(ref custom) = self.config.custom_config {
                if self.config.verbose.enabled() {
                    println!("  [Tuning] Skipped (custom config provided)");
                }
                (custom.clone(), None, None)
            } else if adapted_config.tuning_level == TuningLevel::None {
                // Skip tuning, use defaults
                let config = tuning::create_config_for_mode(mode, self.config.tuning_level);
                (config, None, None)
            } else {
                // Print tuning start message only when actually tuning
                if self.config.verbose.enabled() {
                    let (train_samples, val_samples, expected_trials) =
                        self.compute_tuning_info(&adapted_config, mode, &dataset, &validation_dataset);
                    println!(
                        "  [Tuning] Starting {:?} tuning: {} train, {} val samples, ~{} trials...",
                        adapted_config.tuning_level, train_samples, val_samples, expected_trials
                    );
                }
                // CRITICAL: Pass filtered_df (preprocessed) to tuning for LTT raw feature extraction
                // Also pass custom validation dataset if provided
                tuning::tune_hyperparameters(
                    &self.config,
                    &dataset,
                    validation_dataset.as_ref(),
                    mode,
                    &filtered_df,
                    target_col,
                    &profile,
                )?
            };
        phase_times.tuning = phase_start.elapsed();

        // Note: target_transform is now part of Pipeline (built later)

        if self.config.verbose.enabled() {
            let num_trials = ltt_tuning
                .as_ref()
                .map(|t| t.history.linear_trials.len())
                .or_else(|| tree_tuning.as_ref().map(|t| t.num_trials))
                .unwrap_or(0);
            println!("  [Tuning] {} trials completed", num_trials);

            // Print detailed breakdown for LTT mode
            if let Some(ltt) = ltt_tuning.as_ref() {
                println!("  [Tuning] === LinearThenTree Breakdown ===");
                println!(
                    "  [Tuning] Linear phase R²: {:.4} ({:.1}% variance captured)",
                    ltt.linear_r2,
                    ltt.linear_r2 * 100.0
                );

                // Get best linear trial to show residual RMSE
                if let Some(best_linear) = ltt
                    .history
                    .linear_trials
                    .iter()
                    .min_by(|a, b| a.rmse.partial_cmp(&b.rmse).unwrap())
                {
                    println!("  [Tuning] Linear residual RMSE: {:.6}", best_linear.rmse);
                }

                // Get best tree trial to show how much trees improved
                if let Some(best_tree) = ltt
                    .history
                    .tree_trials
                    .iter()
                    .min_by(|a, b| a.residual_rmse.partial_cmp(&b.residual_rmse).unwrap())
                {
                    println!(
                        "  [Tuning] Tree residual RMSE: {:.6}",
                        best_tree.residual_rmse
                    );
                }

                println!("  [Tuning] Combined final RMSE: {:.6}", ltt.final_rmse);
            } else if let Some(tree) = tree_tuning.as_ref() {
                // best_metric is MSE, compute RMSE for display
                let rmse = tree.best_metric.sqrt();
                println!(
                    "  [Tuning] Best validation MSE: {:.6} (RMSE: {:.6})",
                    tree.best_metric, rmse
                );
            }

            println!("  [Tuning] (Computed on validation split during hyperparameter search)");
        }

        // === Phase 7: Train Final Model ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Training,
            progress_pct: TrainingPhase::Training.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("{:?} model", mode)),
        });

        let phase_start = Instant::now();
        // LinearThenTree Feature Extraction Strategy
        // ==========================================
        //
        // Critical design for LinearThenTree mode:
        //
        // 1. TREE COMPONENT needs ALL features (including categoricals, IDs, etc.)
        //    - BinnedDataset contains all preprocessed features
        //    - Trees make splits on the full feature space
        //
        // 2. LINEAR COMPONENT may want a SUBSET (e.g., exclude IDs, categoricals)
        //    - Linear models work best with numeric predictors
        //    - User can configure which features to exclude via LinearFeatureConfig
        //
        // Implementation approach:
        // - Extract ALL features as raw_features array (matches BinnedDataset)
        // - Build linear_feature_indices to select subset for linear model
        // - During training: linear_booster uses selected indices
        // - During prediction: same indices applied to maintain consistency
        //
        // This ensures:
        // - Training and inference use identical feature selection
        // - Trees use full feature space (no information loss)
        // - Linear model uses only appropriate features (better generalization)
        //
        let (feature_extractor, raw_features, linear_indices) = if matches!(
            mode,
            BoostingMode::LinearThenTree
        ) {
            // Step 1: Extract ALL features (no filtering) to match BinnedDataset
            let all_config = LinearFeatureConfig {
                exclude_columns: std::collections::HashSet::new(),
                exclude_categorical: false,
                exclude_id: false,
                exclude_constant: false,
                exclude_boolean: false,
                exclude_datetime: false,
                exclude_text: false,
            };
            let all_extractor =
                crate::dataset::feature_extractor::FeatureExtractor::with_config(all_config);
            let (all_features, _num_all_features) =
                all_extractor.extract(&filtered_df, target_col)?;

            // Step 2: Select features for the linear component based on LinearSelectionMode
            // Trees always use ALL features. Linear uses a subset determined by the mode.
            let col_names: Vec<String> = filtered_df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();

            let num_all_features = col_names.len() - 1; // -1 for target

            // Build feature name list (excluding target)
            let feature_names: Vec<String> = col_names
                .iter()
                .filter(|c| c.as_str() != target_col)
                .cloned()
                .collect();

            use crate::model::universal::config::LinearSelectionMode;
            let selection_mode = &universal_config.linear_selection_mode;

            let mut linear_feature_indices = Vec::new();
            let mut linear_feature_names: Vec<(String, f32)> = Vec::new(); // (name, score)

            match selection_mode {
                LinearSelectionMode::Correlation { threshold } => {
                    let num_rows = all_features.len() / num_all_features;
                    let target_series = filtered_df.column(target_col)?;
                    let target_f32: Vec<f32> = target_series
                        .cast(&polars::datatypes::DataType::Float32)?
                        .f32()?
                        .iter()
                        .map(|v| v.unwrap_or(0.0))
                        .collect();

                    for feat_idx in 0..num_all_features {
                        let feat_vals: Vec<f32> = (0..num_rows)
                            .map(|row| all_features[row * num_all_features + feat_idx])
                            .collect();
                        let corr =
                            crate::features::stats::correlation(&feat_vals, &target_f32).abs();
                        if corr >= *threshold {
                            linear_feature_indices.push(feat_idx);
                            linear_feature_names.push((feature_names[feat_idx].clone(), corr));
                        }
                    }

                    if linear_feature_indices.is_empty() {
                        linear_feature_indices = (0..num_all_features).collect();
                        if self.config.verbose.enabled() {
                            println!(
                                "  [Linear] No features above correlation threshold {:.2}, using all {} features",
                                threshold, num_all_features
                            );
                        }
                    } else if self.config.verbose.enabled() {
                        linear_feature_names.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        println!(
                            "  [Linear] Selected {} features for linear component (|corr| >= {:.2}):",
                            linear_feature_names.len(), threshold
                        );
                        for (name, corr) in &linear_feature_names {
                            println!("    - {} (|r|={:.3})", name, corr);
                        }
                        println!(
                            "  [Linear] Tree component uses all {} features",
                            num_all_features
                        );
                    }
                }
                LinearSelectionMode::Pattern => {
                    for (feat_idx, col_name) in feature_names.iter().enumerate() {
                        let is_engineered = col_name.ends_with("_squared")
                            || col_name.ends_with("_sqrt")
                            || col_name.ends_with("_log")
                            || col_name.ends_with("_log1p")
                            || col_name.contains("_x_") // Interaction features
                            || col_name.contains("_ratio_"); // Ratio features
                        if is_engineered {
                            linear_feature_indices.push(feat_idx);
                            linear_feature_names.push((col_name.clone(), f32::NAN));
                            // No score in Pattern mode
                        }
                    }

                    if linear_feature_indices.is_empty() {
                        linear_feature_indices = (0..num_all_features).collect();
                        if self.config.verbose.enabled() {
                            println!(
                                "  [Linear] No engineered features found, using all {} features",
                                num_all_features
                            );
                        }
                    } else if self.config.verbose.enabled() {
                        println!(
                            "  [Linear] Selected {} engineered features for linear component:",
                            linear_feature_names.len()
                        );
                        for (name, _) in &linear_feature_names {
                            println!("    - {}", name);
                        }
                        println!(
                            "  [Linear] Tree component uses all {} features",
                            num_all_features
                        );
                    }
                }
            }

            // CRITICAL VALIDATION: LinearThenTree needs features for BOTH linear and tree models
            if linear_feature_indices.is_empty() {
                return Err(TreeBoostError::Data(format!(
                        "LinearThenTree mode requires at least one feature for the linear model, but none were found!\n\n\
                         Dataset has {} columns after preprocessing.\n\n\
                         Common causes:\n\
                         1. All features were dropped as useless (constants, IDs, text)\n\
                         2. Feature engineering disabled and no features remain\n\n\
                         Solutions:\n\
                         - Verify your data has valid predictive features\n\
                         - Check column names (avoid 'id', 'index' for real features)\n\
                         - Try enabling feature engineering: AutoConfig::new().with_feature_engineering(FeatureEngineeringMode::Aggressive)",
                        col_names.len()
                    )));
            }

            let num_total_features = col_names.len() - 1; // -1 for target
            if num_total_features == 0 {
                return Err(TreeBoostError::Data("LinearThenTree mode requires features for the tree model, but none were found!\n\n\
                         This should not happen if linear features exist. Please report this as a bug.".to_string()));
            }

            (
                Some(all_extractor),
                Some(all_features),
                Some(linear_feature_indices),
            )
        } else {
            (None, None, None)
        };
        // Handle ensemble configuration by setting ensemble_seeds in UniversalConfig
        let final_config = if let Some(ref ensemble_config) = adapted_config.ensemble.get_config() {
            // Only PureTree and LinearThenTree support ensembles
            if matches!(mode, BoostingMode::PureTree | BoostingMode::LinearThenTree) {
                // Generate ensemble seeds from multi_seed config
                let seeds: Vec<u64> = (0..ensemble_config.multi_seed.n_seeds)
                    .map(|i| ensemble_config.multi_seed.base_seed + i as u64)
                    .collect();

                if self.config.verbose.enabled() {
                    println!("  [Ensemble] Training with {} seeds", seeds.len());
                }

                // Convert StackingConfig to StackingStrategy
                let stacking_strategy = crate::model::universal::config::StackingStrategy::Ridge {
                    alpha: ensemble_config.stacking.alpha,
                    rank_transform: ensemble_config.stacking.rank_transform,
                    fit_intercept: ensemble_config.stacking.fit_intercept,
                    min_weight: ensemble_config.stacking.min_weight,
                };

                // Set ensemble_seeds and stacking_strategy in UniversalConfig
                universal_config
                    .with_ensemble_seeds(seeds)
                    .with_stacking_strategy(stacking_strategy)
            } else {
                if self.config.verbose.enabled() {
                    println!(
                        "  [Ensemble] Skipped (mode {:?} does not support ensembles)",
                        mode
                    );
                }
                universal_config
            }
        } else {
            universal_config
        };

        // Add validation split and early stopping defaults (only if user didn't set them)
        // Skip if user provided presplit validation (they control the split externally)
        let mut final_config = final_config;
        if self.validation_df.is_none() {
            if final_config.validation_ratio == 0.0 {
                final_config = final_config.with_validation_ratio(0.2)?;
            }
            if final_config.early_stopping_rounds == 0 {
                final_config = final_config.with_early_stopping_rounds(50);
            }
        }

        // Check if skip_training is enabled (discovery mode)
        if self.config.skip_training {
            // Save discovered config without training
            if let Some(ref output_dir) = self.config.model_output_dir {
                use std::fs;
                fs::create_dir_all(output_dir)?;

                // Build Pipeline from all discovered settings
                let drop_columns: Vec<String> = profile
                    .drop_columns
                    .iter()
                    .map(|c| c.name.clone())
                    .collect();
                let pipeline = self.build_pipeline(
                    &feature_plan,
                    &pipeline_state,
                    target_transform.clone(),
                    final_config.linear_feature_indices.clone(),
                    &drop_columns,
                )?;

                // Build enriched config with pipeline
                let mut enriched_config = final_config.clone();
                enriched_config.pipeline = Some(pipeline);

                // Save config.json only (no model.rkyv)
                let config_path = output_dir.join("config.json");
                let config_json = serde_json::to_string_pretty(&enriched_config).map_err(|e| {
                    TreeBoostError::Serialization(format!("Failed to serialize config: {}", e))
                })?;
                fs::write(&config_path, config_json)?;

                if self.config.verbose.enabled() {
                    println!("  [Discovery] Config saved: {}", config_path.display());
                    println!("  [Discovery] Skip training enabled - no model trained");
                    println!("  [Discovery] Use UniversalModel::train() with this config for production training");
                }
            }

            // Return BuildResult with skip_training=true and no model
            return Ok(BuildResult {
                model: None,
                skip_training: true,
                mode,
                target_column: target_col.to_string(),
                mode_confidence: Some(mode_confidence),
                preprocessing_plan: Some(preprocessing_plan),
                feature_plan,
                ltt_tuning: None,
                tree_tuning: None,
                column_profile: Some(profile),
                analysis,
                pipeline_state: Some(pipeline_state),
                target_transform: target_transform.clone(),
                build_time: start.elapsed(),
                phase_times,
                validation_dataset: validation_dataset.clone(),
            });
        }

        // Train UniversalModel (handles both single and ensemble internally)
        let max_rounds = final_config.num_rounds; // Save before moving
        let linear_feature_indices_for_pipeline = final_config.linear_feature_indices.clone();
        let mut model = self.train_model(
            &dataset,
            final_config,
            feature_extractor,
            raw_features,
            linear_indices,
            validation_dataset.as_ref(),
        )?;
        phase_times.training = phase_start.elapsed();

        // Build Pipeline and set it on the model BEFORE creating BuildResult
        // This ensures the pipeline is serialized with model.rkyv
        let drop_columns: Vec<String> = profile
            .drop_columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let pipeline = self.build_pipeline(
            &feature_plan,
            &pipeline_state,
            target_transform.clone(),
            linear_feature_indices_for_pipeline,
            &drop_columns,
        )?;
        model.set_pipeline(pipeline);
        model.set_pipeline_state(pipeline_state.clone());

        // Report training metrics
        if self.config.verbose.enabled() {
            let num_trees = model.num_trees();
            let stopped_early = num_trees < max_rounds;

            println!("  [Train] Model trained in {:?}", phase_times.training);
            println!(
                "  [Train] Trees: {}/{} {}",
                num_trees,
                max_rounds,
                if stopped_early {
                    "(early stopped)"
                } else {
                    "(full rounds)"
                }
            );

            // Compute training RMSE in ORIGINAL scale (comparable to Kaggle score)
            // 1. Get raw predictions from model
            let mut train_preds = model.predict(&dataset);

            // 2. Apply inverse transform to predictions (e.g., sigmoid for Logit, clamp for Clamp)
            if let Some(ref transform) = target_transform {
                use crate::preprocessing::TargetTransform;
                let _ = transform.inverse_transform(&mut train_preds);
            }

            // 3. Get targets in original scale
            // Note: For Clamp, targets were never transformed (identity).
            // For Logit, targets were transformed, so we need to inverse-transform them back.
            let mut train_targets = dataset.targets().to_vec();
            if let Some(ref transform) = target_transform {
                use crate::preprocessing::TargetTransform;
                let _ = transform.inverse_transform(&mut train_targets);
            }

            // 4. Compute RMSE in original scale [0, 100]
            let mse: f32 = train_preds
                .iter()
                .zip(train_targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / train_preds.len() as f32;
            let train_rmse = mse.sqrt();
            println!("  [Train] Training RMSE: {:.4}", train_rmse);
        }

        // Emit completion progress
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Complete,
            progress_pct: TrainingPhase::Complete.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("Total: {:?}", start.elapsed())),
        });

        let build_result = BuildResult {
            model: Some(model),
            skip_training: false,
            mode,
            target_column: target_col.to_string(),
            mode_confidence: Some(mode_confidence),
            preprocessing_plan: Some(preprocessing_plan),
            feature_plan,
            ltt_tuning,
            tree_tuning,
            column_profile: Some(profile),
            analysis,
            pipeline_state: Some(pipeline_state), // CRITICAL for inference!
            target_transform: target_transform.clone(),
            build_time: start.elapsed(),
            phase_times,
            validation_dataset,
        };

        // Auto-save artifacts if model_output_dir is configured
        if let Some(ref output_dir) = self.config.model_output_dir {
            self.auto_save_artifacts(&build_result, output_dir)?;
        }

        Ok(build_result)
    }

    /// Profile a DataFrame to understand column types
    fn profile_dataframe(&self, df: &DataFrame, target_col: &str) -> Result<DataFrameProfile> {
        // Skip correlations when mode is fixed and feature engineering disabled
        let skip_correlations =
            !self.config.mode_selection.is_auto() && !self.config.feature_engineering.is_enabled();
        DataFrameProfile::analyze_with_options(df, target_col, skip_correlations)
    }

    /// Plan preprocessing based on profile and model type
    fn plan_preprocessing(
        &self,
        profile: &DataFrameProfile,
    ) -> Result<(ModelType, PreprocessingPlan)> {
        // PRIORITY 1: Check if custom_config provided - if so, use its mode for preprocessing
        // This ensures preprocessing and training use the same mode
        let model_type = if let Some(ref custom) = self.config.custom_config {
            match custom.mode {
                BoostingMode::LinearThenTree => ModelType::LinearThenTree,
                BoostingMode::PureTree | BoostingMode::RandomForest => ModelType::Tree,
            }
        } else {
            // PRIORITY 2: Check if user explicitly set the mode via mode_selection
            match &self.config.mode_selection {
                ModeSelection::Fixed(forced_mode) => {
                    // User explicitly set the mode - respect it
                    match forced_mode {
                        BoostingMode::LinearThenTree => ModelType::LinearThenTree,
                        BoostingMode::PureTree | BoostingMode::RandomForest => ModelType::Tree,
                    }
                }
                ModeSelection::Auto | ModeSelection::AutoWithConfig(_) => {
                    // PRIORITY 3: Auto-detect based on linear signal in data
                    let has_linear_signal = profile.columns.iter().any(|c| {
                        c.target_correlation
                            .map(|r| r.abs() > auto_defaults::LINEAR_SIGNAL_THRESHOLD)
                            .unwrap_or(false)
                    });

                    if has_linear_signal {
                        ModelType::LinearThenTree
                    } else {
                        ModelType::Tree
                    }
                }
            }
        };

        let plan = SmartPreprocessor::infer(profile, model_type);
        Ok((model_type, plan))
    }

    /// Plan feature engineering based on profile
    fn plan_features(&self, profile: &DataFrameProfile) -> Result<FeaturePlan> {
        // Use user-provided feature config, or default if not specified
        let default_config = crate::features::SmartFeatureConfig::default();
        let config_option = self.config.feature_engineering.get_config();
        let config = config_option.as_ref().unwrap_or(&default_config);

        let plan = SmartFeatureEngine::infer_with_config(profile, None, config);
        Ok(plan)
    }

    /// Prepare dataset and return pipeline state (for AutoModel inference)
    fn prepare_dataset_and_state(
        &self,
        df: &DataFrame,
        target_col: &str,
        panel_info: Option<&crate::analysis::PanelDataInfo>,
    ) -> Result<(BinnedDataset, crate::dataset::PipelineState, DataFrame)> {
        // Use DataPipeline to create BinnedDataset
        let pipeline = DataPipeline::with_defaults();

        // Pass None for categorical_columns to enable auto-detection
        // (String/Categorical dtypes will be treated as categorical)
        // Pass date_column as era_column if panel data detected
        let era_column = panel_info.map(|info| info.date_column.as_str());

        let (dataset, pipeline_state, filtered_df) =
            pipeline.process_for_training(df.clone(), target_col, None, era_column)?;

        Ok((dataset, pipeline_state, filtered_df))
    }

    /// Build a Pipeline from training components
    ///
    /// Constructs a sequential Pipeline that captures all transformation steps:
    /// 1. Feature engineering (polynomials, interactions, ratios)
    /// 2. Categorical encoding (with learned state from pipeline_state)
    /// 3. Target transformation (if provided)
    /// 4. Numeric binning (with learned boundaries from pipeline_state)
    /// 5. Linear feature extraction (if linear_feature_indices provided)
    fn build_pipeline(
        &self,
        feature_plan: &Option<FeaturePlan>,
        pipeline_state: &crate::dataset::PipelineState,
        target_transform: Option<crate::preprocessing::TargetTransformKind>,
        linear_feature_indices: Option<Vec<usize>>,
        drop_columns: &[String],
    ) -> Result<Pipeline> {
        let mut pipeline = Pipeline::new();

        // Step 0: DropColumns (MUST be first - remove id, constants, etc.)
        if !drop_columns.is_empty() {
            pipeline.add_step(PipelineStepKind::DropColumns(DropColumnsStep::new(
                drop_columns.to_vec(),
            )));
        }

        // Step 0.5: CustomFeaturesStep (user-defined features BEFORE auto feature engineering)
        // These encode domain-specific knowledge that AutoML cannot discover
        if !self.config.custom_features.is_empty() {
            if self.config.verbose.enabled() {
                println!(
                    "  [CustomFeatures] Applying {} user-defined features",
                    self.config.custom_features.len()
                );
                for feature in &self.config.custom_features {
                    println!("    - {}", feature.name);
                }
            }
            pipeline.add_step(PipelineStepKind::CustomFeatures(CustomFeaturesStep::new(
                self.config.custom_features.clone(),
            )));
        }

        // Step 1: EngineerFeaturesStep (if feature engineering was applied)
        if let Some(ref plan) = feature_plan {
            if !plan.is_empty() {
                pipeline.add_step(PipelineStepKind::EngineerFeatures(
                    EngineerFeaturesStep::new(
                        plan.polynomial_features.clone(),
                        plan.interaction_pairs.clone(),
                        plan.ratio_pairs.clone(),
                    ),
                ));
            }
        }

        // Step 2: EncodeCategoricalsStep (with learned encodings from pipeline_state)
        if !pipeline_state.categorical_encodings.is_empty() {
            let encodings: Vec<(String, CategoryEncoding)> = pipeline_state
                .categorical_encodings
                .iter()
                .map(|enc| {
                    (
                        enc.name.clone(),
                        CategoryEncoding {
                            method: "target".to_string(),
                            category_mapping: enc.category_mapping.clone(),
                            encoding_map: enc.encoding_map.clone(),
                            bin_boundaries: enc.bin_boundaries.clone(),
                        },
                    )
                })
                .collect();
            pipeline.add_step(PipelineStepKind::EncodeCategoricals(
                EncodeCategoricalsState { encodings },
            ));
        }

        // Step 3: TransformTargetStep (if target transform was applied)
        if let Some(transform) = target_transform {
            pipeline.add_step(PipelineStepKind::TransformTarget(TransformTargetStep::new(
                transform,
            )));
        }

        // Step 4: BinNumericFeaturesStep (with learned boundaries from feature_info)
        let boundaries: Vec<(String, Vec<f64>)> = pipeline_state
            .feature_info
            .iter()
            .filter(|fi| !fi.bin_boundaries.is_empty())
            .map(|fi| (fi.name.clone(), fi.bin_boundaries.clone()))
            .collect();
        pipeline.add_step(PipelineStepKind::BinNumericFeatures(
            BinNumericFeaturesState {
                num_bins: 255,
                boundaries,
            },
        ));

        // Step 5: ExtractLinearFeaturesStep (if LinearThenTree mode with feature indices)
        if let Some(indices) = linear_feature_indices {
            pipeline.add_step(PipelineStepKind::ExtractLinearFeatures(
                ExtractLinearFeaturesStep {
                    linear_feature_indices: indices,
                    all_feature_names: pipeline_state.column_order.clone(),
                },
            ));
        }

        // Store training column order so predict_df can reorder inference DataFrames
        pipeline.set_column_order(pipeline_state.column_order.clone());
        pipeline.set_target_column(Some("target".to_string())); // Placeholder, not critical for inference

        Ok(pipeline)
    }

    /// Select boosting mode based on analysis
    fn select_mode(
        &self,
        dataset: &BinnedDataset,
    ) -> Result<(BoostingMode, Option<DatasetAnalysis>, Confidence)> {
        // If custom config provided, use its mode
        if let Some(ref custom) = self.config.custom_config {
            return Ok((custom.mode, None, Confidence::High));
        }

        // If mode is fixed, use it
        if let Some(mode) = self.config.mode_selection.fixed_mode() {
            return Ok((mode, None, Confidence::High));
        }

        // Auto mode is enabled - run analysis

        // Run analysis
        let analysis = DatasetAnalysis::analyze(dataset)?;

        let mode = analysis.recommend_mode();
        let confidence = analysis.confidence();

        Ok((mode, Some(analysis), confidence))
    }

    /// Compute tuning information for display
    fn compute_tuning_info(
        &self,
        config: &AutoConfig,
        mode: BoostingMode,
        dataset: &BinnedDataset,
        validation_dataset: &Option<BinnedDataset>,
    ) -> (usize, usize, usize) {
        // Compute train/val split sizes
        let total_rows = dataset.num_rows();
        let (train_samples, val_samples) = if validation_dataset.is_some() {
            // Custom validation provided
            let val_rows = validation_dataset.as_ref().unwrap().num_rows();
            (total_rows, val_rows)
        } else {
            // Will do internal split - default 0.2 validation ratio
            let val_ratio = auto_defaults::DEFAULT_VALIDATION_RATIO;
            let train_rows = (total_rows as f32 * (1.0 - val_ratio)) as usize;
            let val_rows = total_rows - train_rows;
            (train_rows, val_rows)
        };

        // Compute expected trial count based on TuningLevel
        let expected_trials = match config.tuning_level {
            TuningLevel::Quick => {
                // Quick: 30 samples × 1 iteration
                30
            }
            TuningLevel::Standard => {
                // Standard: 100 samples × 3 iterations
                if matches!(mode, BoostingMode::LinearThenTree) {
                    // LTT does dual-phase tuning (linear + tree), roughly double
                    200
                } else {
                    300
                }
            }
            TuningLevel::Thorough => {
                // Thorough: 150 samples × 15 iterations
                if matches!(mode, BoostingMode::LinearThenTree) {
                    // LTT does dual-phase tuning
                    3000
                } else {
                    2250
                }
            }
            TuningLevel::None => 0,
        };

        (train_samples, val_samples, expected_trials)
    }

    /// Train the final model
    fn train_model(
        &self,
        dataset: &BinnedDataset,
        mut config: UniversalConfig,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
        raw_features: Option<Vec<f32>>,
        linear_indices: Option<Vec<usize>>,
        validation_dataset: Option<&BinnedDataset>,
    ) -> Result<UniversalModel> {
        // Use custom loss or default to MSE
        let default_loss = crate::loss::MseLoss;
        let loss: &dyn crate::loss::LossFunction = self.loss.as_deref().unwrap_or(&default_loss);

        // Add feature_extractor to config
        config.feature_extractor = feature_extractor;

        // Add linear_feature_indices to config (for LinearThenTree mode)
        // This ensures the indices are serialized and used during prediction
        if let Some(indices) = &linear_indices {
            config.linear_feature_indices = Some(indices.clone());
        }

        // Pass raw features AND linear_feature_indices to training if we're in LTT mode
        match (config.mode, raw_features, linear_indices) {
            (crate::model::BoostingMode::LinearThenTree, Some(features), Some(indices)) => {
                UniversalModel::train_with_linear_feature_selection(
                    dataset,
                    &features,
                    &indices,
                    config,
                    loss,
                    validation_dataset,
                )
            }
            _ => {
                // PureTree or RandomForest mode
                if let Some(val_data) = validation_dataset {
                    UniversalModel::train_with_validation(dataset, val_data, config, loss)
                } else {
                    UniversalModel::train(dataset, config, loss)
                }
            }
        }
    }

    /// Detect panel data structure in the DataFrame
    ///
    /// Tries automatic detection first, then falls back to heuristics
    /// for common patterns (e.g., numeric date columns).
    fn detect_panel_structure(
        &self,
        profile: &DataFrameProfile,
        df: &DataFrame,
    ) -> Option<PanelDataInfo> {
        // Try automatic detection first
        if let Some(info) = profile.detect_panel_structure(df) {
            return Some(info);
        }

        // Heuristic: Look for categorical column + numeric column named "date"
        // This handles cases where date is stored as integer (e.g., YYYYMMDD or ordinal)
        // CRITICAL: Exclude low-cardinality categoricals (e.g., "time_of_day" with 3 values = categorical period, not timestamp)

        // Find the date column (must be numeric OR high-cardinality categorical)
        let date_col_name = df
            .get_column_names()
            .iter()
            .find(|c| {
                let lower = c.to_lowercase();
                if !lower.contains("date") && !lower.contains("time") && lower != "dt" {
                    return false;
                }

                // Check the column's dtype and cardinality
                if let Ok(series) = df.column(c) {
                    let dtype = series.dtype();
                    // Accept numeric columns (integer timestamps)
                    if dtype.is_numeric() {
                        return true;
                    }
                    // For categorical: only accept if many unique values (>MIN_DATE_CARDINALITY)
                    // Low cardinality = categorical time period (morning/afternoon), not timestamps
                    if matches!(
                        dtype,
                        polars::prelude::DataType::String
                            | polars::prelude::DataType::Categorical(_, _)
                    ) {
                        if let Ok(n_unique) = series.n_unique() {
                            return n_unique > crate::defaults::analysis::MIN_DATE_CARDINALITY;
                        }
                    }
                }
                false
            })?
            .to_string();

        // Find a categorical column that could be the group
        let group_col = profile.columns.iter().find(|c| {
            c.dtype == crate::analysis::ColumnDataType::Categorical
                && c.cardinality >= 2
                && c.cardinality <= 10000
                && c.cardinality_ratio < 0.5
        })?;

        // Compute statistics
        let num_groups = group_col.cardinality;
        let avg_obs = df.height() as f32 / num_groups as f32;

        Some(PanelDataInfo {
            group_column: group_col.name.clone(),
            date_column: date_col_name,
            num_groups,
            avg_observations_per_group: avg_obs,
            min_observations_per_group: (avg_obs * 0.5) as usize,
            max_observations_per_group: (avg_obs * 1.5) as usize,
            confidence: 0.7, // Lower confidence for heuristic detection
            time_granularity: crate::analysis::TimeGranularity::Daily, // Default assumption
        })
    }

    /// Auto-detect bounded targets and recommend transformation
    ///
    /// Analyzes the target column to detect if values are bounded and suggests
    /// appropriate transformations:
    /// - [0, 1]: Suggest LogitTransform(0.0, 1.0) for probabilities
    /// - Explicit user config: Use specified mode (Logit, Clamp, etc.)
    /// - None: No transformation
    ///
    /// # Returns
    ///
    /// `Ok(Some(TargetTransformKind))` if transform configured
    /// `Ok(None)` if no transformation requested
    fn detect_and_recommend_transform(
        &self,
        df: &DataFrame,
        target_col: &str,
    ) -> Result<Option<crate::preprocessing::TargetTransformKind>> {
        use crate::model::TargetBoundConfig;

        // Handle explicit user configuration
        match &self.config.target_bound_config {
            TargetBoundConfig::None => {
                if self.config.verbose.enabled() {
                    println!("  [TargetBound] No target transformation configured");
                }
                return Ok(None);
            }
            TargetBoundConfig::Logit { min, max } => {
                if self.config.verbose.enabled() {
                    println!(
                        "  [TargetBound] Using Logit transform with fixed bounds: [{:.4}, {:.4}]",
                        min, max
                    );
                }
                return Ok(Some(crate::preprocessing::TargetTransformKind::logit(
                    *min, *max,
                )?));
            }
            TargetBoundConfig::Clamp { min, max } => {
                if self.config.verbose.enabled() {
                    println!(
                        "  [TargetBound] Using Clamp transform with fixed bounds: [{:.4}, {:.4}]",
                        min, max
                    );
                }
                return Ok(Some(crate::preprocessing::TargetTransformKind::clamp(
                    *min, *max,
                )?));
            }
            TargetBoundConfig::LogitEmpirical | TargetBoundConfig::ClampEmpirical => {
                // Need to compute empirical bounds from data
            }
        }

        // For empirical modes, extract target column and compute bounds
        let target_series = df.column(target_col).map_err(|e| {
            TreeBoostError::Data(format!("Target column '{}' not found: {}", target_col, e))
        })?;

        // Convert to f32 values
        let targets: Vec<f32> = match target_series.dtype() {
            polars::prelude::DataType::Float32 => {
                target_series.f32()?.into_no_null_iter().collect()
            }
            polars::prelude::DataType::Float64 => target_series
                .f64()?
                .into_no_null_iter()
                .map(|x| x as f32)
                .collect(),
            polars::prelude::DataType::Int32 => target_series
                .i32()?
                .into_no_null_iter()
                .map(|x| x as f32)
                .collect(),
            polars::prelude::DataType::Int64 => target_series
                .i64()?
                .into_no_null_iter()
                .map(|x| x as f32)
                .collect(),
            _ => {
                // Non-numeric target, cannot apply transformation
                return Ok(None);
            }
        };

        if targets.is_empty() {
            return Ok(None);
        }

        // Compute empirical bounds
        let empirical_min = targets.iter().cloned().fold(f32::INFINITY, f32::min);
        let empirical_max = targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if self.config.verbose.enabled() {
            println!(
                "  [TargetBound] Empirical bounds: [{:.4}, {:.4}]",
                empirical_min, empirical_max
            );
        }

        // Apply empirical transform based on config
        match &self.config.target_bound_config {
            TargetBoundConfig::LogitEmpirical => {
                if self.config.verbose.enabled() {
                    println!(
                        "  [TargetBound] Using Logit transform with empirical bounds: [{:.4}, {:.4}]",
                        empirical_min, empirical_max
                    );
                }
                Ok(Some(crate::preprocessing::TargetTransformKind::logit(
                    empirical_min,
                    empirical_max,
                )?))
            }
            TargetBoundConfig::ClampEmpirical => {
                if self.config.verbose.enabled() {
                    println!(
                        "  [TargetBound] Using Clamp transform with empirical bounds: [{:.4}, {:.4}]",
                        empirical_min, empirical_max
                    );
                }
                Ok(Some(crate::preprocessing::TargetTransformKind::clamp(
                    empirical_min,
                    empirical_max,
                )?))
            }
            // Already handled above
            _ => Ok(None),
        }
    }

    /// Automatically save all artifacts (model, config, metadata) to output directory
    fn auto_save_artifacts(
        &self,
        build_result: &BuildResult,
        output_dir: &std::path::Path,
    ) -> Result<()> {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;

        // If skip_training mode, the config was already saved inline during fit()
        // No model to save, only config was produced
        if build_result.skip_training {
            return Ok(());
        }

        // Extract model (safe to unwrap since skip_training is false)
        let model = build_result
            .model
            .as_ref()
            .expect("model must be Some when skip_training is false");

        // Model already has pipeline set (from fit() before BuildResult creation)
        // Just extract the config for saving as JSON
        let enriched_config = model.config().clone();

        // Save trained model (rkyv format for fast loading)
        // The model's config now includes the Pipeline - it's serialized together
        let model_path = output_dir.join("model.rkyv");
        model.save(&model_path)?;
        if self.config.verbose.enabled() {
            println!("  [AutoSave] Model: {}", model_path.display());
        }

        // Save UniversalConfig as config.json (human-readable view)
        // The model.rkyv is the SINGLE SOURCE OF TRUTH.
        // config.json is extracted FROM the model for human inspection and contains:
        // - Model hyperparameters
        // - Pipeline steps (feature engineering, encoding, binning, target transform)
        // - Linear feature indices
        // - Target column name
        let config_path = output_dir.join("config.json");
        let config_json = serde_json::to_string_pretty(&enriched_config).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize config: {}", e))
        })?;
        fs::write(&config_path, config_json)?;
        if self.config.verbose.enabled() {
            println!("  [AutoSave] Config: {}", config_path.display());
        }

        Ok(())
    }
}

impl Default for AutoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_config_defaults() {
        let config = AutoConfig::default();
        assert_eq!(config.tuning_level, TuningLevel::Standard);
        assert!((config.val_ratio - 0.2).abs() < 0.01);
        assert!(config.feature_engineering.is_enabled());
        assert!(config.preprocessing.is_enabled());
        assert_eq!(config.mode_selection, ModeSelection::Auto);
    }

    #[test]
    fn test_auto_builder_creation() {
        let builder = AutoBuilder::new()
            .with_tuning(TuningLevel::Quick)
            .with_verbose(true);

        assert_eq!(builder.config.tuning_level, TuningLevel::Quick);
        assert!(builder.config.verbose.enabled());
    }

    #[test]
    fn test_tuning_level_variants() {
        assert_eq!(TuningLevel::default(), TuningLevel::Standard);
    }
}
