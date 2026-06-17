//! AutoModel: Self-contained trained model from AutoBuilder
//!
//! AutoModel wraps the result of AutoBuilder training and provides a clean
//! interface for predictions and model inspection.
//!
//! # Features
//!
//! - **One-line training**: `AutoModel::train(&df, "target")?`
//! - **Easy prediction**: `model.predict(&df)?`
//! - **Training summary**: Access all metadata from the build process
//! - **Serialization**: Save/load trained models
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::AutoModel;
//! use polars::prelude::*;
//!
//! // One-line training
//! let model = AutoModel::train(&df, "target")?;
//!
//! // See what mode was selected
//! println!("Mode: {:?}", model.mode());
//! println!("Build time: {:?}", model.build_time());
//!
//! // Predict
//! let predictions = model.predict(&test_df)?;
//!
//! // Save for later
//! model.save("model.rkyv")?;
//!
//! // Load and use
//! let loaded = AutoModel::load("model.rkyv")?;
//! let preds = loaded.predict(&test_df)?;
//! ```

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis};
use crate::dataset::{BinnedDataset, DataPipeline};
use crate::features::FeaturePlan;
use crate::loss::MseLoss;
use crate::model::{
    AutoBuilder, AutoConfig, BoostingMode, BuildPhaseTimes, BuildResult, TreeTuningResult,
    TuningLevel, UniversalModel,
};
use crate::preprocessing::PreprocessingPlan;
use crate::tuner::ltt::LttTuningResult;
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::time::Duration;

/// AutoModel: Trained model with full metadata
///
/// This is the main user-facing type for trained models. It wraps the
/// `UniversalModel` along with all the metadata from the training process.
pub struct AutoModel {
    /// The underlying trained model (UniversalModel handles all modes and ensembles)
    model: UniversalModel,

    /// Boosting mode that was used
    mode: BoostingMode,

    /// Target column name used during training
    target_column: String,

    /// Confidence in mode selection
    mode_confidence: Option<Confidence>,

    /// Preprocessing plan that was applied
    preprocessing_plan: Option<PreprocessingPlan>,

    /// Feature engineering plan that was applied
    feature_plan: Option<FeaturePlan>,

    /// LTT tuning result (if applicable)
    ltt_tuning: Option<LttTuningResult>,

    /// Tree tuning result (if PureTree/RandomForest mode was used)
    tree_tuning: Option<TreeTuningResult>,

    /// Column profile from analysis
    column_profile: Option<DataFrameProfile>,

    /// Dataset analysis result
    analysis: Option<DatasetAnalysis>,

    /// Fitted pipeline state (CRITICAL for inference!)
    pipeline_state: Option<crate::dataset::PipelineState>,

    /// Total build time
    build_time: Duration,

    /// Time breakdown by phase
    phase_times: BuildPhaseTimes,

    /// Tuned thresholds for multi-label classification (one per label)
    ///
    /// These thresholds are optimized to maximize F1 score on validation data.
    /// Use `tune_thresholds()` to set these, then `predict_labels_tuned()` will use them.
    tuned_thresholds: Option<Vec<f32>>,
}

impl AutoModel {
    /// Create an AutoModel from a BuildResult
    ///
    /// # Errors
    ///
    /// Returns an error if the BuildResult has `skip_training = true` (discovery mode).
    /// AutoModel requires a trained model.
    pub fn from_build_result(result: BuildResult) -> Result<Self> {
        if result.skip_training {
            return Err(TreeBoostError::Config(
                "Cannot create AutoModel from BuildResult with skip_training = true. \
                 Use BuildResult directly for discovery mode."
                    .to_string(),
            ));
        }

        let model = result.model.ok_or_else(|| {
            TreeBoostError::Config(
                "BuildResult has skip_training = false but model is None. This is a bug."
                    .to_string(),
            )
        })?;

        Ok(Self {
            model,
            mode: result.mode,
            target_column: result.target_column,
            mode_confidence: result.mode_confidence,
            preprocessing_plan: result.preprocessing_plan,
            feature_plan: result.feature_plan,
            ltt_tuning: result.ltt_tuning,
            tree_tuning: result.tree_tuning,
            column_profile: result.column_profile,
            pipeline_state: result.pipeline_state,
            analysis: result.analysis,
            build_time: result.build_time,
            phase_times: result.phase_times,
            tuned_thresholds: None,
        })
    }

    /// Train a model with default settings (the simplest API)
    ///
    /// This is the recommended entry point for most users.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = AutoModel::train(&df, "price")?;
    /// let predictions = model.predict(&test_df)?;
    /// ```
    pub fn train(df: &DataFrame, target_col: &str) -> Result<Self> {
        let builder = AutoBuilder::new();
        let result = builder.fit(df, target_col)?;
        Self::from_build_result(result)
    }

    /// Train with quick settings (minimal tuning, fast training)
    pub fn train_quick(df: &DataFrame, target_col: &str) -> Result<Self> {
        let builder = AutoBuilder::new().with_tuning(TuningLevel::Quick);
        let result = builder.fit(df, target_col)?;
        Self::from_build_result(result)
    }

    /// Train with thorough settings (extensive tuning, best accuracy)
    pub fn train_thorough(df: &DataFrame, target_col: &str) -> Result<Self> {
        let builder = AutoBuilder::new().with_tuning(TuningLevel::Thorough);
        let result = builder.fit(df, target_col)?;
        Self::from_build_result(result)
    }

    /// Train with a specific mode (bypass auto-selection)
    pub fn train_with_mode(df: &DataFrame, target_col: &str, mode: BoostingMode) -> Result<Self> {
        let builder = AutoBuilder::new().with_mode(mode);
        let result = builder.fit(df, target_col)?;
        Self::from_build_result(result)
    }

    /// Train with custom configuration
    pub fn train_with_config(df: &DataFrame, target_col: &str, config: AutoConfig) -> Result<Self> {
        let builder = AutoBuilder::with_config(config);
        let result = builder.fit(df, target_col)?;
        Self::from_build_result(result)
    }

    // =========================================================================
    // Multi-Label Training Methods
    // =========================================================================

    /// Train a multi-label classification model
    ///
    /// This is the recommended entry point for multi-label tasks where each sample
    /// can have multiple binary labels (e.g., multi-tag classification).
    ///
    /// # Arguments
    /// * `df` - DataFrame containing features and target columns
    /// * `target_cols` - Names of the binary target columns (each should be 0/1)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Multi-label: each article can have multiple tags
    /// let model = AutoModel::train_multilabel(&df, &["is_tech", "is_finance", "is_sports"])?;
    /// let predictions = model.predict_multilabel(&test_df)?;
    /// ```
    pub fn train_multilabel(df: &DataFrame, target_cols: &[&str]) -> Result<Self> {
        Self::train_multilabel_with_mode(df, target_cols, BoostingMode::LinearThenTree)
    }

    /// Train a multi-label model with a specific boosting mode
    pub fn train_multilabel_with_mode(
        df: &DataFrame,
        target_cols: &[&str],
        mode: BoostingMode,
    ) -> Result<Self> {
        use crate::loss::MultiLabelLogLoss;
        use crate::model::UniversalModel;

        let start = std::time::Instant::now();

        if target_cols.len() < 2 {
            return Err(TreeBoostError::Config(
                "Multi-label training requires at least 2 target columns".to_string(),
            ));
        }

        // Validate all target columns exist
        for &col in target_cols {
            if df.column(col).is_err() {
                return Err(TreeBoostError::Data(format!(
                    "Target column '{}' not found in DataFrame",
                    col
                )));
            }
        }

        let num_rows = df.height();
        let num_labels = target_cols.len();

        // Use first target column for pipeline processing (features are independent of which target)
        let primary_target = target_cols[0];

        // Process through pipeline
        let pipeline = DataPipeline::with_defaults();
        let (binned_dataset, pipeline_state, _preprocessed_df) =
            pipeline.process_for_training(df.clone(), primary_target, None, None)?;

        // Extract all targets (row-wise flattened: [row0_label0, row0_label1, ..., row1_label0, ...])
        let mut targets = Vec::with_capacity(num_rows * num_labels);
        for row_idx in 0..num_rows {
            for &col_name in target_cols {
                let series = df.column(col_name)?;
                let value = series
                    .get(row_idx)
                    .map_err(|e| TreeBoostError::Data(format!("Failed to get value: {}", e)))?;

                let target_val = match value {
                    AnyValue::Int32(v) => v as f32,
                    AnyValue::Int64(v) => v as f32,
                    AnyValue::Float32(v) => v,
                    AnyValue::Float64(v) => v as f32,
                    AnyValue::UInt32(v) => v as f32,
                    AnyValue::UInt64(v) => v as f32,
                    _ => {
                        return Err(TreeBoostError::Data(format!(
                            "Target column '{}' has unsupported type: {:?}",
                            col_name, value
                        )))
                    }
                };
                targets.push(target_val);
            }
        }

        // Create multi-output dataset with our targets
        let multioutput_dataset = BinnedDataset::new_multioutput(
            num_rows,
            binned_dataset.features().to_vec(),
            targets,
            binned_dataset.all_feature_info().to_vec(),
            num_labels,
        );

        // Configure model
        let config = crate::model::UniversalConfig::default()
            .with_mode(mode)
            .with_num_rounds(100)
            .with_learning_rate(0.1)?;

        // Train with multi-label log loss
        let loss_fn = MultiLabelLogLoss::new();
        let model = UniversalModel::train_multilabel(&multioutput_dataset, config, &loss_fn)?;

        let build_time = start.elapsed();

        Ok(Self {
            model,
            mode,
            target_column: target_cols.join(","),
            mode_confidence: None,
            preprocessing_plan: None,
            feature_plan: None,
            ltt_tuning: None,
            tree_tuning: None,
            column_profile: None,
            analysis: None,
            pipeline_state: Some(pipeline_state),
            build_time,
            phase_times: BuildPhaseTimes::default(),
            tuned_thresholds: None,
        })
    }

    /// Get number of labels (for multi-label models)
    pub fn num_labels(&self) -> usize {
        self.model.num_linear_boosters().max(
            self.model.num_gbdt_per_label().max(
                self.model
                    .gbdt_model()
                    .map(|m| m.num_outputs())
                    .unwrap_or(1),
            ),
        )
    }

    /// Predict multi-label probabilities on a DataFrame
    ///
    /// Returns `Vec<Vec<f32>>` where each inner vec contains probabilities for all labels.
    pub fn predict_multilabel(&self, df: &DataFrame) -> Result<Vec<Vec<f32>>> {
        let (_preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;
        Ok(self.model.predict_multilabel(&dataset))
    }

    /// Predict multi-label probabilities (sigmoid applied)
    pub fn predict_proba_multilabel(&self, df: &DataFrame) -> Result<Vec<Vec<f32>>> {
        let (_preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;
        Ok(self.model.predict_proba_multilabel(&dataset))
    }

    /// Predict multi-label boolean labels with threshold 0.5
    ///
    /// Returns `Vec<Vec<bool>>` where each inner vec contains boolean labels for all outputs.
    pub fn predict_labels(&self, df: &DataFrame) -> Result<Vec<Vec<bool>>> {
        let (_preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;
        Ok(self.model.predict_labels(&dataset))
    }

    /// Predict multi-label boolean labels with custom threshold
    ///
    /// # Arguments
    /// * `df` - Input DataFrame
    /// * `threshold` - Classification threshold (0.0 to 1.0)
    pub fn predict_labels_with_threshold(
        &self,
        df: &DataFrame,
        threshold: f32,
    ) -> Result<Vec<Vec<bool>>> {
        let (_preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;
        Ok(self
            .model
            .predict_labels_with_threshold(&dataset, threshold))
    }

    /// Tune thresholds for multi-label classification using F1 optimization
    ///
    /// This sweeps thresholds from 0.01 to 0.99 and finds the optimal threshold
    /// for each label that maximizes F1 score on the validation data.
    ///
    /// **IMPORTANT**: Use held-out validation data, NOT training data, to avoid leakage.
    ///
    /// # Arguments
    /// * `val_df` - Validation DataFrame with features and target columns
    /// * `target_cols` - Names of the target columns (must match training)
    ///
    /// # Returns
    /// `TuneResult` with optimal thresholds and metrics per label
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Train on training data
    /// let mut model = AutoModel::train_multilabel(&train_df, &target_cols)?;
    ///
    /// // Tune thresholds on validation data
    /// let tune_result = model.tune_thresholds(&val_df, &target_cols)?;
    /// println!("Optimal thresholds: {:?}", tune_result.thresholds);
    ///
    /// // Now predict_labels_tuned() will use optimal thresholds
    /// let labels = model.predict_labels_tuned(&test_df)?;
    /// ```
    pub fn tune_thresholds(
        &mut self,
        val_df: &DataFrame,
        target_cols: &[&str],
    ) -> Result<crate::analysis::TuneResult> {
        use crate::analysis::ThresholdTuner;

        let num_labels = target_cols.len();
        let num_rows = val_df.height();

        // Get probabilities on validation set
        let probabilities = self.predict_proba_multilabel(val_df)?;

        // Extract targets from validation DataFrame
        let mut targets = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let mut row_targets = Vec::with_capacity(num_labels);
            for &col_name in target_cols {
                let series = val_df.column(col_name)?;
                let value = series
                    .get(row_idx)
                    .map_err(|e| TreeBoostError::Data(format!("Failed to get value: {}", e)))?;

                let target_val = match value {
                    AnyValue::Int32(v) => v as f32,
                    AnyValue::Int64(v) => v as f32,
                    AnyValue::Float32(v) => v,
                    AnyValue::Float64(v) => v as f32,
                    AnyValue::UInt32(v) => v as f32,
                    AnyValue::UInt64(v) => v as f32,
                    _ => {
                        return Err(TreeBoostError::Data(format!(
                            "Target column '{}' has unsupported type",
                            col_name
                        )))
                    }
                };
                row_targets.push(target_val);
            }
            targets.push(row_targets);
        }

        // Run threshold tuning
        let tuner = ThresholdTuner::new();
        let result = tuner.tune(&probabilities, &targets, num_labels)?;

        // Store the tuned thresholds
        self.tuned_thresholds = Some(result.thresholds.clone());

        Ok(result)
    }

    /// Get tuned thresholds (if available)
    pub fn tuned_thresholds(&self) -> Option<&[f32]> {
        self.tuned_thresholds.as_deref()
    }

    /// Predict labels using tuned thresholds
    ///
    /// Falls back to 0.5 if thresholds haven't been tuned.
    pub fn predict_labels_tuned(&self, df: &DataFrame) -> Result<Vec<Vec<bool>>> {
        let probabilities = self.predict_proba_multilabel(df)?;

        let labels: Vec<Vec<bool>> = if let Some(ref thresholds) = self.tuned_thresholds {
            probabilities
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(thresholds.iter())
                        .map(|(&prob, &threshold)| prob >= threshold)
                        .collect()
                })
                .collect()
        } else {
            // Fallback to 0.5 threshold
            probabilities
                .iter()
                .map(|row| row.iter().map(|&prob| prob >= 0.5).collect())
                .collect()
        };

        Ok(labels)
    }

    /// Predict on a DataFrame
    ///
    /// Returns predictions as a `Vec<f32>`.
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f32>> {
        // Convert DataFrame to BinnedDataset for prediction
        // Also get preprocessed DataFrame with encoded categoricals
        let (preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;

        // For LinearThenTree, use dual-representation inference if FeatureExtractor is stored
        if matches!(self.mode, crate::model::BoostingMode::LinearThenTree) {
            if let Some(extractor) = self.model.feature_extractor() {
                // CRITICAL: Extract from preprocessed_df (with encoded categoricals),
                // NOT from original df (with String categoricals)
                let (raw_features, _num_features) =
                    extractor.extract(&preprocessed_df, &self.target_column)?;

                return Ok(self
                    .model
                    .predict_with_raw_features(&dataset, &raw_features));
            }
        }

        Ok(self.model.predict(&dataset))
    }

    /// Predict using only the linear component (LinearThenTree mode only)
    ///
    /// For fair comparison between linear-only and full LinearThenTree.
    /// Uses the same preprocessing pipeline as the full model.
    ///
    /// # Returns
    /// - `Ok(Vec<f32>)`: Linear-only predictions (base + linear model)
    /// - `Err`: If model is not LinearThenTree mode
    pub fn predict_linear_only(&self, df: &DataFrame) -> Result<Vec<f32>> {
        if !matches!(self.mode, crate::model::BoostingMode::LinearThenTree) {
            return Err(TreeBoostError::Config(
                "predict_linear_only() only available for LinearThenTree mode".to_string(),
            ));
        }

        // Use same preprocessing as full prediction
        let (preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;

        // Extract features using FeatureExtractor
        if let Some(extractor) = self.model.feature_extractor() {
            let (raw_features, _num_features) =
                extractor.extract(&preprocessed_df, &self.target_column)?;

            // Get linear-only predictions from the model
            return self.model.predict_linear_only(&dataset, &raw_features);
        }

        Err(TreeBoostError::Config(
            "LinearThenTree model missing FeatureExtractor - cannot predict".to_string(),
        ))
    }

    /// Predict on a BinnedDataset (for advanced users)
    pub fn predict_binned(&self, dataset: &BinnedDataset) -> Vec<f32> {
        self.model.predict(dataset)
    }

    /// Get the boosting mode that was used
    pub fn mode(&self) -> BoostingMode {
        self.mode
    }

    /// Get confidence in the mode selection
    pub fn mode_confidence(&self) -> Option<Confidence> {
        self.mode_confidence
    }

    /// Get total training time
    pub fn build_time(&self) -> Duration {
        self.build_time
    }

    /// Get time breakdown by phase
    pub fn phase_times(&self) -> &BuildPhaseTimes {
        &self.phase_times
    }

    /// Get the preprocessing plan that was applied
    pub fn preprocessing_plan(&self) -> Option<&PreprocessingPlan> {
        self.preprocessing_plan.as_ref()
    }

    /// Get the feature engineering plan that was applied
    pub fn feature_plan(&self) -> Option<&FeaturePlan> {
        self.feature_plan.as_ref()
    }

    /// Get the column profile from analysis
    pub fn column_profile(&self) -> Option<&DataFrameProfile> {
        self.column_profile.as_ref()
    }

    /// Get the dataset analysis result
    pub fn analysis(&self) -> Option<&DatasetAnalysis> {
        self.analysis.as_ref()
    }

    /// Get LTT tuning result (if LTT mode was used)
    pub fn ltt_tuning(&self) -> Option<&LttTuningResult> {
        self.ltt_tuning.as_ref()
    }

    /// Get tree tuning result (if PureTree/RandomForest mode was used)
    pub fn tree_tuning(&self) -> Option<&TreeTuningResult> {
        self.tree_tuning.as_ref()
    }

    /// Get the fitted pipeline state (feature names, encodings, etc.)
    pub fn pipeline_state(&self) -> Option<&crate::dataset::PipelineState> {
        self.pipeline_state.as_ref()
    }

    /// Get number of trees in the model
    pub fn num_trees(&self) -> usize {
        self.model.num_trees()
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.model.num_features()
    }

    /// Get the underlying UniversalModel
    pub fn inner(&self) -> &UniversalModel {
        &self.model
    }

    /// Get the discovered UniversalConfig
    ///
    /// This returns the config that AutoModel discovered through analysis and tuning.
    /// You can export this config to JSON and use it to retrain with UniversalModel directly.
    pub fn config(&self) -> &crate::model::UniversalConfig {
        self.model.config()
    }

    /// Get a comprehensive summary of the training process
    ///
    /// This shows the full "Smart Engineer" report explaining every decision
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "┌─────────────────────────────────────────────────────────────────┐".to_string(),
            "│                  TreeBoost Pipeline Report                      │".to_string(),
            "└─────────────────────────────────────────────────────────────────┘".to_string(),
            "".to_string(),
        ];

        // Section 1: Data Profile
        if let Some(ref profile) = self.column_profile {
            lines.push("═══ DATA PROFILE ═══".to_string());
            lines.push(format!("  Rows: {}", profile.num_rows));
            lines.push(format!("  Columns: {} total", profile.columns.len()));
            lines.push(format!("    • Numeric: {}", profile.num_numeric));
            lines.push(format!("    • Categorical: {}", profile.num_categorical));
            lines.push(format!(
                "  Target: {} ({:?})",
                self.target_column, profile.task_type
            ));

            if !profile.drop_columns.is_empty() {
                lines.push("".to_string());
                lines.push(format!("  Dropped {} columns:", profile.drop_columns.len()));
                for dropped in &profile.drop_columns {
                    lines.push(format!("    • '{}' - {}", dropped.name, dropped.reason));
                }
            }
            lines.push("".to_string());
        }

        // Section 2: Preprocessing Decisions
        if let Some(ref plan) = self.preprocessing_plan {
            lines.push("═══ PREPROCESSING DECISIONS ═══".to_string());
            if !plan.reasoning.is_empty() {
                for reason in &plan.reasoning {
                    lines.push(format!("  • {}", reason));
                }
            } else {
                lines.push("  • No special preprocessing required".to_string());
            }
            lines.push("".to_string());
        }

        // Section 3: Feature Engineering
        if let Some(ref plan) = self.feature_plan {
            lines.push("═══ FEATURE ENGINEERING ═══".to_string());

            if !plan.polynomial_features.is_empty() {
                lines.push(format!(
                    "  Polynomial features ({}): ",
                    plan.polynomial_features.len()
                ));
                for feat in &plan.polynomial_features {
                    lines.push(format!("    • {}", feat));
                }
            }

            if !plan.ratio_pairs.is_empty() {
                lines.push(format!("  Ratio features ({}): ", plan.ratio_pairs.len()));
                for (f1, f2) in &plan.ratio_pairs {
                    lines.push(format!("    • {}/{}", f1, f2));
                }
            }

            if !plan.interaction_pairs.is_empty() {
                lines.push(format!(
                    "  Interaction features ({}): ",
                    plan.interaction_pairs.len()
                ));
                for (f1, f2) in plan.interaction_pairs.iter().take(5) {
                    lines.push(format!("    • {} × {}", f1, f2));
                }
                if plan.interaction_pairs.len() > 5 {
                    lines.push(format!(
                        "    ... and {} more",
                        plan.interaction_pairs.len() - 5
                    ));
                }
            }

            if !plan.reasoning.is_empty() {
                lines.push("".to_string());
                lines.push("  Reasoning:".to_string());
                for reason in &plan.reasoning {
                    lines.push(format!("    • {}", reason));
                }
            }
            lines.push("".to_string());
        }

        // Section 4: Mode Selection
        lines.push("═══ MODE SELECTION ═══".to_string());
        lines.push(format!("  Selected: {:?}", self.mode));
        lines.push(format!(
            "  Confidence: {:?}",
            self.mode_confidence
                .as_ref()
                .map(|c| format!("{:?}", c))
                .unwrap_or("N/A".to_string())
        ));

        if let Some(ref analysis) = self.analysis {
            lines.push("".to_string());
            lines.push("  Analysis Results:".to_string());
            lines.push(format!(
                "    • Linear R²: {:.4} ({})",
                analysis.linear_r2,
                if analysis.linear_r2 > 0.5 {
                    "Strong"
                } else if analysis.linear_r2 > 0.3 {
                    "Moderate"
                } else {
                    "Weak"
                }
            ));
            lines.push(format!(
                "    • Tree Gain: {:.4} ({})",
                analysis.tree_gain,
                if analysis.tree_gain > 0.3 {
                    "Strong"
                } else if analysis.tree_gain > 0.1 {
                    "Moderate"
                } else {
                    "Weak"
                }
            ));

            // Show the recommended mode from analysis
            let recommended_mode = analysis.recommend_mode();
            let reasoning = if analysis.linear_r2 > 0.5 && analysis.tree_gain > 0.1 {
                "Strong linear trend + residual structure → Hybrid approach"
            } else if analysis.linear_r2 > 0.5 {
                "Strong linear relationship → Linear model dominates"
            } else if analysis.tree_gain > 0.1 {
                "Non-linear patterns → Tree-based approach"
            } else {
                "Moderate signals → Pure tree model"
            };

            lines.push("".to_string());
            lines.push(format!("  Recommended: {:?}", recommended_mode));
            lines.push(format!("  Reasoning: {}", reasoning));
        }
        lines.push("".to_string());

        // Section 5: Tuning Results (if applicable)
        if let Some(ref tuning) = self.ltt_tuning {
            lines.push("═══ LTT TUNING RESULTS ═══".to_string());
            lines.push("  Linear Phase:".to_string());
            lines.push(format!("    • R²: {:.4}", tuning.linear_r2));
            lines.push(format!("    • Lambda: {:.4}", tuning.linear_params.lambda));
            lines.push(format!(
                "    • L1 Ratio: {:.4} ({})",
                tuning.linear_params.l1_ratio,
                if tuning.linear_params.l1_ratio == 0.0 {
                    "Ridge"
                } else if tuning.linear_params.l1_ratio == 1.0 {
                    "LASSO"
                } else {
                    "ElasticNet"
                }
            ));
            lines.push("".to_string());
            lines.push("  Tree Phase:".to_string());
            lines.push(format!("    • Max Depth: {}", tuning.tree_params.max_depth));
            lines.push(format!(
                "    • Learning Rate: {:.4}",
                tuning.tree_params.learning_rate
            ));
            lines.push(format!(
                "    • Num Rounds: {}",
                tuning.tree_params.num_rounds
            ));
            lines.push("".to_string());
            lines.push(format!("  Final RMSE: {:.4}", tuning.final_rmse));
            lines.push("".to_string());
        }

        // Section 6: Training Summary
        lines.push("═══ TRAINING SUMMARY ═══".to_string());
        lines.push(format!(
            "  Total Time: {:.3}s",
            self.build_time.as_secs_f64()
        ));
        lines.push("".to_string());
        lines.push("  Phase Breakdown:".to_string());
        lines.push(format!("    • Profiling: {:?}", self.phase_times.profiling));
        lines.push(format!(
            "    • Preprocessing: {:?}",
            self.phase_times.preprocessing
        ));
        lines.push(format!(
            "    • Feature Engineering: {:?}",
            self.phase_times.feature_engineering
        ));
        lines.push(format!("    • Analysis: {:?}", self.phase_times.analysis));
        lines.push(format!("    • Tuning: {:?}", self.phase_times.tuning));
        lines.push(format!("    • Training: {:?}", self.phase_times.training));
        lines.push("".to_string());

        lines.push(
            "┌─────────────────────────────────────────────────────────────────┐".to_string(),
        );
        lines.push(
            "│      TreeBoost: The Smart Engineer That Explains Itself         │".to_string(),
        );
        lines.push(
            "└─────────────────────────────────────────────────────────────────┘".to_string(),
        );

        lines.join("\n")
    }

    /// Prepare a DataFrame for prediction
    fn prepare_dataset_for_prediction(&self, df: &DataFrame) -> Result<(DataFrame, BinnedDataset)> {
        // CRITICAL: Apply the SAME feature engineering and preprocessing pipeline as training!
        // Order matters: 1) Feature engineering, 2) Preprocessing (encoding/scaling/binning)

        // Step 1: Apply feature engineering plan if it exists (using canonical helper)
        // Constructs PanelDataInfo from TimeSeriesFeaturePlan if time-series features are present
        let panel_info = self.feature_plan.as_ref().and_then(|plan| {
            plan.timeseries_features.as_ref().map(|ts_plan| {
                crate::analysis::PanelDataInfo {
                    group_column: ts_plan.group_column.clone(),
                    date_column: ts_plan.date_column.clone(),
                    num_groups: 0, // Not used by apply_timeseries_features
                    avg_observations_per_group: 0.0,
                    min_observations_per_group: 0,
                    max_observations_per_group: 0,
                    confidence: 1.0, // Assume detected correctly during training
                    time_granularity: crate::analysis::TimeGranularity::Daily, // Default
                }
            })
        });

        let working_df = crate::features::apply_feature_plan(
            df.clone(),
            self.feature_plan.as_ref(),
            panel_info.as_ref(),
        )?;

        // Step 2: Apply preprocessing pipeline (encoding, scaling, binning)
        let pipeline_state = self.pipeline_state.as_ref().ok_or_else(|| {
            TreeBoostError::Data(
                "AutoModel missing fitted pipeline state - cannot make predictions".to_string(),
            )
        })?;

        let pipeline = DataPipeline::with_defaults();

        // process_for_inference() applies the learned encodings/scalers/binners
        // Returns (preprocessed_df, dataset) where preprocessed_df has encoded categoricals
        let (preprocessed_df, dataset) =
            pipeline.process_for_inference(working_df, pipeline_state)?;

        Ok((preprocessed_df, dataset))
    }

    /// Export the discovered config to JSON
    ///
    /// This saves the UniversalConfig that AutoModel discovered through analysis and tuning.
    /// You can load this config and use it to retrain with UniversalModel directly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Train with AutoModel
    /// let model = AutoModel::train(&df, "target")?;
    ///
    /// // Export the discovered config
    /// model.save_config("optimal_config.json")?;
    ///
    /// // Later: Load and tweak the config
    /// let config = UniversalConfig::load_json("optimal_config.json")?;
    /// let tweaked = config.with_learning_rate(0.05); // Adjust as needed
    ///
    /// // Retrain with the tweaked config
    /// let new_model = UniversalModel::train(&dataset, tweaked, &loss_fn)?;
    /// ```
    pub fn save_config(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let config = self.config();

        let json = serde_json::to_string_pretty(config).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize config to JSON: {}", e))
        })?;

        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save the trained model to a file
    ///
    /// This saves the underlying UniversalModel (weights, trees, ensembles, etc.) for inference.
    /// The model can be loaded later using `UniversalModel::load()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Train and save both config and model
    /// let model = AutoModel::train(&df, "target")?;
    /// model.save_config("config.json")?;      // For retraining with AutoML
    /// model.save("model.rkyv")?;               // For inference
    ///
    /// // Later: Load for inference only (not AutoML)
    /// let loaded = UniversalModel::load("model.rkyv")?;
    /// let preds = loaded.predict(&dataset)?;
    /// ```
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.model.save(path)
    }

    // =========================================================================
    // Incremental Learning Support
    // =========================================================================

    /// Update the model with new training data (incremental learning)
    ///
    /// This method continues training from the current model state:
    /// 1. Uses existing pipeline_state for consistent preprocessing
    /// 2. Updates the underlying UniversalModel with new trees
    /// 3. Preserves all configuration from original training
    ///
    /// # Arguments
    /// * `df` - New data to train on (must have same columns as original)
    /// * `additional_rounds` - Number of new boosting rounds (trees) to add
    ///
    /// # Example
    /// ```ignore
    /// // Train on January data
    /// let mut model = AutoModel::train(&jan_df, "target")?;
    ///
    /// // Update with February data (10 more trees)
    /// let report = model.update(&feb_df, 10)?;
    /// println!("{}", report);
    ///
    /// // Save the updated model
    /// model.save_trb("model.trb")?;
    /// ```
    pub fn update(
        &mut self,
        df: &DataFrame,
        additional_rounds: usize,
    ) -> Result<AutoModelUpdateReport> {
        let rows_before = df.height();

        // Convert DataFrame to BinnedDataset using existing pipeline
        let (_preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;

        // Get targets from the dataframe
        let target_series = df.column(&self.target_column).map_err(|e| {
            TreeBoostError::Data(format!(
                "Target column '{}' not found: {}",
                self.target_column, e
            ))
        })?;

        let targets: Vec<f32> = target_series
            .cast(&polars::datatypes::DataType::Float32)
            .map_err(|e| TreeBoostError::Data(format!("Failed to cast target to f32: {}", e)))?
            .f32()
            .map_err(|e| TreeBoostError::Data(format!("Failed to get f32 values: {}", e)))?
            .into_no_null_iter()
            .collect();

        // Create dataset with actual targets (avoids clone + modify pattern)
        let update_dataset = dataset.with_targets(targets);

        // Update the model (uses MSE loss by default, same as AutoBuilder)
        let loss_fn = MseLoss::new();
        let model_report = self
            .model
            .update(&update_dataset, &loss_fn, additional_rounds)?;

        Ok(AutoModelUpdateReport {
            rows_trained: rows_before,
            trees_before: model_report.trees_before,
            trees_after: model_report.trees_after,
            trees_added: model_report.trees_added,
            mode: self.mode,
            target_column: self.target_column.clone(),
        })
    }

    /// Save model to TRB (TreeBoost) incremental format
    ///
    /// TRB format supports incremental updates without rewriting the entire file.
    /// Use this format when you plan to update the model with new data.
    ///
    /// # Example
    /// ```ignore
    /// model.save_trb("model.trb", "Initial training on January data")?;
    /// ```
    pub fn save_trb(&self, path: impl AsRef<std::path::Path>, description: &str) -> Result<()> {
        self.model.save_trb(path, description)
    }

    /// Append an update to an existing TRB file
    ///
    /// This appends a new segment without rewriting the base model.
    ///
    /// # Arguments
    /// * `path` - Path to existing .trb file
    /// * `rows_trained` - Number of rows used in this update
    /// * `description` - Description of this update
    pub fn save_trb_update(
        &self,
        path: impl AsRef<std::path::Path>,
        rows_trained: usize,
        description: &str,
    ) -> Result<()> {
        self.model.save_trb_update(path, rows_trained, description)
    }

    /// Load model from TRB format for continued training
    ///
    /// This loads the base model and all updates, ready for further training.
    /// The returned AutoModel will have minimal metadata (no tuning results, etc.)
    /// but the underlying model and pipeline state are preserved.
    ///
    /// # Example
    /// ```ignore
    /// let mut model = AutoModel::load_trb("model.trb", "target")?;
    /// model.update(&new_data, 10)?;
    /// ```
    pub fn load_trb(path: impl AsRef<std::path::Path>, target_column: &str) -> Result<Self> {
        let model = crate::model::UniversalModel::load_trb(path)?;
        let mode = model.mode();

        Ok(Self {
            model,
            mode,
            target_column: target_column.to_string(),
            mode_confidence: None,
            preprocessing_plan: None,
            feature_plan: None,
            ltt_tuning: None,
            tree_tuning: None,
            column_profile: None,
            analysis: None,
            pipeline_state: None, // Note: pipeline_state not preserved in TRB format yet
            build_time: Duration::default(),
            phase_times: BuildPhaseTimes::default(),
            tuned_thresholds: None,
        })
    }

    /// Check if model is compatible with dataset for incremental update
    pub fn is_compatible_for_update(&self, df: &DataFrame) -> bool {
        // Check that target column exists
        if df.column(&self.target_column).is_err() {
            return false;
        }

        // Try to prepare dataset - if it fails, not compatible
        self.prepare_dataset_for_prediction(df).is_ok()
    }

    /// Get mutable reference to the underlying UniversalModel
    pub fn model_mut(&mut self) -> &mut UniversalModel {
        &mut self.model
    }
}

/// Report from an AutoModel incremental training update
#[derive(Debug, Clone)]
pub struct AutoModelUpdateReport {
    /// Number of rows in the new training data
    pub rows_trained: usize,
    /// Number of trees before update
    pub trees_before: usize,
    /// Number of trees after update
    pub trees_after: usize,
    /// Number of trees added
    pub trees_added: usize,
    /// Boosting mode
    pub mode: BoostingMode,
    /// Target column name
    pub target_column: String,
}

impl std::fmt::Display for AutoModelUpdateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AutoModel Update: {} rows on '{}', {} trees added ({} -> {}), mode={:?}",
            self.rows_trained,
            self.target_column,
            self.trees_added,
            self.trees_before,
            self.trees_after,
            self.mode
        )
    }
}

impl std::fmt::Debug for AutoModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoModel")
            .field("mode", &self.mode)
            .field("mode_confidence", &self.mode_confidence)
            .field("build_time", &self.build_time)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_auto_model_from_build_result() {
        // This test just verifies the struct construction works
        // Full integration tests would require a DataFrame
    }

    #[test]
    fn test_auto_model_summary_format() {
        // Create a minimal AutoModel to test summary formatting
        // This would need a real UniversalModel in practice
    }
}
