//! Configuration types for AutoBuilder
//!
//! This module contains all configuration structs, enums, and result types
//! used by the AutoBuilder interface.

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis};
use crate::dataset::feature_extractor::LinearFeatureConfig;
use crate::defaults::{auto as auto_defaults, tuning::seeds as seeds_defaults};
use crate::ensemble::{MultiSeedConfig, SelectionConfig, StackingConfig};
use crate::features::FeaturePlan;
use crate::model::progress::{ProgressCallback, QuietProgress};
use crate::model::universal::ModeSelection;
use crate::model::{BoostingMode, UniversalModel};
use crate::preprocessing::PreprocessingPlan;
use crate::tuner::ltt::LttTuningResult;
use crate::tuner::{OptimizationMetric, TaskType as TunerTaskType};
use std::sync::Arc;
use std::time::Duration;

/// Granular verbosity control for AutoBuilder output.
///
/// Each field controls a category of output. Use `Verbosity::all()` for
/// everything, `Verbosity::light()` for training output without noisy details,
/// or construct manually for fine-grained control.
#[derive(Debug, Clone)]
pub struct Verbosity {
    /// Show preprocessing step details (imputers, encoders per column)
    pub preprocessing: bool,
    /// Show column profiling results
    pub profiling: bool,
    /// Show feature engineering details
    pub features: bool,
    /// Show autotuner progress
    pub tuning: bool,
    /// Show general training progress
    pub training: bool,
}

impl Verbosity {
    /// Everything enabled
    pub fn all() -> Self {
        Self {
            preprocessing: true,
            profiling: true,
            features: true,
            tuning: true,
            training: true,
        }
    }

    /// Training output without noisy preprocessing details
    pub fn light() -> Self {
        Self {
            preprocessing: false,
            profiling: true,
            features: true,
            tuning: true,
            training: true,
        }
    }

    /// All output disabled
    pub fn silent() -> Self {
        Self {
            preprocessing: false,
            profiling: false,
            features: false,
            tuning: false,
            training: false,
        }
    }

    /// Whether any verbose output is enabled (used for general checks)
    pub fn enabled(&self) -> bool {
        self.training || self.profiling || self.features || self.tuning || self.preprocessing
    }
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::silent()
    }
}

impl From<bool> for Verbosity {
    fn from(v: bool) -> Self {
        if v {
            Self::light()
        } else {
            Self::silent()
        }
    }
}

/// Tuning intensity level for AutoBuilder's hyperparameter optimization.
///
/// ## When to Use TuningLevel
///
/// Use `TuningLevel` when working with **`AutoBuilder`** or **`AutoConfig`**.
/// This enum controls AutoBuilder's automatic tuning pipeline intensity.
///
/// ## Relationship to Other Preset Enums
///
/// TreeBoost has several tuning-related preset enums for different contexts:
///
/// | Enum | Context | Purpose |
/// |------|---------|---------|
/// | **`TuningLevel`** | `AutoBuilder`, `AutoConfig` | High-level: AutoML tuning intensity |
/// | **`TunerPreset`** | `AutoTuner`, `TunerConfig` | Mid-level: Manual tuning intensity |
/// | **`TreeTunerPreset`** | `TreeTunerConfig` | Low-level: Tree-only tuning intensity |
///
/// **Mapping between presets:**
/// - `TuningLevel::Quick` ≈ `TunerPreset::Quick` ≈ `TreeTunerPreset::Quick`
/// - `TuningLevel::Standard` ≈ `TunerPreset::Balanced` ≈ `TreeTunerPreset::Standard`
/// - `TuningLevel::Thorough` ≈ `TunerPreset::Thorough` ≈ `TreeTunerPreset::Thorough`
/// - `TuningLevel::None` = No tuning (use provided hyperparameters)
///
/// ## Variants
///
/// ### `Quick`
/// Minimal tuning with sensible defaults.
/// - **Best for**: Quick experiments, small datasets, CI/debugging
/// - **Time**: Seconds to minutes
/// - **Quality**: Good baseline, not optimized
///
/// ### `Standard` (Default)
/// Moderate tuning with balanced speed and quality.
/// - **Best for**: Most real-world use cases
/// - **Time**: Minutes to tens of minutes
/// - **Quality**: Well-tuned, production-ready
///
/// ### `Thorough`
/// Extensive hyperparameter search for maximum accuracy.
/// - **Best for**: Competitions, research, when accuracy > time
/// - **Time**: Tens of minutes to hours
/// - **Quality**: Highly optimized
///
/// ### `None`
/// No tuning - use provided hyperparameters as-is.
/// - **Best for**: Already-tuned configs, reproducibility
/// - **Time**: Instant (no tuning overhead)
/// - **Quality**: Depends on provided hyperparameters
///
/// ## Examples
///
/// ```ignore
/// use treeboost::{AutoBuilder, TuningLevel};
///
/// // Quick experiments
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::Quick)
///     .fit(&df, "target")?;
///
/// // Production model (default)
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::Standard)  // or omit (default)
///     .fit(&df, "target")?;
///
/// // Maximum accuracy
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::Thorough)
///     .fit(&df, "target")?;
///
/// // No tuning (use specific hyperparameters)
/// let config = UniversalConfig::new().with_num_rounds(100).with_learning_rate(0.1);
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::None)
///     .with_config(config)
///     .fit(&df, "target")?;
/// ```
///
/// ## See Also
///
/// - [`crate::tuner::TunerPreset`] - For manual tuning with `AutoTuner`
/// - [`TreeTunerPreset`] - For tree-specific tuning with `TreeTunerConfig`
/// - [`crate::model::AutoBuilder::with_tuning`] - Set the tuning level
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TuningLevel {
    /// Minimal tuning with sensible defaults
    Quick,

    /// Moderate tuning with balanced speed and quality
    #[default]
    Standard,

    /// Extensive tuning for maximum accuracy
    Thorough,

    /// No tuning - use provided hyperparameters as-is
    None,
}

impl TuningLevel {
    /// Convert to the equivalent `TunerPreset` for use with `AutoTuner`.
    ///
    /// This provides a sensible mapping between `TuningLevel` (used in `AutoBuilder`)
    /// and `TunerPreset` (used in `AutoTuner`/`TunerConfig`).
    ///
    /// # Mapping
    ///
    /// - `Quick` → `TunerPreset::Quick`
    /// - `Standard` → `TunerPreset::Balanced`
    /// - `Thorough` → `TunerPreset::Thorough`
    /// - `None` → `TunerPreset::Quick` (minimal tuning as fallback)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::{TuningLevel, TunerPreset};
    ///
    /// let level = TuningLevel::Standard;
    /// let tuner_preset = level.to_tuner_preset();
    /// assert_eq!(tuner_preset, TunerPreset::Balanced);
    /// ```
    pub fn to_tuner_preset(self) -> crate::tuner::TunerPreset {
        use crate::tuner::TunerPreset;
        match self {
            TuningLevel::Quick => TunerPreset::Quick,
            TuningLevel::Standard => TunerPreset::Balanced,
            TuningLevel::Thorough => TunerPreset::Thorough,
            TuningLevel::None => TunerPreset::Quick, // Fallback to minimal tuning
        }
    }

    /// Convert to the equivalent `TreeTunerPreset` for use with `TreeTunerConfig`.
    ///
    /// This provides a sensible mapping between `TuningLevel` (used in `AutoBuilder`)
    /// and `TreeTunerPreset` (used in `TreeTunerConfig`).
    ///
    /// # Mapping
    ///
    /// - `Quick` → `TreeTunerPreset::Quick`
    /// - `Standard` → `TreeTunerPreset::Standard`
    /// - `Thorough` → `TreeTunerPreset::Thorough`
    /// - `None` → `TreeTunerPreset::Quick` (minimal tuning as fallback)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::{TuningLevel, TreeTunerPreset};
    ///
    /// let level = TuningLevel::Thorough;
    /// let tree_preset = level.to_tree_tuner_preset();
    /// assert_eq!(tree_preset, TreeTunerPreset::Thorough);
    /// ```
    pub fn to_tree_tuner_preset(self) -> TreeTunerPreset {
        match self {
            TuningLevel::Quick => TreeTunerPreset::Quick,
            TuningLevel::Standard => TreeTunerPreset::Standard,
            TuningLevel::Thorough => TreeTunerPreset::Thorough,
            TuningLevel::None => TreeTunerPreset::Quick, // Fallback to minimal tuning
        }
    }
}

/// Target bound configuration for bounded regression problems.
///
/// Controls how target bounds are determined and what transformation to apply.
/// This configuration is stored in the Pipeline and can be replicated from saved models.
///
/// # Modes
///
/// | Mode | Bounds Source | Transformation | Use When |
/// |------|--------------|----------------|----------|
/// | `Logit { min, max }` | User-specified | Logit transform | Known theoretical bounds (e.g., scores [0, 100]) |
/// | `LogitEmpirical` | From data min/max | Logit transform | Want logit but let data determine bounds |
/// | `Clamp { min, max }` | User-specified | Clamp only | Logit causes issues, simpler approach |
/// | `ClampEmpirical` | From data min/max | Clamp only | Simple bounded regression from data |
/// | `None` | N/A | No transform | Unbounded targets |
///
/// # Example
///
/// ```ignore
/// use treeboost::model::{AutoConfig, TargetBoundConfig};
///
/// // Exam scores with known bounds [0, 100]
/// let config = AutoConfig::new()
///     .with_target_bound_config(TargetBoundConfig::Logit { min: 0.0, max: 100.0 });
///
/// // Let data determine bounds, use simple clamp
/// let config = AutoConfig::new()
///     .with_target_bound_config(TargetBoundConfig::ClampEmpirical);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TargetBoundConfig {
    /// No target transformation (unbounded regression)
    #[default]
    None,

    /// Logit transform with user-specified bounds.
    /// Maps [min, max] → (-∞, +∞) via logit, predictions via sigmoid back to [min, max].
    /// Best when theoretical bounds are known (e.g., exam scores always [0, 100]).
    Logit { min: f32, max: f32 },

    /// Logit transform with bounds from training data min/max.
    /// Automatically determines bounds from empirical data.
    LogitEmpirical,

    /// Clamp-only with user-specified bounds.
    /// No transform during training, predictions clamped to [min, max] at inference.
    /// Simpler than Logit, use when Logit causes issues.
    Clamp { min: f32, max: f32 },

    /// Clamp-only with bounds from training data min/max.
    /// Simpler bounded regression using empirical bounds.
    ClampEmpirical,
}

/// Ensemble strategy for AutoBuilder (PureTree only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoEnsembleMethod {
    /// Simple average of selected models
    SimpleAverage,
    /// Ridge stacking of selected models (default)
    RidgeStacking,
}

/// Ensemble configuration for AutoBuilder
#[derive(Debug, Clone)]
pub struct AutoEnsembleConfig {
    pub method: AutoEnsembleMethod,
    pub multi_seed: MultiSeedConfig,
    pub selection: SelectionConfig,
    pub stacking: StackingConfig,
}

impl Default for AutoEnsembleConfig {
    fn default() -> Self {
        Self {
            method: AutoEnsembleMethod::RidgeStacking,
            multi_seed: MultiSeedConfig::default(),
            selection: SelectionConfig::default(),
            stacking: StackingConfig::default(),
        }
    }
}

impl AutoEnsembleConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, method: AutoEnsembleMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_multi_seed_config(mut self, config: MultiSeedConfig) -> Self {
        self.multi_seed = config;
        self
    }

    pub fn with_selection_config(mut self, config: SelectionConfig) -> Self {
        self.selection = config;
        self
    }

    pub fn with_stacking_config(mut self, config: StackingConfig) -> Self {
        self.stacking = config;
        self
    }
}

/// Feature engineering mode for AutoBuilder.
///
/// Controls which feature engineering techniques to apply.
///
/// # Examples
///
/// ```ignore
/// use treeboost::{AutoBuilder, FeatureEngineeringMode};
///
/// // Disable all feature engineering
/// let model = AutoBuilder::new()
///     .with_feature_engineering(FeatureEngineeringMode::None)
///     .fit(&df, "target")?;
///
/// // Use default feature engineering
/// let model = AutoBuilder::new()
///     .with_feature_engineering(FeatureEngineeringMode::Default)
///     .fit(&df, "target")?;
///
/// // Custom feature configuration
/// let model = AutoBuilder::new()
///     .with_feature_engineering(FeatureEngineeringMode::Custom(
///         SmartFeatureConfig::default()
///             .with_enable_polynomial(false)
///             .with_top_n_interactions(10)
///     ))
///     .fit(&df, "target")?;
/// ```
#[derive(Debug, Clone, Default)]
pub enum FeatureEngineeringMode {
    /// No feature engineering
    None,
    /// Minimal features (polynomial only, max 20)
    Minimal,
    /// Standard features (polynomial + interactions, max 50)
    #[default]
    Default,
    /// Aggressive features (all types, max 100)
    Aggressive,
    /// Custom configuration
    Custom(crate::features::SmartFeatureConfig),
}

impl FeatureEngineeringMode {
    /// Check if feature engineering is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get the feature configuration
    pub fn get_config(&self) -> Option<crate::features::SmartFeatureConfig> {
        use crate::features::{SmartFeatureConfig, SmartFeaturePreset};
        match self {
            Self::None => None,
            Self::Minimal => {
                Some(SmartFeatureConfig::default().with_preset(SmartFeaturePreset::Minimal))
            }
            Self::Default => Some(SmartFeatureConfig::default()),
            Self::Aggressive => {
                Some(SmartFeatureConfig::default().with_preset(SmartFeaturePreset::Aggressive))
            }
            Self::Custom(config) => Some(config.clone()),
        }
    }
}

/// Preprocessing mode for AutoBuilder.
///
/// Controls which preprocessing transformations to apply.
///
/// # Examples
///
/// ```ignore
/// use treeboost::{AutoBuilder, PreprocessingMode};
///
/// // No preprocessing
/// let model = AutoBuilder::new()
///     .with_preprocessing(PreprocessingMode::None)
///     .fit(&df, "target")?;
///
/// // Strict preprocessing (aggressive outlier removal, missing value handling)
/// let model = AutoBuilder::new()
///     .with_preprocessing(PreprocessingMode::Strict)
///     .fit(&df, "target")?;
///
/// // Custom preprocessing configuration
/// let model = AutoBuilder::new()
///     .with_preprocessing(PreprocessingMode::Custom(
///         SmartPreprocessConfig::default()
///             .with_preset(SmartPreprocessPreset::Permissive)
///     ))
///     .fit(&df, "target")?;
/// ```
#[derive(Debug, Clone, Default)]
pub enum PreprocessingMode {
    /// No preprocessing
    None,
    /// Minimal preprocessing (basic imputation only)
    Minimal,
    /// Standard preprocessing (default)
    #[default]
    Default,
    /// Strict preprocessing (aggressive cleaning)
    Strict,
    /// Custom configuration
    Custom(crate::preprocessing::SmartPreprocessConfig),
}

impl PreprocessingMode {
    /// Check if preprocessing is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get the preprocessing configuration
    pub fn get_config(&self) -> Option<crate::preprocessing::SmartPreprocessConfig> {
        use crate::preprocessing::{SmartPreprocessConfig, SmartPreprocessPreset};
        match self {
            Self::None => None,
            Self::Minimal => Some(
                SmartPreprocessConfig::default().with_preset(SmartPreprocessPreset::Permissive),
            ),
            Self::Default => Some(SmartPreprocessConfig::default()),
            Self::Strict => {
                Some(SmartPreprocessConfig::default().with_preset(SmartPreprocessPreset::Strict))
            }
            Self::Custom(config) => Some(config.clone()),
        }
    }
}

/// Ensemble mode for AutoBuilder.
///
/// Controls whether and how to build model ensembles (PureTree only).
///
/// # Examples
///
/// ```ignore
/// use treeboost::{AutoBuilder, EnsembleMode, AutoEnsembleMethod};
///
/// // No ensemble
/// let model = AutoBuilder::new()
///     .with_ensemble(EnsembleMode::Disabled)
///     .fit(&df, "target")?;
///
/// // Default ensemble (Ridge stacking with 5 seeds)
/// let model = AutoBuilder::new()
///     .with_ensemble(EnsembleMode::Default)
///     .fit(&df, "target")?;
///
/// // Simple averaging instead of stacking
/// let model = AutoBuilder::new()
///     .with_ensemble(EnsembleMode::WithMethod(AutoEnsembleMethod::SimpleAverage))
///     .fit(&df, "target")?;
///
/// // Custom ensemble configuration
/// let model = AutoBuilder::new()
///     .with_ensemble(EnsembleMode::Custom(
///         AutoEnsembleConfig::new()
///             .with_method(AutoEnsembleMethod::RidgeStacking)
///             .with_multi_seed_config(MultiSeedConfig::new(10))
///     ))
///     .fit(&df, "target")?;
/// ```
#[derive(Debug, Clone, Default)]
pub enum EnsembleMode {
    /// No ensemble (single model)
    #[default]
    Disabled,
    /// Default ensemble (Ridge stacking with 5 seeds)
    Default,
    /// Ensemble with specific method
    WithMethod(AutoEnsembleMethod),
    /// Custom ensemble configuration
    Custom(AutoEnsembleConfig),
}

impl EnsembleMode {
    /// Check if ensemble is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Get the ensemble configuration
    pub fn get_config(&self) -> Option<AutoEnsembleConfig> {
        match self {
            Self::Disabled => None,
            Self::Default => Some(AutoEnsembleConfig::default()),
            Self::WithMethod(method) => Some(AutoEnsembleConfig::default().with_method(*method)),
            Self::Custom(config) => Some(config.clone()),
        }
    }
}

/// AutoBuilder configuration
pub struct AutoConfig {
    /// Tuning intensity
    pub tuning_level: TuningLevel,

    /// Validation split ratio (default: 0.2)
    pub val_ratio: f32,

    /// Feature engineering mode (replaces auto_features + feature_config)
    pub feature_engineering: FeatureEngineeringMode,

    /// Preprocessing mode (replaces auto_preprocessing)
    pub preprocessing: PreprocessingMode,

    /// Mode selection strategy (Auto or Fixed)
    pub mode_selection: ModeSelection,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Verbose output control
    pub verbose: Verbosity,

    /// Maximum time budget for training (None = no limit)
    /// AutoBuilder will adapt tuning intensity to fit within this budget
    pub time_budget: Option<Duration>,

    /// Progress callback for tracking training phases
    pub progress_callback: Arc<dyn ProgressCallback>,

    /// Configuration for extracting features for linear models (LinearThenTree)
    pub linear_feature_config: LinearFeatureConfig,

    /// Custom UniversalConfig to use (bypasses tuning if provided with TuningLevel::None)
    pub custom_config: Option<crate::model::UniversalConfig>,

    /// Ensemble mode (replaces ensemble: Option<AutoEnsembleConfig>)
    pub ensemble: EnsembleMode,

    /// Custom tree tuner configuration (overrides tuning_level for tree-based modes)
    pub tree_tuner_config: Option<TreeTunerConfig>,

    /// Output directory for saving final model artifacts (model.rkyv, config.json)
    /// If None, artifacts are not automatically saved
    ///
    /// Note: config.json now contains everything (hyperparameters, feature plan, preprocessing plan, target column)
    pub model_output_dir: Option<std::path::PathBuf>,

    /// Backend type for computation (Auto, Cuda, Wgpu, Scalar, etc.)
    pub backend_type: crate::backend::BackendType,

    /// Skip final model training (discovery mode)
    ///
    /// When true:
    /// - Runs profiling, feature engineering, preprocessing, and hyperparameter tuning
    /// - Saves config.json with discovered settings
    /// - Does NOT train final model or save model.rkyv
    ///
    /// Use this for fast config discovery on sampled data, then train on full data later.
    pub skip_training: bool,

    /// Target bound configuration for bounded regression.
    ///
    /// Controls how target bounds are determined and what transformation to apply.
    /// See [`TargetBoundConfig`] for available modes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, TargetBoundConfig};
    ///
    /// // Exam scores: known bounds [0, 100] with Logit transform
    /// let config = AutoConfig::new()
    ///     .with_target_bound_config(TargetBoundConfig::Logit { min: 0.0, max: 100.0 });
    ///
    /// // Simple clamp using empirical min/max from data
    /// let config = AutoConfig::new()
    ///     .with_target_bound_config(TargetBoundConfig::ClampEmpirical);
    /// ```
    ///
    /// Default is `TargetBoundConfig::None` (no transformation).
    pub target_bound_config: TargetBoundConfig,

    /// Custom user-defined features.
    ///
    /// These features are applied BEFORE automatic feature engineering and encode
    /// domain-specific knowledge that the AutoML system cannot discover on its own.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, CustomFeature, FeatureOp};
    ///
    /// // Manual formula with LUT mappings (like winning Kaggle notebooks)
    /// let manual_formula = CustomFeature::formula("manual_formula")
    ///     .add_term("study_hours", 6.0)
    ///     .add_term("class_attendance", 0.35)
    ///     .add_term("sleep_hours", 1.5)
    ///     .add_lut("sleep_quality", &[("good", 5.0), ("average", 0.0), ("poor", -5.0)])
    ///     .add_lut("study_method", &[("coaching", 10.0), ("mixed", 5.0), ("self-study", 0.0)])
    ///     .build();
    ///
    /// // Trigonometric feature for cyclical patterns
    /// let trig_feature = CustomFeature::new(
    ///     "study_hours_sin",
    ///     FeatureOp::sin("study_hours", 12.0)
    /// );
    ///
    /// let config = AutoConfig::new()
    ///     .with_custom_features(vec![manual_formula, trig_feature]);
    /// ```
    pub custom_features: Vec<super::CustomFeature>,
}

impl Clone for AutoConfig {
    fn clone(&self) -> Self {
        Self {
            tuning_level: self.tuning_level,
            val_ratio: self.val_ratio,
            feature_engineering: self.feature_engineering.clone(),
            preprocessing: self.preprocessing.clone(),
            mode_selection: self.mode_selection.clone(),
            seed: self.seed,
            verbose: self.verbose.clone(),
            time_budget: self.time_budget,
            progress_callback: Arc::clone(&self.progress_callback),
            linear_feature_config: self.linear_feature_config.clone(),
            custom_config: self.custom_config.clone(),
            ensemble: self.ensemble.clone(),
            tree_tuner_config: self.tree_tuner_config.clone(),
            model_output_dir: self.model_output_dir.clone(),
            backend_type: self.backend_type,
            skip_training: self.skip_training,
            target_bound_config: self.target_bound_config.clone(),
            custom_features: self.custom_features.clone(),
        }
    }
}

impl std::fmt::Debug for AutoConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoConfig")
            .field("tuning_level", &self.tuning_level)
            .field("val_ratio", &self.val_ratio)
            .field("feature_engineering", &self.feature_engineering)
            .field("preprocessing", &self.preprocessing)
            .field("mode_selection", &self.mode_selection)
            .field("seed", &self.seed)
            .field("verbose", &self.verbose)
            .field("time_budget", &self.time_budget)
            .field("progress_callback", &"<callback>")
            .field("linear_feature_config", &self.linear_feature_config)
            .field("ensemble", &self.ensemble)
            .field("tree_tuner_config", &self.tree_tuner_config)
            .field("backend_type", &self.backend_type)
            .field("skip_training", &self.skip_training)
            .field("target_bound_config", &self.target_bound_config)
            .field(
                "custom_features",
                &format!("{} features", self.custom_features.len()),
            )
            .finish()
    }
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            tuning_level: TuningLevel::Standard,
            val_ratio: auto_defaults::DEFAULT_VALIDATION_RATIO,
            feature_engineering: FeatureEngineeringMode::Default,
            preprocessing: PreprocessingMode::Default,
            mode_selection: ModeSelection::Auto, // AutoBuilder defaults to automatic mode selection
            seed: seeds_defaults::DEFAULT_SEED,
            verbose: Verbosity::silent(),
            time_budget: None,
            progress_callback: Arc::new(QuietProgress),
            linear_feature_config: LinearFeatureConfig::default(),
            custom_config: None,
            ensemble: EnsembleMode::Disabled,
            tree_tuner_config: None,
            model_output_dir: None,
            backend_type: crate::backend::BackendType::Auto,
            skip_training: false,
            target_bound_config: TargetBoundConfig::None,
            custom_features: Vec::new(),
        }
    }
}

impl AutoConfig {
    /// Create a new AutoConfig with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tuning level
    pub fn with_tuning(mut self, level: TuningLevel) -> Self {
        self.tuning_level = level;
        self
    }

    /// Set validation split ratio for **random** train/validation split.
    ///
    /// **For cross-sectional (i.i.d.) data only** where rows are independent.
    /// For time-series or panel data, use `AutoBuilder::with_presplit_validation` instead.
    pub fn with_random_validation_split(mut self, ratio: f32) -> Self {
        self.val_ratio = ratio.clamp(0.1, 0.4);
        self
    }

    /// Set feature engineering mode
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoConfig, FeatureEngineeringMode};
    /// use treeboost::features::SmartFeatureConfig;
    ///
    /// // Disable feature engineering
    /// let config = AutoConfig::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::None);
    ///
    /// // Use default features
    /// let config = AutoConfig::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::Default);
    ///
    /// // Custom configuration
    /// let config = AutoConfig::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::Custom(
    ///         SmartFeatureConfig::default()
    ///             .with_enable_polynomial(false)
    ///             .with_top_n_interactions(10)
    ///     ));
    /// ```
    pub fn with_feature_engineering(mut self, mode: FeatureEngineeringMode) -> Self {
        self.feature_engineering = mode;
        self
    }

    /// Set preprocessing mode
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoConfig, PreprocessingMode};
    ///
    /// // No preprocessing
    /// let config = AutoConfig::new()
    ///     .with_preprocessing(PreprocessingMode::None);
    ///
    /// // Strict preprocessing (aggressive cleaning)
    /// let config = AutoConfig::new()
    ///     .with_preprocessing(PreprocessingMode::Strict);
    /// ```
    pub fn with_preprocessing(mut self, mode: PreprocessingMode) -> Self {
        self.preprocessing = mode;
        self
    }

    /// Set mode selection strategy
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, ModeSelection, BoostingMode};
    ///
    /// // Automatic mode selection (default)
    /// let config = AutoConfig::default()
    ///     .with_mode_selection(ModeSelection::Auto);
    ///
    /// // Force a specific mode
    /// let config = AutoConfig::default()
    ///     .with_mode_selection(ModeSelection::Fixed(BoostingMode::PureTree));
    /// ```
    pub fn with_mode_selection(mut self, selection: ModeSelection) -> Self {
        self.mode_selection = selection;
        self
    }

    /// Force a specific boosting mode (convenience method)
    ///
    /// Equivalent to `with_mode_selection(ModeSelection::Fixed(mode))`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, BoostingMode};
    ///
    /// let config = AutoConfig::default()
    ///     .with_mode(BoostingMode::LinearThenTree);
    /// ```
    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.mode_selection = ModeSelection::Fixed(mode);
        self
    }

    /// Set ensemble mode (PureTree only)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoConfig, EnsembleMode, AutoEnsembleMethod, AutoEnsembleConfig};
    ///
    /// // No ensemble
    /// let config = AutoConfig::new()
    ///     .with_ensemble(EnsembleMode::Disabled);
    ///
    /// // Default ensemble
    /// let config = AutoConfig::new()
    ///     .with_ensemble(EnsembleMode::Default);
    ///
    /// // Specific method
    /// let config = AutoConfig::new()
    ///     .with_ensemble(EnsembleMode::WithMethod(AutoEnsembleMethod::SimpleAverage));
    ///
    /// // Custom configuration
    /// let config = AutoConfig::new()
    ///     .with_ensemble(EnsembleMode::Custom(AutoEnsembleConfig::new()));
    /// ```
    pub fn with_ensemble(mut self, mode: EnsembleMode) -> Self {
        self.ensemble = mode;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose.into();
        self
    }

    /// Set backend type for computation
    ///
    /// Use BackendType::Auto (default) for automatic detection: CUDA > WGPU > Scalar
    pub fn with_backend(mut self, backend_type: crate::backend::BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Enable skip training mode (discovery only)
    ///
    /// When enabled, AutoBuilder will:
    /// - Run profiling, feature engineering, preprocessing, and hyperparameter tuning
    /// - Save config.json with all discovered settings
    /// - Skip final model training (no model.rkyv saved)
    ///
    /// Use this for fast config discovery on sampled data.
    /// Then load the config and train with UniversalModel on full data.
    ///
    /// # Example
    /// ```ignore
    /// // Discovery on sample data
    /// let config = AutoConfig::new().with_skip_training(true);
    /// AutoBuilder::with_config(config).fit(&sample_df, "target")?;
    /// // Saves config.json only
    ///
    /// // Train on full data
    /// let config = UniversalConfig::load("config.json")?;
    /// let model = UniversalModel::train(&full_dataset, config, &loss)?;
    /// ```
    pub fn with_skip_training(mut self, skip: bool) -> Self {
        self.skip_training = skip;
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
    /// use treeboost::model::{AutoConfig, TargetBoundConfig};
    ///
    /// // Exam scores: known bounds [0, 100] with Logit transform
    /// let config = AutoConfig::new()
    ///     .with_target_bound_config(TargetBoundConfig::Logit { min: 0.0, max: 100.0 });
    ///
    /// // Simple clamp using empirical min/max from data
    /// let config = AutoConfig::new()
    ///     .with_target_bound_config(TargetBoundConfig::ClampEmpirical);
    ///
    /// // Clamp to known bounds (no transform during training)
    /// let config = AutoConfig::new()
    ///     .with_target_bound_config(TargetBoundConfig::Clamp { min: 0.0, max: 100.0 });
    /// ```
    pub fn with_target_bound_config(mut self, config: TargetBoundConfig) -> Self {
        self.target_bound_config = config;
        self
    }

    /// Set custom user-defined features.
    ///
    /// Custom features are applied BEFORE automatic feature engineering and encode
    /// domain-specific knowledge that the AutoML system cannot discover on its own.
    ///
    /// These features are:
    /// - Applied in order before any automatic feature engineering
    /// - Serialized with the model for consistent inference
    /// - Available for both training and prediction
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, CustomFeature, FeatureOp};
    ///
    /// // Manual formula like winning Kaggle notebooks
    /// let manual_formula = CustomFeature::formula("manual_formula")
    ///     .add_term("study_hours", 6.0)
    ///     .add_term("class_attendance", 0.35)
    ///     .add_lut("sleep_quality", &[("good", 5.0), ("average", 0.0), ("poor", -5.0)])
    ///     .build();
    ///
    /// // Sin transform for cyclical patterns
    /// let trig = CustomFeature::new("study_hours_sin", FeatureOp::sin("study_hours", 12.0));
    ///
    /// let config = AutoConfig::new()
    ///     .with_custom_features(vec![manual_formula, trig]);
    /// ```
    pub fn with_custom_features(mut self, features: Vec<super::CustomFeature>) -> Self {
        self.custom_features = features;
        self
    }

    /// Add a single custom feature.
    ///
    /// Appends to existing custom features rather than replacing them.
    pub fn add_custom_feature(mut self, feature: super::CustomFeature) -> Self {
        self.custom_features.push(feature);
        self
    }

    /// Set explicit target bounds for bounded regression (deprecated)
    ///
    /// **Deprecated:** Use [`with_target_bound_config`] instead for more control.
    ///
    /// This method is equivalent to `with_target_bound_config(TargetBoundConfig::Logit { min, max })`.
    #[deprecated(since = "0.2.0", note = "Use with_target_bound_config() instead")]
    pub fn with_target_bounds(mut self, min: f32, max: f32) -> Self {
        assert!(
            min < max,
            "Target bounds must have min < max, got min={}, max={}",
            min,
            max
        );
        self.target_bound_config = TargetBoundConfig::Logit { min, max };
        self
    }

    /// Set time budget for training
    ///
    /// AutoBuilder will adapt tuning intensity to fit within this budget.
    /// For example, if 60 seconds is allocated and profiling takes 10 seconds,
    /// the remaining 50 seconds will be split between tuning and training.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    /// let config = AutoConfig::new()
    ///     .with_time_budget(Duration::from_secs(60)); // 1 minute max
    /// ```
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = Some(budget);
        self
    }

    /// Set progress callback for tracking training phases
    ///
    /// Use this to receive updates as training progresses through each phase.
    /// Useful for long-running training to show progress to users.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::ConsoleProgress;
    /// use std::sync::Arc;
    ///
    /// let config = AutoConfig::new()
    ///     .with_progress_callback(Arc::new(ConsoleProgress::detailed()));
    /// ```
    pub fn with_progress_callback(mut self, callback: Arc<dyn ProgressCallback>) -> Self {
        self.progress_callback = callback;
        self
    }

    /// Set configuration for linear model features (LinearThenTree mode)
    ///
    /// Use this to control which features are used for the linear component
    /// in LinearThenTree mode. This can improve accuracy by excluding features
    /// that are not useful for linear regression (e.g., high-cardinality categoricals).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::dataset::feature_extractor::LinearFeatureConfig;
    ///
    /// let config = AutoConfig::new()
    ///     .with_linear_feature_config(LinearFeatureConfig::default()
    ///         .with_exclude_patterns(&vec!["id_".to_string(), "rank_".to_string()]));
    /// ```
    pub fn with_linear_feature_config(mut self, config: LinearFeatureConfig) -> Self {
        self.linear_feature_config = config;
        self
    }

    /// Set custom UniversalConfig (overrides tuning)
    ///
    /// Use this to bypass tuning and provide your own hyperparameters.
    /// The custom config will be used regardless of tuning level.
    pub fn with_custom_config(mut self, config: crate::model::UniversalConfig) -> Self {
        self.custom_config = Some(config);
        self
    }

    /// Set output directory for tuning logs (simple API)
    ///
    /// **This is the recommended way** to enable CSV logging for tuning trials.
    /// AutoConfig automatically configures TreeTunerConfig with the output directory
    /// based on your tuning level (Quick, Standard, Thorough).
    ///
    /// The directory will contain timestamped subdirectories (run_YYYYMMDD_HHMMSS/) with:
    /// - `iteration_N.csv` - Trial-by-trial results for each tuning iteration
    /// - `best_params.json` - Best hyperparameters and validation metrics
    /// - `summary.json` - Run metadata
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::path::Path;
    /// use treeboost::model::{AutoConfig, TuningLevel};
    ///
    /// let config = AutoConfig::new()
    ///     .with_output_dir(Path::new("my_experiment/tuning_logs"))  // Simple!
    ///     .with_tuning(TuningLevel::Quick)
    ///     .with_random_validation_split(0.2);
    /// ```
    pub fn with_output_dir(mut self, dir: &std::path::Path) -> Self {
        // Set model output directory
        // Tuner logs will automatically be saved to {dir}/autotuner/ by tune_tree_model() and tune_ltt()
        self.model_output_dir = Some(dir.to_path_buf());

        // DO NOT set tree_tuner_config.output.dir here!
        // Let tune_tree_model() handle appending "/autotuner" to avoid path conflicts
        self
    }

    /// Set custom tree tuner configuration (advanced API for power users)
    ///
    /// Use this to provide custom hyperparameter search configuration for PureTree
    /// and RandomForest modes. This allows fine-grained control over the number of
    /// samples, iterations, depth ranges, and other tuning parameters.
    ///
    /// **Most users should use `with_output_dir()` instead** - it's simpler and handles
    /// configuration automatically based on your tuning level.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, TreeTunerConfig};
    /// use treeboost::defaults::auto as auto_defaults;
    ///
    /// let custom_tuner = TreeTunerConfig {
    ///     max_depth_range: (3, 8),
    ///     learning_rate_range: auto_defaults::STANDARD_LR_RANGE,
    ///     n_samples: 50,
    ///     n_iterations: 1,
    ///     max_rounds: 200,
    ///     early_stopping_rounds: 10,
    ///     validation_ratio: 0.2,
    ///     improvement_threshold: 0.001,
    ///     min_f1_score: 0.85,
    /// };
    ///
    /// let config = AutoConfig::new()
    ///     .with_tree_tuner_config(custom_tuner);
    /// ```
    pub fn with_tree_tuner_config(mut self, config: TreeTunerConfig) -> Self {
        self.tree_tuner_config = Some(config);
        self
    }
}

/// Tuning result for tree-based models (PureTree, RandomForest)
#[derive(Debug, Clone)]
pub struct TreeTuningResult {
    /// Number of tuning trials evaluated
    pub num_trials: usize,
    /// Best validation metric achieved
    pub best_metric: f32,
    /// Best hyperparameters found
    pub best_params: std::collections::HashMap<String, f32>,
}

/// Depth range configuration for tree tuning.
#[derive(Debug, Clone, Copy)]
pub struct DepthConfig {
    /// Minimum depth to explore
    pub min: usize,
    /// Maximum depth to explore
    pub max: usize,
}

impl DepthConfig {
    /// Create depth config from range tuple
    pub fn new(min: usize, max: usize) -> Self {
        Self { min, max }
    }

    /// Create from tuple (min, max)
    pub fn from_range(range: (usize, usize)) -> Self {
        Self {
            min: range.0,
            max: range.1,
        }
    }

    /// Convert to range tuple
    pub fn to_range(self) -> (usize, usize) {
        (self.min, self.max)
    }
}

/// Learning rate range configuration for tree tuning.
#[derive(Debug, Clone, Copy)]
pub struct LearningRateConfig {
    /// Minimum learning rate to explore (log scale)
    pub min: f32,
    /// Maximum learning rate to explore (log scale)
    pub max: f32,
}

impl LearningRateConfig {
    /// Create learning rate config from range tuple
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Create from tuple (min, max)
    pub fn from_range(range: (f32, f32)) -> Self {
        Self {
            min: range.0,
            max: range.1,
        }
    }

    /// Convert to range tuple
    pub fn to_range(self) -> (f32, f32) {
        (self.min, self.max)
    }
}

/// Search configuration for tree tuning.
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// Number of Latin Hypercube samples per iteration
    pub n_samples: usize,
    /// Number of zoom iterations (iterative refinement)
    pub n_iterations: usize,
}

impl SearchConfig {
    /// Create search config
    pub fn new(n_samples: usize, n_iterations: usize) -> Self {
        Self {
            n_samples,
            n_iterations,
        }
    }
}

/// Stopping criteria configuration for tree tuning.
#[derive(Debug, Clone, Copy)]
pub struct StoppingConfig {
    /// Early stopping rounds (stop if no improvement)
    pub early_stopping_rounds: usize,
    /// Validation ratio for early stopping
    pub validation_ratio: f32,
    /// Improvement threshold for stopping iterations (e.g., 0.001 = 0.1%)
    pub improvement_threshold: f32,
    /// Minimum F1 score required before stopping (classification only)
    pub min_f1_score: f32,
}

impl StoppingConfig {
    /// Create stopping config
    pub fn new(
        early_stopping_rounds: usize,
        validation_ratio: f32,
        improvement_threshold: f32,
        min_f1_score: f32,
    ) -> Self {
        Self {
            early_stopping_rounds,
            validation_ratio,
            improvement_threshold,
            min_f1_score,
        }
    }
}

/// Output and metric configuration for tree tuning.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Optional output directory for CSV logging (None = no logging)
    pub dir: Option<std::path::PathBuf>,
    /// Metric to optimize (ValidationLoss, F1Score, RocAuc, RankIc)
    pub metric: OptimizationMetric,
}

impl OutputConfig {
    /// Create output config
    pub fn new(dir: Option<std::path::PathBuf>, metric: OptimizationMetric) -> Self {
        Self { dir, metric }
    }
}

/// Configuration for tree-based model tuning (PureTree, RandomForest)
///
/// This config groups parameters into logical sub-configs for better organization:
/// - `depth`: Depth range to explore
/// - `learning_rate`: Learning rate range to explore
/// - `search`: Latin Hypercube sampling and iteration parameters
/// - `stopping`: Early stopping and convergence criteria
/// - `output`: Logging and metric optimization
///
/// # Examples
///
/// ```ignore
/// use treeboost::TreeTunerConfig;
///
/// // Use preset
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Standard);
///
/// // Custom config
/// let config = TreeTunerConfig::new()
///     .with_depth_range(3, 10)
///     .with_n_samples(200)
///     .with_n_iterations(5);
/// ```
#[derive(Debug, Clone)]
pub struct TreeTunerConfig {
    /// Depth range configuration
    pub depth: DepthConfig,
    /// Learning rate range configuration
    pub learning_rate: LearningRateConfig,
    /// Search space sampling configuration
    pub search: SearchConfig,
    /// Maximum boosting rounds per trial
    pub max_rounds: usize,
    /// Stopping criteria configuration
    pub stopping: StoppingConfig,
    /// Output and metric configuration
    pub output: OutputConfig,
    /// Task type (Regression, BinaryClassification, MultiClassClassification)
    ///
    /// When set to Some, overrides the profile-detected task type.
    /// Use this to explicitly set regression for stock/quant data where
    /// Rank IC should be computed.
    pub task_type: Option<TunerTaskType>,
}

/// Tuning intensity preset for tree-specific hyperparameter optimization.
///
/// ## When to Use TreeTunerPreset
///
/// Use `TreeTunerPreset` when working with **`TreeTunerConfig`**.
/// This enum controls tree-specific tuning intensity for PureTree and RandomForest modes.
///
/// ## Relationship to Other Preset Enums
///
/// TreeBoost has several tuning-related preset enums for different contexts:
///
/// | Enum | Context | Purpose |
/// |------|---------|---------|
/// | **`TuningLevel`** | `AutoBuilder`, `AutoConfig` | High-level: AutoML tuning intensity |
/// | **`TunerPreset`** | `AutoTuner`, `TunerConfig` | Mid-level: Manual tuning intensity |
/// | **`TreeTunerPreset`** | `TreeTunerConfig` | Low-level: Tree-only tuning intensity |
///
/// **Mapping between presets:**
/// - `TreeTunerPreset::Quick` ≈ `TuningLevel::Quick` ≈ `TunerPreset::Quick`
/// - `TreeTunerPreset::Standard` ≈ `TuningLevel::Standard` ≈ `TunerPreset::Balanced`
/// - `TreeTunerPreset::Thorough` ≈ `TuningLevel::Thorough` ≈ `TunerPreset::Thorough`
///
/// ## Variants
///
/// ### `Quick`
/// Fast tree tuning with shallow depth range.
/// - **Best for**: Prototyping, debugging, quick experiments
/// - **Depth range**: [3-6]
/// - **Samples**: 30 Latin Hypercube samples
/// - **Iterations**: 1 (no zoom)
/// - **Time**: Minutes
/// - **Quality**: Good starting point, not fully optimized
///
/// ### `Standard` (Default)
/// Balanced tree tuning with moderate depth range.
/// - **Best for**: Most production use cases
/// - **Depth range**: [3-8]
/// - **Samples**: 100 Latin Hypercube samples
/// - **Iterations**: 3 (with zoom refinement)
/// - **Time**: Tens of minutes
/// - **Quality**: Well-tuned, production-ready
///
/// ### `Thorough`
/// Extensive tree tuning with deep depth range.
/// - **Best for**: Maximum accuracy, competitions, research
/// - **Depth range**: [3-10]
/// - **Samples**: 150 Latin Hypercube samples
/// - **Iterations**: 15 (aggressive zoom)
/// - **Time**: Hours
/// - **Quality**: Near-optimal hyperparameters
///
/// ## Examples
///
/// ```ignore
/// use treeboost::{TreeTunerConfig, TreeTunerPreset};
///
/// // Quick tree tuning
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Quick);
///
/// // Standard tree tuning (default)
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Standard);
///
/// // Thorough tree tuning
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Thorough);
/// ```
///
/// ## See Also
///
/// - [`TuningLevel`] - For high-level AutoML tuning with `AutoBuilder`
/// - [`crate::tuner::TunerPreset`] - For manual tuning with `AutoTuner`
/// - [`TreeTunerConfig::with_preset`] - Apply a preset to tree tuner configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeTunerPreset {
    /// Fast tree tuning with shallow depth range [3-6]
    Quick,
    /// Balanced tree tuning with moderate depth range [3-8]
    Standard,
    /// Extensive tree tuning with deep depth range [3-10]
    Thorough,
}

impl TreeTunerConfig {
    fn preset_quick() -> Self {
        Self {
            depth: DepthConfig::from_range(auto_defaults::QUICK_DEPTH_RANGE),
            learning_rate: LearningRateConfig::from_range(auto_defaults::QUICK_LR_RANGE),
            search: SearchConfig::new(30, 1),
            max_rounds: 100,
            stopping: StoppingConfig::new(10, auto_defaults::DEFAULT_VALIDATION_RATIO, 0.001, 0.80),
            output: OutputConfig::new(None, OptimizationMetric::ValidationLoss),
            task_type: None, // Use profile detection
        }
    }

    fn preset_standard() -> Self {
        Self {
            depth: DepthConfig::from_range(auto_defaults::STANDARD_DEPTH_RANGE),
            learning_rate: LearningRateConfig::from_range(auto_defaults::STANDARD_LR_RANGE),
            search: SearchConfig::new(100, 3),
            max_rounds: 200,
            stopping: StoppingConfig::new(10, auto_defaults::DEFAULT_VALIDATION_RATIO, 0.001, 0.85),
            output: OutputConfig::new(None, OptimizationMetric::ValidationLoss),
            task_type: None, // Use profile detection
        }
    }

    fn preset_thorough() -> Self {
        Self {
            depth: DepthConfig::from_range(auto_defaults::THOROUGH_DEPTH_RANGE),
            learning_rate: LearningRateConfig::from_range(auto_defaults::STANDARD_LR_RANGE),
            search: SearchConfig::new(150, 15),
            max_rounds: 200,
            stopping: StoppingConfig::new(10, auto_defaults::DEFAULT_VALIDATION_RATIO, 0.001, 0.85),
            output: OutputConfig::new(None, OptimizationMetric::ValidationLoss),
            task_type: None, // Use profile detection
        }
    }

    /// Set the optimization metric
    pub fn with_optimization_metric(mut self, metric: OptimizationMetric) -> Self {
        self.output.metric = metric;
        self
    }

    /// Set the task type explicitly (overrides profile detection)
    pub fn with_task_type(mut self, task_type: TunerTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    /// Create a new TreeTunerConfig with standard defaults.
    pub fn new() -> Self {
        Self::preset_standard()
    }

    /// Set depth range for tree tuning.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_depth_range(3, 10);
    /// ```
    pub fn with_depth_range(mut self, min: usize, max: usize) -> Self {
        self.depth = DepthConfig::new(min, max);
        self
    }

    /// Set learning rate range for tree tuning.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_learning_rate_range(0.01, 0.3);
    /// ```
    pub fn with_learning_rate_range(mut self, min: f32, max: f32) -> Self {
        self.learning_rate = LearningRateConfig::new(min, max);
        self
    }

    /// Set number of Latin Hypercube samples per iteration.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_n_samples(200);
    /// ```
    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.search.n_samples = n_samples;
        self
    }

    /// Set number of zoom iterations.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_n_iterations(5);
    /// ```
    pub fn with_n_iterations(mut self, n_iterations: usize) -> Self {
        self.search.n_iterations = n_iterations;
        self
    }

    /// Set maximum boosting rounds per trial.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_max_rounds(500);
    /// ```
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Set early stopping rounds.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_early_stopping(20);
    /// ```
    pub fn with_early_stopping(mut self, rounds: usize) -> Self {
        self.stopping.early_stopping_rounds = rounds;
        self
    }

    /// Set validation ratio for early stopping.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_validation_ratio(0.15);
    /// ```
    pub fn with_validation_ratio(mut self, ratio: f32) -> Self {
        self.stopping.validation_ratio = ratio;
        self
    }

    /// Set improvement threshold for iteration stopping.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_improvement_threshold(0.005);
    /// ```
    pub fn with_improvement_threshold(mut self, threshold: f32) -> Self {
        self.stopping.improvement_threshold = threshold;
        self
    }

    /// Set minimum F1 score for classification.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_min_f1_score(0.90);
    /// ```
    pub fn with_min_f1_score(mut self, score: f32) -> Self {
        self.stopping.min_f1_score = score;
        self
    }

    /// Set output directory for CSV logging.
    ///
    /// # Example
    /// ```ignore
    /// let config = TreeTunerConfig::new().with_output_dir(Some("logs/tuning".into()));
    /// ```
    pub fn with_output_dir(mut self, dir: Option<std::path::PathBuf>) -> Self {
        self.output.dir = dir;
        self
    }

    /// Set the entire depth configuration.
    ///
    /// # Example
    /// ```ignore
    /// let depth_cfg = DepthConfig::new(5, 12);
    /// let config = TreeTunerConfig::new().with_depth_config(depth_cfg);
    /// ```
    pub fn with_depth_config(mut self, depth: DepthConfig) -> Self {
        self.depth = depth;
        self
    }

    /// Set the entire learning rate configuration.
    ///
    /// # Example
    /// ```ignore
    /// let lr_cfg = LearningRateConfig::new(0.01, 0.5);
    /// let config = TreeTunerConfig::new().with_learning_rate_config(lr_cfg);
    /// ```
    pub fn with_learning_rate_config(mut self, learning_rate: LearningRateConfig) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the entire search configuration.
    ///
    /// # Example
    /// ```ignore
    /// let search_cfg = SearchConfig::new(200, 10);
    /// let config = TreeTunerConfig::new().with_search_config(search_cfg);
    /// ```
    pub fn with_search_config(mut self, search: SearchConfig) -> Self {
        self.search = search;
        self
    }

    /// Set the entire stopping configuration.
    ///
    /// # Example
    /// ```ignore
    /// let stopping_cfg = StoppingConfig::new(15, 0.15, 0.002, 0.88);
    /// let config = TreeTunerConfig::new().with_stopping_config(stopping_cfg);
    /// ```
    pub fn with_stopping_config(mut self, stopping: StoppingConfig) -> Self {
        self.stopping = stopping;
        self
    }

    /// Set the entire output configuration.
    ///
    /// # Example
    /// ```ignore
    /// let output_cfg = OutputConfig::new(Some("logs".into()), OptimizationMetric::F1Score);
    /// let config = TreeTunerConfig::new().with_output_config(output_cfg);
    /// ```
    pub fn with_output_config(mut self, output: OutputConfig) -> Self {
        self.output = output;
        self
    }

    /// Apply a preset configuration.
    pub fn with_preset(preset: TreeTunerPreset) -> Self {
        match preset {
            TreeTunerPreset::Quick => Self::preset_quick(),
            TreeTunerPreset::Standard => Self::preset_standard(),
            TreeTunerPreset::Thorough => Self::preset_thorough(),
        }
    }
}

impl Default for TreeTunerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Build result containing the trained model and metadata
#[derive(Debug)]
pub struct BuildResult {
    /// The trained model (UniversalModel handles all modes and ensembles).
    ///
    /// None if `skip_training` was enabled (discovery mode).
    pub model: Option<UniversalModel>,

    /// Whether training was skipped (discovery mode).
    ///
    /// When true, `model` will be None and only configuration was discovered.
    /// The enriched config is saved to disk and can be used for training with UniversalModel.
    pub skip_training: bool,

    /// The boosting mode used
    pub mode: BoostingMode,

    /// Target column name used during training
    pub target_column: String,

    /// Mode selection confidence (if auto mode was used)
    pub mode_confidence: Option<Confidence>,

    /// Preprocessing plan that was applied
    pub preprocessing_plan: Option<PreprocessingPlan>,

    /// Feature engineering plan that was applied
    pub feature_plan: Option<FeaturePlan>,

    /// LTT tuning result (if LTT mode was used)
    pub ltt_tuning: Option<LttTuningResult>,

    /// Tree tuning result (if PureTree/RandomForest mode was used)
    pub tree_tuning: Option<TreeTuningResult>,

    /// Column profile from analysis
    pub column_profile: Option<DataFrameProfile>,

    /// Dataset analysis result
    pub analysis: Option<DatasetAnalysis>,

    /// Fitted pipeline state for inference (CRITICAL for prediction!)
    pub pipeline_state: Option<crate::dataset::PipelineState>,

    /// Target transformation applied (if any)
    pub target_transform: Option<crate::preprocessing::TargetTransformKind>,

    /// Total build time
    pub build_time: Duration,

    /// Time breakdown by phase
    pub phase_times: BuildPhaseTimes,

    /// Binned validation dataset (for debugging pipeline consistency)
    /// Only set when presplit validation is used.
    pub validation_dataset: Option<crate::dataset::BinnedDataset>,
}

/// Time breakdown for build phases
#[derive(Debug, Clone, Default)]
pub struct BuildPhaseTimes {
    pub profiling: Duration,
    pub preprocessing: Duration,
    pub feature_engineering: Duration,
    pub analysis: Duration,
    pub tuning: Duration,
    pub training: Duration,
}
