//! Configuration for UniversalModel

use crate::dataset::feature_extractor::FeatureExtractor;
use crate::defaults::{
    ensemble as ensemble_defaults, learners::gbdt as gbdt_defaults,
    learners::tree as tree_defaults, learners::universal as universal_defaults,
    tuning::seeds as seeds_defaults,
};
use crate::learner::{LinearConfig, LinearPreset, TreeConfig, TreePreset};
use crate::model::universal::mode::BoostingMode;
use crate::{Result, TreeBoostError};
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// StackingStrategy - Serializable stacking config
// =============================================================================

/// Stacking strategy for ensemble combination
///
/// Specifies how to combine predictions from multiple ensemble members.
///
/// This is a serializable alternative to the `Box<dyn Stacker>` trait object,
/// enabling full persistence of ensemble configurations. UniversalModel stores
/// this enum and constructs the appropriate stacker at runtime.
///
/// # Variants
///
/// - **Ridge**: Uses Ridge regression to learn optimal weights for each ensemble member.
///   Recommended when ensemble members have different prediction scales or accuracies.
///
/// - **Average**: Simple equal-weight averaging across all members.
///   Recommended for homogeneous ensembles where all members have similar quality.
///
/// # Note on Extensibility
///
/// Only Ridge and Average strategies are currently exposed because they are the most
/// commonly used and both are fully serializable. If you need custom stacking logic,
/// train multiple independent models and combine them manually, or use the TrainedMember
/// API directly with EnsembleBuilder.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub enum StackingStrategy {
    /// Ridge regression stacking with regularization
    ///
    /// Learns optimal weights via Ridge regression on out-of-fold predictions.
    /// Ridge regularization prevents overfitting when stacking.
    Ridge {
        /// Ridge regularization parameter (alpha). Higher values = stronger regularization.
        /// Typical range: 0.001 to 0.1. Default: 0.01
        alpha: f32,
        /// Whether to apply rank transformation to member predictions before stacking.
        /// Useful when members have different prediction scales (e.g., one predicts [0, 100],
        /// another [0, 1]). Rank transformation makes them comparable. Default: false
        rank_transform: bool,
        /// Whether to fit an intercept term. Set to false if you want zero-centered predictions.
        /// Default: true
        fit_intercept: bool,
        /// Minimum weight magnitude threshold. Weights smaller than this are set to zero,
        /// creating sparse stacking weights. Typical range: 0.0 to 0.1. Default: 0.01
        min_weight: f32,
    },
    /// Simple averaging (equal weights)
    ///
    /// Combines members by simple arithmetic mean: `(pred_1 + pred_2 + ... + pred_n) / n`
    /// No learning required. Fast and effective for balanced, diverse ensembles.
    Average,
}

impl Default for StackingStrategy {
    fn default() -> Self {
        Self::Ridge {
            alpha: ensemble_defaults::DEFAULT_STACKING_ALPHA,
            rank_transform: ensemble_defaults::DEFAULT_RANK_TRANSFORM,
            fit_intercept: ensemble_defaults::DEFAULT_FIT_INTERCEPT,
            min_weight: ensemble_defaults::DEFAULT_MIN_WEIGHT,
        }
    }
}

// =============================================================================
// LinearSelectionMode
// =============================================================================

/// How to select features for the linear component in LinearThenTree mode
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub enum LinearSelectionMode {
    /// Select features by correlation with target (|r| >= threshold)
    /// This captures real linear signal: trends, spatial gradients, price levels
    Correlation {
        /// Minimum absolute correlation to include a feature (default: 0.05)
        threshold: f32,
    },
    /// Select features by naming convention (legacy behavior)
    /// Matches: *_squared, *_sqrt, *_log, *_log1p, *_x_*, *_ratio_*
    Pattern,
}

impl Default for LinearSelectionMode {
    fn default() -> Self {
        Self::Correlation { threshold: 0.05 }
    }
}

// =============================================================================
// UniversalConfig
// =============================================================================

/// Configuration for UniversalModel
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct UniversalConfig {
    /// Boosting mode
    pub mode: BoostingMode,

    /// Number of boosting rounds (trees for PureTree/LinearThenTree, or total trees for RF)
    pub num_rounds: usize,

    /// Tree configuration
    pub tree_config: TreeConfig,

    /// Linear configuration (for LinearThenTree mode)
    pub linear_config: LinearConfig,

    /// Learning rate for gradient boosting (not used in RandomForest mode)
    pub learning_rate: f32,

    /// Row subsampling ratio (0.0-1.0)
    pub subsample: f32,

    /// Column subsampling ratio per tree (0.0-1.0, 1.0 = use all features)
    pub colsample: f32,

    /// Validation ratio for early stopping (0.0 to disable)
    pub validation_ratio: f32,

    /// Early stopping rounds (0 to disable)
    pub early_stopping_rounds: usize,

    /// Calibration set ratio for conformal prediction (0.0 to disable)
    pub calibration_ratio: f32,

    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    pub conformal_quantile: f32,

    /// Random seed
    pub seed: u64,

    /// Number of linear boosting rounds before trees (LinearThenTree mode)
    pub linear_rounds: usize,

    /// How to select features for the linear component (LinearThenTree mode)
    ///
    /// - `Correlation { threshold }`: Select features with |correlation| >= threshold (default)
    /// - `Pattern`: Legacy naming convention matching (*_squared, *_x_*, etc.)
    ///
    /// Ignored if `linear_feature_indices` is set (explicit override takes priority).
    pub linear_selection_mode: LinearSelectionMode,

    /// Feature indices for linear model in LinearThenTree mode
    ///
    /// When set, specifies which feature indices (after preprocessing and feature engineering)
    /// should be used by the linear model. The tree model uses the complementary set
    /// (all features NOT in this list).
    ///
    /// This is critical for LinearThenTree feature partitioning:
    /// - Linear model: Uses engineered features (polynomials, interactions)
    /// - Tree model: Uses original features (numeric + categorical-encoded)
    ///
    /// If None, both linear and tree use all features (backward compatibility).
    ///
    /// **Serialization**: This field MUST be serialized in both rkyv and JSON to ensure
    /// prediction uses the same feature split as training.
    pub linear_feature_indices: Option<Vec<usize>>,

    /// Maximum memory (MB) for LinearThenTree raw feature extraction
    ///
    /// LinearThenTree mode unpacks binned u8 data to f32 (4x expansion).
    /// Set this to limit memory usage. If exceeded:
    /// - `0` = No limit (default, may cause OOM on large datasets)
    /// - `> 0` = Error if estimated memory exceeds this limit
    ///
    /// **Rule of thumb**: 100M rows × 100 features = ~40GB memory
    pub max_linear_memory_mb: usize,

    /// Feature extractor for LinearThenTree mode
    ///
    /// Used to extract raw numeric features from DataFrame for the linear component.
    /// If None, LinearThenTree expects pre-extracted raw features.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    pub feature_extractor: Option<FeatureExtractor>,

    /// Multi-seed ensemble configuration
    ///
    /// If Some, trains multiple GBDTs with different random seeds and stacks them.
    /// - For PureTree mode: Trains N GBDTs directly
    /// - For LinearThenTree mode: Trains N GBDTs on linear residuals
    /// - For RandomForest mode: Ignored (RF already uses multiple trees)
    ///
    /// If None, trains a single GBDT (standard behavior).
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_ensemble_seeds(vec![42, 43, 44, 45, 46]); // 5-model ensemble
    /// ```
    pub ensemble_seeds: Option<Vec<u64>>,

    /// Stacking strategy for ensemble combination
    ///
    /// Only used if `ensemble_seeds.is_some()`.
    /// Determines how predictions from multiple GBDTs are combined.
    pub stacking_strategy: StackingStrategy,

    /// Backend type for histogram building (CPU/GPU)
    ///
    /// Determines which backend to use for gradient histogram computation:
    /// - `Auto`: Automatically select best available backend
    /// - `Cuda`: Force NVIDIA CUDA (requires `cuda` feature)
    /// - `Wgpu`: Force GPU via WGPU/Vulkan (requires `gpu` feature)
    /// - `Scalar`: Force CPU (portable fallback)
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    pub backend_type: crate::backend::BackendType,

    /// Sequential data transformation pipeline
    ///
    /// **CRITICAL**: This captures ALL transformation steps in execution order:
    /// 1. Feature engineering (polynomials, interactions, ratios, time-series)
    /// 2. Categorical encoding (CMS filter → Target encoding)
    /// 3. Target transformation (Logit for bounded targets)
    /// 4. Numeric binning (quantile boundaries)
    /// 5. Linear feature extraction (for LinearThenTree mode)
    ///
    /// Each step contains both the transformation logic AND learned state (encodings, boundaries).
    /// This ensures training and inference use THE EXACT SAME transformations.
    ///
    /// When None, no preprocessing is applied (data is assumed to already be processed).
    ///
    /// # Serialization
    ///
    /// Pipeline is fully serialized in both rkyv (model.rkyv) and serde (config.json).
    /// The model.rkyv file is the SINGLE SOURCE OF TRUTH.
    /// config.json is a human-readable view extracted from the model.
    ///
    /// # Example
    /// ```ignore
    /// // AutoBuilder builds pipeline during training, saves in model.rkyv
    /// let result = AutoBuilder::new().fit(&train_df, "target")?;
    /// result.model.save("model.rkyv")?;  // Contains model + pipeline
    ///
    /// // For inference: Load model (includes pipeline), call predict_df
    /// let model = UniversalModel::load("model.rkyv")?;
    /// let preds = model.predict_df(&test_df)?;  // Pipeline auto-applied
    /// ```
    pub pipeline: Option<crate::model::Pipeline>,

    /// DataPipeline state for inference (bin boundaries, encoding maps, column order)
    ///
    /// This is the authoritative state used by `predict_df()` to bin and encode
    /// inference data identically to training. Stored alongside the Pipeline for
    /// backwards compatibility, but `predict_df()` uses this for actual inference.
    ///
    /// Skipped by serde because rkyv handles serialization of the full model
    /// (including pipeline state) via zero-copy. JSON round-tripping of config
    /// alone does not need this field.
    #[serde(skip)]
    pub pipeline_state: Option<crate::dataset::PipelineState>,
}

impl Default for UniversalConfig {
    fn default() -> Self {
        Self {
            mode: BoostingMode::PureTree,
            num_rounds: gbdt_defaults::DEFAULT_NUM_ROUNDS,
            tree_config: TreeConfig::default(),
            linear_config: LinearConfig::default(),
            learning_rate: tree_defaults::DEFAULT_LEARNING_RATE,
            subsample: gbdt_defaults::DEFAULT_SUBSAMPLE,
            colsample: 1.0,
            validation_ratio: gbdt_defaults::DEFAULT_GBDT_VALIDATION_RATIO,
            early_stopping_rounds: gbdt_defaults::DEFAULT_EARLY_STOPPING_ROUNDS,
            calibration_ratio: gbdt_defaults::DEFAULT_CALIBRATION_RATIO,
            conformal_quantile: gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE,
            seed: seeds_defaults::DEFAULT_SEED,
            linear_rounds: universal_defaults::DEFAULT_LINEAR_ROUNDS, // Single round with many CD iterations (fit once)
            linear_selection_mode: LinearSelectionMode::default(),
            linear_feature_indices: None, // No feature filtering by default (backward compat)
            max_linear_memory_mb: universal_defaults::DEFAULT_MAX_LINEAR_MEMORY_MB, // No limit by default
            feature_extractor: None,
            ensemble_seeds: None, // No ensemble by default
            stacking_strategy: StackingStrategy::default(),
            backend_type: crate::backend::BackendType::Auto,
            pipeline: None,
            pipeline_state: None,
        }
    }
}

/// Presets for common UniversalModel configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniversalPreset {
    /// Standard GBDT (GBDTConfig wrapper).
    PureTree,
    /// Linear first, then tree residuals.
    LinearThenTree,
    /// Bagged trees for variance reduction.
    RandomForest,
    /// LinearThenTree + aggressive linear shrinkage.
    TimeSeries,
    /// RandomForest + regularized trees.
    NoisyTabular,
    /// PureTree + conformal calibration.
    UncertaintyAware,
}

impl UniversalConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: UniversalPreset) -> Self {
        match preset {
            UniversalPreset::PureTree => {
                self.mode = BoostingMode::PureTree;
            }
            UniversalPreset::LinearThenTree => {
                self.mode = BoostingMode::LinearThenTree;
            }
            UniversalPreset::RandomForest => {
                self.mode = BoostingMode::RandomForest;
            }
            UniversalPreset::TimeSeries => {
                self.mode = BoostingMode::LinearThenTree;
                self.linear_config = self.linear_config.with_preset(LinearPreset::Aggressive);
            }
            UniversalPreset::NoisyTabular => {
                self.mode = BoostingMode::RandomForest;
                self.tree_config = self.tree_config.with_preset(TreePreset::Regularized);
            }
            UniversalPreset::UncertaintyAware => {
                self.mode = BoostingMode::PureTree;
                self.calibration_ratio = gbdt_defaults::CONFORMAL_CALIBRATION_RATIO;
                self.conformal_quantile = gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE;
            }
        }
        self
    }

    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_num_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    pub fn with_tree_config(mut self, config: TreeConfig) -> Self {
        self.tree_config = config;
        self
    }

    pub fn with_linear_config(mut self, config: LinearConfig) -> Self {
        self.linear_config = config;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Result<Self> {
        if lr <= 0.0 || lr > 1.0 {
            return Err(TreeBoostError::Config(format!(
                "learning_rate must be in (0, 1], got {}",
                lr
            )));
        }
        self.learning_rate = lr;
        Ok(self)
    }

    pub fn with_subsample(mut self, ratio: f32) -> Result<Self> {
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(TreeBoostError::Config(format!(
                "subsample must be in (0, 1], got {}",
                ratio
            )));
        }
        self.subsample = ratio;
        Ok(self)
    }

    pub fn with_colsample(mut self, ratio: f32) -> Result<Self> {
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(TreeBoostError::Config(format!(
                "colsample must be in (0, 1], got {}",
                ratio
            )));
        }
        self.colsample = ratio;
        Ok(self)
    }

    pub fn with_validation_ratio(mut self, ratio: f32) -> Result<Self> {
        if !(0.0..=0.5).contains(&ratio) {
            return Err(TreeBoostError::Config(format!(
                "validation_ratio must be in [0, 0.5], got {}",
                ratio
            )));
        }
        self.validation_ratio = ratio;
        Ok(self)
    }

    pub fn with_early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = rounds;
        self
    }

    /// Enable conformal calibration for uncertainty estimates.
    pub fn with_conformal_calibration(mut self, ratio: f32, quantile: f32) -> Result<Self> {
        if !(0.0..=0.5).contains(&ratio) {
            return Err(TreeBoostError::Config(format!(
                "calibration_ratio must be in [0, 0.5], got {}",
                ratio
            )));
        }
        if !(0.5..=0.99).contains(&quantile) {
            return Err(TreeBoostError::Config(format!(
                "conformal_quantile must be in [0.5, 0.99], got {}",
                quantile
            )));
        }
        self.calibration_ratio = ratio;
        self.conformal_quantile = quantile;
        Ok(self)
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_linear_rounds(mut self, rounds: usize) -> Self {
        self.linear_rounds = rounds;
        self
    }

    /// Set feature indices for linear model in LinearThenTree mode
    ///
    /// Specifies which feature indices should be used by the linear model.
    /// The tree model uses the complementary set (all features NOT in this list).
    ///
    /// # Arguments
    /// - `indices`: Feature indices for the linear model (e.g., engineered polynomial/interaction features)
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_linear_feature_indices(vec![0, 1, 4, 5]); // Linear uses features 0,1,4,5; trees use 2,3,6,7,...
    /// ```
    /// Set linear feature selection mode
    pub fn with_linear_selection_mode(mut self, mode: LinearSelectionMode) -> Self {
        self.linear_selection_mode = mode;
        self
    }

    pub fn with_linear_feature_indices(mut self, indices: Vec<usize>) -> Self {
        self.linear_feature_indices = Some(indices);
        self
    }

    /// Set maximum memory (MB) for LinearThenTree raw feature extraction
    ///
    /// # Arguments
    /// - `mb`: Maximum memory in megabytes (0 = no limit)
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_max_linear_memory_mb(4096); // 4GB limit
    /// ```
    pub fn with_max_linear_memory_mb(mut self, mb: usize) -> Self {
        self.max_linear_memory_mb = mb;
        self
    }

    /// Set the feature extractor for LinearThenTree mode
    ///
    /// The feature extractor is used to extract raw numeric features from DataFrames
    /// for the linear component in LinearThenTree mode. This allows automatic feature
    /// selection and handling of non-numeric columns.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_feature_extractor(Some(FeatureExtractor::default()));
    /// ```
    pub fn with_feature_extractor(mut self, extractor: Option<FeatureExtractor>) -> Self {
        self.feature_extractor = extractor;
        self
    }

    /// Estimate memory usage (bytes) for LinearThenTree raw feature extraction
    pub fn estimate_linear_memory(&self, num_rows: usize, num_features: usize) -> usize {
        // f32 = 4 bytes per element
        num_rows * num_features * 4
    }

    /// Set ensemble seeds for multi-model training
    ///
    /// Trains multiple GBDTs with different random seeds and stacks their predictions.
    /// - For PureTree mode: Trains N GBDTs directly
    /// - For LinearThenTree mode: Trains N GBDTs on linear residuals
    /// - For RandomForest mode: Ignored (RF already uses multiple trees)
    ///
    /// # Arguments
    /// - `seeds`: Vector of random seeds (typically 3-10 seeds)
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_ensemble_seeds(vec![42, 43, 44, 45, 46]);
    /// ```
    pub fn with_ensemble_seeds(mut self, seeds: Vec<u64>) -> Self {
        self.ensemble_seeds = Some(seeds);
        self
    }

    /// Set stacking strategy for ensemble combination
    ///
    /// Only used if `ensemble_seeds.is_some()`.
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_ensemble_seeds(vec![42, 43, 44])
    ///     .with_stacking_strategy(StackingStrategy::Ridge {
    ///         alpha: 1.0,
    ///         rank_transform: true,
    ///         fit_intercept: true,
    ///         min_weight: 0.0,
    ///     });
    /// ```
    pub fn with_stacking_strategy(mut self, strategy: StackingStrategy) -> Self {
        self.stacking_strategy = strategy;
        self
    }

    /// Set backend type for histogram building.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::BackendType;
    ///
    /// let config = UniversalConfig::new()
    ///     .with_backend(BackendType::Cuda); // Force CUDA backend
    /// ```
    pub fn with_backend(mut self, backend_type: crate::backend::BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }
}
