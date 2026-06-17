//! GBDT training configuration

use crate::backend::{BackendType, GpuMode};
use crate::dataset::OrderingStrategy;
use crate::defaults::{
    learners::gbdt as gbdt_defaults, learners::tree as tree_defaults,
    tuning::seeds as seeds_defaults,
};
use crate::loss::{BinaryLogLoss, LossFunction, MseLoss, PseudoHuberLoss};
use crate::tree::MonotonicConstraint;
use rkyv::{Archive, Deserialize, Serialize};

/// Output type for the model (used during prediction)
///
/// This determines how raw tree outputs are transformed and structured.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
    Default,
)]
pub enum OutputType {
    /// Single regression output (raw value)
    #[default]
    Regression,
    /// Single binary classification output (sigmoid)
    Binary,
    /// Multi-class classification (softmax over K classes)
    MultiClass,
    /// Multi-label classification (sigmoid per label, independent)
    MultiLabel,
}

impl OutputType {
    /// Returns true if this is a classification output type
    pub fn is_classification(&self) -> bool {
        !matches!(self, OutputType::Regression)
    }

    /// Returns true if this is a multi-output type
    pub fn is_multi_output(&self) -> bool {
        matches!(self, OutputType::MultiClass | OutputType::MultiLabel)
    }
}

/// Loss function type for serialization
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
    Default,
)]
pub enum LossType {
    /// Mean Squared Error (regression)
    #[default]
    Mse,
    /// Pseudo-Huber Loss with given delta (robust regression)
    PseudoHuber { delta: f32 },
    /// Binary Log Loss / Cross-Entropy (binary classification)
    BinaryLogLoss,
    /// Multi-class Log Loss / Softmax Cross-Entropy (multi-class classification)
    MultiClassLogLoss { num_classes: usize },
    /// Multi-label Log Loss (sigmoid per label, independent binary per output)
    MultiLabelLogLoss { num_outputs: usize },
    /// Multi-label Focal Loss (sigmoid per label with focusing parameter)
    MultiLabelFocalLoss { num_outputs: usize, gamma: f32 },
}

impl LossType {
    /// Create a boxed loss function for regression, binary, and multi-label losses
    ///
    /// # Panics
    /// Panics if called with `MultiClassLogLoss` - multi-class has a separate
    /// training path that handles gradients differently.
    pub fn create(&self) -> Box<dyn LossFunction> {
        match self {
            LossType::Mse => Box::new(MseLoss::new()),
            LossType::PseudoHuber { delta } => Box::new(PseudoHuberLoss::new(*delta)),
            LossType::BinaryLogLoss => Box::new(BinaryLogLoss::new()),
            LossType::MultiClassLogLoss { .. } => {
                panic!(
                    "MultiClassLogLoss does not implement LossFunction trait. \
                     Use train_binned_multiclass() which handles multi-class gradients directly."
                )
            }
            LossType::MultiLabelLogLoss { .. } => Box::new(crate::loss::MultiLabelLogLoss::new()),
            LossType::MultiLabelFocalLoss { gamma, .. } => {
                Box::new(crate::loss::MultiLabelFocalLoss::new(*gamma))
            }
        }
    }

    /// Create a loss function for multi-label training with configurable alpha
    pub fn create_multilabel(&self, alpha: Option<f32>) -> Box<dyn LossFunction> {
        match self {
            LossType::MultiLabelLogLoss { .. } => Box::new(crate::loss::MultiLabelLogLoss::new()),
            LossType::MultiLabelFocalLoss { gamma, .. } => {
                let mut loss = crate::loss::MultiLabelFocalLoss::new(*gamma);
                if let Some(a) = alpha {
                    loss = loss.with_alpha(a);
                }
                Box::new(loss)
            }
            _ => self.create(),
        }
    }

    /// Returns true if this is a classification loss
    pub fn is_classification(&self) -> bool {
        matches!(
            self,
            LossType::BinaryLogLoss
                | LossType::MultiClassLogLoss { .. }
                | LossType::MultiLabelLogLoss { .. }
                | LossType::MultiLabelFocalLoss { .. }
        )
    }

    /// Returns true if this is a multi-class classification loss
    pub fn is_multiclass(&self) -> bool {
        matches!(self, LossType::MultiClassLogLoss { .. })
    }

    /// Returns true if this is a multi-label classification loss
    pub fn is_multilabel(&self) -> bool {
        matches!(
            self,
            LossType::MultiLabelLogLoss { .. } | LossType::MultiLabelFocalLoss { .. }
        )
    }

    /// Get number of classes (for multi-class classification)
    pub fn num_classes(&self) -> Option<usize> {
        match self {
            LossType::MultiClassLogLoss { num_classes } => Some(*num_classes),
            _ => None,
        }
    }

    /// Get number of outputs (for multi-label classification)
    pub fn num_outputs(&self) -> Option<usize> {
        match self {
            LossType::MultiLabelLogLoss { num_outputs } => Some(*num_outputs),
            LossType::MultiLabelFocalLoss { num_outputs, .. } => Some(*num_outputs),
            LossType::MultiClassLogLoss { num_classes } => Some(*num_classes),
            _ => Some(1), // Regression and binary have 1 output
        }
    }

    /// Get the corresponding OutputType for this loss
    pub fn output_type(&self) -> OutputType {
        match self {
            LossType::Mse | LossType::PseudoHuber { .. } => OutputType::Regression,
            LossType::BinaryLogLoss => OutputType::Binary,
            LossType::MultiClassLogLoss { .. } => OutputType::MultiClass,
            LossType::MultiLabelLogLoss { .. } | LossType::MultiLabelFocalLoss { .. } => {
                OutputType::MultiLabel
            }
        }
    }
}

/// Presets for common GBDT configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbdtPreset {
    /// **Balanced defaults** - good starting point for clean, low-dimensional data.
    ///
    /// **Use when:**
    /// - Dataset is well-curated with mostly relevant features
    /// - Feature count < 100
    /// - Low noise-to-signal ratio
    ///
    /// **Characteristics:**
    /// - No row/column sampling (subsample=1.0, colsample=1.0)
    /// - May overfit on noisy or high-dimensional data
    Standard,

    /// **Shallow trees + GOSS** for faster training on large datasets.
    ///
    /// **Use when:**
    /// - Training speed is critical
    /// - Dataset is very large (>1M rows)
    /// - Willing to trade some accuracy for speed
    ///
    /// **Characteristics:**
    /// - Shallow trees (depth=4)
    /// - Gradient-based One-Side Sampling (GOSS) enabled
    /// - 3-5x faster than Standard
    Speed,

    /// **Deeper trees + feature/row bagging** for maximum predictive accuracy.
    ///
    /// **Use when:**
    /// - Accuracy is paramount
    /// - Training time is not a constraint
    /// - Dataset has complex interactions
    ///
    /// **Characteristics:**
    /// - Deep trees (depth=10)
    /// - Lower learning rate (0.05) + more rounds (200)
    /// - Feature bagging (colsample=0.8) prevents overfitting
    /// - Row bagging (subsample=0.8) improves generalization
    Accuracy,

    /// **No subsampling** - use all data every iteration (small datasets only).
    ///
    /// **Use when:**
    /// - Dataset is very small (<10k rows)
    /// - Every sample matters for learning
    /// - Data is clean and low-dimensional
    ///
    /// **⚠️ Warning:** Can overfit on noisy data!
    SmallData,

    /// **Aggressive subsampling + GOSS** for very large datasets.
    ///
    /// **Use when:**
    /// - Dataset is massive (>10M rows)
    /// - Memory is constrained
    /// - Training speed is critical
    ///
    /// **Characteristics:**
    /// - Row subsampling (subsample=0.8)
    /// - GOSS enabled for additional speedup
    /// - Efficient memory usage
    LargeData,

    /// **Robust to noise and irrelevant features** - prevents overfitting via bagging.
    ///
    /// **Use when:**
    /// - High-dimensional data (>100 features)
    /// - Many irrelevant or noisy features
    /// - Risk of overfitting to noise
    /// - Feature importance is unknown
    ///
    /// **Characteristics:**
    /// - Feature bagging (colsample=0.8) - Random Forest-like robustness
    /// - Row bagging (subsample=0.8) - variance reduction
    /// - Ignores ~20% of features per split (focuses on signal)
    ///
    /// **Evidence:** Our tests show naive GBDT with 1000 features (990 noise)
    /// has 99,366% train/val gap. This preset reduces it to 140% (993× improvement!).
    ///
    /// **Recommended for:**
    /// - Financial data with many technical indicators
    /// - Biological data with high-throughput measurements
    /// - Any data where you suspect irrelevant features
    ///
    /// **See:** `tests/rf_robustness.rs` for empirical validation
    Robust,

    /// **Enable conformal prediction** for uncertainty quantification.
    ///
    /// **Use when:**
    /// - Need prediction intervals (not just point estimates)
    /// - Uncertainty quantification is critical
    /// - Decision-making under uncertainty
    ///
    /// **Characteristics:**
    /// - Calibration ratio = 0.2 (20% of data for calibration)
    /// - Quantile = 0.9 (90% coverage intervals)
    Conformal,
}

/// GBDT training configuration
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct GBDTConfig {
    // Ensemble parameters
    /// Number of boosting rounds (trees)
    pub num_rounds: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f32,

    // Tree parameters
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Maximum number of leaves per tree
    pub max_leaves: usize,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: usize,
    /// Minimum hessian sum required in a leaf
    pub min_hessian_leaf: f32,
    /// Minimum gain to make a split
    pub min_gain: f32,

    // Regularization
    /// L2 regularization (lambda)
    pub lambda: f32,
    /// Shannon Entropy regularization weight (beta)
    pub entropy_weight: f32,

    // Loss function
    /// Loss function type
    pub loss_type: LossType,

    // Subsampling
    /// Row subsampling ratio (0.0-1.0) for random subsampling
    pub subsample: f32,
    /// Column subsampling ratio (0.0-1.0)
    pub colsample: f32,

    // GOSS (Gradient-based One-Side Sampling)
    /// Enable GOSS sampling (overrides random subsample when enabled)
    pub goss_enabled: bool,
    /// Ratio of large-gradient samples to keep (default: 0.2 = top 20%)
    pub goss_top_rate: f32,
    /// Ratio of small-gradient samples to randomly sample (default: 0.1 = 10%)
    pub goss_other_rate: f32,

    // Binning
    /// Number of histogram bins
    pub num_bins: usize,

    // Conformal prediction
    /// Calibration set ratio for conformal prediction (0.0 to disable)
    pub calibration_ratio: f32,
    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    pub conformal_quantile: f32,

    // Early stopping
    /// Number of rounds with no improvement before stopping (0 to disable)
    pub early_stopping_rounds: usize,
    /// Minimum trees before early stopping can trigger (default: 20)
    ///
    /// Prevents early stopping from killing models too early.
    /// Early stopping will only check after this many trees have been trained.
    pub min_early_stopping_trees: usize,
    /// Ratio of data to use for validation (0.0 to disable early stopping)
    pub validation_ratio: f32,

    // Performance optimizations (all ON by default)
    /// Use parallel prediction via Rayon (default: true)
    pub parallel_prediction: bool,
    /// Reorder columns by feature importance for cache locality (default: true)
    pub column_reordering: bool,
    /// Column reordering strategy (default: ByImportance)
    pub reordering_strategy: OrderingStrategy,
    /// Use 4-bit packing for low-cardinality features (default: true)
    pub packed_dataset: bool,
    /// Use parallel gradient computation (default: false)
    /// Experimental: may not provide stable speedups, benchmark before enabling
    pub parallel_gradient: bool,

    // Backend selection
    /// Backend type for histogram building (default: Auto = GPU for large datasets, CPU otherwise)
    #[rkyv(with = rkyv::with::Skip)]
    pub backend_type: BackendType,

    /// GPU execution mode for GPU backends (default: Auto).
    ///
    /// - `Auto`: Automatically select optimal mode per backend
    ///   - CUDA: Full (low dispatch latency makes it worthwhile)
    ///   - WGPU: Hybrid (high dispatch latency makes Full slower)
    /// - `Hybrid`: GPU histogram + CPU partition/split (best-first tree growth)
    /// - `Full`: Full GPU pipeline with level-wise tree growth
    ///
    /// Ignored when using CPU-only backends (Scalar, AVX-512, SVE2).
    #[rkyv(with = rkyv::with::Skip)]
    pub gpu_mode: GpuMode,

    /// Enable GPU subgroup operations for histogram building (default: false)
    ///
    /// Subgroups can reduce atomic contention when multiple threads write to the same
    /// histogram bin. However, benchmarks show minimal benefit on modern NVIDIA GPUs
    /// (~1.0x speedup). May help on older AMD or Intel GPUs with slower atomics.
    #[rkyv(with = rkyv::with::Skip)]
    pub use_gpu_subgroups: bool,

    // Monotonic constraints
    /// Monotonic constraints per feature (empty = no constraints)
    pub monotonic_constraints: Vec<MonotonicConstraint>,

    // Interaction constraints (groups of features that can interact)
    /// Feature interaction groups: each inner Vec is a group of features that can interact
    /// Features not in any group can interact with all features
    pub interaction_groups: Vec<Vec<usize>>,

    // Era-based splitting (Directional Era Splitting / DES)
    /// Enable era-based split finding for robust/invariant learning (default: false)
    ///
    /// When enabled, only accepts splits where ALL eras agree on the split direction.
    /// This filters out spurious correlations that work in some eras but not others,
    /// learning only invariant patterns that generalize across time periods/environments.
    ///
    /// Use cases:
    /// - Financial ML (market regimes shift over time)
    /// - Time series with distribution shift
    /// - Multi-environment/multi-site data
    /// - Numerai-style competitions with era labels
    ///
    /// Requires passing `era_indices` to the training method.
    pub era_splitting: bool,

    /// Enable noise pruning (PaloBoost-style validation)
    ///
    /// When enabled with `era_splitting`, treats eras as:
    /// - Era 0: Training set (in-bag)
    /// - Era 1: Validation set (out-of-bag)
    ///
    /// Splits are found using Era 0 gain, but rejected if Era 1 gain <= 0.
    /// This prevents overfitting by validating each split against held-out data.
    ///
    /// Use cases:
    /// - Noisy datasets where standard regularization isn't enough
    /// - High-dimensional data with spurious correlations
    /// - Temporal validation (train on past, validate on future)
    pub noise_pruning: bool,

    /// Noise pruning threshold for validation gain
    /// Splits are rejected if validation gain <= this threshold
    /// Default: -0.1 (allows small negative gains to handle distribution shift)
    /// Stricter: 0.0 (classic PaloBoost, only accept positive validation gains)
    /// More permissive: -0.2 (for high temporal distribution shift)
    pub noise_pruning_threshold: f32,

    /// Post-pruning gamma threshold (Cost-Complexity Pruning)
    ///
    /// Applied after tree growth to collapse branches where split gain < gamma.
    /// This solves the "Horizon Effect" where greedy algorithms miss good splits
    /// behind initially weak ones.
    ///
    /// Set to 0.0 to disable post-pruning (default).
    /// Typical values: 0.0 (disabled) to 5.0 (aggressive pruning).
    ///
    /// **Purpose**: Accuracy improvement (complements noise_pruning which is for safety)
    pub post_pruning_gamma: f32,

    /// Enable adaptive missing value direction selection
    ///
    /// When enabled, evaluates both child directions for missing values at each split
    /// and selects the direction that maximizes gain. When disabled (default), missing
    /// values follow the left child direction.
    ///
    /// Beneficial when missing values carry predictive information distinct from observed
    /// values. May increase training time due to additional gain evaluations.
    pub use_missing_value_learning: bool,

    // Random seed for reproducibility
    /// Random seed for train/validation splitting and subsampling
    pub seed: u64,
}

impl Default for GBDTConfig {
    fn default() -> Self {
        Self {
            // Ensemble
            num_rounds: gbdt_defaults::DEFAULT_NUM_ROUNDS,
            learning_rate: tree_defaults::DEFAULT_LEARNING_RATE,

            // Tree
            max_depth: tree_defaults::DEFAULT_MAX_DEPTH,
            max_leaves: tree_defaults::DEFAULT_MAX_LEAVES,
            min_samples_leaf: tree_defaults::DEFAULT_MIN_SAMPLES_LEAF,
            min_hessian_leaf: tree_defaults::DEFAULT_MIN_HESSIAN_LEAF,
            min_gain: tree_defaults::DEFAULT_MIN_GAIN,

            // Regularization
            lambda: tree_defaults::DEFAULT_TREE_LAMBDA,
            entropy_weight: tree_defaults::DEFAULT_ENTROPY_WEIGHT,

            // Loss
            loss_type: LossType::Mse,

            // Subsampling
            subsample: gbdt_defaults::DEFAULT_SUBSAMPLE,
            colsample: tree_defaults::DEFAULT_COLSAMPLE,

            // GOSS (disabled by default, most effective on large datasets)
            goss_enabled: false,
            goss_top_rate: gbdt_defaults::DEFAULT_GOSS_TOP_RATE,
            goss_other_rate: gbdt_defaults::DEFAULT_GOSS_OTHER_RATE,

            // Binning
            num_bins: gbdt_defaults::DEFAULT_NUM_BINS,

            // Conformal
            calibration_ratio: gbdt_defaults::DEFAULT_CALIBRATION_RATIO,
            conformal_quantile: gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE,

            // Early stopping (disabled by default)
            early_stopping_rounds: gbdt_defaults::DEFAULT_EARLY_STOPPING_ROUNDS,
            min_early_stopping_trees: gbdt_defaults::MIN_EARLY_STOPPING_TREES,
            validation_ratio: gbdt_defaults::DEFAULT_GBDT_VALIDATION_RATIO,

            // Performance optimizations (all ON by default)
            parallel_prediction: true,
            column_reordering: true,
            reordering_strategy: OrderingStrategy::ByImportance,
            packed_dataset: true,
            parallel_gradient: false, // Enable for large datasets (100k+ rows)

            // Backend selection (Auto = GPU for large datasets, CPU otherwise)
            backend_type: BackendType::Auto,
            gpu_mode: GpuMode::Auto,  // Auto-select: CUDA→Full, WGPU→Hybrid
            use_gpu_subgroups: false, // Disabled by default (minimal benefit on modern NVIDIA)

            // Monotonic constraints
            monotonic_constraints: Vec::new(),

            // Interaction constraints
            interaction_groups: Vec::new(),

            // Era-based splitting (disabled by default)
            era_splitting: false,

            // Noise pruning (disabled by default)
            noise_pruning: false,
            noise_pruning_threshold: -0.1, // Default threshold

            // Post-pruning (disabled by default)
            post_pruning_gamma: 0.0, // 0.0 = disabled

            // Missing value learning (disabled by default)
            use_missing_value_learning: false,

            // Random seed (matches legacy hardcoded value for backwards compatibility)
            seed: seeds_defaults::GBDT_SEED,
        }
    }
}

impl GBDTConfig {
    /// Create a new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a preset configuration.
    ///
    /// # Example
    /// ```no_run
    /// use treeboost::{GBDTConfig, GbdtPreset};
    ///
    /// // For high-dimensional, noisy data
    /// let config = GBDTConfig::default()
    ///     .with_preset(GbdtPreset::Robust);
    ///
    /// // For maximum accuracy
    /// let config = GBDTConfig::default()
    ///     .with_preset(GbdtPreset::Accuracy);
    /// ```
    pub fn with_preset(mut self, preset: GbdtPreset) -> Self {
        match preset {
            GbdtPreset::Standard => {
                // Use defaults (no changes needed)
            }
            GbdtPreset::Speed => {
                self.max_depth = tree_defaults::SHALLOW_MAX_DEPTH;
                self.goss_enabled = true;
                self.goss_top_rate = gbdt_defaults::DEFAULT_GOSS_TOP_RATE;
                self.goss_other_rate = gbdt_defaults::DEFAULT_GOSS_OTHER_RATE;
            }
            GbdtPreset::Accuracy => {
                self.max_depth = tree_defaults::DEEP_MAX_DEPTH;
                self.learning_rate = tree_defaults::DEFAULT_LEARNING_RATE * 0.5;
                self.num_rounds = gbdt_defaults::DEFAULT_NUM_ROUNDS * 2;
                // Feature/row sampling for better generalization
                self.subsample = gbdt_defaults::ROBUST_SUBSAMPLE;
                self.colsample = tree_defaults::ROBUST_COLSAMPLE;
            }
            GbdtPreset::SmallData => {
                self.subsample = gbdt_defaults::DEFAULT_SUBSAMPLE;
                self.colsample = tree_defaults::DEFAULT_COLSAMPLE;
                self.goss_enabled = false;
            }
            GbdtPreset::LargeData => {
                self.subsample = gbdt_defaults::LARGE_DATA_SUBSAMPLE;
                self.goss_enabled = true;
                self.goss_top_rate = gbdt_defaults::DEFAULT_GOSS_TOP_RATE;
                self.goss_other_rate = gbdt_defaults::DEFAULT_GOSS_OTHER_RATE;
            }
            GbdtPreset::Robust => {
                // Feature bagging (Random Forest-like robustness)
                self.colsample = tree_defaults::ROBUST_COLSAMPLE;
                // Row bagging (variance reduction)
                self.subsample = gbdt_defaults::ROBUST_SUBSAMPLE;
                // Disable GOSS (incompatible with bagging)
                self.goss_enabled = false;
            }
            GbdtPreset::Conformal => {
                self.calibration_ratio = gbdt_defaults::CONFORMAL_CALIBRATION_RATIO;
                self.conformal_quantile = gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE;
            }
        }
        self
    }

    /// Set number of boosting rounds
    pub fn with_num_rounds(mut self, num_rounds: usize) -> Self {
        self.num_rounds = num_rounds;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum tree depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set maximum leaves per tree
    pub fn with_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    /// Set L2 regularization
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set Shannon Entropy regularization weight
    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    /// Set loss function to MSE
    pub fn with_mse_loss(mut self) -> Self {
        self.loss_type = LossType::Mse;
        self
    }

    /// Set loss function to Pseudo-Huber
    pub fn with_pseudo_huber_loss(mut self, delta: f32) -> Self {
        self.loss_type = LossType::PseudoHuber { delta };
        self
    }

    /// Set loss function to Binary Log Loss (for binary classification)
    ///
    /// Uses sigmoid activation for probability output.
    /// Targets should be 0 or 1.
    pub fn with_binary_logloss(mut self) -> Self {
        self.loss_type = LossType::BinaryLogLoss;
        self
    }

    /// Set loss function to Multi-class Log Loss (for multi-class classification)
    ///
    /// Uses softmax activation for probability output.
    /// Targets should be class indices: 0, 1, 2, ..., num_classes-1.
    ///
    /// This trains K trees per round (one per class) and combines predictions
    /// via softmax for final class probabilities.
    ///
    /// # Arguments
    /// * `num_classes` - Number of classes (K)
    pub fn with_multiclass_logloss(mut self, num_classes: usize) -> crate::Result<Self> {
        if num_classes < 2 {
            return Err(crate::TreeBoostError::Config(format!(
                "num_classes must be >= 2, got {}",
                num_classes
            )));
        }
        self.loss_type = LossType::MultiClassLogLoss { num_classes };
        Ok(self)
    }

    /// Set loss function to Multi-label Log Loss (for multi-label classification)
    ///
    /// Uses sigmoid activation per label (independent binary classification per output).
    /// Targets should be binary (0/1) for each label, stored row-wise flattened:
    /// `[row0_label0, row0_label1, ..., row1_label0, ...]`
    ///
    /// This trains trees with vector leaf values that predict all labels simultaneously.
    ///
    /// # Arguments
    /// * `num_outputs` - Number of labels/outputs
    pub fn with_multilabel_logloss(mut self, num_outputs: usize) -> crate::Result<Self> {
        if num_outputs < 2 {
            return Err(crate::TreeBoostError::Config(format!(
                "num_outputs must be >= 2 for multi-label, got {}",
                num_outputs
            )));
        }
        self.loss_type = LossType::MultiLabelLogLoss { num_outputs };
        Ok(self)
    }

    /// Set loss function to Multi-label Focal Loss (for imbalanced multi-label classification)
    ///
    /// Focal loss down-weights easy examples to focus training on hard examples.
    /// Uses sigmoid activation per label (independent binary classification per output).
    ///
    /// # Arguments
    /// * `num_outputs` - Number of labels/outputs
    /// * `gamma` - Focusing parameter (typical values: 1-5, higher = more focusing)
    ///   - gamma=0 is equivalent to standard log loss
    ///   - gamma=2 is a common choice
    pub fn with_multilabel_focal_loss(
        mut self,
        num_outputs: usize,
        gamma: f32,
    ) -> crate::Result<Self> {
        if num_outputs < 2 {
            return Err(crate::TreeBoostError::Config(format!(
                "num_outputs must be >= 2 for multi-label, got {}",
                num_outputs
            )));
        }
        if gamma < 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "gamma must be >= 0, got {}",
                gamma
            )));
        }
        self.loss_type = LossType::MultiLabelFocalLoss { num_outputs, gamma };
        Ok(self)
    }

    /// Set row subsampling ratio
    pub fn with_subsample(mut self, ratio: f32) -> crate::Result<Self> {
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "subsample must be in (0, 1], got {}",
                ratio
            )));
        }
        self.subsample = ratio;
        Ok(self)
    }

    /// Set column subsampling ratio
    pub fn with_colsample(mut self, ratio: f32) -> crate::Result<Self> {
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "colsample must be in (0, 1], got {}",
                ratio
            )));
        }
        self.colsample = ratio;
        Ok(self)
    }

    /// Enable conformal prediction
    pub fn with_conformal(mut self, calibration_ratio: f32, quantile: f32) -> crate::Result<Self> {
        if !(0.0..1.0).contains(&calibration_ratio) {
            return Err(crate::TreeBoostError::Config(format!(
                "calibration_ratio must be in [0, 1), got {}",
                calibration_ratio
            )));
        }
        if quantile <= 0.0 || quantile >= 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "quantile must be in (0, 1), got {}",
                quantile
            )));
        }
        self.calibration_ratio = calibration_ratio;
        self.conformal_quantile = quantile;
        Ok(self)
    }

    /// Enable early stopping
    ///
    /// # Arguments
    /// * `rounds` - Number of consecutive rounds without improvement before stopping
    /// * `validation_ratio` - Fraction of data to use for validation (e.g., 0.1 for 10%)
    pub fn with_early_stopping(
        mut self,
        rounds: usize,
        validation_ratio: f32,
    ) -> crate::Result<Self> {
        if rounds == 0 {
            return Err(crate::TreeBoostError::Config(
                "early_stopping_rounds must be > 0".to_string(),
            ));
        }
        if validation_ratio <= 0.0 || validation_ratio >= 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "validation_ratio must be in (0, 1), got {}",
                validation_ratio
            )));
        }
        self.early_stopping_rounds = rounds;
        self.validation_ratio = validation_ratio;
        Ok(self)
    }

    /// Set minimum trees before early stopping can trigger
    ///
    /// Prevents early stopping from killing models too early.
    /// Default is 20 trees (`defaults::learners::gbdt::MIN_EARLY_STOPPING_TREES`).
    pub fn with_min_early_stopping_trees(mut self, min_trees: usize) -> Self {
        self.min_early_stopping_trees = min_trees;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set minimum hessian per leaf
    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    /// Set minimum gain for splitting
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Enable/disable parallel prediction (default: enabled)
    pub fn with_parallel_prediction(mut self, enabled: bool) -> Self {
        self.parallel_prediction = enabled;
        self
    }

    /// Enable/disable column reordering for cache locality (default: enabled)
    pub fn with_column_reordering(mut self, enabled: bool) -> Self {
        self.column_reordering = enabled;
        self
    }

    /// Set column reordering strategy
    pub fn with_reordering_strategy(mut self, strategy: OrderingStrategy) -> Self {
        self.reordering_strategy = strategy;
        self
    }

    /// Enable/disable 4-bit packed dataset for memory optimization (default: enabled)
    pub fn with_packed_dataset(mut self, enabled: bool) -> Self {
        self.packed_dataset = enabled;
        self
    }

    /// Enable parallel gradient computation (default: false)
    ///
    /// Experimental: may not provide stable speedups, benchmark before enabling.
    /// Use examples/find_crossover.rs to test on your specific data.
    pub fn with_parallel_gradient(mut self, enabled: bool) -> Self {
        self.parallel_gradient = enabled;
        self
    }

    /// Set the backend for histogram building
    ///
    /// # Backend Types
    /// - `Auto` (default): Uses GPU for datasets >= 50K rows, CPU otherwise
    /// - `Scalar`: Force CPU (AVX2/NEON optimized)
    /// - `Wgpu`: Force GPU (requires `gpu` feature)
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::{GBDTConfig, BackendType};
    ///
    /// // Force GPU backend for training
    /// let config = GBDTConfig::new()
    ///     .with_backend(BackendType::Wgpu);
    ///
    /// // Force CPU backend for reproducibility
    /// let config = GBDTConfig::new()
    ///     .with_backend(BackendType::Scalar);
    /// ```
    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Set the GPU execution mode
    ///
    /// # GPU Modes
    /// - `Auto` (default): Automatically select optimal mode per backend
    ///   - CUDA: Full (low dispatch latency)
    ///   - WGPU: Hybrid (high dispatch latency makes Full slower)
    /// - `Hybrid`: GPU histogram + CPU partition/split (best-first tree growth)
    /// - `Full`: Full GPU pipeline with level-wise tree growth
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::{GBDTConfig, BackendType, GpuMode};
    ///
    /// // Force full GPU mode for CUDA (level-wise tree growth)
    /// let config = GBDTConfig::new()
    ///     .with_backend(BackendType::Cuda)
    ///     .with_gpu_mode(GpuMode::Full);
    ///
    /// // Force hybrid mode (best-first tree growth with GPU histograms)
    /// let config = GBDTConfig::new()
    ///     .with_backend(BackendType::Wgpu)
    ///     .with_gpu_mode(GpuMode::Hybrid);
    /// ```
    pub fn with_gpu_mode(mut self, mode: GpuMode) -> Self {
        self.gpu_mode = mode;
        self
    }

    /// Enable or disable GPU subgroup operations
    ///
    /// Subgroups can reduce atomic contention but show minimal benefit on modern
    /// NVIDIA GPUs. May help on older AMD or Intel GPUs.
    ///
    /// Default: false (disabled)
    pub fn with_gpu_subgroups(mut self, enabled: bool) -> Self {
        self.use_gpu_subgroups = enabled;
        self
    }

    /// Disable all performance optimizations (for debugging/comparison)
    pub fn without_optimizations(mut self) -> Self {
        self.parallel_prediction = false;
        self.column_reordering = false;
        self.packed_dataset = false;
        self.parallel_gradient = false;
        self
    }

    /// Enable/disable GOSS (Gradient-based One-Side Sampling)
    pub fn with_goss(mut self, enabled: bool) -> Self {
        self.goss_enabled = enabled;
        self
    }

    /// Configure GOSS sampling rates
    ///
    /// # Arguments
    /// * `top_rate` - Ratio of large-gradient samples to keep (default: 0.2)
    /// * `other_rate` - Ratio of small-gradient samples to randomly sample (default: 0.1)
    pub fn with_goss_rates(mut self, top_rate: f32, other_rate: f32) -> Self {
        self.goss_enabled = true;
        self.goss_top_rate = top_rate;
        self.goss_other_rate = other_rate;
        self
    }

    /// Set monotonic constraints for features
    ///
    /// The vector should have one entry per feature. Features beyond the
    /// vector length are treated as unconstrained.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::{GBDTConfig, MonotonicConstraint};
    ///
    /// // Feature 0: increasing, Feature 1: decreasing, Feature 2: none
    /// let config = GBDTConfig::new()
    ///     .with_monotonic_constraints(vec![
    ///         MonotonicConstraint::Increasing,
    ///         MonotonicConstraint::Decreasing,
    ///         MonotonicConstraint::None,
    ///     ]);
    /// ```
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Set feature interaction constraints
    ///
    /// Features in the same group can interact (appear together in a tree path).
    /// Features in different groups cannot be used together.
    /// Features not in any group can interact with all features.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::GBDTConfig;
    ///
    /// // Features 0,1,2 can interact; features 3,4 can interact
    /// // Feature 5 is unconstrained (can interact with any)
    /// let config = GBDTConfig::new()
    ///     .with_interaction_groups(vec![vec![0, 1, 2], vec![3, 4]]);
    /// ```
    pub fn with_interaction_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.interaction_groups = groups;
        self
    }

    /// Enable era-based split finding (Directional Era Splitting)
    ///
    /// When enabled, only accepts splits where ALL eras agree on the split direction.
    /// This filters out spurious correlations that work in some eras but not others.
    ///
    /// Requires passing `era_indices` to the training method.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::GBDTConfig;
    ///
    /// let config = GBDTConfig::new()
    ///     .with_era_splitting(true);
    ///
    /// // Then train with era indices:
    /// // let model = GBDTModel::train_with_eras(&features, num_features, &targets, &era_indices, config)?;
    /// ```
    pub fn with_era_splitting(mut self, enabled: bool) -> Self {
        self.era_splitting = enabled;
        self
    }

    /// Enable noise pruning (PaloBoost-style validation)
    ///
    /// When enabled with `era_splitting`, treats Era 0 as training and Era 1 as validation.
    /// Splits are found using Era 0 gain, but rejected if Era 1 gain <= 0.
    ///
    /// This prevents overfitting by validating each split against held-out data.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::GBDTConfig;
    ///
    /// let config = GBDTConfig::new()
    ///     .with_era_splitting(true)
    ///     .with_noise_pruning(true);
    ///
    /// // Set era indices: 0 for train rows, 1 for validation rows
    /// // dataset.set_era_indices(era_indices);
    /// ```
    pub fn with_noise_pruning(mut self, enabled: bool) -> Self {
        self.noise_pruning = enabled;
        self
    }

    /// Set noise pruning threshold
    ///
    /// Splits are rejected if validation gain <= this threshold.
    /// - 0.0: Classic PaloBoost (only accept positive validation gains)
    /// - -0.1: Default (allows small negative gains, handles distribution shift)
    /// - -0.2: More permissive (for high temporal distribution shift)
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::GBDTConfig;
    ///
    /// let config = GBDTConfig::new()
    ///     .with_era_splitting(true)
    ///     .with_noise_pruning(true)
    ///     .with_noise_pruning_threshold(-0.15);  // Custom threshold
    /// ```
    pub fn with_noise_pruning_threshold(mut self, threshold: f32) -> Self {
        self.noise_pruning_threshold = threshold;
        self
    }

    /// Set post-pruning gamma threshold (Cost-Complexity Pruning)
    ///
    /// Applied after tree growth to collapse branches where split gain < gamma.
    /// This solves the "Horizon Effect" where greedy algorithms miss good splits
    /// behind initially weak ones.
    ///
    /// # Arguments
    /// * `gamma` - Minimum gain threshold. Set to 0.0 to disable (default).
    ///
    /// # Strategy
    /// 1. Grow trees with low/no min_gain (allow deep exploration)
    /// 2. Post-prune to remove branches that didn't contribute enough
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::GBDTConfig;
    ///
    /// let config = GBDTConfig::new()
    ///     .with_min_gain(0.0)           // Allow all splits during growth
    ///     .with_post_pruning_gamma(1.0); // Then prune weak branches
    /// ```
    ///
    /// Returns error if gamma is negative.
    pub fn with_post_pruning_gamma(mut self, gamma: f32) -> Result<Self, crate::TreeBoostError> {
        if gamma < 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "post_pruning_gamma must be >= 0.0, got {}",
                gamma
            )));
        }
        self.post_pruning_gamma = gamma;
        Ok(self)
    }

    /// Enable adaptive missing value direction selection
    ///
    /// When enabled, evaluates both child directions for missing values at each split
    /// and selects the direction that maximizes gain. When disabled (default), missing
    /// values follow the left child direction.
    ///
    /// Beneficial when missing values carry predictive information distinct from observed
    /// values. May increase training time due to additional gain evaluations.
    pub fn with_missing_value_learning(mut self, enabled: bool) -> Self {
        self.use_missing_value_learning = enabled;
        self
    }

    /// Set random seed for reproducibility
    ///
    /// Controls train/validation splitting and subsampling randomization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_rounds == 0 {
            return Err("num_rounds must be > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        if self.max_depth == 0 {
            return Err("max_depth must be > 0".to_string());
        }
        if self.max_leaves == 0 {
            return Err("max_leaves must be > 0".to_string());
        }
        if self.lambda < 0.0 {
            return Err("lambda must be >= 0".to_string());
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err("subsample must be in (0, 1]".to_string());
        }
        if self.colsample <= 0.0 || self.colsample > 1.0 {
            return Err("colsample must be in (0, 1]".to_string());
        }
        if self.goss_enabled {
            if self.goss_top_rate <= 0.0 || self.goss_top_rate >= 1.0 {
                return Err("goss_top_rate must be in (0, 1)".to_string());
            }
            if self.goss_other_rate <= 0.0 || self.goss_other_rate >= 1.0 {
                return Err("goss_other_rate must be in (0, 1)".to_string());
            }
            if self.goss_top_rate + self.goss_other_rate >= 1.0 {
                return Err("goss_top_rate + goss_other_rate must be < 1.0".to_string());
            }
        }
        if self.validation_ratio < 0.0 || self.validation_ratio >= 1.0 {
            return Err("validation_ratio must be in [0, 1)".to_string());
        }
        // Can't use both conformal calibration and early stopping validation from same data
        if self.calibration_ratio > 0.0 && self.validation_ratio > 0.0 {
            let total_holdout = self.calibration_ratio + self.validation_ratio;
            if total_holdout >= 1.0 {
                return Err("calibration_ratio + validation_ratio must be < 1.0".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GBDTConfig::default();

        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.max_depth, 6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = GBDTConfig::new()
            .with_num_rounds(50)
            .with_learning_rate(0.05)
            .with_max_depth(4)
            .with_pseudo_huber_loss(1.0)
            .with_entropy_weight(0.1)
            .with_conformal(0.1, 0.9)
            .unwrap();

        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.max_depth, 4);
        assert_eq!(config.loss_type, LossType::PseudoHuber { delta: 1.0 });
        assert_eq!(config.entropy_weight, 0.1);
        assert_eq!(config.calibration_ratio, 0.1);
    }

    #[test]
    fn test_config_validation() {
        let invalid = GBDTConfig::default().with_num_rounds(0);
        assert!(invalid.validate().is_err());

        let invalid = GBDTConfig {
            subsample: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
