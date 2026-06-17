//! LTT (LinearThenTree) AutoTuning
//!
//! Sequential hyperparameter tuning for LinearThenTree mode. LTT has TWO separate
//! hyperparameter spaces that must be tuned in sequence:
//!
//! 1. **Phase 1**: Tune linear model (alpha, l1_ratio)
//! 2. **Phase 1.5**: Select shrinkage factor based on linear R²
//! 3. **Phase 2**: Tune tree model on residuals (depth, learning_rate, etc.)
//! 4. **Phase 3**: Joint refinement (extrapolation damping)
//!
//! # Why Sequential?
//!
//! Tree hyperparameters depend on the LINEAR model's residuals. You CANNOT tune
//! them in parallel - the linear model must complete first to produce residuals
//! for tree training.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::tuner::ltt::{LttTuner, LttTunerConfig};
//!
//! // Prepare raw features (not binned) for linear model
//! let raw_features: Vec<f32> = /* ... */;
//! let targets: Vec<f32> = /* ... */;
//! let num_features = 10;
//!
//! let config = LttTunerConfig::default();
//! let tuner = LttTuner::new(config);
//!
//! let result = tuner.tune(&raw_features, num_features, &targets)?;
//! println!("Best linear lambda: {}", result.linear_params.lambda);
//! println!("Best tree depth: {}", result.tree_params.max_depth);
//! ```

use crate::analysis::{compute_mse, compute_r2};
use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::{BinnedDataset, FeatureInfo, QuantileBinner};
use crate::defaults::{
    learners::linear as linear_defaults, learners::tree as tree_defaults,
    tuning::ltt as ltt_defaults, tuning::seeds as seeds_defaults,
};
use crate::learner::{LinearBooster, LinearConfig, WeakLearner};
use crate::{Result, TreeBoostError};
use std::time::{Duration, Instant};

// =============================================================================
// Data Structures
// =============================================================================

/// Linear phase hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct LinearHyperparams {
    /// Regularization strength (higher = more regularization)
    /// Range: [0.001, 10.0], log scale
    pub lambda: f32,

    /// L1/L2 ratio: 0 = Ridge (pure L2), 1 = LASSO (pure L1), between = ElasticNet
    /// Range: [0.0, 1.0]
    pub l1_ratio: f32,

    /// Shrinkage factor for ensemble weighting
    /// Controls how much linear model contributes to final prediction
    /// Range: [0.1, 1.0]. Higher = trust linear more.
    pub shrinkage_factor: f32,

    /// Extrapolation damping toward target mean for OOD safety
    /// Range: [0.0, 0.5]
    pub extrapolation_damping: f32,
}

/// Presets for linear hyperparameters in LTT tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearHyperparamsPreset {
    Ridge,
    Lasso,
    ElasticNet,
}

impl Default for LinearHyperparams {
    fn default() -> Self {
        Self {
            lambda: linear_defaults::DEFAULT_LAMBDA,
            l1_ratio: linear_defaults::DEFAULT_L1_RATIO, // Ridge by default
            shrinkage_factor: ltt_defaults::DEFAULT_LTT_SHRINKAGE,
            extrapolation_damping: linear_defaults::DEFAULT_EXTRAPOLATION_DAMPING,
        }
    }
}

impl LinearHyperparams {
    /// Convert to LinearConfig
    pub fn to_config(&self) -> LinearConfig {
        LinearConfig::default()
            .with_lambda(self.lambda)
            .expect("tuned lambda is valid")
            .with_l1_ratio(self.l1_ratio)
            .expect("tuned l1_ratio is valid")
            .with_shrinkage_factor(self.shrinkage_factor)
            .expect("tuned shrinkage_factor is valid")
            .with_extrapolation_damping(self.extrapolation_damping)
            .expect("tuned extrapolation_damping is valid")
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: LinearHyperparamsPreset) -> Self {
        match preset {
            LinearHyperparamsPreset::Ridge => {
                self.l1_ratio = 0.0;
            }
            LinearHyperparamsPreset::Lasso => {
                self.l1_ratio = 1.0;
            }
            LinearHyperparamsPreset::ElasticNet => {
                self.l1_ratio = 0.5;
            }
        }
        self
    }
}

/// Tree phase hyperparameters (applied to residuals)
#[derive(Debug, Clone, Copy)]
pub struct TreeHyperparams {
    /// Maximum tree depth
    /// Range: [3, 12]
    pub max_depth: u32,

    /// Learning rate (step size shrinkage)
    /// Range: [0.01, 0.3]
    pub learning_rate: f32,

    /// Number of boosting rounds
    /// Range: [100, 2000]
    pub num_rounds: u32,

    /// Minimum sum of hessians in a leaf
    /// Range: [1.0, 10.0]
    pub min_child_weight: f32,

    /// Row subsampling ratio
    /// Range: [0.6, 1.0]
    pub subsample: f32,

    /// Column subsampling ratio per tree
    /// Range: [0.6, 1.0]
    pub colsample_bytree: f32,
}

/// Presets for tree hyperparameters in LTT tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeHyperparamsPreset {
    /// depth=4, lr=0.05, rounds=1000, min_weight=3.0
    Conservative,
    /// depth=8, lr=0.15, rounds=500, min_weight=1.0
    Aggressive,
}

impl Default for TreeHyperparams {
    fn default() -> Self {
        Self {
            max_depth: tree_defaults::DEFAULT_MAX_DEPTH as u32,
            learning_rate: tree_defaults::DEFAULT_LEARNING_RATE,
            num_rounds: 500,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
        }
    }
}

impl TreeHyperparams {
    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: TreeHyperparamsPreset) -> Self {
        match preset {
            TreeHyperparamsPreset::Conservative => {
                self.max_depth = 4;
                self.learning_rate = 0.05;
                self.num_rounds = 1000;
                self.min_child_weight = 3.0;
                self.subsample = 0.8;
                self.colsample_bytree = 0.8;
            }
            TreeHyperparamsPreset::Aggressive => {
                self.max_depth = 8;
                self.learning_rate = 0.15;
                self.num_rounds = 500;
                self.min_child_weight = 1.0;
                self.subsample = 1.0;
                self.colsample_bytree = 1.0;
            }
        }
        self
    }
}

/// Combined LTT configuration
#[derive(Debug, Clone, Copy, Default)]
pub struct LttConfig {
    pub linear: LinearHyperparams,
    pub tree: TreeHyperparams,
}

/// Result of LTT tuning
#[derive(Debug, Clone)]
pub struct LttTuningResult {
    /// Best linear hyperparameters
    pub linear_params: LinearHyperparams,
    /// Best tree hyperparameters
    pub tree_params: TreeHyperparams,
    /// Linear phase R² on validation
    pub linear_r2: f32,
    /// Final combined RMSE on validation
    pub final_rmse: f32,
    /// Total tuning time
    pub total_time: Duration,
    /// Phase timings
    pub phase_times: PhaseTimes,
    /// Tuning history
    pub history: LttTuningHistory,
}

/// Time breakdown by phase
#[derive(Debug, Clone, Default)]
pub struct PhaseTimes {
    pub linear_phase: Duration,
    pub tree_phase: Duration,
    pub joint_phase: Duration,
}

/// Tuning history for logging/debugging
#[derive(Debug, Clone, Default)]
pub struct LttTuningHistory {
    pub linear_trials: Vec<LinearTrial>,
    pub tree_trials: Vec<TreeTrial>,
    pub joint_trials: Vec<JointTrial>,
}

/// Single linear tuning trial
#[derive(Debug, Clone)]
pub struct LinearTrial {
    pub lambda: f32,
    pub l1_ratio: f32,
    pub r2: f32,
    pub rmse: f32,
}

/// Single tree tuning trial
#[derive(Debug, Clone)]
pub struct TreeTrial {
    pub max_depth: u32,
    pub learning_rate: f32,
    /// Actual number of trees built (after early stopping)
    pub num_trees: u32,
    pub residual_rmse: f32,
}

/// Single joint refinement trial
#[derive(Debug, Clone)]
pub struct JointTrial {
    pub extrapolation_damping: f32,
    pub combined_rmse: f32,
}

// =============================================================================
// Data Split - Encapsulates train/validation data to reduce parameter count
// =============================================================================

/// Encapsulates train/validation split data
///
/// This struct groups related data to reduce function parameter count
/// and make the API cleaner.
struct DataSplit<'a> {
    train_features: &'a [f32],
    train_targets: &'a [f32],
    val_features: &'a [f32],
    val_targets: &'a [f32],
    num_features: usize,
}

impl<'a> DataSplit<'a> {
    fn new(
        train_features: &'a [f32],
        train_targets: &'a [f32],
        val_features: &'a [f32],
        val_targets: &'a [f32],
        num_features: usize,
    ) -> Self {
        Self {
            train_features,
            train_targets,
            val_features,
            val_targets,
            num_features,
        }
    }
}

/// Result of evaluating a linear configuration
struct LinearEvalResult {
    train_preds: Vec<f32>,
    val_preds: Vec<f32>,
    r2: f32,
    rmse: f32,
}

// =============================================================================
// LTT Tuner Configuration
// =============================================================================

/// LTT Tuner configuration
#[derive(Debug, Clone)]
pub struct LttTunerConfig {
    /// Validation split ratio (must be in (0.0, 1.0))
    pub val_ratio: f32,

    // Linear phase config
    /// Lambda values to try (log scale)
    pub lambda_values: Vec<f32>,
    /// L1 ratio values to try
    pub l1_ratio_values: Vec<f32>,

    // Tree phase config
    /// Max depth values to try
    pub max_depth_values: Vec<u32>,
    /// Learning rate values to try
    pub learning_rate_values: Vec<f32>,
    /// Num rounds values to try
    pub num_rounds_values: Vec<u32>,
    /// Subsample ratio values to try (row sampling)
    pub subsample_values: Vec<f32>,
    /// Colsample ratio values to try (column sampling per tree)
    pub colsample_values: Vec<f32>,

    // Shrinkage factor tuning (Phase 1.5)
    /// Shrinkage factor values to try for ensemble weighting
    /// These control how much linear model contributes vs trees
    pub shrinkage_factor_values: Vec<f32>,

    // Joint refinement config (extrapolation damping)
    /// Extrapolation damping values to try
    pub extrapolation_damping_values: Vec<f32>,
    /// Enable joint refinement phase
    pub enable_joint_refinement: bool,

    // AutoTuner configuration for tree phase
    /// Number of iterations for tree phase AutoTuner (progressive zoom)
    /// - Quick: 1 iteration (grid search only)
    /// - Standard: 2 iterations (grid + 1 zoom)
    /// - Thorough: 3+ iterations (grid + multiple zooms)
    pub tree_tuner_iterations: usize,

    // Output configuration
    /// Output directory for tuning logs (AutoTuner logs will be in {output_dir}/autotuner/)
    pub output_dir: Option<std::path::PathBuf>,

    /// Seed for reproducibility
    pub seed: u64,
}

/// Presets for LTT tuner configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LttTunerPreset {
    /// Coarse grid, no joint refinement - fast.
    Quick,
    /// Standard grid with joint refinement.
    Standard,
    /// Dense grid, joint refinement - comprehensive.
    Thorough,
    /// Skip phase 2/joint refinement, focus on shrinkage only.
    ShrinkageOnly,
}

impl Default for LttTunerConfig {
    fn default() -> Self {
        Self {
            val_ratio: ltt_defaults::DEFAULT_LTT_VAL_RATIO,
            // Linear phase: 4×3 = 12 trials
            lambda_values: ltt_defaults::DEFAULT_LAMBDA_GRID.to_vec(),
            l1_ratio_values: ltt_defaults::DEFAULT_L1_RATIO_GRID.to_vec(), // Ridge, ElasticNet, LASSO
            // Tree phase: 3×3×3×3×3 = 243 trials (with subsample and colsample)
            max_depth_values: ltt_defaults::DEFAULT_MAX_DEPTH_GRID.to_vec(),
            learning_rate_values: ltt_defaults::DEFAULT_LR_GRID.to_vec(),
            num_rounds_values: ltt_defaults::DEFAULT_ROUNDS_GRID.to_vec(),
            subsample_values: ltt_defaults::DEFAULT_SUBSAMPLE_GRID.to_vec(),
            colsample_values: ltt_defaults::DEFAULT_COLSAMPLE_GRID.to_vec(),
            // Shrinkage factor: 5 values centered around typical optimal (0.5-0.9)
            shrinkage_factor_values: ltt_defaults::DEFAULT_SHRINKAGE_GRID.to_vec(),
            // Extrapolation damping: usually 0 unless OOD is a concern
            extrapolation_damping_values: ltt_defaults::DEFAULT_EXTRAPOLATION_DAMPING_GRID.to_vec(),
            enable_joint_refinement: true,
            tree_tuner_iterations: 2, // Standard: 2 iterations (grid + 1 zoom)
            output_dir: None,
            seed: seeds_defaults::DEFAULT_SEED,
        }
    }
}

impl LttTunerConfig {
    /// Set the output directory for tuning logs
    pub fn with_output_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.output_dir = Some(dir);
        self
    }

    /// Apply a preset configuration.
    pub fn with_preset(self, preset: LttTunerPreset) -> Self {
        let output_dir = self.output_dir.clone(); // Preserve output_dir
        let mut config = match preset {
            LttTunerPreset::Quick => Self {
                val_ratio: ltt_defaults::DEFAULT_LTT_VAL_RATIO,
                lambda_values: ltt_defaults::QUICK_LAMBDA_GRID.to_vec(),
                l1_ratio_values: ltt_defaults::QUICK_L1_RATIO_GRID.to_vec(),
                max_depth_values: ltt_defaults::QUICK_MAX_DEPTH_GRID.to_vec(),
                learning_rate_values: ltt_defaults::QUICK_LR_GRID.to_vec(),
                num_rounds_values: ltt_defaults::QUICK_ROUNDS_GRID.to_vec(),
                subsample_values: ltt_defaults::QUICK_SUBSAMPLE_GRID.to_vec(),
                colsample_values: ltt_defaults::QUICK_COLSAMPLE_GRID.to_vec(),
                // Quick: 3 shrinkage values
                shrinkage_factor_values: ltt_defaults::QUICK_SHRINKAGE_GRID.to_vec(),
                extrapolation_damping_values: ltt_defaults::QUICK_EXTRAPOLATION_DAMPING_GRID
                    .to_vec(),
                enable_joint_refinement: false, // Skip for quick mode
                tree_tuner_iterations: 2, // Quick: 2 iterations to match regular AutoTuner::Quick
                output_dir: None,
                seed: seeds_defaults::DEFAULT_SEED,
            },
            LttTunerPreset::Standard => Self::default(),
            LttTunerPreset::Thorough => Self {
                val_ratio: ltt_defaults::DEFAULT_LTT_VAL_RATIO,
                lambda_values: ltt_defaults::THOROUGH_LAMBDA_GRID.to_vec(),
                l1_ratio_values: ltt_defaults::THOROUGH_L1_RATIO_GRID.to_vec(),
                max_depth_values: ltt_defaults::THOROUGH_MAX_DEPTH_GRID.to_vec(),
                learning_rate_values: ltt_defaults::THOROUGH_LR_GRID.to_vec(),
                num_rounds_values: ltt_defaults::THOROUGH_ROUNDS_GRID.to_vec(),
                subsample_values: ltt_defaults::THOROUGH_SUBSAMPLE_GRID.to_vec(),
                colsample_values: ltt_defaults::THOROUGH_COLSAMPLE_GRID.to_vec(),
                // Thorough: 7 shrinkage values
                shrinkage_factor_values: ltt_defaults::THOROUGH_SHRINKAGE_GRID.to_vec(),
                extrapolation_damping_values: ltt_defaults::THOROUGH_EXTRAPOLATION_DAMPING_GRID
                    .to_vec(),
                enable_joint_refinement: true,
                tree_tuner_iterations: 3, // Thorough: grid + 2 zoom iterations
                output_dir: None,
                seed: seeds_defaults::DEFAULT_SEED,
            },
            LttTunerPreset::ShrinkageOnly => {
                let mut config = Self::default();
                let linear_defaults = LinearHyperparams::default();
                let tree_defaults = TreeHyperparams::default();
                config.lambda_values = vec![linear_defaults.lambda];
                config.l1_ratio_values = vec![linear_defaults.l1_ratio];
                config.max_depth_values = vec![tree_defaults.max_depth];
                config.learning_rate_values = vec![tree_defaults.learning_rate];
                config.num_rounds_values = vec![tree_defaults.num_rounds];
                config.subsample_values = vec![tree_defaults.subsample];
                config.colsample_values = vec![tree_defaults.colsample_bytree];
                config.enable_joint_refinement = false;
                config
            }
        };
        config.output_dir = output_dir; // Restore output_dir
        config
    }

    /// Validate configuration parameters
    fn validate(&self) -> Result<()> {
        // Validate val_ratio
        if self.val_ratio <= 0.0 || self.val_ratio >= 1.0 {
            return Err(TreeBoostError::Config(format!(
                "val_ratio must be in (0.0, 1.0), got {}",
                self.val_ratio
            )));
        }

        // Validate non-empty configuration grids
        if self.lambda_values.is_empty() {
            return Err(TreeBoostError::Config(
                "lambda_values cannot be empty".into(),
            ));
        }
        if self.l1_ratio_values.is_empty() {
            return Err(TreeBoostError::Config(
                "l1_ratio_values cannot be empty".into(),
            ));
        }
        if self.max_depth_values.is_empty() {
            return Err(TreeBoostError::Config(
                "max_depth_values cannot be empty".into(),
            ));
        }
        if self.learning_rate_values.is_empty() {
            return Err(TreeBoostError::Config(
                "learning_rate_values cannot be empty".into(),
            ));
        }
        if self.num_rounds_values.is_empty() {
            return Err(TreeBoostError::Config(
                "num_rounds_values cannot be empty".into(),
            ));
        }
        if self.shrinkage_factor_values.is_empty() {
            return Err(TreeBoostError::Config(
                "shrinkage_factor_values cannot be empty".into(),
            ));
        }
        if self.enable_joint_refinement && self.extrapolation_damping_values.is_empty() {
            return Err(TreeBoostError::Config(
                "extrapolation_damping_values cannot be empty when joint refinement is enabled"
                    .into(),
            ));
        }

        Ok(())
    }
}

// =============================================================================
// LTT Tuner Implementation
// =============================================================================

/// LTT Tuner for sequential hyperparameter optimization
pub struct LttTuner {
    config: LttTunerConfig,
}

impl LttTuner {
    /// Create a new LTT tuner with the given configuration
    pub fn new(config: LttTunerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(LttTunerConfig::default())
    }

    /// Get estimated number of trials
    pub fn estimated_trials(&self) -> usize {
        let linear_trials = self.config.lambda_values.len() * self.config.l1_ratio_values.len();
        let tree_trials = self.config.max_depth_values.len()
            * self.config.learning_rate_values.len()
            * self.config.num_rounds_values.len();
        let joint_trials = if self.config.enable_joint_refinement {
            self.config.extrapolation_damping_values.len()
        } else {
            0
        };
        linear_trials + tree_trials + joint_trials
    }

    /// Run sequential tuning
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    /// * `targets` - Target values
    ///
    /// # Returns
    /// Best configuration and tuning history
    ///
    /// # Errors
    /// Returns error if:
    /// - Configuration is invalid (empty grids, bad val_ratio)
    /// - Input data is invalid (empty, mismatched dimensions)
    /// - Split produces empty train/validation sets
    pub fn tune(
        &self,
        features: &[f32],
        num_features: usize,
        targets: &[f32],
        linear_indices: &[usize],
    ) -> Result<LttTuningResult> {
        // === Comprehensive input validation ===
        self.validate_inputs(features, num_features, targets)?;

        let start = Instant::now();
        let mut history = LttTuningHistory::default();
        let mut phase_times = PhaseTimes::default();

        let num_rows = targets.len();

        // Create train/val split indices
        let val_size = ((num_rows as f32) * self.config.val_ratio).ceil() as usize;
        let train_size = num_rows - val_size;

        // Validate split produces valid sets
        if train_size == 0 {
            return Err(TreeBoostError::Data(format!(
                "Train/val split produced empty training set (val_ratio={}, num_rows={})",
                self.config.val_ratio, num_rows
            )));
        }
        if val_size == 0 {
            return Err(TreeBoostError::Data(format!(
                "Train/val split produced empty validation set (val_ratio={}, num_rows={})",
                self.config.val_ratio, num_rows
            )));
        }

        // Simple split (last val_ratio% as validation)
        let train_indices: Vec<usize> = (0..train_size).collect();
        let val_indices: Vec<usize> = (train_size..num_rows).collect();

        // Extract train/val data
        let (train_features, train_targets) =
            Self::extract_split(features, targets, num_features, &train_indices);
        let (val_features, val_targets) =
            Self::extract_split(features, targets, num_features, &val_indices);

        let split = DataSplit::new(
            &train_features,
            &train_targets,
            &val_features,
            &val_targets,
            num_features,
        );

        // === PHASE 1: Tune linear model (lambda, l1_ratio) ===
        let phase1_start = Instant::now();
        let (mut best_linear, linear_r2, linear_train_preds, linear_val_preds) =
            self.tune_linear_phase(&split, &mut history)?;
        phase_times.linear_phase = phase1_start.elapsed();

        // === PHASE 1.5: Select shrinkage_factor based on linear R² ===
        // This determines how much to trust linear vs trees
        let best_shrinkage =
            self.select_shrinkage_factor(&linear_train_preds, &linear_val_preds, &split);
        best_linear.shrinkage_factor = best_shrinkage;

        // Compute residuals WITH shrinkage applied
        // Trees will learn to fix what's left AFTER shrinkage scaling
        let train_residuals: Vec<f32> = train_targets
            .iter()
            .zip(linear_train_preds.iter())
            .map(|(&t, &p)| t - best_shrinkage * p)
            .collect();
        let val_residuals: Vec<f32> = val_targets
            .iter()
            .zip(linear_val_preds.iter())
            .map(|(&t, &p)| t - best_shrinkage * p)
            .collect();

        // === PHASE 2: Tune tree model on shrinkage-adjusted residuals ===
        let phase2_start = Instant::now();
        let best_tree = self.tune_tree_phase(
            &split,
            &train_residuals,
            &val_residuals,
            linear_indices,
            &mut history,
        )?;
        phase_times.tree_phase = phase2_start.elapsed();

        // === PHASE 3: Joint refinement (extrapolation damping, optional) ===
        let mut final_linear = best_linear;
        if self.config.enable_joint_refinement {
            let phase3_start = Instant::now();
            final_linear = self.tune_joint_phase(&split, &best_linear, &mut history)?;
            phase_times.joint_phase = phase3_start.elapsed();
        }

        // Compute final RMSE (combined LTT: shrinkage*linear + tree)
        let final_rmse = self.compute_final_rmse(&final_linear, &best_tree, &split)?;

        Ok(LttTuningResult {
            linear_params: final_linear,
            tree_params: best_tree,
            linear_r2,
            final_rmse,
            total_time: start.elapsed(),
            phase_times,
            history,
        })
    }

    /// Comprehensive input validation
    fn validate_inputs(
        &self,
        features: &[f32],
        num_features: usize,
        targets: &[f32],
    ) -> Result<()> {
        // Validate configuration
        self.config.validate()?;

        // Validate num_features
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "num_features must be greater than 0".into(),
            ));
        }

        // Validate non-empty inputs
        if targets.is_empty() {
            return Err(TreeBoostError::Data("targets cannot be empty".into()));
        }
        if features.is_empty() {
            return Err(TreeBoostError::Data("features cannot be empty".into()));
        }

        // Validate feature matrix dimensions
        let num_rows = targets.len();
        let expected_features = num_rows * num_features;
        if features.len() != expected_features {
            return Err(TreeBoostError::Data(format!(
                "Feature matrix size mismatch: expected {} ({}×{}), got {}",
                expected_features,
                num_rows,
                num_features,
                features.len()
            )));
        }

        // Validate minimum data size for meaningful tuning
        let min_rows = 10; // Need at least some data for train/val split
        if num_rows < min_rows {
            return Err(TreeBoostError::Data(format!(
                "Insufficient data for tuning: need at least {} rows, got {}",
                min_rows, num_rows
            )));
        }

        Ok(())
    }

    /// Extract features and targets for a subset of indices
    fn extract_split(
        features: &[f32],
        targets: &[f32],
        num_features: usize,
        indices: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut split_features = Vec::with_capacity(indices.len() * num_features);
        let mut split_targets = Vec::with_capacity(indices.len());

        for &idx in indices {
            for f in 0..num_features {
                split_features.push(features[idx * num_features + f]);
            }
            split_targets.push(targets[idx]);
        }

        (split_features, split_targets)
    }

    // =========================================================================
    // Helper: Linear Model Evaluation (eliminates code duplication)
    // =========================================================================

    /// Train and evaluate a linear model configuration
    ///
    /// This helper method encapsulates the common pattern of:
    /// 1. Creating a LinearBooster with given config
    /// 2. Fitting on training data
    /// 3. Predicting on validation data
    /// 4. Computing metrics (R², RMSE)
    ///
    /// Used by tune_linear_phase, tune_joint_phase, and compute_final_rmse.
    fn evaluate_linear_config(config: LinearConfig, split: &DataSplit) -> Result<LinearEvalResult> {
        let mut booster = LinearBooster::new(split.num_features, config);

        // Fit on training data
        let train_preds = booster.fit_direct(
            split.train_features,
            split.num_features,
            split.train_targets,
        )?;

        // Predict on validation data
        let val_preds = booster.predict_batch(split.val_features, split.num_features);

        // Compute metrics using shared utilities
        let r2 = compute_r2(split.val_targets, &val_preds);
        let mse = compute_mse(split.val_targets, &val_preds);
        let rmse = mse.sqrt();

        Ok(LinearEvalResult {
            train_preds,
            val_preds,
            r2,
            rmse,
        })
    }

    // =========================================================================
    // Phase 1: Linear Model Tuning
    // =========================================================================

    /// Phase 1: Tune linear model hyperparameters (lambda, l1_ratio)
    ///
    /// Returns: (best_params, best_r2, train_preds, val_preds)
    fn tune_linear_phase(
        &self,
        split: &DataSplit,
        history: &mut LttTuningHistory,
    ) -> Result<(LinearHyperparams, f32, Vec<f32>, Vec<f32>)> {
        let mut best_params = LinearHyperparams::default();
        let mut best_r2 = f32::NEG_INFINITY;
        let mut best_train_preds: Option<Vec<f32>> = None;
        let mut best_val_preds: Option<Vec<f32>> = None;

        for &lambda in &self.config.lambda_values {
            for &l1_ratio in &self.config.l1_ratio_values {
                let config = LinearConfig::default()
                    .with_lambda(lambda)?
                    .with_l1_ratio(l1_ratio)?;

                let eval_result = Self::evaluate_linear_config(config, split)?;

                history.linear_trials.push(LinearTrial {
                    lambda,
                    l1_ratio,
                    r2: eval_result.r2,
                    rmse: eval_result.rmse,
                });

                if eval_result.r2 > best_r2 {
                    best_r2 = eval_result.r2;
                    best_params = LinearHyperparams {
                        lambda,
                        l1_ratio,
                        shrinkage_factor: ltt_defaults::DEFAULT_LTT_SHRINKAGE,
                        extrapolation_damping: 0.0,
                    };
                    best_train_preds = Some(eval_result.train_preds);
                    best_val_preds = Some(eval_result.val_preds);
                }
            }
        }

        // Safety: We validated config has non-empty grids, so at least one iteration ran
        let train_preds = best_train_preds
            .ok_or_else(|| TreeBoostError::Training("Linear phase produced no results".into()))?;
        let val_preds = best_val_preds
            .ok_or_else(|| TreeBoostError::Training("Linear phase produced no results".into()))?;

        Ok((best_params, best_r2, train_preds, val_preds))
    }

    // =========================================================================
    // Phase 1.5: Shrinkage Factor Selection
    // =========================================================================

    /// Phase 1.5: Select optimal shrinkage_factor (ensemble weight tuning)
    ///
    /// After the linear model is trained, this phase decides how much weight to
    /// give the linear model in the final ensemble:
    /// - shrinkage = 1.0 → fully trust linear predictions (trees fit residuals as-is)
    /// - shrinkage < 1.0 → scale down linear predictions, trees fit larger residuals
    ///
    /// # Strategy
    ///
    /// Train a tiny tree model on residuals for each candidate shrinkage and
    /// select the value that minimizes combined validation RMSE.
    ///
    /// # Arguments
    /// * `linear_train_preds` - Training predictions from best linear model
    /// * `linear_val_preds` - Validation predictions from best linear model
    /// * `split` - Train/validation split data
    fn select_shrinkage_factor(
        &self,
        linear_train_preds: &[f32],
        linear_val_preds: &[f32],
        split: &DataSplit,
    ) -> f32 {
        // If only one shrinkage value configured, use it
        if self.config.shrinkage_factor_values.len() <= 1 {
            return self
                .config
                .shrinkage_factor_values
                .first()
                .copied()
                .unwrap_or(ltt_defaults::DEFAULT_LTT_SHRINKAGE);
        }

        let candidates = self.config.shrinkage_factor_values.clone();

        let (train_binned, val_binned, feature_info) = Self::build_probe_binned_datasets(split);

        struct ShrinkageScore {
            shrinkage: f32,
            score: f32,
        }

        let mut scores: Vec<ShrinkageScore> = Vec::with_capacity(candidates.len());

        for &shrinkage in &candidates {
            let train_residuals: Vec<f32> = split
                .train_targets
                .iter()
                .zip(linear_train_preds.iter())
                .map(|(&t, &p)| t - shrinkage * p)
                .collect();
            let val_residuals: Vec<f32> = split
                .val_targets
                .iter()
                .zip(linear_val_preds.iter())
                .map(|(&t, &p)| t - shrinkage * p)
                .collect();

            let train_dataset = BinnedDataset::new(
                split.train_targets.len(),
                train_binned.clone(),
                train_residuals,
                feature_info.clone(),
            );
            let val_dataset = BinnedDataset::new(
                split.val_targets.len(),
                val_binned.clone(),
                val_residuals,
                feature_info.clone(),
            );

            let probe_config = GBDTConfig::new()
                .with_mse_loss()
                .with_num_rounds(ltt_defaults::SHRINKAGE_PROBE_ROUNDS)
                .with_learning_rate(ltt_defaults::SHRINKAGE_PROBE_LR)
                .with_max_depth(ltt_defaults::SHRINKAGE_PROBE_DEPTH)
                .with_min_samples_leaf(ltt_defaults::SHRINKAGE_PROBE_MIN_SAMPLES_LEAF)
                .with_seed(self.config.seed);

            let probe_model =
                GBDTModel::train_binned(&train_dataset, probe_config).unwrap_or_else(|_| {
                    GBDTModel::train_binned(
                        &train_dataset,
                        GBDTConfig::new()
                            .with_mse_loss()
                            .with_num_rounds(1)
                            .with_max_depth(1)
                            .with_seed(self.config.seed),
                    )
                    .expect("Probe fallback should succeed")
                });

            let residual_preds = probe_model.predict(&val_dataset);
            let combined_preds: Vec<f32> = linear_val_preds
                .iter()
                .zip(residual_preds.iter())
                .map(|(&p, &r)| shrinkage * p + r)
                .collect();

            let abs_errors: Vec<f32> = combined_preds
                .iter()
                .zip(split.val_targets.iter())
                .map(|(&p, &t)| (p - t).abs())
                .collect();
            let mae = abs_errors.iter().sum::<f32>() / abs_errors.len().max(1) as f32;
            let rmse = compute_mse(split.val_targets, &combined_preds).sqrt();

            let mean_error = mae;
            let std = {
                let var = abs_errors
                    .iter()
                    .map(|&e| {
                        let d = e - mean_error;
                        d * d
                    })
                    .sum::<f32>()
                    / abs_errors.len().max(1) as f32;
                var.sqrt()
            };

            let score = rmse + 0.5 * mae + 0.2 * std;

            scores.push(ShrinkageScore { shrinkage, score });
        }

        let best_by = |f: fn(&ShrinkageScore) -> f32| {
            scores
                .iter()
                .min_by(|a, b| f(a).partial_cmp(&f(b)).unwrap())
                .map(|s| (s.shrinkage, f(s)))
                .unwrap()
        };

        let (best_shrinkage, _best_metric) = best_by(|s| s.score);
        best_shrinkage
    }

    fn build_probe_binned_datasets(split: &DataSplit) -> (Vec<u8>, Vec<u8>, Vec<FeatureInfo>) {
        let num_train_rows = split.train_targets.len();
        let num_val_rows = split.val_targets.len();
        let num_features = split.num_features;
        let binner = QuantileBinner::new(ltt_defaults::SHRINKAGE_PROBE_BINS);

        let mut feature_info = Vec::with_capacity(num_features);
        let mut boundaries: Vec<Vec<f64>> = Vec::with_capacity(num_features);

        for feat in 0..num_features {
            let mut combined = Vec::with_capacity(num_train_rows + num_val_rows);
            for row in 0..num_train_rows {
                combined.push(split.train_features[row * num_features + feat] as f64);
            }
            for row in 0..num_val_rows {
                combined.push(split.val_features[row * num_features + feat] as f64);
            }
            let bins = binner.compute_boundaries(&combined);
            boundaries.push(bins.clone());
            feature_info.push(binner.create_feature_info(format!("f{}", feat), bins));
        }

        let train_binned = Self::bin_features(
            split.train_features,
            num_train_rows,
            num_features,
            &boundaries,
            &binner,
        );
        let val_binned = Self::bin_features(
            split.val_features,
            num_val_rows,
            num_features,
            &boundaries,
            &binner,
        );

        (train_binned, val_binned, feature_info)
    }

    fn bin_features(
        features: &[f32],
        num_rows: usize,
        num_features: usize,
        boundaries: &[Vec<f64>],
        binner: &QuantileBinner,
    ) -> Vec<u8> {
        let mut binned = Vec::with_capacity(num_rows * num_features);
        for feat in 0..num_features {
            let mut values = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                values.push(features[row * num_features + feat] as f64);
            }
            let column = binner.bin_column(&values, &boundaries[feat]);
            binned.extend_from_slice(&column);
        }
        binned
    }

    // =========================================================================
    // Phase 2: Tree Model Tuning
    // =========================================================================

    /// Phase 2: Tune tree model hyperparameters on residuals
    ///
    /// Uses AutoTuner to perform real model training and validation on residuals.
    /// This replaces the old heuristic scoring approach with proper hyperparameter
    /// optimization.
    ///
    /// Trees only use features NOT used by linear model (e.g., categorical features,
    /// excluding polynomial/interaction features that linear already used).
    ///
    /// # Arguments
    ///
    /// * `split` - Train/validation data split (features and original targets)
    /// * `train_residuals` - Training residuals from linear model (used as targets for tree)
    /// * `val_residuals` - Validation residuals from linear model
    /// * `linear_indices` - Feature indices used by linear model
    /// * `history` - Tuning history to append tree trials to
    ///
    /// # Returns
    ///
    /// Best tree hyperparameters found by AutoTuner
    fn tune_tree_phase(
        &self,
        split: &DataSplit,
        train_residuals: &[f32],
        val_residuals: &[f32],
        linear_indices: &[usize],
        history: &mut LttTuningHistory,
    ) -> Result<TreeHyperparams> {
        // Compute tree_indices = all features NOT in linear_indices
        // Linear model uses polynomial/interaction features → tree uses categorical/other features
        let all_indices: std::collections::HashSet<usize> = (0..split.num_features).collect();
        let linear_set: std::collections::HashSet<usize> = linear_indices.iter().copied().collect();
        let mut tree_indices: Vec<usize> = all_indices.difference(&linear_set).copied().collect();
        tree_indices.sort_unstable(); // Keep deterministic order

        // Create BinnedDatasets with residuals as targets (all features initially)
        let (train_binned_full, val_binned_full, feature_info_full) =
            Self::build_probe_binned_datasets(split);

        // Build full datasets first, then filter to tree_indices
        let train_dataset_full = BinnedDataset::new(
            split.train_targets.len(),
            train_binned_full,
            train_residuals.to_vec(),
            feature_info_full.clone(),
        );
        let val_dataset_full = BinnedDataset::new(
            split.val_targets.len(),
            val_binned_full,
            val_residuals.to_vec(),
            feature_info_full,
        );

        // Filter to only tree features (exclude linear model features)
        let train_dataset = train_dataset_full.subset_features(&tree_indices);
        let val_dataset = val_dataset_full.subset_features(&tree_indices);

        // Configure AutoTuner with tree parameter space
        use crate::tuner::autotuner::AutoTuner;
        use crate::tuner::config::{
            EvalStrategy, OptimizationMetric, ParamBounds, ParameterSpace, TaskType, TunableParam,
            TunerConfig, TunerPreset,
        };

        // Use SpacePreset::Regression as base, then extend with Colsample
        // Regression preset includes: MaxDepth, LearningRate, Subsample, Lambda, EntropyWeight
        // We add:
        // - Colsample: [0.5, 1.0] - Important feature subsampling regularization
        //
        // Note: We do NOT tune NumRounds. Instead, we set a high max (e.g., 1000) and use
        // early stopping to find the optimal tree count per config. This is standard practice:
        // early stopping is more adaptive than grid-searching num_rounds.
        //
        // LttTunerConfig's grid values (QUICK_LR_GRID=[0.05, 0.1], etc.) are only used
        // for the initial linear phase grid search, NOT for the tree phase AutoTuner.
        use crate::tuner::config::SpacePreset;
        let space = ParameterSpace::with_preset(SpacePreset::Regression).with_param(
            TunableParam::Colsample,
            ParamBounds::continuous(0.5, 1.0),
            1.0,
        );

        // Map LttTunerPreset to TunerPreset for iteration counts:
        // - Quick → Quick (2 iterations)
        // - Standard → Balanced (5 iterations)
        // - Thorough → Thorough (7 iterations)
        let tuner_preset = match self.config.tree_tuner_iterations {
            1 => TunerPreset::SmokeTest,    // For testing only
            2 => TunerPreset::Quick,        // Quick: 2 iterations
            3..=4 => TunerPreset::Balanced, // Standard: 3-4 iterations → Balanced
            _ => TunerPreset::Thorough,     // Thorough: 5+ iterations
        };

        // Configure tuner with custom space + preset settings
        // Note: When using tune_with_validation(), the validation_ratio is just a dummy value
        // needed for config validation. It won't actually be used since we provide pre-split data.
        let mut tuner_config = TunerConfig::new()
            .with_preset(tuner_preset)
            .with_space(space) // Override preset's default space
            .with_eval_strategy(EvalStrategy::Holdout {
                validation_ratio: 0.2, // Dummy value (won't be used, but must be non-zero for validation)
                folds: 1,              // Single fold
            })
            .with_optimization_metric(OptimizationMetric::ValidationLoss)
            .with_task_type(TaskType::Regression)
            .with_iterations(self.config.tree_tuner_iterations) // Override preset iterations if specified
            .with_parallel(true)
            .with_seed(self.config.seed)
            .with_verbose(false); // LttTuner controls verbosity

        // Configure output directory for AutoTuner logs: {output_dir}/autotuner/
        if let Some(ref output_dir) = self.config.output_dir {
            let autotuner_dir = output_dir.join("autotuner");
            tuner_config = tuner_config.with_output_dir(&autotuner_dir);
        }

        // Create base GBDT config
        let base_config = GBDTConfig::new()
            .with_mse_loss()
            .with_seed(self.config.seed);

        // Run AutoTuner
        let mut tuner = AutoTuner::<GBDTModel>::new(base_config).with_config(tuner_config);
        let (best_gbdt_config, tuner_history) =
            tuner.tune_with_validation(&train_dataset, &val_dataset)?;

        // Extract TreeHyperparams from best GBDTConfig
        let best_params = TreeHyperparams {
            max_depth: best_gbdt_config.max_depth as u32,
            learning_rate: best_gbdt_config.learning_rate,
            num_rounds: best_gbdt_config.num_rounds as u32,
            min_child_weight: best_gbdt_config.min_hessian_leaf,
            subsample: best_gbdt_config.subsample,
            colsample_bytree: best_gbdt_config.colsample,
        };

        // Convert AutoTuner history to LttTuningHistory format
        for trial in tuner_history.trials() {
            // Extract validation RMSE from trial (val_loss is MSE, need sqrt)
            let residual_rmse = trial.val_loss.sqrt(); // MSE -> RMSE

            history.tree_trials.push(TreeTrial {
                max_depth: *trial.params.get("max_depth").unwrap() as u32,
                learning_rate: *trial.params.get("learning_rate").unwrap(),
                num_trees: trial.num_trees as u32,
                residual_rmse,
            });
        }

        Ok(best_params)
    }

    // =========================================================================
    // Phase 3: Joint Refinement
    // =========================================================================

    /// Phase 3: Joint refinement (extrapolation damping tuning)
    ///
    /// Fine-tunes the extrapolation damping parameter which controls how
    /// predictions are dampened toward the target mean for out-of-distribution
    /// safety.
    fn tune_joint_phase(
        &self,
        split: &DataSplit,
        linear_params: &LinearHyperparams,
        history: &mut LttTuningHistory,
    ) -> Result<LinearHyperparams> {
        let mut best_params = *linear_params;
        let mut best_rmse = f32::INFINITY;

        for &damping in &self.config.extrapolation_damping_values {
            let config = LinearConfig::default()
                .with_lambda(linear_params.lambda)?
                .with_l1_ratio(linear_params.l1_ratio)?
                .with_shrinkage_factor(linear_params.shrinkage_factor)?
                .with_extrapolation_damping(damping)?;

            let eval_result = Self::evaluate_linear_config(config, split)?;

            history.joint_trials.push(JointTrial {
                extrapolation_damping: damping,
                combined_rmse: eval_result.rmse,
            });

            if eval_result.rmse < best_rmse {
                best_rmse = eval_result.rmse;
                best_params.extrapolation_damping = damping;
            }
        }

        Ok(best_params)
    }

    /// Compute final RMSE with best parameters (full LTT: shrinkage*linear + tree)
    fn compute_final_rmse(
        &self,
        linear_params: &LinearHyperparams,
        tree_params: &TreeHyperparams,
        split: &DataSplit,
    ) -> Result<f32> {
        // Evaluate full LTT model: pred = shrinkage_factor * linear_pred + tree_pred

        // 1. Train linear model and get predictions
        let linear_config = linear_params.to_config();
        let mut linear_booster = LinearBooster::new(split.num_features, linear_config);

        let _train_preds = linear_booster.fit_direct(
            split.train_features,
            split.num_features,
            split.train_targets,
        )?;

        let linear_val_preds = linear_booster.predict_batch(split.val_features, split.num_features);

        // 2. Compute shrinkage-adjusted residuals for tree training
        let shrinkage = linear_params.shrinkage_factor;
        let train_residuals: Vec<f32> = split
            .train_targets
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let linear_pred =
                    linear_booster.predict_row(split.train_features, split.num_features, i);
                t - shrinkage * linear_pred
            })
            .collect();

        // 3. Build binned dataset for tree training
        let (train_binned, val_binned, feature_info) = Self::build_probe_binned_datasets(split);
        let train_dataset = BinnedDataset::new(
            split.train_targets.len(),
            train_binned,
            train_residuals,
            feature_info.clone(),
        );

        // 4. Train tree model
        let tree_config = GBDTConfig::new()
            .with_max_depth(tree_params.max_depth as usize)
            .with_learning_rate(tree_params.learning_rate)
            .with_num_rounds(tree_params.num_rounds as usize)
            .with_min_hessian_leaf(tree_params.min_child_weight)
            .with_subsample(tree_params.subsample)?
            .with_colsample(tree_params.colsample_bytree)?
            .with_mse_loss();
        let tree_model = GBDTModel::train_binned(&train_dataset, tree_config)?;

        // 5. Get tree predictions on validation
        let val_dataset = BinnedDataset::new(
            split.val_targets.len(),
            val_binned,
            vec![0.0; split.val_targets.len()], // dummy targets
            feature_info,
        );
        let tree_val_preds = tree_model.predict(&val_dataset);

        // 6. Combine predictions: shrinkage*linear + tree
        let combined_preds: Vec<f32> = linear_val_preds
            .iter()
            .zip(tree_val_preds.iter())
            .map(|(&lp, &tp)| shrinkage * lp + tp)
            .collect();

        // 7. Compute RMSE
        let mse: f32 = split
            .val_targets
            .iter()
            .zip(combined_preds.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum::<f32>()
            / split.val_targets.len() as f32;

        Ok(mse.sqrt())
    }
}

impl Default for LttTuner {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_hyperparams_default() {
        let params = LinearHyperparams::default();
        assert_eq!(params.lambda, 1.0);
        assert_eq!(params.l1_ratio, 0.0); // Ridge
        assert_eq!(params.shrinkage_factor, ltt_defaults::DEFAULT_LTT_SHRINKAGE);
        assert_eq!(params.extrapolation_damping, 0.0);
    }

    #[test]
    fn test_tree_hyperparams_default() {
        let params = TreeHyperparams::default();
        assert_eq!(params.max_depth, 6);
        assert_eq!(params.learning_rate, 0.1);
        assert_eq!(params.num_rounds, 500);
    }

    #[test]
    fn test_ltt_tuner_config_presets() {
        let quick = LttTunerConfig::default().with_preset(LttTunerPreset::Quick);
        let thorough = LttTunerConfig::default().with_preset(LttTunerPreset::Thorough);

        assert!(quick.lambda_values.len() < thorough.lambda_values.len());
        assert!(quick.max_depth_values.len() < thorough.max_depth_values.len());
    }

    #[test]
    fn test_ltt_tuner_estimated_trials() {
        let config = LttTunerConfig::default();
        let tuner = LttTuner::new(config.clone());

        let linear_trials = config.lambda_values.len() * config.l1_ratio_values.len();
        let tree_trials = config.max_depth_values.len()
            * config.learning_rate_values.len()
            * config.num_rounds_values.len();
        let joint_trials = config.extrapolation_damping_values.len();

        assert_eq!(
            tuner.estimated_trials(),
            linear_trials + tree_trials + joint_trials
        );
    }

    #[test]
    fn test_config_validation_empty_grids() {
        let config = LttTunerConfig {
            lambda_values: vec![],
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("lambda_values"));
    }

    #[test]
    fn test_config_validation_bad_val_ratio() {
        let config = LttTunerConfig {
            val_ratio: 1.5, // Invalid
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("val_ratio"));
    }

    #[test]
    fn test_input_validation_empty_targets() {
        let tuner = LttTuner::with_defaults();
        let features = vec![1.0, 2.0, 3.0];
        let targets: Vec<f32> = vec![];
        let linear_indices: Vec<usize> = vec![0];

        let result = tuner.tune(&features, 1, &targets, &linear_indices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_input_validation_zero_features() {
        let tuner = LttTuner::with_defaults();
        let features = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        let linear_indices: Vec<usize> = vec![];

        let result = tuner.tune(&features, 0, &targets, &linear_indices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_features"));
    }

    #[test]
    fn test_input_validation_dimension_mismatch() {
        let tuner = LttTuner::with_defaults();
        let features = vec![1.0, 2.0, 3.0]; // 3 elements
        let targets = vec![1.0, 2.0]; // 2 rows, but features suggest 3×1
        let linear_indices: Vec<usize> = vec![0];

        let result = tuner.tune(&features, 1, &targets, &linear_indices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    #[test]
    fn test_ltt_tuner_tune() {
        // Simple linear data: y = 2*x + 1 + noise
        let num_features = 1;
        let num_rows = 100;

        let mut features = Vec::with_capacity(num_rows * num_features);
        let mut targets = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let x = (i as f32) / 10.0;
            features.push(x);
            targets.push(2.0 * x + 1.0 + (i as f32 % 3.0) * 0.1); // y = 2x + 1 + small noise
        }

        let config = LttTunerConfig::default().with_preset(LttTunerPreset::Quick);
        let tuner = LttTuner::new(config);
        let linear_indices: Vec<usize> = (0..num_features).collect();

        let result = tuner
            .tune(&features, num_features, &targets, &linear_indices)
            .expect("LTT tuner should successfully fit linear data");

        // Should find reasonable hyperparameters
        assert!(result.linear_r2 > 0.5, "R² should be > 0.5 for linear data");
        assert!(result.final_rmse < 5.0, "RMSE should be reasonable");
        assert!(!result.history.linear_trials.is_empty());
        assert!(!result.history.tree_trials.is_empty());
    }

    #[test]
    fn test_linear_params_to_config() {
        let params = LinearHyperparams {
            lambda: 0.5,
            l1_ratio: 0.3,
            shrinkage_factor: 0.8,
            extrapolation_damping: 0.1,
        };

        let config = params.to_config();

        assert!((config.lambda - 0.5).abs() < 1e-6);
        assert!((config.l1_ratio - 0.3).abs() < 1e-6);
        assert!((config.shrinkage_factor - 0.8).abs() < 1e-6);
        assert!((config.extrapolation_damping - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_shrinkage_factor_selection_strong_linear() {
        let mut config = LttTunerConfig::default();
        let shrinkage_grid = vec![0.3, 0.7];
        config.shrinkage_factor_values = shrinkage_grid.clone();
        let tuner = LttTuner::new(config);

        let train_features = vec![0.0, 1.0, 2.0, 3.0];
        let val_features = vec![4.0, 5.0];
        let train_targets = vec![0.0, 1.0, 2.0, 3.0];
        let val_targets = vec![4.0, 5.0];
        let split = DataSplit::new(
            &train_features,
            &train_targets,
            &val_features,
            &val_targets,
            1,
        );
        let train_preds = vec![0.0, 1.0, 2.0, 3.0];
        let val_preds = vec![4.0, 5.0];

        let shrinkage = tuner.select_shrinkage_factor(&train_preds, &val_preds, &split);

        assert!(
            shrinkage_grid.contains(&shrinkage),
            "Shrinkage should be selected from configured grid"
        );
    }

    #[test]
    fn test_shrinkage_factor_selection_weak_linear() {
        let mut config = LttTunerConfig::default();
        let shrinkage_grid = vec![0.3, 0.7];
        config.shrinkage_factor_values = shrinkage_grid.clone();
        let tuner = LttTuner::new(config);

        let train_features = vec![0.0, 1.0, 2.0, 3.0];
        let val_features = vec![4.0, 5.0];
        let train_targets = vec![0.0, 1.0, 2.0, 3.0];
        let val_targets = vec![4.0, 5.0];
        let split = DataSplit::new(
            &train_features,
            &train_targets,
            &val_features,
            &val_targets,
            1,
        );
        let train_preds = vec![3.0, 2.0, 1.0, 0.0];
        let val_preds = vec![1.0, 0.0];

        let shrinkage = tuner.select_shrinkage_factor(&train_preds, &val_preds, &split);

        assert!(
            shrinkage_grid.contains(&shrinkage),
            "Shrinkage should be selected from configured grid"
        );
    }

    #[test]
    fn test_constants_are_reasonable() {
        // Verify our named constants have sensible values.
        // Bind to runtime locals so these are genuine runtime checks of the
        // configured constant values, not compile-time constant assertions.
        let strong_r2 = std::hint::black_box(ltt_defaults::STRONG_LINEAR_R2);
        let weak_r2 = std::hint::black_box(ltt_defaults::WEAK_LINEAR_R2);
        let high_shrinkage_min = std::hint::black_box(ltt_defaults::HIGH_SHRINKAGE_MIN);
        let low_shrinkage_max = std::hint::black_box(ltt_defaults::LOW_SHRINKAGE_MAX);
        let default_shrinkage = std::hint::black_box(ltt_defaults::DEFAULT_LTT_SHRINKAGE);
        let high_variance_threshold = std::hint::black_box(ltt_defaults::HIGH_VARIANCE_THRESHOLD);
        let max_depth_threshold = std::hint::black_box(ltt_defaults::MAX_DEPTH_THRESHOLD);
        let min_lr_threshold = std::hint::black_box(ltt_defaults::MIN_LR_THRESHOLD);

        assert!(strong_r2 > weak_r2);
        assert!(high_shrinkage_min > 0.0 && high_shrinkage_min < 1.0);
        assert!(low_shrinkage_max > 0.0 && low_shrinkage_max < 1.0);
        assert!(default_shrinkage > 0.0 && default_shrinkage <= 1.0);
        assert!(high_variance_threshold > 0.0);
        assert!(max_depth_threshold > 0);
        assert!(min_lr_threshold > 0.0);
    }
}
