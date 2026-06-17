//! AutoTuner configuration types
//!
//! This module defines the configuration structures for the hyperparameter tuner:
//! - `ParamBounds`: Bounds and scaling for a single parameter
//! - `ParamDef`: A parameter definition with name, bounds, and center
//! - `ParameterSpace`: Collection of parameters to tune
//! - `EvalStrategy`: How to evaluate candidates (holdout vs K-fold)
//! - `GridStrategy`: How to generate candidate points
//! - `TuningMode`: Optimistic (fast) vs Realistic (accurate) evaluation
//! - `ModelFormat`: Output format for saving models
//! - `TunerConfig`: Main tuner configuration

use std::collections::HashMap;
use std::path::PathBuf;

use crate::defaults::tuning::{seeds as seeds_defaults, tuner as tuner_defaults};
use crate::TreeBoostError;

/// Type-safe tunable parameter names
///
/// This enum provides compile-time type safety for hyperparameter names,
/// eliminating string typos and runtime validation errors.
///
/// # Parameter Categories
///
/// ## GBDTModel Parameters
/// - **Tree structure**: `MaxDepth`
/// - **Learning**: `LearningRate`, `NumRounds`
/// - **Sampling**: `Subsample`, `Colsample`, `GossTopRate`, `GossOtherRate`
/// - **Regularization**: `Lambda`, `EntropyWeight`, `MinSamplesLeaf`, `MinHessianLeaf`, `MinGain`
///
/// ## UniversalModel Parameters
/// - **Mode selection**: `Mode` (categorical: PureTree, LinearThenTree, RandomForest)
/// - **Common**: `NumRounds`, `LearningRate`, `Subsample`, `ValidationRatio`, `EarlyStoppingRounds`
/// - **Linear**: `LinearRounds`, `LinearLambda`, `LinearMaxIter`
/// - **Tree**: `TreeMaxDepth`, `TreeMaxLeaves`, `TreeLambda`
///
/// # Example
///
/// ```
/// use treeboost::tuner::{TunableParam, ParamBounds, ParamDef};
///
/// // Type-safe parameter definition
/// let param = ParamDef::new(
///     TunableParam::MaxDepth,
///     ParamBounds::discrete(2, 12),
///     6.0,
/// );
///
/// // Parsing from string
/// let parsed = TunableParam::parse("learning_rate")?;
/// assert_eq!(parsed, TunableParam::LearningRate);
///
/// // Converting to string
/// assert_eq!(TunableParam::Lambda.to_string(), "lambda");
/// # Ok::<(), String>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TunableParam {
    // === GBDTModel Parameters ===
    /// Tree maximum depth (discrete: 2-12)
    MaxDepth,
    /// Learning rate / shrinkage (continuous log: 0.01-0.5)
    LearningRate,
    /// Row subsampling ratio (continuous: 0.5-1.0)
    Subsample,
    /// Feature subsampling ratio (continuous: 0.5-1.0)
    Colsample,
    /// L2 regularization strength (continuous: 0.0-10.0)
    Lambda,
    /// Shannon entropy regularization weight (continuous: 0.0-0.5)
    EntropyWeight,
    /// Minimum samples per leaf (discrete: 1-100)
    MinSamplesLeaf,
    /// Minimum hessian sum per leaf (continuous: 0.0-10.0)
    MinHessianLeaf,
    /// Minimum split gain threshold (continuous: 0.0-1.0)
    MinGain,
    /// Number of boosting rounds (discrete: 50-500)
    NumRounds,
    /// GOSS top gradient sample ratio (continuous: 0.1-0.4)
    GossTopRate,
    /// GOSS other gradient sample ratio (continuous: 0.05-0.2)
    GossOtherRate,

    // === UniversalModel Parameters ===
    /// Boosting mode (categorical: PureTree, LinearThenTree, RandomForest)
    Mode,
    /// Validation set ratio for early stopping (continuous: 0.0-0.5)
    ValidationRatio,
    /// Early stopping patience rounds (discrete: 5-50)
    EarlyStoppingRounds,
    /// Number of linear boosting rounds (discrete: 1-30)
    LinearRounds,
    /// Tree max depth in UniversalModel (discrete: 2-12)
    TreeMaxDepth,
    /// Tree max leaves in UniversalModel (discrete: 8-256)
    TreeMaxLeaves,
    /// Tree L2 regularization in UniversalModel (continuous: 0.0-10.0)
    TreeLambda,
    /// Linear L2 regularization (continuous log: 0.01-10.0)
    LinearLambda,
    /// Linear coordinate descent max iterations (discrete: 10-1000)
    LinearMaxIter,
}

impl TunableParam {
    /// Parse parameter name from string
    ///
    /// # Errors
    ///
    /// Returns error if the parameter name is not recognized.
    ///
    /// # Example
    ///
    /// ```
    /// use treeboost::tuner::TunableParam;
    ///
    /// let param = TunableParam::parse("max_depth")?;
    /// assert_eq!(param, TunableParam::MaxDepth);
    ///
    /// let err = TunableParam::parse("invalid_param");
    /// assert!(err.is_err());
    /// # Ok::<(), String>(())
    /// ```
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            // GBDTModel parameters
            "max_depth" => Ok(Self::MaxDepth),
            "learning_rate" => Ok(Self::LearningRate),
            "subsample" => Ok(Self::Subsample),
            "colsample" => Ok(Self::Colsample),
            "lambda" => Ok(Self::Lambda),
            "entropy_weight" => Ok(Self::EntropyWeight),
            "min_samples_leaf" => Ok(Self::MinSamplesLeaf),
            "min_hessian_leaf" => Ok(Self::MinHessianLeaf),
            "min_gain" => Ok(Self::MinGain),
            "num_rounds" => Ok(Self::NumRounds),
            "goss_top_rate" => Ok(Self::GossTopRate),
            "goss_other_rate" => Ok(Self::GossOtherRate),
            // UniversalModel parameters
            "mode" => Ok(Self::Mode),
            "validation_ratio" => Ok(Self::ValidationRatio),
            "early_stopping_rounds" => Ok(Self::EarlyStoppingRounds),
            "linear_rounds" => Ok(Self::LinearRounds),
            "tree_max_depth" => Ok(Self::TreeMaxDepth),
            "tree_max_leaves" => Ok(Self::TreeMaxLeaves),
            "tree_lambda" => Ok(Self::TreeLambda),
            "linear_lambda" => Ok(Self::LinearLambda),
            "linear_max_iter" => Ok(Self::LinearMaxIter),
            _ => Err(format!(
                "Unknown parameter '{}'. Valid parameters: {:?}",
                s,
                Self::all_names()
            )),
        }
    }

    /// Convert parameter to its string representation
    ///
    /// # Example
    ///
    /// ```
    /// use treeboost::tuner::TunableParam;
    ///
    /// assert_eq!(TunableParam::MaxDepth.to_string(), "max_depth");
    /// assert_eq!(TunableParam::LearningRate.to_string(), "learning_rate");
    /// ```
    pub fn to_string(&self) -> &'static str {
        match self {
            // GBDTModel parameters
            Self::MaxDepth => "max_depth",
            Self::LearningRate => "learning_rate",
            Self::Subsample => "subsample",
            Self::Colsample => "colsample",
            Self::Lambda => "lambda",
            Self::EntropyWeight => "entropy_weight",
            Self::MinSamplesLeaf => "min_samples_leaf",
            Self::MinHessianLeaf => "min_hessian_leaf",
            Self::MinGain => "min_gain",
            Self::NumRounds => "num_rounds",
            Self::GossTopRate => "goss_top_rate",
            Self::GossOtherRate => "goss_other_rate",
            // UniversalModel parameters
            Self::Mode => "mode",
            Self::ValidationRatio => "validation_ratio",
            Self::EarlyStoppingRounds => "early_stopping_rounds",
            Self::LinearRounds => "linear_rounds",
            Self::TreeMaxDepth => "tree_max_depth",
            Self::TreeMaxLeaves => "tree_max_leaves",
            Self::TreeLambda => "tree_lambda",
            Self::LinearLambda => "linear_lambda",
            Self::LinearMaxIter => "linear_max_iter",
        }
    }

    /// Get all valid parameter names
    pub fn all_names() -> &'static [&'static str] {
        &[
            // GBDTModel
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample",
            "lambda",
            "entropy_weight",
            "min_samples_leaf",
            "min_hessian_leaf",
            "min_gain",
            "num_rounds",
            "goss_top_rate",
            "goss_other_rate",
            // UniversalModel
            "mode",
            "validation_ratio",
            "early_stopping_rounds",
            "linear_rounds",
            "tree_max_depth",
            "tree_max_leaves",
            "tree_lambda",
            "linear_lambda",
            "linear_max_iter",
        ]
    }

    /// Get GBDTModel-specific parameter names
    pub fn gbdt_params() -> &'static [Self] {
        &[
            Self::MaxDepth,
            Self::LearningRate,
            Self::Subsample,
            Self::Colsample,
            Self::Lambda,
            Self::EntropyWeight,
            Self::MinSamplesLeaf,
            Self::MinHessianLeaf,
            Self::MinGain,
            Self::NumRounds,
            Self::GossTopRate,
            Self::GossOtherRate,
        ]
    }

    /// Get UniversalModel-specific parameter names
    pub fn universal_params() -> &'static [Self] {
        &[
            Self::Mode,
            Self::NumRounds,
            Self::LearningRate,
            Self::Subsample,
            Self::ValidationRatio,
            Self::EarlyStoppingRounds,
            Self::LinearRounds,
            Self::TreeMaxDepth,
            Self::TreeMaxLeaves,
            Self::TreeLambda,
            Self::LinearLambda,
            Self::LinearMaxIter,
        ]
    }
}

/// Model serialization format
///
/// Determines how the best model is saved after tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Rkyv format - zero-copy deserialization, fastest loading
    ///
    /// Uses rkyv for blazing fast model loading with memory mapping.
    /// Best for production inference where load time matters.
    /// File extension: `.rkyv`
    Rkyv,
}

impl ModelFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Rkyv => "rkyv",
        }
    }

    /// Get the filename for the best model in this format
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Rkyv => "best_model.rkyv",
        }
    }
}

/// Bounds and scaling for a tunable parameter
#[derive(Debug, Clone, PartialEq)]
pub enum ParamBounds {
    /// Continuous parameter with min, max, and optional log scaling
    ///
    /// When `log_scale` is true, values are sampled uniformly in log space.
    /// This is useful for parameters like learning_rate where the difference
    /// between 0.01 and 0.1 is more significant than between 0.1 and 0.2.
    Continuous { min: f32, max: f32, log_scale: bool },
    /// Discrete integer parameter with min, max, and step size
    ///
    /// Values are sampled at intervals of `step` between min and max.
    /// For example, `Discrete { min: 2, max: 12, step: 2 }` produces [2, 4, 6, 8, 10, 12].
    Discrete { min: usize, max: usize, step: usize },
    /// Categorical parameter with a fixed set of string values
    ///
    /// Used for parameters like boosting_mode where values are discrete choices.
    /// The center is stored as an index into the values array.
    Categorical { values: Vec<String> },
}

impl ParamBounds {
    /// Create continuous bounds without log scaling
    pub fn continuous(min: f32, max: f32) -> Self {
        Self::Continuous {
            min,
            max,
            log_scale: false,
        }
    }

    /// Create continuous bounds with log scaling
    pub fn log_continuous(min: f32, max: f32) -> Self {
        Self::Continuous {
            min,
            max,
            log_scale: true,
        }
    }

    /// Create discrete bounds with step size 1
    pub fn discrete(min: usize, max: usize) -> Self {
        Self::Discrete { min, max, step: 1 }
    }

    /// Create discrete bounds with custom step size
    pub fn discrete_step(min: usize, max: usize, step: usize) -> Self {
        Self::Discrete { min, max, step }
    }

    /// Create categorical bounds from string values
    pub fn categorical(values: Vec<String>) -> Self {
        Self::Categorical { values }
    }

    /// Create categorical bounds from str slice
    pub fn categorical_from_strs(values: &[&str]) -> Self {
        Self::Categorical {
            values: values.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Clamp a value to be within bounds
    ///
    /// For categorical parameters, clamps to valid index range [0, len-1]
    pub fn clamp(&self, value: f32) -> f32 {
        match self {
            Self::Continuous { min, max, .. } => value.clamp(*min, *max),
            Self::Discrete { min, max, step } => {
                let clamped = (value as usize).clamp(*min, *max);
                // Round to nearest step
                let steps = (clamped - min) / step;
                (min + steps * step) as f32
            }
            Self::Categorical { values } => {
                // For categorical, value is an index
                let max_idx = values.len().saturating_sub(1);
                (value as usize).clamp(0, max_idx) as f32
            }
        }
    }

    /// Check if a value is within bounds
    pub fn contains(&self, value: f32) -> bool {
        match self {
            Self::Continuous { min, max, .. } => value >= *min && value <= *max,
            Self::Discrete { min, max, .. } => {
                let v = value as usize;
                v >= *min && v <= *max
            }
            Self::Categorical { values } => {
                let idx = value as usize;
                idx < values.len()
            }
        }
    }

    /// Get the minimum value
    pub fn min_value(&self) -> f32 {
        match self {
            Self::Continuous { min, .. } => *min,
            Self::Discrete { min, .. } => *min as f32,
            Self::Categorical { .. } => 0.0,
        }
    }

    /// Get the maximum value
    pub fn max_value(&self) -> f32 {
        match self {
            Self::Continuous { max, .. } => *max,
            Self::Discrete { max, .. } => *max as f32,
            Self::Categorical { values } => values.len().saturating_sub(1) as f32,
        }
    }

    /// Check if this uses log scaling
    pub fn is_log_scale(&self) -> bool {
        matches!(
            self,
            Self::Continuous {
                log_scale: true,
                ..
            }
        )
    }

    /// Check if this is a categorical parameter
    pub fn is_categorical(&self) -> bool {
        matches!(self, Self::Categorical { .. })
    }

    /// Get categorical values (if categorical)
    pub fn categorical_values(&self) -> Option<&[String]> {
        match self {
            Self::Categorical { values } => Some(values),
            _ => None,
        }
    }

    /// Get categorical value by index
    pub fn get_categorical_value(&self, index: usize) -> Option<&str> {
        match self {
            Self::Categorical { values } => values.get(index).map(|s| s.as_str()),
            _ => None,
        }
    }

    /// Get index for categorical value
    pub fn categorical_index(&self, value: &str) -> Option<usize> {
        match self {
            Self::Categorical { values } => values.iter().position(|v| v == value),
            _ => None,
        }
    }
}

/// Definition of a single tunable parameter
#[derive(Debug, Clone)]
pub struct ParamDef {
    /// Parameter name (type-safe enum)
    pub name: TunableParam,
    /// Bounds and scaling
    pub bounds: ParamBounds,
    /// Current center point for grid generation
    pub center: f32,
}

impl ParamDef {
    /// Create a new parameter definition
    pub fn new(name: TunableParam, bounds: ParamBounds, center: f32) -> Self {
        let center = bounds.clamp(center);
        Self {
            name,
            bounds,
            center,
        }
    }

    /// Create a parameter definition from string name (for backward compatibility)
    ///
    /// # Errors
    ///
    /// Returns error if the parameter name is not recognized.
    pub fn from_str(name: &str, bounds: ParamBounds, center: f32) -> Result<Self, String> {
        let param = TunableParam::parse(name)?;
        Ok(Self::new(param, bounds, center))
    }

    /// Get the parameter name as a string
    pub fn name_str(&self) -> &'static str {
        self.name.to_string()
    }

    /// Update the center point (clamped to bounds)
    pub fn set_center(&mut self, center: f32) {
        self.center = self.bounds.clamp(center);
    }
}

/// Search space: collection of parameters to tune
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    params: Vec<ParamDef>,
}

/// Presets for parameter search spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpacePreset {
    /// max_depth, learning_rate only.
    Minimal,
    /// Standard regression parameters.
    Regression,
    /// Classification-optimized parameters.
    Classification,
    /// All parameters including GOSS and colsample.
    Exhaustive,
    /// UniversalModel mode selection + common params.
    Universal,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::with_preset(SpacePreset::Regression)
    }
}

impl ParameterSpace {
    /// Create an empty parameter space
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Create a preset search space.
    pub fn with_preset(preset: SpacePreset) -> Self {
        match preset {
            SpacePreset::Minimal => Self::minimal_space(),
            SpacePreset::Regression => Self::regression_space(),
            SpacePreset::Classification => Self::classification_space(),
            SpacePreset::Exhaustive => Self::exhaustive(),
            SpacePreset::Universal => Self::universal_space(),
        }
    }

    /// Create an exhaustive search space (adds GOSS and colsample).
    pub fn exhaustive() -> Self {
        use TunableParam::*;
        Self {
            params: vec![
                ParamDef::new(MaxDepth, ParamBounds::discrete(2, 12), 6.0),
                ParamDef::new(LearningRate, ParamBounds::log_continuous(0.01, 0.5), 0.1),
                ParamDef::new(Subsample, ParamBounds::continuous(0.5, 1.0), 0.8),
                ParamDef::new(Colsample, ParamBounds::continuous(0.5, 1.0), 1.0),
                ParamDef::new(Lambda, ParamBounds::continuous(0.0, 10.0), 1.0),
                ParamDef::new(EntropyWeight, ParamBounds::continuous(0.0, 0.5), 0.0),
                ParamDef::new(GossTopRate, ParamBounds::continuous(0.1, 0.4), 0.2),
                ParamDef::new(GossOtherRate, ParamBounds::continuous(0.05, 0.2), 0.1),
            ],
        }
    }

    /// Create search space for UniversalModel focusing on mode selection only
    ///
    /// Use this when you want to find the best mode without tuning other parameters.
    pub fn universal_mode_only() -> Self {
        Self {
            params: vec![ParamDef::new(
                TunableParam::Mode,
                ParamBounds::categorical_from_strs(&["PureTree", "LinearThenTree", "RandomForest"]),
                0.0,
            )],
        }
    }

    fn regression_space() -> Self {
        use TunableParam::*;
        Self {
            params: vec![
                ParamDef::new(MaxDepth, ParamBounds::discrete(2, 12), 6.0),
                ParamDef::new(LearningRate, ParamBounds::log_continuous(0.01, 0.5), 0.1),
                ParamDef::new(Subsample, ParamBounds::continuous(0.5, 1.0), 0.8),
                ParamDef::new(Lambda, ParamBounds::continuous(0.0, 10.0), 1.0),
                ParamDef::new(EntropyWeight, ParamBounds::continuous(0.0, 0.5), 0.0),
            ],
        }
    }

    fn classification_space() -> Self {
        use TunableParam::*;
        Self {
            params: vec![
                ParamDef::new(MaxDepth, ParamBounds::discrete(2, 10), 5.0),
                ParamDef::new(LearningRate, ParamBounds::log_continuous(0.01, 0.3), 0.1),
                ParamDef::new(Subsample, ParamBounds::continuous(0.6, 1.0), 0.8),
                ParamDef::new(Lambda, ParamBounds::continuous(0.0, 5.0), 1.0),
                ParamDef::new(EntropyWeight, ParamBounds::continuous(0.0, 0.3), 0.0),
            ],
        }
    }

    fn minimal_space() -> Self {
        use TunableParam::*;
        Self {
            params: vec![
                ParamDef::new(MaxDepth, ParamBounds::discrete(3, 10), 6.0),
                ParamDef::new(LearningRate, ParamBounds::log_continuous(0.01, 0.3), 0.1),
            ],
        }
    }

    fn universal_space() -> Self {
        use TunableParam::*;
        Self {
            params: vec![
                // Mode selection (categorical)
                ParamDef::new(
                    Mode,
                    ParamBounds::categorical_from_strs(&[
                        "PureTree",
                        "LinearThenTree",
                        "RandomForest",
                    ]),
                    0.0, // PureTree is default (index 0)
                ),
                // Common parameters
                ParamDef::new(NumRounds, ParamBounds::discrete(50, 200), 100.0),
                ParamDef::new(LearningRate, ParamBounds::log_continuous(0.01, 0.3), 0.1),
                ParamDef::new(Subsample, ParamBounds::continuous(0.6, 1.0), 0.8),
                // Tree parameters
                ParamDef::new(TreeMaxDepth, ParamBounds::discrete(3, 10), 6.0),
                ParamDef::new(TreeLambda, ParamBounds::continuous(0.0, 10.0), 1.0),
            ],
        }
    }

    /// Create search space for UniversalModel with LinearThenTree focus
    ///
    /// Includes parameters relevant to LinearThenTree mode.
    pub fn universal_linear_then_tree() -> Self {
        use TunableParam::*;
        Self {
            params: vec![
                ParamDef::new(
                    NumRounds,
                    ParamBounds::discrete(30, 150),
                    50.0, // Fewer rounds needed with linear component
                ),
                ParamDef::new(LearningRate, ParamBounds::log_continuous(0.01, 0.3), 0.1),
                ParamDef::new(LinearRounds, ParamBounds::discrete(5, 30), 10.0),
                ParamDef::new(LinearLambda, ParamBounds::log_continuous(0.01, 10.0), 1.0),
                ParamDef::new(TreeMaxDepth, ParamBounds::discrete(3, 8), 5.0),
            ],
        }
    }

    /// Add or update a parameter in the search space
    ///
    /// If a parameter with the same name exists, it will be replaced.
    pub fn with_param(mut self, name: TunableParam, bounds: ParamBounds, center: f32) -> Self {
        // Remove existing param with same name
        self.params.retain(|p| p.name != name);
        self.params.push(ParamDef::new(name, bounds, center));
        self
    }

    /// Remove a parameter from tuning
    ///
    /// The parameter will use its default value from the base config.
    pub fn without_param(mut self, name: TunableParam) -> Self {
        self.params.retain(|p| p.name != name);
        self
    }

    /// Get a parameter by name
    pub fn get(&self, name: TunableParam) -> Option<&ParamDef> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Get a mutable parameter by name
    pub fn get_mut(&mut self, name: TunableParam) -> Option<&mut ParamDef> {
        self.params.iter_mut().find(|p| p.name == name)
    }

    /// Get all parameters
    pub fn params(&self) -> &[ParamDef] {
        &self.params
    }

    /// Get mutable access to all parameters
    pub fn params_mut(&mut self) -> &mut [ParamDef] {
        &mut self.params
    }

    /// Number of parameters in the search space
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Check if the search space is empty
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get parameter names in consistent order
    pub fn param_names(&self) -> Vec<&'static str> {
        self.params.iter().map(|p| p.name.to_string()).collect()
    }

    /// Get current centers as a HashMap
    pub fn centers(&self) -> HashMap<&'static str, f32> {
        self.params
            .iter()
            .map(|p| (p.name.to_string(), p.center))
            .collect()
    }

    /// Update centers from a HashMap (accepts string keys for backward compatibility)
    pub fn set_centers(&mut self, centers: &HashMap<String, f32>) {
        for param in &mut self.params {
            let name_str = param.name.to_string();
            if let Some(&center) = centers.get(name_str) {
                param.set_center(center);
            }
        }
    }

    /// Constrain parameter bounds based on historical results
    ///
    /// Reads iteration_*.csv files from a previous run directory and constrains
    /// the search space to the min/max values of the top performing trials.
    ///
    /// # Arguments
    /// * `history_dir` - Path to a previous run directory (e.g., "results/run_20251228_143022")
    /// * `top_percentile` - Fraction of top trials to use (e.g., 0.2 = top 20%)
    /// * `metric_column` - Column name to sort by ("val_metric", "roc_auc", or "f1_score")
    /// * `higher_is_better` - Whether higher values are better for the metric
    ///
    /// # Example
    /// ```ignore
    /// let space = ParameterSpace::with_preset(SpacePreset::Classification)
    ///     .constrain_from_history(
    ///         "results/run_20251228_143022",
    ///         0.2,  // Top 20%
    ///         "roc_auc",
    ///         true, // Higher is better
    ///     )?;
    /// ```
    pub fn constrain_from_history<P: AsRef<std::path::Path>>(
        mut self,
        history_dir: P,
        top_percentile: f32,
        metric_column: &str,
        higher_is_better: bool,
    ) -> crate::Result<Self> {
        use std::fs;

        let dir = history_dir.as_ref();
        if !dir.exists() {
            return Err(TreeBoostError::Data(format!(
                "History directory not found: {}",
                dir.display()
            )));
        }

        // Find all iteration_*.csv files
        let mut csv_files: Vec<_> = fs::read_dir(dir)
            .map_err(|e| TreeBoostError::Data(format!("Failed to read directory: {}", e)))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                let name = path.file_name()?.to_str()?;
                if name.starts_with("iteration_") && name.ends_with(".csv") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if csv_files.is_empty() {
            return Err(TreeBoostError::Data(
                "No iteration_*.csv files found in history directory".into(),
            ));
        }

        csv_files.sort();

        // Parse all trials from CSV files
        let mut all_trials: Vec<HashMap<String, f32>> = Vec::new();

        for csv_path in &csv_files {
            let mut reader = csv::Reader::from_path(csv_path)
                .map_err(|e| TreeBoostError::Data(format!("Failed to open CSV: {}", e)))?;

            let headers: Vec<String> = reader
                .headers()
                .map_err(|e| TreeBoostError::Data(format!("Failed to read headers: {}", e)))?
                .iter()
                .map(|s| s.to_string())
                .collect();

            for result in reader.records() {
                let record = result
                    .map_err(|e| TreeBoostError::Data(format!("Failed to read record: {}", e)))?;
                let mut trial: HashMap<String, f32> = HashMap::new();

                for (i, value) in record.iter().enumerate() {
                    if let Some(header) = headers.get(i) {
                        if let Ok(v) = value.parse::<f32>() {
                            trial.insert(header.clone(), v);
                        }
                    }
                }

                if trial.contains_key(metric_column) {
                    all_trials.push(trial);
                }
            }
        }

        if all_trials.is_empty() {
            return Err(TreeBoostError::Data(
                "No valid trials found in CSV files".into(),
            ));
        }

        // Sort by metric (higher or lower is better)
        all_trials.sort_by(|a, b| {
            let a_val = a.get(metric_column).copied().unwrap_or(f32::NAN);
            let b_val = b.get(metric_column).copied().unwrap_or(f32::NAN);
            if higher_is_better {
                b_val
                    .partial_cmp(&a_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a_val
                    .partial_cmp(&b_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Take top N%
        let n_top = ((all_trials.len() as f32) * top_percentile).ceil() as usize;
        let n_top = n_top.max(1); // At least 1 trial
        let top_trials = &all_trials[..n_top.min(all_trials.len())];

        // For each parameter, find min/max in top trials and constrain bounds
        for param in &mut self.params {
            let param_name_str = param.name.to_string();
            let values: Vec<f32> = top_trials
                .iter()
                .filter_map(|t| t.get(param_name_str).copied())
                .filter(|v| !v.is_nan())
                .collect();

            if values.is_empty() {
                continue;
            }

            let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Update bounds based on type
            match &mut param.bounds {
                ParamBounds::Continuous { min, max, .. } => {
                    // Add a small margin (10%) to avoid getting stuck at edges
                    let range = max_val - min_val;
                    let margin = range * 0.1;
                    *min = (min_val - margin).max(*min);
                    *max = (max_val + margin).min(*max);
                }
                ParamBounds::Discrete { min, max, step } => {
                    let new_min = (min_val as usize).max(*min);
                    let new_max = (max_val as usize).min(*max);
                    // Round to step boundaries
                    *min = ((new_min - *min) / *step) * *step + *min;
                    *max = ((new_max - *min) / *step) * *step + *min;
                }
                ParamBounds::Categorical { .. } => {
                    // Categorical bounds don't narrow - all categories remain valid
                }
            }

            // Update center to be in the middle of new bounds
            let mid = match &param.bounds {
                ParamBounds::Continuous { min, max, .. } => (*min + *max) / 2.0,
                ParamBounds::Discrete { min, max, .. } => ((*min + *max) / 2) as f32,
                ParamBounds::Categorical { values } => {
                    // For categorical, center stays at index 0 (first category)
                    // unless the current center is still valid
                    let current = param.center as usize;
                    if current < values.len() {
                        current as f32
                    } else {
                        0.0
                    }
                }
            };
            param.set_center(mid);
        }

        Ok(self)
    }
}

/// Strategy for evaluating candidate configurations
///
/// # Choosing a Strategy
///
/// - **Holdout**: Train/validation split. Use `folds=1` for simple holdout (fast),
///   or `folds=5` for 5-fold CV (robust, 5x slower but better generalization).
///
/// - **Conformal**: O(1) evaluation using conformal interval width as the metric.
///   Instead of optimizing for lowest error, optimizes for tightest prediction intervals.
///   Models that overfit will have wide intervals to maintain coverage guarantee.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvalStrategy {
    /// Train/validation split with optional k-fold cross-validation
    ///
    /// - `folds=1`: Simple holdout (fast, may have high variance)
    /// - `folds=5`: 5-fold CV (robust, 5x slower)
    Holdout {
        /// Fraction of data to use for validation (e.g., 0.2 = 20%)
        validation_ratio: f32,
        /// Number of folds (1 = simple holdout, 5 = 5-fold CV)
        folds: usize,
    },
    /// Conformal prediction-based evaluation with optional k-fold
    ///
    /// **O(1) evaluation** - Uses the conformal quantile `q` as the optimization metric.
    /// Lower `q` means tighter prediction intervals, indicating a more confident model.
    ///
    /// - `folds=1`: Simple conformal (fastest)
    /// - `folds=5`: Conformal 5-fold (robust, still fast per-fold)
    ///
    /// Benefits:
    /// - Fast evaluation (no prediction loop needed, just read `q`)
    /// - Penalizes overfitting naturally (overfit models need wide intervals for coverage)
    /// - Optimizes for "certainty" rather than just "accuracy"
    Conformal {
        /// Fraction of data for calibration set (e.g., 0.2 = 20%)
        calibration_ratio: f32,
        /// Coverage quantile (e.g., 0.9 = 90% coverage guarantee)
        quantile: f32,
        /// Number of folds (1 = simple conformal, 5 = conformal 5-fold)
        folds: usize,
    },
}

impl Default for EvalStrategy {
    fn default() -> Self {
        Self::Holdout {
            validation_ratio: 0.2,
            folds: 1,
        }
    }
}

impl EvalStrategy {
    /// Create holdout strategy with given validation ratio (single fold)
    pub fn holdout(validation_ratio: f32) -> Self {
        Self::Holdout {
            validation_ratio,
            folds: 1,
        }
    }

    /// Create conformal prediction-based evaluation strategy (single fold)
    ///
    /// Uses the conformal quantile `q` as the optimization metric.
    /// Lower `q` = tighter intervals = more confident model.
    ///
    /// # Arguments
    /// * `calibration_ratio` - Fraction for calibration set (e.g., 0.2)
    /// * `quantile` - Coverage quantile (e.g., 0.9 for 90% coverage)
    pub fn conformal(calibration_ratio: f32, quantile: f32) -> Self {
        Self::Conformal {
            calibration_ratio,
            quantile,
            folds: 1,
        }
    }

    /// Create conformal strategy with default 90% coverage
    pub fn conformal_90(calibration_ratio: f32) -> Self {
        Self::conformal(calibration_ratio, 0.9)
    }

    /// Set the number of folds for cross-validation
    ///
    /// - `folds=1`: Simple single split (default)
    /// - `folds=5`: 5-fold cross-validation (recommended)
    /// - `folds=10`: 10-fold cross-validation (more robust, slower)
    pub fn with_folds(mut self, folds: usize) -> Self {
        match &mut self {
            Self::Holdout { folds: f, .. } => *f = folds,
            Self::Conformal { folds: f, .. } => *f = folds,
        }
        self
    }

    /// Get the number of folds
    pub fn folds(&self) -> usize {
        match self {
            Self::Holdout { folds, .. } => *folds,
            Self::Conformal { folds, .. } => *folds,
        }
    }

    /// Automatically select strategy based on dataset size
    ///
    /// - < 1,000 samples: 5-fold CV
    /// - 1,000 - 5,000 samples: 3-fold CV
    /// - > 5,000 samples: 20% holdout
    pub fn auto(num_samples: usize) -> Self {
        if num_samples < 1_000 {
            Self::holdout(0.2).with_folds(5)
        } else if num_samples < 5_000 {
            Self::holdout(0.2).with_folds(3)
        } else {
            Self::holdout(0.2)
        }
    }

    /// Validate the strategy
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Holdout {
                validation_ratio,
                folds,
            } => {
                if *validation_ratio <= 0.0 || *validation_ratio >= 1.0 {
                    return Err(format!(
                        "validation_ratio must be in (0, 1), got {}",
                        validation_ratio
                    ));
                }
                if *folds == 0 {
                    return Err("folds must be >= 1".into());
                }
            }
            Self::Conformal {
                calibration_ratio,
                quantile,
                folds,
            } => {
                if *calibration_ratio <= 0.0 || *calibration_ratio >= 1.0 {
                    return Err(format!(
                        "calibration_ratio must be in (0, 1), got {}",
                        calibration_ratio
                    ));
                }
                if *quantile <= 0.0 || *quantile >= 1.0 {
                    return Err(format!("quantile must be in (0, 1), got {}", quantile));
                }
                if *folds == 0 {
                    return Err("folds must be >= 1".into());
                }
            }
        }
        Ok(())
    }
}

/// Strategy for generating candidate hyperparameter configurations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridStrategy {
    /// Full Cartesian product grid
    ///
    /// Generates `points_per_dim^n` candidates where n = number of parameters.
    /// For 5 parameters with 3 points each: 3^5 = 243 candidates per iteration.
    Cartesian {
        /// Number of points per dimension (typically 3)
        points_per_dim: usize,
    },
    /// Latin Hypercube Sampling
    ///
    /// Space-filling design that ensures good coverage with fewer samples.
    /// More efficient than Cartesian for high-dimensional spaces.
    LatinHypercube {
        /// Total number of samples to generate
        n_samples: usize,
    },
    /// Pure random sampling
    ///
    /// Simple but may miss important regions. Good for very large spaces.
    Random {
        /// Total number of samples to generate
        n_samples: usize,
    },
}

impl Default for GridStrategy {
    fn default() -> Self {
        Self::Cartesian { points_per_dim: 3 }
    }
}

impl GridStrategy {
    /// Create Cartesian grid with given points per dimension
    pub fn cartesian(points_per_dim: usize) -> Self {
        Self::Cartesian { points_per_dim }
    }

    /// Create Latin Hypercube sampling with given sample count
    pub fn lhs(n_samples: usize) -> Self {
        Self::LatinHypercube { n_samples }
    }

    /// Create random sampling with given sample count
    pub fn random(n_samples: usize) -> Self {
        Self::Random { n_samples }
    }

    /// Get the number of candidates this strategy will generate
    pub fn num_candidates(&self, num_params: usize) -> usize {
        match self {
            Self::Cartesian { points_per_dim } => points_per_dim.pow(num_params as u32),
            Self::LatinHypercube { n_samples } => *n_samples,
            Self::Random { n_samples } => *n_samples,
        }
    }

    /// Validate the strategy
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Cartesian { points_per_dim } => {
                if *points_per_dim < 2 {
                    return Err(format!(
                        "points_per_dim must be >= 2, got {}",
                        points_per_dim
                    ));
                }
            }
            Self::LatinHypercube { n_samples } | Self::Random { n_samples } => {
                if *n_samples < 1 {
                    return Err(format!("n_samples must be >= 1, got {}", n_samples));
                }
            }
        }
        Ok(())
    }
}

/// Tuning mode: trade-off between speed and accuracy
///
/// Controls how the tuner handles data encoding during evaluation.
/// This is critical for classification tasks with categorical features
/// that use target encoding.
///
/// # Target Leakage Problem
///
/// When target encoding is applied to ALL data before tuning, the validation
/// metrics are optimistically biased because validation rows' target values
/// "leak" into their own encodings.
///
/// # Mode Comparison
///
/// | Mode       | Speed      | Accuracy   | Use Case                           |
/// |------------|------------|------------|------------------------------------|
/// | Optimistic | Fast (1x)  | Biased     | Quick exploration, large datasets  |
/// | Realistic  | Slow (~3x) | Unbiased   | Final tuning, small datasets       |
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TuningMode {
    /// Fast mode: uses pre-encoded data
    ///
    /// Target encoding is done once on ALL data before tuning.
    /// This causes target leakage in validation metrics, but is fast.
    ///
    /// **Symptoms of leakage**: Internal F1 score is much higher than
    /// final evaluation (e.g., 95% during tuning → 76% on holdout).
    ///
    /// Use for:
    /// - Quick exploration of hyperparameter space
    /// - Large datasets where realistic mode is too slow
    /// - Numeric-only data (no target encoding needed)
    #[default]
    Optimistic,

    /// Accurate mode: encodes per train/validation split
    ///
    /// For each trial, encoding is fit ONLY on training data, then applied
    /// to validation data. This prevents target leakage.
    ///
    /// ~3x slower than optimistic mode (encoding done per trial).
    ///
    /// Use for:
    /// - Classification with categorical features
    /// - Small datasets where leakage causes significant bias
    /// - Final hyperparameter selection before production
    Realistic,
}

impl TuningMode {
    /// Create optimistic (fast) tuning mode
    pub fn optimistic() -> Self {
        Self::Optimistic
    }

    /// Create realistic (accurate) tuning mode
    pub fn realistic() -> Self {
        Self::Realistic
    }

    /// Check if this is optimistic mode
    pub fn is_optimistic(&self) -> bool {
        matches!(self, Self::Optimistic)
    }

    /// Check if this is realistic mode
    pub fn is_realistic(&self) -> bool {
        matches!(self, Self::Realistic)
    }
}

/// Metric to use for selecting the "best" trial
///
/// This determines which metric is used to compare trials and select the winner.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OptimizationMetric {
    /// Use validation loss (LogLoss for classification, MSE for regression)
    /// Lower is better. This is the default.
    #[default]
    ValidationLoss,
    /// Use F1 score (binary/multi-class classification)
    /// Higher is better.
    F1Score,
    /// Use ROC-AUC (binary classification only)
    /// Higher is better. Measures ranking quality.
    RocAuc,
    /// Use Rank IC (regression only)
    /// Higher is better. Spearman rank correlation between predictions and targets.
    /// Common in quantitative finance for measuring prediction ranking quality.
    RankIc,
}

impl OptimizationMetric {
    /// Check if higher values are better for this metric
    pub fn higher_is_better(&self) -> bool {
        match self {
            Self::ValidationLoss => false,
            Self::F1Score => true,
            Self::RocAuc => true,
            Self::RankIc => true,
        }
    }

    /// Get the metric name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::ValidationLoss => "validation_loss",
            Self::F1Score => "f1_score",
            Self::RocAuc => "roc_auc",
            Self::RankIc => "rank_ic",
        }
    }
}

/// Task type for the tuner
///
/// Determines which metrics are computed and displayed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TaskType {
    /// Regression task (MSE, MAE, etc.)
    /// No F1 or ROC-AUC computed.
    Regression,
    /// Binary classification (LogLoss, F1, ROC-AUC)
    #[default]
    BinaryClassification,
    /// Multi-class classification (LogLoss, F1)
    /// ROC-AUC is not computed (would need one-vs-rest).
    MultiClassClassification,
}

impl TaskType {
    /// Check if this is a classification task
    pub fn is_classification(&self) -> bool {
        matches!(
            self,
            Self::BinaryClassification | Self::MultiClassClassification
        )
    }

    /// Check if this is binary classification
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::BinaryClassification)
    }

    /// Check if this is regression
    pub fn is_regression(&self) -> bool {
        matches!(self, Self::Regression)
    }
}

/// Main tuner configuration
#[derive(Debug, Clone)]
pub struct TunerConfig {
    /// Search space definition
    pub space: ParameterSpace,
    /// Number of zoom iterations (max)
    pub n_iterations: usize,
    /// Initial spread factor (fraction of range to explore)
    ///
    /// 0.5 means explore +/- 50% around the center on the first iteration.
    pub initial_spread: f32,
    /// Zoom factor per iteration
    ///
    /// After each iteration, spread *= zoom_factor.
    /// 0.5 means halve the search radius each iteration.
    pub zoom_factor: f32,
    /// Grid generation strategy
    pub grid_strategy: GridStrategy,
    /// Evaluation strategy
    pub eval_strategy: EvalStrategy,
    /// Tuning mode: optimistic (fast) or realistic (accurate)
    ///
    /// - Optimistic: Uses pre-encoded data. Fast but may have target leakage.
    /// - Realistic: Encodes per train/val split. Slower but no leakage.
    ///
    /// Use `tune()` for optimistic mode (pre-encoded BinnedDataset).
    /// Use `tune_dataframe()` for realistic mode (raw DataFrame + encoding config).
    pub tuning_mode: TuningMode,
    /// Enable parallel trial evaluation (for CPU backends only)
    ///
    /// When enabled, CPU-based trials are evaluated concurrently using Rayon.
    /// GPU backends always run sequentially to avoid CUDA context contention.
    /// Since GPU trials are fast (~1-2s each), sequential execution is acceptable.
    pub parallel_trials: bool,
    /// Maximum number of parallel trials (0 = auto)
    pub n_parallel: usize,
    /// Number of boosting rounds per trial
    pub num_rounds: usize,

    // Inner loop: Early stopping for individual model training
    /// Early stopping rounds per trial (0 = train all rounds)
    ///
    /// When > 0, training stops if validation loss doesn't improve for this many rounds.
    /// This is the INNER LOOP stopping criterion.
    pub early_stopping_rounds: usize,
    /// Validation ratio for early stopping (e.g., 0.2 = 20%)
    pub validation_ratio: f32,

    // Outer loop: Diminishing returns for hyperparameter search
    /// Minimum relative improvement to continue zooming (e.g., 0.001 = 0.1%)
    ///
    /// If the best metric improves by less than this between iterations, stop searching.
    /// This is the OUTER LOOP stopping criterion (diminishing returns check).
    pub improvement_threshold: f32,

    // Balance check for classification (prevents stopping with unbalanced models)
    /// Minimum F1 score required before stopping (default: 0.8 = 80%)
    ///
    /// For classification tasks, prevents stopping early when the model is
    /// unbalanced. If F1 score is below this threshold, the tuner will continue
    /// searching even if no improvement is found.
    ///
    /// Set to 0.0 to disable this check.
    pub min_f1_score: f32,

    /// Random seed for reproducibility
    pub seed: u64,
    /// Enable verbose logging
    pub verbose: bool,
    /// Metric to use for selecting the "best" trial
    ///
    /// - `ValidationLoss`: Use val_metric (lower is better) - default
    /// - `F1Score`: Use F1 score (higher is better)
    /// - `RocAuc`: Use ROC-AUC (higher is better) - ranking quality
    pub optimization_metric: OptimizationMetric,
    /// Task type (regression, binary classification, multi-class)
    ///
    /// Determines which metrics are computed:
    /// - Regression: only loss (MSE)
    /// - BinaryClassification: loss, F1, ROC-AUC
    /// - MultiClassClassification: loss, F1
    pub task_type: TaskType,
    /// Output directory for logging results (None = no logging)
    ///
    /// When set, creates a timestamped run directory with:
    /// - `iteration_N.csv` - trial results per iteration (streaming)
    /// - `best_params.json` - best hyperparameters
    /// - `best_model.{format}` - serialized best model (for each format in save_model_formats)
    /// - `summary.json` - run metadata
    pub output_dir: Option<PathBuf>,
    /// Formats to save the best model in after tuning
    ///
    /// When non-empty and output_dir is set, retrains with the best config
    /// on the full dataset and saves to `best_model.{format}` for each format.
    ///
    /// Empty by default (no model saving). Use `with_save_model_formats()` to enable.
    ///
    /// # Example
    /// ```ignore
    /// let config = TunerConfig::new()
    ///     .with_output_dir("results")
    ///     .with_save_model_formats(vec![ModelFormat::Rkyv]);
    /// ```
    pub save_model_formats: Vec<ModelFormat>,
}

/// Tuning intensity preset for AutoTuner's hyperparameter optimization.
///
/// ## When to Use TunerPreset
///
/// Use `TunerPreset` when working with **`AutoTuner`** or **`TunerConfig`**.
/// This enum controls the intensity of manual hyperparameter tuning via the AutoTuner API.
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
/// - `TunerPreset::SmokeTest` → Minimal testing only (CI/debug)
/// - `TunerPreset::Quick` ≈ `TuningLevel::Quick` ≈ `TreeTunerPreset::Quick`
/// - `TunerPreset::Balanced` ≈ `TuningLevel::Standard` ≈ `TreeTunerPreset::Standard`
/// - `TunerPreset::Thorough` ≈ `TuningLevel::Thorough` ≈ `TreeTunerPreset::Thorough`
///
/// ## Variants
///
/// ### `SmokeTest`
/// Absolute minimum tuning for testing only.
/// - **Best for**: CI pipelines, unit tests, sanity checks
/// - **Iterations**: 1 (no zoom)
/// - **Time**: Seconds
/// - **Quality**: Not production-ready, just validates code works
///
/// ### `Quick`
/// Fast exploration with loose convergence thresholds.
/// - **Best for**: Prototyping, small datasets, time-constrained experiments
/// - **Iterations**: 2
/// - **Time**: Minutes
/// - **Quality**: Decent baseline, may not be fully converged
///
/// ### `Balanced` (Default)
/// Balanced search with moderate iterations.
/// - **Best for**: Most real-world use cases, production models
/// - **Iterations**: 5
/// - **Time**: Tens of minutes
/// - **Quality**: Well-tuned and production-ready
///
/// ### `Thorough`
/// Deep search with strict convergence thresholds.
/// - **Best for**: Competitions, research, maximum accuracy
/// - **Iterations**: 7+
/// - **Time**: Hours
/// - **Quality**: Highly optimized, near-optimal hyperparameters
///
/// ## Examples
///
/// ```ignore
/// use treeboost::{AutoTuner, TunerConfig, TunerPreset, GBDTConfig};
///
/// // Quick tuning
/// let config = TunerConfig::with_preset(TunerPreset::Quick);
/// let tuner = AutoTuner::new(GBDTConfig::default()).with_config(config);
/// let (best_config, history) = tuner.tune(&dataset)?;
///
/// // Production tuning (default)
/// let tuner = AutoTuner::new(GBDTConfig::default()); // Balanced is default
/// let (best_config, history) = tuner.tune(&dataset)?;
///
/// // Thorough tuning
/// let config = TunerConfig::with_preset(TunerPreset::Thorough);
/// let tuner = AutoTuner::new(GBDTConfig::default()).with_config(config);
/// let (best_config, history) = tuner.tune(&dataset)?;
/// ```
///
/// ## See Also
///
/// - [`crate::model::TuningLevel`] - For high-level AutoML tuning with `AutoBuilder`
/// - [`crate::model::TreeTunerPreset`] - For tree-specific tuning with `TreeTunerConfig`
/// - [`TunerConfig::with_preset`] - Apply a preset to tuner configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunerPreset {
    /// 1 iteration, minimal rounds - CI/Debug only.
    SmokeTest,
    /// 2 iterations, loose thresholds - fast exploration.
    Quick,
    /// 5 iterations - balanced default.
    Balanced,
    /// 7+ iterations, strict thresholds - deep search.
    Thorough,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            space: ParameterSpace::with_preset(SpacePreset::Regression),
            n_iterations: tuner_defaults::DEFAULT_N_ITERATIONS, // Max zoom iterations
            initial_spread: tuner_defaults::DEFAULT_INITIAL_SPREAD, // Explore full range initially (100%)
            zoom_factor: tuner_defaults::DEFAULT_ZOOM_FACTOR, // Keep 80% each iteration (remove outliers)
            grid_strategy: GridStrategy::Cartesian { points_per_dim: 3 },
            eval_strategy: EvalStrategy::Holdout {
                validation_ratio: tuner_defaults::DEFAULT_TUNER_VAL_RATIO,
                folds: 1,
            },
            tuning_mode: TuningMode::Optimistic, // Fast by default (for backwards compat)
            parallel_trials: true,               // Enable parallel trials by default
            n_parallel: 0,                       // Auto-detect
            num_rounds: tuner_defaults::DEFAULT_TUNER_ROUNDS, // Rounds per trial

            // Inner loop: Early stopping for individual models
            early_stopping_rounds: tuner_defaults::DEFAULT_TUNER_EARLY_STOP, // Stop if no improvement
            validation_ratio: tuner_defaults::DEFAULT_TUNER_VAL_RATIO,

            // Outer loop: Diminishing returns for hyperparameter search
            improvement_threshold: tuner_defaults::DEFAULT_IMPROVEMENT_THRESHOLD,

            // Balance check for classification
            min_f1_score: tuner_defaults::DEFAULT_MIN_F1_SCORE,

            seed: seeds_defaults::DEFAULT_SEED,
            verbose: true,
            optimization_metric: OptimizationMetric::ValidationLoss,
            task_type: TaskType::Regression,
            output_dir: None,
            save_model_formats: Vec::new(), // No model saving by default
        }
    }
}

impl TunerConfig {
    /// Create a new tuner config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: TunerPreset) -> Self {
        match preset {
            TunerPreset::SmokeTest => {
                self.n_iterations = tuner_defaults::SMOKE_TEST_N_ITERATIONS;
                self.num_rounds = tuner_defaults::QUICK_TUNER_ROUNDS;
                self.early_stopping_rounds = tuner_defaults::QUICK_TUNER_EARLY_STOP;
                self.improvement_threshold = tuner_defaults::QUICK_IMPROVEMENT_THRESHOLD;
            }
            TunerPreset::Quick => {
                self.n_iterations = tuner_defaults::QUICK_N_ITERATIONS;
                self.num_rounds = tuner_defaults::QUICK_TUNER_ROUNDS;
                self.early_stopping_rounds = tuner_defaults::QUICK_TUNER_EARLY_STOP;
                self.improvement_threshold = tuner_defaults::QUICK_IMPROVEMENT_THRESHOLD;
            }
            TunerPreset::Balanced => {
                self.n_iterations = tuner_defaults::DEFAULT_N_ITERATIONS;
                self.num_rounds = tuner_defaults::DEFAULT_TUNER_ROUNDS;
                self.early_stopping_rounds = tuner_defaults::DEFAULT_TUNER_EARLY_STOP;
                self.improvement_threshold = tuner_defaults::DEFAULT_IMPROVEMENT_THRESHOLD;
            }
            TunerPreset::Thorough => {
                self.n_iterations = tuner_defaults::THOROUGH_N_ITERATIONS;
                self.num_rounds = tuner_defaults::THOROUGH_TUNER_ROUNDS;
                self.early_stopping_rounds = tuner_defaults::THOROUGH_TUNER_EARLY_STOP;
                self.improvement_threshold = tuner_defaults::THOROUGH_IMPROVEMENT_THRESHOLD;
            }
        }
        self
    }

    // Builder methods

    /// Set the parameter space
    pub fn with_space(mut self, space: ParameterSpace) -> Self {
        self.space = space;
        self
    }

    /// Set the number of zoom iterations
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Set the initial spread factor
    pub fn with_initial_spread(mut self, spread: f32) -> Self {
        self.initial_spread = spread;
        self
    }

    /// Set the zoom factor
    pub fn with_zoom_factor(mut self, factor: f32) -> Self {
        self.zoom_factor = factor;
        self
    }

    /// Set the grid generation strategy
    pub fn with_grid_strategy(mut self, strategy: GridStrategy) -> Self {
        self.grid_strategy = strategy;
        self
    }

    /// Set the evaluation strategy
    pub fn with_eval_strategy(mut self, strategy: EvalStrategy) -> Self {
        self.eval_strategy = strategy;
        self
    }

    /// Enable or disable parallel trial evaluation
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel_trials = enabled;
        self
    }

    /// Set the maximum number of parallel trials
    pub fn with_n_parallel(mut self, n: usize) -> Self {
        self.n_parallel = n;
        self
    }

    /// Set number of boosting rounds per trial
    pub fn with_num_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    /// Set early stopping for individual model training (inner loop)
    ///
    /// When > 0, each model's training stops if validation loss doesn't improve
    /// for this many consecutive rounds.
    ///
    /// # Arguments
    /// * `rounds` - Number of rounds without improvement before stopping
    /// * `validation_ratio` - Fraction of data for validation (e.g., 0.2 for 20%)
    pub fn with_early_stopping(mut self, rounds: usize, validation_ratio: f32) -> Self {
        self.early_stopping_rounds = rounds;
        self.validation_ratio = validation_ratio;
        self
    }

    /// Disable early stopping (train all num_rounds per trial)
    pub fn without_early_stopping(mut self) -> Self {
        self.early_stopping_rounds = 0;
        self
    }

    /// Set improvement threshold for hyperparameter search (outer loop)
    ///
    /// If the best metric improves by less than this threshold between zoom
    /// iterations, the search stops early (diminishing returns).
    ///
    /// # Arguments
    /// * `threshold` - Minimum relative improvement (e.g., 0.001 = 0.1%)
    pub fn with_improvement_threshold(mut self, threshold: f32) -> Self {
        self.improvement_threshold = threshold;
        self
    }

    /// Set the minimum F1 score for classification tasks
    ///
    /// For classification, prevents stopping early when the model is
    /// unbalanced. If F1 score is below this threshold, the tuner
    /// continues searching.
    ///
    /// # Arguments
    /// * `min_f1` - Minimum required F1 score (e.g., 0.5 = 50%)
    pub fn with_min_f1_score(mut self, min_f1: f32) -> Self {
        self.min_f1_score = min_f1;
        self
    }

    /// Set the tuning mode (optimistic or realistic)
    ///
    /// - `Optimistic`: Fast, uses pre-encoded data. May have target leakage.
    /// - `Realistic`: Slower, encodes per split. No target leakage.
    ///
    /// Use realistic mode for classification with categorical features
    /// where accurate F1 estimates are important.
    pub fn with_tuning_mode(mut self, mode: TuningMode) -> Self {
        self.tuning_mode = mode;
        self
    }

    /// Use optimistic (fast) tuning mode
    ///
    /// Equivalent to `with_tuning_mode(TuningMode::Optimistic)`.
    /// Uses pre-encoded data. Fast but may have target leakage.
    pub fn optimistic(mut self) -> Self {
        self.tuning_mode = TuningMode::Optimistic;
        self
    }

    /// Use realistic (accurate) tuning mode
    ///
    /// Equivalent to `with_tuning_mode(TuningMode::Realistic)`.
    /// Encodes per train/val split. Slower but no target leakage.
    pub fn realistic(mut self) -> Self {
        self.tuning_mode = TuningMode::Realistic;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the metric to optimize (determines "best" trial)
    ///
    /// # Arguments
    /// * `metric` - The metric to use for comparing trials
    ///
    /// # Examples
    /// ```ignore
    /// // Optimize for ROC-AUC (ranking quality):
    /// let config = TunerConfig::new()
    ///     .with_optimization_metric(OptimizationMetric::RocAuc);
    ///
    /// // Optimize for F1 score:
    /// let config = TunerConfig::new()
    ///     .with_optimization_metric(OptimizationMetric::F1Score);
    /// ```
    pub fn with_optimization_metric(mut self, metric: OptimizationMetric) -> Self {
        self.optimization_metric = metric;
        self
    }

    /// Set the task type (regression, binary classification, multi-class)
    ///
    /// This determines which metrics are computed during tuning:
    /// - `Regression`: only loss (MSE)
    /// - `BinaryClassification`: loss, F1, ROC-AUC
    /// - `MultiClassClassification`: loss, F1
    pub fn with_task_type(mut self, task_type: TaskType) -> Self {
        self.task_type = task_type;
        self
    }

    /// Set the output directory for logging results
    ///
    /// When set, creates a timestamped run directory with iteration CSVs,
    /// best params JSON, and serialized model.
    pub fn with_output_dir<P: AsRef<std::path::Path>>(mut self, path: P) -> Self {
        self.output_dir = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the formats to save the best model in
    ///
    /// When non-empty and output_dir is set, retrains with the best config
    /// on the full dataset and saves to `best_model.{format}` for each format.
    ///
    /// # Example
    /// ```ignore
    /// // Save in rkyv format (fastest loading)
    /// let config = TunerConfig::new()
    ///     .with_output_dir("results")
    ///     .with_save_model_formats(vec![ModelFormat::Rkyv]);
    /// ```
    pub fn with_save_model_formats(mut self, formats: Vec<ModelFormat>) -> Self {
        self.save_model_formats = formats;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.n_iterations == 0 {
            return Err("n_iterations must be > 0".into());
        }
        if self.initial_spread <= 0.0 || self.initial_spread > 1.0 {
            return Err(format!(
                "initial_spread must be in (0, 1], got {}",
                self.initial_spread
            ));
        }
        if self.zoom_factor <= 0.0 || self.zoom_factor >= 1.0 {
            return Err(format!(
                "zoom_factor must be in (0, 1), got {}",
                self.zoom_factor
            ));
        }
        if self.num_rounds == 0 {
            return Err("num_rounds must be > 0".into());
        }

        // Note: No need to validate parameter names anymore - TunableParam enum ensures type safety
        self.grid_strategy.validate()?;
        self.eval_strategy.validate()?;

        Ok(())
    }

    /// Get the spread factor for a given iteration
    pub fn spread_for_iteration(&self, iteration: usize) -> f32 {
        self.initial_spread * self.zoom_factor.powi(iteration as i32)
    }

    /// Estimate total number of trials
    pub fn estimated_trials(&self) -> usize {
        let candidates_per_iter = self.grid_strategy.num_candidates(self.space.len());
        candidates_per_iter * self.n_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_bounds_continuous() {
        let bounds = ParamBounds::continuous(0.0, 1.0);
        assert_eq!(bounds.clamp(-0.5), 0.0);
        assert_eq!(bounds.clamp(0.5), 0.5);
        assert_eq!(bounds.clamp(1.5), 1.0);
        assert!(bounds.contains(0.5));
        assert!(!bounds.contains(-0.1));
        assert!(!bounds.is_log_scale());
    }

    #[test]
    fn test_param_bounds_log_continuous() {
        let bounds = ParamBounds::log_continuous(0.01, 1.0);
        assert!(bounds.is_log_scale());
        assert_eq!(bounds.min_value(), 0.01);
        assert_eq!(bounds.max_value(), 1.0);
    }

    #[test]
    fn test_param_bounds_discrete() {
        let bounds = ParamBounds::discrete(2, 10);
        assert_eq!(bounds.clamp(1.0), 2.0);
        assert_eq!(bounds.clamp(5.0), 5.0);
        assert_eq!(bounds.clamp(15.0), 10.0);
        assert!(!bounds.is_log_scale());
    }

    #[test]
    fn test_param_bounds_discrete_step() {
        let bounds = ParamBounds::discrete_step(2, 10, 2);
        // Value 5 should round down to 4 (nearest step from 2)
        assert_eq!(bounds.clamp(5.0), 4.0);
        assert_eq!(bounds.clamp(6.0), 6.0);
    }

    #[test]
    fn test_param_def() {
        let mut param = ParamDef::new(
            TunableParam::MaxDepth,
            ParamBounds::continuous(0.0, 10.0),
            5.0,
        );
        assert_eq!(param.name, TunableParam::MaxDepth);
        assert_eq!(param.center, 5.0);

        param.set_center(15.0); // Out of bounds
        assert_eq!(param.center, 10.0); // Clamped
    }

    #[test]
    fn test_parameter_space_default() {
        use TunableParam::*;
        let space = ParameterSpace::with_preset(SpacePreset::Regression);
        assert_eq!(space.len(), 5);
        assert!(space.get(MaxDepth).is_some());
        assert!(space.get(LearningRate).is_some());
        assert!(space.get(Subsample).is_some());
        assert!(space.get(Lambda).is_some());
        assert!(space.get(EntropyWeight).is_some());
    }

    #[test]
    fn test_parameter_space_with_param() {
        use TunableParam::*;
        let space = ParameterSpace::with_preset(SpacePreset::Minimal).with_param(
            Colsample,
            ParamBounds::continuous(0.5, 1.0),
            0.8,
        );

        assert_eq!(space.len(), 3);
        assert!(space.get(Colsample).is_some());
    }

    #[test]
    fn test_parameter_space_without_param() {
        use TunableParam::*;
        let space =
            ParameterSpace::with_preset(SpacePreset::Regression).without_param(EntropyWeight);

        assert_eq!(space.len(), 4);
        assert!(space.get(EntropyWeight).is_none());
    }

    #[test]
    fn test_parameter_space_centers() {
        use TunableParam::*;
        let mut space = ParameterSpace::with_preset(SpacePreset::Minimal);
        let centers = space.centers();
        assert_eq!(centers.get("max_depth"), Some(&6.0));
        assert_eq!(centers.get("learning_rate"), Some(&0.1));

        let mut new_centers = HashMap::new();
        new_centers.insert("max_depth".into(), 8.0);
        space.set_centers(&new_centers);
        assert_eq!(space.get(MaxDepth).unwrap().center, 8.0);
    }

    #[test]
    fn test_parameter_space_add_and_remove() {
        use TunableParam::*;
        // Test that adding a parameter increases space size
        let space = ParameterSpace::new()
            .with_param(MaxDepth, ParamBounds::discrete(2, 10), 6.0)
            .with_param(LearningRate, ParamBounds::log_continuous(0.01, 0.3), 0.1);
        assert_eq!(space.len(), 2);

        // Test that removing a parameter decreases space size
        let space = space.without_param(LearningRate);
        assert_eq!(space.len(), 1);
        assert!(space.get(MaxDepth).is_some());
        assert!(space.get(LearningRate).is_none());
    }

    #[test]
    fn test_eval_strategy() {
        let holdout = EvalStrategy::holdout(0.2);
        assert!(holdout.validate().is_ok());
        assert_eq!(holdout.folds(), 1);

        let holdout_5fold = EvalStrategy::holdout(0.2).with_folds(5);
        assert!(holdout_5fold.validate().is_ok());
        assert_eq!(holdout_5fold.folds(), 5);

        let conformal = EvalStrategy::conformal(0.2, 0.9).with_folds(3);
        assert!(conformal.validate().is_ok());
        assert_eq!(conformal.folds(), 3);

        let invalid_holdout = EvalStrategy::holdout(1.5);
        assert!(invalid_holdout.validate().is_err());

        let invalid_folds = EvalStrategy::holdout(0.2).with_folds(0);
        assert!(invalid_folds.validate().is_err()); // folds must be >= 1
    }

    #[test]
    fn test_eval_strategy_auto() {
        assert!(matches!(
            EvalStrategy::auto(500),
            EvalStrategy::Holdout { folds: 5, .. }
        ));
        assert!(matches!(
            EvalStrategy::auto(2000),
            EvalStrategy::Holdout { folds: 3, .. }
        ));
        assert!(matches!(
            EvalStrategy::auto(10000),
            EvalStrategy::Holdout { folds: 1, .. }
        ));
    }

    #[test]
    fn test_grid_strategy() {
        let cart = GridStrategy::cartesian(3);
        assert_eq!(cart.num_candidates(5), 243); // 3^5

        let lhs = GridStrategy::lhs(50);
        assert_eq!(lhs.num_candidates(5), 50);

        let rand = GridStrategy::random(100);
        assert_eq!(rand.num_candidates(5), 100);
    }

    #[test]
    fn test_grid_strategy_validate() {
        assert!(GridStrategy::cartesian(3).validate().is_ok());
        assert!(GridStrategy::cartesian(1).validate().is_err());
        assert!(GridStrategy::lhs(0).validate().is_err());
    }

    #[test]
    fn test_tuner_config_default() {
        let config = TunerConfig::default();
        assert_eq!(config.n_iterations, 5);
        assert_eq!(config.initial_spread, 1.0); // Full range initially
        assert_eq!(config.zoom_factor, 0.8); // Keep 80% each iteration
        assert_eq!(config.early_stopping_rounds, 10);
        assert_eq!(config.validation_ratio, 0.2);
        assert_eq!(config.improvement_threshold, 0.001);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tuner_config_quick() {
        let config = TunerConfig::default().with_preset(TunerPreset::Quick);
        assert_eq!(config.n_iterations, 2);
        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.early_stopping_rounds, 5);
        assert_eq!(config.improvement_threshold, 0.01); // 1% threshold
    }

    #[test]
    fn test_tuner_config_thorough() {
        let config = TunerConfig::default().with_preset(TunerPreset::Thorough);
        assert_eq!(config.n_iterations, 7);
        assert_eq!(config.num_rounds, 200);
        assert_eq!(config.early_stopping_rounds, 20);
        assert_eq!(config.improvement_threshold, 0.0001); // 0.01% threshold
    }

    #[test]
    fn test_tuner_config_builders() {
        let config = TunerConfig::new()
            .with_iterations(5)
            .with_num_rounds(200)
            .with_seed(123)
            .with_verbose(false);

        assert_eq!(config.n_iterations, 5);
        assert_eq!(config.num_rounds, 200);
        assert_eq!(config.seed, 123);
        assert!(!config.verbose);
    }

    #[test]
    fn test_tuner_config_spread_for_iteration() {
        let config = TunerConfig::default();
        // With initial_spread=1.0 and zoom_factor=0.8:
        // iteration 0: 1.0 * 0.8^0 = 1.0 (100%)
        // iteration 1: 1.0 * 0.8^1 = 0.8 (80%)
        // iteration 2: 1.0 * 0.8^2 = 0.64 (64%)
        assert_eq!(config.spread_for_iteration(0), 1.0);
        assert_eq!(config.spread_for_iteration(1), 0.8);
        assert!((config.spread_for_iteration(2) - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_tuner_config_estimated_trials() {
        let config = TunerConfig::default();
        // 3^5 = 243 candidates per iteration * 5 iterations = 1215
        assert_eq!(config.estimated_trials(), 1215);
    }

    #[test]
    fn test_tuner_config_validate() {
        assert!(TunerConfig::default().validate().is_ok());

        let invalid = TunerConfig::default().with_iterations(0);
        assert!(invalid.validate().is_err());

        let invalid = TunerConfig::default().with_initial_spread(0.0);
        assert!(invalid.validate().is_err());

        let invalid = TunerConfig::default().with_zoom_factor(1.0);
        assert!(invalid.validate().is_err());

        let invalid = TunerConfig::default().with_num_rounds(0);
        assert!(invalid.validate().is_err());
    }
}
