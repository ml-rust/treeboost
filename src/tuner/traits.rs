//! Traits for model-agnostic hyperparameter tuning
//!
//! This module defines the `TunableModel` trait that allows AutoTuner to work
//! with different model types (GBDTModel, UniversalModel, etc.) without
//! code duplication.

use std::collections::HashMap;

use crate::dataset::BinnedDataset;
use crate::Result;

/// Parameter value that can be numeric or categorical
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Numeric parameter (continuous or discrete)
    Numeric(f32),
    /// Categorical parameter (stored as string)
    Categorical(String),
}

impl ParamValue {
    /// Get as numeric value (returns 0.0 for categorical)
    pub fn as_numeric(&self) -> f32 {
        match self {
            Self::Numeric(v) => *v,
            Self::Categorical(_) => 0.0,
        }
    }

    /// Get as categorical string (returns None for numeric)
    pub fn as_categorical(&self) -> Option<&str> {
        match self {
            Self::Categorical(s) => Some(s),
            Self::Numeric(_) => None,
        }
    }

    /// Check if this is a numeric value
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Numeric(_))
    }

    /// Check if this is a categorical value
    pub fn is_categorical(&self) -> bool {
        matches!(self, Self::Categorical(_))
    }
}

impl From<f32> for ParamValue {
    fn from(v: f32) -> Self {
        Self::Numeric(v)
    }
}

impl From<String> for ParamValue {
    fn from(s: String) -> Self {
        Self::Categorical(s)
    }
}

impl From<&str> for ParamValue {
    fn from(s: &str) -> Self {
        Self::Categorical(s.to_string())
    }
}

/// Trait for models that can be tuned by AutoTuner
///
/// This trait abstracts the model training, prediction, and configuration
/// so that AutoTuner can work with different model types without code duplication.
///
/// # Required Methods
///
/// These methods MUST be implemented for your model:
///
/// - [`train`](Self::train) - Train a model on a dataset
/// - [`predict`](Self::predict) - Generate predictions
/// - [`num_trees`](Self::num_trees) - Return number of boosters/trees
/// - [`apply_params`](Self::apply_params) - Apply hyperparameters to config
/// - [`valid_params`](Self::valid_params) - List valid parameter names
/// - [`default_config`](Self::default_config) - Create default configuration
///
/// # Optional Methods (with defaults)
///
/// Override these for additional functionality:
///
/// - [`train_with_validation`](Self::train_with_validation) - Override if your model supports
///   early stopping with explicit validation data
/// - [`is_gpu_config`](Self::is_gpu_config) - Override if your model can use GPU backends
///   (returns `false` by default, causing sequential trial execution)
/// - [`get_learning_rate`](Self::get_learning_rate) - Override if learning rate is tunable
/// - [`configure_validation`](Self::configure_validation) - Override to enable early stopping
/// - [`set_num_rounds`](Self::set_num_rounds) - Override to configure training rounds
/// - [`save_rkyv`](Self::save_rkyv) / [`save_bincode`](Self::save_bincode) - Override to
///   enable model serialization after tuning
///
/// # Example Implementation
///
/// ```ignore
/// impl TunableModel for MyModel {
///     type Config = MyConfig;
///
///     fn train(dataset: &BinnedDataset, config: &Self::Config) -> Result<Self> {
///         // Your training logic
///     }
///
///     fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
///         // Your prediction logic
///     }
///
///     fn num_trees(&self) -> usize {
///         self.trees.len()
///     }
///
///     fn apply_params(config: &mut Self::Config, params: &HashMap<String, ParamValue>) {
///         for (name, value) in params {
///             match (name.as_str(), value) {
///                 ("learning_rate", ParamValue::Numeric(v)) => config.lr = *v,
///                 ("max_depth", ParamValue::Numeric(v)) => config.depth = *v as usize,
///                 _ => {}
///             }
///         }
///     }
///
///     fn valid_params() -> &'static [&'static str] {
///         &["learning_rate", "max_depth"]
///     }
///
///     fn default_config() -> Self::Config {
///         MyConfig::default()
///     }
/// }
/// ```
pub trait TunableModel: Clone + Send + Sync + Sized {
    /// Configuration type for this model
    type Config: Clone + Send + Sync;

    /// Train a model on the given dataset with the given configuration
    fn train(dataset: &BinnedDataset, config: &Self::Config) -> Result<Self>;

    /// Train a model with explicit validation set for early stopping
    ///
    /// Default implementation ignores validation set and calls `train()`.
    /// Override this for models that support early stopping with explicit validation.
    fn train_with_validation(
        train_data: &BinnedDataset,
        val_data: &BinnedDataset,
        val_targets: &[f32],
        config: &Self::Config,
    ) -> Result<Self> {
        let _ = (val_data, val_targets); // Suppress unused warnings
        Self::train(train_data, config)
    }

    /// Predict on a dataset
    fn predict(&self, dataset: &BinnedDataset) -> Vec<f32>;

    /// Get the number of trees/boosters in the trained model
    fn num_trees(&self) -> usize;

    /// Apply parameter values to a configuration
    ///
    /// This method updates the configuration based on the parameter map.
    /// Parameters not in the map should retain their default values.
    fn apply_params(config: &mut Self::Config, params: &HashMap<String, ParamValue>);

    /// Get the list of valid parameter names for this model type
    fn valid_params() -> &'static [&'static str];

    /// Create a default configuration
    fn default_config() -> Self::Config;

    /// Check if the configuration uses GPU backend
    ///
    /// Used to disable parallel trials when GPU is in use.
    /// Default returns false (assumes CPU).
    fn is_gpu_config(config: &Self::Config) -> bool {
        let _ = config;
        false
    }

    /// Get learning rate from configuration
    ///
    /// Used for early stopping analysis and learning rate scheduling.
    /// Default returns 0.1.
    fn get_learning_rate(config: &Self::Config) -> f32 {
        let _ = config;
        0.1
    }

    /// Configure validation settings for early stopping
    ///
    /// Sets up the configuration for early stopping with the given parameters.
    fn configure_validation(
        config: &mut Self::Config,
        validation_ratio: f32,
        early_stopping_rounds: usize,
    ) {
        let _ = (config, validation_ratio, early_stopping_rounds);
    }

    /// Configure number of training rounds
    fn set_num_rounds(config: &mut Self::Config, num_rounds: usize) {
        let _ = (config, num_rounds);
    }

    /// Save the model to a file in rkyv format
    ///
    /// Default implementation returns an error. Override for models that support serialization.
    fn save_rkyv(&self, path: &std::path::Path) -> Result<()> {
        let _ = path;
        Err(crate::TreeBoostError::Config(
            "Model serialization not supported for this model type".to_string(),
        ))
    }

    /// Save the model to a file in bincode format
    ///
    /// Default implementation returns an error. Override for models that support serialization.
    fn save_bincode(&self, path: &std::path::Path) -> Result<()> {
        let _ = path;
        Err(crate::TreeBoostError::Config(
            "Model serialization not supported for this model type".to_string(),
        ))
    }

    /// Check if this model type supports conformal prediction
    ///
    /// Default returns false. Override for models with conformal support.
    fn supports_conformal() -> bool {
        false
    }

    /// Get the conformal quantile from a trained model
    ///
    /// Returns `None` if conformal prediction is not supported or not calibrated.
    /// Override for models that support conformal prediction (e.g., GBDTModel).
    fn conformal_quantile(&self) -> Option<f32> {
        None
    }

    /// Configure conformal prediction settings in the model config
    ///
    /// Default implementation does nothing. Override for models that support conformal.
    fn configure_conformal(config: &mut Self::Config, calibration_ratio: f32, quantile: f32) {
        let _ = (config, calibration_ratio, quantile);
    }
}

/// Helper trait to convert legacy f32 param maps to ParamValue maps
pub trait ParamMapExt {
    /// Convert a `HashMap<String, f32>` to `HashMap<String, ParamValue>`
    ///
    /// Note: This simple conversion treats all values as numeric.
    /// For proper categorical handling, use `to_param_values_with_space`.
    fn to_param_values(&self) -> HashMap<String, ParamValue>;

    /// Convert a `HashMap<String, f32>` to `HashMap<String, ParamValue>` with space context
    ///
    /// This method properly converts categorical parameter indices to their string values
    /// by looking up the index in the ParameterSpace.
    fn to_param_values_with_space(
        &self,
        space: &super::config::ParameterSpace,
    ) -> HashMap<String, ParamValue>;
}

impl ParamMapExt for HashMap<String, f32> {
    fn to_param_values(&self) -> HashMap<String, ParamValue> {
        self.iter()
            .map(|(k, v)| (k.clone(), ParamValue::Numeric(*v)))
            .collect()
    }

    fn to_param_values_with_space(
        &self,
        space: &super::config::ParameterSpace,
    ) -> HashMap<String, ParamValue> {
        use super::config::TunableParam;

        self.iter()
            .map(|(k, v)| {
                // Check if this parameter is categorical
                if let Ok(param_name) = TunableParam::parse(k) {
                    if let Some(param_def) = space.get(param_name) {
                        if let Some(cat_value) = param_def.bounds.get_categorical_value(*v as usize)
                        {
                            return (k.clone(), ParamValue::Categorical(cat_value.to_string()));
                        }
                    }
                }
                // Default to numeric
                (k.clone(), ParamValue::Numeric(*v))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_value_numeric() {
        let v = ParamValue::Numeric(2.5);
        assert!(v.is_numeric());
        assert!(!v.is_categorical());
        assert!((v.as_numeric() - 2.5).abs() < 1e-6);
        assert!(v.as_categorical().is_none());
    }

    #[test]
    fn test_param_value_categorical() {
        let v = ParamValue::Categorical("PureTree".to_string());
        assert!(v.is_categorical());
        assert!(!v.is_numeric());
        assert_eq!(v.as_categorical(), Some("PureTree"));
        assert!((v.as_numeric() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_param_value_from() {
        let v1: ParamValue = 2.5f32.into();
        assert!(v1.is_numeric());

        let v2: ParamValue = "LinearThenTree".into();
        assert!(v2.is_categorical());

        let v3: ParamValue = String::from("RandomForest").into();
        assert!(v3.is_categorical());
    }

    #[test]
    fn test_param_map_ext() {
        let mut map = HashMap::new();
        map.insert("learning_rate".to_string(), 0.1f32);
        map.insert("max_depth".to_string(), 6.0f32);

        let param_values = map.to_param_values();
        assert_eq!(param_values.len(), 2);
        assert!(param_values["learning_rate"].is_numeric());
        assert!((param_values["learning_rate"].as_numeric() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_param_map_ext_with_space() {
        use crate::tuner::config::{ParamBounds, ParameterSpace, TunableParam};

        // Create a parameter space with both numeric and categorical params
        let space = ParameterSpace::new()
            .with_param(
                TunableParam::LearningRate,
                ParamBounds::continuous(0.01, 0.5),
                0.1,
            )
            .with_param(
                TunableParam::Mode,
                ParamBounds::categorical(vec![
                    "PureTree".to_string(),
                    "LinearThenTree".to_string(),
                    "RandomForest".to_string(),
                ]),
                0.0, // Index 0 = "PureTree"
            );

        // Create params with categorical index
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), 0.15f32);
        params.insert("mode".to_string(), 1.0f32); // Index 1 = "LinearThenTree"

        let param_values = params.to_param_values_with_space(&space);

        // Numeric param should stay numeric
        assert!(param_values["learning_rate"].is_numeric());
        assert!((param_values["learning_rate"].as_numeric() - 0.15).abs() < 1e-6);

        // Categorical param should be converted to string
        assert!(param_values["mode"].is_categorical());
        assert_eq!(
            param_values["mode"].as_categorical(),
            Some("LinearThenTree")
        );
    }

    #[test]
    fn test_param_map_ext_with_space_unknown_param() {
        use crate::tuner::config::{ParamBounds, ParameterSpace, TunableParam};

        // Create a minimal parameter space
        let space = ParameterSpace::new().with_param(
            TunableParam::LearningRate,
            ParamBounds::continuous(0.01, 0.5),
            0.1,
        );

        // Create params with an unknown parameter
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), 0.15f32);
        params.insert("unknown_param".to_string(), 42.0f32);

        let param_values = params.to_param_values_with_space(&space);

        // Known param
        assert!(param_values["learning_rate"].is_numeric());

        // Unknown param should still be numeric (fallback)
        assert!(param_values["unknown_param"].is_numeric());
        assert!((param_values["unknown_param"].as_numeric() - 42.0).abs() < 1e-6);
    }
}
