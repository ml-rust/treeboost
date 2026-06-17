//! Python bindings for tuner configuration types
//!
//! Provides wrapper types for:
//! - ParamBounds: Bounds and scaling for parameters
//! - ParameterSpace: Collection of parameters to tune
//! - TunerConfig: Main tuner configuration

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::tuner::{
    ModelFormat, ParamBounds, ParameterSpace, SpacePreset, TunableParam, TunerConfig, TunerPreset,
};

use super::enums::{
    PyEvalStrategy, PyGridStrategy, PyModelFormat, PyOptimizationMetric, PyTaskType, PyTuningMode,
};

/// Python wrapper for ParamBounds
///
/// Defines bounds and scaling for a tunable parameter.
///
/// - Continuous: Float range with optional log scaling
/// - Discrete: Integer range with step size
#[pyclass(from_py_object, name = "ParamBounds")]
#[derive(Clone)]
pub struct PyParamBounds {
    pub(crate) inner: ParamBounds,
}

#[pymethods]
impl PyParamBounds {
    /// Create continuous bounds without log scaling
    ///
    /// Args:
    ///     min: Minimum value
    ///     max: Maximum value
    #[staticmethod]
    fn continuous(min: f32, max: f32) -> PyResult<Self> {
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        Ok(Self {
            inner: ParamBounds::continuous(min, max),
        })
    }

    /// Create continuous bounds with log scaling
    ///
    /// Values are sampled uniformly in log space. Useful for
    /// parameters like learning_rate where the difference between
    /// 0.01 and 0.1 is more significant than 0.1 and 0.2.
    ///
    /// Args:
    ///     min: Minimum value (must be positive)
    ///     max: Maximum value
    #[staticmethod]
    fn log_continuous(min: f32, max: f32) -> PyResult<Self> {
        if min <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be positive for log scaling",
            ));
        }
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        Ok(Self {
            inner: ParamBounds::log_continuous(min, max),
        })
    }

    /// Create discrete bounds with step size 1
    ///
    /// Args:
    ///     min: Minimum integer value
    ///     max: Maximum integer value
    #[staticmethod]
    fn discrete(min: usize, max: usize) -> PyResult<Self> {
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        Ok(Self {
            inner: ParamBounds::discrete(min, max),
        })
    }

    /// Create discrete bounds with custom step size
    ///
    /// Args:
    ///     min: Minimum integer value
    ///     max: Maximum integer value
    ///     step: Step size between values
    #[staticmethod]
    fn discrete_step(min: usize, max: usize, step: usize) -> PyResult<Self> {
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        if step == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "step must be positive",
            ));
        }
        Ok(Self {
            inner: ParamBounds::discrete_step(min, max, step),
        })
    }

    /// Clamp a value to be within bounds
    fn clamp(&self, value: f32) -> f32 {
        self.inner.clamp(value)
    }

    /// Check if a value is within bounds
    fn contains(&self, value: f32) -> bool {
        self.inner.contains(value)
    }

    /// Get the minimum value
    #[getter]
    fn min_value(&self) -> f32 {
        self.inner.min_value()
    }

    /// Get the maximum value
    #[getter]
    fn max_value(&self) -> f32 {
        self.inner.max_value()
    }

    /// Check if this uses log scaling
    #[getter]
    fn is_log_scale(&self) -> bool {
        self.inner.is_log_scale()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ParamBounds::Continuous {
                min,
                max,
                log_scale,
            } => {
                if *log_scale {
                    format!("ParamBounds.log_continuous({}, {})", min, max)
                } else {
                    format!("ParamBounds.continuous({}, {})", min, max)
                }
            }
            ParamBounds::Discrete { min, max, step } => {
                if *step == 1 {
                    format!("ParamBounds.discrete({}, {})", min, max)
                } else {
                    format!("ParamBounds.discrete_step({}, {}, {})", min, max, step)
                }
            }
            ParamBounds::Categorical { values } => {
                format!("ParamBounds.categorical([{}])", values.join(", "))
            }
        }
    }
}

impl From<ParamBounds> for PyParamBounds {
    fn from(bounds: ParamBounds) -> Self {
        Self { inner: bounds }
    }
}

/// Python wrapper for ParameterSpace
///
/// Collection of parameters to tune during hyperparameter search.
#[pyclass(from_py_object, name = "ParameterSpace")]
#[derive(Clone)]
pub struct PyParameterSpace {
    pub(crate) inner: ParameterSpace,
}

#[pymethods]
impl PyParameterSpace {
    /// Create an empty parameter space
    #[new]
    fn new() -> Self {
        Self {
            inner: ParameterSpace::new(),
        }
    }

    /// Create a preset search space.
    ///
    /// Valid presets: minimal, regression, classification, exhaustive, universal.
    #[staticmethod]
    fn preset(preset: &str) -> PyResult<Self> {
        let preset = match preset.to_lowercase().as_str() {
            "minimal" => SpacePreset::Minimal,
            "regression" => SpacePreset::Regression,
            "classification" => SpacePreset::Classification,
            "exhaustive" => SpacePreset::Exhaustive,
            "universal" => SpacePreset::Universal,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown preset (use: minimal, regression, classification, exhaustive, universal)",
                ));
            }
        };
        Ok(Self {
            inner: ParameterSpace::with_preset(preset),
        })
    }

    /// Add or update a parameter in the search space
    ///
    /// If a parameter with the same name exists, it will be replaced.
    ///
    /// Args:
    ///     name: Parameter name (must match GBDTConfig field)
    ///     bounds: Parameter bounds
    ///     center: Initial center point for grid generation
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_param(&self, name: &str, bounds: &PyParamBounds, center: f32) -> PyResult<Self> {
        let param = TunableParam::parse(name).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_param(param, bounds.inner.clone(), center),
        })
    }

    /// Add a continuous parameter
    ///
    /// Args:
    ///     name: Parameter name
    ///     min: Minimum value
    ///     max: Maximum value
    ///     center: Initial center point
    ///
    /// Returns:
    ///     Self for method chaining
    fn add_continuous(&self, name: &str, min: f32, max: f32, center: f32) -> PyResult<Self> {
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        let param = TunableParam::parse(name).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_param(param, ParamBounds::continuous(min, max), center),
        })
    }

    /// Add a log-scaled continuous parameter
    ///
    /// Args:
    ///     name: Parameter name
    ///     min: Minimum value (must be positive)
    ///     max: Maximum value
    ///     center: Initial center point
    ///
    /// Returns:
    ///     Self for method chaining
    fn add_log_continuous(&self, name: &str, min: f32, max: f32, center: f32) -> PyResult<Self> {
        if min <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be positive for log scaling",
            ));
        }
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        let param = TunableParam::parse(name).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: self.inner.clone().with_param(
                param,
                ParamBounds::log_continuous(min, max),
                center,
            ),
        })
    }

    /// Add a discrete parameter
    ///
    /// Args:
    ///     name: Parameter name
    ///     min: Minimum integer value
    ///     max: Maximum integer value
    ///     center: Initial center point
    ///
    /// Returns:
    ///     Self for method chaining
    fn add_discrete(&self, name: &str, min: usize, max: usize, center: f32) -> PyResult<Self> {
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        let param = TunableParam::parse(name).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_param(param, ParamBounds::discrete(min, max), center),
        })
    }

    /// Add an integer range parameter with step
    ///
    /// Args:
    ///     name: Parameter name
    ///     min: Minimum value
    ///     max: Maximum value
    ///     step: Step size
    ///     center: Initial center point
    ///
    /// Returns:
    ///     Self for method chaining
    fn add_integer_range(
        &self,
        name: &str,
        min: usize,
        max: usize,
        step: usize,
        center: f32,
    ) -> PyResult<Self> {
        if min >= max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min must be less than max",
            ));
        }
        if step == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "step must be positive",
            ));
        }
        let param = TunableParam::parse(name).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: self.inner.clone().with_param(
                param,
                ParamBounds::discrete_step(min, max, step),
                center,
            ),
        })
    }

    /// Remove a parameter from tuning
    ///
    /// The parameter will use its default value from the base config.
    ///
    /// Args:
    ///     name: Parameter name to remove
    ///
    /// Returns:
    ///     Self for method chaining
    fn without_param(&self, name: &str) -> PyResult<Self> {
        let param = TunableParam::parse(name).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: self.inner.clone().without_param(param),
        })
    }

    /// Number of parameters in the search space
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the search space is empty
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get parameter names
    fn param_names(&self) -> Vec<String> {
        self.inner
            .param_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Get current centers as a dictionary
    fn centers(&self) -> HashMap<String, f32> {
        self.inner
            .centers()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    fn __repr__(&self) -> String {
        let names = self.inner.param_names();
        format!("ParameterSpace([{}])", names.join(", "))
    }
}

impl From<ParameterSpace> for PyParameterSpace {
    fn from(space: ParameterSpace) -> Self {
        Self { inner: space }
    }
}

/// Python wrapper for TunerConfig
///
/// Main configuration for the hyperparameter tuner.
#[pyclass(from_py_object, name = "TunerConfig")]
#[derive(Clone)]
pub struct PyTunerConfig {
    pub(crate) inner: TunerConfig,
}

#[pymethods]
impl PyTunerConfig {
    /// Create a new tuner config with default settings
    #[new]
    fn new() -> Self {
        Self {
            inner: TunerConfig::default(),
        }
    }

    /// Create a tuner config from a preset name.
    ///
    /// Valid presets: smoketest, quick, balanced, thorough.
    #[staticmethod]
    fn preset(preset: &str) -> PyResult<Self> {
        let preset = match preset.to_lowercase().as_str() {
            "smoketest" | "smoke_test" => TunerPreset::SmokeTest,
            "quick" => TunerPreset::Quick,
            "balanced" => TunerPreset::Balanced,
            "thorough" => TunerPreset::Thorough,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown preset (use: smoketest, quick, balanced, thorough)",
                ));
            }
        };
        Ok(Self {
            inner: TunerConfig::default().with_preset(preset),
        })
    }

    /// Return a new config with the given preset applied.
    ///
    /// Valid presets: smoketest, quick, balanced, thorough.
    fn with_preset(&self, preset: &str) -> PyResult<Self> {
        let preset = match preset.to_lowercase().as_str() {
            "smoketest" | "smoke_test" => TunerPreset::SmokeTest,
            "quick" => TunerPreset::Quick,
            "balanced" => TunerPreset::Balanced,
            "thorough" => TunerPreset::Thorough,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown preset (use: smoketest, quick, balanced, thorough)",
                ));
            }
        };
        Ok(Self {
            inner: self.inner.clone().with_preset(preset),
        })
    }

    // Builder methods

    /// Set the parameter space
    fn with_space(&self, space: &PyParameterSpace) -> Self {
        Self {
            inner: self.inner.clone().with_space(space.inner.clone()),
        }
    }

    /// Set the number of zoom iterations
    fn with_iterations(&self, n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "iterations must be > 0",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_iterations(n),
        })
    }

    /// Set the initial spread factor
    fn with_initial_spread(&self, spread: f32) -> PyResult<Self> {
        if spread <= 0.0 || spread > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_spread must be in (0, 1]",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_initial_spread(spread),
        })
    }

    /// Set the zoom factor
    fn with_zoom_factor(&self, factor: f32) -> PyResult<Self> {
        if factor <= 0.0 || factor >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "zoom_factor must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_zoom_factor(factor),
        })
    }

    /// Set the grid generation strategy
    fn with_grid_strategy(&self, strategy: &PyGridStrategy) -> Self {
        Self {
            inner: self.inner.clone().with_grid_strategy(strategy.inner),
        }
    }

    /// Set the evaluation strategy
    fn with_eval_strategy(&self, strategy: &PyEvalStrategy) -> Self {
        Self {
            inner: self.inner.clone().with_eval_strategy(strategy.inner),
        }
    }

    /// Enable or disable parallel trial evaluation
    fn with_parallel(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().with_parallel(enabled),
        }
    }

    /// Set the maximum number of parallel trials
    fn with_n_parallel(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_n_parallel(n),
        }
    }

    /// Set number of boosting rounds per trial
    fn with_num_rounds(&self, rounds: usize) -> PyResult<Self> {
        if rounds == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_rounds must be > 0",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_num_rounds(rounds),
        })
    }

    /// Set early stopping for individual model training
    ///
    /// Args:
    ///     rounds: Number of rounds without improvement before stopping
    ///     validation_ratio: Fraction of data for validation (e.g., 0.2)
    fn with_early_stopping(&self, rounds: usize, validation_ratio: f32) -> PyResult<Self> {
        if validation_ratio <= 0.0 || validation_ratio >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "validation_ratio must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_early_stopping(rounds, validation_ratio),
        })
    }

    /// Disable early stopping
    fn without_early_stopping(&self) -> Self {
        Self {
            inner: self.inner.clone().without_early_stopping(),
        }
    }

    /// Set improvement threshold for hyperparameter search
    ///
    /// Args:
    ///     threshold: Minimum relative improvement (e.g., 0.001 = 0.1%)
    fn with_improvement_threshold(&self, threshold: f32) -> PyResult<Self> {
        if threshold < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threshold must be non-negative",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_improvement_threshold(threshold),
        })
    }

    /// Set the minimum F1 score for classification
    ///
    /// Args:
    ///     min_f1: Minimum required F1 score (e.g., 0.5 = 50%)
    fn with_min_f1_score(&self, min_f1: f32) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&min_f1) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_f1 must be in [0, 1]",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_min_f1_score(min_f1),
        })
    }

    /// Set the tuning mode
    fn with_tuning_mode(&self, mode: &PyTuningMode) -> Self {
        Self {
            inner: self.inner.clone().with_tuning_mode(mode.inner),
        }
    }

    /// Use optimistic (fast) tuning mode
    fn optimistic(&self) -> Self {
        Self {
            inner: self.inner.clone().optimistic(),
        }
    }

    /// Use realistic (accurate) tuning mode
    fn realistic(&self) -> Self {
        Self {
            inner: self.inner.clone().realistic(),
        }
    }

    /// Set the random seed
    fn with_seed(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().with_seed(seed),
        }
    }

    /// Enable or disable verbose logging
    fn with_verbose(&self, verbose: bool) -> Self {
        Self {
            inner: self.inner.clone().with_verbose(verbose),
        }
    }

    /// Set the metric to optimize
    fn with_optimization_metric(&self, metric: &PyOptimizationMetric) -> Self {
        Self {
            inner: self.inner.clone().with_optimization_metric(metric.inner),
        }
    }

    /// Set the task type
    fn with_task_type(&self, task_type: &PyTaskType) -> Self {
        Self {
            inner: self.inner.clone().with_task_type(task_type.inner),
        }
    }

    /// Set the output directory for logging results
    fn with_output_dir(&self, path: &str) -> Self {
        Self {
            inner: self.inner.clone().with_output_dir(path),
        }
    }

    /// Set the formats to save the best model in
    fn with_save_model_formats(&self, formats: Vec<PyModelFormat>) -> Self {
        let rust_formats: Vec<ModelFormat> = formats.iter().map(|f| f.inner).collect();
        Self {
            inner: self.inner.clone().with_save_model_formats(rust_formats),
        }
    }

    // Getters

    /// Number of zoom iterations
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.n_iterations
    }

    /// Initial spread factor
    #[getter]
    fn initial_spread(&self) -> f32 {
        self.inner.initial_spread
    }

    /// Zoom factor
    #[getter]
    fn zoom_factor(&self) -> f32 {
        self.inner.zoom_factor
    }

    /// Number of boosting rounds per trial
    #[getter]
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds
    }

    /// Early stopping rounds
    #[getter]
    fn early_stopping_rounds(&self) -> usize {
        self.inner.early_stopping_rounds
    }

    /// Validation ratio for early stopping
    #[getter]
    fn validation_ratio(&self) -> f32 {
        self.inner.validation_ratio
    }

    /// Improvement threshold for outer loop
    #[getter]
    fn improvement_threshold(&self) -> f32 {
        self.inner.improvement_threshold
    }

    /// Minimum F1 score for classification
    #[getter]
    fn min_f1_score(&self) -> f32 {
        self.inner.min_f1_score
    }

    /// Whether parallel evaluation is enabled
    #[getter]
    fn parallel_trials(&self) -> bool {
        self.inner.parallel_trials
    }

    /// Random seed
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Verbose logging enabled
    #[getter]
    fn verbose(&self) -> bool {
        self.inner.verbose
    }

    /// Get the parameter space
    #[getter]
    fn space(&self) -> PyParameterSpace {
        self.inner.space.clone().into()
    }

    /// Get the tuning mode
    #[getter]
    fn tuning_mode(&self) -> PyTuningMode {
        self.inner.tuning_mode.into()
    }

    /// Get the optimization metric
    #[getter]
    fn optimization_metric(&self) -> PyOptimizationMetric {
        self.inner.optimization_metric.into()
    }

    /// Get the task type
    #[getter]
    fn task_type(&self) -> PyTaskType {
        self.inner.task_type.into()
    }

    /// Get the grid strategy
    #[getter]
    fn grid_strategy(&self) -> PyGridStrategy {
        self.inner.grid_strategy.into()
    }

    /// Get the eval strategy
    #[getter]
    fn eval_strategy(&self) -> PyEvalStrategy {
        self.inner.eval_strategy.into()
    }

    /// Output directory (None if not set)
    #[getter]
    fn output_dir(&self) -> Option<String> {
        self.inner
            .output_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
    }

    /// Validate the configuration
    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Estimate total number of trials
    fn estimated_trials(&self) -> usize {
        self.inner.estimated_trials()
    }

    /// Get the spread factor for a given iteration
    fn spread_for_iteration(&self, iteration: usize) -> f32 {
        self.inner.spread_for_iteration(iteration)
    }

    fn __repr__(&self) -> String {
        format!(
            "TunerConfig(iterations={}, rounds={}, grid={:?})",
            self.inner.n_iterations, self.inner.num_rounds, self.inner.grid_strategy
        )
    }
}

impl From<TunerConfig> for PyTunerConfig {
    fn from(config: TunerConfig) -> Self {
        Self { inner: config }
    }
}

/// Register tuner config classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParamBounds>()?;
    m.add_class::<PyParameterSpace>()?;
    m.add_class::<PyTunerConfig>()?;
    Ok(())
}
