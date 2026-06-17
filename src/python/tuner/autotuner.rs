//! Python bindings for AutoTuner
//!
//! Provides PyAutoTuner - the main hyperparameter tuning interface.

use pyo3::prelude::*;

use crate::booster::{GBDTConfig, GBDTModel};
use crate::tuner::AutoTuner;

use super::callback::{validate_callable, PyProgressCallback};
use super::config::{PyParameterSpace, PyTunerConfig};
use super::enums::PyEvalStrategy;
use super::results::PySearchHistory;
use crate::python::bindings::PyGBDTConfig;
use crate::python::dataset::PyBinnedDataset;

/// Python wrapper for AutoTuner
///
/// Hyperparameter tuner using Iterative Grid Search (Auto-Zoom).
///
/// Example:
/// ```python
/// from treeboost import GBDTConfig, AutoTuner, TunerConfig
///
/// # Create base config and tuner
/// base_config = GBDTConfig()
/// tuner = AutoTuner(base_config)
///
/// # Configure tuning
/// tuner = (
///     tuner
///     .with_config(TunerConfig.preset("quick"))
///     .with_callback(lambda trial, curr, total: print(f"{curr}/{total}"))
///     .with_seed(42)
/// )
///
/// # Tune with pre-binned dataset (optimistic mode)
/// best_config, history = tuner.tune(dataset)
/// ```
#[pyclass(name = "AutoTuner")]
pub struct PyAutoTuner {
    base_config: GBDTConfig,
    tuner_config: Option<crate::tuner::TunerConfig>,
    callback: Option<Py<PyAny>>,
    seed: Option<u64>,
}

#[pymethods]
impl PyAutoTuner {
    /// Create a new AutoTuner with the given base configuration
    ///
    /// The base configuration provides default values for all parameters
    /// not being tuned.
    ///
    /// Args:
    ///     base_config: Base GBDTConfig with default parameter values
    #[new]
    fn new(base_config: &PyGBDTConfig) -> Self {
        Self {
            base_config: base_config.inner().clone(),
            tuner_config: None,
            callback: None,
            seed: None,
        }
    }

    /// Set the tuner configuration
    ///
    /// Args:
    ///     config: TunerConfig with tuning settings
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_config(&self, py: Python<'_>, config: &PyTunerConfig) -> Self {
        Self {
            base_config: self.base_config.clone(),
            tuner_config: Some(config.inner.clone()),
            callback: self.callback.as_ref().map(|c| c.clone_ref(py)),
            seed: self.seed,
        }
    }

    /// Set the parameter space
    ///
    /// Args:
    ///     space: ParameterSpace defining parameters to tune
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_space(&self, py: Python<'_>, space: &PyParameterSpace) -> Self {
        let mut config = self.tuner_config.clone().unwrap_or_default();
        config.space = space.inner.clone();
        Self {
            base_config: self.base_config.clone(),
            tuner_config: Some(config),
            callback: self.callback.as_ref().map(|c| c.clone_ref(py)),
            seed: self.seed,
        }
    }

    /// Set the number of iterations
    ///
    /// Args:
    ///     n: Number of zoom iterations
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_iterations(&self, py: Python<'_>, n: usize) -> Self {
        let mut config = self.tuner_config.clone().unwrap_or_default();
        config.n_iterations = n;
        Self {
            base_config: self.base_config.clone(),
            tuner_config: Some(config),
            callback: self.callback.as_ref().map(|c| c.clone_ref(py)),
            seed: self.seed,
        }
    }

    /// Set the evaluation strategy
    ///
    /// Args:
    ///     strategy: EvalStrategy (holdout or conformal)
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_eval_strategy(&self, py: Python<'_>, strategy: &PyEvalStrategy) -> Self {
        let mut config = self.tuner_config.clone().unwrap_or_default();
        config.eval_strategy = strategy.inner;
        Self {
            base_config: self.base_config.clone(),
            tuner_config: Some(config),
            callback: self.callback.as_ref().map(|c| c.clone_ref(py)),
            seed: self.seed,
        }
    }

    /// Enable or disable parallel trial evaluation
    ///
    /// Args:
    ///     enabled: Whether to run trials in parallel
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_parallel(&self, py: Python<'_>, enabled: bool) -> Self {
        let mut config = self.tuner_config.clone().unwrap_or_default();
        config.parallel_trials = enabled;
        Self {
            base_config: self.base_config.clone(),
            tuner_config: Some(config),
            callback: self.callback.as_ref().map(|c| c.clone_ref(py)),
            seed: self.seed,
        }
    }

    /// Set a progress callback
    ///
    /// The callback receives (trial, current, total) where:
    /// - trial: TrialResult with trial details
    /// - current: Current trial number (1-indexed)
    /// - total: Total expected trials
    ///
    /// Args:
    ///     callback: Python callable(trial, current, total)
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_callback(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<Self> {
        validate_callable(py, &callback)?;
        Ok(Self {
            base_config: self.base_config.clone(),
            tuner_config: self.tuner_config.clone(),
            callback: Some(callback),
            seed: self.seed,
        })
    }

    /// Set the random seed for reproducibility
    ///
    /// Args:
    ///     seed: Random seed value
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_seed(&self, py: Python<'_>, seed: u64) -> Self {
        Self {
            base_config: self.base_config.clone(),
            tuner_config: self.tuner_config.clone(),
            callback: self.callback.as_ref().map(|c| c.clone_ref(py)),
            seed: Some(seed),
        }
    }

    /// Run hyperparameter tuning on a pre-binned dataset
    ///
    /// This is the optimistic mode - uses pre-encoded data.
    /// Fast but may have target leakage if target encoding was applied.
    ///
    /// Args:
    ///     dataset: Pre-binned BinnedDataset
    ///
    /// Returns:
    ///     Tuple of (best_config, history)
    fn tune(
        &self,
        py: Python<'_>,
        dataset: &PyBinnedDataset,
    ) -> PyResult<(PyGBDTConfig, PySearchHistory)> {
        // Build the AutoTuner
        let mut tuner = AutoTuner::<GBDTModel>::new(self.base_config.clone());

        if let Some(config) = &self.tuner_config {
            tuner = tuner.with_config(config.clone());
        }

        if let Some(seed) = self.seed {
            tuner = tuner.with_seed(seed);
        }

        // Set up callback if provided
        if let Some(callback_obj) = &self.callback {
            let py_callback = PyProgressCallback::new(callback_obj.clone_ref(py));
            tuner = tuner.with_callback(move |trial, current, total| {
                py_callback.call(trial, current, total);
            });
        }

        // Release GIL during tuning for better performance
        let result = py.detach(|| tuner.tune(&dataset.inner));

        match result {
            Ok((best_config, history)) => {
                let py_config = PyGBDTConfig::from_inner(best_config);
                let py_history = PySearchHistory::from(history);
                Ok((py_config, py_history))
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Tuning failed: {}",
                e
            ))),
        }
    }

    /// Get the current tuner configuration
    #[getter]
    fn config(&self) -> Option<PyTunerConfig> {
        self.tuner_config
            .as_ref()
            .map(|c| PyTunerConfig { inner: c.clone() })
    }

    /// Get the base GBDT configuration
    #[getter]
    fn base(&self) -> PyGBDTConfig {
        PyGBDTConfig::from_inner(self.base_config.clone())
    }

    fn __repr__(&self) -> String {
        let config_info = if let Some(config) = &self.tuner_config {
            format!(
                "iterations={}, rounds={}",
                config.n_iterations, config.num_rounds
            )
        } else {
            "default config".to_string()
        };
        format!("AutoTuner({})", config_info)
    }
}

/// Register AutoTuner with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAutoTuner>()?;
    Ok(())
}
