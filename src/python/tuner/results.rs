//! Python bindings for tuner result types
//!
//! Provides wrapper types for:
//! - TrialResult: Result of a single trial evaluation
//! - SearchHistory: Collection of all trials with best tracking

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::tuner::{OptimizationMetric, SearchHistory, TrialResult};

use super::enums::PyOptimizationMetric;

/// Python wrapper for TrialResult
///
/// Contains the results of a single hyperparameter trial.
#[pyclass(from_py_object, name = "TrialResult")]
#[derive(Clone)]
pub struct PyTrialResult {
    inner: TrialResult,
}

#[pymethods]
impl PyTrialResult {
    /// Unique trial identifier
    #[getter]
    fn trial_id(&self) -> usize {
        self.inner.trial_id
    }

    /// Iteration (zoom level) when this trial was run
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.iteration
    }

    /// Hyperparameter values used
    #[getter]
    fn params(&self) -> HashMap<String, f32> {
        self.inner.params.clone()
    }

    /// Validation metric (lower is better for MSE/LogLoss)
    #[getter]
    fn val_metric(&self) -> f32 {
        self.inner.val_loss
    }

    /// Training metric
    #[getter]
    fn train_metric(&self) -> f32 {
        self.inner.train_loss
    }

    /// Number of trees actually trained
    #[getter]
    fn num_trees(&self) -> usize {
        self.inner.num_trees
    }

    /// Training time in milliseconds
    #[getter]
    fn train_time_ms(&self) -> u64 {
        self.inner.train_time_ms
    }

    /// F1 score for classification (None for regression)
    #[getter]
    fn f1_score(&self) -> Option<f32> {
        self.inner.f1_score
    }

    /// ROC-AUC score for binary classification
    #[getter]
    fn roc_auc(&self) -> Option<f64> {
        self.inner.roc_auc
    }

    /// Get a specific parameter value
    fn get_param(&self, name: &str) -> Option<f32> {
        self.inner.params.get(name).copied()
    }

    fn __repr__(&self) -> String {
        let mut parts = vec![
            format!("trial_id={}", self.inner.trial_id),
            format!("iteration={}", self.inner.iteration),
            format!("val_metric={:.6}", self.inner.val_loss),
            format!("num_trees={}", self.inner.num_trees),
        ];
        if let Some(f1) = self.inner.f1_score {
            parts.push(format!("f1={:.4}", f1));
        }
        if let Some(auc) = self.inner.roc_auc {
            parts.push(format!("roc_auc={:.4}", auc));
        }
        format!("TrialResult({})", parts.join(", "))
    }
}

impl From<TrialResult> for PyTrialResult {
    fn from(result: TrialResult) -> Self {
        Self { inner: result }
    }
}

impl From<&TrialResult> for PyTrialResult {
    fn from(result: &TrialResult) -> Self {
        Self {
            inner: result.clone(),
        }
    }
}

/// Python wrapper for SearchHistory
///
/// Tracks all trial results and maintains the best trial.
#[pyclass(from_py_object, name = "SearchHistory")]
#[derive(Clone)]
pub struct PySearchHistory {
    pub(crate) inner: SearchHistory,
}

#[pymethods]
impl PySearchHistory {
    /// Create a new empty history with default optimization metric
    #[new]
    fn new() -> Self {
        Self {
            inner: SearchHistory::new(),
        }
    }

    /// Create a new history with a specific optimization metric
    #[staticmethod]
    fn with_metric(metric: &PyOptimizationMetric) -> Self {
        Self {
            inner: SearchHistory::with_metric(metric.inner),
        }
    }

    /// Get the best trial so far
    fn best(&self) -> Option<PyTrialResult> {
        self.inner.best().map(|t| t.into())
    }

    /// Get all trials
    fn trials(&self) -> Vec<PyTrialResult> {
        self.inner.trials().iter().map(|t| t.into()).collect()
    }

    /// Get trials for a specific iteration
    fn trials_for_iteration(&self, iteration: usize) -> Vec<PyTrialResult> {
        self.inner
            .trials_for_iteration(iteration)
            .into_iter()
            .map(|t| t.into())
            .collect()
    }

    /// Number of trials
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if history is empty
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the optimization metric being used
    #[getter]
    fn optimization_metric(&self) -> PyOptimizationMetric {
        self.inner.optimization_metric().into()
    }

    /// Get the top N trials by metric
    ///
    /// Args:
    ///     n: Number of top trials to return
    ///
    /// Returns:
    ///     List of top N trials sorted by metric
    fn top_n(&self, n: usize) -> Vec<PyTrialResult> {
        let trials = self.inner.trials();
        let higher_is_better = self.inner.optimization_metric().higher_is_better();

        let mut sorted: Vec<_> = trials.iter().collect();
        sorted.sort_by(|a, b| {
            let metric_a = get_metric_value(a, self.inner.optimization_metric());
            let metric_b = get_metric_value(b, self.inner.optimization_metric());
            if higher_is_better {
                metric_b
                    .partial_cmp(&metric_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                metric_a
                    .partial_cmp(&metric_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        sorted.into_iter().take(n).map(|t| t.into()).collect()
    }

    /// Export history to JSON string
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    fn __repr__(&self) -> String {
        let best_info = if let Some(best) = self.inner.best() {
            format!(
                ", best_trial_id={}, best_val={:.6}",
                best.trial_id, best.val_loss
            )
        } else {
            String::new()
        };
        format!("SearchHistory(len={}{})", self.inner.len(), best_info)
    }
}

/// Helper to extract metric value for sorting
fn get_metric_value(trial: &TrialResult, metric: OptimizationMetric) -> f32 {
    match metric {
        OptimizationMetric::ValidationLoss => trial.val_loss,
        OptimizationMetric::F1Score => trial.f1_score.unwrap_or(0.0),
        OptimizationMetric::RocAuc => trial.roc_auc.unwrap_or(0.0) as f32,
        OptimizationMetric::RankIc => trial.rank_ic.unwrap_or(0.0) as f32,
    }
}

impl From<SearchHistory> for PySearchHistory {
    fn from(history: SearchHistory) -> Self {
        Self { inner: history }
    }
}

/// Register tuner result classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrialResult>()?;
    m.add_class::<PySearchHistory>()?;
    Ok(())
}
