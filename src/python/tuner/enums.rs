//! Python bindings for tuner enum types
//!
//! Provides wrapper types for:
//! - TuningMode: Optimistic vs Realistic evaluation
//! - OptimizationMetric: What metric to optimize
//! - TaskType: Regression, Binary, Multi-class
//! - ModelFormat: Rkyv vs Bincode serialization
//! - GridStrategy: Cartesian, LHS, Random sampling
//! - EvalStrategy: Holdout or Conformal evaluation

use pyo3::prelude::*;

use crate::tuner::{
    EvalStrategy, GridStrategy, ModelFormat, OptimizationMetric, TaskType, TuningMode,
};

/// Python wrapper for TuningMode
///
/// Controls how the tuner handles data encoding during evaluation.
///
/// - Optimistic: Fast mode, uses pre-encoded data (may have target leakage)
/// - Realistic: Accurate mode, encodes per split (no target leakage)
#[pyclass(from_py_object, name = "TuningMode", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTuningMode {
    pub(crate) inner: TuningMode,
}

#[pymethods]
impl PyTuningMode {
    /// Create optimistic (fast) tuning mode
    ///
    /// Uses pre-encoded data. Fast but may have target leakage
    /// when target encoding is applied before tuning.
    #[staticmethod]
    fn optimistic() -> Self {
        Self {
            inner: TuningMode::Optimistic,
        }
    }

    /// Create realistic (accurate) tuning mode
    ///
    /// Encodes per train/validation split. Slower but no target leakage.
    /// Use for classification with categorical features.
    #[staticmethod]
    fn realistic() -> Self {
        Self {
            inner: TuningMode::Realistic,
        }
    }

    /// Check if this is optimistic mode
    #[getter]
    fn is_optimistic(&self) -> bool {
        self.inner.is_optimistic()
    }

    /// Check if this is realistic mode
    #[getter]
    fn is_realistic(&self) -> bool {
        self.inner.is_realistic()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            TuningMode::Optimistic => "TuningMode.optimistic()",
            TuningMode::Realistic => "TuningMode.realistic()",
        }
    }
}

impl From<TuningMode> for PyTuningMode {
    fn from(mode: TuningMode) -> Self {
        Self { inner: mode }
    }
}

/// Python wrapper for OptimizationMetric
///
/// Determines which metric is used to select the "best" trial.
///
/// - ValidationLoss: Lower is better (default)
/// - F1Score: Higher is better
/// - RocAuc: Higher is better (binary classification)
#[pyclass(from_py_object, name = "OptimizationMetric", eq)]
#[derive(Clone, PartialEq)]
pub struct PyOptimizationMetric {
    pub(crate) inner: OptimizationMetric,
}

#[pymethods]
impl PyOptimizationMetric {
    /// Use validation loss (lower is better)
    ///
    /// LogLoss for classification, MSE for regression.
    #[staticmethod]
    fn val_loss() -> Self {
        Self {
            inner: OptimizationMetric::ValidationLoss,
        }
    }

    /// Use F1 score (higher is better)
    ///
    /// Harmonic mean of precision and recall.
    #[staticmethod]
    fn f1_score() -> Self {
        Self {
            inner: OptimizationMetric::F1Score,
        }
    }

    /// Use ROC-AUC (higher is better)
    ///
    /// Measures ranking quality. Binary classification only.
    #[staticmethod]
    fn roc_auc() -> Self {
        Self {
            inner: OptimizationMetric::RocAuc,
        }
    }

    /// Use Rank IC (higher is better)
    ///
    /// Spearman rank correlation between predictions and targets. Regression only.
    #[staticmethod]
    fn rank_ic() -> Self {
        Self {
            inner: OptimizationMetric::RankIc,
        }
    }

    /// Check if higher values are better for this metric
    #[getter]
    fn higher_is_better(&self) -> bool {
        self.inner.higher_is_better()
    }

    /// Get the metric name
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            OptimizationMetric::ValidationLoss => "OptimizationMetric.val_loss()",
            OptimizationMetric::F1Score => "OptimizationMetric.f1_score()",
            OptimizationMetric::RocAuc => "OptimizationMetric.roc_auc()",
            OptimizationMetric::RankIc => "OptimizationMetric.rank_ic()",
        }
    }
}

impl From<OptimizationMetric> for PyOptimizationMetric {
    fn from(metric: OptimizationMetric) -> Self {
        Self { inner: metric }
    }
}

/// Python wrapper for TaskType
///
/// Determines which metrics are computed during tuning.
///
/// - Regression: Only loss (MSE)
/// - BinaryClassification: Loss, F1, ROC-AUC
/// - MultiClassClassification: Loss, F1
#[pyclass(from_py_object, name = "TaskType", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTaskType {
    pub(crate) inner: TaskType,
}

#[pymethods]
impl PyTaskType {
    /// Regression task (MSE, MAE)
    #[staticmethod]
    fn regression() -> Self {
        Self {
            inner: TaskType::Regression,
        }
    }

    /// Binary classification (LogLoss, F1, ROC-AUC)
    #[staticmethod]
    fn binary_classification() -> Self {
        Self {
            inner: TaskType::BinaryClassification,
        }
    }

    /// Multi-class classification (LogLoss, F1)
    #[staticmethod]
    fn multi_class_classification() -> Self {
        Self {
            inner: TaskType::MultiClassClassification,
        }
    }

    /// Check if this is a classification task
    #[getter]
    fn is_classification(&self) -> bool {
        self.inner.is_classification()
    }

    /// Check if this is binary classification
    #[getter]
    fn is_binary(&self) -> bool {
        self.inner.is_binary()
    }

    /// Check if this is regression
    #[getter]
    fn is_regression(&self) -> bool {
        self.inner.is_regression()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            TaskType::Regression => "TaskType.regression()",
            TaskType::BinaryClassification => "TaskType.binary_classification()",
            TaskType::MultiClassClassification => "TaskType.multi_class_classification()",
        }
    }
}

impl From<TaskType> for PyTaskType {
    fn from(task_type: TaskType) -> Self {
        Self { inner: task_type }
    }
}

/// Python wrapper for ModelFormat
///
/// Determines how the best model is saved after tuning.
///
/// - Rkyv: Zero-copy deserialization, fastest loading
/// - Bincode: Compact binary, serde-based
#[pyclass(from_py_object, name = "ModelFormat", eq)]
#[derive(Clone, PartialEq)]
pub struct PyModelFormat {
    pub(crate) inner: ModelFormat,
}

#[pymethods]
impl PyModelFormat {
    /// Rkyv format - zero-copy deserialization
    ///
    /// Fastest model loading, best for production inference.
    /// File extension: .rkyv
    #[staticmethod]
    fn rkyv() -> Self {
        Self {
            inner: ModelFormat::Rkyv,
        }
    }

    /// Bincode format - compact binary serialization
    ///
    /// Good balance of size and compatibility.
    /// File extension: .bin
    #[staticmethod]
    fn bincode() -> Self {
        Self {
            inner: ModelFormat::Bincode,
        }
    }

    /// Get the file extension for this format
    #[getter]
    fn extension(&self) -> &'static str {
        self.inner.extension()
    }

    /// Get the filename for the best model
    #[getter]
    fn filename(&self) -> &'static str {
        self.inner.filename()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            ModelFormat::Rkyv => "ModelFormat.rkyv()",
            ModelFormat::Bincode => "ModelFormat.bincode()",
        }
    }
}

impl From<ModelFormat> for PyModelFormat {
    fn from(format: ModelFormat) -> Self {
        Self { inner: format }
    }
}

/// Python wrapper for GridStrategy
///
/// Strategy for generating candidate hyperparameter configurations.
///
/// - Cartesian: Full grid (points_per_dim^n candidates)
/// - LatinHypercube: Space-filling design
/// - Random: Pure random sampling
#[pyclass(from_py_object, name = "GridStrategy", eq)]
#[derive(Clone, PartialEq)]
pub struct PyGridStrategy {
    pub(crate) inner: GridStrategy,
}

#[pymethods]
impl PyGridStrategy {
    /// Full Cartesian product grid
    ///
    /// Generates points_per_dim^n candidates where n = number of parameters.
    ///
    /// Args:
    ///     points_per_dim: Number of points per dimension (typically 3)
    #[staticmethod]
    #[pyo3(signature = (points_per_dim=3))]
    fn cartesian(points_per_dim: usize) -> PyResult<Self> {
        if points_per_dim < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "points_per_dim must be >= 2",
            ));
        }
        Ok(Self {
            inner: GridStrategy::cartesian(points_per_dim),
        })
    }

    /// Latin Hypercube Sampling
    ///
    /// Space-filling design that ensures good coverage with fewer samples.
    /// More efficient than Cartesian for high-dimensional spaces.
    ///
    /// Args:
    ///     n_samples: Total number of samples to generate
    #[staticmethod]
    fn lhs(n_samples: usize) -> PyResult<Self> {
        if n_samples < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_samples must be >= 1",
            ));
        }
        Ok(Self {
            inner: GridStrategy::lhs(n_samples),
        })
    }

    /// Pure random sampling
    ///
    /// Simple but may miss important regions.
    ///
    /// Args:
    ///     n_samples: Total number of samples to generate
    #[staticmethod]
    fn random(n_samples: usize) -> PyResult<Self> {
        if n_samples < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_samples must be >= 1",
            ));
        }
        Ok(Self {
            inner: GridStrategy::random(n_samples),
        })
    }

    /// Get the number of candidates for a given number of parameters
    fn num_candidates(&self, num_params: usize) -> usize {
        self.inner.num_candidates(num_params)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            GridStrategy::Cartesian { points_per_dim } => {
                format!("GridStrategy.cartesian({})", points_per_dim)
            }
            GridStrategy::LatinHypercube { n_samples } => {
                format!("GridStrategy.lhs({})", n_samples)
            }
            GridStrategy::Random { n_samples } => {
                format!("GridStrategy.random({})", n_samples)
            }
        }
    }
}

impl From<GridStrategy> for PyGridStrategy {
    fn from(strategy: GridStrategy) -> Self {
        Self { inner: strategy }
    }
}

/// Python wrapper for EvalStrategy
///
/// Strategy for evaluating candidate configurations.
///
/// - Holdout: Train/validation split with optional k-fold CV
/// - Conformal: Prediction interval-based evaluation
#[pyclass(from_py_object, name = "EvalStrategy", eq)]
#[derive(Clone, PartialEq)]
pub struct PyEvalStrategy {
    pub(crate) inner: EvalStrategy,
}

#[pymethods]
impl PyEvalStrategy {
    /// Create holdout strategy with given validation ratio
    ///
    /// Args:
    ///     validation_ratio: Fraction for validation (e.g., 0.2 = 20%)
    ///
    /// Returns:
    ///     EvalStrategy configured for holdout evaluation
    #[staticmethod]
    #[pyo3(signature = (validation_ratio=0.2))]
    fn holdout(validation_ratio: f32) -> PyResult<Self> {
        if validation_ratio <= 0.0 || validation_ratio >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "validation_ratio must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: EvalStrategy::holdout(validation_ratio),
        })
    }

    /// Create conformal prediction-based evaluation
    ///
    /// Uses the conformal quantile as the optimization metric.
    /// Lower quantile = tighter intervals = more confident model.
    ///
    /// Args:
    ///     calibration_ratio: Fraction for calibration (e.g., 0.2)
    ///     coverage: Coverage quantile (e.g., 0.9 for 90%)
    #[staticmethod]
    #[pyo3(signature = (calibration_ratio=0.2, coverage=0.9))]
    fn conformal(calibration_ratio: f32, coverage: f32) -> PyResult<Self> {
        if calibration_ratio <= 0.0 || calibration_ratio >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "calibration_ratio must be in (0, 1)",
            ));
        }
        if coverage <= 0.0 || coverage >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coverage must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: EvalStrategy::conformal(calibration_ratio, coverage),
        })
    }

    /// Auto-select strategy based on dataset size
    ///
    /// - < 1,000 samples: 5-fold CV
    /// - 1,000 - 5,000 samples: 3-fold CV
    /// - > 5,000 samples: 20% holdout
    #[staticmethod]
    fn auto(num_samples: usize) -> Self {
        Self {
            inner: EvalStrategy::auto(num_samples),
        }
    }

    /// Set the number of folds for cross-validation
    ///
    /// Returns a new EvalStrategy with the specified fold count.
    ///
    /// Args:
    ///     folds: Number of folds (1 = simple split, 5 = 5-fold CV)
    fn with_folds(&self, folds: usize) -> PyResult<Self> {
        if folds == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "folds must be >= 1",
            ));
        }
        Ok(Self {
            inner: self.inner.with_folds(folds),
        })
    }

    /// Get the number of folds
    #[getter]
    fn folds(&self) -> usize {
        self.inner.folds()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            EvalStrategy::Holdout {
                validation_ratio,
                folds,
            } => {
                if folds == 1 {
                    format!("EvalStrategy.holdout({})", validation_ratio)
                } else {
                    format!(
                        "EvalStrategy.holdout({}).with_folds({})",
                        validation_ratio, folds
                    )
                }
            }
            EvalStrategy::Conformal {
                calibration_ratio,
                quantile,
                folds,
            } => {
                if folds == 1 {
                    format!(
                        "EvalStrategy.conformal({}, {})",
                        calibration_ratio, quantile
                    )
                } else {
                    format!(
                        "EvalStrategy.conformal({}, {}).with_folds({})",
                        calibration_ratio, quantile, folds
                    )
                }
            }
        }
    }
}

impl From<EvalStrategy> for PyEvalStrategy {
    fn from(strategy: EvalStrategy) -> Self {
        Self { inner: strategy }
    }
}

/// Register tuner enum classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTuningMode>()?;
    m.add_class::<PyOptimizationMetric>()?;
    m.add_class::<PyTaskType>()?;
    m.add_class::<PyModelFormat>()?;
    m.add_class::<PyGridStrategy>()?;
    m.add_class::<PyEvalStrategy>()?;
    Ok(())
}
