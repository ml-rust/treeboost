//! Python bindings for loss functions
//!
//! Provides wrappers for MseLoss, PseudoHuberLoss, BinaryLogLoss,
//! MultiClassLogLoss, and activation functions.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::loss::{
    sigmoid, softmax, BinaryLogLoss, LossFunction, MseLoss, MultiClassLogLoss, PseudoHuberLoss,
};

/// Python wrapper for Mean Squared Error loss
///
/// L(y, ŷ) = 0.5 * (y - ŷ)²
///
/// Properties:
/// - Gradient: ŷ - y
/// - Hessian: 1.0 (constant)
/// - Sensitive to outliers
#[pyclass(from_py_object, name = "MseLoss")]
#[derive(Clone)]
pub struct PyMseLoss {
    inner: MseLoss,
}

#[pymethods]
impl PyMseLoss {
    /// Create a new MSE loss
    #[new]
    fn new() -> Self {
        Self {
            inner: MseLoss::new(),
        }
    }

    /// Compute loss value
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        self.inner.loss(target, prediction)
    }

    /// Compute gradient
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        self.inner.gradient(target, prediction)
    }

    /// Compute hessian (always 1.0 for MSE)
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        self.inner.hessian(target, prediction)
    }

    /// Compute gradient and hessian together
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        self.inner.gradient_hessian(target, prediction)
    }

    /// Loss function name
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> &'static str {
        "MseLoss()"
    }
}

/// Python wrapper for Pseudo-Huber loss
///
/// Smooth approximation to Huber loss that transitions from L2 to L1 behavior.
/// More robust to outliers than MSE.
#[pyclass(from_py_object, name = "PseudoHuberLoss")]
#[derive(Clone)]
pub struct PyPseudoHuberLoss {
    inner: PseudoHuberLoss,
}

#[pymethods]
impl PyPseudoHuberLoss {
    /// Create a new Pseudo-Huber loss
    ///
    /// Args:
    ///     delta: Transition parameter (default 1.0)
    ///         - Small delta (0.1): More robust, slower convergence
    ///         - Large delta (10.0): More like MSE, faster convergence
    ///
    /// Example:
    ///     loss = PseudoHuberLoss(delta=1.0)
    #[new]
    #[pyo3(signature = (delta=1.0))]
    fn new(delta: f32) -> PyResult<Self> {
        if delta <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "delta must be positive",
            ));
        }
        Ok(Self {
            inner: PseudoHuberLoss::new(delta),
        })
    }

    /// Delta parameter
    #[getter]
    fn delta(&self) -> f32 {
        self.inner.delta()
    }

    /// Compute loss value
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        self.inner.loss(target, prediction)
    }

    /// Compute gradient (bounded, unlike MSE)
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        self.inner.gradient(target, prediction)
    }

    /// Compute hessian (approaches 0 for large errors)
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        self.inner.hessian(target, prediction)
    }

    /// Compute gradient and hessian together
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        self.inner.gradient_hessian(target, prediction)
    }

    /// Loss function name
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!("PseudoHuberLoss(delta={})", self.inner.delta())
    }
}

/// Python wrapper for Binary Log Loss (Cross-Entropy)
///
/// For binary classification with targets in {0, 1}.
///
/// L(y, ŷ) = -[y * log(p) + (1-y) * log(1-p)]
/// where p = sigmoid(ŷ)
#[pyclass(from_py_object, name = "BinaryLogLoss")]
#[derive(Clone)]
pub struct PyBinaryLogLoss {
    inner: BinaryLogLoss,
}

#[pymethods]
impl PyBinaryLogLoss {
    /// Create a new Binary Log Loss
    #[new]
    fn new() -> Self {
        Self {
            inner: BinaryLogLoss::new(),
        }
    }

    /// Create with custom epsilon for numerical stability
    #[staticmethod]
    fn with_eps(eps: f32) -> PyResult<Self> {
        if eps <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eps must be positive",
            ));
        }
        Ok(Self {
            inner: BinaryLogLoss::with_eps(eps),
        })
    }

    /// Compute loss value
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        self.inner.loss(target, prediction)
    }

    /// Compute gradient: sigmoid(prediction) - target
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        self.inner.gradient(target, prediction)
    }

    /// Compute hessian: p * (1 - p)
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        self.inner.hessian(target, prediction)
    }

    /// Compute gradient and hessian together
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        self.inner.gradient_hessian(target, prediction)
    }

    /// Convert raw prediction to probability via sigmoid
    fn to_probability(&self, raw: f32) -> f32 {
        self.inner.to_probability(raw)
    }

    /// Convert probability to class (0 or 1) with threshold
    #[pyo3(signature = (prob, threshold=0.5))]
    fn to_class(&self, prob: f32, threshold: f32) -> u32 {
        self.inner.to_class(prob, threshold)
    }

    /// Loss function name
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> &'static str {
        "BinaryLogLoss()"
    }
}

/// Python wrapper for Multi-Class Log Loss (Softmax Cross-Entropy)
///
/// For multi-class classification with K classes.
/// Targets should be class indices: 0, 1, 2, ..., K-1.
#[pyclass(from_py_object, name = "MultiClassLogLoss")]
#[derive(Clone)]
pub struct PyMultiClassLogLoss {
    inner: MultiClassLogLoss,
}

#[pymethods]
impl PyMultiClassLogLoss {
    /// Create a new Multi-Class Log Loss
    ///
    /// Args:
    ///     num_classes: Number of classes (must be >= 2)
    ///
    /// Example:
    ///     loss = MultiClassLogLoss(num_classes=10)
    #[new]
    fn new(num_classes: usize) -> PyResult<Self> {
        if num_classes < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_classes must be >= 2",
            ));
        }
        Ok(Self {
            inner: MultiClassLogLoss::new(num_classes),
        })
    }

    /// Number of classes
    #[getter]
    fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }

    /// Compute gradient and hessian for a specific class
    ///
    /// Args:
    ///     target_class: True class index
    ///     class_idx: Class we're computing gradient for
    ///     raw_preds: Raw predictions for all classes (numpy array)
    ///
    /// Returns:
    ///     (gradient, hessian) tuple for the specified class
    fn gradient_hessian_for_class<'py>(
        &self,
        target_class: usize,
        class_idx: usize,
        raw_preds: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(f32, f32)> {
        let preds = raw_preds.as_array();
        if preds.len() != self.inner.num_classes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "raw_preds length {} doesn't match num_classes {}",
                preds.len(),
                self.inner.num_classes()
            )));
        }
        if target_class >= self.inner.num_classes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "target_class {} >= num_classes {}",
                target_class,
                self.inner.num_classes()
            )));
        }
        if class_idx >= self.inner.num_classes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "class_idx {} >= num_classes {}",
                class_idx,
                self.inner.num_classes()
            )));
        }

        let preds_vec: Vec<f32> = preds.to_vec();
        Ok(self
            .inner
            .gradient_hessian_for_class(target_class, class_idx, &preds_vec))
    }

    /// Compute gradient and hessian for all classes
    ///
    /// Args:
    ///     target_class: True class index
    ///     raw_preds: Raw predictions for all classes (numpy array)
    ///
    /// Returns:
    ///     Tuple of (gradients, hessians) where each is a list for all classes
    fn gradient_hessian_all_classes<'py>(
        &self,
        target_class: usize,
        raw_preds: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let preds = raw_preds.as_array();
        if preds.len() != self.inner.num_classes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "raw_preds length {} doesn't match num_classes {}",
                preds.len(),
                self.inner.num_classes()
            )));
        }
        if target_class >= self.inner.num_classes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "target_class {} >= num_classes {}",
                target_class,
                self.inner.num_classes()
            )));
        }

        let preds_vec: Vec<f32> = preds.to_vec();
        Ok(self
            .inner
            .gradient_hessian_all_classes(target_class, &preds_vec))
    }

    /// Loss function name
    #[getter]
    fn name(&self) -> &'static str {
        "multi_class_log_loss"
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiClassLogLoss(num_classes={})",
            self.inner.num_classes()
        )
    }
}

/// Numerically stable sigmoid function
///
/// sigmoid(x) = 1 / (1 + exp(-x))
///
/// Args:
///     x: Input value
///
/// Returns:
///     Sigmoid output in (0, 1)
#[pyfunction]
fn py_sigmoid(x: f32) -> f32 {
    sigmoid(x)
}

/// Vectorized sigmoid function
///
/// Args:
///     x: Input numpy array
///
/// Returns:
///     Numpy array of sigmoid outputs
#[pyfunction]
fn sigmoid_batch<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> Bound<'py, PyArray1<f32>> {
    let x_arr = x.as_array();
    let result: Vec<f32> = x_arr.iter().map(|&v| sigmoid(v)).collect();
    PyArray1::from_vec(py, result)
}

/// Numerically stable softmax function
///
/// softmax(x)_i = exp(x_i) / sum(exp(x_j))
///
/// Args:
///     raw_scores: Input numpy array of raw scores
///
/// Returns:
///     Numpy array of probabilities (sum to 1.0)
#[pyfunction]
fn py_softmax<'py>(
    py: Python<'py>,
    raw_scores: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    let scores = raw_scores.as_array();
    let scores_vec: Vec<f32> = scores.to_vec();
    let result = softmax(&scores_vec);
    PyArray1::from_vec(py, result)
}

/// Register loss classes and functions with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMseLoss>()?;
    m.add_class::<PyPseudoHuberLoss>()?;
    m.add_class::<PyBinaryLogLoss>()?;
    m.add_class::<PyMultiClassLogLoss>()?;
    m.add_function(wrap_pyfunction!(py_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_softmax, m)?)?;
    Ok(())
}
