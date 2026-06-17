//! Python bindings for prediction types
//!
//! Provides wrappers for Prediction and ConformalPredictor.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::inference::{ConformalPredictor, Prediction};

/// Python wrapper for Prediction
///
/// A prediction result with optional confidence interval.
#[pyclass(from_py_object, name = "Prediction")]
#[derive(Clone)]
pub struct PyPrediction {
    inner: Prediction,
}

#[pymethods]
impl PyPrediction {
    /// Create a point prediction without intervals
    ///
    /// Args:
    ///     value: Point prediction value
    #[staticmethod]
    fn point(value: f32) -> Self {
        Self {
            inner: Prediction::point(value),
        }
    }

    /// Create a prediction with confidence interval
    ///
    /// Args:
    ///     point: Point prediction value
    ///     lower: Lower bound of interval
    ///     upper: Upper bound of interval
    #[staticmethod]
    fn with_interval(point: f32, lower: f32, upper: f32) -> Self {
        Self {
            inner: Prediction::with_interval(point, lower, upper),
        }
    }

    /// Point prediction value
    #[getter]
    fn get_point(&self) -> f32 {
        self.inner.point
    }

    /// Lower bound of interval (None if no interval)
    #[getter]
    fn lower(&self) -> Option<f32> {
        self.inner.lower
    }

    /// Upper bound of interval (None if no interval)
    #[getter]
    fn upper(&self) -> Option<f32> {
        self.inner.upper
    }

    /// Check if this prediction has confidence intervals
    #[getter]
    fn has_interval(&self) -> bool {
        self.inner.has_interval()
    }

    /// Get interval width (None if no interval)
    #[getter]
    fn interval_width(&self) -> Option<f32> {
        self.inner.interval_width()
    }

    fn __repr__(&self) -> String {
        if self.inner.has_interval() {
            format!(
                "Prediction(point={:.4}, lower={:.4}, upper={:.4})",
                self.inner.point,
                self.inner.lower.unwrap(),
                self.inner.upper.unwrap()
            )
        } else {
            format!("Prediction(point={:.4})", self.inner.point)
        }
    }
}

impl From<Prediction> for PyPrediction {
    fn from(pred: Prediction) -> Self {
        Self { inner: pred }
    }
}

/// Python wrapper for ConformalPredictor
///
/// Computes prediction intervals with distribution-free coverage guarantees.
/// Uses split conformal prediction with finite-sample guarantees.
#[pyclass(from_py_object, name = "ConformalPredictor")]
#[derive(Clone)]
pub struct PyConformalPredictor {
    inner: ConformalPredictor,
}

#[pymethods]
impl PyConformalPredictor {
    /// Create from calibration residuals
    ///
    /// Args:
    ///     residuals: Numpy array of absolute residuals |y - ŷ| from calibration set
    ///     coverage: Desired coverage level (e.g., 0.9 for 90%)
    ///
    /// Example:
    ///     predictor = ConformalPredictor.from_residuals(residuals, coverage=0.9)
    #[staticmethod]
    #[pyo3(signature = (residuals, coverage=0.9))]
    fn from_residuals<'py>(residuals: PyReadonlyArray1<'py, f32>, coverage: f32) -> PyResult<Self> {
        if coverage <= 0.0 || coverage >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coverage must be in (0, 1)",
            ));
        }

        let residuals_arr = residuals.as_array();
        let residuals_vec: Vec<f32> = residuals_arr.to_vec();

        Ok(Self {
            inner: ConformalPredictor::from_residuals(&residuals_vec, coverage),
        })
    }

    /// Create from precomputed quantile
    ///
    /// Args:
    ///     quantile: Quantile value for symmetric intervals
    ///     coverage: Coverage level used to compute this quantile
    ///
    /// Example:
    ///     predictor = ConformalPredictor.from_quantile(quantile=0.15, coverage=0.9)
    #[staticmethod]
    #[pyo3(signature = (quantile, coverage=0.9))]
    fn from_quantile(quantile: f32, coverage: f32) -> PyResult<Self> {
        if coverage <= 0.0 || coverage >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coverage must be in (0, 1)",
            ));
        }
        if quantile < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "quantile must be non-negative",
            ));
        }

        Ok(Self {
            inner: ConformalPredictor::from_quantile(quantile, coverage),
        })
    }

    /// Create prediction interval for a single point prediction
    ///
    /// Args:
    ///     point: Point prediction value
    ///
    /// Returns:
    ///     Prediction with interval [point - q, point + q]
    fn predict(&self, point: f32) -> PyPrediction {
        self.inner.predict(point).into()
    }

    /// Create prediction intervals for multiple predictions
    ///
    /// Args:
    ///     points: Numpy array of point predictions
    ///
    /// Returns:
    ///     List of Prediction objects with intervals
    fn predict_batch<'py>(&self, points: PyReadonlyArray1<'py, f32>) -> Vec<PyPrediction> {
        let points_arr = points.as_array();
        let points_vec: Vec<f32> = points_arr.to_vec();
        self.inner
            .predict_batch(&points_vec)
            .into_iter()
            .map(|p| p.into())
            .collect()
    }

    /// Get lower bounds for batch predictions as numpy array
    fn predict_batch_lower<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray1<'py, f32>,
    ) -> Bound<'py, PyArray1<f32>> {
        let points_arr = points.as_array();
        let lower: Vec<f32> = points_arr
            .iter()
            .map(|&p| p - self.inner.quantile())
            .collect();
        PyArray1::from_vec(py, lower)
    }

    /// Get upper bounds for batch predictions as numpy array
    fn predict_batch_upper<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray1<'py, f32>,
    ) -> Bound<'py, PyArray1<f32>> {
        let points_arr = points.as_array();
        let upper: Vec<f32> = points_arr
            .iter()
            .map(|&p| p + self.inner.quantile())
            .collect();
        PyArray1::from_vec(py, upper)
    }

    /// Quantile value used for intervals
    #[getter]
    fn quantile(&self) -> f32 {
        self.inner.quantile()
    }

    /// Coverage level (e.g., 0.9 for 90%)
    #[getter]
    fn coverage(&self) -> f32 {
        self.inner.coverage()
    }

    /// Compute empirical coverage on test data
    ///
    /// Returns the fraction of test points where the true value
    /// falls within the predicted interval.
    ///
    /// Args:
    ///     true_values: Numpy array of actual target values
    ///     predictions: Numpy array of predicted values
    ///
    /// Returns:
    ///     Empirical coverage (fraction in [0, 1])
    fn empirical_coverage<'py>(
        &self,
        true_values: PyReadonlyArray1<'py, f32>,
        predictions: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<f32> {
        let true_arr = true_values.as_array();
        let pred_arr = predictions.as_array();

        if true_arr.len() != pred_arr.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "true_values length {} doesn't match predictions length {}",
                true_arr.len(),
                pred_arr.len()
            )));
        }

        let true_vec: Vec<f32> = true_arr.to_vec();
        let pred_vec: Vec<f32> = pred_arr.to_vec();

        Ok(self.inner.empirical_coverage(&true_vec, &pred_vec))
    }

    fn __repr__(&self) -> String {
        format!(
            "ConformalPredictor(quantile={:.4}, coverage={:.2})",
            self.inner.quantile(),
            self.inner.coverage()
        )
    }
}

impl From<ConformalPredictor> for PyConformalPredictor {
    fn from(pred: ConformalPredictor) -> Self {
        Self { inner: pred }
    }
}

/// Register inference classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPrediction>()?;
    m.add_class::<PyConformalPredictor>()?;
    Ok(())
}
