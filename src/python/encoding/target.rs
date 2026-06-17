//! Python bindings for target encoding
//!
//! Provides wrappers for OrderedTargetEncoder and EncodingMap.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::encoding::{EncodingMap, OrderedTargetEncoder};

/// Python wrapper for OrderedTargetEncoder
///
/// Encodes categorical features using smoothed target means while
/// preventing target leakage through ordered/streaming approach.
#[pyclass(name = "OrderedTargetEncoder")]
pub struct PyOrderedTargetEncoder {
    inner: OrderedTargetEncoder,
}

#[pymethods]
impl PyOrderedTargetEncoder {
    /// Create a new ordered target encoder
    ///
    /// Args:
    ///     smoothing: M-estimate smoothing parameter (typically 1.0 to 10.0).
    ///                Higher values pull encodings toward global mean.
    ///
    /// Example:
    ///     encoder = OrderedTargetEncoder(smoothing=10.0)
    #[new]
    #[pyo3(signature = (smoothing=10.0))]
    fn new(smoothing: f64) -> PyResult<Self> {
        if smoothing < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "smoothing must be non-negative",
            ));
        }
        Ok(Self {
            inner: OrderedTargetEncoder::new(smoothing),
        })
    }

    /// Reset the encoder state (clear all statistics)
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Encode a single value using current statistics, then update
    ///
    /// This is the ordered/streaming approach:
    /// 1. Compute encoding using statistics from prior rows only
    /// 2. Update statistics with this row's values
    ///
    /// Args:
    ///     category: Category string to encode
    ///     target: Target value for this observation
    ///
    /// Returns:
    ///     Encoded value (smoothed target mean based on prior statistics)
    fn encode_and_update(&mut self, category: &str, target: f64) -> f64 {
        self.inner.encode_and_update(category, target)
    }

    /// Encode an entire column in streaming order
    ///
    /// Resets internal state and encodes all rows in order.
    /// Each row only sees statistics from prior rows.
    ///
    /// Args:
    ///     categories: List of category strings
    ///     targets: Numpy array of target values
    ///
    /// Returns:
    ///     Numpy array of encoded values
    fn encode_column<'py>(
        &mut self,
        py: Python<'py>,
        categories: Vec<String>,
        targets: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let targets_arr = targets.as_array();

        if categories.len() != targets_arr.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "categories length {} doesn't match targets length {}",
                categories.len(),
                targets_arr.len()
            )));
        }

        let targets_vec: Vec<f64> = targets_arr.to_vec();
        let encoded = self.inner.encode_column(&categories, &targets_vec);

        Ok(PyArray1::from_vec(py, encoded))
    }

    /// Encode using final statistics (for inference)
    ///
    /// After training, use this to encode new data using the learned
    /// statistics without updating them.
    ///
    /// Args:
    ///     category: Category string to encode
    ///
    /// Returns:
    ///     Encoded value using final training statistics
    fn encode_inference(&self, category: &str) -> f64 {
        self.inner.encode_inference(category)
    }

    /// Encode a batch using final statistics (for inference)
    ///
    /// Args:
    ///     categories: List of category strings
    ///
    /// Returns:
    ///     Numpy array of encoded values
    fn encode_inference_batch<'py>(
        &self,
        py: Python<'py>,
        categories: Vec<String>,
    ) -> Bound<'py, PyArray1<f64>> {
        let encoded: Vec<f64> = categories
            .iter()
            .map(|c| self.inner.encode_inference(c))
            .collect();
        PyArray1::from_vec(py, encoded)
    }

    /// Get the learned encoding map for serialization
    ///
    /// Returns an EncodingMap that can be used for inference
    /// without needing the full encoder state.
    fn get_encoding_map(&self) -> PyEncodingMap {
        PyEncodingMap {
            inner: self.inner.get_encoding_map(),
        }
    }

    fn __repr__(&self) -> String {
        "OrderedTargetEncoder()".to_string()
    }
}

/// Python wrapper for serializable encoding map
///
/// Provides a lightweight encoding lookup that can be saved
/// and used for inference without the full encoder state.
#[pyclass(from_py_object, name = "EncodingMap")]
#[derive(Clone)]
pub struct PyEncodingMap {
    inner: EncodingMap,
}

#[pymethods]
impl PyEncodingMap {
    /// Encode a single category
    ///
    /// Args:
    ///     category: Category string to encode
    ///
    /// Returns:
    ///     Encoded value (returns default_value for unknown categories)
    fn encode(&self, category: &str) -> f64 {
        self.inner.encode(category)
    }

    /// Encode a batch of categories
    ///
    /// Args:
    ///     categories: List of category strings
    ///
    /// Returns:
    ///     Numpy array of encoded values
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        categories: Vec<String>,
    ) -> Bound<'py, PyArray1<f64>> {
        let encoded = self.inner.encode_batch(&categories);
        PyArray1::from_vec(py, encoded)
    }

    /// Default value for unknown categories
    #[getter]
    fn default_value(&self) -> f64 {
        self.inner.default_value
    }

    /// Smoothing parameter used during training
    #[getter]
    fn smoothing(&self) -> f64 {
        self.inner.smoothing
    }

    /// Number of known categories
    #[getter]
    fn num_categories(&self) -> usize {
        self.inner.encodings.len()
    }

    /// Get all category-to-value mappings as a list of tuples
    fn items(&self) -> Vec<(String, f64)> {
        self.inner.encodings.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodingMap(num_categories={}, default={:.4})",
            self.inner.encodings.len(),
            self.inner.default_value
        )
    }

    fn __len__(&self) -> usize {
        self.inner.encodings.len()
    }
}

impl From<EncodingMap> for PyEncodingMap {
    fn from(map: EncodingMap) -> Self {
        Self { inner: map }
    }
}

/// Register encoding classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrderedTargetEncoder>()?;
    m.add_class::<PyEncodingMap>()?;
    Ok(())
}
