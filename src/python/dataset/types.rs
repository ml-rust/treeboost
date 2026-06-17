//! Python bindings for dataset types
//!
//! Provides wrappers for FeatureType, FeatureInfo, and BinnedDataset.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};

/// Python wrapper for FeatureType enum
#[pyclass(from_py_object, name = "FeatureType", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyFeatureType {
    inner: FeatureType,
}

#[pymethods]
impl PyFeatureType {
    /// Create a numeric feature type (for continuous values)
    #[staticmethod]
    fn numeric() -> Self {
        Self {
            inner: FeatureType::Numeric,
        }
    }

    /// Create a categorical feature type
    #[staticmethod]
    fn categorical() -> Self {
        Self {
            inner: FeatureType::Categorical,
        }
    }

    /// Check if this is a numeric feature type
    #[getter]
    fn is_numeric(&self) -> bool {
        matches!(self.inner, FeatureType::Numeric)
    }

    /// Check if this is a categorical feature type
    #[getter]
    fn is_categorical(&self) -> bool {
        matches!(self.inner, FeatureType::Categorical)
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            FeatureType::Numeric => "FeatureType.numeric()",
            FeatureType::Categorical => "FeatureType.categorical()",
        }
    }
}

impl From<FeatureType> for PyFeatureType {
    fn from(ft: FeatureType) -> Self {
        Self { inner: ft }
    }
}

impl From<PyFeatureType> for FeatureType {
    fn from(pft: PyFeatureType) -> Self {
        pft.inner
    }
}

/// Python wrapper for feature metadata
#[pyclass(from_py_object, name = "FeatureInfo")]
#[derive(Clone)]
pub struct PyFeatureInfo {
    inner: FeatureInfo,
}

#[pymethods]
impl PyFeatureInfo {
    /// Feature name
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Feature type (numeric or categorical)
    #[getter]
    fn feature_type(&self) -> PyFeatureType {
        self.inner.feature_type.into()
    }

    /// Number of bins used (max 256)
    #[getter]
    fn num_bins(&self) -> u8 {
        self.inner.num_bins
    }

    /// Bin boundaries for numeric features (empty for categorical)
    #[getter]
    fn bin_boundaries<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.bin_boundaries)
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureInfo(name='{}', type={:?}, num_bins={})",
            self.inner.name, self.inner.feature_type, self.inner.num_bins
        )
    }
}

impl From<FeatureInfo> for PyFeatureInfo {
    fn from(fi: FeatureInfo) -> Self {
        Self { inner: fi }
    }
}

impl From<&FeatureInfo> for PyFeatureInfo {
    fn from(fi: &FeatureInfo) -> Self {
        Self { inner: fi.clone() }
    }
}

impl From<PyFeatureInfo> for FeatureInfo {
    fn from(pfi: PyFeatureInfo) -> Self {
        pfi.inner
    }
}

/// Python wrapper for binned dataset
///
/// Provides read-only access to the internal binned representation.
/// This is primarily used for advanced users who want to work with
/// pre-binned data directly.
#[pyclass(name = "BinnedDataset")]
pub struct PyBinnedDataset {
    pub(crate) inner: BinnedDataset,
}

#[pymethods]
impl PyBinnedDataset {
    /// Create a BinnedDataset from numpy arrays
    ///
    /// Args:
    ///     features: 2D numpy array of u8 bins (column-major: [feature][row])
    ///     targets: 1D numpy array of f32 target values
    ///     feature_info: List of FeatureInfo objects
    ///
    /// Note: Features should be in column-major format (features first, then rows).
    /// This is the internal format used by TreeBoost.
    #[staticmethod]
    #[pyo3(signature = (features, targets, feature_info))]
    fn from_arrays<'py>(
        features: PyReadonlyArray2<'py, u8>,
        targets: PyReadonlyArray1<'py, f32>,
        feature_info: Vec<PyFeatureInfo>,
    ) -> PyResult<Self> {
        let features_arr = features.as_array();
        let targets_arr = targets.as_array();

        let num_features = features_arr.nrows();
        let num_rows = features_arr.ncols();

        // Validate dimensions
        if targets_arr.len() != num_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "targets length {} doesn't match num_rows {}",
                targets_arr.len(),
                num_rows
            )));
        }
        if feature_info.len() != num_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "feature_info length {} doesn't match num_features {}",
                feature_info.len(),
                num_features
            )));
        }

        // Convert to column-major flat Vec
        let mut features_flat = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features_flat.push(features_arr[[f, r]]);
            }
        }

        let targets_vec: Vec<f32> = targets_arr.to_vec();
        let info_vec: Vec<FeatureInfo> = feature_info.into_iter().map(|fi| fi.inner).collect();

        Ok(Self {
            inner: BinnedDataset::new(num_rows, features_flat, targets_vec, info_vec),
        })
    }

    /// Number of samples (rows)
    #[getter]
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    /// Number of features (columns)
    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Get target values as numpy array
    #[getter]
    fn targets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.targets())
    }

    /// Get feature info for all features
    #[getter]
    fn feature_info(&self) -> Vec<PyFeatureInfo> {
        self.inner
            .all_feature_info()
            .iter()
            .map(|fi| fi.into())
            .collect()
    }

    /// Get feature info for a specific feature by index
    fn feature_info_at(&self, feature_idx: usize) -> PyResult<PyFeatureInfo> {
        if feature_idx >= self.inner.num_features() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "feature index {} out of range (num_features={})",
                feature_idx,
                self.inner.num_features()
            )));
        }
        Ok(self.inner.feature_info(feature_idx).into())
    }

    /// Get bin values for a feature column as numpy array
    fn feature_column<'py>(
        &self,
        py: Python<'py>,
        feature_idx: usize,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if feature_idx >= self.inner.num_features() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "feature index {} out of range (num_features={})",
                feature_idx,
                self.inner.num_features()
            )));
        }
        Ok(PyArray1::from_slice(
            py,
            self.inner.feature_column(feature_idx),
        ))
    }

    /// Get bin value for a specific row and feature
    fn get_bin(&self, row_idx: usize, feature_idx: usize) -> PyResult<u8> {
        if row_idx >= self.inner.num_rows() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "row index {} out of range (num_rows={})",
                row_idx,
                self.inner.num_rows()
            )));
        }
        if feature_idx >= self.inner.num_features() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "feature index {} out of range (num_features={})",
                feature_idx,
                self.inner.num_features()
            )));
        }
        Ok(self.inner.get_bin(row_idx, feature_idx))
    }

    /// Check if a feature is sparse
    fn is_sparse(&self, feature_idx: usize) -> PyResult<bool> {
        if feature_idx >= self.inner.num_features() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "feature index {} out of range",
                feature_idx
            )));
        }
        Ok(self.inner.is_sparse(feature_idx))
    }

    /// Number of sparse features
    #[getter]
    fn num_sparse_features(&self) -> usize {
        self.inner.num_sparse_features()
    }

    /// Maximum number of bins across all features
    #[getter]
    fn max_bins(&self) -> u8 {
        self.inner.max_bins()
    }

    /// Whether dataset supports 4-bit packing (all features have <=16 bins)
    #[getter]
    fn supports_4bit(&self) -> bool {
        self.inner.supports_4bit()
    }

    /// Check if era indices are available
    #[getter]
    fn has_eras(&self) -> bool {
        self.inner.has_eras()
    }

    /// Number of eras (0 if no era indices)
    #[getter]
    fn num_eras(&self) -> usize {
        self.inner.num_eras()
    }

    fn __repr__(&self) -> String {
        format!(
            "BinnedDataset(num_rows={}, num_features={}, max_bins={}, sparse_features={})",
            self.inner.num_rows(),
            self.inner.num_features(),
            self.inner.max_bins(),
            self.inner.num_sparse_features()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.num_rows()
    }
}

impl From<BinnedDataset> for PyBinnedDataset {
    fn from(ds: BinnedDataset) -> Self {
        Self { inner: ds }
    }
}

/// Register dataset type classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFeatureType>()?;
    m.add_class::<PyFeatureInfo>()?;
    m.add_class::<PyBinnedDataset>()?;
    Ok(())
}
