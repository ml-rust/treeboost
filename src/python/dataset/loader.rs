//! Python bindings for DatasetLoader
//!
//! Provides data loading from CSV/Parquet files and NumPy arrays.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::dataset::{BinnedDataset, DatasetBinner, DatasetLoader, FeatureInfo};

use super::types::{PyBinnedDataset, PyFeatureInfo};

/// Python wrapper for DatasetLoader
///
/// Loads tabular data from various sources and converts to BinnedDataset.
///
/// Example:
/// ```python
/// from treeboost import DatasetLoader
///
/// # Create loader with 255 bins (default)
/// loader = DatasetLoader(num_bins=255)
///
/// # Load from CSV
/// dataset = loader.load_csv("data.csv", target="price", features=["f1", "f2"])
///
/// # Load from Parquet
/// dataset = loader.load_parquet("data.parquet", target="price")
///
/// # Load from NumPy arrays
/// dataset = loader.from_numpy(X, y, feature_names=["f1", "f2"])
/// ```
#[pyclass(name = "DatasetLoader")]
pub struct PyDatasetLoader {
    inner: DatasetLoader,
    num_bins: usize,
}

#[pymethods]
impl PyDatasetLoader {
    /// Create a new dataset loader
    ///
    /// Args:
    ///     num_bins: Number of bins for quantile binning (default: 255, max 255)
    #[new]
    #[pyo3(signature = (num_bins=255))]
    fn new(num_bins: usize) -> PyResult<Self> {
        if num_bins == 0 || num_bins > 255 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_bins must be between 1 and 255",
            ));
        }
        Ok(Self {
            inner: DatasetLoader::new(num_bins),
            num_bins,
        })
    }

    /// Number of bins used for quantile binning
    #[getter]
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Load a CSV file into a BinnedDataset
    ///
    /// Args:
    ///     path: Path to the CSV file
    ///     target: Name of the target column
    ///     features: Optional list of feature column names. If None, uses all columns except target.
    ///
    /// Returns:
    ///     BinnedDataset with quantile-binned features
    #[pyo3(signature = (path, target, features=None))]
    fn load_csv(
        &self,
        py: Python<'_>,
        path: &str,
        target: &str,
        features: Option<Vec<String>>,
    ) -> PyResult<PyBinnedDataset> {
        let feature_refs: Option<Vec<&str>> = features
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = py.detach(|| self.inner.load_csv(path, target, feature_refs.as_deref()));

        match result {
            Ok(dataset) => Ok(PyBinnedDataset::from(dataset)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load CSV: {}",
                e
            ))),
        }
    }

    /// Load a Parquet file into a BinnedDataset
    ///
    /// Args:
    ///     path: Path to the Parquet file
    ///     target: Name of the target column
    ///     features: Optional list of feature column names. If None, uses all columns except target.
    ///
    /// Returns:
    ///     BinnedDataset with quantile-binned features
    #[pyo3(signature = (path, target, features=None))]
    fn load_parquet(
        &self,
        py: Python<'_>,
        path: &str,
        target: &str,
        features: Option<Vec<String>>,
    ) -> PyResult<PyBinnedDataset> {
        let feature_refs: Option<Vec<&str>> = features
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = py.detach(|| {
            self.inner
                .load_parquet(path, target, feature_refs.as_deref())
        });

        match result {
            Ok(dataset) => Ok(PyBinnedDataset::from(dataset)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load Parquet: {}",
                e
            ))),
        }
    }

    /// Create a BinnedDataset from NumPy arrays
    ///
    /// Args:
    ///     features: 2D numpy array of shape (num_samples, num_features)
    ///     targets: 1D numpy array of shape (num_samples,)
    ///     feature_names: Optional list of feature names
    ///
    /// Returns:
    ///     BinnedDataset with quantile-binned features
    #[pyo3(signature = (features, targets, feature_names=None))]
    #[allow(clippy::wrong_self_convention)] // reason: name fixed by Python API
    fn from_numpy<'py>(
        &self,
        features: PyReadonlyArray2<'py, f64>,
        targets: PyReadonlyArray1<'py, f32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyBinnedDataset> {
        let features_arr = features.as_array();
        let targets_arr = targets.as_array();

        let num_rows = features_arr.nrows();
        let num_features = features_arr.ncols();

        // Validate dimensions
        if targets_arr.len() != num_rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "targets length {} doesn't match num_rows {}",
                targets_arr.len(),
                num_rows
            )));
        }

        // Generate feature names if not provided
        let names: Vec<String> = feature_names.unwrap_or_else(|| {
            (0..num_features)
                .map(|i| format!("feature_{}", i))
                .collect()
        });

        if names.len() != num_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "feature_names length {} doesn't match num_features {}",
                names.len(),
                num_features
            )));
        }

        // Create binner
        let binner = DatasetBinner::new(self.num_bins);

        // Process each column
        let mut all_binned: Vec<Vec<u8>> = Vec::with_capacity(num_features);
        let mut all_info: Vec<FeatureInfo> = Vec::with_capacity(num_features);

        for (f, name) in names.iter().enumerate() {
            // Extract column
            let col: Vec<f64> = features_arr.column(f).to_vec();

            let (binned, info) =
                binner
                    .process_numeric_column(name.clone(), &col)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to process column '{}': {}",
                            names[f], e
                        ))
                    })?;

            all_binned.push(binned);
            all_info.push(info);
        }

        // Flatten to column-major layout
        let mut features_flat = Vec::with_capacity(num_rows * num_features);
        for binned_col in all_binned {
            features_flat.extend(binned_col);
        }

        let targets_vec: Vec<f32> = targets_arr.to_vec();

        Ok(PyBinnedDataset::from(BinnedDataset::new(
            num_rows,
            features_flat,
            targets_vec,
            all_info,
        )))
    }

    /// Load a CSV file for prediction using existing feature info
    ///
    /// Uses the bin boundaries from training to ensure consistent binning.
    ///
    /// Args:
    ///     path: Path to the CSV file
    ///     feature_info: List of FeatureInfo from training dataset
    ///
    /// Returns:
    ///     BinnedDataset with consistent binning (dummy targets)
    fn load_csv_for_prediction(
        &self,
        py: Python<'_>,
        path: &str,
        feature_info: Vec<PyFeatureInfo>,
    ) -> PyResult<PyBinnedDataset> {
        let info_vec: Vec<FeatureInfo> = feature_info.into_iter().map(|fi| fi.into()).collect();

        let result = py.detach(|| self.inner.load_csv_for_prediction(path, &info_vec));

        match result {
            Ok(dataset) => Ok(PyBinnedDataset::from(dataset)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load CSV for prediction: {}",
                e
            ))),
        }
    }

    /// Load a Parquet file for prediction using existing feature info
    ///
    /// Uses the bin boundaries from training to ensure consistent binning.
    ///
    /// Args:
    ///     path: Path to the Parquet file
    ///     feature_info: List of FeatureInfo from training dataset
    ///
    /// Returns:
    ///     BinnedDataset with consistent binning (dummy targets)
    fn load_parquet_for_prediction(
        &self,
        py: Python<'_>,
        path: &str,
        feature_info: Vec<PyFeatureInfo>,
    ) -> PyResult<PyBinnedDataset> {
        let info_vec: Vec<FeatureInfo> = feature_info.into_iter().map(|fi| fi.into()).collect();

        let result = py.detach(|| self.inner.load_parquet_for_prediction(path, &info_vec));

        match result {
            Ok(dataset) => Ok(PyBinnedDataset::from(dataset)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load Parquet for prediction: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!("DatasetLoader(num_bins={})", self.num_bins)
    }
}

/// Register DatasetLoader with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetLoader>()?;
    Ok(())
}
