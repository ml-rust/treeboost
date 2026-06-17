//! Python bindings for DataPipeline
//!
//! Provides the dirty data pipeline for handling messy real-world data:
//! 1. Count-Min Sketch filtering for rare categories
//! 2. Ordered Target Encoding with M-Estimate smoothing
//! 3. T-Digest quantile binning

use pyo3::prelude::*;

use crate::dataset::{DataPipeline, PipelineConfig, PipelineState};

use super::types::{PyBinnedDataset, PyFeatureInfo};

/// Python wrapper for PipelineConfig
///
/// Configuration for the dirty data pipeline.
///
/// Example:
/// ```python
/// from treeboost import PipelineConfig
///
/// config = (
///     PipelineConfig()
///     .with_num_bins(255)
///     .with_cms_params(eps=0.001, confidence=0.99, min_count=5)
///     .with_smoothing(10.0)
/// )
/// ```
#[pyclass(from_py_object, name = "PipelineConfig")]
#[derive(Clone)]
pub struct PyPipelineConfig {
    pub(crate) inner: PipelineConfig,
}

#[pymethods]
impl PyPipelineConfig {
    /// Create a new PipelineConfig with default settings
    #[new]
    fn new() -> Self {
        Self {
            inner: PipelineConfig::default(),
        }
    }

    /// Set the number of bins for quantile binning
    ///
    /// Args:
    ///     num_bins: Number of bins (1-255, default: 255)
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_num_bins(&self, num_bins: usize) -> PyResult<Self> {
        if num_bins == 0 || num_bins > 255 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_bins must be between 1 and 255",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_num_bins(num_bins),
        })
    }

    /// Set Count-Min Sketch parameters for rare category filtering
    ///
    /// Args:
    ///     eps: Error tolerance (default: 0.001 = 0.1%)
    ///     confidence: Confidence level (default: 0.99)
    ///     min_count: Minimum count for a category to be kept (default: 5)
    ///
    /// Returns:
    ///     Self for method chaining
    #[pyo3(signature = (eps=0.001, confidence=0.99, min_count=5))]
    fn with_cms_params(&self, eps: f64, confidence: f64, min_count: u64) -> PyResult<Self> {
        if eps <= 0.0 || eps >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eps must be between 0 and 1",
            ));
        }
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "confidence must be between 0 and 1",
            ));
        }
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_cms_params(eps, confidence, min_count),
        })
    }

    /// Set the M-Estimate smoothing parameter for target encoding
    ///
    /// Args:
    ///     smoothing: Smoothing parameter (default: 10.0)
    ///         - Larger values bias towards global mean
    ///         - Smaller values trust category means more
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_smoothing(&self, smoothing: f64) -> PyResult<Self> {
        if smoothing < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "smoothing must be non-negative",
            ));
        }
        Ok(Self {
            inner: self.inner.clone().with_smoothing(smoothing),
        })
    }

    /// Number of bins for quantile binning
    #[getter]
    fn num_bins(&self) -> usize {
        self.inner.num_bins
    }

    /// CMS error tolerance
    #[getter]
    fn cms_eps(&self) -> f64 {
        self.inner.cms_eps
    }

    /// CMS confidence level
    #[getter]
    fn cms_confidence(&self) -> f64 {
        self.inner.cms_confidence
    }

    /// Minimum category count
    #[getter]
    fn min_category_count(&self) -> u64 {
        self.inner.min_category_count
    }

    /// Target encoding smoothing parameter
    #[getter]
    fn target_encoding_smoothing(&self) -> f64 {
        self.inner.target_encoding_smoothing
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineConfig(num_bins={}, cms_eps={}, min_count={}, smoothing={})",
            self.inner.num_bins,
            self.inner.cms_eps,
            self.inner.min_category_count,
            self.inner.target_encoding_smoothing
        )
    }
}

/// Python wrapper for PipelineState
///
/// Learned state from training that is needed for inference.
/// Can be serialized and loaded for production deployment.
#[pyclass(from_py_object, name = "PipelineState")]
#[derive(Clone)]
pub struct PyPipelineState {
    pub(crate) inner: PipelineState,
}

#[pymethods]
impl PyPipelineState {
    /// Get feature info for all columns
    #[getter]
    fn feature_info(&self) -> Vec<PyFeatureInfo> {
        self.inner.feature_info.iter().map(|fi| fi.into()).collect()
    }

    /// Column names in order
    #[getter]
    fn column_order(&self) -> Vec<String> {
        self.inner.column_order.clone()
    }

    /// Indices of categorical columns
    #[getter]
    fn categorical_indices(&self) -> Vec<usize> {
        self.inner.categorical_indices.clone()
    }

    /// Number of categorical columns
    #[getter]
    fn num_categorical(&self) -> usize {
        self.inner.categorical_encodings.len()
    }

    /// Number of columns
    fn __len__(&self) -> usize {
        self.inner.column_order.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineState(columns={}, categorical={})",
            self.inner.column_order.len(),
            self.inner.categorical_encodings.len()
        )
    }
}

impl From<PipelineState> for PyPipelineState {
    fn from(state: PipelineState) -> Self {
        Self { inner: state }
    }
}

/// Python wrapper for DataPipeline
///
/// The dirty data pipeline for handling messy real-world data:
/// 1. Count-Min Sketch filtering (rare categories -> "unknown")
/// 2. Ordered Target Encoding (with M-Estimate smoothing)
/// 3. T-Digest quantile binning
///
/// Example:
/// ```python
/// from treeboost import DataPipeline, PipelineConfig
///
/// # Create pipeline with config
/// config = PipelineConfig().with_num_bins(255).with_smoothing(10.0)
/// pipeline = DataPipeline(config)
///
/// # Train: fit and transform
/// dataset, state = pipeline.load_csv_for_training(
///     "train.csv",
///     target="price",
///     categoricals=["city", "type"]
/// )
///
/// # Inference: transform using learned state
/// test_dataset = pipeline.load_csv_for_inference("test.csv", state)
/// ```
#[pyclass(name = "DataPipeline")]
pub struct PyDataPipeline {
    inner: DataPipeline,
}

#[pymethods]
impl PyDataPipeline {
    /// Create a new DataPipeline
    ///
    /// Args:
    ///     config: Optional PipelineConfig. If None, uses default config.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyPipelineConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: DataPipeline::new(cfg),
        }
    }

    /// Create a DataPipeline with default configuration
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: DataPipeline::default_config(),
        }
    }

    /// Load and process a CSV file for training
    ///
    /// Applies the full pipeline:
    /// 1. CMS filtering for rare categories
    /// 2. Ordered target encoding
    /// 3. Quantile binning
    ///
    /// Args:
    ///     path: Path to the CSV file
    ///     target: Name of the target column
    ///     categoricals: Optional list of categorical column names.
    ///                   If None, auto-detects string columns.
    ///
    /// Returns:
    ///     Tuple of (BinnedDataset, PipelineState)
    #[pyo3(signature = (path, target, categoricals=None))]
    fn load_csv_for_training(
        &self,
        py: Python<'_>,
        path: &str,
        target: &str,
        categoricals: Option<Vec<String>>,
    ) -> PyResult<(PyBinnedDataset, PyPipelineState)> {
        let cat_refs: Option<Vec<&str>> = categoricals
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = py.detach(|| {
            self.inner
                .load_csv_for_training(path, target, cat_refs.as_deref())
        });

        match result {
            Ok((dataset, state, _)) => {
                Ok((PyBinnedDataset::from(dataset), PyPipelineState::from(state)))
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to process CSV for training: {}",
                e
            ))),
        }
    }

    /// Load and process a Parquet file for training
    ///
    /// Applies the full pipeline:
    /// 1. CMS filtering for rare categories
    /// 2. Ordered target encoding
    /// 3. Quantile binning
    ///
    /// Args:
    ///     path: Path to the Parquet file
    ///     target: Name of the target column
    ///     categoricals: Optional list of categorical column names.
    ///                   If None, auto-detects string columns.
    ///
    /// Returns:
    ///     Tuple of (BinnedDataset, PipelineState)
    #[pyo3(signature = (path, target, categoricals=None))]
    fn load_parquet_for_training(
        &self,
        py: Python<'_>,
        path: &str,
        target: &str,
        categoricals: Option<Vec<String>>,
    ) -> PyResult<(PyBinnedDataset, PyPipelineState)> {
        let cat_refs: Option<Vec<&str>> = categoricals
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = py.detach(|| {
            self.inner
                .load_parquet_for_training(path, target, cat_refs.as_deref())
        });

        match result {
            Ok((dataset, state, _)) => {
                Ok((PyBinnedDataset::from(dataset), PyPipelineState::from(state)))
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to process Parquet for training: {}",
                e
            ))),
        }
    }

    /// Load and process a CSV file for inference using learned state
    ///
    /// Applies encodings and binning learned during training.
    ///
    /// Args:
    ///     path: Path to the CSV file
    ///     state: PipelineState from training
    ///
    /// Returns:
    ///     BinnedDataset with consistent encoding/binning (dummy targets)
    fn load_csv_for_inference(
        &self,
        py: Python<'_>,
        path: &str,
        state: &PyPipelineState,
    ) -> PyResult<PyBinnedDataset> {
        let result = py.detach(|| self.inner.load_csv_for_inference(path, &state.inner));

        match result {
            Ok(dataset) => Ok(PyBinnedDataset::from(dataset)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to process CSV for inference: {}",
                e
            ))),
        }
    }

    /// Load and process a Parquet file for inference using learned state
    ///
    /// Applies encodings and binning learned during training.
    ///
    /// Args:
    ///     path: Path to the Parquet file
    ///     state: PipelineState from training
    ///
    /// Returns:
    ///     BinnedDataset with consistent encoding/binning (dummy targets)
    fn load_parquet_for_inference(
        &self,
        py: Python<'_>,
        path: &str,
        state: &PyPipelineState,
    ) -> PyResult<PyBinnedDataset> {
        let result = py.detach(|| self.inner.load_parquet_for_inference(path, &state.inner));

        match result {
            Ok(dataset) => Ok(PyBinnedDataset::from(dataset)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to process Parquet for inference: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> &'static str {
        "DataPipeline(...)"
    }
}

/// Register pipeline classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipelineConfig>()?;
    m.add_class::<PyPipelineState>()?;
    m.add_class::<PyDataPipeline>()?;
    Ok(())
}
