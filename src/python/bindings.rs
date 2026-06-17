//! PyO3 bindings for TreeBoost GBDT
//!
//! This module provides Python wrappers around the core Rust GBDT implementation,
//! exposing a Python-friendly API while maintaining Rust performance.
//!
//! # Example
//!
//! ```python
//! import numpy as np
//! from treeboost import GBDTConfig, GBDTModel
//!
//! # Create configuration with builder pattern (recommended)
//! config = (
//!     GBDTConfig()
//!     .with_num_rounds(100)
//!     .with_max_depth(6)
//!     .with_learning_rate(0.1)
//!     .with_backend("cuda")
//! )
//!
//! # Train model (features: 2D array, targets: 1D array)
//! model = GBDTModel.train(features, targets, config)
//!
//! # Predict
//! predictions = model.predict(features)
//!
//! # Save/load model
//! model.save("model.rkyv")
//! loaded = GBDTModel.load("model.rkyv")
//! ```

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use crate::backend::{BackendType, GpuMode};
use crate::booster::{GBDTConfig, GBDTModel, GbdtPreset, LossType};
use crate::serialize;
use crate::tree::MonotonicConstraint;
use crate::tuner::ModelFormat;

/// Extract features from a 2D numpy array (f64 or f32) into a flat Vec<f64>
///
/// Returns (num_features, flattened_row_major_data)
fn extract_features_array<'py>(features: &Bound<'py, PyAny>) -> PyResult<(usize, Vec<f64>)> {
    // Try f64 first (most common for numpy), then f32
    if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f64>>() {
        let arr = arr.as_array();
        let num_rows = arr.nrows();
        let num_cols = arr.ncols();
        let mut raw = Vec::with_capacity(num_rows * num_cols);
        for row in arr.rows() {
            raw.extend(row.iter().copied());
        }
        Ok((num_cols, raw))
    } else if let Ok(arr) = features.extract::<PyReadonlyArray2<'py, f32>>() {
        let arr = arr.as_array();
        let num_rows = arr.nrows();
        let num_cols = arr.ncols();
        let mut raw = Vec::with_capacity(num_rows * num_cols);
        for row in arr.rows() {
            raw.extend(row.iter().map(|&v| v as f64));
        }
        Ok((num_cols, raw))
    } else {
        Err(PyValueError::new_err(
            "features must be a 2D numpy array of float32 or float64",
        ))
    }
}

/// Validate that extracted features match the expected count
fn validate_feature_count(actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "Expected {} features, got {}",
            expected, actual
        )));
    }
    Ok(())
}

/// Parse a model format string into ModelFormat enum
fn parse_model_format(format: &str) -> PyResult<ModelFormat> {
    match format.to_lowercase().as_str() {
        "rkyv" => Ok(ModelFormat::Rkyv),
        "bincode" | "bin" => Ok(ModelFormat::Bincode),
        _ => Err(PyValueError::new_err("format must be 'rkyv' or 'bincode'")),
    }
}

/// Python wrapper for monotonic constraint
///
/// Controls the direction of feature effects on predictions.
///
/// Example:
/// ```python
/// from treeboost import MonotonicConstraint
///
/// # Feature 0: increasing, Feature 1: decreasing, Feature 2: none
/// constraints = [
///     MonotonicConstraint.increasing(),
///     MonotonicConstraint.decreasing(),
///     MonotonicConstraint.none()
/// ]
/// config = config.with_monotonic_constraints(constraints)
/// ```
#[pyclass(from_py_object, name = "MonotonicConstraint", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyMonotonicConstraint {
    inner: MonotonicConstraint,
}

#[pymethods]
impl PyMonotonicConstraint {
    /// Create an increasing constraint (larger values -> larger predictions)
    #[staticmethod]
    fn increasing() -> Self {
        Self {
            inner: MonotonicConstraint::Increasing,
        }
    }

    /// Create a decreasing constraint (larger values -> smaller predictions)
    #[staticmethod]
    fn decreasing() -> Self {
        Self {
            inner: MonotonicConstraint::Decreasing,
        }
    }

    /// Create no constraint (no monotonic restriction)
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: MonotonicConstraint::None,
        }
    }

    /// Check if this is an increasing constraint
    #[getter]
    fn is_increasing(&self) -> bool {
        matches!(self.inner, MonotonicConstraint::Increasing)
    }

    /// Check if this is a decreasing constraint
    #[getter]
    fn is_decreasing(&self) -> bool {
        matches!(self.inner, MonotonicConstraint::Decreasing)
    }

    /// Check if this is no constraint
    #[getter]
    fn is_none(&self) -> bool {
        matches!(self.inner, MonotonicConstraint::None)
    }

    /// Convert to integer value (1 = increasing, -1 = decreasing, 0 = none)
    #[allow(clippy::wrong_self_convention)] // reason: name fixed by Python API
    fn to_int(&self) -> i32 {
        match self.inner {
            MonotonicConstraint::Increasing => 1,
            MonotonicConstraint::Decreasing => -1,
            MonotonicConstraint::None => 0,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            MonotonicConstraint::Increasing => "MonotonicConstraint.increasing()",
            MonotonicConstraint::Decreasing => "MonotonicConstraint.decreasing()",
            MonotonicConstraint::None => "MonotonicConstraint.none()",
        }
    }
}

impl From<MonotonicConstraint> for PyMonotonicConstraint {
    fn from(mc: MonotonicConstraint) -> Self {
        Self { inner: mc }
    }
}

impl From<PyMonotonicConstraint> for MonotonicConstraint {
    fn from(pmc: PyMonotonicConstraint) -> Self {
        pmc.inner
    }
}

/// Python wrapper for GBDT training configuration
#[pyclass(from_py_object, name = "GBDTConfig")]
#[derive(Clone)]
pub struct PyGBDTConfig {
    inner: GBDTConfig,
}

#[pymethods]
impl PyGBDTConfig {
    /// Create a new configuration with default values
    #[new]
    fn new() -> Self {
        Self {
            inner: GBDTConfig::default(),
        }
    }

    /// Create a new configuration from a preset name.
    ///
    /// Valid presets: standard, speed, accuracy, smalldata, largedata, conformal.
    #[staticmethod]
    fn preset(preset: &str) -> PyResult<Self> {
        let preset = match preset.to_lowercase().as_str() {
            "standard" => GbdtPreset::Standard,
            "speed" => GbdtPreset::Speed,
            "accuracy" => GbdtPreset::Accuracy,
            "smalldata" | "small_data" => GbdtPreset::SmallData,
            "largedata" | "large_data" => GbdtPreset::LargeData,
            "conformal" => GbdtPreset::Conformal,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown preset (use: standard, speed, accuracy, smalldata, largedata, conformal)",
                ));
            }
        };
        Ok(Self {
            inner: GBDTConfig::default().with_preset(preset),
        })
    }

    /// Return a new config with the given preset applied.
    ///
    /// Valid presets: standard, speed, accuracy, smalldata, largedata, conformal.
    fn with_preset(&self, preset: &str) -> PyResult<Self> {
        let mut inner = self.inner.clone();
        inner = match preset.to_lowercase().as_str() {
            "standard" => inner.with_preset(GbdtPreset::Standard),
            "speed" => inner.with_preset(GbdtPreset::Speed),
            "accuracy" => inner.with_preset(GbdtPreset::Accuracy),
            "smalldata" | "small_data" => inner.with_preset(GbdtPreset::SmallData),
            "largedata" | "large_data" => inner.with_preset(GbdtPreset::LargeData),
            "conformal" => inner.with_preset(GbdtPreset::Conformal),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown preset (use: standard, speed, accuracy, smalldata, largedata, conformal)",
                ));
            }
        };
        Ok(Self { inner })
    }

    // Ensemble parameters

    /// Number of boosting rounds (trees)
    #[getter]
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds
    }

    #[setter]
    fn set_num_rounds(&mut self, value: usize) {
        self.inner.num_rounds = value;
    }

    /// Learning rate (optimization step size for trees)
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f32) {
        self.inner.learning_rate = value;
    }

    // Tree parameters

    /// Maximum depth of each tree
    #[getter]
    fn max_depth(&self) -> usize {
        self.inner.max_depth
    }

    #[setter]
    fn set_max_depth(&mut self, value: usize) {
        self.inner.max_depth = value;
    }

    /// Maximum number of leaves per tree
    #[getter]
    fn max_leaves(&self) -> usize {
        self.inner.max_leaves
    }

    #[setter]
    fn set_max_leaves(&mut self, value: usize) {
        self.inner.max_leaves = value;
    }

    /// Minimum samples required in a leaf
    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }

    #[setter]
    fn set_min_samples_leaf(&mut self, value: usize) {
        self.inner.min_samples_leaf = value;
    }

    /// Minimum hessian sum required in a leaf
    #[getter]
    fn min_hessian_leaf(&self) -> f32 {
        self.inner.min_hessian_leaf
    }

    #[setter]
    fn set_min_hessian_leaf(&mut self, value: f32) {
        self.inner.min_hessian_leaf = value;
    }

    /// Minimum gain to make a split
    #[getter]
    fn min_gain(&self) -> f32 {
        self.inner.min_gain
    }

    #[setter]
    fn set_min_gain(&mut self, value: f32) {
        self.inner.min_gain = value;
    }

    // Regularization

    /// L2 regularization (lambda)
    #[getter]
    fn lambda_(&self) -> f32 {
        self.inner.lambda
    }

    #[setter]
    fn set_lambda(&mut self, value: f32) {
        self.inner.lambda = value;
    }

    /// Shannon Entropy regularization weight (beta)
    #[getter]
    fn entropy_weight(&self) -> f32 {
        self.inner.entropy_weight
    }

    #[setter]
    fn set_entropy_weight(&mut self, value: f32) {
        self.inner.entropy_weight = value;
    }

    // Loss function

    /// Set loss function to MSE
    fn use_mse_loss(&mut self) {
        self.inner.loss_type = LossType::Mse;
    }

    /// Set loss function to Pseudo-Huber with given delta
    fn use_pseudo_huber_loss(&mut self, delta: f32) {
        self.inner.loss_type = LossType::PseudoHuber { delta };
    }

    /// Set loss function to Binary Log Loss (for binary classification)
    ///
    /// Uses sigmoid activation for probability output.
    /// Targets should be 0 or 1.
    fn use_binary_logloss(&mut self) {
        self.inner.loss_type = LossType::BinaryLogLoss;
    }

    /// Set loss function to Multi-class Log Loss (for multi-class classification)
    ///
    /// Uses softmax activation for probability output.
    /// Targets should be class indices: 0, 1, 2, ..., num_classes-1.
    ///
    /// This trains K trees per round (one per class) and combines predictions
    /// via softmax for final class probabilities.
    ///
    /// Args:
    ///     num_classes: Number of classes (K), must be >= 2
    fn use_multiclass_logloss(&mut self, num_classes: usize) -> PyResult<()> {
        if num_classes < 2 {
            return Err(PyValueError::new_err("num_classes must be >= 2"));
        }
        self.inner.loss_type = LossType::MultiClassLogLoss { num_classes };
        Ok(())
    }

    // Subsampling

    /// Row subsampling ratio (0.0-1.0)
    #[getter]
    fn subsample(&self) -> f32 {
        self.inner.subsample
    }

    #[setter]
    fn set_subsample(&mut self, value: f32) {
        self.inner.subsample = value;
    }

    /// Column subsampling ratio (0.0-1.0)
    #[getter]
    fn colsample(&self) -> f32 {
        self.inner.colsample
    }

    #[setter]
    fn set_colsample(&mut self, value: f32) {
        self.inner.colsample = value;
    }

    // Binning

    /// Number of histogram bins
    #[getter]
    fn num_bins(&self) -> usize {
        self.inner.num_bins
    }

    #[setter]
    fn set_num_bins(&mut self, value: usize) {
        self.inner.num_bins = value;
    }

    // Conformal prediction

    /// Calibration set ratio for conformal prediction (0.0 to disable)
    #[getter]
    fn calibration_ratio(&self) -> f32 {
        self.inner.calibration_ratio
    }

    #[setter]
    fn set_calibration_ratio(&mut self, value: f32) {
        self.inner.calibration_ratio = value;
    }

    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    #[getter]
    fn conformal_quantile(&self) -> f32 {
        self.inner.conformal_quantile
    }

    #[setter]
    fn set_conformal_quantile(&mut self, value: f32) {
        self.inner.conformal_quantile = value;
    }

    // Early stopping

    /// Number of rounds with no improvement before stopping (0 to disable)
    #[getter]
    fn early_stopping_rounds(&self) -> usize {
        self.inner.early_stopping_rounds
    }

    #[setter]
    fn set_early_stopping_rounds(&mut self, value: usize) {
        self.inner.early_stopping_rounds = value;
    }

    /// Ratio of data to use for validation (0.0 to disable early stopping)
    #[getter]
    fn validation_ratio(&self) -> f32 {
        self.inner.validation_ratio
    }

    #[setter]
    fn set_validation_ratio(&mut self, value: f32) {
        self.inner.validation_ratio = value;
    }

    // Performance optimizations

    /// Use parallel prediction via Rayon (default: true)
    #[getter]
    fn parallel_prediction(&self) -> bool {
        self.inner.parallel_prediction
    }

    #[setter]
    fn set_parallel_prediction(&mut self, value: bool) {
        self.inner.parallel_prediction = value;
    }

    /// Reorder columns by feature importance for cache locality (default: true)
    #[getter]
    fn column_reordering(&self) -> bool {
        self.inner.column_reordering
    }

    #[setter]
    fn set_column_reordering(&mut self, value: bool) {
        self.inner.column_reordering = value;
    }

    /// Use 4-bit packing for low-cardinality features (default: true)
    #[getter]
    fn packed_dataset(&self) -> bool {
        self.inner.packed_dataset
    }

    #[setter]
    fn set_packed_dataset(&mut self, value: bool) {
        self.inner.packed_dataset = value;
    }

    /// Use parallel gradient computation (default: false)
    /// Experimental: may not provide stable speedups, benchmark before enabling
    #[getter]
    fn parallel_gradient(&self) -> bool {
        self.inner.parallel_gradient
    }

    #[setter]
    fn set_parallel_gradient(&mut self, value: bool) {
        self.inner.parallel_gradient = value;
    }

    // Backend selection

    /// Enable GPU subgroup operations (default: True for Python)
    ///
    /// Subgroups can reduce atomic contention in GPU histogram building.
    /// Benchmarks show minimal benefit on modern NVIDIA GPUs (~1.0x),
    /// but may help on older AMD or Intel GPUs.
    #[getter]
    fn use_gpu_subgroups(&self) -> bool {
        self.inner.use_gpu_subgroups
    }

    #[setter]
    fn set_use_gpu_subgroups(&mut self, value: bool) {
        self.inner.use_gpu_subgroups = value;
    }

    /// Get current backend type as string
    #[getter]
    fn backend(&self) -> &'static str {
        match self.inner.backend_type {
            BackendType::Auto => "auto",
            BackendType::Scalar => "cpu",
            BackendType::Wgpu => "wgpu",
            BackendType::Avx512 => "avx512",
            BackendType::Sve2 => "sve2",
            BackendType::Cuda => "cuda",
            BackendType::Rocm => "rocm",
            BackendType::Metal => "metal",
        }
    }

    /// Set the backend for histogram building
    ///
    /// Args:
    ///     value: One of "auto", "cpu", "gpu", "wgpu", "cuda"
    ///         - "auto": Select best available (CUDA > WGPU > CPU)
    ///         - "cpu": Force CPU (AVX2/NEON optimized)
    ///         - "gpu": Select best GPU (CUDA > WGPU), same as "auto"
    ///         - "wgpu": Force WGPU (all GPUs via Vulkan/Metal/DX12)
    ///         - "cuda": Force CUDA (NVIDIA only, fastest)
    ///
    /// Example:
    ///     config.backend = "cuda"
    #[setter]
    fn set_backend(&mut self, value: &str) -> PyResult<()> {
        self.inner.backend_type = match value.to_lowercase().as_str() {
            "auto" | "gpu" => BackendType::Auto,  // gpu = auto-select best GPU (CUDA > WGPU)
            "cpu" | "scalar" => BackendType::Scalar,
            "wgpu" => BackendType::Wgpu,
            "cuda" => BackendType::Cuda,
            "rocm" => BackendType::Rocm,
            "metal" => BackendType::Metal,
            _ => return Err(PyValueError::new_err(
                "backend must be one of: 'auto' (or 'gpu'), 'cpu' (or 'scalar'), 'wgpu', 'cuda', 'rocm', 'metal'"
            )),
        };
        Ok(())
    }

    /// Get current GPU mode as string
    #[getter]
    fn gpu_mode(&self) -> &'static str {
        match self.inner.gpu_mode {
            GpuMode::Auto => "auto",
            GpuMode::Hybrid => "hybrid",
            GpuMode::Full => "full",
        }
    }

    /// Set the GPU execution mode
    ///
    /// Args:
    ///     value: One of "auto", "hybrid", "full"
    ///         - "auto": Select optimal mode per backend (CUDA→Full, WGPU→Hybrid)
    ///         - "hybrid": GPU histogram + CPU partition (best-first tree growth)
    ///         - "full": Full GPU pipeline (level-wise tree growth)
    ///
    /// Example:
    ///     config.gpu_mode = "full"
    #[setter]
    fn set_gpu_mode(&mut self, value: &str) -> PyResult<()> {
        self.inner.gpu_mode = match value.to_lowercase().as_str() {
            "auto" => GpuMode::Auto,
            "hybrid" => GpuMode::Hybrid,
            "full" => GpuMode::Full,
            _ => {
                return Err(PyValueError::new_err(
                    "gpu_mode must be one of: 'auto', 'hybrid', 'full'",
                ))
            }
        };
        Ok(())
    }

    // Monotonic constraints

    /// Set monotonic constraints for features
    ///
    /// Args:
    ///     constraints: List of constraint values per feature
    ///         - 1 = Increasing (larger values -> larger predictions)
    ///         - -1 = Decreasing (larger values -> smaller predictions)
    ///         - 0 = None (no constraint)
    ///
    /// Example:
    ///     config.set_monotonic_constraints([1, -1, 0])  # Feature 0: inc, Feature 1: dec, Feature 2: none
    fn set_monotonic_constraints(&mut self, constraints: Vec<i32>) -> PyResult<()> {
        let parsed: Result<Vec<MonotonicConstraint>, _> = constraints
            .into_iter()
            .map(|c| match c {
                1 => Ok(MonotonicConstraint::Increasing),
                -1 => Ok(MonotonicConstraint::Decreasing),
                0 => Ok(MonotonicConstraint::None),
                _ => Err(PyValueError::new_err(
                    "Constraint must be 1 (increasing), -1 (decreasing), or 0 (none)",
                )),
            })
            .collect();

        self.inner.monotonic_constraints = parsed?;
        Ok(())
    }

    // Era splitting (Directional Era Splitting)

    /// Enable Directional Era Splitting (DES)
    ///
    /// When enabled, splits are only accepted if ALL eras agree on the
    /// split direction. This filters out spurious correlations that work
    /// in some time periods but not others.
    ///
    /// Requires passing era_indices to train_with_eras().
    #[getter]
    fn era_splitting(&self) -> bool {
        self.inner.era_splitting
    }

    #[setter]
    fn set_era_splitting(&mut self, value: bool) {
        self.inner.era_splitting = value;
    }

    // Interaction constraints

    /// Set feature interaction constraints
    ///
    /// Features in the same group can interact (appear together in a tree path).
    /// Features in different groups cannot be used together.
    /// Features not in any group can interact with all features.
    ///
    /// Args:
    ///     groups: List of feature index groups, e.g., [[0, 1, 2], [3, 4]]
    ///
    /// Example:
    ///     config.set_interaction_groups([[0, 1, 2], [3, 4]])
    fn set_interaction_groups(&mut self, groups: Vec<Vec<usize>>) {
        self.inner.interaction_groups = groups;
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTConfig(num_rounds={}, learning_rate={}, max_depth={}, max_leaves={}, backend='{}', gpu_mode='{}')",
            self.inner.num_rounds,
            self.inner.learning_rate,
            self.inner.max_depth,
            self.inner.max_leaves,
            self.backend(),
            self.gpu_mode()
        )
    }

    // ========== Builder Pattern Methods ==========
    // All return Self for method chaining (immutable builder pattern)

    /// Set number of boosting rounds (trees)
    fn with_num_rounds(&self, value: usize) -> PyResult<Self> {
        if value == 0 {
            return Err(PyValueError::new_err("num_rounds must be >= 1"));
        }
        let mut new = self.clone();
        new.inner.num_rounds = value;
        Ok(new)
    }

    /// Set learning rate (shrinkage)
    fn with_learning_rate(&self, value: f32) -> PyResult<Self> {
        if value <= 0.0 || value > 1.0 {
            return Err(PyValueError::new_err("learning_rate must be in (0.0, 1.0]"));
        }
        let mut new = self.clone();
        new.inner.learning_rate = value;
        Ok(new)
    }

    /// Set maximum depth of each tree
    fn with_max_depth(&self, value: usize) -> PyResult<Self> {
        if value == 0 {
            return Err(PyValueError::new_err("max_depth must be >= 1"));
        }
        let mut new = self.clone();
        new.inner.max_depth = value;
        Ok(new)
    }

    /// Set maximum number of leaves per tree
    fn with_max_leaves(&self, value: usize) -> PyResult<Self> {
        if value < 2 {
            return Err(PyValueError::new_err("max_leaves must be >= 2"));
        }
        let mut new = self.clone();
        new.inner.max_leaves = value;
        Ok(new)
    }

    /// Set minimum samples required in a leaf
    fn with_min_samples_leaf(&self, value: usize) -> PyResult<Self> {
        if value == 0 {
            return Err(PyValueError::new_err("min_samples_leaf must be >= 1"));
        }
        let mut new = self.clone();
        new.inner.min_samples_leaf = value;
        Ok(new)
    }

    /// Set minimum hessian sum required in a leaf
    fn with_min_hessian_leaf(&self, value: f32) -> PyResult<Self> {
        if value < 0.0 {
            return Err(PyValueError::new_err("min_hessian_leaf must be >= 0.0"));
        }
        let mut new = self.clone();
        new.inner.min_hessian_leaf = value;
        Ok(new)
    }

    /// Set minimum gain to make a split
    fn with_min_gain(&self, value: f32) -> PyResult<Self> {
        if value < 0.0 {
            return Err(PyValueError::new_err("min_gain must be >= 0.0"));
        }
        let mut new = self.clone();
        new.inner.min_gain = value;
        Ok(new)
    }

    /// Set L2 regularization (lambda)
    fn with_lambda(&self, value: f32) -> PyResult<Self> {
        if value < 0.0 {
            return Err(PyValueError::new_err("lambda must be >= 0.0"));
        }
        let mut new = self.clone();
        new.inner.lambda = value;
        Ok(new)
    }

    /// Set Shannon Entropy regularization weight (beta)
    fn with_entropy_weight(&self, value: f32) -> PyResult<Self> {
        if value < 0.0 {
            return Err(PyValueError::new_err("entropy_weight must be >= 0.0"));
        }
        let mut new = self.clone();
        new.inner.entropy_weight = value;
        Ok(new)
    }

    /// Set row subsampling ratio (0.0-1.0)
    fn with_subsample(&self, value: f32) -> PyResult<Self> {
        if value <= 0.0 || value > 1.0 {
            return Err(PyValueError::new_err("subsample must be in (0.0, 1.0]"));
        }
        let mut new = self.clone();
        new.inner.subsample = value;
        Ok(new)
    }

    /// Set column subsampling ratio (0.0-1.0)
    fn with_colsample(&self, value: f32) -> PyResult<Self> {
        if value <= 0.0 || value > 1.0 {
            return Err(PyValueError::new_err("colsample must be in (0.0, 1.0]"));
        }
        let mut new = self.clone();
        new.inner.colsample = value;
        Ok(new)
    }

    /// Set number of histogram bins
    fn with_num_bins(&self, value: usize) -> PyResult<Self> {
        if !(2..=255).contains(&value) {
            return Err(PyValueError::new_err("num_bins must be in [2, 255]"));
        }
        let mut new = self.clone();
        new.inner.num_bins = value;
        Ok(new)
    }

    /// Set calibration set ratio for conformal prediction (0.0 to disable)
    fn with_calibration_ratio(&self, value: f32) -> PyResult<Self> {
        if !(0.0..1.0).contains(&value) {
            return Err(PyValueError::new_err(
                "calibration_ratio must be in [0.0, 1.0)",
            ));
        }
        let mut new = self.clone();
        new.inner.calibration_ratio = value;
        Ok(new)
    }

    /// Set conformal prediction quantile (e.g., 0.9 for 90% coverage)
    fn with_conformal_quantile(&self, value: f32) -> PyResult<Self> {
        if value <= 0.0 || value >= 1.0 {
            return Err(PyValueError::new_err(
                "conformal_quantile must be in (0.0, 1.0)",
            ));
        }
        let mut new = self.clone();
        new.inner.conformal_quantile = value;
        Ok(new)
    }

    /// Set early stopping rounds (0 to disable)
    fn with_early_stopping_rounds(&self, value: usize) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.early_stopping_rounds = value;
        Ok(new)
    }

    /// Set validation ratio for early stopping (0.0 to disable)
    fn with_validation_ratio(&self, value: f32) -> PyResult<Self> {
        if !(0.0..1.0).contains(&value) {
            return Err(PyValueError::new_err(
                "validation_ratio must be in [0.0, 1.0)",
            ));
        }
        let mut new = self.clone();
        new.inner.validation_ratio = value;
        Ok(new)
    }

    /// Enable/disable parallel prediction
    fn with_parallel_prediction(&self, value: bool) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.parallel_prediction = value;
        Ok(new)
    }

    /// Enable/disable column reordering for cache locality
    fn with_column_reordering(&self, value: bool) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.column_reordering = value;
        Ok(new)
    }

    /// Enable/disable 4-bit packing for low-cardinality features
    fn with_packed_dataset(&self, value: bool) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.packed_dataset = value;
        Ok(new)
    }

    /// Enable/disable parallel gradient computation
    fn with_parallel_gradient(&self, value: bool) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.parallel_gradient = value;
        Ok(new)
    }

    /// Enable/disable GPU subgroup operations
    fn with_gpu_subgroups(&self, value: bool) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.use_gpu_subgroups = value;
        Ok(new)
    }

    /// Set backend type ("auto", "cpu", "gpu", "wgpu", "cuda", "rocm", "metal")
    fn with_backend(&self, value: &str) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.backend_type = match value.to_lowercase().as_str() {
            "auto" | "gpu" => BackendType::Auto,
            "cpu" | "scalar" => BackendType::Scalar,
            "wgpu" => BackendType::Wgpu,
            "cuda" => BackendType::Cuda,
            "rocm" => BackendType::Rocm,
            "metal" => BackendType::Metal,
            _ => return Err(PyValueError::new_err(
                "backend must be one of: 'auto' (or 'gpu'), 'cpu' (or 'scalar'), 'wgpu', 'cuda', 'rocm', 'metal'"
            )),
        };
        Ok(new)
    }

    /// Set GPU execution mode ("auto", "hybrid", "full")
    fn with_gpu_mode(&self, value: &str) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.gpu_mode = match value.to_lowercase().as_str() {
            "auto" => GpuMode::Auto,
            "hybrid" => GpuMode::Hybrid,
            "full" => GpuMode::Full,
            _ => {
                return Err(PyValueError::new_err(
                    "gpu_mode must be one of: 'auto', 'hybrid', 'full'",
                ))
            }
        };
        Ok(new)
    }

    /// Enable/disable Directional Era Splitting
    fn with_era_splitting(&self, value: bool) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.era_splitting = value;
        Ok(new)
    }

    /// Use MSE loss function
    fn with_mse_loss(&self) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.loss_type = LossType::Mse;
        Ok(new)
    }

    /// Use Pseudo-Huber loss function with given delta
    fn with_pseudo_huber_loss(&self, delta: f32) -> PyResult<Self> {
        if delta <= 0.0 {
            return Err(PyValueError::new_err("delta must be > 0.0"));
        }
        let mut new = self.clone();
        new.inner.loss_type = LossType::PseudoHuber { delta };
        Ok(new)
    }

    /// Use Binary Log Loss for binary classification
    fn with_binary_logloss(&self) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.loss_type = LossType::BinaryLogLoss;
        Ok(new)
    }

    /// Use Multi-class Log Loss for multi-class classification
    fn with_multiclass_logloss(&self, num_classes: usize) -> PyResult<Self> {
        if num_classes < 2 {
            return Err(PyValueError::new_err("num_classes must be >= 2"));
        }
        let mut new = self.clone();
        new.inner.loss_type = LossType::MultiClassLogLoss { num_classes };
        Ok(new)
    }

    /// Set monotonic constraints for features (using integers)
    ///
    /// Args:
    ///     constraints: List of constraint values per feature
    ///         - 1 = Increasing, -1 = Decreasing, 0 = None
    ///
    /// Note: Prefer with_constraints() with MonotonicConstraint enum for clearer code
    fn with_monotonic_constraints(&self, constraints: Vec<i32>) -> PyResult<Self> {
        let parsed: Result<Vec<MonotonicConstraint>, _> = constraints
            .into_iter()
            .map(|c| match c {
                1 => Ok(MonotonicConstraint::Increasing),
                -1 => Ok(MonotonicConstraint::Decreasing),
                0 => Ok(MonotonicConstraint::None),
                _ => Err(PyValueError::new_err(
                    "Constraint must be 1 (increasing), -1 (decreasing), or 0 (none)",
                )),
            })
            .collect();

        let mut new = self.clone();
        new.inner.monotonic_constraints = parsed?;
        Ok(new)
    }

    /// Set monotonic constraints for features (using enum)
    ///
    /// Args:
    ///     constraints: List of MonotonicConstraint per feature
    ///
    /// Example:
    ///     config.with_constraints([
    ///         MonotonicConstraint.increasing(),
    ///         MonotonicConstraint.decreasing(),
    ///         MonotonicConstraint.none()
    ///     ])
    fn with_constraints(&self, constraints: Vec<PyMonotonicConstraint>) -> PyResult<Self> {
        let parsed: Vec<MonotonicConstraint> = constraints.into_iter().map(|c| c.into()).collect();

        let mut new = self.clone();
        new.inner.monotonic_constraints = parsed;
        Ok(new)
    }

    /// Set feature interaction groups
    fn with_interaction_groups(&self, groups: Vec<Vec<usize>>) -> PyResult<Self> {
        let mut new = self.clone();
        new.inner.interaction_groups = groups;
        Ok(new)
    }
}

// Internal methods for use by other Python binding modules
impl PyGBDTConfig {
    /// Get reference to inner config (for use by tuner module)
    pub fn inner(&self) -> &GBDTConfig {
        &self.inner
    }

    /// Create from inner config (for use by tuner module)
    pub fn from_inner(config: GBDTConfig) -> Self {
        Self { inner: config }
    }
}

/// Python wrapper for trained GBDT model
#[pyclass(name = "GBDTModel")]
pub struct PyGBDTModel {
    model: GBDTModel,
}

#[pymethods]
impl PyGBDTModel {
    /// Train a GBDT model from numpy arrays
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///     targets: 1D numpy array of shape (n_samples,)
    ///     config: GBDTConfig instance
    ///     feature_names: Optional list of feature names
    ///     output_dir: Optional directory to save model and config.json
    ///                 If provided, saves model.rkyv and config.json for reproducibility
    ///
    /// Returns:
    ///     Trained GBDTModel
    ///
    /// Example:
    ///     # Train and save automatically
    ///     model = GBDTModel.train(features, targets, config, output_dir="my_model")
    ///     # Creates: my_model/model.rkyv and my_model/config.json
    #[staticmethod]
    #[pyo3(signature = (features, targets, config, feature_names=None, output_dir=None))]
    fn train<'py>(
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        targets: PyReadonlyArray1<'py, f32>,
        config: &PyGBDTConfig,
        feature_names: Option<Vec<String>>,
        output_dir: Option<String>,
    ) -> PyResult<Self> {
        let features_arr = features.as_array();
        let targets_arr = targets.as_array();

        let num_rows = features_arr.nrows();
        let num_features = features_arr.ncols();

        // Convert to row-major flat Vec<f32> for Rust high-level API
        let mut features_flat: Vec<f32> = Vec::with_capacity(num_rows * num_features);
        for row in features_arr.rows() {
            features_flat.extend(row.iter().copied());
        }

        let targets_vec: Vec<f32> = targets_arr.to_vec();

        // Train model using high-level Rust API (release GIL during training)
        // Binning is now done in Rust with Rayon parallelization
        let model = py
            .detach(|| {
                GBDTModel::train(
                    &features_flat,
                    num_features,
                    &targets_vec,
                    config.inner.clone(),
                    feature_names,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // If output_dir provided, save model and config
        if let Some(ref dir) = output_dir {
            model
                .save_to_directory(dir, &config.inner, &[ModelFormat::Rkyv])
                .map_err(|e| {
                    PyIOError::new_err(format!("Failed to save to output directory: {}", e))
                })?;
        }

        Ok(Self { model })
    }

    /// Train a GBDT model with Directional Era Splitting (DES)
    ///
    /// Era splitting filters out spurious correlations by requiring all eras
    /// to agree on split direction. This is useful for time-series or financial
    /// data where patterns may not generalize across time periods.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///     targets: 1D numpy array of shape (n_samples,)
    ///     era_indices: 1D numpy array of era indices (uint16), shape (n_samples,)
    ///     config: GBDTConfig instance (must have era_splitting=True)
    ///     feature_names: Optional list of feature names
    ///
    /// Returns:
    ///     Trained GBDTModel
    ///
    /// Example:
    ///     config = GBDTConfig()
    ///     config.era_splitting = True
    ///     model = GBDTModel.train_with_eras(features, targets, era_indices, config)
    #[staticmethod]
    #[pyo3(signature = (features, targets, era_indices, config, feature_names=None))]
    fn train_with_eras<'py>(
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        targets: PyReadonlyArray1<'py, f32>,
        era_indices: PyReadonlyArray1<'py, u16>,
        config: &PyGBDTConfig,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let features_arr = features.as_array();
        let targets_arr = targets.as_array();
        let era_indices_arr = era_indices.as_array();

        let num_rows = features_arr.nrows();
        let num_features = features_arr.ncols();

        // Validate era_indices length
        if era_indices_arr.len() != num_rows {
            return Err(PyValueError::new_err(format!(
                "era_indices length {} doesn't match number of rows {}",
                era_indices_arr.len(),
                num_rows
            )));
        }

        // Validate era_splitting is enabled
        if !config.inner.era_splitting {
            return Err(PyValueError::new_err(
                "era_splitting must be True in config when using train_with_eras",
            ));
        }

        // Convert to row-major flat Vec<f32> for Rust high-level API
        let mut features_flat: Vec<f32> = Vec::with_capacity(num_rows * num_features);
        for row in features_arr.rows() {
            features_flat.extend(row.iter().copied());
        }

        let targets_vec: Vec<f32> = targets_arr.to_vec();
        let era_indices_vec: Vec<u16> = era_indices_arr.to_vec();

        // Train model using high-level Rust API (release GIL during training)
        let model = py
            .detach(|| {
                GBDTModel::train_with_eras(
                    &features_flat,
                    num_features,
                    &targets_vec,
                    &era_indices_vec,
                    config.inner.clone(),
                    feature_names,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { model })
    }

    /// Predict for new data
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     1D numpy array of predictions
    #[pyo3(signature = (features))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (num_features, raw_features) = extract_features_array(features)?;
        validate_feature_count(num_features, self.model.num_features())?;

        // Predict using raw values (release GIL)
        let predictions = py.detach(|| self.model.predict_raw(&raw_features));

        Ok(PyArray1::from_vec(py, predictions))
    }

    /// Predict with conformal intervals
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     Tuple of (predictions, lower_bounds, upper_bounds) as numpy arrays
    #[pyo3(signature = (features))]
    #[allow(clippy::type_complexity)] // reason: complex return tuple kept inline
    fn predict_with_intervals<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let (num_features, raw_features) = extract_features_array(features)?;
        validate_feature_count(num_features, self.model.num_features())?;

        // Predict with intervals (release GIL)
        let (preds, lower, upper) =
            py.detach(|| self.model.predict_raw_with_intervals(&raw_features));

        Ok((
            PyArray1::from_vec(py, preds),
            PyArray1::from_vec(py, lower),
            PyArray1::from_vec(py, upper),
        ))
    }

    /// Predict class probabilities (for binary classification)
    ///
    /// Applies sigmoid to raw predictions to get probabilities in [0, 1].
    /// Only meaningful when trained with `use_binary_logloss()`.
    ///
    /// For multi-class models, use `predict_proba_multiclass()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     1D numpy array of probabilities (probability of class 1)
    #[pyo3(signature = (features))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Extract and validate features first (most specific error)
        let (num_features, raw_features) = extract_features_array(features)?;
        validate_feature_count(num_features, self.model.num_features())?;

        // Then check model type
        if self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_proba() is for binary classification only. \
                 For multi-class models, use predict_proba_multiclass() instead.",
            ));
        }

        // Predict probabilities (release GIL)
        let proba = py.detach(|| self.model.predict_proba_raw(&raw_features));

        Ok(PyArray1::from_vec(py, proba))
    }

    /// Predict class labels (for binary classification)
    ///
    /// Applies sigmoid to raw predictions and thresholds.
    /// Only meaningful when trained with `use_binary_logloss()`.
    ///
    /// For multi-class models, use `predict_class_multiclass()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///     threshold: Classification threshold (default 0.5)
    ///
    /// Returns:
    ///     1D numpy array of class labels (0 or 1)
    #[pyo3(signature = (features, threshold = 0.5))]
    fn predict_class<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
        threshold: f32,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        // Extract and validate features first (most specific error)
        let (num_features, raw_features) = extract_features_array(features)?;
        validate_feature_count(num_features, self.model.num_features())?;

        // Then check model type
        if self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_class() is for binary classification only. \
                 For multi-class models, use predict_class_multiclass() instead.",
            ));
        }

        // Predict classes (release GIL)
        let classes = py.detach(|| self.model.predict_class_raw(&raw_features, threshold));

        Ok(PyArray1::from_vec(py, classes))
    }

    // Multi-class classification methods

    /// Check if this is a multi-class model
    #[getter]
    fn is_multiclass(&self) -> bool {
        self.model.is_multiclass()
    }

    /// Get number of classes (0 for regression/binary)
    #[getter]
    fn num_classes(&self) -> usize {
        self.model.get_num_classes()
    }

    /// Predict class probabilities for multi-class classification
    ///
    /// Applies softmax to raw predictions to get probabilities for each class.
    /// Only meaningful when trained with `use_multiclass_logloss()`.
    ///
    /// For binary classification, use `predict_proba()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     2D numpy array of shape (n_samples, n_classes) with probabilities
    #[pyo3(signature = (features))]
    fn predict_proba_multiclass<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        use numpy::PyArray2;

        // Extract and validate features first (most specific error)
        let (num_features, raw_features) = extract_features_array(features)?;
        validate_feature_count(num_features, self.model.num_features())?;

        // Then check model type
        if !self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_proba_multiclass() is for multi-class models only. \
                 For binary classification, use predict_proba() instead.",
            ));
        }

        // Use the raw prediction method (no binning needed, release GIL)
        let proba = py.detach(|| self.model.predict_proba_multiclass_raw(&raw_features));

        // Validate array is not jagged before conversion
        if proba.is_empty() {
            return Err(PyValueError::new_err("No predictions returned"));
        }
        let expected_cols = proba[0].len();
        if !proba.iter().all(|row| row.len() == expected_cols) {
            return Err(PyValueError::new_err(
                "Internal error: jagged probability array returned",
            ));
        }

        // Convert Vec<Vec<f32>> to 2D numpy array via nested vec
        PyArray2::from_vec2(py, &proba)
            .map_err(|e| PyValueError::new_err(format!("Failed to create numpy array: {:?}", e)))
    }

    /// Predict class labels for multi-class classification
    ///
    /// Returns the class with highest probability (argmax of softmax).
    /// Only meaningful when trained with `use_multiclass_logloss()`.
    ///
    /// For binary classification, use `predict_class()` instead.
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    ///               Accepts float32 or float64 arrays
    ///
    /// Returns:
    ///     1D numpy array of class labels (0, 1, 2, ..., K-1)
    #[pyo3(signature = (features))]
    fn predict_class_multiclass<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        // Extract and validate features first (most specific error)
        let (num_features, raw_features) = extract_features_array(features)?;
        validate_feature_count(num_features, self.model.num_features())?;

        // Then check model type
        if !self.model.is_multiclass() {
            return Err(PyValueError::new_err(
                "predict_class_multiclass() is for multi-class models only. \
                 For binary classification, use predict_class() instead.",
            ));
        }

        // Use the raw prediction method (no binning needed, release GIL)
        let classes = py.detach(|| self.model.predict_class_multiclass_raw(&raw_features));

        Ok(PyArray1::from_vec(py, classes))
    }

    /// Get feature importance (gain-based)
    ///
    /// Returns:
    ///     1D numpy array of normalized feature importance
    fn feature_importance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let importances = self.model.feature_importance();
        PyArray1::from_vec(py, importances)
    }

    /// Save model to file
    ///
    /// Args:
    ///     path: Path to save the model
    ///     format: Serialization format ("rkyv" or "bincode", default: "rkyv")
    ///         - "rkyv": Zero-copy deserialization, fastest loading (recommended)
    ///         - "bincode" (or "bin"): Compact binary format, serde-based
    #[pyo3(signature = (path, format="rkyv"))]
    fn save(&self, path: &str, format: &str) -> PyResult<()> {
        let model_format = parse_model_format(format)?;
        match model_format {
            ModelFormat::Rkyv => serialize::save_model(&self.model, path),
            ModelFormat::Bincode => serialize::save_model_bincode(&self.model, path),
        }
        .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Save model and config to a directory for reproducibility
    ///
    /// Creates the directory if needed and saves:
    /// - model.rkyv (or model.bin): The trained model
    /// - config.json: Training configuration for reproducibility
    ///
    /// Args:
    ///     output_dir: Directory to save model and config
    ///     config: GBDTConfig used for training (for config.json)
    ///     formats: Serialization format(s) - either a single string ("rkyv" or "bincode")
    ///              or a list of strings (["rkyv", "bincode"]). Default: "rkyv"
    ///
    /// Example:
    ///     model = GBDTModel.train(features, targets, config)
    ///
    ///     # Single format (default)
    ///     model.save_to_directory("my_model", config)
    ///     # Creates: my_model/model.rkyv and my_model/config.json
    ///
    ///     # Multiple formats
    ///     model.save_to_directory("my_model", config, formats=["rkyv", "bincode"])
    ///     # Creates: my_model/model.rkyv, my_model/model.bin, and my_model/config.json
    #[pyo3(signature = (output_dir, config, formats=None))]
    fn save_to_directory(
        &self,
        output_dir: &str,
        config: &PyGBDTConfig,
        formats: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // Parse formats - can be None (default), a string, or a list of strings
        let model_formats = match formats {
            None => vec![ModelFormat::Rkyv], // Default to rkyv
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    // Single string format
                    vec![parse_model_format(&s)?]
                } else if let Ok(list) = obj.extract::<Vec<String>>() {
                    // List of format strings
                    if list.is_empty() {
                        return Err(PyValueError::new_err("formats list must not be empty"));
                    }
                    list.iter()
                        .map(|s| parse_model_format(s))
                        .collect::<PyResult<Vec<_>>>()?
                } else {
                    return Err(PyValueError::new_err(
                        "formats must be a string ('rkyv' or 'bincode') or a list of strings",
                    ));
                }
            }
        };

        self.model
            .save_to_directory(output_dir, &config.inner, &model_formats)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load model from file
    ///
    /// Args:
    ///     path: Path to the model file
    ///     format: Serialization format ("rkyv" or "bincode", default: "rkyv")
    ///         - "rkyv": Zero-copy deserialization, fastest loading (recommended)
    ///         - "bincode" (or "bin"): Compact binary format, serde-based
    ///
    /// Returns:
    ///     Loaded GBDTModel
    #[staticmethod]
    #[pyo3(signature = (path, format="rkyv"))]
    fn load(path: &str, format: &str) -> PyResult<Self> {
        let model_format = parse_model_format(format)?;
        let model = match model_format {
            ModelFormat::Rkyv => serialize::load_model(path),
            ModelFormat::Bincode => serialize::load_model_bincode(path),
        }
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(Self { model })
    }

    /// Get number of trees in the ensemble
    #[getter]
    fn num_trees(&self) -> usize {
        self.model.num_trees()
    }

    /// Get number of features
    #[getter]
    fn num_features(&self) -> usize {
        self.model.num_features()
    }

    /// Get base prediction value
    #[getter]
    fn base_prediction(&self) -> f32 {
        self.model.base_prediction()
    }

    /// Get conformal quantile (if calibrated)
    #[getter]
    fn conformal_quantile(&self) -> Option<f32> {
        self.model.conformal_quantile()
    }

    /// Get feature names
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model
            .feature_info()
            .iter()
            .map(|info| info.name.clone())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "GBDTModel(num_trees={}, num_features={}, base_prediction={:.4})",
            self.model.num_trees(),
            self.model.num_features(),
            self.model.base_prediction()
        )
    }
}

/// Register Python module classes and functions
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMonotonicConstraint>()?;
    m.add_class::<PyGBDTConfig>()?;
    m.add_class::<PyGBDTModel>()?;
    Ok(())
}
