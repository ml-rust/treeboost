//! Python bindings for dataset splitting utilities
//!
//! Provides wrappers for HoldoutSplit and KFoldSplit.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::dataset::{split_holdout, split_kfold, HoldoutSplit, KFoldSplit};

/// Python wrapper for holdout split result
///
/// Provides train/validation/calibration index sets from a single random split.
#[pyclass(name = "HoldoutSplit")]
pub struct PyHoldoutSplit {
    inner: HoldoutSplit,
}

#[pymethods]
impl PyHoldoutSplit {
    /// Create a holdout split
    ///
    /// Splits indices into train, validation, and optionally calibration sets.
    /// All index arrays are sorted for cache-friendly access.
    ///
    /// Args:
    ///     n_samples: Total number of samples
    ///     val_ratio: Fraction for validation set (0.0 to skip)
    ///     calib_ratio: Fraction for calibration set (0.0 to skip)
    ///     seed: Random seed for reproducibility
    ///
    /// Example:
    ///     split = HoldoutSplit.create(10000, val_ratio=0.2, calib_ratio=0.1, seed=42)
    ///     print(f"Train: {len(split.train_indices)}")
    #[staticmethod]
    #[pyo3(signature = (n_samples, val_ratio=0.2, calib_ratio=0.0, seed=42))]
    fn create(n_samples: usize, val_ratio: f32, calib_ratio: f32, seed: u64) -> PyResult<Self> {
        if n_samples == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_samples must be > 0",
            ));
        }
        if !(0.0..1.0).contains(&val_ratio) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "val_ratio must be in [0.0, 1.0)",
            ));
        }
        if !(0.0..1.0).contains(&calib_ratio) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "calib_ratio must be in [0.0, 1.0)",
            ));
        }
        if val_ratio + calib_ratio >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "val_ratio + calib_ratio must be < 1.0",
            ));
        }

        Ok(Self {
            inner: split_holdout(n_samples, val_ratio, calib_ratio, seed),
        })
    }

    /// Training set indices as numpy array
    #[getter]
    fn train_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_slice(py, &self.inner.train)
    }

    /// Validation set indices as numpy array
    #[getter]
    fn validation_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_slice(py, &self.inner.validation)
    }

    /// Calibration set indices as numpy array
    #[getter]
    fn calibration_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_slice(py, &self.inner.calibration)
    }

    /// Number of training samples
    #[getter]
    fn train_len(&self) -> usize {
        self.inner.train_len()
    }

    /// Number of validation samples
    #[getter]
    fn val_len(&self) -> usize {
        self.inner.val_len()
    }

    /// Number of calibration samples
    #[getter]
    fn calib_len(&self) -> usize {
        self.inner.calib_len()
    }

    fn __repr__(&self) -> String {
        format!(
            "HoldoutSplit(train={}, val={}, calib={})",
            self.inner.train_len(),
            self.inner.val_len(),
            self.inner.calib_len()
        )
    }
}

impl From<HoldoutSplit> for PyHoldoutSplit {
    fn from(split: HoldoutSplit) -> Self {
        Self { inner: split }
    }
}

/// Python wrapper for K-fold cross-validation split
///
/// Partitions indices into k disjoint folds for cross-validation.
#[pyclass(name = "KFoldSplit")]
pub struct PyKFoldSplit {
    inner: KFoldSplit,
}

#[pymethods]
impl PyKFoldSplit {
    /// Create a K-fold split
    ///
    /// Partitions indices into k disjoint folds of approximately equal size.
    /// Each fold's indices are sorted for cache-friendly access.
    ///
    /// Args:
    ///     n_samples: Total number of samples
    ///     k: Number of folds (must be >= 2)
    ///     seed: Random seed for reproducibility
    ///
    /// Example:
    ///     kfold = KFoldSplit.create(10000, k=5, seed=42)
    ///     for i in range(5):
    ///         train_idx, val_idx = kfold.get_fold(i)
    #[staticmethod]
    #[pyo3(signature = (n_samples, k=5, seed=42))]
    fn create(n_samples: usize, k: usize, seed: u64) -> PyResult<Self> {
        if n_samples == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_samples must be > 0",
            ));
        }
        if k < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k must be >= 2 for cross-validation",
            ));
        }
        if k > n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k cannot exceed n_samples",
            ));
        }

        Ok(Self {
            inner: split_kfold(n_samples, k, seed)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Number of folds
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// Get train and validation indices for a specific fold
    ///
    /// Args:
    ///     fold_idx: The fold to use as validation (0-indexed)
    ///
    /// Returns:
    ///     Tuple of (train_indices, val_indices) as numpy arrays
    #[allow(clippy::type_complexity)] // reason: complex return tuple kept inline
    fn get_fold<'py>(
        &self,
        py: Python<'py>,
        fold_idx: usize,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        if fold_idx >= self.inner.k() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "fold_idx {} out of range (k={})",
                fold_idx,
                self.inner.k()
            )));
        }

        let (train, val) = self.inner.get_fold(fold_idx);
        Ok((PyArray1::from_vec(py, train), PyArray1::from_vec(py, val)))
    }

    /// Get the size of each fold
    fn fold_sizes(&self) -> Vec<usize> {
        self.inner.folds.iter().map(|f| f.len()).collect()
    }

    fn __repr__(&self) -> String {
        let sizes = self.fold_sizes();
        let min_size = sizes.iter().min().unwrap_or(&0);
        let max_size = sizes.iter().max().unwrap_or(&0);
        format!(
            "KFoldSplit(k={}, fold_sizes={}..{})",
            self.inner.k(),
            min_size,
            max_size
        )
    }
}

impl From<KFoldSplit> for PyKFoldSplit {
    fn from(split: KFoldSplit) -> Self {
        Self { inner: split }
    }
}

/// Register split classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHoldoutSplit>()?;
    m.add_class::<PyKFoldSplit>()?;
    Ok(())
}
