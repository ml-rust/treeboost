//! Python callback bridging for tuner progress
//!
//! Provides a mechanism to call Python functions from Rust
//! during tuning, reacquiring the GIL as needed.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::tuner::TrialResult;

use super::results::PyTrialResult;

/// Wrapper for Python callback that can be called from Rust threads
pub struct PyProgressCallback {
    callback: Arc<Py<PyAny>>,
}

impl PyProgressCallback {
    /// Create a new callback wrapper from a Python callable
    pub fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback: Arc::new(callback),
        }
    }

    /// Call the Python callback with trial result
    ///
    /// Reacquires the GIL to safely call into Python.
    ///
    /// Arguments:
    /// - trial: The completed trial result
    /// - current: Current trial number (1-indexed)
    /// - total: Total number of trials
    pub fn call(&self, trial: &TrialResult, current: usize, total: usize) {
        // Reacquire GIL to call Python
        Python::attach(|py| {
            let py_trial = PyTrialResult::from(trial.clone());
            let py_trial_obj = Py::new(py, py_trial).unwrap();

            let args = PyTuple::new(
                py,
                &[
                    py_trial_obj.into_any(),
                    current.into_pyobject(py).unwrap().into_any().unbind(),
                    total.into_pyobject(py).unwrap().into_any().unbind(),
                ],
            )
            .unwrap();

            if let Err(e) = self.callback.call1(py, args) {
                // Log error but don't panic - tuning should continue
                eprintln!("Warning: Progress callback failed: {}", e);
            }
        });
    }
}

// PyProgressCallback is Send+Sync because it holds Arc<Py<PyAny>>
// and only accesses the Python object while holding the GIL
unsafe impl Send for PyProgressCallback {}
unsafe impl Sync for PyProgressCallback {}

/// Validate that a Python object is callable
pub fn validate_callable(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<()> {
    if !obj.bind(py).is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "callback must be callable",
        ));
    }
    Ok(())
}
