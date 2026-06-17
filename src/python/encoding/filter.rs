//! Python bindings for category filtering
//!
//! Provides wrappers for CountMinSketch, CategoryFilter, and CategoryMapping.

use pyo3::prelude::*;

use crate::encoding::{CategoryFilter, CategoryMapping, CountMinSketch};

/// Python wrapper for Count-Min Sketch
///
/// A probabilistic data structure for approximate frequency counting.
/// Uses sub-linear space and never underestimates counts.
#[pyclass(name = "CountMinSketch")]
pub struct PyCountMinSketch {
    inner: CountMinSketch,
}

#[pymethods]
impl PyCountMinSketch {
    /// Create a new Count-Min Sketch
    ///
    /// Args:
    ///     eps: Error tolerance (e.g., 0.01 for 1% error)
    ///     confidence: Confidence level (e.g., 0.99 for 99%)
    ///
    /// Example:
    ///     cms = CountMinSketch(eps=0.001, confidence=0.99)
    #[new]
    #[pyo3(signature = (eps=0.001, confidence=0.99))]
    fn new(eps: f64, confidence: f64) -> PyResult<Self> {
        if eps <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eps must be positive",
            ));
        }
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "confidence must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: CountMinSketch::new(eps, confidence),
        })
    }

    /// Increment the count for a hash value by 1
    fn inc(&mut self, hash: u64) {
        self.inner.inc(hash);
    }

    /// Increment the count for a hash value by a specified amount
    fn inc_by(&mut self, hash: u64, count: u64) {
        self.inner.inc_by(hash, count);
    }

    /// Estimate the count for a hash value
    fn estimate(&self, hash: u64) -> u64 {
        self.inner.estimate(hash)
    }

    /// Reset all counters to zero
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Divide all counters by 2 (useful for time decay)
    fn halve(&mut self) {
        self.inner.halve();
    }

    /// Table width
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Table depth (number of hash functions)
    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Memory usage in bytes
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "CountMinSketch(width={}, depth={}, memory={}B)",
            self.inner.width(),
            self.inner.depth(),
            self.inner.memory_bytes()
        )
    }
}

/// Python wrapper for CategoryFilter
///
/// Filters rare categories to "unknown" using Count-Min Sketch
/// for probabilistic counting. Essential for handling high-cardinality
/// categorical features with typos and rare values.
#[pyclass(name = "CategoryFilter")]
pub struct PyCategoryFilter {
    inner: CategoryFilter,
}

#[pymethods]
impl PyCategoryFilter {
    /// Create a new category filter
    ///
    /// Args:
    ///     eps: Error tolerance (e.g., 0.001 for 0.1% error)
    ///     confidence: Confidence level (e.g., 0.99 for 99%)
    ///     min_count: Minimum frequency to keep a category
    ///
    /// Example:
    ///     filter = CategoryFilter(eps=0.001, confidence=0.99, min_count=5)
    #[new]
    #[pyo3(signature = (eps=0.001, confidence=0.99, min_count=5))]
    fn new(eps: f64, confidence: f64, min_count: u64) -> PyResult<Self> {
        if eps <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eps must be positive",
            ));
        }
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "confidence must be in (0, 1)",
            ));
        }
        Ok(Self {
            inner: CategoryFilter::new(eps, confidence, min_count),
        })
    }

    /// Create with default parameters for high-cardinality data
    ///
    /// Uses eps=0.001, confidence=0.99, min_count=5.
    #[staticmethod]
    fn default_for_high_cardinality() -> Self {
        Self {
            inner: CategoryFilter::default_for_high_cardinality(),
        }
    }

    /// First pass: count a single category
    fn count(&mut self, category: &str) {
        self.inner.count(category);
    }

    /// First pass: count a batch of categories
    fn count_batch(&mut self, categories: Vec<String>) {
        for cat in &categories {
            self.inner.count(cat);
        }
    }

    /// Finalize the filter by identifying frequent categories
    ///
    /// Must be called after counting and before filtering.
    /// Pass all unique categories seen during counting.
    fn finalize(&mut self, unique_categories: Vec<String>) {
        self.inner.finalize(unique_categories);
    }

    /// Check if a category is frequent enough to keep
    fn is_frequent(&self, category: &str) -> bool {
        self.inner.is_frequent(category)
    }

    /// Get the estimated count for a category
    fn estimate_count(&self, category: &str) -> u64 {
        self.inner.estimate_count(category)
    }

    /// Filter a category: returns the category if frequent, "unknown" otherwise
    fn filter(&self, category: &str) -> String {
        self.inner.filter(category).to_string()
    }

    /// Filter a batch of categories
    fn filter_batch(&self, categories: Vec<String>) -> Vec<String> {
        categories
            .iter()
            .map(|c| self.inner.filter(c).to_string())
            .collect()
    }

    /// Number of frequent categories identified
    #[getter]
    fn num_frequent(&self) -> usize {
        self.inner.num_frequent()
    }

    /// Get all frequent categories as a list
    fn frequent_categories(&self) -> Vec<String> {
        self.inner.frequent_categories().iter().cloned().collect()
    }

    /// Memory usage of the sketch in bytes
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Create a CategoryMapping from this filter
    fn to_mapping(&self) -> PyCategoryMapping {
        PyCategoryMapping {
            inner: CategoryMapping::from_filter(&self.inner),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CategoryFilter(num_frequent={}, memory={}B)",
            self.inner.num_frequent(),
            self.inner.memory_bytes()
        )
    }
}

/// Python wrapper for CategoryMapping
///
/// Provides a serializable mapping from categories to indices.
/// Uses binary search for efficient lookup.
#[pyclass(from_py_object, name = "CategoryMapping")]
#[derive(Clone)]
pub struct PyCategoryMapping {
    inner: CategoryMapping,
}

#[pymethods]
impl PyCategoryMapping {
    /// Create a mapping from a CategoryFilter
    #[staticmethod]
    fn from_filter(filter: &PyCategoryFilter) -> Self {
        Self {
            inner: CategoryMapping::from_filter(&filter.inner),
        }
    }

    /// Get index for a category (uses binary search)
    ///
    /// Returns unknown_idx for categories not in the mapping.
    fn get_index(&self, category: &str) -> u32 {
        self.inner.get_index(category)
    }

    /// Get indices for a batch of categories
    fn get_indices(&self, categories: Vec<String>) -> Vec<u32> {
        categories.iter().map(|c| self.inner.get_index(c)).collect()
    }

    /// Index used for unknown categories
    #[getter]
    fn unknown_idx(&self) -> u32 {
        self.inner.unknown_idx
    }

    /// Total number of categories including unknown
    #[getter]
    fn num_categories(&self) -> usize {
        self.inner.num_categories()
    }

    /// Get all category-to-index mappings as a list of tuples
    fn items(&self) -> Vec<(String, u32)> {
        self.inner.category_to_idx.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CategoryMapping(num_categories={}, unknown_idx={})",
            self.inner.num_categories(),
            self.inner.unknown_idx
        )
    }

    fn __len__(&self) -> usize {
        self.inner.num_categories()
    }
}

impl From<CategoryMapping> for PyCategoryMapping {
    fn from(mapping: CategoryMapping) -> Self {
        Self { inner: mapping }
    }
}

/// Register filter classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCountMinSketch>()?;
    m.add_class::<PyCategoryFilter>()?;
    m.add_class::<PyCategoryMapping>()?;
    Ok(())
}
