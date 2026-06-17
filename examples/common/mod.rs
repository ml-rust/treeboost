//! Shared utilities for TreeBoost examples
//!
//! This module provides common functionality used across multiple examples
//! to avoid code duplication.

#![allow(dead_code)]

use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

/// Simple Linear Congruential Generator (LCG) for reproducible random numbers.
///
/// Uses the classic parameters from MINSTD for simplicity.
/// Not cryptographically secure - for example/demo purposes only.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next random f32 in range [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7FFF) as f32 / 32767.0
    }

    /// Generate a random f32 in range [min, max)
    pub fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Extract a subset of rows from a BinnedDataset.
///
/// Creates a new dataset containing rows from `start_row` to `end_row` (exclusive).
/// Useful for creating train/test/calibration splits.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `start_row` - Starting row index (inclusive)
/// * `end_row` - Ending row index (exclusive)
///
/// # Returns
/// A new BinnedDataset containing only the specified rows
pub fn extract_subset(dataset: &BinnedDataset, start_row: usize, end_row: usize) -> BinnedDataset {
    let n_rows = end_row - start_row;
    let n_features = dataset.num_features();

    // Extract features (column-major)
    let mut features = Vec::with_capacity(n_rows * n_features);
    for f in 0..n_features {
        let col = dataset.feature_column(f);
        for &val in &col[start_row..end_row] {
            features.push(val);
        }
    }

    // Extract targets
    let targets: Vec<f32> = dataset.targets()[start_row..end_row].to_vec();

    // Clone feature info
    let feature_info: Vec<FeatureInfo> = (0..n_features)
        .map(|i| {
            let info = dataset.feature_info(i);
            FeatureInfo {
                name: info.name.clone(),
                feature_type: info.feature_type,
                num_bins: info.num_bins,
                bin_boundaries: info.bin_boundaries.clone(),
                impute_value: info.impute_value,
            }
        })
        .collect();

    BinnedDataset::new(n_rows, features, targets, feature_info)
}

/// Create feature info for synthetic numeric features.
///
/// # Arguments
/// * `n_features` - Number of features
/// * `prefix` - Name prefix (e.g., "feature" -> "feature_0", "feature_1", ...)
pub fn create_feature_info(n_features: usize, prefix: &str) -> Vec<FeatureInfo> {
    (0..n_features)
        .map(|i| FeatureInfo {
            name: format!("{}_{}", prefix, i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        })
        .collect()
}
