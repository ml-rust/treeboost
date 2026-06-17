//! Shared test utilities for TreeBoost integration tests
// reason: each integration test file is its own crate; some helpers are
// only used by a subset of them, so they appear unused per-crate.
#![allow(dead_code)]

use polars::prelude::*;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::Result;

/// Create a synthetic regression dataset for testing
///
/// Generates deterministic pseudo-random data using a seed for reproducibility.
/// Target: y = f0 * 10 + f1 * 5 + noise
pub fn create_synthetic_dataset(n: usize, seed: u64) -> BinnedDataset {
    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_rand = || -> f32 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    let num_features = 5;
    let mut features = Vec::with_capacity(n * num_features);

    // Generate features (column-major)
    for _f in 0..num_features {
        for _r in 0..n {
            features.push((next_rand() * 255.0) as u8);
        }
    }

    // Generate targets: y = f0 * 10 + f1 * 5 + noise
    let targets: Vec<f32> = (0..n)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n + i] as f32 / 255.0;
            f0 * 10.0 + f1 * 5.0 + next_rand() * 0.5
        })
        .collect();

    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("feature_{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        })
        .collect();

    BinnedDataset::new(n, features, targets, feature_info)
}

/// Generate linear trend data for LTT extrapolation testing
///
/// Creates data with pure linear relationship: y = slope * x + noise
/// Designed for testing extrapolation: train on x ∈ [0, train_max], predict on x > train_max
///
/// # Arguments
/// * `n` - Number of samples
/// * `slope` - Linear coefficient (default: 3.5)
/// * `noise_std` - Standard deviation of Gaussian noise (default: 1.0)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// DataFrame with columns: ["x", "target"], where x ∈ [0, n-1]
pub fn generate_linear_trend(n: usize, slope: f64, noise_std: f64, seed: u64) -> Result<DataFrame> {
    generate_linear_trend_range(0.0, n as f64, slope, noise_std, seed)
}

/// Generate linear trend data with specified x range
///
/// Creates data with pure linear relationship: y = slope * x + noise
///
/// # Arguments
/// * `x_start` - Starting x value
/// * `x_end` - Ending x value (exclusive)
/// * `slope` - Linear coefficient
/// * `noise_std` - Standard deviation of Gaussian noise
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// DataFrame with columns: ["x", "target"], where x ∈ [x_start, x_end)
pub fn generate_linear_trend_range(
    x_start: f64,
    x_end: f64,
    slope: f64,
    noise_std: f64,
    seed: u64,
) -> Result<DataFrame> {
    let n = (x_end - x_start) as usize;
    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_rand = || -> f64 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let uniform = ((state >> 16) & 0x7FFF) as f64 / 32767.0;
        // Box-Muller transform for Gaussian noise
        let u1 = uniform;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = ((state >> 16) & 0x7FFF) as f64 / 32767.0;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z * noise_std
    };

    let x: Vec<f64> = (0..n).map(|i| x_start + i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| slope * xi + next_rand()).collect();

    df! {
        "x" => x,
        "target" => y,
    }
    .map_err(|e| {
        treeboost::TreeBoostError::Config(format!("Failed to create linear trend DataFrame: {}", e))
    })
}

/// Generate linear trend with seasonality for advanced LTT testing
///
/// Creates data with: y = slope * x + amplitude * sin(x / period) + noise
///
/// # Arguments
/// * `n` - Number of samples
/// * `slope` - Linear coefficient
/// * `amplitude` - Amplitude of sine wave
/// * `period` - Period of sine wave (higher = slower oscillation)
/// * `noise_std` - Standard deviation of Gaussian noise
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// DataFrame with columns: ["x", "target"], where x ∈ [0, n-1]
pub fn generate_linear_trend_with_seasonality(
    n: usize,
    slope: f64,
    amplitude: f64,
    period: f64,
    noise_std: f64,
    seed: u64,
) -> Result<DataFrame> {
    generate_linear_trend_with_seasonality_range(
        0.0, n as f64, slope, amplitude, period, noise_std, seed,
    )
}

/// Generate linear trend with seasonality for specified x range
///
/// Creates data with: y = slope * x + amplitude * sin(x / period) + noise
///
/// # Arguments
/// * `x_start` - Starting x value
/// * `x_end` - Ending x value (exclusive)
/// * `slope` - Linear coefficient
/// * `amplitude` - Amplitude of sine wave
/// * `period` - Period of sine wave (higher = slower oscillation)
/// * `noise_std` - Standard deviation of Gaussian noise
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// DataFrame with columns: ["x", "target"], where x ∈ [x_start, x_end)
pub fn generate_linear_trend_with_seasonality_range(
    x_start: f64,
    x_end: f64,
    slope: f64,
    amplitude: f64,
    period: f64,
    noise_std: f64,
    seed: u64,
) -> Result<DataFrame> {
    let n = (x_end - x_start) as usize;
    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_rand = || -> f64 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let uniform = ((state >> 16) & 0x7FFF) as f64 / 32767.0;
        // Box-Muller transform for Gaussian noise
        let u1 = uniform;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = ((state >> 16) & 0x7FFF) as f64 / 32767.0;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z * noise_std
    };

    let x: Vec<f64> = (0..n).map(|i| x_start + i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| slope * xi + amplitude * (xi / period).sin() + next_rand())
        .collect();

    df! {
        "x" => x,
        "target" => y,
    }
    .map_err(|e| {
        treeboost::TreeBoostError::Config(format!(
            "Failed to create linear+seasonal DataFrame: {}",
            e
        ))
    })
}

/// Generate multi-class classification data with configurable class imbalance
///
/// Creates synthetic data for multi-class classification with controlled class distribution.
/// Features are generated with different separability across classes.
///
/// # Arguments
/// * `n` - Total number of samples
/// * `num_classes` - Number of classes (2-10)
/// * `imbalance` - Class distribution ratios (must sum to ~1.0, length = num_classes)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// DataFrame with columns: ["f1", "f2", "f3", "target"], where target ∈ [0, num_classes)
///
/// # Example
/// ```ignore
/// // Generate 1000 samples with 3 classes: 60% class 0, 30% class 1, 10% class 2
/// let df = generate_multiclass_data(1000, 3, vec![0.6, 0.3, 0.1], 42)?;
/// ```
pub fn generate_multiclass_data(
    n: usize,
    num_classes: usize,
    imbalance: Vec<f64>,
    seed: u64,
) -> Result<DataFrame> {
    if !(2..=10).contains(&num_classes) {
        return Err(treeboost::TreeBoostError::Config(
            "num_classes must be between 2 and 10".to_string(),
        ));
    }
    if imbalance.len() != num_classes {
        return Err(treeboost::TreeBoostError::Config(format!(
            "imbalance vector length ({}) must match num_classes ({})",
            imbalance.len(),
            num_classes
        )));
    }
    let sum: f64 = imbalance.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(treeboost::TreeBoostError::Config(format!(
            "imbalance ratios must sum to 1.0, got {}",
            sum
        )));
    }

    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_rand = || -> f64 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) & 0x7FFF) as f64 / 32767.0
    };
    let mut next_gaussian = || -> f64 {
        let u1 = next_rand();
        let u2 = next_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Calculate samples per class based on imbalance
    let mut samples_per_class = vec![0; num_classes];
    let mut remaining = n;
    for (i, &ratio) in imbalance.iter().enumerate() {
        if i == num_classes - 1 {
            samples_per_class[i] = remaining;
        } else {
            samples_per_class[i] = (n as f64 * ratio).round() as usize;
            remaining -= samples_per_class[i];
        }
    }

    let mut f1_vals = Vec::with_capacity(n);
    let mut f2_vals = Vec::with_capacity(n);
    let mut f3_vals = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);

    // Generate class-specific clusters
    for (class_id, &n_samples) in samples_per_class.iter().enumerate() {
        let class_center_1 = (class_id as f64) * 3.0; // Separate classes along f1
        let class_center_2 = ((class_id % 2) as f64) * 4.0; // Alternate along f2

        for _ in 0..n_samples {
            // Features with class-specific means and noise
            let f1 = class_center_1 + next_gaussian() * 1.5;
            let f2 = class_center_2 + next_gaussian() * 1.5;
            let f3 = next_gaussian() * 2.0; // Uninformative feature

            f1_vals.push(f1);
            f2_vals.push(f2);
            f3_vals.push(f3);
            targets.push(class_id as i32);
        }
    }

    df! {
        "f1" => f1_vals,
        "f2" => f2_vals,
        "f3" => f3_vals,
        "target" => targets,
    }
    .map_err(|e| {
        treeboost::TreeBoostError::Config(format!("Failed to create multi-class DataFrame: {}", e))
    })
}

/// Generate multi-label classification data with configurable label correlation
///
/// Creates synthetic data for multi-label classification where samples can have multiple labels.
/// Labels can be independent or correlated based on the correlations parameter.
///
/// # Arguments
/// * `n` - Number of samples
/// * `num_labels` - Number of labels (2-10)
/// * `correlations` - List of (label_i, label_j) pairs that should be correlated
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// DataFrame with columns: ["f1", "f2", "f3", "f4", "label_0", "label_1", ..., "label_N"]
/// where each label_i ∈ {0, 1}
///
/// # Example
/// ```ignore
/// // Generate 1000 samples with 3 labels: labels 0 & 1 are correlated
/// let df = generate_multilabel_data(1000, 3, vec![(0, 1)], 42)?;
/// ```
pub fn generate_multilabel_data(
    n: usize,
    num_labels: usize,
    correlations: Vec<(usize, usize)>,
    seed: u64,
) -> Result<DataFrame> {
    if !(2..=10).contains(&num_labels) {
        return Err(treeboost::TreeBoostError::Config(
            "num_labels must be between 2 and 10".to_string(),
        ));
    }

    // Validate correlations
    for &(i, j) in &correlations {
        if i >= num_labels || j >= num_labels {
            return Err(treeboost::TreeBoostError::Config(format!(
                "Invalid correlation ({}, {}): labels must be < {}",
                i, j, num_labels
            )));
        }
    }

    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_rand = || -> f64 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) & 0x7FFF) as f64 / 32767.0
    };
    // Separate Gaussian generator that doesn't borrow next_rand
    let mut gaussian_state = seed.wrapping_mul(7919); // Different initial seed
    let mut next_gaussian = || -> f64 {
        gaussian_state = gaussian_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = ((gaussian_state >> 16) & 0x7FFF) as f64 / 32767.0;
        gaussian_state = gaussian_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = ((gaussian_state >> 16) & 0x7FFF) as f64 / 32767.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut f1_vals = Vec::with_capacity(n);
    let mut f2_vals = Vec::with_capacity(n);
    let mut f3_vals = Vec::with_capacity(n);
    let mut f4_vals = Vec::with_capacity(n);
    let mut label_vecs: Vec<Vec<i32>> = vec![Vec::with_capacity(n); num_labels];

    for _ in 0..n {
        // Generate features
        let f1 = next_gaussian() * 2.0;
        let f2 = next_gaussian() * 2.0;
        let f3 = next_gaussian() * 2.0;
        let f4 = next_gaussian() * 2.0;

        f1_vals.push(f1);
        f2_vals.push(f2);
        f3_vals.push(f3);
        f4_vals.push(f4);

        // Generate labels with dependencies
        let mut labels = vec![0; num_labels];

        // Base probabilities from features
        for (label_id, label) in labels.iter_mut().enumerate() {
            let signal = match label_id {
                0 => f1 + f2 * 0.5,              // Label 0 depends on f1, f2
                1 => f2 * 0.8 + f3 * 0.3,        // Label 1 depends on f2, f3
                2 => f3 - f1 * 0.4,              // Label 2 depends on f3, f1
                _ => f4 * 0.6 + next_gaussian(), // Other labels depend on f4
            };

            // Sigmoid to get probability
            let prob = 1.0 / (1.0 + (-signal).exp());
            *label = if next_rand() < prob { 1 } else { 0 };
        }

        // Apply correlations: if label i is 1 and (i, j) is correlated, increase prob of j
        for &(i, j) in &correlations {
            if labels[i] == 1 && next_rand() < 0.7 {
                // 70% chance of correlation
                labels[j] = 1;
            }
        }

        // Store labels
        for (label_id, &label_val) in labels.iter().enumerate() {
            label_vecs[label_id].push(label_val);
        }
    }

    // Build DataFrame
    let mut df_builder = df! {
        "f1" => f1_vals,
        "f2" => f2_vals,
        "f3" => f3_vals,
        "f4" => f4_vals,
    }
    .map_err(|e| {
        treeboost::TreeBoostError::Config(format!("Failed to create multi-label DataFrame: {}", e))
    })?;

    // Add label columns
    for (label_id, label_vec) in label_vecs.into_iter().enumerate() {
        let label_series = Series::new(format!("label_{}", label_id).into(), label_vec);
        let _ = df_builder.with_column(label_series.into()).map_err(|e| {
            treeboost::TreeBoostError::Config(format!("Failed to add label column: {}", e))
        })?;
    }

    Ok(df_builder)
}
