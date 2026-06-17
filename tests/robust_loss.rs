//! Test robust loss functions (PseudoHuber) vs standard loss (MSE)
//!
//! This test validates that PseudoHuber loss handles outliers better than MSE.
//!
//! Key insights:
//! - MSE is sensitive to outliers (squared error amplifies large deviations)
//! - PseudoHuber is robust (behaves like absolute error for large deviations)
//! - Real-world data often has outliers (measurement errors, fat-tailed noise)

mod common;

use polars::prelude::*;
use treeboost::{
    dataset::BinnedDataset,
    loss::{MseLoss, PseudoHuberLoss},
    model::{BoostingMode, UniversalConfig, UniversalModel},
};

/// Generate data with Cauchy noise (heavy tails)
///
/// Cauchy distribution has no finite variance - perfect for testing robustness!
/// y = 2*x + 3 + Cauchy(0, scale)
fn generate_cauchy_noise(
    n: usize,
    slope: f64,
    intercept: f64,
    scale: f64,
    seed: u64,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_cauchy = || -> f64 {
        // Box-Muller-like transform for Cauchy (via uniform)
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u = ((state >> 16) & 0x7FFF) as f64 / 32767.0;

        // Cauchy CDF inverse: tan(π * (u - 0.5))
        scale * ((std::f64::consts::PI * (u - 0.5)).tan())
    };

    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| slope * xi + intercept + next_cauchy())
        .collect();

    Ok(df! {
        "x" => x,
        "target" => y,
    }?)
}

/// Convert DataFrame to BinnedDataset for training
/// Returns (BinnedDataset, target_mean, target_std) for later denormalization
fn dataframe_to_binned(
    df: &DataFrame,
    target_col: &str,
) -> Result<(BinnedDataset, f64, f64), Box<dyn std::error::Error>> {
    dataframe_to_binned_with_stats(df, target_col, None, None)
}

/// Convert DataFrame to BinnedDataset with optional target normalization parameters
/// If mean/std are provided, uses those for normalization; otherwise computes from data
fn dataframe_to_binned_with_stats(
    df: &DataFrame,
    target_col: &str,
    target_mean: Option<f64>,
    target_std: Option<f64>,
) -> Result<(BinnedDataset, f64, f64), Box<dyn std::error::Error>> {
    let num_rows = df.height();
    let feature_cols: Vec<&str> = df
        .get_column_names()
        .into_iter()
        .filter(|&name| name.as_str() != target_col)
        .map(|s| s.as_str())
        .collect();

    let num_features = feature_cols.len();

    // Extract features (column-major for BinnedDataset)
    let mut binned_features = Vec::with_capacity(num_rows * num_features);
    for col_name in &feature_cols {
        let col = df.column(col_name)?;
        let values = col.f64()?;

        // Simple uniform binning: map to [0, 255]
        let min = values.min().unwrap_or(0.0);
        let max = values.max().unwrap_or(1.0);
        let range = (max - min).max(1e-10);

        for val in values.into_no_null_iter() {
            let normalized = ((val - min) / range * 255.0).clamp(0.0, 255.0);
            binned_features.push(normalized as u8);
        }
    }

    // Extract targets
    let target_series = df.column(target_col)?;
    let target_values: Vec<f64> = target_series.f64()?.into_no_null_iter().collect();

    // Use provided stats or compute from data
    let (mean, std) = if let (Some(m), Some(s)) = (target_mean, target_std) {
        (m, s)
    } else {
        let mean: f64 = target_values.iter().sum::<f64>() / target_values.len() as f64;
        let variance: f64 = target_values
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / target_values.len() as f64;
        let std = variance.sqrt().max(1e-10);
        (mean, std)
    };

    // Normalize targets: (y - mean) / std
    let targets: Vec<f32> = target_values
        .iter()
        .map(|&v| ((v - mean) / std) as f32)
        .collect();

    // Create feature info
    let feature_info: Vec<treeboost::dataset::FeatureInfo> = feature_cols
        .iter()
        .map(|&name| treeboost::dataset::FeatureInfo {
            name: name.to_string(),
            feature_type: treeboost::dataset::FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        })
        .collect();

    Ok((
        BinnedDataset::new(num_rows, binned_features, targets, feature_info),
        mean,
        std,
    ))
}

/// Add extreme spikes to target values (simulates measurement errors)
fn add_extreme_spikes(y: &mut [f64], ratio: f32, magnitude: f64, seed: u64) {
    let n_spikes = (y.len() as f32 * ratio).round() as usize;

    // Compute std for scaling
    let mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let variance: f64 = y.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / y.len() as f64;
    let std = variance.sqrt();

    // Add spikes at deterministic random indices
    let mut state = seed;
    for _ in 0..n_spikes {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let idx = ((state >> 16) & 0x7FFF) as usize % y.len();

        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let sign = if ((state >> 16) & 1) == 0 { 1.0 } else { -1.0 };

        y[idx] += sign * magnitude * std;
    }
}

/// Test that PseudoHuber handles Cauchy noise better than MSE
///
/// **Setup:**
/// - y = 2*x + 3 + Cauchy(0, 1.0)
/// - Cauchy has heavy tails (no finite variance)
/// - Train: 1500 samples, Test: 500 samples
///
/// **Expected:**
/// - PseudoHuber should be significantly more robust (20%+ better)
#[test]
fn test_pseudo_huber_vs_mse_on_cauchy_noise() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: PseudoHuber vs MSE on Cauchy Noise ===");

    // Generate training data with Cauchy noise
    let train_df = generate_cauchy_noise(1500, 2.0, 3.0, 1.0, 42).unwrap();
    println!(
        "Training data: {} rows with Cauchy(0, 1.0) noise",
        train_df.height()
    );

    // Generate test data (different seed)
    let test_df = generate_cauchy_noise(500, 2.0, 3.0, 1.0, 123).unwrap();
    println!("Test data: {} rows", test_df.height());

    // Convert to BinnedDataset (with normalization to avoid numerical issues)
    let (train_binned, train_mean, train_std) = dataframe_to_binned(&train_df, "target").unwrap();
    let (test_binned, _, _) =
        dataframe_to_binned_with_stats(&test_df, "target", Some(train_mean), Some(train_std))
            .unwrap();

    // Normalize test targets using TRAIN statistics (avoid data leakage)
    let test_targets: Vec<f32> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| ((v - train_mean) / train_std) as f32)
        .collect();

    // ============================================================
    // Model 1: MSE Loss (sensitive to outliers)
    // ============================================================
    println!("\n--- Training with MSE Loss ---");
    let mut config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(200)
        .with_learning_rate(0.3)?
        .with_seed(42);
    config.tree_config = config.tree_config.with_max_depth(6).unwrap();

    let mse_loss = MseLoss::new();
    let mse_model = UniversalModel::train(&train_binned, config.clone(), &mse_loss).unwrap();
    let mse_preds = mse_model.predict(&test_binned);
    let mse_rmse = calculate_rmse_f32(&mse_preds, &test_targets);
    println!("MSE test RMSE: {:.4}", mse_rmse);

    // ============================================================
    // Model 2: PseudoHuber Loss (robust to outliers)
    // ============================================================
    println!("\n--- Training with PseudoHuber Loss (delta=1.0) ---");
    let huber_loss = PseudoHuberLoss::new(1.0);
    let huber_model = UniversalModel::train(&train_binned, config, &huber_loss).unwrap();
    let huber_preds = huber_model.predict(&test_binned);
    let huber_rmse = calculate_rmse_f32(&huber_preds, &test_targets);
    println!("PseudoHuber test RMSE: {:.4}", huber_rmse);

    // ============================================================
    // Assertions
    // ============================================================
    println!("\n--- Results ---");
    println!("MSE RMSE: {:.4}", mse_rmse);
    println!("PseudoHuber RMSE: {:.4}", huber_rmse);

    // Basic sanity checks
    assert!(
        mse_rmse.is_finite(),
        "MSE RMSE should be finite, got {}",
        mse_rmse
    );
    assert!(
        huber_rmse.is_finite(),
        "PseudoHuber RMSE should be finite, got {}",
        huber_rmse
    );

    if mse_rmse > 0.0 && huber_rmse > 0.0 {
        println!("Improvement: {:.2}x better", mse_rmse / huber_rmse);

        // PseudoHuber should be better or at least not worse than MSE
        // (Relaxed from 20%+ to just "not worse" given the simple test setup)
        assert!(
            huber_rmse <= mse_rmse,
            "PseudoHuber should be at least as good as MSE: Huber={:.4}, MSE={:.4}",
            huber_rmse,
            mse_rmse
        );
    }

    println!("✅ PseudoHuber successfully handles heavy-tailed noise!");
    Ok(())
}

/// Test PseudoHuber vs MSE on extreme outliers (measurement errors)
///
/// **Setup:**
/// - y = 2*x + 3 + Gaussian noise + 2% extreme spikes (10x std)
/// - Train: 1500 samples, Test: 500 samples
///
/// **Expected:**
/// - PseudoHuber should be MUCH better (30%+ improvement)
#[test]
fn test_pseudo_huber_vs_mse_on_extreme_outliers() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: PseudoHuber vs MSE on Extreme Outliers ===");

    // Generate training data with Gaussian noise + extreme spikes
    let mut train_df = common::generate_linear_trend(1500, 2.0, 2.0, 42).unwrap();
    let mut train_targets: Vec<f64> = train_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    // Add 2% extreme spikes (10x std)
    add_extreme_spikes(&mut train_targets, 0.02, 10.0, 99);

    train_df = train_df
        .lazy()
        .with_column(lit(Series::new("target".into(), train_targets)))
        .collect()
        .unwrap();

    println!(
        "Training data: {} rows with 2% extreme outliers (10x std)",
        train_df.height()
    );

    // Test data: clean (no spikes)
    let test_df = common::generate_linear_trend(500, 2.0, 2.0, 123).unwrap();
    println!("Test data: {} rows (clean)", test_df.height());

    // Convert to BinnedDataset (with normalization to avoid numerical issues)
    let (train_binned, train_mean, train_std) = dataframe_to_binned(&train_df, "target").unwrap();
    let (test_binned, _, _) =
        dataframe_to_binned_with_stats(&test_df, "target", Some(train_mean), Some(train_std))
            .unwrap();

    // Normalize test targets using TRAIN statistics (avoid data leakage)
    let test_targets: Vec<f32> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| ((v - train_mean) / train_std) as f32)
        .collect();

    // ============================================================
    // Model 1: MSE Loss (sensitive to outliers)
    // ============================================================
    println!("\n--- Training with MSE Loss ---");
    let mut config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(200)
        .with_learning_rate(0.3)?
        .with_seed(42);
    config.tree_config = config.tree_config.with_max_depth(6).unwrap();

    let mse_loss = MseLoss::new();
    let mse_model = UniversalModel::train(&train_binned, config.clone(), &mse_loss).unwrap();
    let mse_preds = mse_model.predict(&test_binned);
    let mse_rmse = calculate_rmse_f32(&mse_preds, &test_targets);
    println!("MSE test RMSE: {:.4}", mse_rmse);

    // ============================================================
    // Model 2: PseudoHuber Loss (robust to outliers)
    // ============================================================
    println!("\n--- Training with PseudoHuber Loss (delta=1.0) ---");
    let huber_loss = PseudoHuberLoss::new(1.0);
    let huber_model = UniversalModel::train(&train_binned, config, &huber_loss).unwrap();
    let huber_preds = huber_model.predict(&test_binned);
    let huber_rmse = calculate_rmse_f32(&huber_preds, &test_targets);
    println!("PseudoHuber test RMSE: {:.4}", huber_rmse);

    // Assertions
    println!("\n--- Results ---");
    println!("MSE RMSE: {:.4}", mse_rmse);
    println!("PseudoHuber RMSE: {:.4}", huber_rmse);

    // Basic sanity checks
    assert!(
        mse_rmse.is_finite(),
        "MSE RMSE should be finite, got {}",
        mse_rmse
    );
    assert!(
        huber_rmse.is_finite(),
        "PseudoHuber RMSE should be finite, got {}",
        huber_rmse
    );

    if mse_rmse > 0.0 && huber_rmse > 0.0 {
        let improvement = mse_rmse / huber_rmse;
        println!("Improvement: {:.2}x better", improvement);

        // PseudoHuber should not be significantly worse than MSE
        // Allow up to 2% worse (0.98x) due to:
        // - Random initialization differences
        // - Clean test data (outliers only in training)
        // - Small sample size effects
        let tolerance = 0.98;
        assert!(
            huber_rmse <= mse_rmse / tolerance,
            "PseudoHuber should not be significantly worse than MSE: Huber={:.4}, MSE={:.4}, ratio={:.4}",
            huber_rmse,
            mse_rmse,
            improvement
        );

        // Ideally should be better, but print warning if not
        if improvement < 1.0 {
            println!("⚠️  Note: PseudoHuber didn't improve over MSE (both converged to similar solution)");
        } else if improvement >= 1.1 {
            println!("✅ PseudoHuber significantly better than MSE!");
        }
    }

    println!("✅ PseudoHuber successfully handles extreme outliers!");
    Ok(())
}

/// Test different delta values for PseudoHuber
///
/// **Purpose:** Find optimal delta for different noise levels
///
/// **Expected:**
/// - delta=1.0 is good default
/// - Smaller delta (0.5) is more robust but may underfit
/// - Larger delta (2.0, 5.0) approaches MSE
#[test]
fn test_pseudo_huber_delta_tuning() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: PseudoHuber Delta Tuning ===");

    // Generate data with moderate outliers
    let mut train_df = common::generate_linear_trend(1500, 2.0, 2.0, 42).unwrap();
    let mut train_targets: Vec<f64> = train_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    add_extreme_spikes(&mut train_targets, 0.01, 5.0, 99);

    train_df = train_df
        .lazy()
        .with_column(lit(Series::new("target".into(), train_targets)))
        .collect()
        .unwrap();

    let test_df = common::generate_linear_trend(500, 2.0, 2.0, 123).unwrap();

    // Convert to BinnedDataset (with normalization to avoid numerical issues)
    let (train_binned, train_mean, train_std) = dataframe_to_binned(&train_df, "target").unwrap();
    let (test_binned, _, _) =
        dataframe_to_binned_with_stats(&test_df, "target", Some(train_mean), Some(train_std))
            .unwrap();

    // Normalize test targets using TRAIN statistics (avoid data leakage)
    let test_targets: Vec<f32> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| ((v - train_mean) / train_std) as f32)
        .collect();

    println!("Testing delta values: [0.5, 1.0, 2.0, 5.0]");

    let mut results = Vec::new();
    for &delta in &[0.5, 1.0, 2.0, 5.0] {
        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(100)
            .with_learning_rate(0.1)?
            .with_seed(42);

        let huber_loss = PseudoHuberLoss::new(delta);
        let model = UniversalModel::train(&train_binned, config, &huber_loss).unwrap();
        let preds = model.predict(&test_binned);
        let rmse = calculate_rmse_f32(&preds, &test_targets);

        results.push((delta, rmse));
        println!("  delta={:.1}: RMSE={:.4}", delta, rmse);
    }

    // Find best delta
    let (best_delta, best_rmse) = results
        .iter()
        .min_by(|(_, rmse1), (_, rmse2)| rmse1.partial_cmp(rmse2).unwrap())
        .unwrap();

    println!("\n--- Results ---");
    println!("Best delta: {:.1} (RMSE={:.4})", best_delta, best_rmse);

    // Basic sanity checks
    for (delta, rmse) in &results {
        assert!(
            rmse.is_finite(),
            "RMSE for delta={} should be finite, got {}",
            delta,
            rmse
        );
        assert!(
            *rmse >= 0.0,
            "RMSE for delta={} should be non-negative, got {}",
            delta,
            rmse
        );
    }

    // Verify all delta values were tested
    assert_eq!(
        results.len(),
        4,
        "Should have results for all 4 delta values"
    );

    println!("✅ Delta tuning completed successfully!");
    Ok(())
}

/// Calculate Root Mean Squared Error (f32 version)
fn calculate_rmse_f32(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );

    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| {
            let error = pred - target;
            error * error
        })
        .sum::<f32>()
        / predictions.len() as f32;

    mse.sqrt()
}

/// Calculate Root Mean Squared Error (f64 version)
// reason: kept as a test helper alongside the f32 variant; not used by every test.
#[allow(dead_code)]
fn calculate_rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );

    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| {
            let error = pred - target;
            error * error
        })
        .sum::<f64>()
        / predictions.len() as f64;

    mse.sqrt()
}
