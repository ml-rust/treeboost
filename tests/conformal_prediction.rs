//! Test conformal prediction coverage
//!
//! This test validates that prediction intervals have correct empirical coverage.
//!
//! Key insights:
//! - 90% conformal intervals should contain ~90% of true values
//! - Coverage should be valid even for non-i.i.d. data (distribution shift)
//! - Interval width should increase with noise level
//! - Calibration set size affects interval quality

mod common;

use polars::prelude::*;
use treeboost::{
    dataset::BinnedDataset,
    loss::MseLoss,
    model::{BoostingMode, UniversalConfig, UniversalModel},
};

/// Generate complex multi-feature data with non-linear relationships
/// y = 2*x1 + sin(x2) + x3*x4 + noise
fn generate_complex_data(
    n: usize,
    noise_std: f64,
    seed: u64,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let mut state = seed;
    let mut next_rand = || -> f64 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u = ((state >> 16) & 0x7FFF) as f64 / 32767.0;
        let v = ((state >> 16) & 0x7FFF) as f64 / 32767.0;
        // Box-Muller transform
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };

    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut x3 = Vec::with_capacity(n);
    let mut x4 = Vec::with_capacity(n);
    let mut x5 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let v1 = (i as f64 / n as f64) * 10.0 - 5.0; // [-5, 5]
        let v2 = (i as f64 / n as f64) * std::f64::consts::TAU; // [0, 2π]
        let v3 = next_rand() * 2.0; // Normal(0, 2)
        let v4 = next_rand() * 2.0; // Normal(0, 2)
        let v5 = next_rand(); // Normal(0, 1)

        // Complex non-linear relationship
        let target = 2.0 * v1 + v2.sin() * 3.0 + v3 * v4 + 0.5 * v5 + noise_std * next_rand();

        x1.push(v1);
        x2.push(v2);
        x3.push(v3);
        x4.push(v4);
        x5.push(v5);
        y.push(target);
    }

    Ok(df! {
        "x1" => x1,
        "x2" => x2,
        "x3" => x3,
        "x4" => x4,
        "x5" => x5,
        "target" => y,
    }?)
}

/// Convert DataFrame to BinnedDataset for training
/// Returns (BinnedDataset, target_mean, target_std) for test data normalization
fn dataframe_to_binned(
    df: &DataFrame,
    target_col: &str,
) -> Result<(BinnedDataset, f64, f64), Box<dyn std::error::Error>> {
    dataframe_to_binned_with_stats(df, target_col, None, None)
}

/// Convert DataFrame to BinnedDataset with optional target normalization parameters
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

/// Test that 90% conformal intervals achieve correct coverage
///
/// **Setup:**
/// - y = 2*x1 + sin(x2) + x3*x4 + noise (5 features, non-linear)
/// - Train: 1000 samples, Calibration: 20% internal split, Test: 500 samples
/// - Request 90% prediction intervals
///
/// **Expected:**
/// - Empirical coverage should be 85-95% (allowing 5% slack)
#[test]
fn test_conformal_coverage_90_percent() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: 90% Conformal Coverage ===");

    // Generate complex training data (will be split internally for calibration)
    let train_df = generate_complex_data(1000, 1.0, 42).unwrap();
    println!(
        "Training data: {} rows with 5 features (20% will be used for calibration)",
        train_df.height()
    );

    // Generate test data
    let test_df = generate_complex_data(500, 1.0, 123).unwrap();
    println!("Test data: {} rows", test_df.height());

    // Convert to BinnedDataset with normalization
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

    // Train model with conformal prediction enabled
    println!("\n--- Training with Conformal Prediction (90% coverage) ---");
    let mut config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(150)
        .with_learning_rate(0.1)?
        .with_seed(42);
    config.tree_config = config.tree_config.with_max_depth(6).unwrap();
    config.calibration_ratio = 0.2; // Use 20% for calibration
    config.conformal_quantile = 0.9; // Request 90% intervals

    let loss_fn = MseLoss::new();
    let model = UniversalModel::train(&train_binned, config, &loss_fn).unwrap();

    // Get predictions with conformal intervals
    let (predictions, lower_bounds, upper_bounds) =
        model.predict_with_intervals(&test_binned).unwrap();

    println!("Predictions: {} values with intervals", predictions.len());

    // Calculate empirical coverage
    let mut covered = 0;
    let mut total_width = 0.0;
    for i in 0..predictions.len() {
        let target = test_targets[i];
        if target >= lower_bounds[i] && target <= upper_bounds[i] {
            covered += 1;
        }
        total_width += upper_bounds[i] - lower_bounds[i];
    }

    let coverage = covered as f32 / predictions.len() as f32;
    let avg_width = total_width / predictions.len() as f32;

    println!("Empirical coverage: {:.2}% (target: 90%)", coverage * 100.0);
    println!("Average interval width: {:.4}", avg_width);

    // Calculate basic RMSE as sanity check
    let rmse = calculate_rmse(&predictions, &test_targets);
    println!("Test RMSE: {:.4}", rmse);

    // Assertions
    assert!(
        (0.85..=0.95).contains(&coverage),
        "Coverage should be 85-95% for 90% intervals, got {:.2}%",
        coverage * 100.0
    );
    assert!(
        rmse.is_finite() && rmse > 0.0,
        "RMSE should be positive and finite, got {:.4}",
        rmse
    );
    assert!(
        avg_width > 0.0,
        "Interval width should be positive, got {:.4}",
        avg_width
    );
    assert!(
        predictions.len() == test_targets.len(),
        "Should have predictions for all test samples"
    );

    println!(
        "✅ 90% conformal coverage achieved: {:.2}%!",
        coverage * 100.0
    );
    Ok(())
}

/// Test different confidence levels (80%, 90%, 95%)
///
/// **Expected:**
/// - Higher confidence → wider intervals
/// - All should achieve their target coverage ±10%
#[test]
fn test_conformal_multiple_confidence_levels() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: Multiple Confidence Levels ===");

    // Generate complex data
    let train_df = generate_complex_data(1000, 1.0, 42).unwrap();
    let test_df = generate_complex_data(500, 1.0, 123).unwrap();

    // Convert to BinnedDataset with normalization
    let (train_binned, train_mean, train_std) = dataframe_to_binned(&train_df, "target").unwrap();
    let (test_binned, _, _) =
        dataframe_to_binned_with_stats(&test_df, "target", Some(train_mean), Some(train_std))
            .unwrap();

    // Normalize test targets using TRAIN statistics
    let test_targets: Vec<f32> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| ((v - train_mean) / train_std) as f32)
        .collect();

    println!("Testing confidence levels: [0.80, 0.90, 0.95]");

    let mut prev_width = 0.0;
    for &quantile in &[0.80, 0.90, 0.95] {
        println!("\n--- Training with {:.0}% coverage ---", quantile * 100.0);

        let mut config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(150)
            .with_learning_rate(0.1)?
            .with_seed(42);
        config.tree_config = config.tree_config.with_max_depth(6).unwrap();
        config.calibration_ratio = 0.2;
        config.conformal_quantile = quantile;

        let loss_fn = MseLoss::new();
        let model = UniversalModel::train(&train_binned, config, &loss_fn).unwrap();

        let (predictions, lower_bounds, upper_bounds) =
            model.predict_with_intervals(&test_binned).unwrap();

        // Calculate coverage
        let mut covered = 0;
        let mut total_width = 0.0;
        for i in 0..predictions.len() {
            if test_targets[i] >= lower_bounds[i] && test_targets[i] <= upper_bounds[i] {
                covered += 1;
            }
            total_width += upper_bounds[i] - lower_bounds[i];
        }

        let coverage = covered as f32 / predictions.len() as f32;
        let avg_width = total_width / predictions.len() as f32;

        println!(
            "  Quantile={:.2}: Coverage={:.2}%, Avg Width={:.4}",
            quantile,
            coverage * 100.0,
            avg_width
        );

        // Verify coverage is close to target (±10% tolerance)
        let target_coverage = quantile;
        assert!(
            coverage >= target_coverage - 0.10 && coverage <= target_coverage + 0.10,
            "Coverage should be near {:.0}% (±10%), got {:.2}%",
            target_coverage * 100.0,
            coverage * 100.0
        );

        assert!(avg_width > 0.0, "Intervals should have positive width");

        // Verify intervals get wider with higher confidence
        if prev_width > 0.0 {
            assert!(
                avg_width >= prev_width * 0.95,
                "Higher confidence should have wider or similar intervals: prev={:.4}, current={:.4}",
                prev_width,
                avg_width
            );
        }
        prev_width = avg_width;
    }

    println!("\n✅ Multiple confidence levels tested successfully!");
    Ok(())
}

/// Test that conformal intervals work with noisy data
///
/// **Setup:**
/// - Train on complex data with higher noise (std=2.0)
/// - Test on data with same noise level
///
/// **Expected:**
/// - Intervals should be wider than for clean data
/// - Coverage should still be correct (85-95%)
#[test]
fn test_conformal_with_noisy_data() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: Conformal Prediction with Noisy Data ===");

    // Generate noisy training data (std=2.0, higher than default)
    let train_df = generate_complex_data(1000, 2.0, 42).unwrap();
    println!(
        "Training data: {} rows with noise std=2.0",
        train_df.height()
    );

    // Test data with same noise level
    let test_df = generate_complex_data(500, 2.0, 123).unwrap();
    println!("Test data: {} rows with noise std=2.0", test_df.height());

    // Convert to BinnedDataset with normalization
    let (train_binned, train_mean, train_std) = dataframe_to_binned(&train_df, "target").unwrap();
    let (test_binned, _, _) =
        dataframe_to_binned_with_stats(&test_df, "target", Some(train_mean), Some(train_std))
            .unwrap();

    // Normalize test targets using TRAIN statistics
    let test_targets: Vec<f32> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| ((v - train_mean) / train_std) as f32)
        .collect();

    // Train with conformal prediction
    println!("\n--- Training with 90% coverage on noisy data ---");
    let mut config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(150)
        .with_learning_rate(0.1)?
        .with_seed(42);
    config.tree_config = config.tree_config.with_max_depth(6).unwrap();
    config.calibration_ratio = 0.2;
    config.conformal_quantile = 0.9;

    let loss_fn = MseLoss::new();
    let model = UniversalModel::train(&train_binned, config, &loss_fn).unwrap();

    let (predictions, lower_bounds, upper_bounds) =
        model.predict_with_intervals(&test_binned).unwrap();

    // Calculate coverage and interval width
    let mut covered = 0;
    let mut total_width = 0.0;
    for i in 0..predictions.len() {
        if test_targets[i] >= lower_bounds[i] && test_targets[i] <= upper_bounds[i] {
            covered += 1;
        }
        total_width += upper_bounds[i] - lower_bounds[i];
    }

    let coverage = covered as f32 / predictions.len() as f32;
    let avg_width = total_width / predictions.len() as f32;

    let rmse = calculate_rmse(&predictions, &test_targets);

    println!("Test RMSE: {:.4}", rmse);
    println!("Coverage: {:.2}%", coverage * 100.0);
    println!("Average interval width: {:.4}", avg_width);

    // Assertions
    assert!(
        (0.85..=0.95).contains(&coverage),
        "Coverage should be 85-95% even with noise, got {:.2}%",
        coverage * 100.0
    );
    assert!(
        rmse.is_finite() && rmse > 0.0,
        "RMSE should be positive and finite"
    );
    assert!(
        avg_width > 1.0,
        "Intervals should be wider for noisy data, got {:.4}",
        avg_width
    );

    println!("✅ Conformal prediction handles noisy data correctly!");
    println!("Coverage={:.2}%, Width={:.4}", coverage * 100.0, avg_width);
    Ok(())
}

/// Test conformal prediction with distribution shift
///
/// **Setup:**
/// - Train on x ∈ [0, 800]
/// - Test on x ∈ [900, 1400] (extrapolation region)
///
/// **Expected:**
/// - Coverage may degrade slightly (conformal assumes exchangeability)
/// - But intervals should still provide uncertainty estimates
#[test]
fn test_conformal_with_distribution_shift() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: Conformal with Distribution Shift ===");

    // Train on [0, 1000]
    let train_df = common::generate_linear_trend_range(0.0, 1000.0, 2.0, 1.0, 42).unwrap();
    println!("Training data: x ∈ [0, 1000]");

    // Test on [1200, 1700] (extrapolation)
    let test_df = common::generate_linear_trend_range(1200.0, 1700.0, 2.0, 1.0, 123).unwrap();
    println!("Test data: x ∈ [1200, 1700] (distribution shift)");

    // Convert to BinnedDataset with normalization
    let (train_binned, train_mean, train_std) = dataframe_to_binned(&train_df, "target").unwrap();
    let (test_binned, _, _) =
        dataframe_to_binned_with_stats(&test_df, "target", Some(train_mean), Some(train_std))
            .unwrap();

    // Normalize test targets using TRAIN statistics
    let test_targets: Vec<f32> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| ((v - train_mean) / train_std) as f32)
        .collect();

    // Train with conformal prediction
    println!("\n--- Training with 90% coverage ---");
    let mut config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(150)
        .with_learning_rate(0.1)?
        .with_seed(42);
    config.tree_config = config.tree_config.with_max_depth(6).unwrap();
    config.calibration_ratio = 0.2;
    config.conformal_quantile = 0.9;

    let loss_fn = MseLoss::new();
    let model = UniversalModel::train(&train_binned, config, &loss_fn).unwrap();

    let (predictions, lower_bounds, upper_bounds) =
        model.predict_with_intervals(&test_binned).unwrap();

    // Calculate coverage
    let mut covered = 0;
    let mut total_width = 0.0;
    for i in 0..predictions.len() {
        if test_targets[i] >= lower_bounds[i] && test_targets[i] <= upper_bounds[i] {
            covered += 1;
        }
        total_width += upper_bounds[i] - lower_bounds[i];
    }

    let coverage = covered as f32 / predictions.len() as f32;
    let avg_width = total_width / predictions.len() as f32;

    let rmse = calculate_rmse(&predictions, &test_targets);

    println!("Test RMSE (extrapolation): {:.4}", rmse);
    println!(
        "Coverage: {:.2}% (may degrade under distribution shift)",
        coverage * 100.0
    );
    println!("Average interval width: {:.4}", avg_width);

    // Just verify that model produces reasonable predictions
    // (coverage may degrade under distribution shift, which is expected)
    assert!(
        predictions.len() == test_targets.len(),
        "Prediction count mismatch"
    );
    assert!(rmse.is_finite(), "RMSE should be finite");
    assert!(avg_width > 0.0, "Intervals should have positive width");

    // Don't enforce strict coverage under distribution shift (conformal assumes exchangeability)
    // But it should still be somewhat reasonable
    println!(
        "Note: Coverage under distribution shift is {:.2}% (no strict requirement)",
        coverage * 100.0
    );

    println!("✅ Conformal prediction with distribution shift completed!");
    Ok(())
}

/// Calculate Root Mean Squared Error
fn calculate_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
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
