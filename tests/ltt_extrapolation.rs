//! Test LTT (LinearThenTree) extrapolation capability
//!
//! This test validates that LinearThenTree mode can extrapolate beyond the training range,
//! while PureTree mode cannot (flatlines at the max training value).
//!
//! Key insight: Trees learn piecewise constants and cannot extrapolate. The linear component
//! in LTT should capture the trend and enable extrapolation.

mod common;

use treeboost::{
    learner::{LinearConfig, TreeConfig},
    model::{AutoConfig, AutoModel, BoostingMode, TuningLevel, UniversalConfig},
};

/// Test that LTT extrapolates linear trends while PureTree flatlines
///
/// **Setup:**
/// - Generate: y = 3.5 * x + noise
/// - Train: x ∈ [0, 100]
/// - Test: x ∈ [100, 200] (extrapolation range)
///
/// **Expected:**
/// - LTT: Should extrapolate the linear trend (low RMSE)
/// - PureTree: Should flatline at max training value (high RMSE)
/// - LTT test RMSE < PureTree test RMSE * 0.5 (2x better)
#[test]
fn test_ltt_extrapolates_linear_trend() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: LTT Extrapolation vs PureTree Flatline ===");

    // Generate training data: x ∈ [0, 100], y = 3.5 * x + noise
    let train_df = common::generate_linear_trend(100, 3.5, 5.0, 42).unwrap();
    println!("Training data: {} rows, x ∈ [0, 100]", train_df.height());

    // Generate test data: x ∈ [100, 200] (extrapolation range)
    let test_df = common::generate_linear_trend_range(100.0, 200.0, 3.5, 5.0, 123).unwrap();
    println!("Test data: {} rows, x ∈ [100, 200]", test_df.height());

    // Get true targets for RMSE calculation
    let test_targets: Vec<f64> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    // ============================================================
    // Model 1: LinearThenTree (should extrapolate)
    // ============================================================

    let linear_config = LinearConfig::default()
        .with_preset(treeboost::LinearPreset::Ridge)
        .with_lambda(0.01) // Light regularization
        .expect("valid lambda")
        .with_shrinkage_factor(1.0)
        .expect("valid shrinkage")
        .with_max_iter(500)
        .expect("valid max_iter");

    let tree_config = TreeConfig::default()
        .with_max_depth(4)
        .expect("valid max_depth")
        .with_min_samples_leaf(5)
        .expect("valid min_samples_leaf");

    let ltt_univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_config(linear_config.clone())
        .with_tree_config(tree_config.clone())
        .with_num_rounds(50)
        .with_learning_rate(0.1)?;

    let ltt_config = AutoConfig::new()
        .with_feature_engineering(treeboost::model::FeatureEngineeringMode::None)
        .with_tuning(TuningLevel::None)
        .with_custom_config(ltt_univ_config);

    println!("\n--- Training LinearThenTree model ---");
    let ltt_model = AutoModel::train_with_config(&train_df, "target", ltt_config).unwrap();

    // Predict on extrapolation range
    let ltt_preds_f32 = ltt_model.predict(&test_df).unwrap();
    let ltt_preds: Vec<f64> = ltt_preds_f32.iter().map(|&x| x as f64).collect();

    // Calculate RMSE
    let ltt_rmse = calculate_rmse(&ltt_preds, &test_targets);
    println!("LTT extrapolation RMSE: {:.4}", ltt_rmse);

    // ============================================================
    // Model 2: PureTree (should flatline)
    // ============================================================

    let tree_univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::PureTree)
        .with_tree_config(tree_config)
        .with_num_rounds(50)
        .with_learning_rate(0.1)?;

    let tree_config = AutoConfig::new()
        .with_feature_engineering(treeboost::model::FeatureEngineeringMode::None)
        .with_tuning(TuningLevel::None)
        .with_custom_config(tree_univ_config);

    println!("\n--- Training PureTree model ---");
    let tree_model = AutoModel::train_with_config(&train_df, "target", tree_config).unwrap();

    // Predict on extrapolation range
    let tree_preds_f32 = tree_model.predict(&test_df).unwrap();
    let tree_preds: Vec<f64> = tree_preds_f32.iter().map(|&x| x as f64).collect();

    // Calculate RMSE
    let tree_rmse = calculate_rmse(&tree_preds, &test_targets);
    println!("PureTree extrapolation RMSE: {:.4}", tree_rmse);

    // ============================================================
    // Assertions
    // ============================================================

    println!("\n--- Results ---");
    println!("LTT RMSE: {:.4}", ltt_rmse);
    println!("PureTree RMSE: {:.4}", tree_rmse);
    println!("Improvement ratio: {:.2}x better", tree_rmse / ltt_rmse);

    // LTT should extrapolate (low error)
    assert!(
        ltt_rmse < 50.0,
        "LTT should extrapolate with reasonable error, got RMSE = {}",
        ltt_rmse
    );

    // PureTree should flatline (high error)
    assert!(
        tree_rmse > 100.0,
        "PureTree should fail to extrapolate (flatline), got RMSE = {}",
        tree_rmse
    );

    // LTT should be significantly better (at least 2x)
    assert!(
        ltt_rmse < tree_rmse * 0.5,
        "LTT should be 2x better at extrapolation: LTT RMSE = {:.4}, PureTree RMSE = {:.4}",
        ltt_rmse,
        tree_rmse
    );

    println!("✅ LTT successfully extrapolates linear trends!");
    Ok(())
}

/// Test LTT with trend + seasonality (more realistic scenario)
///
/// **Setup:**
/// - Generate: y = 3.5 * x + 10 * sin(x / 10) + noise
/// - Train: x ∈ [0, 100]
/// - Test: x ∈ [100, 200]
///
/// **Expected:**
/// - LTT should still extrapolate better than PureTree
/// - Gap may be smaller than pure linear case (seasonality is harder)
#[test]
fn test_ltt_extrapolates_with_seasonality() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test: LTT Extrapolation with Seasonality ===");

    // Generate training data: y = 3.5 * x + 10 * sin(x / 10) + noise
    let train_df =
        common::generate_linear_trend_with_seasonality(100, 3.5, 10.0, 10.0, 5.0, 42).unwrap();
    println!(
        "Training data: {} rows, y = 3.5*x + 10*sin(x/10) + noise",
        train_df.height()
    );

    // Generate test data: x ∈ [100, 200]
    let test_df = common::generate_linear_trend_with_seasonality_range(
        100.0, 200.0, 3.5, 10.0, 10.0, 5.0, 123,
    )
    .unwrap();

    let test_targets: Vec<f64> = test_df
        .column("target")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    // Configure models
    let linear_config = LinearConfig::default()
        .with_preset(treeboost::LinearPreset::Ridge)
        .with_lambda(0.01)
        .expect("valid lambda")
        .with_shrinkage_factor(1.0)
        .expect("valid shrinkage")
        .with_max_iter(500)
        .expect("valid max_iter");

    let tree_config = TreeConfig::default()
        .with_max_depth(4)
        .expect("valid max_depth")
        .with_min_samples_leaf(5)
        .expect("valid min_samples_leaf");

    // Train LTT
    let ltt_univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_config(linear_config.clone())
        .with_tree_config(tree_config.clone())
        .with_num_rounds(50)
        .with_learning_rate(0.1)?;

    let ltt_config = AutoConfig::new()
        .with_feature_engineering(treeboost::model::FeatureEngineeringMode::None)
        .with_tuning(TuningLevel::None)
        .with_custom_config(ltt_univ_config);

    println!("\n--- Training LinearThenTree ---");
    let ltt_model = AutoModel::train_with_config(&train_df, "target", ltt_config).unwrap();
    let ltt_preds_f32 = ltt_model.predict(&test_df).unwrap();
    let ltt_preds: Vec<f64> = ltt_preds_f32.iter().map(|&x| x as f64).collect();
    let ltt_rmse = calculate_rmse(&ltt_preds, &test_targets);

    // Train PureTree
    let tree_univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::PureTree)
        .with_tree_config(tree_config)
        .with_num_rounds(50)
        .with_learning_rate(0.1)?;

    let tree_config = AutoConfig::new()
        .with_feature_engineering(treeboost::model::FeatureEngineeringMode::None)
        .with_tuning(TuningLevel::None)
        .with_custom_config(tree_univ_config);

    println!("--- Training PureTree ---");
    let tree_model = AutoModel::train_with_config(&train_df, "target", tree_config).unwrap();
    let tree_preds_f32 = tree_model.predict(&test_df).unwrap();
    let tree_preds: Vec<f64> = tree_preds_f32.iter().map(|&x| x as f64).collect();
    let tree_rmse = calculate_rmse(&tree_preds, &test_targets);

    println!("\n--- Results ---");
    println!("LTT RMSE: {:.4}", ltt_rmse);
    println!("PureTree RMSE: {:.4}", tree_rmse);
    println!("Improvement ratio: {:.2}x better", tree_rmse / ltt_rmse);

    // LTT should still be significantly better (though gap may be smaller than pure linear)
    assert!(
        ltt_rmse < tree_rmse * 0.7,
        "LTT should extrapolate better even with seasonality: LTT = {:.4}, PureTree = {:.4}",
        ltt_rmse,
        tree_rmse
    );

    println!("✅ LTT successfully extrapolates trends with seasonality!");
    Ok(())
}

/// Calculate Root Mean Squared Error
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
