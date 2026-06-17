//! Correctness benchmark: Verify TreeBoost predictions match competitor implementations
//!
//! This benchmark tests that all GBDT implementations produce similar predictions
//! on the same data, verifying algorithmic correctness.

use rand::prelude::*;
use std::time::Instant;

// TreeBoost imports
use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

// Competitor imports
use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};
use gbdt::config::Config;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;

/// Generate synthetic regression dataset with known pattern
fn generate_regression_data(
    num_rows: usize,
    num_features: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut features = Vec::with_capacity(num_rows * num_features);
    let mut targets = Vec::with_capacity(num_rows);

    for _ in 0..num_rows {
        let mut row_sum = 0.0;
        for f in 0..num_features {
            let val: f64 = rng.random_range(0.0..10.0);
            features.push(val);
            // Target is weighted sum of features with some noise
            row_sum += val * (f as f64 + 1.0) * 0.1;
        }
        let noise: f64 = rng.random_range(-0.5..0.5);
        targets.push(row_sum + noise);
    }

    (features, targets)
}

/// Convert to TreeBoost format (column-major binned)
fn to_treeboost_dataset(
    features: &[f64],
    targets: &[f64],
    num_rows: usize,
    num_features: usize,
) -> BinnedDataset {
    let mut all_binned = Vec::with_capacity(num_rows * num_features);
    let mut all_info = Vec::with_capacity(num_features);

    for f in 0..num_features {
        let mut col_values: Vec<f64> = (0..num_rows)
            .map(|r| features[r * num_features + f])
            .collect();

        col_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let num_bins = 255usize;
        let mut boundaries = Vec::with_capacity(num_bins - 1);
        for i in 1..num_bins {
            let idx = (i * col_values.len()) / num_bins;
            let val = col_values[idx.min(col_values.len() - 1)];
            if boundaries.is_empty() || val > *boundaries.last().unwrap() {
                boundaries.push(val);
            }
        }

        for r in 0..num_rows {
            let val = features[r * num_features + f];
            let bin = boundaries
                .binary_search_by(|b| b.partial_cmp(&val).unwrap())
                .unwrap_or_else(|i| i) as u8;
            all_binned.push(bin);
        }

        all_info.push(FeatureInfo {
            name: format!("f{}", f),
            feature_type: FeatureType::Numeric,
            num_bins: (boundaries.len() + 1).min(255) as u8,
            bin_boundaries: boundaries,
            impute_value: 0.0,
        });
    }

    let targets_f32: Vec<f32> = targets.iter().map(|&t| t as f32).collect();
    BinnedDataset::new(num_rows, all_binned, targets_f32, all_info)
}

/// Convert to gbdt-rs format
fn to_gbdt_data(
    features: &[f64],
    targets: &[f64],
    num_rows: usize,
    num_features: usize,
) -> Vec<Data> {
    (0..num_rows)
        .map(|r| {
            let row_features: Vec<f32> = (0..num_features)
                .map(|f| features[r * num_features + f] as f32)
                .collect();
            Data::new_training_data(row_features, 1.0, targets[r] as f32, None)
        })
        .collect()
}

/// Convert to forust format (column-major f64)
fn to_forust_matrix(features: &[f64], num_rows: usize, num_features: usize) -> Vec<f64> {
    let mut col_major = vec![0.0; num_rows * num_features];
    for r in 0..num_rows {
        for f in 0..num_features {
            col_major[f * num_rows + r] = features[r * num_features + f];
        }
    }
    col_major
}

/// Compute Mean Squared Error
fn mse(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n
}

/// Compute Root Mean Squared Error
fn rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    mse(predictions, targets).sqrt()
}

/// Compute Mean Absolute Error
fn mae(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n
}

/// Compute R² (coefficient of determination)
fn r_squared(predictions: &[f64], targets: &[f64]) -> f64 {
    let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = targets.iter().map(|t| (t - mean_target).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}

/// Compute correlation coefficient between two prediction sets
fn correlation(pred_a: &[f64], pred_b: &[f64]) -> f64 {
    let n = pred_a.len() as f64;
    let mean_a = pred_a.iter().sum::<f64>() / n;
    let mean_b = pred_b.iter().sum::<f64>() / n;

    let cov: f64 = pred_a
        .iter()
        .zip(pred_b.iter())
        .map(|(a, b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>()
        / n;

    let std_a = (pred_a.iter().map(|a| (a - mean_a).powi(2)).sum::<f64>() / n).sqrt();
    let std_b = (pred_b.iter().map(|b| (b - mean_b).powi(2)).sum::<f64>() / n).sqrt();

    if std_a == 0.0 || std_b == 0.0 {
        0.0
    } else {
        cov / (std_a * std_b)
    }
}

/// Compute max absolute difference between predictions
fn max_abs_diff(pred_a: &[f64], pred_b: &[f64]) -> f64 {
    pred_a
        .iter()
        .zip(pred_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn print_separator() {
    println!("{}", "=".repeat(80));
}

fn print_header(title: &str) {
    println!();
    print_separator();
    println!("{:^80}", title);
    print_separator();
}

fn main() {
    println!("\n🔬 TREEBOOST CORRECTNESS VERIFICATION BENCHMARK\n");
    println!("This benchmark verifies that TreeBoost produces correct predictions");
    println!("by comparing against gbdt-rs and forust implementations.\n");

    // Configuration
    let num_features = 10;
    let num_rounds = 50;
    let max_depth = 6;
    let learning_rate = 0.1;
    let train_rows = 5_000;
    let test_rows = 1_000;

    println!("Configuration:");
    println!("  Features:      {}", num_features);
    println!("  Rounds:        {}", num_rounds);
    println!("  Max Depth:     {}", max_depth);
    println!("  Learning Rate: {}", learning_rate);
    println!("  Train Rows:    {}", train_rows);
    println!("  Test Rows:     {}", test_rows);

    // Generate data
    print_header("DATA GENERATION");
    let (train_features, train_targets) = generate_regression_data(train_rows, num_features, 42);
    let (test_features, test_targets) = generate_regression_data(test_rows, num_features, 123);

    println!("Training data statistics:");
    let train_mean = train_targets.iter().sum::<f64>() / train_targets.len() as f64;
    let train_std = (train_targets
        .iter()
        .map(|t| (t - train_mean).powi(2))
        .sum::<f64>()
        / train_targets.len() as f64)
        .sqrt();
    println!("  Target mean: {:.4}", train_mean);
    println!("  Target std:  {:.4}", train_std);
    println!(
        "  Target range: [{:.4}, {:.4}]",
        train_targets.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        train_targets
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Train TreeBoost
    print_header("TRAINING TREEBOOST");
    let treeboost_train =
        to_treeboost_dataset(&train_features, &train_targets, train_rows, num_features);
    let treeboost_config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(max_depth)
        .with_learning_rate(learning_rate)
        .with_min_samples_leaf(5);

    let start = Instant::now();
    let treeboost_model = GBDTModel::train_binned(&treeboost_train, treeboost_config).unwrap();
    let treeboost_train_time = start.elapsed();
    println!("✓ TreeBoost trained in {:?}", treeboost_train_time);
    println!("  Trees: {}", treeboost_model.num_trees());

    // Train gbdt-rs
    print_header("TRAINING GBDT-RS");
    let mut gbdt_train = to_gbdt_data(&train_features, &train_targets, train_rows, num_features);
    let mut gbdt_cfg = Config::new();
    gbdt_cfg.set_feature_size(num_features);
    gbdt_cfg.set_max_depth(max_depth as u32);
    gbdt_cfg.set_iterations(num_rounds);
    gbdt_cfg.set_shrinkage(learning_rate);
    gbdt_cfg.set_loss("SquaredError");
    gbdt_cfg.set_min_leaf_size(5);

    let start = Instant::now();
    let mut gbdt_model = GBDT::new(&gbdt_cfg);
    gbdt_model.fit(&mut gbdt_train);
    let gbdt_train_time = start.elapsed();
    println!("✓ gbdt-rs trained in {:?}", gbdt_train_time);

    // Train forust
    print_header("TRAINING FORUST");
    let forust_train_features = to_forust_matrix(&train_features, train_rows, num_features);
    let forust_train_matrix = Matrix::new(&forust_train_features, train_rows, num_features);

    let start = Instant::now();
    let mut forust_model = GradientBooster::default()
        .set_objective_type(ObjectiveType::SquaredLoss)
        .set_iterations(num_rounds)
        .set_max_depth(max_depth)
        .set_learning_rate(learning_rate)
        .set_min_leaf_weight(5.0)
        .set_parallel(false);
    forust_model
        .fit_unweighted(&forust_train_matrix, &train_targets, None)
        .unwrap();
    let forust_train_time = start.elapsed();
    println!("✓ forust trained in {:?}", forust_train_time);

    // Generate predictions on test set
    print_header("PREDICTION");

    // TreeBoost predictions
    let treeboost_test =
        to_treeboost_dataset(&test_features, &test_targets, test_rows, num_features);
    let start = Instant::now();
    let treeboost_preds: Vec<f64> = treeboost_model
        .predict(&treeboost_test)
        .iter()
        .map(|&p| p as f64)
        .collect();
    let treeboost_pred_time = start.elapsed();
    println!("TreeBoost prediction: {:?}", treeboost_pred_time);

    // gbdt-rs predictions
    let gbdt_test = to_gbdt_data(&test_features, &test_targets, test_rows, num_features);
    let start = Instant::now();
    let gbdt_preds: Vec<f64> = gbdt_model
        .predict(&gbdt_test)
        .iter()
        .map(|&p| p as f64)
        .collect();
    let gbdt_pred_time = start.elapsed();
    println!("gbdt-rs prediction: {:?}", gbdt_pred_time);

    // forust predictions
    let forust_test_features = to_forust_matrix(&test_features, test_rows, num_features);
    let forust_test_matrix = Matrix::new(&forust_test_features, test_rows, num_features);
    let start = Instant::now();
    let forust_preds: Vec<f64> = forust_model.predict(&forust_test_matrix, false);
    let forust_pred_time = start.elapsed();
    println!("forust prediction: {:?}", forust_pred_time);

    // Evaluate accuracy against ground truth
    print_header("ACCURACY VS GROUND TRUTH");

    println!(
        "\n{:^20} {:>12} {:>12} {:>12} {:>12}",
        "Implementation", "MSE", "RMSE", "MAE", "R²"
    );
    println!("{}", "-".repeat(72));

    let tb_mse = mse(&treeboost_preds, &test_targets);
    let tb_rmse = rmse(&treeboost_preds, &test_targets);
    let tb_mae = mae(&treeboost_preds, &test_targets);
    let tb_r2 = r_squared(&treeboost_preds, &test_targets);
    println!(
        "{:^20} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "TreeBoost", tb_mse, tb_rmse, tb_mae, tb_r2
    );

    let gbdt_mse = mse(&gbdt_preds, &test_targets);
    let gbdt_rmse = rmse(&gbdt_preds, &test_targets);
    let gbdt_mae = mae(&gbdt_preds, &test_targets);
    let gbdt_r2 = r_squared(&gbdt_preds, &test_targets);
    println!(
        "{:^20} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "gbdt-rs", gbdt_mse, gbdt_rmse, gbdt_mae, gbdt_r2
    );

    let for_mse = mse(&forust_preds, &test_targets);
    let for_rmse = rmse(&forust_preds, &test_targets);
    let for_mae = mae(&forust_preds, &test_targets);
    let for_r2 = r_squared(&forust_preds, &test_targets);
    println!(
        "{:^20} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "forust", for_mse, for_rmse, for_mae, for_r2
    );

    // Cross-implementation comparison
    print_header("CROSS-IMPLEMENTATION AGREEMENT");

    println!("\nPrediction correlation matrix:");
    println!(
        "{:^20} {:>15} {:>15} {:>15}",
        "", "TreeBoost", "gbdt-rs", "forust"
    );
    println!("{}", "-".repeat(65));

    let corr_tb_gbdt = correlation(&treeboost_preds, &gbdt_preds);
    let corr_tb_for = correlation(&treeboost_preds, &forust_preds);
    let corr_gbdt_for = correlation(&gbdt_preds, &forust_preds);

    println!(
        "{:^20} {:>15.6} {:>15.6} {:>15.6}",
        "TreeBoost", 1.0, corr_tb_gbdt, corr_tb_for
    );
    println!(
        "{:^20} {:>15.6} {:>15.6} {:>15.6}",
        "gbdt-rs", corr_tb_gbdt, 1.0, corr_gbdt_for
    );
    println!(
        "{:^20} {:>15.6} {:>15.6} {:>15.6}",
        "forust", corr_tb_for, corr_gbdt_for, 1.0
    );

    println!("\nMax absolute prediction difference:");
    println!(
        "  TreeBoost vs gbdt-rs: {:.6}",
        max_abs_diff(&treeboost_preds, &gbdt_preds)
    );
    println!(
        "  TreeBoost vs forust:  {:.6}",
        max_abs_diff(&treeboost_preds, &forust_preds)
    );
    println!(
        "  gbdt-rs vs forust:    {:.6}",
        max_abs_diff(&gbdt_preds, &forust_preds)
    );

    // Prediction distribution comparison
    print_header("PREDICTION DISTRIBUTION");

    fn stats(preds: &[f64]) -> (f64, f64, f64, f64) {
        let n = preds.len() as f64;
        let mean = preds.iter().sum::<f64>() / n;
        let std = (preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n).sqrt();
        let min = preds.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = preds.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (mean, std, min, max)
    }

    println!(
        "\n{:^20} {:>12} {:>12} {:>12} {:>12}",
        "Implementation", "Mean", "Std", "Min", "Max"
    );
    println!("{}", "-".repeat(72));

    let (tb_mean, tb_std, tb_min, tb_max) = stats(&treeboost_preds);
    println!(
        "{:^20} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
        "TreeBoost", tb_mean, tb_std, tb_min, tb_max
    );

    let (gb_mean, gb_std, gb_min, gb_max) = stats(&gbdt_preds);
    println!(
        "{:^20} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
        "gbdt-rs", gb_mean, gb_std, gb_min, gb_max
    );

    let (fo_mean, fo_std, fo_min, fo_max) = stats(&forust_preds);
    println!(
        "{:^20} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
        "forust", fo_mean, fo_std, fo_min, fo_max
    );

    let (tgt_mean, tgt_std, tgt_min, tgt_max) = stats(&test_targets);
    println!(
        "{:^20} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
        "Ground Truth", tgt_mean, tgt_std, tgt_min, tgt_max
    );

    // Sample predictions
    print_header("SAMPLE PREDICTIONS (First 10 rows)");

    println!(
        "\n{:>6} {:>12} {:>12} {:>12} {:>12}",
        "Row", "Truth", "TreeBoost", "gbdt-rs", "forust"
    );
    println!("{}", "-".repeat(60));

    for i in 0..10.min(test_rows) {
        println!(
            "{:>6} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            i, test_targets[i], treeboost_preds[i], gbdt_preds[i], forust_preds[i]
        );
    }

    // Final verdict
    print_header("CORRECTNESS VERDICT");

    let min_correlation = 0.95;
    let max_r2_diff = 0.05;

    let corr_pass = corr_tb_gbdt >= min_correlation && corr_tb_for >= min_correlation;
    let r2_pass = (tb_r2 - gbdt_r2).abs() <= max_r2_diff && (tb_r2 - for_r2).abs() <= max_r2_diff;

    println!("\nVerification criteria:");
    println!(
        "  Correlation ≥ {:.2}: {} (TreeBoost-gbdt: {:.4}, TreeBoost-forust: {:.4})",
        min_correlation,
        if corr_pass { "✓ PASS" } else { "✗ FAIL" },
        corr_tb_gbdt,
        corr_tb_for
    );
    println!(
        "  R² difference ≤ {:.2}: {} (vs gbdt: {:.4}, vs forust: {:.4})",
        max_r2_diff,
        if r2_pass { "✓ PASS" } else { "✗ FAIL" },
        (tb_r2 - gbdt_r2).abs(),
        (tb_r2 - for_r2).abs()
    );

    println!();
    if corr_pass && r2_pass {
        println!("🎉 OVERALL: ✓ PASS - TreeBoost produces correct predictions!");
        println!("   Predictions are highly correlated with competitor implementations.");
    } else {
        println!("⚠️  OVERALL: NEEDS INVESTIGATION");
        if !corr_pass {
            println!("   - Prediction correlation is lower than expected");
        }
        if !r2_pass {
            println!("   - R² difference with competitors is higher than expected");
        }
    }

    // Relative performance summary
    print_header("RELATIVE PERFORMANCE SUMMARY");

    println!("\nAccuracy ranking (by R²):");
    let mut rankings = [
        ("TreeBoost", tb_r2),
        ("gbdt-rs", gbdt_r2),
        ("forust", for_r2),
    ];
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, (name, r2)) in rankings.iter().enumerate() {
        println!("  {}. {} (R² = {:.6})", i + 1, name, r2);
    }

    println!("\nTraining time ranking:");
    let mut train_times = [
        ("TreeBoost", treeboost_train_time),
        ("gbdt-rs", gbdt_train_time),
        ("forust", forust_train_time),
    ];
    train_times.sort_by_key(|a| a.1);
    for (i, (name, time)) in train_times.iter().enumerate() {
        println!("  {}. {} ({:?})", i + 1, name, time);
    }

    println!("\nPrediction time ranking:");
    let mut pred_times = [
        ("TreeBoost", treeboost_pred_time),
        ("gbdt-rs", gbdt_pred_time),
        ("forust", forust_pred_time),
    ];
    pred_times.sort_by_key(|a| a.1);
    for (i, (name, time)) in pred_times.iter().enumerate() {
        println!("  {}. {} ({:?})", i + 1, name, time);
    }

    print_separator();
    println!();
}
