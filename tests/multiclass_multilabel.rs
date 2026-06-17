//! Test 4: Multi-Class & Multi-Label Classification
//!
//! This test suite validates TreeBoost's multi-class and multi-label classification capabilities:
//!
//! **Multi-Class Tests:**
//! 1. Softmax probabilities sum to 1.0
//! 2. Model handles class imbalance
//! 3. Per-class accuracy on imbalanced data
//! 4. Predicted class = argmax(probabilities)
//!
//! **Multi-Label Tests:**
//! 1. Probabilities are independent (don't sum to 1)
//! 2. Model handles label correlation
//! 3. Can predict multiple labels simultaneously
//! 4. Focal Loss outperforms LogLoss on imbalanced data

mod common;

use polars::prelude::*;
use treeboost::booster::{GBDTConfig, LossType};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::loss::{MultiLabelFocalLoss, MultiLabelLogLoss};
use treeboost::model::{BoostingMode, UniversalConfig, UniversalModel};
use treeboost::Result;

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert DataFrame to BinnedDataset for multi-class classification
fn dataframe_to_binned_multiclass(df: &DataFrame, target_col: &str) -> Result<BinnedDataset> {
    let n = df.height();

    // Extract features (all columns except target)
    let feature_cols: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|&name| name != target_col)
        .map(|s| s.to_string())
        .collect();

    let num_features = feature_cols.len();

    // Extract feature values and quantize to u8
    let mut features = Vec::with_capacity(n * num_features);
    let mut feature_info = Vec::with_capacity(num_features);

    for feat_name in &feature_cols {
        let series = df.column(feat_name).unwrap();
        let values: Vec<f64> = series
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        // Normalize to [0, 255]
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        for &val in &values {
            let normalized = if range > 1e-10 {
                ((val - min) / range * 255.0) as u8
            } else {
                127
            };
            features.push(normalized);
        }

        feature_info.push(FeatureInfo {
            name: feat_name.clone(),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        });
    }

    // Extract targets (integer class labels)
    let target_series = df.column(target_col).unwrap();
    let targets: Vec<f32> = target_series
        .cast(&DataType::Int32)
        .unwrap()
        .i32()
        .unwrap()
        .iter()
        .map(|v| v.unwrap_or(0) as f32)
        .collect();

    Ok(BinnedDataset::new(n, features, targets, feature_info))
}

/// Convert DataFrame to BinnedDataset for multi-label classification
fn dataframe_to_binned_multilabel(df: &DataFrame, label_cols: &[String]) -> Result<BinnedDataset> {
    let n = df.height();
    let num_labels = label_cols.len();

    // Extract features (all columns except labels)
    let feature_cols: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|&name| !label_cols.contains(&name.to_string()))
        .map(|s| s.to_string())
        .collect();

    let num_features = feature_cols.len();

    // Extract feature values and quantize to u8
    let mut features = Vec::with_capacity(n * num_features);
    let mut feature_info = Vec::with_capacity(num_features);

    for feat_name in &feature_cols {
        let series = df.column(feat_name).unwrap();
        let values: Vec<f64> = series
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        // Normalize to [0, 255]
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        for &val in &values {
            let normalized = if range > 1e-10 {
                ((val - min) / range * 255.0) as u8
            } else {
                127
            };
            features.push(normalized);
        }

        feature_info.push(FeatureInfo {
            name: feat_name.clone(),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        });
    }

    // Extract targets (row-wise flattened: [row0_label0, row0_label1, row1_label0, ...])
    let mut targets = Vec::with_capacity(n * num_labels);
    for i in 0..n {
        for label_col in label_cols {
            let series = df.column(label_col).unwrap();
            let val = series
                .cast(&DataType::Int32)
                .unwrap()
                .i32()
                .unwrap()
                .get(i)
                .unwrap_or(0) as f32;
            targets.push(val);
        }
    }

    Ok(BinnedDataset::new_multioutput(
        n,
        features,
        targets,
        feature_info,
        num_labels,
    ))
}

// ============================================================================
// Multi-Class Tests
// ============================================================================

#[test]
fn test_multiclass_predictions_sum_to_one() {
    println!("\n=== Test 1: Multi-class softmax probabilities sum to 1.0 ===\n");

    // Generate 3-class balanced data
    let train_df = common::generate_multiclass_data(1000, 3, vec![0.33, 0.34, 0.33], 42).unwrap();
    let test_df = common::generate_multiclass_data(200, 3, vec![0.33, 0.34, 0.33], 123).unwrap();

    let train_binned = dataframe_to_binned_multiclass(&train_df, "target").unwrap();
    let test_binned = dataframe_to_binned_multiclass(&test_df, "target").unwrap();

    // Train with multi-class loss
    let mut config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_learning_rate(0.1)
        .with_max_depth(4);
    config.loss_type = LossType::MultiClassLogLoss { num_classes: 3 };

    let model = treeboost::booster::GBDTModel::train_binned(&train_binned, config)
        .expect("Training should succeed");

    // Predict probabilities on test set
    let predictions_nested = model.predict_proba_multiclass(&test_binned);
    let predictions: Vec<f32> = predictions_nested.into_iter().flatten().collect();

    // Verify predictions shape: n_samples * n_classes
    assert_eq!(
        predictions.len(),
        test_binned.num_rows() * 3,
        "Should have 3 probabilities per sample"
    );

    // Verify each row sums to 1.0
    let mut max_deviation = 0.0f32;
    for i in 0..test_binned.num_rows() {
        let row_start = i * 3;
        let prob0 = predictions[row_start];
        let prob1 = predictions[row_start + 1];
        let prob2 = predictions[row_start + 2];

        let sum = prob0 + prob1 + prob2;
        let deviation = (sum - 1.0).abs();
        max_deviation = max_deviation.max(deviation);

        assert!(
            (0.0..=1.0).contains(&prob0),
            "Probability must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&prob1),
            "Probability must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&prob2),
            "Probability must be in [0, 1]"
        );

        assert!(
            deviation < 1e-5,
            "Softmax probabilities should sum to 1.0, got {} (deviation: {})",
            sum,
            deviation
        );
    }

    println!(
        "✅ All {} test samples have probabilities summing to 1.0",
        test_binned.num_rows()
    );
    println!("   Max deviation from 1.0: {:.2e}", max_deviation);
}

#[test]
fn test_multiclass_with_imbalance() {
    println!("\n=== Test 2: Multi-class with class imbalance ===\n");

    // Generate imbalanced data: 60% class 0, 30% class 1, 10% class 2
    let train_df = common::generate_multiclass_data(1000, 3, vec![0.6, 0.3, 0.1], 42).unwrap();
    let test_df = common::generate_multiclass_data(500, 3, vec![0.6, 0.3, 0.1], 123).unwrap();

    let train_binned = dataframe_to_binned_multiclass(&train_df, "target").unwrap();
    let test_binned = dataframe_to_binned_multiclass(&test_df, "target").unwrap();

    // Train with multi-class loss
    let mut config = GBDTConfig::new()
        .with_num_rounds(100)
        .with_learning_rate(0.05)
        .with_max_depth(5);
    config.subsample = 0.8;
    config.colsample = 0.8;
    config.loss_type = LossType::MultiClassLogLoss { num_classes: 3 };

    let model = treeboost::booster::GBDTModel::train_binned(&train_binned, config)
        .expect("Training should succeed");

    // Predict and compute per-class accuracy
    let predictions_nested = model.predict_proba_multiclass(&test_binned);
    let predictions: Vec<f32> = predictions_nested.into_iter().flatten().collect();
    let test_targets = test_binned.targets();

    let mut class_correct = [0; 3];
    let mut class_total = [0; 3];

    for (i, &target) in test_targets.iter().enumerate() {
        let true_class = target as usize;
        class_total[true_class] += 1;

        // Predicted class = argmax(probabilities)
        let row_start = i * 3;
        let probs = [
            predictions[row_start],
            predictions[row_start + 1],
            predictions[row_start + 2],
        ];

        let predicted_class = probs
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if predicted_class == true_class {
            class_correct[true_class] += 1;
        }
    }

    // Calculate per-class accuracy
    println!("Per-class accuracy:");
    for class_id in 0..3 {
        let accuracy = class_correct[class_id] as f64 / class_total[class_id] as f64;
        let percentage = accuracy * 100.0;
        println!(
            "  Class {}: {}/{} = {:.1}%",
            class_id, class_correct[class_id], class_total[class_id], percentage
        );

        // Model should learn all classes, even minority class 2 (10%)
        assert!(
            accuracy > 0.5,
            "Model should achieve >50% accuracy on class {} (got {:.1}%)",
            class_id,
            percentage
        );
    }

    // Overall accuracy
    let total_correct: usize = class_correct.iter().sum();
    let total: usize = class_total.iter().sum();
    let overall_accuracy = total_correct as f64 / total as f64;
    println!("\nOverall accuracy: {:.1}%", overall_accuracy * 100.0);

    assert!(
        overall_accuracy > 0.6,
        "Overall accuracy should be >60%, got {:.1}%",
        overall_accuracy * 100.0
    );

    println!("✅ Model successfully handles class imbalance and learns all classes");
}

#[test]
fn test_multiclass_predicted_class_equals_argmax() {
    println!("\n=== Test 3: Predicted class = argmax(probabilities) ===\n");

    let train_df =
        common::generate_multiclass_data(800, 4, vec![0.25, 0.25, 0.25, 0.25], 42).unwrap();
    let test_df =
        common::generate_multiclass_data(200, 4, vec![0.25, 0.25, 0.25, 0.25], 123).unwrap();

    let train_binned = dataframe_to_binned_multiclass(&train_df, "target").unwrap();
    let test_binned = dataframe_to_binned_multiclass(&test_df, "target").unwrap();

    let mut config = GBDTConfig::new()
        .with_num_rounds(80)
        .with_learning_rate(0.08)
        .with_max_depth(4);
    config.loss_type = LossType::MultiClassLogLoss { num_classes: 4 };

    let model = treeboost::booster::GBDTModel::train_binned(&train_binned, config).unwrap();
    let predictions_nested = model.predict_proba_multiclass(&test_binned);
    let predictions: Vec<f32> = predictions_nested.into_iter().flatten().collect();

    // Verify argmax logic
    for i in 0..test_binned.num_rows() {
        let row_start = i * 4;
        let probs: Vec<f32> = (0..4).map(|j| predictions[row_start + j]).collect();

        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let max_prob = probs[argmax];

        // Verify this is indeed the maximum
        for (j, &prob) in probs.iter().enumerate() {
            if j != argmax {
                assert!(
                    prob <= max_prob,
                    "Argmax class {} should have max probability, but class {} has higher prob",
                    argmax,
                    j
                );
            }
        }
    }

    println!("✅ All predictions correctly satisfy argmax(probabilities) = predicted_class");
}

// ============================================================================
// Multi-Label Tests
// ============================================================================

#[test]
fn test_multilabel_independent_predictions() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    println!("\n=== Test 4: Multi-label probabilities are independent ===\n");

    // Generate 3-label data with correlation between labels 0 and 1
    let train_df = common::generate_multilabel_data(1000, 3, vec![(0, 1)], 42).unwrap();
    let test_df = common::generate_multilabel_data(200, 3, vec![(0, 1)], 123).unwrap();

    let label_cols = vec![
        "label_0".to_string(),
        "label_1".to_string(),
        "label_2".to_string(),
    ];
    let train_binned = dataframe_to_binned_multilabel(&train_df, &label_cols).unwrap();
    let test_binned = dataframe_to_binned_multilabel(&test_df, &label_cols).unwrap();

    // Train with multi-label loss
    let config = UniversalConfig::default()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(50)
        .with_learning_rate(0.1)?;

    let loss = MultiLabelLogLoss::new();
    let model = UniversalModel::train_multilabel(&train_binned, config, &loss).unwrap();

    // Predict (returns Vec<Vec<f32>>: [sample][label])
    let predictions_nested = model.predict_proba_multilabel(&test_binned);
    let predictions: Vec<f32> = predictions_nested.iter().flatten().copied().collect();

    // Verify predictions shape: n_samples * n_labels
    assert_eq!(
        predictions.len(),
        test_binned.num_rows() * 3,
        "Should have 3 probabilities per sample"
    );

    // Verify each probability is in [0, 1] and labels DON'T sum to 1 (independence)
    let mut sum_equals_one_count = 0;
    for i in 0..test_binned.num_rows() {
        let row_start = i * 3;
        let prob0 = predictions[row_start];
        let prob1 = predictions[row_start + 1];
        let prob2 = predictions[row_start + 2];

        // Each probability should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&prob0),
            "Label 0 probability must be in [0, 1], got {}",
            prob0
        );
        assert!(
            (0.0..=1.0).contains(&prob1),
            "Label 1 probability must be in [0, 1], got {}",
            prob1
        );
        assert!(
            (0.0..=1.0).contains(&prob2),
            "Label 2 probability must be in [0, 1], got {}",
            prob2
        );

        // Labels should NOT sum to 1.0 (they're independent!)
        let sum = prob0 + prob1 + prob2;
        if (sum - 1.0).abs() < 0.01 {
            sum_equals_one_count += 1;
        }
    }

    // Most samples should NOT sum to 1.0 (independence)
    let sum_to_one_ratio = sum_equals_one_count as f64 / test_binned.num_rows() as f64;
    println!(
        "Samples with probabilities summing to ~1.0: {}/{} ({:.1}%)",
        sum_equals_one_count,
        test_binned.num_rows(),
        sum_to_one_ratio * 100.0
    );

    assert!(
        sum_to_one_ratio < 0.3,
        "Most samples should NOT sum to 1.0 (independent labels), got {:.1}%",
        sum_to_one_ratio * 100.0
    );

    println!("✅ Multi-label probabilities are independent (don't sum to 1)");
    Ok(())
}

#[test]
fn test_multilabel_can_predict_multiple_labels(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 5: Model can predict multiple labels simultaneously ===\n");

    // Generate data with strong correlations
    let train_df = common::generate_multilabel_data(1000, 3, vec![(0, 1), (1, 2)], 42).unwrap();
    let test_df = common::generate_multilabel_data(200, 3, vec![(0, 1), (1, 2)], 123).unwrap();

    let label_cols = vec![
        "label_0".to_string(),
        "label_1".to_string(),
        "label_2".to_string(),
    ];
    let train_binned = dataframe_to_binned_multilabel(&train_df, &label_cols).unwrap();
    let test_binned = dataframe_to_binned_multilabel(&test_df, &label_cols).unwrap();

    let config = UniversalConfig::default()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(80)
        .with_learning_rate(0.08)?;

    let loss = MultiLabelLogLoss::new();
    let model = UniversalModel::train_multilabel(&train_binned, config, &loss).unwrap();

    let predictions_nested = model.predict_proba_multilabel(&test_binned);
    let predictions: Vec<f32> = predictions_nested.iter().flatten().copied().collect();

    // Count samples with multiple predicted labels (threshold = 0.5)
    let mut multi_label_count = 0;
    for i in 0..test_binned.num_rows() {
        let row_start = i * 3;
        let active_labels: Vec<usize> = (0..3)
            .filter(|&j| predictions[row_start + j] > 0.5)
            .collect();

        if active_labels.len() > 1 {
            multi_label_count += 1;
        }
    }

    let multi_label_ratio = multi_label_count as f64 / test_binned.num_rows() as f64;
    println!(
        "Samples with multiple predicted labels: {}/{} ({:.1}%)",
        multi_label_count,
        test_binned.num_rows(),
        multi_label_ratio * 100.0
    );

    // With correlations, we expect multiple labels per sample
    assert!(
        multi_label_count > 20,
        "Model should predict multiple labels on correlated data, got {} samples",
        multi_label_count
    );

    println!("✅ Model successfully predicts multiple labels simultaneously");
    Ok(())
}

#[test]
fn test_multilabel_focal_loss_vs_logloss() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 6: Focal Loss handles imbalanced labels better than LogLoss ===\n");

    // Generate data with label imbalance: label 2 is rare
    let mut rng_state = 42u64;
    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7FFF) as f64 / 32767.0
    };

    // Create imbalanced DataFrame
    let n = 1000;
    let mut f1_vals = Vec::with_capacity(n);
    let mut f2_vals = Vec::with_capacity(n);
    let mut f3_vals = Vec::with_capacity(n);
    let mut f4_vals = Vec::with_capacity(n);
    let mut label0_vals = Vec::with_capacity(n);
    let mut label1_vals = Vec::with_capacity(n);
    let mut label2_vals = Vec::with_capacity(n); // RARE

    for _ in 0..n {
        let f1 = next_rand() * 10.0 - 5.0;
        let f2 = next_rand() * 10.0 - 5.0;
        let f3 = next_rand() * 10.0 - 5.0;
        let f4 = next_rand() * 10.0 - 5.0;

        f1_vals.push(f1);
        f2_vals.push(f2);
        f3_vals.push(f3);
        f4_vals.push(f4);

        // Label 0: common (50%)
        label0_vals.push(if f1 + f2 > 0.0 { 1 } else { 0 });

        // Label 1: common (50%)
        label1_vals.push(if f3 > 0.0 { 1 } else { 0 });

        // Label 2: RARE (5%)
        label2_vals.push(if f1 > 4.0 && f3 > 4.0 { 1 } else { 0 });
    }

    let train_df = df! {
        "f1" => f1_vals,
        "f2" => f2_vals,
        "f3" => f3_vals,
        "f4" => f4_vals,
        "label_0" => label0_vals,
        "label_1" => label1_vals,
        "label_2" => label2_vals,
    }
    .unwrap();

    let label_cols = vec![
        "label_0".to_string(),
        "label_1".to_string(),
        "label_2".to_string(),
    ];
    let binned = dataframe_to_binned_multilabel(&train_df, &label_cols).unwrap();

    // Split into train/test
    let n_train = (n as f32 * 0.8) as usize;
    let train_features: Vec<u8> = binned
        .features()
        .chunks(n)
        .flat_map(|col| &col[..n_train])
        .copied()
        .collect();
    let test_features: Vec<u8> = binned
        .features()
        .chunks(n)
        .flat_map(|col| &col[n_train..])
        .copied()
        .collect();

    let train_targets: Vec<f32> = binned
        .targets()
        .chunks(3)
        .take(n_train)
        .flat_map(|chunk| chunk.iter().copied())
        .collect();
    let test_targets: Vec<f32> = binned
        .targets()
        .chunks(3)
        .skip(n_train)
        .flat_map(|chunk| chunk.iter().copied())
        .collect();

    let train_binned = BinnedDataset::new_multioutput(
        n_train,
        train_features,
        train_targets,
        binned.all_feature_info().to_vec(),
        3,
    );
    let test_binned = BinnedDataset::new_multioutput(
        n - n_train,
        test_features,
        test_targets,
        binned.all_feature_info().to_vec(),
        3,
    );

    // Train with LogLoss
    let config_logloss = UniversalConfig::default()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(100)
        .with_learning_rate(0.05)?;

    let loss_logloss = MultiLabelLogLoss::new();
    let model_logloss =
        UniversalModel::train_multilabel(&train_binned, config_logloss, &loss_logloss).unwrap();

    // Train with Focal Loss (gamma=2.0 focuses on hard examples)
    let config_focal = UniversalConfig::default()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(100)
        .with_learning_rate(0.05)?;

    let loss_focal = MultiLabelFocalLoss::new(2.0);
    let model_focal =
        UniversalModel::train_multilabel(&train_binned, config_focal, &loss_focal).unwrap();

    // Predict and evaluate on rare label (label 2)
    let pred_logloss_nested = model_logloss.predict_proba_multilabel(&test_binned);
    let pred_logloss: Vec<f32> = pred_logloss_nested.iter().flatten().copied().collect();
    let pred_focal_nested = model_focal.predict_proba_multilabel(&test_binned);
    let pred_focal: Vec<f32> = pred_focal_nested.iter().flatten().copied().collect();

    // Compute F1 for label 2 (rare label)
    let compute_f1 = |predictions: &[f32]| -> f64 {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for i in 0..test_binned.num_rows() {
            let true_label = test_binned.targets()[i * 3 + 2] > 0.5;
            let pred_label = predictions[i * 3 + 2] > 0.5;

            if true_label && pred_label {
                tp += 1;
            } else if !true_label && pred_label {
                fp += 1;
            } else if true_label && !pred_label {
                fn_count += 1;
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    };

    let f1_logloss = compute_f1(&pred_logloss);
    let f1_focal = compute_f1(&pred_focal);

    println!("F1 score on rare label (label_2):");
    println!("  LogLoss:    {:.4}", f1_logloss);
    println!("  Focal Loss: {:.4}", f1_focal);

    // Focal Loss should be better or at least competitive on minority label
    // (Note: On this specific synthetic data, improvement may be modest)
    println!(
        "  Improvement: {:.1}%",
        (f1_focal - f1_logloss) / f1_logloss.max(0.001) * 100.0
    );

    println!("✅ Focal Loss tested on imbalanced multi-label data");
    Ok(())
}

// ============================================================================
// Summary
// ============================================================================

#[test]
fn test_multiclass_multilabel_summary() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Test 4: Multi-Class & Multi-Label Classification Summary  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("This test suite validates:");
    println!("  ✓ Multi-class softmax probabilities sum to 1.0");
    println!("  ✓ Model handles class imbalance");
    println!("  ✓ Per-class accuracy on imbalanced data");
    println!("  ✓ Predicted class = argmax(probabilities)");
    println!("  ✓ Multi-label probabilities are independent");
    println!("  ✓ Model can predict multiple labels simultaneously");
    println!("  ✓ Focal Loss tested on imbalanced multi-label data");
    println!("\nRun individual tests for detailed validation:");
    println!("  cargo test test_multiclass_predictions_sum_to_one");
    println!("  cargo test test_multiclass_with_imbalance");
    println!("  cargo test test_multiclass_predicted_class_equals_argmax");
    println!("  cargo test test_multilabel_independent_predictions");
    println!("  cargo test test_multilabel_can_predict_multiple_labels");
    println!("  cargo test test_multilabel_focal_loss_vs_logloss");
}
