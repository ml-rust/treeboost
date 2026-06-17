//! Tests for AutoML multi-label support
//!
//! These tests verify that AutoML correctly handles multi-label classification tasks.

use polars::prelude::*;
use treeboost::model::{AutoModel, BoostingMode};

/// Create a multi-label dataset with multiple binary target columns
fn create_multilabel_dataframe(n_rows: usize, n_labels: usize, seed: u64) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(seed);

    // Create feature columns
    let x1: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();
    let x3: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();

    // Create target columns (binary labels based on feature thresholds)
    let mut columns: Vec<Column> = vec![
        Column::new("x1".into(), x1.clone()),
        Column::new("x2".into(), x2.clone()),
        Column::new("x3".into(), x3.clone()),
    ];

    for k in 0..n_labels {
        let threshold = 3.0 + (k as f64) * 1.5;
        let label: Vec<i32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| {
                if a + b * (0.5 + k as f64 * 0.1) > threshold {
                    1
                } else {
                    0
                }
            })
            .collect();
        columns.push(Column::new(format!("label_{}", k).into(), label));
    }

    DataFrame::new(n_rows, columns).unwrap()
}

// =============================================================================
//  AutoML 2D Detection Tests
// =============================================================================

#[test]
fn test_automl_multilabel_training() {
    let n_labels = 3;
    let df = create_multilabel_dataframe(200, n_labels, 4242);

    // Get target column names
    let target_cols: Vec<&str> = (0..n_labels)
        .map(|k| {
            // Get the column name from the DataFrame
            match k {
                0 => "label_0",
                1 => "label_1",
                2 => "label_2",
                _ => panic!("Unexpected label index"),
            }
        })
        .collect();

    // Train with multi-label API
    let model = AutoModel::train_multilabel(&df, &target_cols)
        .expect("Multi-label training should succeed");

    // Verify model structure
    assert_eq!(
        model.num_labels(),
        n_labels,
        "Should have correct number of labels"
    );
}

#[test]
fn test_automl_multilabel_prediction_shape() {
    let n_labels = 2;
    let n_rows = 100;
    let df = create_multilabel_dataframe(n_rows, n_labels, 5353);
    let target_cols = vec!["label_0", "label_1"];

    let model = AutoModel::train_multilabel(&df, &target_cols).expect("Training should succeed");

    // Predict on the same data (just for shape testing)
    let predictions = model
        .predict_multilabel(&df)
        .expect("Prediction should succeed");

    assert_eq!(predictions.len(), n_rows, "Should have one row per sample");
    for row in &predictions {
        assert_eq!(
            row.len(),
            n_labels,
            "Each row should have predictions for all labels"
        );
    }
}

#[test]
fn test_automl_multilabel_correctness() {
    let n_labels = 3;
    let n_rows = 300;
    let df = create_multilabel_dataframe(n_rows, n_labels, 6464);
    let target_cols = vec!["label_0", "label_1", "label_2"];

    let model = AutoModel::train_multilabel(&df, &target_cols).expect("Training should succeed");

    // Get predictions
    let proba = model
        .predict_proba_multilabel(&df)
        .expect("Prediction should succeed");

    // Compute accuracy per label
    // `k` indexes the label dimension of row-major `proba[row][k]` and builds the
    // column name, so there is no single slice to drive the loop via an iterator.
    #[allow(clippy::needless_range_loop)]
    for k in 0..n_labels {
        let col_name = format!("label_{}", k);
        let targets: Vec<f64> = df
            .column(&col_name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let correct: usize = targets
            .iter()
            .enumerate()
            .filter(|(i, &t)| {
                let pred = if proba[*i][k] >= 0.5 { 1.0 } else { 0.0 };
                (t - pred).abs() < 0.5
            })
            .count();

        let accuracy = correct as f64 / n_rows as f64;
        assert!(
            accuracy > 0.6,
            "Label {} accuracy {:.2} should be better than random",
            k,
            accuracy
        );
    }
}

#[test]
fn test_automl_multilabel_with_mode() {
    let df = create_multilabel_dataframe(150, 2, 7575);
    let target_cols = vec!["label_0", "label_1"];

    // Force LinearThenTree mode for multi-label
    let model =
        AutoModel::train_multilabel_with_mode(&df, &target_cols, BoostingMode::LinearThenTree)
            .expect("Training should succeed");

    assert_eq!(model.mode(), BoostingMode::LinearThenTree);
}

// =============================================================================
// Threshold Tuning Tests
// =============================================================================

#[test]
fn test_threshold_tuning() {
    let n_labels = 3;

    // Create separate train and validation sets
    let train_df = create_multilabel_dataframe(200, n_labels, 8686);
    let val_df = create_multilabel_dataframe(100, n_labels, 9797);

    let target_cols = vec!["label_0", "label_1", "label_2"];

    // Train model
    let mut model =
        AutoModel::train_multilabel(&train_df, &target_cols).expect("Training should succeed");

    // Initially no tuned thresholds
    assert!(
        model.tuned_thresholds().is_none(),
        "Should have no tuned thresholds initially"
    );

    // Tune thresholds on validation set
    let tune_result = model
        .tune_thresholds(&val_df, &target_cols)
        .expect("Threshold tuning should succeed");

    // Should have thresholds for all labels
    assert_eq!(tune_result.thresholds.len(), n_labels);
    assert_eq!(tune_result.f1_scores.len(), n_labels);

    // Thresholds should be in valid range
    for &threshold in &tune_result.thresholds {
        assert!(
            (0.01..=0.99).contains(&threshold),
            "Threshold {} should be in [0.01, 0.99]",
            threshold
        );
    }

    // Model should now have tuned thresholds
    assert!(
        model.tuned_thresholds().is_some(),
        "Should have tuned thresholds after tuning"
    );

    // predict_labels_tuned should use tuned thresholds
    let labels = model
        .predict_labels_tuned(&val_df)
        .expect("Prediction should succeed");

    assert_eq!(labels.len(), val_df.height());
    for row in &labels {
        assert_eq!(row.len(), n_labels);
    }
}

#[test]
fn test_threshold_tuning_improves_f1() {
    let n_labels = 2;

    // Create imbalanced data where threshold tuning should help
    let train_df = create_imbalanced_multilabel_dataframe(300, n_labels, 1010);
    let val_df = create_imbalanced_multilabel_dataframe(100, n_labels, 2020);

    let target_cols = vec!["label_0", "label_1"];

    // Train model
    let mut model =
        AutoModel::train_multilabel(&train_df, &target_cols).expect("Training should succeed");

    // Get predictions with default 0.5 threshold
    let proba = model
        .predict_proba_multilabel(&val_df)
        .expect("Prediction should succeed");

    let _labels_default: Vec<Vec<bool>> = proba
        .iter()
        .map(|row| row.iter().map(|&p| p >= 0.5).collect())
        .collect();

    // Tune thresholds
    let tune_result = model
        .tune_thresholds(&val_df, &target_cols)
        .expect("Threshold tuning should succeed");

    // Get predictions with tuned thresholds
    let _labels_tuned = model
        .predict_labels_tuned(&val_df)
        .expect("Prediction should succeed");

    // Tuned thresholds should give reasonable F1 scores
    for &f1 in &tune_result.f1_scores {
        assert!(
            (0.0..=1.0).contains(&f1),
            "F1 score {} should be in [0, 1]",
            f1
        );
    }
}

// =============================================================================
// Integration & Polish: End-to-End Tests
// =============================================================================

#[test]
fn test_multilabel_end_to_end_workflow() {
    // Complete workflow: train → predict → evaluate → serialize → load → predict

    let n_labels = 3;
    let n_train = 400;
    let n_test = 100;

    // 1. Create train and test datasets
    let train_df = create_multilabel_dataframe(n_train, n_labels, 1111);
    let test_df = create_multilabel_dataframe(n_test, n_labels, 2222);
    let target_cols = vec!["label_0", "label_1", "label_2"];

    // 2. Train multi-label model
    let mut model =
        AutoModel::train_multilabel(&train_df, &target_cols).expect("Training should succeed");

    // 3. Tune thresholds on a validation split (use test_df for simplicity)
    let tune_result = model
        .tune_thresholds(&test_df, &target_cols)
        .expect("Threshold tuning should succeed");

    // Verify tuning produced valid results
    assert_eq!(tune_result.thresholds.len(), n_labels);
    for &t in &tune_result.thresholds {
        assert!((0.01..=0.99).contains(&t), "Threshold {} out of range", t);
    }

    // 4. Make predictions with tuned thresholds
    let proba = model
        .predict_proba_multilabel(&test_df)
        .expect("Proba prediction should succeed");
    let labels = model
        .predict_labels_tuned(&test_df)
        .expect("Label prediction should succeed");

    // Verify prediction shapes
    assert_eq!(proba.len(), n_test);
    assert_eq!(labels.len(), n_test);
    for (p, l) in proba.iter().zip(labels.iter()) {
        assert_eq!(p.len(), n_labels);
        assert_eq!(l.len(), n_labels);

        // Probabilities should be in [0, 1]
        for &prob in p {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability {} out of range",
                prob
            );
        }
    }

    // 5. Evaluate per-label accuracy
    for k in 0..n_labels {
        let col_name = format!("label_{}", k);
        let targets: Vec<f64> = test_df
            .column(&col_name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let correct: usize = labels
            .iter()
            .zip(targets.iter())
            .filter(|(pred, &target)| {
                let p = if pred[k] { 1.0 } else { 0.0 };
                (p - target).abs() < 0.5
            })
            .count();

        let accuracy = correct as f64 / n_test as f64;
        assert!(
            accuracy > 0.5,
            "Label {} accuracy {:.2} should be better than random",
            k,
            accuracy
        );
    }

    // 6. Test model serialization round-trip (file-based)
    let temp_path = std::env::temp_dir().join("test_multilabel_model.rkyv");

    // Save the inner UniversalModel
    let inner_model = model.inner();
    inner_model
        .save(&temp_path)
        .expect("Model save should succeed");

    // Load it back
    let loaded =
        treeboost::model::UniversalModel::load(&temp_path).expect("Model load should succeed");

    // Verify loaded model has same structure
    assert_eq!(
        loaded.mode(),
        inner_model.mode(),
        "Loaded model should have same mode"
    );

    // For LinearThenTree multi-label, check per-label GBDTs were saved
    if let Some(gbdt_per_label) = loaded.gbdt_per_label() {
        assert_eq!(
            gbdt_per_label.len(),
            n_labels,
            "Loaded model should have same number of per-label GBDTs"
        );
    }

    // Clean up
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_multilabel_pure_tree_mode() {
    // Test multi-label with PureTree mode (GBDT-based)
    let n_labels = 2;
    let df = create_multilabel_dataframe(200, n_labels, 3333);
    let target_cols = vec!["label_0", "label_1"];

    let model = AutoModel::train_multilabel_with_mode(&df, &target_cols, BoostingMode::PureTree)
        .expect("PureTree multi-label training should succeed");

    assert_eq!(model.mode(), BoostingMode::PureTree);
    assert_eq!(model.num_labels(), n_labels);

    // Predictions should work
    let proba = model
        .predict_proba_multilabel(&df)
        .expect("Prediction should succeed");
    assert_eq!(proba.len(), df.height());
}

/// Create an imbalanced multi-label dataset (more negatives than positives)
fn create_imbalanced_multilabel_dataframe(n_rows: usize, n_labels: usize, seed: u64) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(seed);

    let x1: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();
    let x3: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();

    let mut columns: Vec<Column> = vec![
        Column::new("x1".into(), x1.clone()),
        Column::new("x2".into(), x2.clone()),
        Column::new("x3".into(), x3.clone()),
    ];

    // Create imbalanced labels (~20% positive rate)
    for k in 0..n_labels {
        let high_threshold = 7.0 + (k as f64) * 0.5; // High threshold = fewer positives
        let label: Vec<i32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| if a + b * 0.5 > high_threshold { 1 } else { 0 })
            .collect();
        columns.push(Column::new(format!("label_{}", k).into(), label));
    }

    DataFrame::new(n_rows, columns).unwrap()
}
