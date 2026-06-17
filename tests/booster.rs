//! Integration tests for core GBDT training and prediction

mod common;

use treeboost::booster::{GBDTConfig, GBDTModel, OutputType};
use treeboost::inference::ConformalPredictor;
use treeboost::serialize::{load_model, save_model};

use common::create_synthetic_dataset;

#[test]
fn test_basic_training_and_prediction() {
    let dataset = create_synthetic_dataset(1000, 42);

    let config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    assert_eq!(model.num_trees(), 50);

    let predictions = model.predict(&dataset);
    assert_eq!(predictions.len(), 1000);

    // Check predictions are in reasonable range
    let targets = dataset.targets();
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    // MSE should be reasonably low after 50 rounds
    assert!(mse < 5.0, "MSE {} is too high", mse);
}

#[test]
fn test_pseudo_huber_loss() {
    // Create dataset with outliers
    let mut dataset = create_synthetic_dataset(500, 123);

    // Add outliers to targets (simulate dirty data)
    let targets = dataset.targets_mut();
    targets[0] = 1000.0; // Extreme outlier
    targets[10] = -500.0;
    targets[50] = 2000.0;

    // Train with Pseudo-Huber loss (should be robust to outliers)
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(3)
        .with_pseudo_huber_loss(1.0);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    let predictions = model.predict(&dataset);

    // Predictions for non-outlier points should be reasonable
    // (not pulled towards extreme values)
    let non_outlier_predictions: Vec<f32> = predictions
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 0 && *i != 10 && *i != 50)
        .map(|(_, &p)| p)
        .collect();

    let mean_pred: f32 =
        non_outlier_predictions.iter().sum::<f32>() / non_outlier_predictions.len() as f32;

    // Mean prediction should be in reasonable range (not pulled to extremes)
    assert!(
        mean_pred > 0.0 && mean_pred < 20.0,
        "Mean prediction {} is extreme",
        mean_pred
    );
}

#[test]
fn test_conformal_prediction() {
    let dataset = create_synthetic_dataset(500, 456);

    // Train with conformal prediction enabled
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(4)
        .with_conformal(0.2, 0.9)
        .unwrap(); // 20% calibration, 90% coverage

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    assert!(model.conformal_quantile().is_some());

    let (predictions, lower, upper) = model.predict_with_intervals(&dataset);

    // All intervals should be valid
    for i in 0..predictions.len() {
        assert!(
            lower[i] < predictions[i],
            "Lower bound should be less than prediction"
        );
        assert!(
            upper[i] > predictions[i],
            "Upper bound should be greater than prediction"
        );
        assert!(lower[i] < upper[i], "Lower should be less than upper");
    }
}

#[test]
fn test_model_serialization() {
    let dataset = create_synthetic_dataset(200, 789);

    let config = GBDTConfig::new().with_num_rounds(10).with_max_depth(3);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");
    let original_predictions = model.predict(&dataset);

    // Save model
    let temp_dir = tempfile::tempdir().expect("Should create temp dir");
    let model_path = temp_dir.path().join("model.rkyv");

    save_model(&model, &model_path).expect("Should save model");

    // Load model
    let loaded_model = load_model(&model_path).expect("Should load model");

    // Verify
    assert_eq!(loaded_model.num_trees(), model.num_trees());
    assert_eq!(loaded_model.base_prediction(), model.base_prediction());

    let loaded_predictions = loaded_model.predict(&dataset);

    for (orig, loaded) in original_predictions.iter().zip(loaded_predictions.iter()) {
        assert!(
            (orig - loaded).abs() < 1e-6,
            "Predictions should match after load"
        );
    }
}

#[test]
fn test_feature_importance() {
    let dataset = create_synthetic_dataset(500, 321);

    let config = GBDTConfig::new().with_num_rounds(50).with_max_depth(5);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");
    let importances = model.feature_importance();

    assert_eq!(importances.len(), 5);

    // Importances should sum to ~1
    let total: f32 = importances.iter().sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "Importances should sum to 1, got {}",
        total
    );

    // All importances should be non-negative
    for (i, &imp) in importances.iter().enumerate() {
        assert!(
            imp >= 0.0,
            "Importance for feature {} should be non-negative",
            i
        );
    }

    // First two features should have higher importance (they define the target)
    // This is a soft check - may not always hold due to correlation
    let top_two: f32 = importances[0] + importances[1];
    assert!(
        top_two > 0.2,
        "Top two features should have significant importance"
    );
}

#[test]
fn test_conformal_predictor() {
    let predictions: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let actuals: Vec<f32> = vec![1.1, 1.9, 3.2, 3.8, 5.1];

    // Compute residuals
    let residuals: Vec<f32> = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (a - p).abs())
        .collect();

    let predictor = ConformalPredictor::from_residuals(&residuals, 0.9);

    // New predictions with intervals
    let new_predictions = vec![2.5, 3.5];
    let intervals = predictor.predict_batch(&new_predictions);

    assert_eq!(intervals.len(), 2);

    for interval in &intervals {
        let lower = interval.lower.unwrap();
        let upper = interval.upper.unwrap();
        assert!(lower < interval.point);
        assert!(interval.point < upper);
    }

    // Coverage should be at target level (approximately)
    let test_preds = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let test_actuals: Vec<f32> = test_preds.iter().map(|x| x + 0.1).collect();

    let coverage = predictor.empirical_coverage(&test_actuals, &test_preds);
    // Coverage should be high for these well-calibrated predictions
    assert!(coverage > 0.5, "Coverage {} is too low", coverage);
}

#[test]
fn test_entropy_regularization() {
    let dataset = create_synthetic_dataset(500, 654);

    // Train without entropy regularization
    let config_no_entropy = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(6)
        .with_entropy_weight(0.0);

    let model_no_entropy =
        GBDTModel::train_binned(&dataset, config_no_entropy).expect("Training should succeed");

    // Train with entropy regularization
    let config_entropy = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(6)
        .with_entropy_weight(0.1);

    let model_entropy =
        GBDTModel::train_binned(&dataset, config_entropy).expect("Training should succeed");

    // Both models should produce reasonable predictions
    let preds_no_entropy = model_no_entropy.predict(&dataset);
    let preds_entropy = model_entropy.predict(&dataset);

    // Basic sanity checks
    assert_eq!(preds_no_entropy.len(), 500);
    assert_eq!(preds_entropy.len(), 500);

    // Predictions should be similar but not identical
    let diff: f32 = preds_no_entropy
        .iter()
        .zip(preds_entropy.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / preds_no_entropy.len() as f32;

    // Some difference is expected due to regularization
    assert!(diff > 0.0, "Entropy regularization should have some effect");
}

#[test]
fn test_max_leaves_constraint() {
    let dataset = create_synthetic_dataset(500, 987);

    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_leaves(8)
        .with_max_depth(10); // High max_depth, but leaves constrained

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Each tree should have at most 8 leaves
    for tree in model.trees() {
        assert!(
            tree.num_leaves() <= 8,
            "Tree has {} leaves, expected <= 8",
            tree.num_leaves()
        );
    }
}

// =============================================================================
// GBDTModel Structure Tests
// =============================================================================

#[test]
fn test_model_output_type_regression() {
    let dataset = create_synthetic_dataset(200, 111);

    let config = GBDTConfig::new().with_num_rounds(10).with_max_depth(3);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Verify model structure for regression
    assert_eq!(model.output_type(), OutputType::Regression);
    assert_eq!(model.num_outputs(), 1);
    assert_eq!(model.base_predictions().len(), 1);
    assert!(!model.is_multiclass());
    assert!(!model.is_multilabel());
}

#[test]
fn test_model_output_type_binary() {
    let mut dataset = create_synthetic_dataset(200, 222);

    // Convert to binary classification (0/1 targets)
    for t in dataset.targets_mut() {
        *t = if *t > 7.5 { 1.0 } else { 0.0 };
    }

    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(3)
        .with_binary_logloss();

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Verify model structure for binary classification
    assert_eq!(model.output_type(), OutputType::Binary);
    assert_eq!(model.num_outputs(), 1);
    assert_eq!(model.base_predictions().len(), 1);
    assert!(!model.is_multiclass());
    assert!(!model.is_multilabel());
}

#[test]
fn test_model_num_outputs_accessor() {
    let dataset = create_synthetic_dataset(200, 333);

    // Regression model
    let config = GBDTConfig::new().with_num_rounds(5);
    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    assert_eq!(model.num_outputs(), 1);
    assert_eq!(model.base_predictions().len(), 1);

    // Verify base_prediction() accessor still works for backward compatibility
    let base = model.base_prediction();
    assert_eq!(base, model.base_predictions()[0]);
}

#[test]
fn test_config_multilabel_logloss() {
    // Valid configuration
    let config = GBDTConfig::new()
        .with_multilabel_logloss(3)
        .expect("Should accept num_outputs >= 2");

    assert!(config.loss_type.is_multilabel());
    assert_eq!(config.loss_type.num_outputs(), Some(3));
    assert_eq!(config.loss_type.output_type(), OutputType::MultiLabel);
}

#[test]
fn test_config_multilabel_focal_loss() {
    // Valid configuration
    let config = GBDTConfig::new()
        .with_multilabel_focal_loss(4, 2.0)
        .expect("Should accept valid focal loss config");

    assert!(config.loss_type.is_multilabel());
    assert_eq!(config.loss_type.num_outputs(), Some(4));
    assert_eq!(config.loss_type.output_type(), OutputType::MultiLabel);
}

#[test]
fn test_config_multilabel_validation() {
    // Invalid: num_outputs < 2
    let result = GBDTConfig::new().with_multilabel_logloss(1);
    assert!(result.is_err(), "Should reject num_outputs < 2");

    // Invalid: negative gamma
    let result = GBDTConfig::new().with_multilabel_focal_loss(3, -1.0);
    assert!(result.is_err(), "Should reject negative gamma");

    // Valid edge cases
    let config = GBDTConfig::new()
        .with_multilabel_focal_loss(2, 0.0)
        .expect("gamma=0 is valid (reduces to logloss)");
    assert!(config.loss_type.is_multilabel());
}

#[test]
fn test_loss_type_output_type_mapping() {
    use treeboost::booster::LossType;

    // Regression losses
    assert_eq!(LossType::Mse.output_type(), OutputType::Regression);
    assert_eq!(
        LossType::PseudoHuber { delta: 1.0 }.output_type(),
        OutputType::Regression
    );

    // Binary classification
    assert_eq!(LossType::BinaryLogLoss.output_type(), OutputType::Binary);

    // Multi-class
    assert_eq!(
        LossType::MultiClassLogLoss { num_classes: 3 }.output_type(),
        OutputType::MultiClass
    );

    // Multi-label
    assert_eq!(
        LossType::MultiLabelLogLoss { num_outputs: 3 }.output_type(),
        OutputType::MultiLabel
    );
    assert_eq!(
        LossType::MultiLabelFocalLoss {
            num_outputs: 3,
            gamma: 2.0
        }
        .output_type(),
        OutputType::MultiLabel
    );
}

// =============================================================================
// Vector Tree Shape Tests
// =============================================================================

/// Helper to create a multi-output dataset for testing
fn create_multilabel_dataset(n: usize, num_outputs: usize, seed: u64) -> treeboost::BinnedDataset {
    use treeboost::dataset::{FeatureInfo, FeatureType};

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

    // Generate multi-output targets (row-wise flattened)
    // Each output is a binary label based on different feature thresholds
    let mut targets = Vec::with_capacity(n * num_outputs);
    for i in 0..n {
        for k in 0..num_outputs {
            // Each output depends on a different feature combination
            let f0 = features[i] as f32 / 255.0;
            let fk = features[(k % num_features) * n + i] as f32 / 255.0;
            // Binary target based on feature threshold
            let threshold = 0.3 + (k as f32) * 0.1;
            let target = if f0 + fk > threshold { 1.0 } else { 0.0 };
            targets.push(target);
        }
    }

    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("feature_{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        })
        .collect();

    treeboost::BinnedDataset::new_multioutput(n, features, targets, feature_info, num_outputs)
}

#[test]
fn test_multilabel_tree_shape() {
    // Test that multi-label training produces VectorTrees (unified architecture)
    let num_outputs = 3;
    let num_rounds = 10;
    let dataset = create_multilabel_dataset(200, num_outputs, 444);

    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(3)
        .with_multilabel_logloss(num_outputs)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // With unified VectorTree: 1 tree per round (not N trees per round)
    assert_eq!(
        model.num_trees(),
        num_rounds,
        "Multi-label should have 1 VectorTree per round"
    );

    // Verify VectorTree architecture
    assert!(model.uses_vector_trees(), "Should use VectorTrees");
    for tree in model.trees() {
        assert!(tree.is_vector());
        assert_eq!(tree.num_outputs(), num_outputs);
    }

    // Verify model structure
    assert_eq!(model.output_type(), OutputType::MultiLabel);
    assert_eq!(model.num_outputs(), num_outputs);
    assert!(model.is_multilabel());
    assert!(!model.is_multiclass());

    // Base predictions should have one value per output
    assert_eq!(model.base_predictions().len(), num_outputs);
}

#[test]
fn test_multilabel_focal_loss_training() {
    let num_outputs = 3;
    let dataset = create_multilabel_dataset(200, num_outputs, 555);

    // Train with focal loss (gamma=2)
    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(3)
        .with_multilabel_focal_loss(num_outputs, 2.0)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    assert_eq!(model.output_type(), OutputType::MultiLabel);
    assert_eq!(model.num_outputs(), num_outputs);
    assert_eq!(model.base_predictions().len(), num_outputs);
}

#[test]
fn test_multilabel_predictions_shape() {
    let num_outputs = 3;
    let num_rows = 100;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 666);

    let config = GBDTConfig::new()
        .with_num_rounds(5)
        .with_max_depth(3)
        .with_multilabel_logloss(num_outputs)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Verify model structure
    assert_eq!(model.num_outputs(), num_outputs);
    assert!(model.uses_vector_trees());

    // Use the proper predict_multilabel API
    let raw_preds = model.predict_multilabel(&dataset);
    assert_eq!(raw_preds.len(), num_rows);
    for row_pred in &raw_preds {
        assert_eq!(row_pred.len(), num_outputs);
    }

    // Verify probabilities are in valid range
    let proba = model.predict_proba_multilabel(&dataset);
    for row in &proba {
        for &p in row {
            assert!((0.0..=1.0).contains(&p), "Probability out of range: {}", p);
        }
    }

    // Verify some learning happened (predictions should vary)
    let flat: Vec<f32> = raw_preds.iter().flat_map(|r| r.iter().copied()).collect();
    let unique_preds: std::collections::HashSet<u32> =
        flat.iter().map(|&p| (p * 1000.0) as u32).collect();
    assert!(unique_preds.len() > 1, "Predictions should vary");
}

// =============================================================================
// Vector Prediction Shape Tests
// =============================================================================

#[test]
fn test_predict_multilabel() {
    let num_outputs = 3;
    let num_rows = 50;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 777);

    let config = GBDTConfig::new()
        .with_num_rounds(5)
        .with_max_depth(3)
        .with_multilabel_logloss(num_outputs)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Test predict_multilabel returns correct shape
    let raw_preds = model.predict_multilabel(&dataset);
    assert_eq!(
        raw_preds.len(),
        num_rows,
        "Should have predictions for all rows"
    );
    for row_pred in &raw_preds {
        assert_eq!(
            row_pred.len(),
            num_outputs,
            "Each row should have num_outputs predictions"
        );
    }

    // Test predict_proba_multilabel returns probabilities in [0, 1]
    let proba = model.predict_proba_multilabel(&dataset);
    assert_eq!(proba.len(), num_rows);
    for row_proba in &proba {
        assert_eq!(row_proba.len(), num_outputs);
        for &p in row_proba {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability must be in [0, 1], got {}",
                p
            );
        }
    }

    // Test predict_labels returns boolean predictions
    let labels = model.predict_labels(&dataset);
    assert_eq!(labels.len(), num_rows);
    for row_labels in &labels {
        assert_eq!(row_labels.len(), num_outputs);
    }
}

#[test]
fn test_predict_labels_with_threshold() {
    let num_outputs = 2;
    let num_rows = 100;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 888);

    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(4)
        .with_multilabel_logloss(num_outputs)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    let proba = model.predict_proba_multilabel(&dataset);

    // Verify probabilities are in valid range [0, 1]
    for row in &proba {
        for &p in row {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability should be in [0, 1], got {}",
                p
            );
        }
    }

    // With threshold 0.0, all predictions should be positive
    let labels_low = model.predict_labels_with_threshold(&dataset, 0.0);
    let all_positive: bool = labels_low.iter().all(|row| row.iter().all(|&l| l));
    assert!(all_positive, "All labels should be true with threshold 0.0");

    // With threshold 1.0, all predictions should be negative
    let labels_high = model.predict_labels_with_threshold(&dataset, 1.0);
    let all_negative: bool = labels_high.iter().all(|row| row.iter().all(|&l| !l));
    assert!(
        all_negative,
        "All labels should be false with threshold 1.0"
    );

    // With threshold 0.5 (default), should match predict_labels
    let labels_default = model.predict_labels(&dataset);
    let labels_half = model.predict_labels_with_threshold(&dataset, 0.5);
    assert_eq!(
        labels_default, labels_half,
        "predict_labels should use threshold 0.5"
    );
}

#[test]
fn test_predict_labels_with_thresholds() {
    let num_outputs = 3;
    let num_rows = 50;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 999);

    let config = GBDTConfig::new()
        .with_num_rounds(5)
        .with_max_depth(3)
        .with_multilabel_logloss(num_outputs)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Per-label thresholds
    let thresholds = vec![0.3, 0.5, 0.7];
    let labels = model.predict_labels_with_thresholds(&dataset, &thresholds);

    assert_eq!(labels.len(), num_rows);
    for row_labels in &labels {
        assert_eq!(row_labels.len(), num_outputs);
    }

    // Verify thresholds are applied correctly
    let proba = model.predict_proba_multilabel(&dataset);
    for (row_idx, (row_proba, row_labels)) in proba.iter().zip(labels.iter()).enumerate() {
        for (label_idx, (&p, &l)) in row_proba.iter().zip(row_labels.iter()).enumerate() {
            let expected = p >= thresholds[label_idx];
            assert_eq!(
                l, expected,
                "Mismatch at row {} label {}: p={}, threshold={}, expected={}, got={}",
                row_idx, label_idx, p, thresholds[label_idx], expected, l
            );
        }
    }
}

// =============================================================================
// Multi-Label Serialization Round-Trip Tests
// =============================================================================

#[test]
fn test_multilabel_model_serialization_roundtrip() {
    let num_outputs = 3;
    let num_rows = 100;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 4242);

    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(4)
        .with_multilabel_logloss(num_outputs)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Get original predictions
    let original_proba = model.predict_proba_multilabel(&dataset);
    let original_labels = model.predict_labels(&dataset);

    // Save model
    let temp_dir = tempfile::tempdir().expect("Should create temp dir");
    let model_path = temp_dir.path().join("multilabel_model.rkyv");

    save_model(&model, &model_path).expect("Should save multi-label model");

    // Load model
    let loaded_model: GBDTModel = load_model(&model_path).expect("Should load multi-label model");

    // Verify model structure is preserved
    assert_eq!(
        loaded_model.num_trees(),
        model.num_trees(),
        "num_trees should match"
    );
    assert_eq!(
        loaded_model.num_outputs(),
        model.num_outputs(),
        "num_outputs should match"
    );
    assert_eq!(
        loaded_model.output_type(),
        model.output_type(),
        "output_type should match"
    );
    assert_eq!(
        loaded_model.output_type(),
        OutputType::MultiLabel,
        "Should be MultiLabel"
    );

    // Verify base_predictions are preserved
    let orig_base = model.base_predictions();
    let loaded_base = loaded_model.base_predictions();
    assert_eq!(
        orig_base.len(),
        loaded_base.len(),
        "base_predictions length should match"
    );
    for (orig, loaded) in orig_base.iter().zip(loaded_base.iter()) {
        assert!(
            (orig - loaded).abs() < 1e-6,
            "base_predictions should match: {} vs {}",
            orig,
            loaded
        );
    }

    // Verify predictions match
    let loaded_proba = loaded_model.predict_proba_multilabel(&dataset);
    assert_eq!(loaded_proba.len(), original_proba.len());
    for (orig_row, loaded_row) in original_proba.iter().zip(loaded_proba.iter()) {
        assert_eq!(orig_row.len(), loaded_row.len());
        for (orig, loaded) in orig_row.iter().zip(loaded_row.iter()) {
            assert!(
                (orig - loaded).abs() < 1e-6,
                "Probabilities should match after load: {} vs {}",
                orig,
                loaded
            );
        }
    }

    // Verify label predictions match
    let loaded_labels = loaded_model.predict_labels(&dataset);
    assert_eq!(
        loaded_labels, original_labels,
        "Label predictions should match after load"
    );
}

#[test]
fn test_multilabel_focal_loss_serialization() {
    let num_outputs = 2;
    let num_rows = 50;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 5555);

    // Train with focal loss
    let config = GBDTConfig::new()
        .with_num_rounds(5)
        .with_max_depth(3)
        .with_multilabel_focal_loss(num_outputs, 2.0)
        .expect("Should accept valid config");

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");
    let original_predictions = model.predict_multilabel(&dataset);

    // Save and load
    let temp_dir = tempfile::tempdir().expect("Should create temp dir");
    let model_path = temp_dir.path().join("focal_loss_model.rkyv");

    save_model(&model, &model_path).expect("Should save focal loss model");
    let loaded_model: GBDTModel = load_model(&model_path).expect("Should load focal loss model");

    // Verify output type preserved
    assert_eq!(loaded_model.output_type(), OutputType::MultiLabel);
    assert_eq!(loaded_model.num_outputs(), num_outputs);

    // Verify predictions match
    let loaded_predictions = loaded_model.predict_multilabel(&dataset);
    for (orig_row, loaded_row) in original_predictions.iter().zip(loaded_predictions.iter()) {
        for (orig, loaded) in orig_row.iter().zip(loaded_row.iter()) {
            assert!(
                (orig - loaded).abs() < 1e-6,
                "Predictions should match after load"
            );
        }
    }
}

// =============================================================================
// Phase 7: Unified Vector Tree Integration Tests
// =============================================================================

#[test]
fn test_vector_tree_multilabel_training() {
    // Verify multi-label uses VectorTree (unified splits), not N scalar trees
    let dataset = create_multilabel_dataset(30, 2, 7001);
    let config = GBDTConfig::new()
        .with_num_rounds(3)
        .with_max_depth(2)
        .with_multilabel_logloss(2)
        .unwrap();

    let model = GBDTModel::train_binned(&dataset, config).unwrap();

    assert!(model.uses_vector_trees());
    assert_eq!(model.num_trees(), 3); // 1 VectorTree per round
    assert!(model.trees()[0].is_vector());
    assert_eq!(model.trees()[0].num_outputs(), 2);
}

#[test]
fn test_vector_tree_serialization() {
    let dataset = create_multilabel_dataset(20, 2, 7002);
    let config = GBDTConfig::new()
        .with_num_rounds(2)
        .with_max_depth(2)
        .with_multilabel_logloss(2)
        .unwrap();

    let model = GBDTModel::train_binned(&dataset, config).unwrap();
    let orig_preds = model.predict_multilabel(&dataset);

    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("model.rkyv");

    save_model(&model, &path).unwrap();
    let loaded: GBDTModel = load_model(&path).unwrap();

    assert!(loaded.uses_vector_trees());
    let loaded_preds = loaded.predict_multilabel(&dataset);

    for (a, b) in orig_preds.iter().zip(loaded_preds.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}

#[test]
fn test_ensemble_tree_type_dispatch() {
    // Scalar (regression)
    let scalar_ds = create_synthetic_dataset(30, 7003);
    let scalar_model = GBDTModel::train_binned(
        &scalar_ds,
        GBDTConfig::new().with_num_rounds(2).with_max_depth(2),
    )
    .unwrap();
    assert!(!scalar_model.uses_vector_trees());
    assert!(scalar_model.trees()[0].is_scalar());

    // Vector (multi-label)
    let vector_ds = create_multilabel_dataset(30, 2, 7004);
    let vector_model = GBDTModel::train_binned(
        &vector_ds,
        GBDTConfig::new()
            .with_num_rounds(2)
            .with_max_depth(2)
            .with_multilabel_logloss(2)
            .unwrap(),
    )
    .unwrap();
    assert!(vector_model.uses_vector_trees());
    assert!(vector_model.trees()[0].is_vector());
}

#[test]
fn test_vector_tree_feature_importance() {
    let dataset = create_multilabel_dataset(50, 2, 7005);
    let config = GBDTConfig::new()
        .with_num_rounds(5)
        .with_max_depth(3)
        .with_multilabel_logloss(2)
        .unwrap();

    let model = GBDTModel::train_binned(&dataset, config).unwrap();
    let importances = model.feature_importance();

    assert_eq!(importances.len(), 5);
    let total: f32 = importances.iter().sum();
    assert!((total - 1.0).abs() < 0.01 || total == 0.0);
    assert!(importances.iter().all(|&x| x >= 0.0));
}

#[test]
fn test_vector_tree_predictions_quality() {
    // Verify VectorTree produces valid, learned predictions

    let num_outputs = 2;
    let num_rows = 50;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 7006);

    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(3)
        .with_multilabel_logloss(num_outputs)
        .unwrap();

    let model = GBDTModel::train_binned(&dataset, config).unwrap();
    let raw_preds = model.predict_multilabel(&dataset);
    let proba = model.predict_proba_multilabel(&dataset);

    // 1. All probabilities in valid range [0, 1]
    for row in &proba {
        for &p in row {
            assert!((0.0..=1.0).contains(&p), "Invalid probability: {}", p);
        }
    }

    // 2. Raw predictions should vary (model learned something)
    let flat: Vec<f32> = raw_preds.iter().flat_map(|r| r.iter().copied()).collect();
    let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max - min > 0.1,
        "Predictions should vary: min={}, max={}",
        min,
        max
    );

    // 3. Compute training loss - should be better than random (0.693 = -ln(0.5))
    let targets = dataset.targets();
    let mut total_loss = 0.0f32;
    for row in 0..num_rows {
        for k in 0..num_outputs {
            let t = targets[row * num_outputs + k];
            let p = proba[row][k].clamp(1e-7, 1.0 - 1e-7);
            total_loss += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        }
    }
    let avg_loss = total_loss / (num_rows * num_outputs) as f32;
    assert!(
        avg_loss < 0.69,
        "Training loss {} should be better than random",
        avg_loss
    );
}
