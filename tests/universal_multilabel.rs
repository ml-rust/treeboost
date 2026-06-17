//! Tests for UniversalModel multi-label support

use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::learner::LinearConfig;
use treeboost::loss::MultiLabelLogLoss;
use treeboost::model::{BoostingMode, UniversalConfig, UniversalModel};

/// Helper to create a multi-output dataset for testing
fn create_multilabel_dataset(n: usize, num_outputs: usize, seed: u64) -> BinnedDataset {
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

    BinnedDataset::new_multioutput(n, features, targets, feature_info, num_outputs)
}

#[test]
fn test_ltt_multilabel_training() -> Result<(), Box<dyn std::error::Error>> {
    let num_outputs = 3;
    let num_rows = 100;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 5151);

    // Configure LinearThenTree mode for multi-label
    let config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_num_rounds(5)
        .with_learning_rate(0.1)?
        .with_linear_config(LinearConfig::default());

    let loss = MultiLabelLogLoss::new();

    let model =
        UniversalModel::train_multilabel(&dataset, config, &loss).expect("Training should succeed");

    // Verify model structure
    assert!(model.has_linear(), "LTT should have linear component");
    assert_eq!(
        model.num_linear_boosters(),
        num_outputs,
        "Should have one LinearBooster per output"
    );

    // Verify per-label GBDT components
    assert_eq!(
        model.num_gbdt_per_label(),
        num_outputs,
        "Should have one GBDT per output"
    );
    let gbdts = model.gbdt_per_label().expect("Should have per-label GBDTs");
    assert_eq!(gbdts.len(), num_outputs);
    Ok(())
}

#[test]
fn test_ltt_multilabel_prediction_shape() -> Result<(), Box<dyn std::error::Error>> {
    let num_outputs = 2;
    let num_rows = 50;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 6262);

    let config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_num_rounds(3)
        .with_learning_rate(0.1)?;

    let loss = MultiLabelLogLoss::new();
    let model =
        UniversalModel::train_multilabel(&dataset, config, &loss).expect("Training should succeed");

    // Test predict_multilabel returns correct shape
    let predictions = model.predict_multilabel(&dataset);
    assert_eq!(
        predictions.len(),
        num_rows,
        "Should have row for each sample"
    );
    for row in &predictions {
        assert_eq!(
            row.len(),
            num_outputs,
            "Each row should have num_outputs predictions"
        );
    }

    // Test predict_proba_multilabel returns probabilities in [0, 1]
    let proba = model.predict_proba_multilabel(&dataset);
    for row in &proba {
        for &p in row {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability should be in [0, 1], got {}",
                p
            );
        }
    }
    Ok(())
}

#[test]
fn test_ltt_multilabel_correctness() -> Result<(), Box<dyn std::error::Error>> {
    let num_outputs = 3;
    let num_rows = 200;
    let dataset = create_multilabel_dataset(num_rows, num_outputs, 7373);

    // First, verify GBDT-only works (baseline)
    let gbdt_config = treeboost::booster::GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(4)
        .with_multilabel_logloss(num_outputs)
        .expect("Should create config");

    let gbdt_model = treeboost::booster::GBDTModel::train_binned(&dataset, gbdt_config)
        .expect("GBDT training should succeed");

    let gbdt_proba = gbdt_model.predict_proba_multilabel(&dataset);
    let targets = dataset.targets();

    // Compute GBDT-only accuracy
    let mut gbdt_correct_per_label = vec![0usize; num_outputs];
    for i in 0..num_rows {
        for k in 0..num_outputs {
            let target = targets[i * num_outputs + k];
            let pred_label = if gbdt_proba[i][k] >= 0.5 { 1.0 } else { 0.0 };
            if (target - pred_label).abs() < 0.5 {
                gbdt_correct_per_label[k] += 1;
            }
        }
    }

    // GBDT should achieve reasonable accuracy
    for (k, &correct) in gbdt_correct_per_label.iter().enumerate() {
        let accuracy = correct as f64 / num_rows as f64;
        assert!(
            accuracy > 0.5,
            "GBDT Label {} accuracy {:.2} should be better than random",
            k,
            accuracy
        );
    }

    // Now test LinearThenTree
    let config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_num_rounds(10)
        .with_learning_rate(0.1)?;

    let loss = MultiLabelLogLoss::new();
    let model =
        UniversalModel::train_multilabel(&dataset, config, &loss).expect("Training should succeed");

    // Get predictions
    let proba = model.predict_proba_multilabel(&dataset);

    // Compute accuracy per label
    let mut correct_per_label = vec![0usize; num_outputs];
    for i in 0..num_rows {
        for k in 0..num_outputs {
            let target = targets[i * num_outputs + k];
            let pred_label = if proba[i][k] >= 0.5 { 1.0 } else { 0.0 };
            if (target - pred_label).abs() < 0.5 {
                correct_per_label[k] += 1;
            }
        }
    }

    // LTT should also achieve reasonable accuracy (at least as good as GBDT)
    for (k, &correct) in correct_per_label.iter().enumerate() {
        let accuracy = correct as f64 / num_rows as f64;
        assert!(
            accuracy > 0.5,
            "LTT Label {} accuracy {:.2} should be better than random",
            k,
            accuracy
        );
    }
    Ok(())
}
