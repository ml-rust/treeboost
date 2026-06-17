//! Random Forest Robustness Test - Noise Feature Handling
//!
//! This test validates that Random Forest (and regularized GBDT with feature sampling)
//! can handle high-dimensional noisy data better than naive GBDT.
//!
//! Test Setup (Needle in a Haystack):
//! - 1000 features: 10 informative, 990 pure Gaussian noise
//! - High noise-to-signal ratio in target (25.0 noise stddev vs ~50 signal range)
//!
//! Expected Behavior:
//! 1. Naive GBDT (no feature sampling): Overfits to noise → large train/val gap
//! 2. Regularized GBDT (with colsample): Ignores noise via feature bagging → small gap
//! 3. Random Forest: Built-in feature bagging → small gap (similar to regularized GBDT)
//!
//! This test is IGNORED by default because it's slow (~2.5 minutes on 1000 features).
//! Run manually with: cargo test test_rf_vs_gbdt_noise_robustness -- --ignored

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use treeboost::{
    BinnedDataset, BoostingMode, FeatureInfo, FeatureType, MseLoss, TreeConfig, UniversalConfig,
    UniversalModel,
};

/// Generate synthetic dataset with 1000 features (10 informative, 990 noise)
///
/// Signal: y = 10*feat_0 + 9*feat_1 + ... + 1*feat_9 + noise
/// where noise ~ N(0, 25) has high variance relative to signal range (~50)
fn generate_noise_dataset(num_rows: usize) -> BinnedDataset {
    let mut rng = StdRng::seed_from_u64(42);

    let num_features = 1000;
    let num_informative = 10;

    // Column-major layout: all rows for feature 0, then all rows for feature 1, etc.
    let mut features = Vec::with_capacity(num_rows * num_features);
    let mut signal = vec![0.0f32; num_rows];

    // Generate 1000 features: 10 informative + 990 noise
    for i in 0..num_features {
        let dist = Normal::new(0.0f32, 1.0f32).unwrap();

        for sig in signal.iter_mut() {
            // Sample value and bin it to u8 range [0, 255]
            let val = dist.sample(&mut rng);
            let binned = ((val + 3.0).clamp(0.0, 6.0) / 6.0 * 255.0) as u8;
            features.push(binned);

            // First 10 features contribute to signal (with decaying weights)
            if i < num_informative {
                let weight = (10 - i) as f32;
                *sig += val * weight;
            }
        }
    }

    // Add high-variance noise to target (validates bagging effectiveness)
    // Signal range ≈ [-50, 50], noise stddev = 25 → challenging noise-to-signal ratio
    let noise_dist = Normal::new(0.0f32, 25.0f32).unwrap();
    let targets: Vec<f32> = signal
        .iter()
        .map(|s| s + noise_dist.sample(&mut rng))
        .collect();

    // Create feature info
    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("feat_{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
            impute_value: 0.0,
        })
        .collect();

    BinnedDataset::new(num_rows, features, targets, feature_info)
}

/// Compute RMSE for model predictions
fn compute_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    mse.sqrt()
}

/// Extract subset of dataset by row indices
fn extract_subset(dataset: &BinnedDataset, start: usize, end: usize) -> BinnedDataset {
    let n_rows = end - start;
    let n_features = dataset.num_features();

    // Extract features (column-major)
    let mut features = Vec::with_capacity(n_rows * n_features);
    for f in 0..n_features {
        let col = dataset.feature_column(f);
        for &val in &col[start..end] {
            features.push(val);
        }
    }

    // Extract targets
    let targets: Vec<f32> = dataset.targets()[start..end].to_vec();

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

#[test]
#[ignore] // SLOW TEST (~2.5min): Training 3 models on 2000×1000 data. Run with: cargo test -- --ignored
fn test_rf_vs_gbdt_noise_robustness() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Random Forest Robustness Test ===");
    println!("Dataset: 2000 rows × 1000 features (10 informative, 990 noise)");
    println!("Comparing: Naive GBDT vs Regularized GBDT vs Random Forest\n");

    let dataset = generate_noise_dataset(2000);
    let loss_fn = MseLoss;

    // Split into train/validation (80/20)
    let split_idx = (dataset.num_rows() as f32 * 0.8) as usize;
    let train_data = extract_subset(&dataset, 0, split_idx);
    let val_data = extract_subset(&dataset, split_idx, dataset.num_rows());

    println!("Training Naive GBDT (no feature sampling)...");
    // 1. Naive GBDT - No feature sampling (prone to overfitting noise)
    let naive_tree_config = TreeConfig::default()
        .with_colsample(1.0)
        .unwrap() // KEY: No feature sampling
        .with_max_depth(8)
        .unwrap();

    let naive_config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(100)
        .with_learning_rate(0.3)?
        .with_subsample(1.0)? // No row sampling either
        .with_tree_config(naive_tree_config)
        .with_seed(42);

    let naive_gbdt = UniversalModel::train(&train_data, naive_config, &loss_fn)
        .expect("Naive GBDT training failed");

    let naive_train_preds = naive_gbdt.predict(&train_data);
    let naive_val_preds = naive_gbdt.predict(&val_data);
    let naive_train_rmse = compute_rmse(&naive_train_preds, train_data.targets());
    let naive_val_rmse = compute_rmse(&naive_val_preds, val_data.targets());
    let naive_gap = (naive_val_rmse / naive_train_rmse - 1.0) * 100.0;

    println!("Training Regularized GBDT (with feature sampling)...");
    // 2. Regularized GBDT - WITH feature sampling (mimics RF behavior)
    let reg_tree_config = TreeConfig::default()
        .with_colsample(0.1)
        .unwrap() // KEY: Feature sampling like RF
        .with_max_depth(5)
        .unwrap();

    let reg_config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(100)
        .with_learning_rate(0.05)?
        .with_subsample(0.8)? // Also add row sampling
        .with_tree_config(reg_tree_config)
        .with_seed(42);

    let reg_gbdt = UniversalModel::train(&train_data, reg_config, &loss_fn)
        .expect("Regularized GBDT training failed");

    let reg_train_preds = reg_gbdt.predict(&train_data);
    let reg_val_preds = reg_gbdt.predict(&val_data);
    let reg_train_rmse = compute_rmse(&reg_train_preds, train_data.targets());
    let reg_val_rmse = compute_rmse(&reg_val_preds, val_data.targets());
    let reg_gap = (reg_val_rmse / reg_train_rmse - 1.0) * 100.0;

    println!("Training Random Forest (built-in feature bagging)...");
    // 3. Random Forest - Built-in feature bagging via colsample
    let rf_tree_config = TreeConfig::default()
        .with_colsample(0.03)
        .unwrap() // sqrt(1000)/1000 ≈ 0.03 (mtry)
        .with_max_depth(5)
        .unwrap();

    let rf_config = UniversalConfig::new()
        .with_mode(BoostingMode::RandomForest)
        .with_num_rounds(100)
        .with_subsample(0.7)? // Bootstrap sampling
        .with_tree_config(rf_tree_config)
        .with_seed(42);

    let rf = UniversalModel::train(&train_data, rf_config, &loss_fn)
        .expect("Random Forest training failed");

    let rf_train_preds = rf.predict(&train_data);
    let rf_val_preds = rf.predict(&val_data);
    let rf_train_rmse = compute_rmse(&rf_train_preds, train_data.targets());
    let rf_val_rmse = compute_rmse(&rf_val_preds, val_data.targets());
    let rf_gap = (rf_val_rmse / rf_train_rmse - 1.0) * 100.0;

    // Print results
    println!("\n=== Results ===");
    println!(
        "Naive GBDT:       Train={:.2}, Val={:.2}, Gap={:.1}%",
        naive_train_rmse, naive_val_rmse, naive_gap
    );
    println!(
        "Regularized GBDT: Train={:.2}, Val={:.2}, Gap={:.1}%",
        reg_train_rmse, reg_val_rmse, reg_gap
    );
    println!(
        "Random Forest:    Train={:.2}, Val={:.2}, Gap={:.1}%",
        rf_train_rmse, rf_val_rmse, rf_gap
    );

    // Assertions

    // 1. Naive GBDT should overfit (large train/val gap)
    let naive_gap_ratio = naive_val_rmse / naive_train_rmse;
    assert!(
        naive_gap_ratio > 1.15,
        "Naive GBDT should overfit on noisy features (expected gap >15%, got {:.1}%)",
        naive_gap
    );

    // 2. Regularized GBDT should be more robust than naive (not necessarily better val RMSE)
    // Focus on the gap reduction, not absolute performance
    assert!(reg_gap < naive_gap * 0.01,
            "Regularized GBDT should have much smaller train/val gap than naive (expected <1% of naive gap, got {:.1}% vs {:.1}%)",
            reg_gap, naive_gap);

    // 3. RF should have smallest train/val gap (best robustness)
    assert!(
        rf_gap < 20.0,
        "RF should have small train/val gap (expected <20%, got {:.1}%)",
        rf_gap
    );

    // 4. Regularized GBDT should not overfit like naive (validation error should be better or similar)
    assert!(reg_val_rmse <= naive_val_rmse * 1.05,
            "Regularized GBDT should have similar or better validation performance than naive (got {:.2} vs {:.2})",
            reg_val_rmse, naive_val_rmse);

    // 5. Both regularized models should have dramatically smaller gaps than naive
    assert!(
        reg_gap < naive_gap * 0.01,
        "Regularized GBDT gap ({:.1}%) should be <1% of naive gap ({:.1}%)",
        reg_gap,
        naive_gap
    );
    assert!(
        rf_gap < naive_gap * 0.001,
        "RF gap ({:.1}%) should be <0.1% of naive gap ({:.1}%)",
        rf_gap,
        naive_gap
    );

    println!("\n✅ All assertions passed!");
    println!("- Naive GBDT overfits to noise (as expected)");
    println!("- Regularized GBDT and RF are robust to noise features");
    println!("- Feature sampling effectively ignores irrelevant features\n");
    Ok(())
}
