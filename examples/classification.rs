//! Binary Classification Example
//!
//! Demonstrates binary classification with:
//! - Synthetic 2D classification data
//! - Binary logistic loss
//! - Probability predictions and class assignment
//! - Classification metrics (accuracy, precision, recall)
//!
//! Run with: cargo run --release --example classification

#[path = "common/mod.rs"]
mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::BinnedDataset;

/// Generate synthetic binary classification dataset
/// Creates two blobs separated in feature space
fn create_classification_dataset(n_samples: usize, seed: u64) -> BinnedDataset {
    let mut rng = common::SimpleRng::new(seed);
    let n_features = 20;

    // Generate features and labels
    // Class 0: features centered around 0.3
    // Class 1: features centered around 0.7
    let mut row_major_features = Vec::with_capacity(n_samples * n_features);
    let targets: Vec<f32> = (0..n_samples)
        .map(|i| {
            let class = if i < n_samples / 2 { 0.0 } else { 1.0 };
            let center = if class == 0.0 { 0.3 } else { 0.7 };

            // Generate features for this sample
            for _f in 0..n_features {
                let noise = (rng.next_f32() - 0.5) * 0.3;
                let feature_val = ((center + noise) * 255.0).clamp(0.0, 255.0) as u8;
                row_major_features.push(feature_val);
            }

            class
        })
        .collect();

    // Reorder to column-major for BinnedDataset
    let mut col_major_features = vec![0u8; n_samples * n_features];
    for f in 0..n_features {
        for r in 0..n_samples {
            col_major_features[f * n_samples + r] = row_major_features[r * n_features + f];
        }
    }

    let feature_info = common::create_feature_info(n_features, "feature");
    BinnedDataset::new(n_samples, col_major_features, targets, feature_info)
}

/// Convert logit to probability using sigmoid function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("TreeBoost: Binary Classification Example");
    println!("{}", "=".repeat(60));
    println!();

    // 1. Create synthetic dataset
    let n_samples = 5000;
    let seed = 42;

    println!("1. Generating synthetic classification dataset...");
    println!("   Samples: {}", n_samples);
    println!("   Classes: 2 (binary)");
    println!("   Features: 20");
    println!();

    let dataset = create_classification_dataset(n_samples, seed);

    // 2. Configure the model
    println!("2. Configuring GBDT model for binary classification...");
    let config = GBDTConfig::new()
        .with_num_rounds(100)
        .with_max_depth(6)
        .with_learning_rate(0.1)
        .with_binary_logloss()
        .with_subsample(0.8)?
        .with_seed(42);

    println!("   Rounds: 100");
    println!("   Max depth: 6");
    println!("   Loss function: Binary Logistic Loss");
    println!();

    // 3. Train the model
    println!("3. Training model...");
    let start = std::time::Instant::now();
    let model = GBDTModel::train_binned(&dataset, config).expect("Training failed");
    let elapsed = start.elapsed();

    println!("   Time: {:.2?}", elapsed);
    println!("   Trees: {}", model.num_trees());
    println!();

    // 4. Make predictions (raw logits)
    println!("4. Making predictions...");
    let logits = model.predict(&dataset);

    // Convert to probabilities using sigmoid
    let probabilities: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();

    // Classify with threshold 0.5
    let threshold = 0.5;
    let predictions: Vec<f32> = probabilities
        .iter()
        .map(|&p| if p >= threshold { 1.0 } else { 0.0 })
        .collect();

    println!();

    // 5. Evaluate performance
    println!("5. Evaluating classification performance...");

    let targets = dataset.targets();

    // Calculate accuracy
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(pred, &target)| (**pred - target).abs() < 0.01)
        .count();
    let accuracy = correct as f32 / predictions.len() as f32;

    // Calculate precision and recall
    let tp = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(pred, &target)| **pred >= 0.5 && target >= 0.5)
        .count();
    let fp = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(pred, &target)| **pred >= 0.5 && target < 0.5)
        .count();
    let fn_ = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(pred, &target)| **pred < 0.5 && target >= 0.5)
        .count();

    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f32 / (tp + fn_) as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    println!("   Accuracy:  {:.4}", accuracy);
    println!("   Precision: {:.4}", precision);
    println!("   Recall:    {:.4}", recall);
    println!("   F1-Score:  {:.4}", f1);
    println!();

    // 6. Feature importance
    println!("6. Feature Importance (top 5):");
    let importances = model.feature_importance();
    let mut indexed: Vec<(usize, f32)> = importances
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (feature_idx, importance) in indexed.iter().take(5) {
        println!("   Feature {}: {:.6}", feature_idx, importance);
    }
    println!();

    // 7. Show probability calibration
    println!("7. Probability Distribution:");

    // Probability bins
    let mut prob_bins = [0usize; 10];
    for prob in &probabilities {
        let bin = ((prob * 10.0).min(9.0)) as usize;
        prob_bins[bin] += 1;
    }

    for (i, count) in prob_bins.iter().enumerate() {
        let start = i as f32 / 10.0;
        let end = (i + 1) as f32 / 10.0;
        let bar = "*".repeat(*count / 50);
        println!("   [{:.1}-{:.1}]: {} ({})", start, end, bar, count);
    }
    println!();

    // 8. Sample predictions
    println!("8. Sample Predictions:");
    for i in (0..predictions.len()).step_by(predictions.len() / 5) {
        println!(
            "   Sample {}: Logit={:7.4}, Prob={:.4}, Pred={}, True={}",
            i, logits[i], probabilities[i], predictions[i] as i32, targets[i] as i32
        );
    }
    println!();

    println!("{}", "=".repeat(60));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(60));
    Ok(())
}
