//! LinearTreeBooster Example
//!
//! Demonstrates the LinearTreeBooster - a hybrid weak learner that combines:
//! - Decision tree partitioning (splits feature space into regions)
//! - Ridge regression in each leaf (fits linear model per region)
//!
//! Benefits:
//! - 10-100x fewer trees needed for same accuracy
//! - Smoother predictions than standard trees
//! - Better extrapolation within leaf regions
//!
//! Run with: cargo run --release --example linear_tree

#[path = "common/mod.rs"]
mod common;

use treeboost::dataset::BinnedDataset;
use treeboost::learner::{LinearConfig, LinearTreeBooster, LinearTreeConfig, TreeConfig};

/// Generate synthetic dataset with local linear relationships
fn create_synthetic_dataset(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (BinnedDataset, Vec<f32>) {
    let mut rng = common::SimpleRng::new(seed);

    // Generate features (column-major layout for BinnedDataset)
    let mut binned_features = Vec::with_capacity(n_samples * n_features);
    for _f in 0..n_features {
        for _r in 0..n_samples {
            binned_features.push((rng.next_f32() * 255.0) as u8);
        }
    }

    // Raw features (row-major layout for LinearTreeBooster)
    let mut raw_features = Vec::with_capacity(n_samples * n_features);
    for r in 0..n_samples {
        for f in 0..n_features {
            let bin_val = binned_features[f * n_samples + r] as f32;
            raw_features.push(bin_val);
        }
    }

    // Generate targets with piecewise linear relationships:
    // - Region 1 (f0 < 128): y = 2*f0 + 1*f1 + noise
    // - Region 2 (f0 >= 128): y = -1*f0 + 3*f1 + 200 + noise
    // This is ideal for LinearTree: tree finds the split, linear models fit each region
    let targets: Vec<f32> = (0..n_samples)
        .map(|i| {
            let f0 = raw_features[i * n_features] / 255.0;
            let f1 = raw_features[i * n_features + 1] / 255.0;
            let noise = rng.next_f32() * 0.1;

            if f0 < 0.5 {
                // Region 1: positive slope on f0
                2.0 * f0 + 1.0 * f1 + noise
            } else {
                // Region 2: negative slope on f0, offset
                -f0 + 3.0 * f1 + 2.0 + noise
            }
        })
        .collect();

    let feature_info = common::create_feature_info(n_features, "feature");
    let dataset = BinnedDataset::new(n_samples, binned_features, targets.clone(), feature_info);

    (dataset, raw_features)
}

/// Calculate R² score
fn r_squared(predictions: &[f32], targets: &[f32]) -> f32 {
    let mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let ss_res: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f32 = targets.iter().map(|t| (t - mean).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TreeBoost: LinearTreeBooster Example");
    println!("{}", "=".repeat(70));
    println!();

    // Create synthetic data with piecewise linear structure
    let n_samples = 3000;
    let n_features = 5;
    let seed = 42;

    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!("Target: Piecewise linear function");
    println!("  - Region 1 (f0 < 0.5): y = 2*f0 + 1*f1");
    println!("  - Region 2 (f0 >= 0.5): y = -1*f0 + 3*f1 + 2.0");
    println!("  (Perfect for LinearTree: tree splits, linear models fit each region)");
    println!();

    let (dataset, raw_features) = create_synthetic_dataset(n_samples, n_features, seed);

    // Compute gradients/hessians for regression (MSE loss)
    // For MSE: gradient = prediction - target, hessian = 1
    // Initial prediction = mean(targets)
    let mean_target = dataset.targets().iter().sum::<f32>() / n_samples as f32;
    let gradients: Vec<f32> = dataset
        .targets()
        .iter()
        .map(|&t| mean_target - t) // negative gradient direction
        .collect();
    let hessians = vec![1.0f32; n_samples];

    // =========================================================================
    // LinearTreeBooster Configuration
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("LinearTreeBooster Configuration");
    println!("{}", "-".repeat(70));

    let config = LinearTreeConfig::new()
        .with_tree_config(
            TreeConfig::default()
                .with_max_depth(3) // Shallow tree - let linear models do the work
                .expect("valid max_depth")
                .with_max_leaves(8)
                .expect("valid max_leaves")
                .with_min_samples_leaf(50)
                .expect("valid min_samples_leaf"),
        )
        .with_linear_config(
            LinearConfig::default()
                .with_lambda(0.1) // Light regularization
                .expect("valid lambda")
                .with_max_iter(100)
                .expect("valid max_iter"),
        )
        .with_min_samples_for_linear(20);

    println!("Tree config:");
    println!("  - max_depth: 3 (shallow - leaves do the heavy lifting)");
    println!("  - max_leaves: 8");
    println!("  - min_samples_leaf: 50");
    println!();
    println!("Linear config (per leaf):");
    println!("  - lambda: 0.1 (L2 regularization)");
    println!("  - max_iter: 100 (coordinate descent iterations)");
    println!();
    println!("min_samples_for_linear: 20");
    println!("  (leaves with fewer samples use constant prediction)");
    println!();

    // =========================================================================
    // Training
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("Training LinearTreeBooster");
    println!("{}", "-".repeat(70));

    let mut booster = LinearTreeBooster::new(config);

    let start = std::time::Instant::now();
    booster
        .fit_on_gradients(&dataset, &raw_features, n_features, &gradients, &hessians)
        .expect("Training failed");
    let elapsed = start.elapsed();

    println!("Training time: {:.2?}", elapsed);
    println!("Leaf models: {}", booster.num_leaf_models());
    println!("Total params: {}", booster.num_params());
    println!();

    // =========================================================================
    // Prediction
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("Prediction & Evaluation");
    println!("{}", "-".repeat(70));

    let predictions = booster.predict_batch(&dataset, &raw_features, n_features);

    // Adjust predictions (we trained on residuals from mean)
    let final_predictions: Vec<f32> = predictions.iter().map(|&p| mean_target + p).collect();

    let r2 = r_squared(&final_predictions, dataset.targets());
    let mse: f32 = final_predictions
        .iter()
        .zip(dataset.targets().iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n_samples as f32;

    println!("R² Score: {:.4}", r2);
    println!("MSE: {:.6}", mse);
    println!();

    // =========================================================================
    // Sample Predictions
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("Sample Predictions");
    println!("{}", "-".repeat(70));
    println!();
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10}",
        "Sample", "f0", "Actual", "Predicted", "Error"
    );
    println!("{}", "-".repeat(58));

    for i in (0..n_samples).step_by(n_samples / 10) {
        let f0 = raw_features[i * n_features] / 255.0;
        let actual = dataset.targets()[i];
        let pred = final_predictions[i];
        let error = (pred - actual).abs();
        println!(
            "{:<8} {:>10.3} {:>10.3} {:>10.3} {:>10.4}",
            i, f0, actual, pred, error
        );
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("{}", "=".repeat(70));
    println!("LinearTreeBooster Benefits");
    println!("{}", "=".repeat(70));
    println!();
    println!("1. Fewer trees needed: Tree partitions space, linear models fit each region");
    println!("2. Smoother predictions: Linear interpolation within leaves");
    println!("3. Better extrapolation: Can extrapolate within leaf boundaries");
    println!("4. Interpretable: Each leaf has explainable linear coefficients");
    println!();
    println!("Use LinearTreeBooster when:");
    println!("- Data has piecewise linear structure");
    println!("- Smooth predictions are important");
    println!("- You need to reduce model complexity");
    println!();
    println!("Example completed successfully!");
}
