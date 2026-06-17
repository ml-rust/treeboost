//! Multi-Label Classification Example
//!
//! Demonstrates multi-label classification with TreeBoost:
//! - Synthetic multi-label dataset (multiple binary targets per sample)
//! - AutoML multi-label training with LinearThenTree mode
//! - Per-label threshold tuning to optimize F1 scores
//! - Label prediction with tuned thresholds
//!
//! Run with: cargo run --release --example multilabel

#[path = "common/mod.rs"]
mod common;

use polars::prelude::*;
use treeboost::model::{AutoModel, BoostingMode};

/// Create a synthetic multi-label DataFrame.
///
/// Each sample has 3 features and 3 binary labels.
/// Labels are determined by thresholds on feature combinations:
/// - label_0: x1 + x2 > 5.0
/// - label_1: x2 + x3 > 6.0
/// - label_2: x1 + x3 > 5.5
fn create_multilabel_dataframe(n_samples: usize, seed: u64) -> DataFrame {
    let mut rng = common::SimpleRng::new(seed);

    // Generate features
    let x1: Vec<f64> = (0..n_samples)
        .map(|_| rng.next_range(0.0, 10.0) as f64)
        .collect();
    let x2: Vec<f64> = (0..n_samples)
        .map(|_| rng.next_range(0.0, 10.0) as f64)
        .collect();
    let x3: Vec<f64> = (0..n_samples)
        .map(|_| rng.next_range(0.0, 10.0) as f64)
        .collect();

    // Generate labels based on feature thresholds
    let label_0: Vec<i32> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| if a + b > 5.0 { 1 } else { 0 })
        .collect();

    let label_1: Vec<i32> = x2
        .iter()
        .zip(x3.iter())
        .map(|(&b, &c)| if b + c > 6.0 { 1 } else { 0 })
        .collect();

    let label_2: Vec<i32> = x1
        .iter()
        .zip(x3.iter())
        .map(|(&a, &c)| if a + c > 5.5 { 1 } else { 0 })
        .collect();

    DataFrame::new(
        n_samples,
        vec![
            Column::new("x1".into(), x1),
            Column::new("x2".into(), x2),
            Column::new("x3".into(), x3),
            Column::new("label_0".into(), label_0),
            Column::new("label_1".into(), label_1),
            Column::new("label_2".into(), label_2),
        ],
    )
    .unwrap()
}

/// Compute per-label accuracy
fn compute_accuracy(predictions: &[Vec<bool>], targets: &[Vec<f64>]) -> Vec<f64> {
    let n_labels = predictions[0].len();
    let n_samples = predictions.len();

    (0..n_labels)
        .map(|k| {
            let correct = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| {
                    let p = if pred[k] { 1.0 } else { 0.0 };
                    (p - target[k]).abs() < 0.5
                })
                .count();
            correct as f64 / n_samples as f64
        })
        .collect()
}

/// Compute per-label F1 scores
fn compute_f1_scores(predictions: &[Vec<bool>], targets: &[Vec<f64>]) -> Vec<f64> {
    let n_labels = predictions[0].len();

    (0..n_labels)
        .map(|k| {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_ = 0;

            for (pred, target) in predictions.iter().zip(targets.iter()) {
                let p = pred[k];
                let t = target[k] >= 0.5;

                if p && t {
                    tp += 1;
                } else if p && !t {
                    fp += 1;
                } else if !p && t {
                    fn_ += 1;
                }
            }

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };

            if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(70));
    println!("TreeBoost: Multi-Label Classification Example");
    println!("{}", "=".repeat(70));
    println!();

    // 1. Create synthetic datasets
    println!("1. Generating synthetic multi-label datasets...");
    let train_df = create_multilabel_dataframe(1000, 42);
    let val_df = create_multilabel_dataframe(300, 123);
    let test_df = create_multilabel_dataframe(200, 456);

    println!("   Training samples:   {}", train_df.height());
    println!("   Validation samples: {}", val_df.height());
    println!("   Test samples:       {}", test_df.height());
    println!("   Features: 3 (x1, x2, x3)");
    println!("   Labels:   3 (label_0, label_1, label_2)");
    println!();

    // 2. Train multi-label model
    let target_cols = vec!["label_0", "label_1", "label_2"];

    println!("2. Training multi-label model with LinearThenTree mode...");
    let start = std::time::Instant::now();

    let mut model = AutoModel::train_multilabel_with_mode(
        &train_df,
        &target_cols,
        BoostingMode::LinearThenTree,
    )?;

    let train_time = start.elapsed();
    println!("   Training time: {:.2?}", train_time);
    println!("   Mode: {:?}", model.mode());
    println!("   Labels: {}", model.num_labels());
    println!();

    // 3. Tune thresholds on validation set
    println!("3. Tuning thresholds on validation set...");
    let tune_result = model.tune_thresholds(&val_df, &target_cols)?;

    println!("   Optimal thresholds:");
    for (k, &threshold) in tune_result.thresholds.iter().enumerate() {
        println!(
            "     label_{}: threshold={:.2}, F1={:.4}, precision={:.4}, recall={:.4}",
            k,
            threshold,
            tune_result.f1_scores[k],
            tune_result.precisions[k],
            tune_result.recalls[k]
        );
    }
    println!();

    // 4. Make predictions on test set
    println!("4. Predicting on test set...");

    // Get probability predictions
    let proba = model.predict_proba_multilabel(&test_df)?;

    // Get label predictions with default 0.5 threshold
    let labels_default = model.predict_labels_with_threshold(&test_df, 0.5)?;

    // Get label predictions with tuned thresholds
    let labels_tuned = model.predict_labels_tuned(&test_df)?;

    println!("   Test samples: {}", test_df.height());
    println!();

    // 5. Extract ground truth for evaluation
    let targets: Vec<Vec<f64>> = (0..test_df.height())
        .map(|i| {
            target_cols
                .iter()
                .map(|col| {
                    test_df
                        .column(col)
                        .unwrap()
                        .cast(&DataType::Float64)
                        .unwrap()
                        .f64()
                        .unwrap()
                        .get(i)
                        .unwrap()
                })
                .collect()
        })
        .collect();

    // 6. Compare default vs tuned thresholds
    println!("5. Comparing default (0.5) vs tuned thresholds...");
    println!();

    let acc_default = compute_accuracy(&labels_default, &targets);
    let f1_default = compute_f1_scores(&labels_default, &targets);

    let acc_tuned = compute_accuracy(&labels_tuned, &targets);
    let f1_tuned = compute_f1_scores(&labels_tuned, &targets);

    println!("   Per-label Accuracy:");
    println!("   {:>10} {:>12} {:>12}", "Label", "Default", "Tuned");
    for k in 0..3 {
        println!(
            "   {:>10} {:>12.4} {:>12.4}",
            format!("label_{}", k),
            acc_default[k],
            acc_tuned[k]
        );
    }
    println!();

    println!("   Per-label F1 Score:");
    println!("   {:>10} {:>12} {:>12}", "Label", "Default", "Tuned");
    for k in 0..3 {
        println!(
            "   {:>10} {:>12.4} {:>12.4}",
            format!("label_{}", k),
            f1_default[k],
            f1_tuned[k]
        );
    }
    println!();

    // 7. Sample predictions
    println!("6. Sample predictions (first 5 samples):");
    println!(
        "   {:>6} {:>20} {:>20} {:>20}",
        "Sample", "Probabilities", "Predicted", "True"
    );
    for i in 0..5.min(test_df.height()) {
        let probs: String = proba[i]
            .iter()
            .map(|p| format!("{:.2}", p))
            .collect::<Vec<_>>()
            .join(", ");
        let preds: String = labels_tuned[i]
            .iter()
            .map(|&b| if b { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(", ");
        let trues: String = targets[i]
            .iter()
            .map(|&t| if t >= 0.5 { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "   {:>6} [{:>18}] [{:>18}] [{:>18}]",
            i, probs, preds, trues
        );
    }
    println!();

    // 8. Summary statistics
    println!("7. Summary:");
    let avg_f1_default: f64 = f1_default.iter().sum::<f64>() / f1_default.len() as f64;
    let avg_f1_tuned: f64 = f1_tuned.iter().sum::<f64>() / f1_tuned.len() as f64;
    println!("   Average F1 (default threshold): {:.4}", avg_f1_default);
    println!("   Average F1 (tuned thresholds):  {:.4}", avg_f1_tuned);

    if avg_f1_tuned > avg_f1_default {
        let improvement = (avg_f1_tuned - avg_f1_default) / avg_f1_default * 100.0;
        println!("   Improvement from tuning: +{:.2}%", improvement);
    }
    println!();

    println!("{}", "=".repeat(70));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(70));

    Ok(())
}
