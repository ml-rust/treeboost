//! Evaluation metrics for hyperparameter tuning
//!
//! Provides metrics for assessing model performance during tuning.
//! All metrics are computed in a numerically stable manner.

use crate::loss::LossFunction;

/// Metric types for evaluating model performance
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Metric {
    /// Mean Squared Error (regression)
    #[default]
    Mse,
    /// Root Mean Squared Error (regression)
    Rmse,
    /// Mean Absolute Error (regression)
    Mae,
    /// Binary log loss (binary classification)
    BinaryLogLoss,
    /// Multi-class log loss (multi-class classification)
    MultiClassLogLoss { n_classes: usize },
    /// Accuracy (classification)
    Accuracy { threshold: f32 },
    /// ROC-AUC (binary classification ranking metric)
    RocAuc,
}

impl Metric {
    /// Create MSE metric
    pub fn mse() -> Self {
        Metric::Mse
    }

    /// Create RMSE metric
    pub fn rmse() -> Self {
        Metric::Rmse
    }

    /// Create MAE metric
    pub fn mae() -> Self {
        Metric::Mae
    }

    /// Create binary log loss metric
    pub fn binary_log_loss() -> Self {
        Metric::BinaryLogLoss
    }

    /// Create multi-class log loss metric
    pub fn multi_class_log_loss(n_classes: usize) -> Self {
        Metric::MultiClassLogLoss { n_classes }
    }

    /// Create accuracy metric with default threshold (0.5)
    pub fn accuracy() -> Self {
        Metric::Accuracy { threshold: 0.5 }
    }

    /// Create accuracy metric with custom threshold
    pub fn accuracy_with_threshold(threshold: f32) -> Self {
        Metric::Accuracy { threshold }
    }

    /// Create ROC-AUC metric (binary classification)
    pub fn roc_auc() -> Self {
        Metric::RocAuc
    }

    /// Auto-select metric from loss function type
    pub fn from_loss_type(loss: &dyn LossFunction) -> Self {
        let name = loss.name();
        match name {
            "mse" | "pseudo_huber" => Metric::Mse,
            "binary_log_loss" => Metric::BinaryLogLoss,
            "multi_class_log_loss" => Metric::MultiClassLogLoss { n_classes: 2 },
            _ => Metric::Mse,
        }
    }

    /// Whether lower values are better for this metric
    pub fn lower_is_better(&self) -> bool {
        match self {
            Metric::Mse | Metric::Rmse | Metric::Mae => true,
            Metric::BinaryLogLoss | Metric::MultiClassLogLoss { .. } => true,
            Metric::Accuracy { .. } | Metric::RocAuc => false,
        }
    }

    /// Compute the metric value
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (raw scores for classification)
    ///   - For regression/binary: one prediction per sample
    ///   - For multi-class: n_classes predictions per sample (logits)
    /// * `targets` - Ground truth values (0/1 for binary, class indices for multi)
    ///
    /// # Returns
    /// The metric value, or f32::INFINITY on error
    pub fn compute(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return f32::INFINITY;
        }

        // For multi-class, predictions has n_samples * n_classes elements
        // For other metrics, predictions.len() == targets.len()
        match self {
            Metric::MultiClassLogLoss { n_classes } => {
                // Multi-class: predictions has n_samples * n_classes elements
                if predictions.len() != targets.len() * n_classes {
                    return f32::INFINITY;
                }
                compute_multi_class_log_loss(predictions, targets, *n_classes)
            }
            _ => {
                // Other metrics: predictions and targets have same length
                if predictions.len() != targets.len() {
                    return f32::INFINITY;
                }
                match self {
                    Metric::Mse => compute_mse(predictions, targets),
                    Metric::Rmse => compute_rmse(predictions, targets),
                    Metric::Mae => compute_mae(predictions, targets),
                    Metric::BinaryLogLoss => compute_binary_log_loss(predictions, targets),
                    Metric::Accuracy { threshold } => {
                        compute_accuracy(predictions, targets, *threshold)
                    }
                    Metric::RocAuc => compute_roc_auc(predictions, targets) as f32,
                    Metric::MultiClassLogLoss { .. } => unreachable!(),
                }
            }
        }
    }

    /// Return the name of the metric
    pub fn name(&self) -> &'static str {
        match self {
            Metric::Mse => "mse",
            Metric::Rmse => "rmse",
            Metric::Mae => "mae",
            Metric::BinaryLogLoss => "binary_log_loss",
            Metric::MultiClassLogLoss { .. } => "multi_class_log_loss",
            Metric::Accuracy { .. } => "accuracy",
            Metric::RocAuc => "roc_auc",
        }
    }
}

/// Compute ROC-AUC score using trapezoidal integration
///
/// Predictions are raw logits (will be converted to probabilities via sigmoid).
/// Targets should be 0.0 or 1.0.
pub fn compute_roc_auc(predictions: &[f32], targets: &[f32]) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }

    // Convert predictions to probabilities
    let probs: Vec<f64> = predictions.iter().map(|&p| sigmoid(p) as f64).collect();
    let targets_f64: Vec<f64> = targets.iter().map(|&t| t as f64).collect();

    // Sort by descending probability
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Count positives and negatives
    let n_pos = targets_f64.iter().filter(|&&t| t > 0.5).count();
    let n_neg = targets_f64.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return 0.5; // Undefined, return random classifier score
    }

    // Compute TPR and FPR at each threshold
    let mut tpr_points = Vec::with_capacity(indices.len() + 1);
    let mut fpr_points = Vec::with_capacity(indices.len() + 1);

    tpr_points.push(0.0);
    fpr_points.push(0.0);

    let mut tp = 0.0;
    let mut fp = 0.0;

    for &idx in &indices {
        if targets_f64[idx] > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        tpr_points.push(tp / n_pos as f64);
        fpr_points.push(fp / n_neg as f64);
    }

    // Trapezoidal integration
    let mut auc = 0.0;
    for i in 1..tpr_points.len() {
        let width = fpr_points[i] - fpr_points[i - 1];
        let height = (tpr_points[i] + tpr_points[i - 1]) / 2.0;
        auc += width * height;
    }

    auc
}

/// Compute Rank IC (Spearman rank correlation coefficient)
///
/// Measures the correlation between prediction ranks and target ranks.
/// Returns a value in [-1, 1] where:
/// - 1.0 = perfect positive correlation (predictions perfectly rank targets)
/// - 0.0 = no correlation
/// - -1.0 = perfect negative correlation
///
/// Common in quantitative finance for measuring prediction quality.
/// Compute Rank IC (Information Coefficient)
///
/// For panel data (e.g., stock returns), IC should be computed per time period:
/// IC = Σ Corr(rank(pred_t), rank(target_t)) / T
///
/// When `era_indices` is provided, computes IC for each era (time period) and averages.
/// When `era_indices` is None, computes a single Spearman correlation (legacy behavior).
pub fn compute_rank_ic(predictions: &[f32], targets: &[f32], era_indices: Option<&[u16]>) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }

    let n = predictions.len();
    if n < 2 {
        return 0.0;
    }

    // If no era indices, compute single Spearman correlation (legacy)
    let Some(eras) = era_indices else {
        return compute_spearman_correlation(predictions, targets);
    };

    if eras.len() != n {
        return 0.0;
    }

    // Group by era and compute IC for each era
    use std::collections::HashMap;
    let mut era_groups: HashMap<u16, (Vec<f32>, Vec<f32>)> = HashMap::new();

    for i in 0..n {
        let era = eras[i];
        era_groups
            .entry(era)
            .or_insert_with(|| (Vec::new(), Vec::new()));
        let group = era_groups.get_mut(&era).unwrap();
        group.0.push(predictions[i]);
        group.1.push(targets[i]);
    }

    // Compute Spearman correlation for each era
    let mut ic_sum = 0.0;
    let mut valid_eras = 0;

    for (_era_id, (era_preds, era_targets)) in era_groups.iter() {
        if era_preds.len() < 2 {
            continue; // Skip eras with < 2 samples
        }
        let ic = compute_spearman_correlation(era_preds, era_targets);
        // Only include if correlation is valid (not NaN)
        if ic.is_finite() {
            ic_sum += ic;
            valid_eras += 1;
        }
    }

    if valid_eras == 0 {
        0.0
    } else {
        ic_sum / valid_eras as f64
    }
}

/// Compute Spearman rank correlation between two vectors
fn compute_spearman_correlation(predictions: &[f32], targets: &[f32]) -> f64 {
    let n = predictions.len();
    if n < 2 {
        return 0.0;
    }

    // Compute ranks for predictions
    let pred_ranks = compute_ranks(predictions);
    // Compute ranks for targets
    let target_ranks = compute_ranks(targets);

    // Compute Pearson correlation of ranks (which is Spearman correlation)
    let n_f64 = n as f64;

    // Mean of ranks
    let mean_pred: f64 = pred_ranks.iter().sum::<f64>() / n_f64;
    let mean_target: f64 = target_ranks.iter().sum::<f64>() / n_f64;

    // Covariance and standard deviations
    let mut cov = 0.0;
    let mut var_pred = 0.0;
    let mut var_target = 0.0;

    for i in 0..n {
        let d_pred = pred_ranks[i] - mean_pred;
        let d_target = target_ranks[i] - mean_target;
        cov += d_pred * d_target;
        var_pred += d_pred * d_pred;
        var_target += d_target * d_target;
    }

    let std_pred = var_pred.sqrt();
    let std_target = var_target.sqrt();

    // Use relative threshold based on expected rank standard deviation
    // For n ranks, expected std dev is roughly sqrt(n²/12), so threshold scales with n
    let epsilon = 1e-10 * n_f64;
    if std_pred < epsilon || std_target < epsilon {
        return 0.0; // No variation in ranks
    }

    cov / (std_pred * std_target)
}

/// Compute ranks with average rank for ties
fn compute_ranks(values: &[f32]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();

    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        let base_val = indexed[i].1;

        // Find all elements with the same value (ties)
        // Use relative comparison for robustness across different value magnitudes
        while j < n {
            let current_val = indexed[j].1;
            let abs_diff = (current_val - base_val).abs();
            let scale = (base_val.abs() + current_val.abs()) / 2.0;

            // Use f32::EPSILON * 4.0 as relative threshold (4x for numerical stability)
            // Falls back to absolute threshold for values near zero
            let threshold = if scale > 1e-6 {
                f32::EPSILON * 4.0 * scale
            } else {
                f32::EPSILON * 4.0
            };

            if abs_diff >= threshold {
                break;
            }
            j += 1;
        }

        // Assign average rank to all ties
        // Ranks are 1-based: positions i..j get average of (i+1)..(j+1)
        let avg_rank = (i + j + 1) as f64 / 2.0; // Average of (i+1) to j
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Compute Mean Squared Error
fn compute_mse(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n
}

/// Compute Root Mean Squared Error
fn compute_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    compute_mse(predictions, targets).sqrt()
}

/// Compute Mean Absolute Error
fn compute_mae(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>()
        / n
}

/// Compute binary log loss (cross-entropy)
///
/// Uses numerically stable implementation with clamping to avoid log(0)
fn compute_binary_log_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    const EPSILON: f32 = 1e-7;

    let n = predictions.len() as f32;
    let sum: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            // Apply sigmoid to raw predictions
            let prob = sigmoid(pred);
            // Clamp to avoid log(0)
            let prob = prob.clamp(EPSILON, 1.0 - EPSILON);
            // Binary cross-entropy
            -(target * prob.ln() + (1.0 - target) * (1.0 - prob).ln())
        })
        .sum();

    sum / n
}

/// Compute multi-class log loss
///
/// Predictions should be arranged as: `[class0_sample0, class1_sample0, ..., class0_sample1, ...]`
fn compute_multi_class_log_loss(predictions: &[f32], targets: &[f32], n_classes: usize) -> f32 {
    if n_classes < 2 {
        return f32::INFINITY;
    }

    const EPSILON: f32 = 1e-7;

    let n_samples = targets.len();
    if predictions.len() != n_samples * n_classes {
        return f32::INFINITY;
    }

    let mut sum = 0.0f32;

    for (i, &target) in targets.iter().enumerate() {
        let class_idx = target as usize;
        if class_idx >= n_classes {
            return f32::INFINITY;
        }

        // Get logits for this sample
        let logits = &predictions[i * n_classes..(i + 1) * n_classes];

        // Softmax with numerical stability (log-sum-exp trick)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_prob = logits[class_idx] - max_logit - exp_sum.ln();

        // Clamp for numerical stability
        sum -= log_prob.max(EPSILON.ln());
    }

    sum / n_samples as f32
}

/// Compute accuracy
fn compute_accuracy(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    let n = predictions.len() as f32;
    let correct: usize = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            // Apply sigmoid to get probability
            let prob = sigmoid(pred);
            let predicted_class = if prob >= threshold { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.5 {
                1
            } else {
                0
            }
        })
        .sum();

    correct as f32 / n
}

/// Sigmoid function
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_lower_is_better() {
        assert!(Metric::Mse.lower_is_better());
        assert!(Metric::Rmse.lower_is_better());
        assert!(Metric::Mae.lower_is_better());
        assert!(Metric::BinaryLogLoss.lower_is_better());
        assert!(Metric::multi_class_log_loss(3).lower_is_better());
        assert!(!Metric::accuracy().lower_is_better());
    }

    #[test]
    fn test_mse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let mse = Metric::Mse.compute(&predictions, &targets);
        assert!((mse - 0.0).abs() < 1e-6);

        let predictions = vec![2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let mse = Metric::Mse.compute(&predictions, &targets);
        assert!((mse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmse() {
        let predictions = vec![2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let rmse = Metric::Rmse.compute(&predictions, &targets);
        assert!((rmse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let predictions = vec![2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let mae = Metric::Mae.compute(&predictions, &targets);
        assert!((mae - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_log_loss() {
        // Perfect predictions should have very low loss
        let predictions = vec![10.0, 10.0, -10.0, -10.0]; // After sigmoid: ~1, ~1, ~0, ~0
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let loss = Metric::BinaryLogLoss.compute(&predictions, &targets);
        assert!(loss < 0.001);

        // Wrong predictions should have high loss
        let predictions = vec![-10.0, -10.0, 10.0, 10.0]; // After sigmoid: ~0, ~0, ~1, ~1
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let loss = Metric::BinaryLogLoss.compute(&predictions, &targets);
        assert!(loss > 5.0);
    }

    #[test]
    fn test_binary_log_loss_numerical_stability() {
        // Extreme values should not produce NaN or Inf
        let predictions = vec![1000.0, -1000.0, 0.0];
        let targets = vec![1.0, 0.0, 0.5];
        let loss = Metric::BinaryLogLoss.compute(&predictions, &targets);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_multi_class_log_loss() {
        // 3 classes, 2 samples
        // Sample 0: true class = 0, logits = [10, 0, 0] -> should predict class 0
        // Sample 1: true class = 2, logits = [0, 0, 10] -> should predict class 2
        let predictions = vec![
            10.0, 0.0, 0.0, // Sample 0
            0.0, 0.0, 10.0, // Sample 1
        ];
        let targets = vec![0.0, 2.0];
        let loss = Metric::multi_class_log_loss(3).compute(&predictions, &targets);
        assert!(loss < 0.001, "Expected loss < 0.001, got {}", loss);

        // Wrong predictions
        let predictions = vec![
            0.0, 0.0, 10.0, // Sample 0: predicts class 2
            10.0, 0.0, 0.0, // Sample 1: predicts class 0
        ];
        let targets = vec![0.0, 2.0];
        let loss = Metric::multi_class_log_loss(3).compute(&predictions, &targets);
        assert!(loss > 5.0);
    }

    #[test]
    fn test_accuracy() {
        // Perfect predictions
        let predictions = vec![10.0, 10.0, -10.0, -10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let acc = Metric::accuracy().compute(&predictions, &targets);
        assert!((acc - 1.0).abs() < 1e-6);

        // All wrong
        let predictions = vec![-10.0, -10.0, 10.0, 10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let acc = Metric::accuracy().compute(&predictions, &targets);
        assert!((acc - 0.0).abs() < 1e-6);

        // 50% correct
        let predictions = vec![10.0, -10.0, 10.0, -10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let acc = Metric::accuracy().compute(&predictions, &targets);
        assert!((acc - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<f32> = vec![];
        assert_eq!(Metric::Mse.compute(&empty, &empty), f32::INFINITY);
    }

    #[test]
    fn test_mismatched_lengths() {
        let predictions = vec![1.0, 2.0];
        let targets = vec![1.0];
        assert_eq!(Metric::Mse.compute(&predictions, &targets), f32::INFINITY);
    }

    #[test]
    fn test_metric_name() {
        assert_eq!(Metric::Mse.name(), "mse");
        assert_eq!(Metric::Rmse.name(), "rmse");
        assert_eq!(Metric::Mae.name(), "mae");
        assert_eq!(Metric::BinaryLogLoss.name(), "binary_log_loss");
        assert_eq!(
            Metric::multi_class_log_loss(3).name(),
            "multi_class_log_loss"
        );
        assert_eq!(Metric::accuracy().name(), "accuracy");
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
        // Test numerical stability with extreme values
        assert!(sigmoid(1000.0).is_finite());
        assert!(sigmoid(-1000.0).is_finite());
    }

    #[test]
    fn test_roc_auc_perfect() {
        // Perfect predictions: all positives ranked higher than negatives
        let predictions = vec![10.0, 10.0, -10.0, -10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let auc = compute_roc_auc(&predictions, &targets);
        assert!((auc - 1.0).abs() < 1e-6, "Expected AUC = 1.0, got {}", auc);
    }

    #[test]
    fn test_roc_auc_worst() {
        // Worst predictions: all negatives ranked higher than positives
        let predictions = vec![-10.0, -10.0, 10.0, 10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let auc = compute_roc_auc(&predictions, &targets);
        assert!((auc - 0.0).abs() < 1e-6, "Expected AUC = 0.0, got {}", auc);
    }

    #[test]
    fn test_roc_auc_random() {
        // Random ordering should give AUC around 0.5
        let predictions = vec![0.5, 0.3, 0.4, 0.6];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        let auc = compute_roc_auc(&predictions, &targets);
        // With this specific ordering, should be 0.5
        assert!((auc - 0.5).abs() < 0.1, "Expected AUC ~ 0.5, got {}", auc);
    }

    #[test]
    fn test_roc_auc_single_class() {
        // All positives or all negatives should return 0.5
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 1.0, 1.0];
        let auc = compute_roc_auc(&predictions, &targets);
        assert!(
            (auc - 0.5).abs() < 1e-6,
            "All-positive should give AUC = 0.5, got {}",
            auc
        );

        let targets = vec![0.0, 0.0, 0.0];
        let auc = compute_roc_auc(&predictions, &targets);
        assert!(
            (auc - 0.5).abs() < 1e-6,
            "All-negative should give AUC = 0.5, got {}",
            auc
        );
    }

    #[test]
    fn test_roc_auc_empty() {
        let empty: Vec<f32> = vec![];
        let auc = compute_roc_auc(&empty, &empty);
        assert!((auc - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_metric_roc_auc() {
        // Test via Metric enum
        let predictions = vec![10.0, 10.0, -10.0, -10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let auc = Metric::RocAuc.compute(&predictions, &targets);
        assert!((auc - 1.0).abs() < 1e-6, "Expected AUC = 1.0, got {}", auc);
        assert!(!Metric::RocAuc.lower_is_better());
        assert_eq!(Metric::RocAuc.name(), "roc_auc");
    }

    #[test]
    fn test_rank_ic_perfect() {
        // Perfect correlation: predictions and targets have same order
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ic = compute_rank_ic(&predictions, &targets, None);
        assert!(
            (ic - 1.0).abs() < 1e-6,
            "Expected Rank IC = 1.0, got {}",
            ic
        );
    }

    #[test]
    fn test_rank_ic_perfect_negative() {
        // Perfect negative correlation: opposite orders
        let predictions = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let targets = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ic = compute_rank_ic(&predictions, &targets, None);
        assert!(
            (ic - (-1.0)).abs() < 1e-6,
            "Expected Rank IC = -1.0, got {}",
            ic
        );
    }

    #[test]
    fn test_rank_ic_no_correlation() {
        // Specific case with zero correlation
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 3.0, 2.0, 4.0]; // Swapped middle two
        let ic = compute_rank_ic(&predictions, &targets, None);
        // This should give 0.8 (not exactly 0)
        assert!(ic.abs() < 1.0, "Rank IC should be in [-1, 1], got {}", ic);
    }

    #[test]
    fn test_rank_ic_with_ties() {
        // Predictions with ties
        let predictions = vec![1.0, 1.0, 3.0, 3.0];
        let targets = vec![10.0, 20.0, 30.0, 40.0];
        let ic = compute_rank_ic(&predictions, &targets, None);
        // Should still give positive correlation
        assert!(ic > 0.0, "Expected positive Rank IC, got {}", ic);
    }

    #[test]
    fn test_rank_ic_empty() {
        let empty: Vec<f32> = vec![];
        let ic = compute_rank_ic(&empty, &empty, None);
        assert!((ic - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rank_ic_single() {
        // Single element should return 0
        let predictions = vec![1.0];
        let targets = vec![1.0];
        let ic = compute_rank_ic(&predictions, &targets, None);
        assert!((ic - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rank_ic_constant_values() {
        // All same values should return 0 (no variation)
        let predictions = vec![1.0, 1.0, 1.0, 1.0];
        let targets = vec![10.0, 20.0, 30.0, 40.0];
        let ic = compute_rank_ic(&predictions, &targets, None);
        assert!((ic - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rank_ic_with_eras() {
        // Test proper era-based IC calculation
        // Era 0: perfect positive correlation (IC = 1.0)
        // Era 1: perfect negative correlation (IC = -1.0)
        // Average IC = 0.0
        let predictions = vec![1.0, 2.0, 3.0, 5.0, 4.0, 3.0];
        let targets = vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0];
        let era_indices = vec![0, 0, 0, 1, 1, 1];

        let ic = compute_rank_ic(&predictions, &targets, Some(&era_indices));
        assert!(
            (ic - 0.0).abs() < 1e-6,
            "Expected average IC = 0.0, got {}",
            ic
        );
    }

    #[test]
    fn test_rank_ic_with_eras_positive() {
        // Era 0: IC ~ 1.0
        // Era 1: IC ~ 1.0
        // Average IC = 1.0
        let predictions = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let targets = vec![10.0, 20.0, 30.0, 5.0, 10.0, 15.0];
        let era_indices = vec![0, 0, 0, 1, 1, 1];

        let ic = compute_rank_ic(&predictions, &targets, Some(&era_indices));
        assert!(
            (ic - 1.0).abs() < 1e-6,
            "Expected average IC = 1.0, got {}",
            ic
        );
    }
}
