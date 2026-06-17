//! Multi-Label Log Loss for multi-output classification
//!
//! Each output dimension is treated as an independent binary classification problem.
//! Uses sigmoid activation per label (NOT softmax which would impose mutual exclusivity).
//!
//! L(y, ŷ) = -Σ_k [y_k * log(p_k) + (1-y_k) * log(1-p_k)]
//!
//! where p_k = sigmoid(ŷ_k) for each label k.

use super::activation::sigmoid;
use super::LossFunction;

/// Multi-Label Log Loss (Binary Cross-Entropy per label)
///
/// For multi-label classification where each sample can belong to multiple classes
/// simultaneously. Each label is treated as an independent binary classification.
///
/// Properties:
/// - Gradient per label: g_k = sigmoid(pred_k) - target_k
/// - Hessian per label: h_k = sigmoid(pred_k) * (1 - sigmoid(pred_k))
/// - Initial prediction per label: log(mean_k / (1 - mean_k)) (log-odds)
///
/// # Data Layout
///
/// Targets and predictions are stored row-wise flattened:
/// `[row0_label0, row0_label1, ..., row0_labelK, row1_label0, ...]`
///
/// # Example
///
/// ```ignore
/// use treeboost::loss::MultiLabelLogLoss;
///
/// let loss = MultiLabelLogLoss::new();
/// let num_outputs = 3;
///
/// // 2 samples × 3 labels
/// let targets = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
/// let predictions = vec![0.5, -0.5, 0.0, -1.0, 1.0, 0.5];
///
/// let mut grads = vec![0.0; 6];
/// let mut hess = vec![0.0; 6];
/// loss.compute_gradients_multi(&targets, &predictions, &mut grads, &mut hess, num_outputs);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MultiLabelLogLoss {
    /// Small epsilon for numerical stability
    eps: f32,
}

impl MultiLabelLogLoss {
    /// Create a new Multi-Label Log Loss
    pub fn new() -> Self {
        Self { eps: 1e-7 }
    }

    /// Create with custom epsilon for numerical stability
    pub fn with_eps(eps: f32) -> Self {
        Self { eps }
    }

    /// Compute loss for a single target-prediction pair
    #[inline]
    pub fn loss_single(&self, target: f32, prediction: f32) -> f32 {
        let p = sigmoid(prediction).clamp(self.eps, 1.0 - self.eps);
        -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
    }

    /// Compute gradient and hessian for a single target-prediction pair
    ///
    /// Returns (gradient, hessian) where:
    /// - gradient = sigmoid(prediction) - target
    /// - hessian = sigmoid(prediction) * (1 - sigmoid(prediction))
    #[inline]
    pub fn gradient_hessian_single(&self, target: f32, prediction: f32) -> (f32, f32) {
        let p = sigmoid(prediction);
        let g = p - target;
        let h = (p * (1.0 - p)).max(self.eps);
        (g, h)
    }

    /// Compute gradients and hessians for multi-output data
    ///
    /// All arrays are row-wise flattened: `[row0_out0, row0_out1, ..., row1_out0, ...]`
    ///
    /// # Arguments
    ///
    /// * `targets` - Flattened targets, length = num_rows * num_outputs
    /// * `predictions` - Flattened predictions (raw logits), same length
    /// * `gradients` - Output gradients, same length
    /// * `hessians` - Output hessians, same length
    /// * `num_outputs` - Number of output labels
    pub fn compute_gradients_multi(
        &self,
        targets: &[f32],
        predictions: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
        num_outputs: usize,
    ) {
        debug_assert_eq!(targets.len(), predictions.len());
        debug_assert_eq!(targets.len(), gradients.len());
        debug_assert_eq!(targets.len(), hessians.len());
        debug_assert!(num_outputs > 0);
        debug_assert_eq!(targets.len() % num_outputs, 0);

        let num_rows = targets.len() / num_outputs;

        // Process in blocks for cache efficiency (8x unrolled)
        const BLOCK_SIZE: usize = 8;
        let full_blocks = num_rows / BLOCK_SIZE;

        for block in 0..full_blocks {
            let base = block * BLOCK_SIZE * num_outputs;

            for output_idx in 0..num_outputs {
                // Unroll 8 rows
                for row_offset in 0..BLOCK_SIZE {
                    let idx = base + row_offset * num_outputs + output_idx;
                    let p = sigmoid(predictions[idx]);
                    gradients[idx] = p - targets[idx];
                    hessians[idx] = (p * (1.0 - p)).max(self.eps);
                }
            }
        }

        // Handle remaining rows
        let remaining_start = full_blocks * BLOCK_SIZE;
        for row in remaining_start..num_rows {
            for output_idx in 0..num_outputs {
                let idx = row * num_outputs + output_idx;
                let p = sigmoid(predictions[idx]);
                gradients[idx] = p - targets[idx];
                hessians[idx] = (p * (1.0 - p)).max(self.eps);
            }
        }
    }

    /// Compute initial predictions (base scores) for each output label
    ///
    /// Returns log-odds for each label: log(p_k / (1 - p_k))
    /// where p_k is the proportion of positive samples for label k.
    ///
    /// # Arguments
    ///
    /// * `targets` - Flattened targets, length = num_rows * num_outputs
    /// * `num_outputs` - Number of output labels
    ///
    /// # Returns
    ///
    /// Vector of length `num_outputs` with initial prediction per label
    pub fn initial_predictions(&self, targets: &[f32], num_outputs: usize) -> Vec<f32> {
        if targets.is_empty() || num_outputs == 0 {
            return vec![0.0; num_outputs];
        }

        let num_rows = targets.len() / num_outputs;
        let mut initial = vec![0.0; num_outputs];

        for (output_idx, init) in initial.iter_mut().enumerate() {
            // Count positives for this label
            let mut positive_count = 0.0;
            for row in 0..num_rows {
                let idx = row * num_outputs + output_idx;
                if targets[idx] > 0.5 {
                    positive_count += 1.0;
                }
            }

            // Compute log-odds
            let p = (positive_count / num_rows as f32).clamp(self.eps, 1.0 - self.eps);
            *init = (p / (1.0 - p)).ln();
        }

        initial
    }

    /// Compute total loss over all samples and labels
    ///
    /// Returns sum of binary cross-entropy over all (sample, label) pairs.
    pub fn compute_loss(&self, targets: &[f32], predictions: &[f32]) -> f32 {
        debug_assert_eq!(targets.len(), predictions.len());

        let mut total_loss = 0.0;
        for i in 0..targets.len() {
            total_loss += self.loss_single(targets[i], predictions[i]);
        }
        total_loss
    }

    /// Compute average loss per sample (mean over samples and labels)
    pub fn compute_mean_loss(
        &self,
        targets: &[f32],
        predictions: &[f32],
        num_outputs: usize,
    ) -> f32 {
        let total = self.compute_loss(targets, predictions);
        let num_rows = targets.len() / num_outputs;
        total / (num_rows * num_outputs) as f32
    }

    /// Convert raw predictions to probabilities via sigmoid
    ///
    /// # Arguments
    ///
    /// * `predictions` - Flattened raw predictions (logits)
    /// * `num_outputs` - Number of output labels
    ///
    /// # Returns
    ///
    /// Flattened probabilities with same layout as input
    pub fn to_probabilities(&self, predictions: &[f32], _num_outputs: usize) -> Vec<f32> {
        predictions.iter().map(|&p| sigmoid(p)).collect()
    }

    /// Convert probabilities to binary predictions (0 or 1) using threshold
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Flattened probabilities
    /// * `threshold` - Classification threshold (default 0.5)
    ///
    /// # Returns
    ///
    /// Flattened binary predictions (0.0 or 1.0)
    pub fn to_labels(&self, probabilities: &[f32], threshold: f32) -> Vec<f32> {
        probabilities
            .iter()
            .map(|&p| if p >= threshold { 1.0 } else { 0.0 })
            .collect()
    }

    /// Convert probabilities to binary predictions using per-label thresholds
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Flattened probabilities
    /// * `thresholds` - Threshold per label
    /// * `num_outputs` - Number of output labels
    ///
    /// # Returns
    ///
    /// Flattened binary predictions (0.0 or 1.0)
    pub fn to_labels_multi_threshold(
        &self,
        probabilities: &[f32],
        thresholds: &[f32],
        num_outputs: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(thresholds.len(), num_outputs);

        let num_rows = probabilities.len() / num_outputs;
        let mut labels = vec![0.0; probabilities.len()];

        for row in 0..num_rows {
            for (output_idx, &threshold) in thresholds.iter().enumerate() {
                let idx = row * num_outputs + output_idx;
                labels[idx] = if probabilities[idx] >= threshold {
                    1.0
                } else {
                    0.0
                };
            }
        }

        labels
    }

    /// Get the name of this loss function (internal method)
    fn name_internal(&self) -> &'static str {
        "multilabel_logloss"
    }
}

/// Implementation of LossFunction trait for integration with GBDT training
impl LossFunction for MultiLabelLogLoss {
    /// Compute loss for a single (target, prediction) pair
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        self.loss_single(target, prediction)
    }

    /// Compute gradient for a single (target, prediction) pair
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        sigmoid(prediction) - target
    }

    /// Compute hessian for a single (target, prediction) pair
    fn hessian(&self, _target: f32, prediction: f32) -> f32 {
        let p = sigmoid(prediction);
        (p * (1.0 - p)).max(self.eps)
    }

    /// Compute gradient and hessian together
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        self.gradient_hessian_single(target, prediction)
    }

    /// Compute gradients for multi-output data (optimized path)
    fn compute_gradients_multi(
        &self,
        targets: &[f32],
        predictions: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
        num_outputs: usize,
    ) {
        // Use the optimized internal method
        MultiLabelLogLoss::compute_gradients_multi(
            self,
            targets,
            predictions,
            gradients,
            hessians,
            num_outputs,
        );
    }

    /// Initial predictions for multi-output data
    fn initial_predictions_multi(&self, targets: &[f32], num_outputs: usize) -> Vec<f32> {
        self.initial_predictions(targets, num_outputs)
    }

    /// Multi-label loss supports any number of outputs
    fn num_outputs(&self) -> Option<usize> {
        None // Any number of outputs
    }

    /// Multi-label loss supports multi-output
    fn supports_multi_output(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        self.name_internal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_hessian_at_zero() {
        let loss = MultiLabelLogLoss::new();

        // At pred=0, sigmoid=0.5
        let (g, h) = loss.gradient_hessian_single(1.0, 0.0);
        assert!((g - (-0.5)).abs() < 1e-6);
        assert!((h - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_direction() {
        let loss = MultiLabelLogLoss::new();

        // When prediction is too low for positive target, gradient should be negative
        let (g_pos, _) = loss.gradient_hessian_single(1.0, -2.0);
        assert!(
            g_pos < 0.0,
            "Gradient should be negative for target=1, low pred"
        );

        // When prediction is too high for negative target, gradient should be positive
        let (g_neg, _) = loss.gradient_hessian_single(0.0, 2.0);
        assert!(
            g_neg > 0.0,
            "Gradient should be positive for target=0, high pred"
        );
    }

    #[test]
    fn test_hessian_always_positive() {
        let loss = MultiLabelLogLoss::new();

        for pred in [-100.0, -10.0, 0.0, 10.0, 100.0] {
            let (_, h) = loss.gradient_hessian_single(0.0, pred);
            assert!(
                h > 0.0,
                "Hessian should always be positive, got {} for pred={}",
                h,
                pred
            );
        }
    }

    #[test]
    fn test_loss_minimized_at_correct_prediction() {
        let loss = MultiLabelLogLoss::new();

        // Target=1, high positive prediction -> low loss
        let l_correct = loss.loss_single(1.0, 10.0);
        // Target=1, prediction=0 -> higher loss
        let l_wrong = loss.loss_single(1.0, 0.0);

        assert!(
            l_correct < l_wrong,
            "Loss should be lower for correct prediction"
        );
    }

    #[test]
    fn test_initial_predictions_balanced() {
        let loss = MultiLabelLogLoss::new();

        // Balanced: 50% positive
        let targets = vec![1.0, 0.0, 1.0, 0.0]; // 2 labels, 2 rows each with [1,0]
        let initial = loss.initial_predictions(&targets, 2);

        // Label 0 is always 1 -> log(1/0) but clamped -> large positive
        // Actually: row0=[1,0], row1=[1,0]
        // Label 0: 2/2 = 100% -> log(0.9999/0.0001) ≈ 9.2
        // Label 1: 0/2 = 0% -> log(0.0001/0.9999) ≈ -9.2
        assert!(initial[0] > 5.0, "init[0] = {}", initial[0]);
        assert!(initial[1] < -5.0, "init[1] = {}", initial[1]);
    }
}
