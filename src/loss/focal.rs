//! Multi-Label Focal Loss for imbalanced multi-output classification
//!
//! Focal Loss addresses class imbalance by down-weighting easy examples and
//! focusing training on hard examples. Each output dimension is treated as
//! an independent binary classification problem with sigmoid activation.
//!
//! FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
//!
//! where:
//! - p_t = p for positive class (y=1), p_t = 1-p for negative class (y=0)
//! - α is a balancing factor (often set to inverse class frequency)
//! - γ is the focusing parameter (typically 2)
//!
//! Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

use super::activation::sigmoid;
use super::LossFunction;

/// Multi-Label Focal Loss
///
/// For imbalanced multi-label classification where some labels are rare.
/// Down-weights easy examples to focus learning on hard examples.
///
/// Properties:
/// - γ=0 reduces to standard log loss
/// - γ>0 reduces gradient magnitude for well-classified examples
/// - Higher γ means more aggressive down-weighting of easy examples
/// - α balances positive/negative classes
///
/// # Example
///
/// ```ignore
/// use treeboost::loss::MultiLabelFocalLoss;
///
/// // gamma=2 is a common choice
/// let focal = MultiLabelFocalLoss::new(2.0);
///
/// // For imbalanced data, set alpha based on class frequency
/// let focal_balanced = MultiLabelFocalLoss::new(2.0).with_alpha(0.75);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MultiLabelFocalLoss {
    /// Focusing parameter (γ)
    /// - γ=0: equivalent to standard log loss
    /// - γ=2: typical choice, good balance
    /// - γ>2: more aggressive focusing on hard examples
    gamma: f32,

    /// Class balancing factor (α)
    /// - α=0.5: balanced (default)
    /// - α>0.5: up-weight positive class
    /// - α<0.5: up-weight negative class
    alpha: f32,

    /// Small epsilon for numerical stability
    eps: f32,
}

impl Default for MultiLabelFocalLoss {
    fn default() -> Self {
        Self {
            gamma: 2.0,
            alpha: 0.5,
            eps: 1e-7,
        }
    }
}

impl MultiLabelFocalLoss {
    /// Create a new Multi-Label Focal Loss with specified gamma
    ///
    /// # Arguments
    ///
    /// * `gamma` - Focusing parameter (typical values: 0-5)
    pub fn new(gamma: f32) -> Self {
        Self {
            gamma,
            alpha: 0.5,
            eps: 1e-7,
        }
    }

    /// Set the alpha balancing factor
    ///
    /// # Arguments
    ///
    /// * `alpha` - Weight for positive class (0-1)
    ///   - alpha=0.75 means positives get 3x weight vs negatives
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set custom epsilon for numerical stability
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Compute the focal weight factor: (1 - p_t)^gamma
    ///
    /// p_t is the probability of the true class:
    /// - For y=1: p_t = p
    /// - For y=0: p_t = 1-p
    #[inline]
    fn focal_weight(&self, p: f32, target: f32) -> f32 {
        let p_t = if target > 0.5 { p } else { 1.0 - p };
        (1.0 - p_t).powf(self.gamma)
    }

    /// Compute the alpha weight based on target
    #[inline]
    fn alpha_weight(&self, target: f32) -> f32 {
        if target > 0.5 {
            self.alpha
        } else {
            1.0 - self.alpha
        }
    }

    /// Compute focal loss for a single target-prediction pair
    ///
    /// FL = -α_t * (1 - p_t)^γ * log(p_t)
    #[inline]
    pub fn loss_single(&self, target: f32, prediction: f32) -> f32 {
        let p = sigmoid(prediction).clamp(self.eps, 1.0 - self.eps);
        let p_t = if target > 0.5 { p } else { 1.0 - p };

        let alpha_t = self.alpha_weight(target);
        let focal_weight = (1.0 - p_t).powf(self.gamma);

        -alpha_t * focal_weight * p_t.ln()
    }

    /// Compute gradient and hessian for a single target-prediction pair
    ///
    /// The gradient of focal loss is:
    /// g = α_t * (1 - p_t)^γ * (p - y)
    ///
    /// This is the standard log loss gradient scaled by the focal weight.
    /// The full derivative includes additional terms from differentiating
    /// the focal weight, but this simplified form works well in practice.
    #[inline]
    pub fn gradient_hessian_single(&self, target: f32, prediction: f32) -> (f32, f32) {
        let p = sigmoid(prediction);

        // Focal weight: (1 - p_t)^gamma
        let focal_weight = self.focal_weight(p, target);
        let alpha_t = self.alpha_weight(target);

        // Combined weight
        let weight = alpha_t * focal_weight;

        // Gradient: weight * (p - target)
        // This is the standard gradient scaled by focal weight
        let g = weight * (p - target);

        // Hessian: weight * p * (1 - p)
        // Approximation that works well for Newton's method
        let h = (weight * p * (1.0 - p)).max(self.eps);

        (g, h)
    }

    /// Compute gradients and hessians for multi-output data
    ///
    /// All arrays are row-wise flattened: `[row0_out0, row0_out1, ..., row1_out0, ...]`
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
                for row_offset in 0..BLOCK_SIZE {
                    let idx = base + row_offset * num_outputs + output_idx;
                    let (g, h) = self.gradient_hessian_single(targets[idx], predictions[idx]);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }
            }
        }

        // Handle remaining rows
        let remaining_start = full_blocks * BLOCK_SIZE;
        for row in remaining_start..num_rows {
            for output_idx in 0..num_outputs {
                let idx = row * num_outputs + output_idx;
                let (g, h) = self.gradient_hessian_single(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }
        }
    }

    /// Compute initial predictions (base scores) for each output label
    ///
    /// Same as log loss: log-odds for each label
    pub fn initial_predictions(&self, targets: &[f32], num_outputs: usize) -> Vec<f32> {
        if targets.is_empty() || num_outputs == 0 {
            return vec![0.0; num_outputs];
        }

        let num_rows = targets.len() / num_outputs;
        let mut initial = vec![0.0; num_outputs];

        for (output_idx, init) in initial.iter_mut().enumerate() {
            let mut positive_count = 0.0;
            for row in 0..num_rows {
                let idx = row * num_outputs + output_idx;
                if targets[idx] > 0.5 {
                    positive_count += 1.0;
                }
            }

            let p = (positive_count / num_rows as f32).clamp(self.eps, 1.0 - self.eps);
            *init = (p / (1.0 - p)).ln();
        }

        initial
    }

    /// Compute total focal loss over all samples and labels
    pub fn compute_loss(&self, targets: &[f32], predictions: &[f32]) -> f32 {
        debug_assert_eq!(targets.len(), predictions.len());

        let mut total_loss = 0.0;
        for i in 0..targets.len() {
            total_loss += self.loss_single(targets[i], predictions[i]);
        }
        total_loss
    }

    /// Convert raw predictions to probabilities via sigmoid
    pub fn to_probabilities(&self, predictions: &[f32]) -> Vec<f32> {
        predictions.iter().map(|&p| sigmoid(p)).collect()
    }

    /// Get the name of this loss function (internal method)
    fn name_internal(&self) -> &'static str {
        "multilabel_focal_loss"
    }

    /// Get the gamma (focusing) parameter
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Get the alpha (balancing) parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

/// Implementation of LossFunction trait for integration with GBDT training
impl LossFunction for MultiLabelFocalLoss {
    /// Compute focal loss for a single (target, prediction) pair
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        self.loss_single(target, prediction)
    }

    /// Compute gradient for a single (target, prediction) pair
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        self.gradient_hessian_single(target, prediction).0
    }

    /// Compute hessian for a single (target, prediction) pair
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        self.gradient_hessian_single(target, prediction).1
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
        MultiLabelFocalLoss::compute_gradients_multi(
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

    /// Multi-label focal loss supports any number of outputs
    fn num_outputs(&self) -> Option<usize> {
        None // Any number of outputs
    }

    /// Multi-label focal loss supports multi-output
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
    fn test_focal_weight_calculation() {
        let focal = MultiLabelFocalLoss::new(2.0);

        // For target=1 and p=0.9 (easy positive): focal_weight = (1-0.9)^2 = 0.01
        let p = 0.9;
        let weight = focal.focal_weight(p, 1.0);
        assert!((weight - 0.01).abs() < 1e-6, "weight = {}", weight);

        // For target=1 and p=0.1 (hard positive): focal_weight = (1-0.1)^2 = 0.81
        let p = 0.1;
        let weight = focal.focal_weight(p, 1.0);
        assert!((weight - 0.81).abs() < 1e-6, "weight = {}", weight);
    }

    #[test]
    fn test_alpha_weight() {
        let focal = MultiLabelFocalLoss::new(2.0).with_alpha(0.75);

        assert!((focal.alpha_weight(1.0) - 0.75).abs() < 1e-6);
        assert!((focal.alpha_weight(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_zero_matches_logloss() {
        use super::super::multilabel::MultiLabelLogLoss;

        let focal = MultiLabelFocalLoss::new(0.0);
        let logloss = MultiLabelLogLoss::new();

        let test_cases = [(1.0, 0.0), (0.0, 2.0), (1.0, -1.0)];

        for (target, pred) in test_cases {
            let (g_focal, _) = focal.gradient_hessian_single(target, pred);
            let (g_logloss, _) = logloss.gradient_hessian_single(target, pred);

            // With alpha=0.5 and gamma=0, focal gradient = 0.5 * logloss gradient
            assert!(
                (g_focal - 0.5 * g_logloss).abs() < 1e-5,
                "target={}, pred={}: focal={}, logloss={}",
                target,
                pred,
                g_focal,
                g_logloss
            );
        }
    }

    #[test]
    fn test_focusing_behavior() {
        let focal = MultiLabelFocalLoss::new(2.0);

        // Easy positive (high p for target=1)
        let (g_easy, _) = focal.gradient_hessian_single(1.0, 5.0);

        // Hard positive (low p for target=1)
        let (g_hard, _) = focal.gradient_hessian_single(1.0, -1.0);

        // Easy example should have smaller gradient magnitude
        // (Both are negative since p < target for hard case is more wrong)
        // Actually for easy case: p ≈ 1, g ≈ 0 (close to target)
        // For hard case: p ≈ 0.27, g = -0.73 * focal_weight
        assert!(
            g_easy.abs() < g_hard.abs(),
            "Easy gradient ({}) should be smaller than hard ({})",
            g_easy.abs(),
            g_hard.abs()
        );
    }
}
