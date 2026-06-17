//! Quantile Loss (Pinball Loss) for quantile regression
//!
//! Estimates conditional quantiles rather than the mean.

use super::{activation::sigmoid, LossFunction};

/// Quantile Loss (Pinball Loss)
///
/// L(y, ŷ) = τ(y - ŷ)  if y ≥ ŷ
///         = (1-τ)(ŷ - y)  if y < ŷ
///
/// Properties:
/// - τ = 0.5 gives median regression (equivalent to MAE)
/// - τ = 0.1 predicts the 10th percentile
/// - τ = 0.9 predicts the 90th percentile
/// - Useful for prediction intervals and risk assessment
///
/// # Smoothed Approximation
///
/// Since quantile loss is not twice differentiable at y = ŷ,
/// we use a smooth approximation with a small delta parameter:
///
/// gradient ≈ τ - sigmoid((y - ŷ) / delta)
/// hessian ≈ sigmoid'((y - ŷ) / delta) / delta
#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    /// Quantile to predict (0 < τ < 1)
    tau: f32,
    /// Smoothing parameter for gradient (smaller = sharper transition)
    delta: f32,
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self::median()
    }
}

impl QuantileLoss {
    /// Create a quantile loss for the given quantile τ
    ///
    /// # Arguments
    /// * `tau` - Quantile to predict (must be in (0, 1))
    ///
    /// # Returns
    /// * `Result<Self>` - Returns error if tau is not in (0, 1)
    ///
    /// # Errors
    /// * Returns error if `tau <= 0.0` or `tau >= 1.0`
    pub fn new(tau: f32) -> crate::Result<Self> {
        if tau <= 0.0 || tau >= 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "tau must be in (0, 1), got {}",
                tau
            )));
        }
        Ok(Self { tau, delta: 0.01 })
    }

    /// Create a quantile loss with custom smoothing
    ///
    /// # Arguments
    /// * `tau` - Quantile to predict (must be in (0, 1))
    /// * `delta` - Smoothing parameter (smaller = sharper, typical: 0.001 to 0.1)
    ///
    /// # Returns
    /// * `Result<Self>` - Returns error if validation fails
    ///
    /// # Errors
    /// * Returns error if `tau <= 0.0` or `tau >= 1.0`
    /// * Returns error if `delta <= 0.0`
    pub fn with_delta(tau: f32, delta: f32) -> crate::Result<Self> {
        if tau <= 0.0 || tau >= 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "tau must be in (0, 1), got {}",
                tau
            )));
        }
        if delta <= 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "delta must be positive, got {}",
                delta
            )));
        }
        Ok(Self { tau, delta })
    }

    /// Median regression (τ = 0.5)
    ///
    /// This is infallible since 0.5 is always valid.
    pub fn median() -> Self {
        // SAFETY: 0.5 is always in (0, 1)
        Self::new(0.5).unwrap()
    }

    /// Lower quantile for prediction intervals (e.g., 10th percentile)
    ///
    /// # Errors
    /// * Returns error if `tau <= 0.0`, `tau >= 0.5`, or `tau >= 1.0`
    pub fn lower(tau: f32) -> crate::Result<Self> {
        if tau >= 0.5 {
            return Err(crate::TreeBoostError::Config(format!(
                "lower quantile should be < 0.5, got {}",
                tau
            )));
        }
        Self::new(tau)
    }

    /// Upper quantile for prediction intervals (e.g., 90th percentile)
    ///
    /// # Errors
    /// * Returns error if `tau <= 0.5`, `tau >= 1.0`, or `tau <= 0.0`
    pub fn upper(tau: f32) -> crate::Result<Self> {
        if tau <= 0.5 {
            return Err(crate::TreeBoostError::Config(format!(
                "upper quantile should be > 0.5, got {}",
                tau
            )));
        }
        Self::new(tau)
    }

    /// Get the quantile parameter
    pub fn tau(&self) -> f32 {
        self.tau
    }
}

impl LossFunction for QuantileLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let residual = target - prediction;
        if residual >= 0.0 {
            self.tau * residual
        } else {
            (self.tau - 1.0) * residual
        }
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        // Smoothed gradient: τ - sigmoid((y - ŷ) / δ)
        // This approximates the subgradient which is:
        //   -τ if y > ŷ
        //   (1-τ) if y < ŷ
        // But we need gradient w.r.t. prediction, so signs flip
        let residual = target - prediction;
        let sig = sigmoid(residual / self.delta);
        self.tau - sig
    }

    #[inline]
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        // Smoothed hessian: sigmoid'(x) / δ = sigmoid(x)(1 - sigmoid(x)) / δ
        let residual = target - prediction;
        let sig = sigmoid(residual / self.delta);
        // Ensure positive hessian (sig*(1-sig) can be very small for large residuals)
        (sig * (1.0 - sig) / self.delta).max(1e-6)
    }

    #[inline]
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        let residual = target - prediction;
        let sig = sigmoid(residual / self.delta);
        let gradient = self.tau - sig;
        let hessian = (sig * (1.0 - sig) / self.delta).max(1e-6);
        (gradient, hessian)
    }

    fn initial_prediction(&self, targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        // For quantile regression, initial prediction should be the sample quantile
        let mut sorted: Vec<f32> = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((targets.len() as f32 - 1.0) * self.tau).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn name(&self) -> &'static str {
        "quantile"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_loss_median() {
        let loss = QuantileLoss::median();

        // Median loss is symmetric (half of absolute error)
        let l1 = loss.loss(10.0, 12.0); // under-predict
        let l2 = loss.loss(10.0, 8.0); // over-predict

        // For median (τ=0.5), loss is symmetric: 0.5 * |error|
        assert!((l1 - 1.0).abs() < 1e-6); // 0.5 * 2 = 1
        assert!((l2 - 1.0).abs() < 1e-6); // 0.5 * 2 = 1
    }

    #[test]
    fn test_quantile_loss_asymmetric() {
        let loss = QuantileLoss::new(0.9).unwrap();

        // τ = 0.9: heavily penalizes under-prediction
        let l_under = loss.loss(10.0, 8.0); // under-predict by 2
        let l_over = loss.loss(10.0, 12.0); // over-predict by 2

        // Under-prediction: τ * error = 0.9 * 2 = 1.8
        assert!((l_under - 1.8).abs() < 1e-6);
        // Over-prediction: (1-τ) * error = 0.1 * 2 = 0.2
        assert!((l_over - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_gradient_direction() {
        let loss = QuantileLoss::median();

        // When prediction > target, gradient should be positive (reduce prediction)
        let g_over = loss.gradient(10.0, 15.0);
        assert!(g_over > 0.0);

        // When prediction < target, gradient should be negative (increase prediction)
        let g_under = loss.gradient(10.0, 5.0);
        assert!(g_under < 0.0);
    }

    #[test]
    fn test_quantile_hessian_positive() {
        let loss = QuantileLoss::new(0.75).unwrap();

        // Hessian should always be positive
        assert!(loss.hessian(10.0, 10.0) > 0.0);
        assert!(loss.hessian(10.0, 15.0) > 0.0);
        assert!(loss.hessian(10.0, 5.0) > 0.0);
    }

    #[test]
    fn test_initial_prediction() {
        let loss = QuantileLoss::median();
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Median of [1,2,3,4,5] is 3
        let init = loss.initial_prediction(&targets);
        assert!((init - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_initial_prediction_quantile() {
        let loss = QuantileLoss::new(0.25).unwrap();
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // 25th percentile should be close to 2
        let init = loss.initial_prediction(&targets);
        assert!((1.0..=3.0).contains(&init));
    }
}
