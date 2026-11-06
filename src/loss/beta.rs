//! Beta loss for bounded continuous regression
//!
//! Implements negative log-likelihood of the Beta distribution for targets in (0, 1).
//! Beta distribution is natural for modeling proportions, probabilities, and bounded
//! continuous targets.
//!
//! # Mathematical Background
//!
//! The Beta distribution with parameters α, β is:
//! ```text
//! Beta(y; α, β) = Γ(α+β)/(Γ(α)Γ(β)) * y^(α-1) * (1-y)^(β-1)
//! ```
//!
//! For regression, we parameterize using:
//! - μ (mean): μ = α/(α+β) ∈ (0, 1)  [predicted by model via sigmoid]
//! - φ (precision): φ = α + β > 0      [fixed hyperparameter]
//!
//! Then: α = μφ, β = (1-μ)φ
//!
//! The loss is the negative log-likelihood:
//! ```text
//! L = -log Beta(y; μφ, (1-μ)φ)
//!   = log Γ(μφ) + log Γ((1-μ)φ) - log Γ(φ) - (μφ-1)log(y) - ((1-μ)φ-1)log(1-y)
//! ```
//!
//! # When to Use Beta Loss
//!
//! **Use Beta Loss when:**
//! - Targets are naturally bounded in (0, 1) (e.g., probabilities, percentages)
//! - You want probabilistic predictions with uncertainty
//! - Target distribution is skewed or U-shaped (Beta handles these well)
//!
//! **DON'T use Beta Loss when:**
//! - Targets can be exactly 0 or 1 (Beta is undefined at boundaries)
//! - For those cases, use LogitTransform + MSE or Focal Loss
//!
//! **Industry Standard Approaches for Bounded Targets:**
//! 1. **LogitTransform + MSE** (most common, what we already have)
//!    - Transform [0,1] → (-∞,∞) via logit, train with MSE, inverse via sigmoid
//!    - Fast, simple, works for exact 0/1 values
//! 2. **Beta Loss** (this implementation)
//!    - Native probabilistic modeling for (0,1) targets
//!    - Requires strict (0,1) bounds (epsilon clipping)
//! 3. **Focal Loss** (for imbalanced proportions)
//!    - Downweights easy examples, focuses on hard ones
//!
//! # Example
//!
//! ```ignore
//! use treeboost::loss::BetaLoss;
//! use treeboost::model::UniversalConfig;
//!
//! // For probability regression (e.g., churn probability)
//! let loss = BetaLoss::new(10.0);  // φ = 10 (moderate precision)
//!
//! let config = UniversalConfig::default()
//!     .with_beta_loss(loss);  // Not yet implemented - use custom loss
//! ```
//!
//! # Precision Parameter (φ)
//!
//! - φ = 1: Uniform prior (maximum uncertainty)
//! - φ = 2-5: Weak prior (flexible, good default)
//! - φ = 10-50: Moderate prior (encouraged toward extremes)
//! - φ = 100+: Strong prior (sharp peaks, low variance)
//!
//! For most applications, φ = 5-20 is reasonable.

use crate::loss::sigmoid;
use crate::{Result, TreeBoostError};

/// Beta loss for bounded continuous regression
///
/// Models targets in (0, 1) using Beta distribution with fixed precision.
#[derive(Debug, Clone)]
pub struct BetaLoss {
    /// Precision parameter φ = α + β
    ///
    /// Controls the "sharpness" of the distribution:
    /// - Small φ (1-5): Flexible, high variance
    /// - Medium φ (10-50): Balanced
    /// - Large φ (100+): Sharp, low variance
    phi: f32,

    /// Epsilon for numerical stability
    ///
    /// Targets and predictions are clamped to [epsilon, 1-epsilon]
    /// to avoid log(0) and division by zero.
    epsilon: f32,
}

impl BetaLoss {
    /// Create a new BetaLoss with specified precision
    ///
    /// # Arguments
    ///
    /// * `phi` - Precision parameter φ = α + β (must be > 0)
    ///
    /// # Recommended Values
    ///
    /// - `phi = 5.0`: Flexible, good starting point
    /// - `phi = 10.0`: Moderate precision, balanced
    /// - `phi = 20.0`: Higher precision, sharper predictions
    ///
    /// # Example
    ///
    /// ```
    /// use treeboost::loss::BetaLoss;
    ///
    /// let loss = BetaLoss::new(10.0).unwrap();  // Moderate precision
    /// ```
    pub fn new(phi: f32) -> Result<Self> {
        if phi <= 0.0 {
            return Err(TreeBoostError::Config(format!(
                "BetaLoss phi must be > 0, got {}",
                phi
            )));
        }

        Ok(Self { phi, epsilon: 1e-7 })
    }

    /// Create BetaLoss with custom epsilon for numerical stability
    ///
    /// # Arguments
    ///
    /// * `phi` - Precision parameter (must be > 0)
    /// * `epsilon` - Clipping threshold for [epsilon, 1-epsilon] (must be in (0, 0.5))
    pub fn with_epsilon(mut self, epsilon: f32) -> Result<Self> {
        if epsilon <= 0.0 || epsilon >= 0.5 {
            return Err(TreeBoostError::Config(format!(
                "BetaLoss epsilon must be in (0, 0.5), got {}",
                epsilon
            )));
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Digamma function ψ(x) = d/dx log Γ(x)
    ///
    /// Uses series approximation for x > 6, iterative recurrence for smaller x.
    ///
    /// Accuracy: ~1e-6 for x > 0.5
    #[inline]
    fn digamma(x: f32) -> f32 {
        // For large x, use asymptotic expansion
        if x > 6.0 {
            let x_inv = 1.0 / x;
            let x_inv2 = x_inv * x_inv;
            return x.ln() - 0.5 * x_inv - x_inv2 / 12.0 + x_inv2 * x_inv2 / 120.0;
        }

        // For x in (0, 6], use iterative recurrence to shift to x > 6
        // ψ(x+1) = ψ(x) + 1/x, so ψ(x) = ψ(x+n) - sum(1/(x+k) for k=0..n-1)
        let mut y = x;
        let mut sum = 0.0;

        // Shift upward to y > 6 using recurrence
        while y < 6.0 {
            sum -= 1.0 / y;
            y += 1.0;
        }

        // Now y > 6, compute ψ(y) using asymptotic expansion
        let y_inv = 1.0 / y;
        let y_inv2 = y_inv * y_inv;
        let psi_y = y.ln() - 0.5 * y_inv - y_inv2 / 12.0 + y_inv2 * y_inv2 / 120.0;

        // Return ψ(x) = ψ(y) + sum
        psi_y + sum
    }

    /// Compute negative log-likelihood for a single target-prediction pair
    ///
    /// # Arguments
    ///
    /// * `y` - True target in (0, 1)
    /// * `mu` - Predicted mean in (0, 1) [sigmoid of raw prediction]
    ///
    /// # Returns
    ///
    /// Negative log-likelihood: -log Beta(y; μφ, (1-μ)φ)
    #[inline]
    fn neg_log_likelihood(&self, y: f32, mu: f32) -> f32 {
        // Clamp to avoid log(0)
        let y_safe = y.clamp(self.epsilon, 1.0 - self.epsilon);
        let mu_safe = mu.clamp(self.epsilon, 1.0 - self.epsilon);

        // Parameters: α = μφ, β = (1-μ)φ
        let alpha = mu_safe * self.phi;
        let beta = (1.0 - mu_safe) * self.phi;

        // NLL = log Γ(α) + log Γ(β) - log Γ(φ) - (α-1)log(y) - (β-1)log(1-y)
        // Using lgamma for log Γ
        let log_gamma_alpha = Self::log_gamma(alpha);
        let log_gamma_beta = Self::log_gamma(beta);
        let log_gamma_phi = Self::log_gamma(self.phi);

        log_gamma_alpha + log_gamma_beta
            - log_gamma_phi
            - (alpha - 1.0) * y_safe.ln()
            - (beta - 1.0) * (1.0 - y_safe).ln()
    }

    /// Log-gamma function ln(Γ(x))
    ///
    /// Uses Stirling's approximation for large x, iterative recurrence for small x.
    #[inline]
    fn log_gamma(x: f32) -> f32 {
        const PI: f32 = std::f32::consts::PI;

        if x > 10.0 {
            // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)ln(x) - x + 0.5·ln(2π) + 1/(12x)
            return (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + 1.0 / (12.0 * x);
        }

        // For x in (0, 10], use iterative recurrence to shift to x > 10
        // Γ(x+1) = x·Γ(x), so ln(Γ(x)) = ln(Γ(x+n)) - sum(ln(x+k) for k=0..n-1)
        let mut y = x;
        let mut log_prod = 0.0;

        // Shift upward to y > 10 using recurrence
        while y < 10.0 {
            log_prod += y.ln();
            y += 1.0;
        }

        // Now y > 10, compute ln(Γ(y)) using Stirling's approximation
        let log_gamma_y = (y - 0.5) * y.ln() - y + 0.5 * (2.0 * PI).ln() + 1.0 / (12.0 * y);

        // Return ln(Γ(x)) = ln(Γ(y)) - log_prod
        log_gamma_y - log_prod
    }
}

impl crate::loss::LossFunction for BetaLoss {
    /// Compute loss value for a single prediction
    ///
    /// Returns negative log-likelihood of Beta distribution.
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let mu = sigmoid(prediction);
        self.neg_log_likelihood(target, mu)
    }

    /// Compute gradient (negative gradient for boosting)
    ///
    /// # Mathematical Derivation
    ///
    /// Given raw prediction `r`, we have μ = sigmoid(r).
    /// The loss is L = -log Beta(y; μφ, (1-μ)φ).
    ///
    /// By chain rule:
    /// ```text
    /// dL/dr = (dL/dμ) * (dμ/dr)
    /// ```
    ///
    /// where:
    /// - dμ/dr = μ(1-μ)  [sigmoid derivative]
    /// - dL/dμ = φ[ψ(μφ) - ψ((1-μ)φ) - log(y/(1-y))]  [from Beta NLL]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        // Clamp target to (epsilon, 1-epsilon)
        let y_safe = target.clamp(self.epsilon, 1.0 - self.epsilon);

        // Apply sigmoid to get μ ∈ (0, 1)
        let mu = sigmoid(prediction);
        let mu_safe = mu.clamp(self.epsilon, 1.0 - self.epsilon);

        // Parameters
        let alpha = mu_safe * self.phi;
        let beta = (1.0 - mu_safe) * self.phi;

        // dL/dμ = φ[ψ(α) - ψ(β) - log(y/(1-y))]
        let digamma_alpha = Self::digamma(alpha);
        let digamma_beta = Self::digamma(beta);
        let log_ratio = (y_safe / (1.0 - y_safe)).ln();

        let dl_dmu = self.phi * (digamma_alpha - digamma_beta - log_ratio);

        // dμ/dr = μ(1-μ)
        let dmu_dr = mu * (1.0 - mu);

        // Gradient: dL/dr = (dL/dμ) * (dμ/dr)
        dl_dmu * dmu_dr
    }

    /// Compute hessian (second derivative)
    ///
    /// Uses an approximation common in GBDT:
    /// ```text
    /// d²L/dr² ≈ |dL/dr| * μ(1-μ)
    /// ```
    ///
    /// This avoids computing the exact hessian which involves trigamma functions.
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        let grad = self.gradient(target, prediction);
        let mu = sigmoid(prediction);
        let dmu_dr = mu * (1.0 - mu);

        // Hessian approximation: |gradient| * sigmoid_derivative + epsilon for stability
        grad.abs() * dmu_dr + 1e-8
    }

    /// Loss function name
    fn name(&self) -> &'static str {
        "BetaLoss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::LossFunction;

    #[test]
    fn test_beta_loss_creation() {
        // Valid phi
        assert!(BetaLoss::new(1.0).is_ok());
        assert!(BetaLoss::new(10.0).is_ok());
        assert!(BetaLoss::new(100.0).is_ok());

        // Invalid phi
        assert!(BetaLoss::new(0.0).is_err());
        assert!(BetaLoss::new(-1.0).is_err());
    }

    #[test]
    fn test_beta_loss_epsilon() {
        let loss = BetaLoss::new(10.0).unwrap();

        // Valid epsilon
        assert!(loss.clone().with_epsilon(0.001).is_ok());
        assert!(loss.clone().with_epsilon(0.1).is_ok());

        // Invalid epsilon
        assert!(loss.clone().with_epsilon(0.0).is_err());
        assert!(loss.clone().with_epsilon(0.5).is_err());
        assert!(loss.clone().with_epsilon(1.0).is_err());
    }

    #[test]
    fn test_digamma_values() {
        // Test known values (approximate)
        let psi_1 = BetaLoss::digamma(1.0);
        assert!(
            (psi_1 + std::f32::consts::EULER_GAMMA).abs() < 0.01,
            "ψ(1) ≈ -γ, got {}",
            psi_1
        );

        let psi_2 = BetaLoss::digamma(2.0);
        assert!(
            (psi_2 - (1.0 - std::f32::consts::EULER_GAMMA)).abs() < 0.01,
            "ψ(2) ≈ 1-γ, got {}",
            psi_2
        );

        // Test monotonicity
        assert!(BetaLoss::digamma(2.0) > BetaLoss::digamma(1.0));
        assert!(BetaLoss::digamma(5.0) > BetaLoss::digamma(2.0));
    }

    #[test]
    fn test_log_gamma_values() {
        // Test known values
        let log_gamma_1 = BetaLoss::log_gamma(1.0);
        assert!(
            log_gamma_1.abs() < 0.01,
            "ln(Γ(1)) = 0, got {}",
            log_gamma_1
        );

        // Γ(2) = 1, so ln(Γ(2)) = 0
        let log_gamma_2 = BetaLoss::log_gamma(2.0);
        assert!(
            log_gamma_2.abs() < 0.01,
            "ln(Γ(2)) = 0, got {}",
            log_gamma_2
        );

        // Test monotonicity for x > 2
        assert!(BetaLoss::log_gamma(5.0) > BetaLoss::log_gamma(3.0));
        assert!(BetaLoss::log_gamma(10.0) > BetaLoss::log_gamma(5.0));
    }

    #[test]
    fn test_beta_loss_gradients() {
        let loss = BetaLoss::new(10.0).unwrap();

        let predictions = [0.0, 1.0, -1.0]; // Raw logits
        let targets = [0.5, 0.8, 0.2];

        // Test gradient computation
        for i in 0..3 {
            let grad = loss.gradient(targets[i], predictions[i]);
            let hess = loss.hessian(targets[i], predictions[i]);

            // Check that values are finite
            assert!(grad.is_finite(), "Gradient should be finite");
            assert!(hess.is_finite(), "Hessian should be finite");

            // Hessian should be positive (for stability)
            assert!(hess > 0.0, "Hessian should be positive");
        }
    }

    #[test]
    fn test_beta_loss_value() {
        let loss = BetaLoss::new(10.0).unwrap();

        let prediction = 0.0; // sigmoid(0) = 0.5
        let target = 0.5;

        let loss_val = loss.loss(target, prediction);

        // Loss should be finite (can be negative for high-density regions where PDF > 1)
        assert!(loss_val.is_finite());

        // For well-matched prediction (μ = y = 0.5 at the mode), loss should be reasonable
        assert!(
            loss_val < 5.0 && loss_val > -5.0,
            "Loss out of reasonable range: {}",
            loss_val
        );
    }

    #[test]
    fn test_beta_loss_boundary_handling() {
        let loss = BetaLoss::new(5.0).unwrap();

        // Near-boundary targets (will be epsilon-clipped)
        let test_cases = [(5.0, 0.99), (-5.0, 0.01), (0.0, 0.5)];

        for (pred, target) in &test_cases {
            let grad = loss.gradient(*target, *pred);
            let hess = loss.hessian(*target, *pred);

            // Should not panic or produce NaN/Inf
            assert!(grad.is_finite(), "Gradient should be finite");
            assert!(hess.is_finite(), "Hessian should be finite");
            assert!(hess > 0.0, "Hessian should be positive");
        }
    }

    #[test]
    fn test_beta_loss_name() {
        let loss = BetaLoss::new(10.0).unwrap();
        assert_eq!(loss.name(), "BetaLoss");
    }
}
