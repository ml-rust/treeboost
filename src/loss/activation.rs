//! Activation functions for loss computation
//!
//! Provides numerically stable implementations of common activation functions.

/// Sigmoid function with numerical stability
///
/// Uses the identity `sigmoid(-x) = 1 - sigmoid(x)` to avoid overflow:
/// - For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
/// - For x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
///
/// # Examples
/// ```
/// use treeboost::loss::sigmoid;
///
/// assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
/// assert!(sigmoid(10.0) > 0.999);
/// assert!(sigmoid(-10.0) < 0.001);
/// ```
#[inline]
pub fn sigmoid(x: f32) -> f32 {
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
    fn test_sigmoid_at_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounded() {
        // Use moderate values to avoid floating-point precision limits
        assert!(sigmoid(-10.0) > 0.0);
        assert!(sigmoid(-10.0) < 0.001);
        assert!(sigmoid(10.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // sigmoid(-x) = 1 - sigmoid(x)
        let x = 2.5;
        assert!((sigmoid(-x) - (1.0 - sigmoid(x))).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_numerical_stability() {
        // Extreme values should not cause NaN or Inf
        let extreme_values = [-1000.0, -100.0, 100.0, 1000.0];
        for x in extreme_values {
            let s = sigmoid(x);
            assert!(s.is_finite(), "sigmoid({}) = {} is not finite", x, s);
            assert!(
                (0.0..=1.0).contains(&s),
                "sigmoid({}) = {} out of [0,1]",
                x,
                s
            );
        }
    }
}
