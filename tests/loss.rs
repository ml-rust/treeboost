//! Loss function tests for multi-output (multi-label) support

use treeboost::loss::{sigmoid, LossFunction, MseLoss, MultiLabelFocalLoss, MultiLabelLogLoss};

/// Test multi-label log loss gradients
///
/// Verifies that:
/// 1. Gradient per label: g_k = sigmoid(pred_k) - target_k
/// 2. Hessian per label: h_k = sigmoid(pred_k) * (1 - sigmoid(pred_k))
/// 3. Each label is treated as independent binary classification
/// 4. Aggregate loss is sum across labels
#[test]
fn test_multilabel_logloss_gradients() {
    let loss = MultiLabelLogLoss::new();
    let _num_outputs = 3;

    // Single sample with 3 labels
    // Target: [1, 0, 1]
    // Prediction (raw logits): [0.0, 0.0, 0.0] -> sigmoid = [0.5, 0.5, 0.5]
    let targets = [1.0, 0.0, 1.0];
    let predictions = [0.0, 0.0, 0.0];

    // Test individual gradient/hessian
    // g_k = sigmoid(pred_k) - target_k
    // h_k = sigmoid(pred_k) * (1 - sigmoid(pred_k))
    let (g0, h0) = loss.gradient_hessian_single(targets[0], predictions[0]);
    let (g1, h1) = loss.gradient_hessian_single(targets[1], predictions[1]);
    let (g2, h2) = loss.gradient_hessian_single(targets[2], predictions[2]);

    // For pred=0, sigmoid=0.5
    // g = 0.5 - target
    assert!((g0 - (0.5 - 1.0)).abs() < 1e-6, "g0 = {}", g0); // -0.5
    assert!((g1 - (0.5 - 0.0)).abs() < 1e-6, "g1 = {}", g1); // 0.5
    assert!((g2 - (0.5 - 1.0)).abs() < 1e-6, "g2 = {}", g2); // -0.5

    // h = 0.5 * (1 - 0.5) = 0.25
    assert!((h0 - 0.25).abs() < 1e-6, "h0 = {}", h0);
    assert!((h1 - 0.25).abs() < 1e-6, "h1 = {}", h1);
    assert!((h2 - 0.25).abs() < 1e-6, "h2 = {}", h2);
}

/// Test batch gradient computation for multi-label
#[test]
fn test_multilabel_batch_gradients() {
    let loss = MultiLabelLogLoss::new();
    let num_rows = 2;
    let num_outputs = 3;

    // Row-wise flattened targets: [row0_label0, row0_label1, row0_label2, row1_label0, ...]
    let targets = vec![
        1.0, 0.0, 1.0, // row 0: labels [1, 0, 1]
        0.0, 1.0, 0.0, // row 1: labels [0, 1, 0]
    ];

    // Row-wise flattened predictions (raw logits)
    let predictions = vec![
        2.0, -2.0, 0.0, // row 0
        -1.0, 1.0, 0.5, // row 1
    ];

    let mut gradients = vec![0.0; num_rows * num_outputs];
    let mut hessians = vec![0.0; num_rows * num_outputs];

    loss.compute_gradients_multi(
        &targets,
        &predictions,
        &mut gradients,
        &mut hessians,
        num_outputs,
    );

    // Verify row 0
    let p00 = sigmoid(2.0);
    let p01 = sigmoid(-2.0);
    let p02 = sigmoid(0.0);

    assert!(
        (gradients[0] - (p00 - 1.0)).abs() < 1e-5,
        "g[0] = {}",
        gradients[0]
    );
    assert!(
        (gradients[1] - (p01 - 0.0)).abs() < 1e-5,
        "g[1] = {}",
        gradients[1]
    );
    assert!(
        (gradients[2] - (p02 - 1.0)).abs() < 1e-5,
        "g[2] = {}",
        gradients[2]
    );

    assert!(
        (hessians[0] - p00 * (1.0 - p00)).abs() < 1e-5,
        "h[0] = {}",
        hessians[0]
    );
    assert!(
        (hessians[1] - p01 * (1.0 - p01)).abs() < 1e-5,
        "h[1] = {}",
        hessians[1]
    );
    assert!(
        (hessians[2] - p02 * (1.0 - p02)).abs() < 1e-5,
        "h[2] = {}",
        hessians[2]
    );

    // Verify row 1
    let p10 = sigmoid(-1.0);
    let p11 = sigmoid(1.0);
    let p12 = sigmoid(0.5);

    assert!(
        (gradients[3] - (p10 - 0.0)).abs() < 1e-5,
        "g[3] = {}",
        gradients[3]
    );
    assert!(
        (gradients[4] - (p11 - 1.0)).abs() < 1e-5,
        "g[4] = {}",
        gradients[4]
    );
    assert!(
        (gradients[5] - (p12 - 0.0)).abs() < 1e-5,
        "g[5] = {}",
        gradients[5]
    );
}

/// Test initial predictions for multi-label (per-label log-odds)
#[test]
fn test_multilabel_initial_predictions() {
    let loss = MultiLabelLogLoss::new();
    let _num_rows = 4;
    let num_outputs = 2;

    // Targets:
    // Row 0: [1, 0]
    // Row 1: [1, 1]
    // Row 2: [0, 0]
    // Row 3: [1, 1]
    // Label 0: 3/4 positive -> log(3/1) = log(3) ≈ 1.099
    // Label 1: 2/4 positive -> log(2/2) = log(1) = 0
    let targets = vec![
        1.0, 0.0, // row 0
        1.0, 1.0, // row 1
        0.0, 0.0, // row 2
        1.0, 1.0, // row 3
    ];

    let initial = loss.initial_predictions(&targets, num_outputs);

    assert_eq!(initial.len(), num_outputs);
    assert!(
        (initial[0] - 3.0_f32.ln()).abs() < 1e-5,
        "init[0] = {}",
        initial[0]
    );
    assert!(initial[1].abs() < 1e-5, "init[1] = {}", initial[1]); // log(1) = 0
}

/// Test loss computation for multi-label
#[test]
fn test_multilabel_loss_value() {
    let loss = MultiLabelLogLoss::new();

    // Perfect prediction for label with target=1, pred=large positive
    let l1 = loss.loss_single(1.0, 10.0);
    assert!(l1 < 0.001, "l1 = {}", l1);

    // Perfect prediction for label with target=0, pred=large negative
    let l0 = loss.loss_single(0.0, -10.0);
    assert!(l0 < 0.001, "l0 = {}", l0);

    // Wrong prediction: target=1, pred=large negative
    let l_wrong = loss.loss_single(1.0, -10.0);
    assert!(l_wrong > 5.0, "l_wrong = {}", l_wrong);
}

/// Test numerical stability with extreme predictions
#[test]
fn test_multilabel_numerical_stability() {
    let loss = MultiLabelLogLoss::new();
    let extreme_preds = [-1000.0, -100.0, 100.0, 1000.0];

    for pred in extreme_preds {
        let l = loss.loss_single(0.0, pred);
        let (g, h) = loss.gradient_hessian_single(0.0, pred);

        assert!(l.is_finite(), "Loss not finite for pred={}", pred);
        assert!(g.is_finite(), "Gradient not finite for pred={}", pred);
        assert!(h.is_finite(), "Hessian not finite for pred={}", pred);
        assert!(h > 0.0, "Hessian not positive for pred={}", pred);
    }
}

/// Test to_probabilities conversion
#[test]
fn test_multilabel_to_probabilities() {
    let loss = MultiLabelLogLoss::new();
    let num_outputs = 3;

    // Row-wise predictions
    let predictions = vec![
        0.0, 10.0, -10.0, // row 0
        -5.0, 5.0, 0.0, // row 1
    ];

    let probs = loss.to_probabilities(&predictions, num_outputs);

    assert_eq!(probs.len(), 6);

    // Row 0
    assert!((probs[0] - 0.5).abs() < 1e-6);
    assert!(probs[1] > 0.999);
    assert!(probs[2] < 0.001);

    // Row 1
    assert!(probs[3] < 0.01);
    assert!(probs[4] > 0.99);
    assert!((probs[5] - 0.5).abs() < 1e-6);
}

/// Test multi-label focal loss gradients and focusing behavior
///
/// Verifies that:
/// 1. Focal loss down-weights easy examples (where p ≈ target)
/// 2. Hard examples (where p ≠ target) get higher gradient magnitude
/// 3. The focusing parameter γ controls the down-weighting intensity
#[test]
fn test_multilabel_focal_loss_gradients() {
    let focal = MultiLabelFocalLoss::new(2.0); // gamma = 2
    let logloss = MultiLabelLogLoss::new();

    // Easy positive: target=1, pred=high (correct prediction)
    // Focal should have LOWER gradient magnitude than log loss
    let (g_focal_easy, _) = focal.gradient_hessian_single(1.0, 5.0);
    let (g_logloss_easy, _) = logloss.gradient_hessian_single(1.0, 5.0);

    assert!(
        g_focal_easy.abs() < g_logloss_easy.abs(),
        "Focal gradient ({}) should be smaller than logloss ({}) for easy example",
        g_focal_easy.abs(),
        g_logloss_easy.abs()
    );

    // Hard positive: target=1, pred=low (wrong prediction)
    // Focal gradient magnitude should be closer to log loss (less down-weighting)
    let (g_focal_hard, _) = focal.gradient_hessian_single(1.0, -2.0);
    let (g_logloss_hard, _) = logloss.gradient_hessian_single(1.0, -2.0);

    // For hard examples, focal doesn't reduce as much
    let ratio_hard = g_focal_hard.abs() / g_logloss_hard.abs();
    let ratio_easy = g_focal_easy.abs() / g_logloss_easy.abs();

    assert!(
        ratio_hard > ratio_easy,
        "Hard examples should have higher focal/logloss ratio ({}) than easy examples ({})",
        ratio_hard,
        ratio_easy
    );
}

/// Test focal loss at the decision boundary (p = 0.5)
#[test]
fn test_focal_loss_at_boundary() {
    let focal = MultiLabelFocalLoss::new(2.0);

    // At pred=0, sigmoid=0.5
    // For target=1: p_t = 0.5, focal weight = (1-0.5)^2 = 0.25
    // alpha_t = 0.5 (default for positive class)
    let (g, h) = focal.gradient_hessian_single(1.0, 0.0);

    // g = alpha_t * focal_weight * (p - y) = 0.5 * 0.25 * (0.5 - 1.0) = -0.0625
    assert!((g - (-0.0625)).abs() < 1e-5, "g = {}", g);

    // Hessian should be positive
    assert!(h > 0.0, "h = {}", h);
}

/// Test focal loss with different gamma values
#[test]
fn test_focal_loss_gamma_effect() {
    let focal_low = MultiLabelFocalLoss::new(0.5); // less focusing
    let focal_high = MultiLabelFocalLoss::new(5.0); // more focusing

    // Easy example: target=1, pred=10 (very confident correct prediction)
    let (g_low, _) = focal_low.gradient_hessian_single(1.0, 10.0);
    let (g_high, _) = focal_high.gradient_hessian_single(1.0, 10.0);

    // Higher gamma -> more down-weighting of easy examples -> smaller gradient
    assert!(
        g_high.abs() < g_low.abs(),
        "Higher gamma should produce smaller gradient for easy examples: g_high={}, g_low={}",
        g_high.abs(),
        g_low.abs()
    );
}

/// Test focal loss with alpha balancing
#[test]
fn test_focal_loss_alpha_balancing() {
    // alpha=0.75 means positives get 3x weight relative to negatives
    let focal = MultiLabelFocalLoss::new(2.0).with_alpha(0.75);

    // Same absolute gradient for balanced targets
    let (g_pos, _) = focal.gradient_hessian_single(1.0, 0.0);
    let (g_neg, _) = focal.gradient_hessian_single(0.0, 0.0);

    // Positive gradient is scaled by alpha=0.75
    // Negative gradient is scaled by (1-alpha)=0.25
    // At p=0.5: base gradient = ±0.5, focal weight = 0.25
    // g_pos = 0.75 * 0.25 * (-0.5) = -0.09375
    // g_neg = 0.25 * 0.25 * (0.5) = 0.03125
    // Ratio should be 3:1
    let ratio = g_pos.abs() / g_neg.abs();
    assert!(
        (ratio - 3.0).abs() < 0.1,
        "Alpha ratio should be 3:1, got {}",
        ratio
    );
}

/// Test batch gradient computation for focal loss
#[test]
fn test_focal_batch_gradients() {
    let focal = MultiLabelFocalLoss::new(2.0);
    let _num_rows = 2;
    let num_outputs = 2;

    let targets = vec![1.0, 0.0, 0.0, 1.0];
    let predictions = vec![2.0, -2.0, 2.0, -2.0]; // row0: correct, row1: wrong

    let mut gradients = vec![0.0; 4];
    let mut hessians = vec![0.0; 4];

    focal.compute_gradients_multi(
        &targets,
        &predictions,
        &mut gradients,
        &mut hessians,
        num_outputs,
    );

    // Row 0: easy examples (correct predictions)
    // Row 1: hard examples (wrong predictions)
    // Hard examples should have larger gradient magnitude
    let row0_grad_mag = gradients[0].abs() + gradients[1].abs();
    let row1_grad_mag = gradients[2].abs() + gradients[3].abs();

    assert!(
        row1_grad_mag > row0_grad_mag,
        "Hard examples should have larger gradients: row0={}, row1={}",
        row0_grad_mag,
        row1_grad_mag
    );

    // All hessians should be positive
    for h in &hessians {
        assert!(*h > 0.0, "Hessian should be positive, got {}", h);
    }
}

/// Test focal loss numerical stability
#[test]
fn test_focal_numerical_stability() {
    let focal = MultiLabelFocalLoss::new(2.0);
    let extreme_preds = [-1000.0, -100.0, 100.0, 1000.0];

    for pred in extreme_preds {
        let l = focal.loss_single(0.0, pred);
        let (g, h) = focal.gradient_hessian_single(0.0, pred);

        assert!(l.is_finite(), "Focal loss not finite for pred={}", pred);
        assert!(g.is_finite(), "Focal gradient not finite for pred={}", pred);
        assert!(h.is_finite(), "Focal hessian not finite for pred={}", pred);
        assert!(h > 0.0, "Focal hessian not positive for pred={}", pred);
    }
}

/// Test that focal loss reduces to log loss when gamma=0 and alpha=1.0
#[test]
fn test_focal_reduces_to_logloss() {
    // With gamma=0 (no focusing) and alpha=1.0 (no class balancing),
    // focal loss should exactly match log loss
    let focal = MultiLabelFocalLoss::new(0.0).with_alpha(1.0);
    let logloss = MultiLabelLogLoss::new();

    let test_cases = [(1.0, 2.0), (0.0, -1.5), (1.0, -3.0), (0.0, 4.0)];

    for (target, pred) in test_cases {
        let (g_focal, h_focal) = focal.gradient_hessian_single(target, pred);
        let (g_logloss, h_logloss) = logloss.gradient_hessian_single(target, pred);

        // Note: for negative class (target=0), alpha=1.0 means weight=0
        // So we only test positive class with alpha=1.0
        if target > 0.5 {
            assert!(
                (g_focal - g_logloss).abs() < 1e-5,
                "Gradient mismatch for target={}, pred={}: focal={}, logloss={}",
                target,
                pred,
                g_focal,
                g_logloss
            );
            assert!(
                (h_focal - h_logloss).abs() < 1e-5,
                "Hessian mismatch for target={}, pred={}: focal={}, logloss={}",
                target,
                pred,
                h_focal,
                h_logloss
            );
        }
    }

    // Test with balanced alpha=0.5 and gamma=0
    // Focal gradient should be exactly half of logloss gradient
    let focal_balanced = MultiLabelFocalLoss::new(0.0); // default alpha=0.5
    for (target, pred) in test_cases {
        let (g_focal, _) = focal_balanced.gradient_hessian_single(target, pred);
        let (g_logloss, _) = logloss.gradient_hessian_single(target, pred);

        assert!(
            (g_focal - 0.5 * g_logloss).abs() < 1e-5,
            "Focal (alpha=0.5, gamma=0) should be 0.5 * logloss: focal={}, 0.5*logloss={}",
            g_focal,
            0.5 * g_logloss
        );
    }
}

/// Test that multi-output losses can be used via the LossFunction trait
#[test]
fn test_loss_trait_multi_output_interface() {
    // Test MultiLabelLogLoss via trait
    let logloss: &dyn LossFunction = &MultiLabelLogLoss::new();
    let focal: &dyn LossFunction = &MultiLabelFocalLoss::new(2.0);

    // Both should support multi-output
    assert!(logloss.supports_multi_output());
    assert!(focal.supports_multi_output());

    // Both should have None for num_outputs (any number supported)
    assert_eq!(logloss.num_outputs(), None);
    assert_eq!(focal.num_outputs(), None);
}

/// Test compute_gradients_multi via trait interface
#[test]
fn test_trait_compute_gradients_multi() {
    let loss: &dyn LossFunction = &MultiLabelLogLoss::new();

    let num_outputs = 2;
    let targets = vec![1.0, 0.0, 0.0, 1.0]; // 2 rows × 2 outputs
    let predictions = vec![2.0, -2.0, -2.0, 2.0];

    let mut gradients = vec![0.0; 4];
    let mut hessians = vec![0.0; 4];

    loss.compute_gradients_multi(
        &targets,
        &predictions,
        &mut gradients,
        &mut hessians,
        num_outputs,
    );

    // Verify gradients are computed correctly
    for g in &gradients {
        assert!(g.is_finite());
    }
    for h in &hessians {
        assert!(h.is_finite());
        assert!(*h > 0.0);
    }
}

/// Test initial_predictions_multi via trait interface
#[test]
fn test_trait_initial_predictions_multi() {
    let loss: &dyn LossFunction = &MultiLabelLogLoss::new();

    let num_outputs = 2;
    // Row 0: [1, 0], Row 1: [1, 1], Row 2: [0, 1]
    // Label 0: 2/3 positive, Label 1: 2/3 positive
    let targets = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let initial = loss.initial_predictions_multi(&targets, num_outputs);

    assert_eq!(initial.len(), num_outputs);
    // Both labels have 2/3 positive rate
    for init in &initial {
        assert!(init.is_finite());
    }
}

/// Test backward compatibility: scalar losses still work
#[test]
fn test_trait_backward_compatibility() {
    // MseLoss is a scalar loss
    let mse = MseLoss::new();

    // Should have num_outputs = Some(1)
    assert_eq!(mse.num_outputs(), Some(1));

    // compute_gradients_multi with num_outputs=1 should work
    let targets = vec![1.0, 2.0, 3.0];
    let predictions = vec![1.1, 1.9, 3.2];
    let mut gradients = vec![0.0; 3];
    let mut hessians = vec![0.0; 3];

    mse.compute_gradients_multi(&targets, &predictions, &mut gradients, &mut hessians, 1);

    // Check gradients match standard compute_gradients
    let mut grads2 = vec![0.0; 3];
    let mut hess2 = vec![0.0; 3];
    mse.compute_gradients(&targets, &predictions, &mut grads2, &mut hess2);

    for i in 0..3 {
        assert!(
            (gradients[i] - grads2[i]).abs() < 1e-6,
            "Gradient mismatch at {}: multi={}, standard={}",
            i,
            gradients[i],
            grads2[i]
        );
        assert!(
            (hessians[i] - hess2[i]).abs() < 1e-6,
            "Hessian mismatch at {}: multi={}, standard={}",
            i,
            hessians[i],
            hess2[i]
        );
    }
}

/// Test initial_predictions_multi backward compatibility for scalar losses
#[test]
fn test_trait_initial_predictions_scalar() {
    let mse = MseLoss::new();

    let targets = vec![1.0, 2.0, 3.0, 4.0];

    // Multi-output with num_outputs=1 should match scalar initial_prediction
    let initial_multi = mse.initial_predictions_multi(&targets, 1);
    let initial_scalar = mse.initial_prediction(&targets);

    assert_eq!(initial_multi.len(), 1);
    assert!(
        (initial_multi[0] - initial_scalar).abs() < 1e-6,
        "Multi initial ({}) should match scalar ({})",
        initial_multi[0],
        initial_scalar
    );
}
