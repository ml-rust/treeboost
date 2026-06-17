//! Tree tests for multi-output vector leaf storage
//!
//! Tree Node Vector Storage (2D Multi-Label Support)

use treeboost::tree::{VectorNode, VectorTree};

/// Test vector node creation and accessors
#[test]
fn test_tree_node_vector_storage() {
    let num_outputs = 3;

    // Create a leaf node with vector values
    let values = vec![1.0, 2.0, 3.0];
    let sum_grads = vec![10.0, 20.0, 30.0];
    let sum_hess = vec![5.0, 10.0, 15.0];

    let leaf = VectorNode::leaf(values.clone(), 2, 100, sum_grads.clone(), sum_hess.clone());

    assert!(leaf.is_leaf());
    assert_eq!(leaf.num_outputs(), num_outputs);
    assert_eq!(leaf.leaf_values(), Some(&values[..]));
    assert_eq!(leaf.depth, 2);
    assert_eq!(leaf.num_samples, 100);

    // Verify gradient/hessian accessors
    assert_eq!(leaf.sum_gradients(), &sum_grads[..]);
    assert_eq!(leaf.sum_hessians(), &sum_hess[..]);

    // Create an internal node
    let internal = VectorNode::internal(
        5,                      // feature_idx
        128,                    // bin_threshold
        5.5,                    // split_value
        1,                      // left_child
        2,                      // right_child
        1,                      // depth
        200,                    // num_samples
        vec![15.0, 25.0, 35.0], // sum_grads
        vec![8.0, 12.0, 16.0],  // sum_hess
    );

    assert!(!internal.is_leaf());
    assert_eq!(internal.num_outputs(), num_outputs);
    assert_eq!(internal.leaf_values(), None);

    let split_info = internal.split_info();
    assert!(split_info.is_some());
    let (feat, bin, val, left, right) = split_info.unwrap();
    assert_eq!(feat, 5);
    assert_eq!(bin, 128);
    assert!((val - 5.5).abs() < 1e-10);
    assert_eq!(left, 1);
    assert_eq!(right, 2);
}

/// Test vector leaf weight computation
#[test]
fn test_vector_leaf_weight_computation() {
    // For multi-output, weight_k = -sum_grad_k / (sum_hess_k + lambda)
    let lambda = 1.0;

    let sum_grads = vec![-10.0, -20.0, -30.0];
    let sum_hess = vec![20.0, 40.0, 60.0];

    let weights = VectorNode::compute_leaf_weights(&sum_grads, &sum_hess, lambda);

    // weight_0 = -(-10) / (20 + 1) = 10/21 ≈ 0.476
    // weight_1 = -(-20) / (40 + 1) = 20/41 ≈ 0.488
    // weight_2 = -(-30) / (60 + 1) = 30/61 ≈ 0.492
    assert!(
        (weights[0] - 10.0 / 21.0).abs() < 1e-6,
        "weight_0 = {}",
        weights[0]
    );
    assert!(
        (weights[1] - 20.0 / 41.0).abs() < 1e-6,
        "weight_1 = {}",
        weights[1]
    );
    assert!(
        (weights[2] - 30.0 / 61.0).abs() < 1e-6,
        "weight_2 = {}",
        weights[2]
    );
}

/// Test VectorTree creation and prediction
#[test]
fn test_vector_tree_predict() {
    // Create a simple tree:
    //        [f0 <= 5]
    //        /        \
    //   leaf=[1,2]   [f1 <= 10]
    //               /         \
    //         leaf=[3,4]   leaf=[5,6]

    let num_outputs = 2;
    let tree = VectorTree::from_nodes(
        vec![
            // Root: internal node, split on feature 0 at bin 5
            VectorNode::internal(0, 5, 5.0, 1, 2, 0, 100, vec![0.0, 0.0], vec![100.0, 100.0]),
            // Left child: leaf with values [1.0, 2.0]
            VectorNode::leaf(vec![1.0, 2.0], 1, 50, vec![0.0, 0.0], vec![50.0, 50.0]),
            // Right child: internal node, split on feature 1 at bin 10
            VectorNode::internal(1, 10, 10.0, 3, 4, 1, 50, vec![0.0, 0.0], vec![50.0, 50.0]),
            // Right-left: leaf with values [3.0, 4.0]
            VectorNode::leaf(vec![3.0, 4.0], 2, 25, vec![0.0, 0.0], vec![25.0, 25.0]),
            // Right-right: leaf with values [5.0, 6.0]
            VectorNode::leaf(vec![5.0, 6.0], 2, 25, vec![0.0, 0.0], vec![25.0, 25.0]),
        ],
        num_outputs,
    );

    assert_eq!(tree.num_outputs(), 2);
    assert_eq!(tree.num_nodes(), 5);
    assert_eq!(tree.num_leaves(), 3);
    assert_eq!(tree.max_depth(), 2);

    // Test predictions
    // f0=3 (<=5): go left -> leaf=[1.0, 2.0]
    let pred1 = tree.predict(|f| if f == 0 { 3 } else { 0 });
    assert_eq!(pred1, vec![1.0, 2.0]);

    // f0=7 (>5): go right, f1=5 (<=10): go left -> leaf=[3.0, 4.0]
    let pred2 = tree.predict(|f| if f == 0 { 7 } else { 5 });
    assert_eq!(pred2, vec![3.0, 4.0]);

    // f0=7 (>5): go right, f1=15 (>10): go right -> leaf=[5.0, 6.0]
    let pred3 = tree.predict(|f| if f == 0 { 7 } else { 15 });
    assert_eq!(pred3, vec![5.0, 6.0]);
}

/// Test batch prediction with accumulation
#[test]
fn test_vector_tree_batch_predict() {
    let num_outputs = 2;

    // Simple tree: all samples go to same leaf
    let tree = VectorTree::from_nodes(
        vec![VectorNode::leaf(
            vec![1.5, 2.5],
            0,
            100,
            vec![0.0, 0.0],
            vec![100.0, 100.0],
        )],
        num_outputs,
    );

    // Test batch prediction with accumulation
    let num_samples = 5;
    let mut predictions = vec![0.0f32; num_samples * num_outputs];

    // First tree contribution
    tree.predict_batch_add(
        |sample_idx, feature_idx| (sample_idx + feature_idx) as u8 % 10,
        num_samples,
        &mut predictions,
    );

    // All samples get [1.5, 2.5] added
    for sample_idx in 0..num_samples {
        let offset = sample_idx * num_outputs;
        assert!((predictions[offset] - 1.5).abs() < 1e-6);
        assert!((predictions[offset + 1] - 2.5).abs() < 1e-6);
    }

    // Second tree contribution (same tree, accumulates)
    tree.predict_batch_add(
        |sample_idx, feature_idx| (sample_idx + feature_idx) as u8 % 10,
        num_samples,
        &mut predictions,
    );

    // All samples now have [3.0, 5.0]
    for sample_idx in 0..num_samples {
        let offset = sample_idx * num_outputs;
        assert!((predictions[offset] - 3.0).abs() < 1e-6);
        assert!((predictions[offset + 1] - 5.0).abs() < 1e-6);
    }
}

/// Test VectorTree with raw value prediction
#[test]
fn test_vector_tree_predict_raw() {
    let num_outputs = 2;

    // Tree: split on feature 0 at value 5.0
    let tree = VectorTree::from_nodes(
        vec![
            VectorNode::internal(
                0,
                127,
                5.0,
                1,
                2,
                0,
                100,
                vec![0.0, 0.0],
                vec![100.0, 100.0],
            ),
            VectorNode::leaf(vec![1.0, 2.0], 1, 50, vec![0.0, 0.0], vec![50.0, 50.0]),
            VectorNode::leaf(vec![3.0, 4.0], 1, 50, vec![0.0, 0.0], vec![50.0, 50.0]),
        ],
        num_outputs,
    );

    // Value 3.0 <= 5.0: go left
    let pred1 = tree.predict_raw(|_| 3.0);
    assert_eq!(pred1, vec![1.0, 2.0]);

    // Value 7.0 > 5.0: go right
    let pred2 = tree.predict_raw(|_| 7.0);
    assert_eq!(pred2, vec![3.0, 4.0]);
}

/// Test single-leaf tree (root is a leaf)
#[test]
fn test_vector_tree_single_leaf() {
    let num_outputs = 3;

    let values = vec![0.5, 1.0, 1.5];
    let tree = VectorTree::new(
        values.clone(),
        100,
        vec![0.0, 0.0, 0.0],
        vec![100.0, 100.0, 100.0],
    );

    assert_eq!(tree.num_nodes(), 1);
    assert_eq!(tree.num_leaves(), 1);
    assert_eq!(tree.max_depth(), 0);
    assert_eq!(tree.num_outputs(), num_outputs);

    // Any prediction should return the root leaf values
    let pred = tree.predict(|_| 0);
    assert_eq!(pred, values);
}
