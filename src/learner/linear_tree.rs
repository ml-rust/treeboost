//! Linear Tree weak learner for gradient boosting
//!
//! A Linear Tree combines decision tree partitioning with linear models in the leaves.
//! Instead of outputting a constant value per leaf, each leaf contains a linear equation:
//! `prediction = w · x + b`
//!
//! # Benefits
//!
//! - **10-100x fewer trees** needed for same accuracy
//! - **Captures local linear relationships** within each partition
//! - **Better extrapolation** within leaf regions
//! - **Smoother predictions** than standard trees
//!
//! # Algorithm
//!
//! 1. Grow a decision tree to partition the feature space
//! 2. For each leaf, collect samples that fall into it
//! 3. Fit Ridge regression on each leaf's samples
//! 4. Prediction: traverse tree → find leaf → apply leaf's linear model
//!
//! # Example
//!
//! ```ignore
//! use treeboost::learner::{LinearTreeBooster, LinearTreeConfig};
//!
//! let config = LinearTreeConfig::default();
//! let mut booster = LinearTreeBooster::new(config);
//!
//! booster.fit_on_gradients(&dataset, &raw_features, num_features, &gradients, &hessians)?;
//! let predictions = booster.predict_batch(&dataset, &raw_features, num_features);
//! ```

use crate::dataset::BinnedDataset;
use crate::defaults::learners::linear_tree as linear_tree_defaults;
use crate::learner::{LinearConfig, TreeConfig};
use crate::tree::{NodeType, Tree};
use crate::Result;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for LinearTreeBooster
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct LinearTreeConfig {
    /// Tree structure configuration
    pub tree_config: TreeConfig,

    /// Linear model configuration for leaves
    pub linear_config: LinearConfig,

    /// Minimum samples in a leaf to fit a linear model
    /// Below this, use constant prediction (standard tree leaf)
    pub min_samples_for_linear: usize,
}

impl Default for LinearTreeConfig {
    fn default() -> Self {
        Self {
            tree_config: TreeConfig::default()
                .with_max_depth(linear_tree_defaults::LINEAR_TREE_MAX_DEPTH) // Shallower trees since leaves are more expressive
                .expect("valid max_depth")
                .with_max_leaves(linear_tree_defaults::LINEAR_TREE_MAX_LEAVES)
                .expect("valid max_leaves")
                .with_min_samples_leaf(linear_tree_defaults::LINEAR_TREE_MIN_SAMPLES_LEAF) // Need enough samples for linear fit
                .expect("valid min_samples_leaf"),
            linear_config: LinearConfig::default()
                .with_lambda(linear_tree_defaults::LINEAR_TREE_LAMBDA)
                .expect("valid lambda")
                .with_max_iter(linear_tree_defaults::LINEAR_TREE_MAX_ITER)
                .expect("valid max_iter"),
            min_samples_for_linear: linear_tree_defaults::MIN_SAMPLES_FOR_LINEAR,
        }
    }
}

impl LinearTreeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tree_config(mut self, config: TreeConfig) -> Self {
        self.tree_config = config;
        self
    }

    pub fn with_linear_config(mut self, config: LinearConfig) -> Self {
        self.linear_config = config;
        self
    }

    pub fn with_min_samples_for_linear(mut self, min_samples: usize) -> Self {
        self.min_samples_for_linear = min_samples.max(2);
        self
    }
}

// =============================================================================
// Leaf Linear Model
// =============================================================================

/// Linear model for a single leaf
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct LeafLinearModel {
    /// Weights (one per feature)
    pub weights: Vec<f32>,
    /// Bias term
    pub bias: f32,
    /// Whether this leaf uses linear model (vs constant)
    pub is_linear: bool,
    /// Constant value (used when is_linear = false)
    pub constant: f32,
}

impl LeafLinearModel {
    /// Create a constant leaf (no linear model)
    pub fn constant(value: f32) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            is_linear: false,
            constant: value,
        }
    }

    /// Create a linear leaf
    pub fn linear(weights: Vec<f32>, bias: f32) -> Self {
        Self {
            weights,
            bias,
            is_linear: true,
            constant: 0.0,
        }
    }

    /// Predict for a single sample
    #[inline]
    pub fn predict(&self, features: &[f32]) -> f32 {
        if !self.is_linear {
            return self.constant;
        }

        let mut pred = self.bias;
        for (i, &w) in self.weights.iter().enumerate() {
            if i < features.len() {
                pred += w * features[i];
            }
        }
        pred
    }
}

// =============================================================================
// LinearTreeBooster
// =============================================================================

/// Linear Tree booster: decision tree with linear models in leaves
///
/// Combines the partitioning power of decision trees with the smoothness
/// of linear regression. Each leaf contains a Ridge regression model
/// fitted on the samples that fall into that leaf.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct LinearTreeBooster {
    /// The tree structure (for partitioning)
    tree: Option<Tree>,

    /// Linear models per leaf (indexed by leaf node index)
    leaf_models: Vec<(usize, LeafLinearModel)>, // (node_idx, model)

    /// Configuration
    config: LinearTreeConfig,

    /// Number of features
    num_features: usize,

    /// Feature means for standardization
    feature_means: Vec<f32>,

    /// Feature stds for standardization
    feature_stds: Vec<f32>,
}

impl LinearTreeBooster {
    /// Create a new LinearTreeBooster
    pub fn new(config: LinearTreeConfig) -> Self {
        Self {
            tree: None,
            leaf_models: Vec::new(),
            config,
            num_features: 0,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &LinearTreeConfig {
        &self.config
    }

    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }

    /// Get the underlying tree
    pub fn tree(&self) -> Option<&Tree> {
        self.tree.as_ref()
    }

    /// Get number of leaf models
    pub fn num_leaf_models(&self) -> usize {
        self.leaf_models.len()
    }

    /// Fit on gradients/hessians
    ///
    /// # Arguments
    /// - `dataset`: Binned dataset for tree structure
    /// - `raw_features`: Raw feature values (row-major: [row0_f0, row0_f1, ..., row1_f0, ...])
    /// - `num_features`: Number of features
    /// - `gradients`: Negative gradient of loss
    /// - `hessians`: Second derivative of loss
    pub fn fit_on_gradients(
        &mut self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<()> {
        let num_rows = dataset.num_rows();
        self.num_features = num_features;

        // Step 1: Compute feature statistics for standardization
        self.compute_feature_stats(raw_features, num_features, num_rows);

        // Step 2: Grow tree structure using TreeGrower
        let grower = self.config.tree_config.build_grower(num_features, None);
        let tree = grower.grow(dataset, gradients, hessians)?;

        // Step 3: Assign samples to leaves
        let leaf_assignments = self.assign_samples_to_leaves(&tree, dataset, num_rows);

        // Step 4: Fit linear models in each leaf
        self.fit_leaf_models(
            &tree,
            &leaf_assignments,
            raw_features,
            num_features,
            gradients,
            hessians,
        );

        self.tree = Some(tree);
        Ok(())
    }

    /// Compute feature means and stds for internal standardization
    fn compute_feature_stats(&mut self, features: &[f32], num_features: usize, num_rows: usize) {
        self.feature_means = vec![0.0; num_features];
        self.feature_stds = vec![1.0; num_features];

        if num_rows == 0 {
            return;
        }

        for j in 0..num_features {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;

            for i in 0..num_rows {
                let val = features[i * num_features + j] as f64;
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / num_rows as f64;
            let variance = (sum_sq / num_rows as f64) - mean * mean;
            let std = variance.max(0.0).sqrt();

            self.feature_means[j] = mean as f32;
            self.feature_stds[j] = if std > 1e-10 { std as f32 } else { 1.0 };
        }
    }

    /// Standardize a feature value
    #[inline]
    fn standardize(&self, value: f32, feature_idx: usize) -> f32 {
        (value - self.feature_means[feature_idx]) / self.feature_stds[feature_idx]
    }

    /// Assign each sample to its leaf node
    fn assign_samples_to_leaves(
        &self,
        tree: &Tree,
        dataset: &BinnedDataset,
        num_rows: usize,
    ) -> Vec<Vec<usize>> {
        // Map: node_idx -> list of sample indices
        let num_nodes = tree.num_nodes();
        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

        for row_idx in 0..num_rows {
            let leaf_idx = self.find_leaf_index(tree, dataset, row_idx);
            assignments[leaf_idx].push(row_idx);
        }

        assignments
    }

    /// Find the leaf node index for a sample
    fn find_leaf_index(&self, tree: &Tree, dataset: &BinnedDataset, row_idx: usize) -> usize {
        let mut node_idx = 0;

        loop {
            let node = tree.get_node(node_idx);
            match node.node_type {
                NodeType::Leaf { .. } => return node_idx,
                NodeType::Internal {
                    feature_idx,
                    bin_threshold,
                    left_child,
                    right_child,
                    ..
                } => {
                    let bin = dataset.get_bin(row_idx, feature_idx);
                    node_idx = if bin <= bin_threshold {
                        left_child
                    } else {
                        right_child
                    };
                }
            }
        }
    }

    /// Fit linear models in each leaf
    fn fit_leaf_models(
        &mut self,
        tree: &Tree,
        leaf_assignments: &[Vec<usize>],
        raw_features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) {
        self.leaf_models.clear();

        for (node_idx, sample_indices) in leaf_assignments.iter().enumerate() {
            let node = tree.get_node(node_idx);

            // Only process leaf nodes
            if !node.is_leaf() {
                continue;
            }

            let num_samples = sample_indices.len();

            // Get default leaf value from tree
            let default_value = match node.node_type {
                NodeType::Leaf { value } => value,
                _ => 0.0,
            };

            // If too few samples, use constant prediction
            if num_samples < self.config.min_samples_for_linear {
                self.leaf_models
                    .push((node_idx, LeafLinearModel::constant(default_value)));
                continue;
            }

            // Fit Ridge regression on leaf samples
            let model = self.fit_ridge_in_leaf(
                sample_indices,
                raw_features,
                num_features,
                gradients,
                hessians,
                default_value,
            );

            self.leaf_models.push((node_idx, model));
        }
    }

    /// Fit Ridge regression for samples in a leaf
    fn fit_ridge_in_leaf(
        &self,
        sample_indices: &[usize],
        raw_features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
        default_value: f32,
    ) -> LeafLinearModel {
        let n = sample_indices.len();
        let lambda = self.config.linear_config.lambda;
        let max_iter = self.config.linear_config.max_iter;
        let tol = self.config.linear_config.tol;
        // Shrinkage factor for linear leaf models (ensemble weighting, not optimization step size)
        let shrinkage = self.config.linear_config.shrinkage_factor;

        // Initialize weights and bias
        let mut weights = vec![0.0f32; num_features];
        let mut bias = default_value;

        // Precompute standardized features for this leaf
        let mut std_features: Vec<Vec<f32>> = Vec::with_capacity(n);
        for &idx in sample_indices {
            let mut row = Vec::with_capacity(num_features);
            for j in 0..num_features {
                let val = raw_features[idx * num_features + j];
                row.push(self.standardize(val, j));
            }
            std_features.push(row);
        }

        // Coordinate descent
        for _iter in 0..max_iter {
            let mut max_change = 0.0f32;

            // Update bias
            {
                let mut grad_bias = 0.0f32;
                let mut hess_bias = 0.0f32;

                for (local_idx, &global_idx) in sample_indices.iter().enumerate() {
                    let h = hessians[global_idx];
                    let g = gradients[global_idx];

                    // Current prediction
                    let mut pred = bias;
                    for (j, &w) in weights.iter().enumerate() {
                        pred += w * std_features[local_idx][j];
                    }

                    // Residual (Newton direction)
                    let residual = pred + g / h.max(1e-10);
                    grad_bias += h * residual;
                    hess_bias += h;
                }

                let delta = -grad_bias / (hess_bias + lambda);
                let delta = delta.clamp(-10.0, 10.0);
                bias += shrinkage * delta;
                max_change = max_change.max(delta.abs());
            }

            // Update each weight
            for j in 0..num_features {
                let mut grad_j = 0.0f32;
                let mut hess_j = 0.0f32;

                for (local_idx, &global_idx) in sample_indices.iter().enumerate() {
                    let h = hessians[global_idx];
                    let g = gradients[global_idx];
                    let x_j = std_features[local_idx][j];

                    // Current prediction
                    let mut pred = bias;
                    for (k, &w) in weights.iter().enumerate() {
                        pred += w * std_features[local_idx][k];
                    }

                    let residual = pred + g / h.max(1e-10);
                    grad_j += h * residual * x_j;
                    hess_j += h * x_j * x_j;
                }

                // L2 regularization
                grad_j += lambda * weights[j];

                let delta = -grad_j / (hess_j + lambda);
                let delta = delta.clamp(-10.0, 10.0);
                weights[j] += shrinkage * delta;
                max_change = max_change.max(delta.abs());
            }

            if max_change < tol {
                break;
            }
        }

        // Convert standardized weights back to original scale
        // w_orig[j] = w_std[j] / std[j]
        // bias_orig = bias_std - Σ (w_std[j] * mean[j] / std[j])
        let mut bias_orig = bias;
        let weights_orig: Vec<f32> = weights
            .iter()
            .zip(self.feature_means.iter())
            .zip(self.feature_stds.iter())
            .map(|((&w, &mean), &std)| {
                bias_orig -= w * mean / std;
                w / std
            })
            .collect();

        LeafLinearModel::linear(weights_orig, bias_orig)
    }

    /// Predict for all rows
    pub fn predict_batch(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
        num_features: usize,
    ) -> Vec<f32> {
        let tree = match &self.tree {
            Some(t) => t,
            None => return vec![0.0; dataset.num_rows()],
        };

        let num_rows = dataset.num_rows();
        let mut predictions = Vec::with_capacity(num_rows);

        // Build leaf model lookup
        let mut leaf_lookup: std::collections::HashMap<usize, &LeafLinearModel> =
            std::collections::HashMap::new();
        for (node_idx, model) in &self.leaf_models {
            leaf_lookup.insert(*node_idx, model);
        }

        for row_idx in 0..num_rows {
            let leaf_idx = self.find_leaf_index(tree, dataset, row_idx);

            let pred = if let Some(model) = leaf_lookup.get(&leaf_idx) {
                // Extract row features
                let row_features: Vec<f32> = (0..num_features)
                    .map(|j| raw_features[row_idx * num_features + j])
                    .collect();
                model.predict(&row_features)
            } else {
                // Fallback to tree's constant value
                tree.predict_row(dataset, row_idx)
            };

            predictions.push(pred);
        }

        predictions
    }

    /// Predict for a single row
    pub fn predict_row(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
        num_features: usize,
        row_idx: usize,
    ) -> f32 {
        let tree = match &self.tree {
            Some(t) => t,
            None => return 0.0,
        };

        let leaf_idx = self.find_leaf_index(tree, dataset, row_idx);

        // Find leaf model
        for (node_idx, model) in &self.leaf_models {
            if *node_idx == leaf_idx {
                let row_features: Vec<f32> = (0..num_features)
                    .map(|j| raw_features[row_idx * num_features + j])
                    .collect();
                return model.predict(&row_features);
            }
        }

        // Fallback
        tree.predict_row(dataset, row_idx)
    }

    /// Add predictions to existing buffer
    pub fn predict_batch_add(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
        num_features: usize,
        predictions: &mut [f32],
    ) {
        let batch_preds = self.predict_batch(dataset, raw_features, num_features);
        for (i, p) in batch_preds.into_iter().enumerate() {
            predictions[i] += p;
        }
    }

    /// Number of parameters (tree complexity + linear weights)
    pub fn num_params(&self) -> usize {
        let tree_params = self.tree.as_ref().map(|t| t.num_leaves()).unwrap_or(0);
        let linear_params: usize = self
            .leaf_models
            .iter()
            .map(|(_, m)| if m.is_linear { m.weights.len() + 1 } else { 1 })
            .sum();
        tree_params + linear_params
    }

    /// Reset the booster
    pub fn reset(&mut self) {
        self.tree = None;
        self.leaf_models.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.1).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: (0..255).map(|b| b as f64).collect(),
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    fn create_raw_features(num_rows: usize, num_features: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for r in 0..num_rows {
            for f in 0..num_features {
                features.push(((r * 3 + f * 7) % 256) as f32);
            }
        }
        features
    }

    #[test]
    fn test_linear_tree_config_defaults() {
        let config = LinearTreeConfig::default();
        assert_eq!(config.tree_config.max_depth, 4);
        assert_eq!(config.min_samples_for_linear, 10);
    }

    #[test]
    fn test_linear_tree_config_builder() {
        let config = LinearTreeConfig::new()
            .with_min_samples_for_linear(20)
            .with_tree_config(
                TreeConfig::default()
                    .with_max_depth(3)
                    .expect("valid max_depth"),
            );

        assert_eq!(config.min_samples_for_linear, 20);
        assert_eq!(config.tree_config.max_depth, 3);
    }

    #[test]
    fn test_linear_tree_booster_creation() {
        let config = LinearTreeConfig::default();
        let booster = LinearTreeBooster::new(config);

        assert!(!booster.is_fitted());
        assert!(booster.tree().is_none());
    }

    #[test]
    fn test_leaf_linear_model_constant() {
        let model = LeafLinearModel::constant(5.0);
        assert!(!model.is_linear);
        assert_eq!(model.predict(&[1.0, 2.0, 3.0]), 5.0);
    }

    #[test]
    fn test_leaf_linear_model_linear() {
        let model = LeafLinearModel::linear(vec![1.0, 2.0], 0.5);
        assert!(model.is_linear);
        // pred = 0.5 + 1.0*1.0 + 2.0*2.0 = 0.5 + 1 + 4 = 5.5
        assert!((model.predict(&[1.0, 2.0]) - 5.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_tree_booster_fit() {
        let dataset = create_test_dataset(100, 3);
        let raw_features = create_raw_features(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| -(i as f32) * 0.1).collect();
        let hessians = vec![1.0; 100];

        let config = LinearTreeConfig::default().with_min_samples_for_linear(5);

        let mut booster = LinearTreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &raw_features, 3, &gradients, &hessians)
            .unwrap();

        assert!(booster.is_fitted());
        assert!(booster.tree().is_some());
        assert!(booster.num_leaf_models() > 0);
    }

    #[test]
    fn test_linear_tree_booster_predict() {
        let dataset = create_test_dataset(100, 3);
        let raw_features = create_raw_features(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| -(i as f32) * 0.1).collect();
        let hessians = vec![1.0; 100];

        let config = LinearTreeConfig::default().with_min_samples_for_linear(5);

        let mut booster = LinearTreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &raw_features, 3, &gradients, &hessians)
            .unwrap();

        let predictions = booster.predict_batch(&dataset, &raw_features, 3);
        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_linear_tree_single_row_matches_batch() {
        let dataset = create_test_dataset(50, 3);
        let raw_features = create_raw_features(50, 3);
        let gradients: Vec<f32> = (0..50).map(|i| -(i as f32) * 0.1).collect();
        let hessians = vec![1.0; 50];

        let config = LinearTreeConfig::default().with_min_samples_for_linear(3);

        let mut booster = LinearTreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &raw_features, 3, &gradients, &hessians)
            .unwrap();

        let batch_preds = booster.predict_batch(&dataset, &raw_features, 3);
        for (i, &batch_pred) in batch_preds.iter().take(50).enumerate() {
            let single_pred = booster.predict_row(&dataset, &raw_features, 3, i);
            assert!(
                (batch_pred - single_pred).abs() < 1e-5,
                "Mismatch at row {}: batch={}, single={}",
                i,
                batch_pred,
                single_pred
            );
        }
    }

    #[test]
    fn test_linear_tree_booster_reset() {
        let dataset = create_test_dataset(50, 3);
        let raw_features = create_raw_features(50, 3);
        let gradients = vec![-1.0; 50];
        let hessians = vec![1.0; 50];

        let config = LinearTreeConfig::default();
        let mut booster = LinearTreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &raw_features, 3, &gradients, &hessians)
            .unwrap();

        assert!(booster.is_fitted());
        booster.reset();
        assert!(!booster.is_fitted());
    }
}
