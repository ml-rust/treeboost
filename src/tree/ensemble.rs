//! Ensemble tree wrapper for scalar and vector trees
//!
//! Provides a unified `EnsembleTree` enum that can hold either:
//! - `Tree` (scalar f32 leaf values) for regression/binary/multi-class
//! - `VectorTree` (`Vec<f32>` leaf values) for multi-label with unified splits
//!
//! This allows `GBDTModel` to support both approaches transparently.

use crate::dataset::BinnedDataset;
use crate::tree::{Tree, VectorTree};
use rkyv::{Archive, Deserialize, Serialize};

/// Unified tree type for ensembles
///
/// Wraps either a scalar `Tree` (one output per tree) or a `VectorTree`
/// (multiple outputs per tree with shared splits).
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub enum EnsembleTree {
    /// Scalar tree with f32 leaf values
    /// Used for: regression, binary classification, multi-class (one-vs-all)
    Scalar(Tree),

    /// Vector tree with `Vec<f32>` leaf values
    /// Used for: multi-label with unified splits (all labels share same tree structure)
    Vector(VectorTree),
}

impl EnsembleTree {
    /// Check if this is a scalar tree
    #[inline]
    pub fn is_scalar(&self) -> bool {
        matches!(self, EnsembleTree::Scalar(_))
    }

    /// Check if this is a vector tree
    #[inline]
    pub fn is_vector(&self) -> bool {
        matches!(self, EnsembleTree::Vector(_))
    }

    /// Get scalar tree reference (panics if vector)
    #[inline]
    pub fn as_scalar(&self) -> &Tree {
        match self {
            EnsembleTree::Scalar(t) => t,
            EnsembleTree::Vector(_) => panic!("Expected scalar tree, got vector tree"),
        }
    }

    /// Get vector tree reference (panics if scalar)
    #[inline]
    pub fn as_vector(&self) -> &VectorTree {
        match self {
            EnsembleTree::Vector(t) => t,
            EnsembleTree::Scalar(_) => panic!("Expected vector tree, got scalar tree"),
        }
    }

    /// Try to get scalar tree reference
    #[inline]
    pub fn try_as_scalar(&self) -> Option<&Tree> {
        match self {
            EnsembleTree::Scalar(t) => Some(t),
            EnsembleTree::Vector(_) => None,
        }
    }

    /// Try to get vector tree reference
    #[inline]
    pub fn try_as_vector(&self) -> Option<&VectorTree> {
        match self {
            EnsembleTree::Vector(t) => Some(t),
            EnsembleTree::Scalar(_) => None,
        }
    }

    /// Number of nodes in the tree
    pub fn num_nodes(&self) -> usize {
        match self {
            EnsembleTree::Scalar(t) => t.num_nodes(),
            EnsembleTree::Vector(t) => t.num_nodes(),
        }
    }

    /// Number of leaves in the tree
    pub fn num_leaves(&self) -> usize {
        match self {
            EnsembleTree::Scalar(t) => t.num_leaves(),
            EnsembleTree::Vector(t) => t.num_leaves(),
        }
    }

    /// Maximum depth of the tree
    pub fn max_depth(&self) -> usize {
        match self {
            EnsembleTree::Scalar(t) => t.max_depth(),
            EnsembleTree::Vector(t) => t.max_depth(),
        }
    }

    /// Number of outputs per leaf
    ///
    /// - Scalar trees: 1
    /// - Vector trees: num_outputs
    pub fn num_outputs(&self) -> usize {
        match self {
            EnsembleTree::Scalar(_) => 1,
            EnsembleTree::Vector(t) => t.num_outputs(),
        }
    }

    /// Predict scalar value for a single sample (for scalar trees)
    ///
    /// # Arguments
    /// * `get_bin` - Function to get bin value for a feature: `|feature_idx| -> u8`
    ///
    /// # Panics
    /// Panics if called on a vector tree. Use `predict_vector` instead.
    #[inline]
    pub fn predict_scalar<F>(&self, get_bin: F) -> f32
    where
        F: Fn(usize) -> u8,
    {
        match self {
            EnsembleTree::Scalar(t) => t.predict(get_bin),
            EnsembleTree::Vector(_) => panic!("Cannot predict_scalar on vector tree"),
        }
    }

    /// Predict scalar value for a single sample (alias for predict_scalar)
    ///
    /// Provided for API compatibility with Tree. For vector trees, use `predict_vector`.
    #[inline]
    pub fn predict<F>(&self, get_bin: F) -> f32
    where
        F: Fn(usize) -> u8,
    {
        self.predict_scalar(get_bin)
    }

    /// Predict vector values for a single sample (for vector trees)
    ///
    /// # Arguments
    /// * `get_bin` - Function to get bin value for a feature: `|feature_idx| -> u8`
    ///
    /// # Panics
    /// Panics if called on a scalar tree. Use `predict_scalar` instead.
    #[inline]
    pub fn predict_vector<F>(&self, get_bin: F) -> Vec<f32>
    where
        F: Fn(usize) -> u8,
    {
        match self {
            EnsembleTree::Vector(t) => t.predict(get_bin),
            EnsembleTree::Scalar(_) => panic!("Cannot predict_vector on scalar tree"),
        }
    }

    /// Batch predict: add this tree's contribution to predictions
    ///
    /// For scalar trees, adds to single prediction per row.
    /// For vector trees, adds to all outputs per row.
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset
    /// * `predictions` - Mutable predictions buffer
    ///   - Scalar: length = num_rows
    ///   - Vector: length = num_rows * num_outputs (row-major)
    pub fn predict_batch_add(&self, dataset: &BinnedDataset, predictions: &mut [f32]) {
        match self {
            EnsembleTree::Scalar(t) => t.predict_batch_add(dataset, predictions),
            EnsembleTree::Vector(t) => {
                t.predict_batch_add(
                    |sample_idx, feature_idx| dataset.get_bin(sample_idx, feature_idx),
                    dataset.num_rows(),
                    predictions,
                );
            }
        }
    }

    /// Predict using raw feature values (for scalar trees)
    ///
    /// Uses stored split_value thresholds for decisions instead of bin indices.
    #[inline]
    pub fn predict_raw<F>(&self, get_feature: F) -> f32
    where
        F: Fn(usize) -> f64,
    {
        match self {
            EnsembleTree::Scalar(t) => t.predict_raw(get_feature),
            EnsembleTree::Vector(_) => {
                panic!("Cannot predict_raw on vector tree - use predict_raw_vector")
            }
        }
    }

    /// Predict using raw feature values (for vector trees)
    #[inline]
    pub fn predict_raw_vector<F>(&self, get_feature: F) -> Vec<f32>
    where
        F: Fn(usize) -> f64,
    {
        match self {
            EnsembleTree::Vector(t) => t.predict_raw(get_feature),
            EnsembleTree::Scalar(_) => panic!("Cannot predict_raw_vector on scalar tree"),
        }
    }

    /// Batch predict with raw features (for scalar trees)
    ///
    /// Adds tree contributions to predictions buffer.
    pub fn predict_batch_add_raw(
        &self,
        features: &[f64],
        num_features: usize,
        predictions: &mut [f32],
    ) {
        match self {
            EnsembleTree::Scalar(t) => t.predict_batch_add_raw(features, num_features, predictions),
            EnsembleTree::Vector(_) => {
                panic!("Cannot predict_batch_add_raw on vector tree - use as_vector().predict_batch_add_raw()")
            }
        }
    }

    /// Get all internal nodes for feature importance calculation
    pub fn internal_nodes(&self) -> Vec<(usize, usize, f32)> {
        // Returns (node_idx, feature_idx, sum_hessians)
        match self {
            EnsembleTree::Scalar(t) => t
                .internal_nodes()
                .filter_map(|(idx, node)| {
                    node.split_info()
                        .map(|(feat_idx, _, _, _, _, _, _)| (idx, feat_idx, node.sum_hessians))
                })
                .collect(),
            EnsembleTree::Vector(t) => t
                .internal_nodes()
                .filter_map(|(idx, node)| {
                    node.split_info().map(|(feat_idx, _, _, _, _)| {
                        // Sum hessians across all outputs for importance
                        let total_hess: f32 = node.sum_hessians().iter().sum();
                        (idx, feat_idx, total_hess)
                    })
                })
                .collect(),
        }
    }
}

impl From<Tree> for EnsembleTree {
    fn from(tree: Tree) -> Self {
        EnsembleTree::Scalar(tree)
    }
}

impl From<VectorTree> for EnsembleTree {
    fn from(tree: VectorTree) -> Self {
        EnsembleTree::Vector(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Node;

    #[test]
    fn test_ensemble_tree_scalar() {
        let tree = Tree::new(0.5, 100, 10.0, 20.0);
        let ensemble = EnsembleTree::from(tree);

        assert!(ensemble.is_scalar());
        assert!(!ensemble.is_vector());
        assert_eq!(ensemble.num_nodes(), 1);
        assert_eq!(ensemble.num_leaves(), 1);
        assert_eq!(ensemble.num_outputs(), 1);
    }

    #[test]
    fn test_ensemble_tree_vector() {
        let tree = VectorTree::new(
            vec![0.5, 1.0, 1.5],
            100,
            vec![10.0, 20.0, 30.0],
            vec![5.0, 10.0, 15.0],
        );
        let ensemble = EnsembleTree::from(tree);

        assert!(!ensemble.is_scalar());
        assert!(ensemble.is_vector());
        assert_eq!(ensemble.num_nodes(), 1);
        assert_eq!(ensemble.num_leaves(), 1);
        assert_eq!(ensemble.num_outputs(), 3);
    }

    #[test]
    fn test_scalar_prediction() {
        let tree = Tree::from_nodes(vec![
            Node::internal(0, 5, 5.0, 1, 2, true, 10.0, 0, 100, 0.0, 100.0),
            Node::leaf(1.0, 1, 50, 0.0, 50.0),
            Node::leaf(2.0, 1, 50, 0.0, 50.0),
        ]);
        let ensemble = EnsembleTree::from(tree);

        // bin=3 <= 5, go left -> leaf=1.0
        assert_eq!(ensemble.predict_scalar(|_| 3), 1.0);

        // bin=7 > 5, go right -> leaf=2.0
        assert_eq!(ensemble.predict_scalar(|_| 7), 2.0);
    }
}
