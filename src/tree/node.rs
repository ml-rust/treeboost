//! Tree node structure

use rkyv::{Archive, Deserialize, Serialize};

/// Node type: internal (split) or leaf
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum NodeType {
    /// Internal node with split
    Internal {
        /// Feature index to split on
        feature_idx: usize,
        /// Bin threshold (samples with bin <= threshold go left)
        bin_threshold: u8,
        /// Actual split value (for raw prediction without binning)
        /// Samples with value <= split_value go left
        split_value: f64,
        /// Left child node index
        left_child: usize,
        /// Right child node index
        right_child: usize,
        /// Default direction for missing values (bin 0)
        /// If true, missing values go left; if false, they go right
        default_left: bool,
        /// Split gain (used for post-pruning)
        gain: f32,
    },
    /// Leaf node with prediction value
    Leaf {
        /// Prediction value (leaf weight)
        value: f32,
    },
}

/// Tree node
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct Node {
    /// Node type (internal or leaf)
    pub node_type: NodeType,
    /// Depth in tree (root = 0)
    pub depth: usize,
    /// Number of samples in this node
    pub num_samples: usize,
    /// Sum of gradients
    pub sum_gradients: f32,
    /// Sum of hessians
    pub sum_hessians: f32,
}

impl Node {
    /// Create a new leaf node
    pub fn leaf(value: f32, depth: usize, num_samples: usize, sum_g: f32, sum_h: f32) -> Self {
        Self {
            node_type: NodeType::Leaf { value },
            depth,
            num_samples,
            sum_gradients: sum_g,
            sum_hessians: sum_h,
        }
    }

    /// Create a new internal node
    ///
    /// All 11 parameters represent distinct, fundamental node properties.
    /// Grouping them into sub-structs would increase verbosity without improving clarity.
    ///
    /// # Parameter Order
    ///
    /// Parameters are ordered semantically:
    /// 1. **Split-specific**: `feature_idx`, `bin_threshold`, `split_value`, `left_child`,
    ///    `right_child`, `default_left`, `gain` - Define the split decision and its quality
    /// 2. **Tree metadata**: `depth`, `num_samples` - Position in tree and node size
    /// 3. **Gradient statistics**: `sum_g`, `sum_h` - Used for leaf weight computation
    ///
    /// The `gain` parameter groups with split-specific fields because it measures split quality
    /// and is used by post-pruning to decide whether to collapse weak splits.
    #[allow(clippy::too_many_arguments)]
    pub fn internal(
        feature_idx: usize,
        bin_threshold: u8,
        split_value: f64,
        left_child: usize,
        right_child: usize,
        default_left: bool,
        gain: f32,
        depth: usize,
        num_samples: usize,
        sum_g: f32,
        sum_h: f32,
    ) -> Self {
        Self {
            node_type: NodeType::Internal {
                feature_idx,
                bin_threshold,
                split_value,
                left_child,
                right_child,
                default_left,
                gain,
            },
            depth,
            num_samples,
            sum_gradients: sum_g,
            sum_hessians: sum_h,
        }
    }

    /// Check if this is a leaf node
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self.node_type, NodeType::Leaf { .. })
    }

    /// Get leaf value, returning None if this is not a leaf node
    #[inline]
    pub fn leaf_value(&self) -> Option<f32> {
        match self.node_type {
            NodeType::Leaf { value } => Some(value),
            NodeType::Internal { .. } => None,
        }
    }

    /// Get split info, returning None if this is not an internal node
    /// Returns (feature_idx, bin_threshold, split_value, left_child, right_child, default_left, gain)
    #[inline]
    pub fn split_info(&self) -> Option<(usize, u8, f64, usize, usize, bool, f32)> {
        match self.node_type {
            NodeType::Internal {
                feature_idx,
                bin_threshold,
                split_value,
                left_child,
                right_child,
                default_left,
                gain,
            } => Some((
                feature_idx,
                bin_threshold,
                split_value,
                left_child,
                right_child,
                default_left,
                gain,
            )),
            NodeType::Leaf { .. } => None,
        }
    }

    /// Compute optimal leaf weight using Newton step
    ///
    /// weight = -sum_gradients / (sum_hessians + lambda)
    #[inline]
    pub fn compute_leaf_weight(sum_g: f32, sum_h: f32, lambda: f32) -> f32 {
        -sum_g / (sum_h + lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_node() {
        let node = Node::leaf(0.5, 2, 100, 10.0, 20.0);

        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(0.5));
        assert_eq!(node.split_info(), None);
        assert_eq!(node.depth, 2);
        assert_eq!(node.num_samples, 100);
    }

    #[test]
    fn test_internal_node() {
        let node = Node::internal(3, 128, 5.5, 1, 2, true, 10.5, 1, 200, 15.0, 30.0);

        assert!(!node.is_leaf());
        assert_eq!(node.leaf_value(), None);
        let split = node.split_info();
        assert!(split.is_some());
        let (f, t, v, l, r, d, g) = split.unwrap();
        assert_eq!(f, 3);
        assert_eq!(t, 128);
        assert!((v - 5.5).abs() < 1e-10);
        assert_eq!(l, 1);
        assert_eq!(r, 2);
        assert!(d);
        assert!((g - 10.5).abs() < 1e-6);
    }

    #[test]
    fn test_leaf_weight() {
        // weight = -sum_g / (sum_h + lambda)
        let weight = Node::compute_leaf_weight(-10.0, 20.0, 0.0);
        assert!((weight - 0.5).abs() < 1e-6);

        let weight_reg = Node::compute_leaf_weight(-10.0, 20.0, 10.0);
        assert!((weight_reg - (10.0 / 30.0)).abs() < 1e-6);
    }
}
