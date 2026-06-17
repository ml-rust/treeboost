//! Full CUDA tree building pipeline.
//!
//! Keeps all data on GPU throughout tree building for minimal PCIe transfers.
//! Uses level-wise tree growth for optimal GPU batching.
//!
//! Key optimizations:
//! - GPU-resident indices: no CPU<->GPU transfers between levels
//! - Histogram subtraction: only build smaller child, compute larger via subtraction
//! - True batched histograms: all nodes in one kernel launch
//! - Double-buffering for indices: swap buffers instead of copying
//! - Only histograms are read back (needed for CPU split finding)

use super::device::CudaDevice;
use super::kernels::{HistogramKernel, NodeRange};
use super::partition::{NodeSplit, PartitionKernel};
use crate::dataset::BinnedDataset;
use crate::histogram::Histogram;
use crate::tree::{Node, NodeType, SplitInfo, Tree};
use crate::utils::approx_equal_relative;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

/// Full CUDA tree builder using level-wise growth with GPU-resident data.
pub struct FullCudaTreeBuilder {
    device: Arc<CudaDevice>,
    histogram_kernel: HistogramKernel,
    partition_kernel: PartitionKernel,

    // GPU-resident buffers (persisted across tree building)
    d_indices_a: Option<CudaSlice<u32>>,
    d_indices_b: Option<CudaSlice<u32>>,
    indices_capacity: usize,
}

impl FullCudaTreeBuilder {
    /// Create a new full CUDA tree builder.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            histogram_kernel: HistogramKernel::new(Arc::clone(&device)),
            partition_kernel: PartitionKernel::new(Arc::clone(&device)),
            device,
            d_indices_a: None,
            d_indices_b: None,
            indices_capacity: 0,
        }
    }

    /// Ensure indices buffers are large enough.
    fn ensure_indices_capacity(&mut self, capacity: usize, max_depth: usize) {
        let multiplier = 1 << (max_depth / 2 + 3);
        let required = capacity * multiplier;
        if self.indices_capacity < required || self.d_indices_a.is_none() {
            self.d_indices_a = Some(self.device.alloc_zeros(required));
            self.d_indices_b = Some(self.device.alloc_zeros(required));
            self.indices_capacity = required;
        }
    }

    /// Build a tree using level-wise GPU pipeline with histogram subtraction.
    #[allow(clippy::too_many_arguments)]
    pub fn build_tree(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
        max_depth: usize,
        max_leaves: usize,
        lambda: f32,
        min_samples_leaf: usize,
        _min_gain: f32, // Unused on GPU: min_gain filtering is handled by SplitFinder on the host side
        learning_rate: f32,
        split_finder: &crate::tree::SplitFinder,
        colsample: f32,
        seed: u64,
    ) -> Tree {
        let num_rows = row_indices.len();
        let num_features = dataset.num_features();

        // Handle edge cases: empty data or no features
        if num_rows == 0 || num_features == 0 {
            return Tree::new(0.0, num_rows, 0.0, 0.0);
        }

        // Generate column subsample mask (per tree)
        let feature_mask = crate::tree::generate_feature_mask(
            num_features,
            colsample,
            seed.wrapping_mul(31337).wrapping_add(num_rows as u64),
        );

        // Cache bins and grad/hess on GPU (once per tree)
        let bins = dataset.as_row_major();
        self.histogram_kernel.ensure_bins_cached(bins);
        self.histogram_kernel
            .ensure_grad_hess_cached(gradients, hessians);
        self.partition_kernel.ensure_bins_cached(bins);

        // Initial indices -> GPU
        let initial_indices: Vec<u32> = row_indices.iter().map(|&r| r as u32).collect();
        self.ensure_indices_capacity(num_rows, max_depth);
        self.device
            .htod_copy_into(&initial_indices, self.d_indices_a.as_mut().unwrap());

        // Compute initial sums
        let total_gradient: f32 = row_indices.iter().map(|&i| gradients[i]).sum();
        let total_hessian: f32 = row_indices.iter().map(|&i| hessians[i]).sum();
        let initial_weight = -total_gradient / (total_hessian + lambda);

        // Initialize tree
        let mut tree = Tree::new(
            initial_weight * learning_rate,
            num_rows,
            total_gradient,
            total_hessian,
        );

        // Level-wise node tracking with histogram subtraction support
        #[derive(Debug, Clone)]
        struct LevelNode {
            node_idx: usize,
            start: u32,
            count: u32,
            sum_gradients: f32,
            sum_hessians: f32,
            // For histogram subtraction
            parent_hist_idx: Option<usize>, // Index into parent_histograms
        }

        // Parent histograms from previous level (for subtraction)
        let mut parent_histograms: Vec<Vec<Histogram>> = Vec::new();

        let mut current_level = vec![LevelNode {
            node_idx: 0,
            start: 0,
            count: num_rows as u32,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
            parent_hist_idx: None,
        }];

        let mut num_leaves = 1;
        let mut use_buffer_a = true;

        // Process level by level
        for depth in 0..max_depth {
            if current_level.is_empty() || num_leaves >= max_leaves {
                break;
            }

            // Determine which nodes need GPU histogram vs subtraction
            // Group siblings by parent and identify smaller one
            let mut nodes_needing_gpu_hist: Vec<usize> = Vec::new();
            let mut subtraction_pairs: Vec<(usize, usize)> = Vec::new(); // (larger_idx, smaller_idx)

            if depth == 0 {
                // Root level - all nodes need GPU histogram
                nodes_needing_gpu_hist = (0..current_level.len()).collect();
            } else {
                // Group nodes by parent
                // Nodes from same parent are consecutive (left then right)
                let mut i = 0;
                while i < current_level.len() {
                    let node_i = &current_level[i];

                    // Check if next node is sibling (same parent)
                    if i + 1 < current_level.len() {
                        let node_j = &current_level[i + 1];
                        if node_i.parent_hist_idx == node_j.parent_hist_idx
                            && node_i.parent_hist_idx.is_some()
                        {
                            // These are siblings - determine smaller
                            if node_i.count <= node_j.count {
                                nodes_needing_gpu_hist.push(i);
                                subtraction_pairs.push((i + 1, i)); // j is larger, i is smaller
                            } else {
                                nodes_needing_gpu_hist.push(i + 1);
                                subtraction_pairs.push((i, i + 1)); // i is larger, j is smaller
                            }
                            i += 2;
                            continue;
                        }
                    }
                    // Single node (no sibling) - needs GPU histogram
                    nodes_needing_gpu_hist.push(i);
                    i += 1;
                }
            }

            // Build node ranges only for nodes needing GPU histogram
            let node_ranges: Vec<NodeRange> = nodes_needing_gpu_hist
                .iter()
                .map(|&idx| NodeRange {
                    start: current_level[idx].start,
                    count: current_level[idx].count,
                })
                .collect();

            // Get current indices buffer
            let d_indices = if use_buffer_a {
                self.d_indices_a.as_ref().unwrap()
            } else {
                self.d_indices_b.as_ref().unwrap()
            };

            // Build histograms only for nodes that need GPU computation
            let gpu_histograms = if !node_ranges.is_empty() {
                self.histogram_kernel
                    .build_histograms_gpu(d_indices, &node_ranges, num_features)
            } else {
                Vec::new()
            };

            // Build full histogram array for all nodes
            // Map GPU histogram index back to node index
            let mut gpu_hist_map: Vec<Option<usize>> = vec![None; current_level.len()];
            for (gpu_idx, &node_idx) in nodes_needing_gpu_hist.iter().enumerate() {
                gpu_hist_map[node_idx] = Some(gpu_idx);
            }

            // Compute all histograms (GPU + subtraction)
            let mut all_histograms: Vec<Vec<Histogram>> = vec![Vec::new(); current_level.len()];

            // First, assign GPU-computed histograms
            for (gpu_idx, &node_idx) in nodes_needing_gpu_hist.iter().enumerate() {
                all_histograms[node_idx] = gpu_histograms[gpu_idx].clone();
            }

            // Then, compute subtraction histograms
            for &(larger_idx, smaller_idx) in &subtraction_pairs {
                let node = &current_level[larger_idx];
                if let Some(parent_idx) = node.parent_hist_idx {
                    let parent_hist = &parent_histograms[parent_idx];
                    let smaller_hist = &all_histograms[smaller_idx];

                    // Larger = Parent - Smaller
                    all_histograms[larger_idx] = parent_hist
                        .iter()
                        .zip(smaller_hist.iter())
                        .map(|(p, s)| Histogram::from_subtraction(p, s))
                        .collect();
                }
            }

            // Find best splits for all nodes
            let mut splits_and_nodes: Vec<(SplitInfo, usize)> = Vec::new();

            for (i, (node, hists)) in current_level.iter().zip(all_histograms.iter()).enumerate() {
                if hists.is_empty() {
                    continue;
                }
                // Convert Vec<Histogram> to NodeHistograms for SplitFinder
                let node_histograms = crate::histogram::NodeHistograms::from_vec(hists.clone());
                if let Some(split) = split_finder.find_best_split_with_features(
                    &node_histograms,
                    node.sum_gradients,
                    node.sum_hessians,
                    node.count,
                    feature_mask.as_deref(),
                ) {
                    splits_and_nodes.push((split, i));
                }
            }

            if splits_and_nodes.is_empty() {
                break;
            }

            // Store current histograms as parent histograms for next level
            // Only store histograms for nodes that will split
            let mut new_parent_histograms: Vec<Vec<Histogram>> = Vec::new();
            let mut parent_hist_indices: Vec<usize> = Vec::new(); // Maps split index to parent_hist index

            for (_, node_idx) in &splits_and_nodes {
                parent_hist_indices.push(new_parent_histograms.len());
                new_parent_histograms.push(all_histograms[*node_idx].clone());
            }

            // Prepare partition splits
            let node_splits: Vec<NodeSplit> = splits_and_nodes
                .iter()
                .map(|(split, node_idx)| {
                    let node = &current_level[*node_idx];
                    NodeSplit {
                        input_start: node.start,
                        input_count: node.count,
                        split_feature: split.feature_idx as u32,
                        split_threshold: split.bin_threshold as u32,
                    }
                })
                .collect();

            // Fused partition on GPU
            let partition_results = if use_buffer_a {
                let d_input = self.d_indices_a.as_ref().unwrap();
                let d_output = self.d_indices_b.as_mut().unwrap();
                self.partition_kernel
                    .partition_fused(d_input, d_output, &node_splits, num_features)
            } else {
                let d_input = self.d_indices_b.as_ref().unwrap();
                let d_output = self.d_indices_a.as_mut().unwrap();
                self.partition_kernel
                    .partition_fused(d_input, d_output, &node_splits, num_features)
            };

            // Build next level and update tree
            let mut next_level: Vec<LevelNode> = Vec::new();

            for (split_idx, ((split, orig_node_idx), &(output_start, left_cnt, right_cnt))) in
                splits_and_nodes
                    .iter()
                    .zip(partition_results.iter())
                    .enumerate()
            {
                if num_leaves >= max_leaves {
                    break;
                }

                let node = &current_level[*orig_node_idx];

                // Validate gradient/hessian sums (catches histogram computation bugs)
                // Use relative error (1e-3 = 0.1%) to handle both large and small values
                debug_assert!(
                    approx_equal_relative(
                        node.sum_gradients,
                        split.left_gradient + split.right_gradient,
                        1e-3
                    ),
                    "Gradient sum mismatch in node {}: left({}) + right({}) != parent({})",
                    node.node_idx,
                    split.left_gradient,
                    split.right_gradient,
                    node.sum_gradients
                );
                debug_assert!(
                    approx_equal_relative(
                        node.sum_hessians,
                        split.left_hessian + split.right_hessian,
                        1e-3
                    ),
                    "Hessian sum mismatch in node {}: left({}) + right({}) != parent({})",
                    node.node_idx,
                    split.left_hessian,
                    split.right_hessian,
                    node.sum_hessians
                );

                let current_node = tree.get_node(node.node_idx);
                let child_depth = current_node.depth + 1;

                let left_weight = -split.left_gradient / (split.left_hessian + lambda);
                let right_weight = -split.right_gradient / (split.right_hessian + lambda);

                let left_count = left_cnt as usize;
                let right_count = right_cnt as usize;

                let left_node = Node::leaf(
                    left_weight * learning_rate,
                    child_depth,
                    left_count,
                    split.left_gradient,
                    split.left_hessian,
                );
                let right_node = Node::leaf(
                    right_weight * learning_rate,
                    child_depth,
                    right_count,
                    split.right_gradient,
                    split.right_hessian,
                );

                let left_idx = tree.add_node(left_node);
                let right_idx = tree.add_node(right_node);

                let split_value = dataset.get_split_value(split.feature_idx, split.bin_threshold);

                let current_node = tree.get_node_mut(node.node_idx);
                current_node.node_type = NodeType::Internal {
                    feature_idx: split.feature_idx,
                    bin_threshold: split.bin_threshold,
                    split_value,
                    left_child: left_idx,
                    right_child: right_idx,
                    default_left: split.default_left,
                    gain: split.gain,
                };

                num_leaves += 1;

                // Add children to next level with parent tracking
                if depth + 1 < max_depth && num_leaves < max_leaves {
                    let left_start = output_start;
                    let right_start = output_start + node.count;
                    let parent_hist_idx = Some(parent_hist_indices[split_idx]);

                    // Add left child first, then right (keeps siblings consecutive)
                    if left_count >= min_samples_leaf {
                        next_level.push(LevelNode {
                            node_idx: left_idx,
                            start: left_start,
                            count: left_cnt,
                            sum_gradients: split.left_gradient,
                            sum_hessians: split.left_hessian,
                            parent_hist_idx,
                        });
                    }
                    if right_count >= min_samples_leaf {
                        next_level.push(LevelNode {
                            node_idx: right_idx,
                            start: right_start,
                            count: right_cnt,
                            sum_gradients: split.right_gradient,
                            sum_hessians: split.right_hessian,
                            parent_hist_idx,
                        });
                    }
                }
            }

            // Update parent histograms for next level
            parent_histograms = new_parent_histograms;

            // Toggle buffer for next level
            use_buffer_a = !use_buffer_a;
            current_level = next_level;
        }

        tree
    }
}
