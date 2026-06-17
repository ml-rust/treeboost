//! Full GPU Tree Building Pipeline
//!
//! Keeps all data on GPU throughout tree building:
//! - Bins, gradients, hessians uploaded once
//! - Histograms built on GPU (with histogram subtraction optimization)
//! - Split finding on CPU (fast for 256 bins)
//! - Row partition on GPU
//! - Only final tree structure downloaded
//!
//! Note: For WGPU, Hybrid mode is generally faster than Full GPU mode due to
//! WGPU's high dispatch overhead (~1-2ms per dispatch). Full GPU mode is
//! primarily useful for CUDA where dispatch overhead is much lower (~10-100μs).

use super::device::GpuDevice;
use super::kernels::HistogramKernel;
use super::partition::{NodeSplit, PartitionKernel, PartitionResult};
use crate::dataset::BinnedDataset;
use crate::histogram::Histogram;
use crate::tree::{Node, NodeType, SplitInfo, Tree};
use crate::utils::approx_equal_relative;
use std::sync::Arc;
use wgpu::Buffer;

/// Full GPU tree builder
pub struct FullGpuTreeBuilder {
    device: Arc<GpuDevice>,
    histogram_kernel: HistogramKernel,
    partition_kernel: PartitionKernel,

    // Persistent GPU buffers (reused across trees)
    bins_buffer: Option<Buffer>,
    indices_buffer: Option<Buffer>,
    indices_buffer_alt: Option<Buffer>, // Double buffer for partition output

    // Cached data for histogram building and partition
    cached_bins_row_major: Vec<u8>,
    cached_bins_packed: Vec<u32>,
    cached_grad_hess: Vec<(f32, f32)>,
    cached_indices: Vec<u32>,

    // Row index array for best-first (mutable during tree building)
    row_indices_array: Vec<u32>,
}

impl FullGpuTreeBuilder {
    pub fn new(device: Arc<GpuDevice>) -> Self {
        let histogram_kernel = HistogramKernel::new(Arc::clone(&device));
        let partition_kernel = PartitionKernel::new(Arc::clone(&device));

        Self {
            device,
            histogram_kernel,
            partition_kernel,
            bins_buffer: None,
            indices_buffer: None,
            indices_buffer_alt: None,
            cached_bins_row_major: Vec::new(),
            cached_bins_packed: Vec::new(),
            cached_grad_hess: Vec::new(),
            cached_indices: Vec::new(),
            row_indices_array: Vec::new(),
        }
    }

    /// Build a tree using fully-GPU pipeline with histogram subtraction
    ///
    /// All operations (histogram, split finding, partition) happen on GPU.
    /// Only the final tree structure is downloaded.
    ///
    /// Key optimization: Histogram subtraction - only build GPU histograms for
    /// the smaller sibling, compute larger via: parent - smaller (~50% less GPU work).
    // reason: kernel/training entry point with many parameters
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
        let _num_bins = 256usize;

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

        // Upload data to GPU (only once per tree or cached across trees)
        self.upload_data(dataset, gradients, hessians, row_indices);

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
            row_start: usize,
            row_count: usize,
            sum_gradients: f32,
            sum_hessians: f32,
            // For histogram subtraction
            parent_hist_idx: Option<usize>, // Index into parent_histograms
        }

        // Parent histograms from previous level (for subtraction)
        let mut parent_histograms: Vec<Vec<Histogram>> = Vec::new();

        let mut current_level = vec![LevelNode {
            node_idx: 0,
            row_start: 0,
            row_count: num_rows,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
            parent_hist_idx: None,
        }];

        let mut num_leaves = 1;
        let mut use_alt_buffer = false;

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
                            if node_i.row_count <= node_j.row_count {
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

            // Step 1: Build histograms only for nodes needing GPU computation
            let node_ranges: Vec<(usize, usize)> = nodes_needing_gpu_hist
                .iter()
                .map(|&idx| (current_level[idx].row_start, current_level[idx].row_count))
                .collect();

            // Use native histogram method to avoid double conversion
            let gpu_histograms = if !node_ranges.is_empty() {
                self.build_histograms_gpu_native(&node_ranges, num_features)
            } else {
                Vec::new()
            };

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

            // Step 2: Find best splits for all nodes (CPU - fast for 256 bins)
            // Using CPU split finding avoids histogram format conversion and GPU upload overhead
            let mut valid_splits: Vec<(usize, SplitInfo, usize)> = Vec::new();

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
                    node.row_count as u32,
                    feature_mask.as_deref(),
                ) {
                    valid_splits.push((i, split, node.node_idx));
                }
            }

            if valid_splits.is_empty() {
                break;
            }

            // Store current histograms as parent histograms for next level
            // Only store histograms for nodes that will split
            let mut new_parent_histograms: Vec<Vec<Histogram>> = Vec::new();
            let mut parent_hist_indices: Vec<usize> = Vec::new(); // Maps split index to parent_hist index

            for (orig_idx, _, _) in &valid_splits {
                parent_hist_indices.push(new_parent_histograms.len());
                new_parent_histograms.push(all_histograms[*orig_idx].clone());
            }

            // Step 3: Partition all nodes with valid splits (GPU)
            let node_splits: Vec<NodeSplit> = valid_splits
                .iter()
                .map(|(orig_idx, split, _)| {
                    let node = &current_level[*orig_idx];
                    NodeSplit {
                        input_start: node.row_start as u32,
                        input_count: node.row_count as u32,
                        output_left_start: node.row_start as u32,
                        output_right_start: node.row_start as u32,
                        split_feature: split.feature_idx as u32,
                        split_threshold: split.bin_threshold as u32,
                        _padding: [0; 2],
                    }
                })
                .collect();

            // Execute partition on GPU (results stay on GPU)
            let partition_results =
                self.partition_gpu(&node_splits, num_features, num_rows, use_alt_buffer);
            use_alt_buffer = !use_alt_buffer;

            // Build next level and update tree structure
            let mut next_level: Vec<LevelNode> = Vec::new();
            let mut write_offset = 0usize;

            for (split_idx, ((_orig_idx, split, tree_node_idx), result)) in valid_splits
                .iter()
                .zip(partition_results.iter())
                .enumerate()
            {
                if num_leaves >= max_leaves {
                    break;
                }

                let current_node = tree.get_node(*tree_node_idx);
                let child_depth = current_node.depth + 1;

                let left_weight = -split.left_gradient / (split.left_hessian + lambda);
                let right_weight = -split.right_gradient / (split.right_hessian + lambda);

                let left_count = result.left_count as usize;
                let right_count = result.right_count as usize;

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

                let current_node = tree.get_node_mut(*tree_node_idx);
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

                // Track children for next level with parent histogram tracking
                let left_start = write_offset;
                write_offset += left_count;
                let right_start = write_offset;
                write_offset += right_count;

                let parent_hist_idx = Some(parent_hist_indices[split_idx]);

                if depth + 1 < max_depth && num_leaves < max_leaves {
                    // Add left child first, then right (keeps siblings consecutive)
                    if left_count >= min_samples_leaf {
                        next_level.push(LevelNode {
                            node_idx: left_idx,
                            row_start: left_start,
                            row_count: left_count,
                            sum_gradients: split.left_gradient,
                            sum_hessians: split.left_hessian,
                            parent_hist_idx,
                        });
                    }
                    if right_count >= min_samples_leaf {
                        next_level.push(LevelNode {
                            node_idx: right_idx,
                            row_start: right_start,
                            row_count: right_count,
                            sum_gradients: split.right_gradient,
                            sum_hessians: split.right_hessian,
                            parent_hist_idx,
                        });
                    }
                }
            }

            // Update parent histograms for next level
            parent_histograms = new_parent_histograms;

            current_level = next_level;
        }

        tree
    }

    /// Build a tree using GPU best-first strategy with histogram subtraction
    ///
    /// This is the optimized approach:
    /// - CPU priority queue (fast, sequential)
    /// - GPU histogram building (for smaller child only)
    /// - CPU histogram subtraction (fast arithmetic)
    /// - GPU partition with in-place CPU array management
    // reason: kernel/training entry point with many parameters
    #[allow(clippy::too_many_arguments)]
    pub fn build_tree_best_first(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
        max_depth: usize,
        max_leaves: usize,
        lambda: f32,
        min_samples_leaf: usize,
        min_gain: f32,
        learning_rate: f32,
        split_finder: &crate::tree::SplitFinder,
        colsample: f32,
        seed: u64,
    ) -> Tree {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

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

        // Upload data once
        self.upload_data(dataset, gradients, hessians, row_indices);

        // Initialize row indices array for in-place partitioning
        self.row_indices_array = row_indices.iter().map(|&r| r as u32).collect();

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

        // Build root histogram using GPU
        let root_row_indices: Vec<usize> =
            self.row_indices_array.iter().map(|&r| r as usize).collect();
        let root_histograms = self.histogram_kernel.build_histograms(
            &self.cached_bins_row_major,
            &self.cached_grad_hess,
            &root_row_indices,
            dataset.num_rows(),
            num_features,
        );

        // Find best split for root
        let node_histograms = crate::histogram::NodeHistograms::from_vec(root_histograms.clone());
        let root_split = split_finder.find_best_split_with_features(
            &node_histograms,
            total_gradient,
            total_hessian,
            num_rows as u32,
            feature_mask.as_deref(),
        );

        // Priority queue entry
        #[derive(Clone)]
        struct SplitCandidate {
            gain: f32,
            node_idx: usize,
            row_start: usize,
            row_end: usize,
            split_info: SplitInfo,
            histograms: Vec<Histogram>,
            /// Total gradient sum for this node (used for debug validation)
            sum_gradients: f32,
            /// Total hessian sum for this node (used for debug validation)
            sum_hessians: f32,
        }

        impl PartialEq for SplitCandidate {
            fn eq(&self, other: &Self) -> bool {
                self.gain == other.gain
            }
        }
        impl Eq for SplitCandidate {}
        impl PartialOrd for SplitCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for SplitCandidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.gain
                    .partial_cmp(&other.gain)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<SplitCandidate> = BinaryHeap::new();
        let mut num_leaves = 1usize;

        // Add root to heap if splittable
        if let Some(split) = root_split {
            if split.gain > min_gain {
                heap.push(SplitCandidate {
                    gain: split.gain,
                    node_idx: 0,
                    row_start: 0,
                    row_end: num_rows,
                    split_info: split,
                    histograms: root_histograms,
                    sum_gradients: total_gradient,
                    sum_hessians: total_hessian,
                });
            }
        }

        // Best-first expansion
        while let Some(candidate) = heap.pop() {
            if num_leaves >= max_leaves {
                break;
            }

            let current_depth = tree.get_node(candidate.node_idx).depth;
            if current_depth >= max_depth {
                continue;
            }

            let split = &candidate.split_info;

            // Validate gradient/hessian sums (catches histogram computation bugs)
            // Use relative error (1e-3 = 0.1%) to handle both large and small values
            debug_assert!(
                approx_equal_relative(
                    candidate.sum_gradients,
                    split.left_gradient + split.right_gradient,
                    1e-3
                ),
                "Gradient sum mismatch in node {}: left({}) + right({}) != parent({})",
                candidate.node_idx,
                split.left_gradient,
                split.right_gradient,
                candidate.sum_gradients
            );
            debug_assert!(
                approx_equal_relative(
                    candidate.sum_hessians,
                    split.left_hessian + split.right_hessian,
                    1e-3
                ),
                "Hessian sum mismatch in node {}: left({}) + right({}) != parent({})",
                candidate.node_idx,
                split.left_hessian,
                split.right_hessian,
                candidate.sum_hessians
            );

            // Partition rows in-place
            let (left_count, right_count) = self.partition_in_place(
                candidate.row_start,
                candidate.row_end,
                split.feature_idx,
                split.bin_threshold,
                num_features,
            );

            let left_start = candidate.row_start;
            let left_end = left_start + left_count;
            let right_start = left_end;
            let right_end = candidate.row_end;

            // Create child nodes
            let left_weight = -split.left_gradient / (split.left_hessian + lambda);
            let right_weight = -split.right_gradient / (split.right_hessian + lambda);

            let child_depth = current_depth + 1;

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

            // Convert parent to internal node
            let split_value = dataset.get_split_value(split.feature_idx, split.bin_threshold);
            let current_node = tree.get_node_mut(candidate.node_idx);
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

            // Determine smaller child for histogram subtraction trick
            let (smaller_idx, smaller_start, smaller_end, smaller_g, smaller_h) =
                if left_count <= right_count {
                    (
                        left_idx,
                        left_start,
                        left_end,
                        split.left_gradient,
                        split.left_hessian,
                    )
                } else {
                    (
                        right_idx,
                        right_start,
                        right_end,
                        split.right_gradient,
                        split.right_hessian,
                    )
                };

            let (larger_idx, larger_start, larger_end, larger_g, larger_h) =
                if left_count <= right_count {
                    (
                        right_idx,
                        right_start,
                        right_end,
                        split.right_gradient,
                        split.right_hessian,
                    )
                } else {
                    (
                        left_idx,
                        left_start,
                        left_end,
                        split.left_gradient,
                        split.left_hessian,
                    )
                };

            // Build histogram for SMALLER child only (GPU)
            if child_depth < max_depth && num_leaves < max_leaves {
                let smaller_count = smaller_end - smaller_start;
                let larger_count = larger_end - larger_start;

                if smaller_count >= min_samples_leaf {
                    let smaller_row_indices: Vec<usize> = self.row_indices_array
                        [smaller_start..smaller_end]
                        .iter()
                        .map(|&r| r as usize)
                        .collect();

                    let smaller_histograms = self.histogram_kernel.build_histograms(
                        &self.cached_bins_row_major,
                        &self.cached_grad_hess,
                        &smaller_row_indices,
                        dataset.num_rows(),
                        num_features,
                    );

                    let node_histograms =
                        crate::histogram::NodeHistograms::from_vec(smaller_histograms.clone());
                    if let Some(smaller_split) = split_finder.find_best_split_with_features(
                        &node_histograms,
                        smaller_g,
                        smaller_h,
                        smaller_count as u32,
                        feature_mask.as_deref(),
                    ) {
                        if smaller_split.gain > min_gain {
                            heap.push(SplitCandidate {
                                gain: smaller_split.gain,
                                node_idx: smaller_idx,
                                row_start: smaller_start,
                                row_end: smaller_end,
                                split_info: smaller_split,
                                histograms: smaller_histograms.clone(),
                                sum_gradients: smaller_g,
                                sum_hessians: smaller_h,
                            });
                        }
                    }

                    // Compute LARGER histogram via subtraction (CPU - fast)
                    if larger_count >= min_samples_leaf {
                        let larger_histograms: Vec<Histogram> = candidate
                            .histograms
                            .iter()
                            .zip(smaller_histograms.iter())
                            .map(|(parent, smaller)| Histogram::from_subtraction(parent, smaller))
                            .collect();

                        let node_histograms =
                            crate::histogram::NodeHistograms::from_vec(larger_histograms.clone());
                        if let Some(larger_split) = split_finder.find_best_split_with_features(
                            &node_histograms,
                            larger_g,
                            larger_h,
                            larger_count as u32,
                            feature_mask.as_deref(),
                        ) {
                            if larger_split.gain > min_gain {
                                heap.push(SplitCandidate {
                                    gain: larger_split.gain,
                                    node_idx: larger_idx,
                                    row_start: larger_start,
                                    row_end: larger_end,
                                    split_info: larger_split,
                                    histograms: larger_histograms,
                                    sum_gradients: larger_g,
                                    sum_hessians: larger_h,
                                });
                            }
                        }
                    }
                }
            }
        }

        tree
    }

    /// In-place partition of row indices
    fn partition_in_place(
        &mut self,
        start: usize,
        end: usize,
        feature_idx: usize,
        bin_threshold: u8,
        num_features: usize,
    ) -> (usize, usize) {
        let mut left = start;
        let mut right = end;

        while left < right {
            let row = self.row_indices_array[left] as usize;
            let bin = self.cached_bins_row_major[row * num_features + feature_idx];

            if bin <= bin_threshold {
                left += 1;
            } else {
                right -= 1;
                self.row_indices_array.swap(left, right);
            }
        }

        let left_count = left - start;
        let right_count = end - left;
        (left_count, right_count)
    }

    fn upload_data(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
    ) {
        // Cache bins (row-major) for histogram building
        self.cached_bins_row_major = dataset.as_row_major().to_vec();

        // Cache grad_hess for histogram building
        self.cached_grad_hess = gradients
            .iter()
            .zip(hessians.iter())
            .map(|(&g, &h)| (g, h))
            .collect();

        // Cache indices
        self.cached_indices = row_indices.iter().map(|&r| r as u32).collect();

        // Cache bins (row-major, packed as u32) for partition kernel
        self.cached_bins_packed = self
            .cached_bins_row_major
            .chunks(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &b)| acc | ((b as u32) << (i * 8)))
            })
            .collect();

        let bins_size = (self.cached_bins_packed.len() * 4) as u64;
        if self.bins_buffer.is_none() || self.bins_buffer.as_ref().unwrap().size() < bins_size {
            self.bins_buffer = Some(self.device.create_storage_buffer("bins", bins_size, false));
        }
        self.device
            .write_buffer(self.bins_buffer.as_ref().unwrap(), &self.cached_bins_packed);

        // Upload indices
        let indices_size = (self.cached_indices.len() * 4) as u64;
        if self.indices_buffer.is_none()
            || self.indices_buffer.as_ref().unwrap().size() < indices_size
        {
            self.indices_buffer = Some(self.device.create_storage_buffer(
                "indices",
                indices_size,
                true,
            ));
            self.indices_buffer_alt = Some(self.device.create_storage_buffer(
                "indices_alt",
                indices_size,
                true,
            ));
        }
        self.device
            .write_buffer(self.indices_buffer.as_ref().unwrap(), &self.cached_indices);
    }

    /// Build histograms on GPU and return as Histogram structs directly.
    /// This avoids the flat f32 conversion when using CPU split finding.
    fn build_histograms_gpu_native(
        &self,
        nodes: &[(usize, usize)], // (row_start, row_count) for each node
        num_features: usize,
    ) -> Vec<Vec<Histogram>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Build row index slices for batched histogram building
        let row_slices: Vec<Vec<usize>> = nodes
            .iter()
            .map(|&(start, count)| {
                self.cached_indices[start..start + count]
                    .iter()
                    .map(|&r| r as usize)
                    .collect()
            })
            .collect();
        let row_slice_refs: Vec<&[usize]> = row_slices.iter().map(|v| v.as_slice()).collect();

        // Use histogram kernel batched API - returns Vec<Vec<Histogram>> directly
        self.histogram_kernel.build_histograms_batched(
            &self.cached_bins_row_major,
            &self.cached_grad_hess,
            &row_slice_refs,
            self.cached_bins_row_major.len() / num_features,
            num_features,
        )
    }

    fn partition_gpu(
        &self,
        node_splits: &[NodeSplit],
        num_features: usize,
        num_rows: usize,
        _use_alt: bool,
    ) -> Vec<PartitionResult> {
        // Use cached bins and indices (no GPU readback needed)
        // This avoids PCIe transfers for bins which never change
        self.partition_kernel.partition_batched(
            &self.cached_bins_packed,
            &self.cached_indices,
            node_splits,
            num_features,
            num_rows,
        )
    }
}
