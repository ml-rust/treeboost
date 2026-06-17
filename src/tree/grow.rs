//! Tree growing with Best-First (Leaf-wise) strategy
//!
//! Uses a sorted row array with position tracking for zero-allocation partitioning.
//! This is the LightGBM approach: a single `Vec<usize>` contains all row indices,
//! partitioned in-place by node. Each node tracks its range (start, end) into this array.
//!
//! # Backend Support
//!
//! Histogram building can use different backends:
//! - Scalar (CPU): Default, uses cache-blocked approach with AVX2/NEON SIMD loads
//! - WGPU (GPU): Uses compute shaders for parallel histogram accumulation
//!
//! Backend selection is automatic based on dataset size and hardware availability.

use crate::backend::{BackendConfig, BackendSelector, BackendType, HistogramBackend};
use crate::dataset::BinnedDataset;
use crate::histogram::{EraHistograms, FusedHistogramBuilder, Histogram, NodeHistograms};
use crate::loss::LossFunction;
use crate::tree::{
    InteractionConstraints, MonotonicConstraint, Node, SplitFinder, SplitInfo, Tree,
};
use crate::Result;

/// Storage for node histograms - either standard or era-stratified
enum NodeHistogramStorage {
    Standard(NodeHistograms),
    Era {
        histograms: EraHistograms,
        per_era_totals: Vec<(f32, f32, u32)>,
    },
}
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::utils::approx_equal_relative;

/// Manages row indices during tree growth with zero-allocation partitioning.
///
/// Instead of storing row indices per node, we maintain a single sorted array
/// where each node's rows are a contiguous slice. Partitioning is done in-place
/// using the Dutch National Flag algorithm.
struct RowPartitioner {
    /// All row indices, partitioned by node (contiguous slices)
    rows: Vec<usize>,
}

impl RowPartitioner {
    /// Create a new partitioner from initial row indices
    fn new(initial_rows: Vec<usize>) -> Self {
        Self { rows: initial_rows }
    }

    /// Get a slice of rows for the given range
    #[inline]
    fn get_rows(&self, start: usize, end: usize) -> &[usize] {
        &self.rows[start..end]
    }

    /// Partition rows in-place using Dutch National Flag algorithm.
    ///
    /// After partitioning, rows[start..mid] go left, rows[mid..end] go right.
    /// Returns the midpoint index.
    #[inline]
    fn partition_in_place(
        &mut self,
        dataset: &BinnedDataset,
        start: usize,
        end: usize,
        feature_idx: usize,
        bin_threshold: u8,
    ) -> usize {
        // Get direct pointer to feature column for faster access
        let feature_column = dataset.feature_column(feature_idx);

        // Dutch National Flag: partition in-place with single pass
        let mut left = start;
        let mut right = end;

        unsafe {
            while left < right {
                // Find first row from left that should go right
                while left < right {
                    let row_idx = *self.rows.get_unchecked(left);
                    let bin = *feature_column.get_unchecked(row_idx);
                    if bin > bin_threshold {
                        break;
                    }
                    left += 1;
                }

                // Find first row from right that should go left
                while left < right {
                    right -= 1;
                    let row_idx = *self.rows.get_unchecked(right);
                    let bin = *feature_column.get_unchecked(row_idx);
                    if bin <= bin_threshold {
                        // Swap and continue
                        self.rows.swap(left, right);
                        left += 1;
                        break;
                    }
                }
            }
        }

        left // Midpoint: rows[start..left] go left, rows[left..end] go right
    }
}

/// Candidate node for splitting (zero-allocation version)
struct SplitCandidate {
    /// Node index in tree
    node_idx: usize,
    /// Start index in RowPartitioner.rows (inclusive)
    row_start: usize,
    /// End index in RowPartitioner.rows (exclusive)
    row_end: usize,
    /// Precomputed histograms (standard or era-stratified)
    histograms: Option<NodeHistogramStorage>,
    /// Best split info (if computed)
    split_info: Option<SplitInfo>,
    /// Total gradient sum for this node (used for debug validation)
    sum_gradients: f32,
    /// Total hessian sum for this node (used for debug validation)
    sum_hessians: f32,
    /// Features used in ancestors (for interaction constraints)
    ancestor_features: Vec<usize>,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain().eq(&other.gain())
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
        // Max-heap by gain
        self.gain()
            .partial_cmp(&other.gain())
            .unwrap_or(Ordering::Equal)
    }
}

impl SplitCandidate {
    fn gain(&self) -> f32 {
        self.split_info
            .as_ref()
            .map(|s| s.gain)
            .unwrap_or(f32::NEG_INFINITY)
    }
}

/// State for a split being processed in a batch
struct BatchedSplitState {
    /// Original candidate info
    _node_idx: usize,
    split_info: SplitInfo,
    parent_histograms: NodeHistogramStorage,
    ancestor_features: Vec<usize>,

    /// Smaller child info
    smaller_start: usize,
    smaller_end: usize,
    smaller_idx: usize,
    smaller_g: f32,
    smaller_h: f32,

    /// Larger child info
    larger_start: usize,
    larger_end: usize,
    larger_idx: usize,
    larger_g: f32,
    larger_h: f32,
}

/// Tree grower configuration
pub struct TreeGrower {
    /// Maximum depth of tree
    max_depth: usize,
    /// Maximum number of leaves
    max_leaves: usize,
    /// L2 regularization (lambda)
    lambda: f32,
    /// Minimum samples per leaf
    min_samples_leaf: usize,
    /// Minimum hessian sum per leaf
    min_hessian_leaf: f32,
    /// Shannon Entropy regularization weight
    entropy_weight: f32,
    /// Minimum gain to make a split
    min_gain: f32,
    /// Learning rate (shrinkage)
    learning_rate: f32,
    /// Column subsampling ratio (0.0-1.0, 1.0 = use all features)
    colsample: f32,
    /// Monotonic constraints per feature
    monotonic_constraints: Vec<MonotonicConstraint>,
    /// Feature interaction constraints
    interaction_constraints: InteractionConstraints,
    /// Backend type for histogram building (Auto = choose based on dataset size)
    backend_type: BackendType,
    /// GPU batch size for histogram dispatch (0 = no batching)
    gpu_batch_size: usize,
    /// Enable GPU subgroup operations (default: false)
    use_gpu_subgroups: bool,
    /// Enable era-based splitting (Directional Era Splitting / DES)
    /// When enabled, only accepts splits where all eras agree on direction
    era_splitting: bool,
    /// Enable noise pruning (PaloBoost-style validation)
    /// When enabled with era_splitting, Era 0 = train, Era 1 = validation
    /// Splits found on Era 0 are rejected if Era 1 gain <= threshold
    noise_pruning: bool,
    /// Noise pruning threshold for validation gain
    /// Splits are rejected if validation gain <= this threshold
    noise_pruning_threshold: f32,
    /// Enable adaptive missing value direction selection
    /// When true, evaluates both directions for missing values and selects optimal path
    use_missing_value_learning: bool,
    /// Cached backend instance (lazily initialized, reused across trees)
    cached_backend: RefCell<Option<Box<dyn HistogramBackend>>>,
}

impl Default for TreeGrower {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_leaves: 31, // 2^5 - 1
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            entropy_weight: 0.0,
            min_gain: 0.0,
            learning_rate: 0.1,
            colsample: 1.0, // Use all features by default
            monotonic_constraints: Vec::new(),
            interaction_constraints: InteractionConstraints::new(),
            backend_type: BackendType::Auto,
            gpu_batch_size: 32, // Default batch size for GPU histogram dispatch
            use_gpu_subgroups: false,
            era_splitting: false,              // Disabled by default
            noise_pruning: false,              // Disabled by default
            noise_pruning_threshold: -0.1,     // Default threshold
            use_missing_value_learning: false, // Disabled by default
            cached_backend: RefCell::new(None),
        }
    }
}

impl std::fmt::Debug for TreeGrower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeGrower")
            .field("max_depth", &self.max_depth)
            .field("max_leaves", &self.max_leaves)
            .field("lambda", &self.lambda)
            .field("min_samples_leaf", &self.min_samples_leaf)
            .field("min_hessian_leaf", &self.min_hessian_leaf)
            .field("entropy_weight", &self.entropy_weight)
            .field("min_gain", &self.min_gain)
            .field("learning_rate", &self.learning_rate)
            .field("colsample", &self.colsample)
            .field("backend_type", &self.backend_type)
            .field("gpu_batch_size", &self.gpu_batch_size)
            .field("use_gpu_subgroups", &self.use_gpu_subgroups)
            .field("era_splitting", &self.era_splitting)
            .field("noise_pruning", &self.noise_pruning)
            .field("noise_pruning_threshold", &self.noise_pruning_threshold)
            .finish()
    }
}

impl Clone for TreeGrower {
    fn clone(&self) -> Self {
        Self {
            max_depth: self.max_depth,
            max_leaves: self.max_leaves,
            lambda: self.lambda,
            min_samples_leaf: self.min_samples_leaf,
            min_hessian_leaf: self.min_hessian_leaf,
            entropy_weight: self.entropy_weight,
            min_gain: self.min_gain,
            learning_rate: self.learning_rate,
            colsample: self.colsample,
            monotonic_constraints: self.monotonic_constraints.clone(),
            interaction_constraints: self.interaction_constraints.clone(),
            backend_type: self.backend_type,
            gpu_batch_size: self.gpu_batch_size,
            use_gpu_subgroups: self.use_gpu_subgroups,
            era_splitting: self.era_splitting,
            noise_pruning: self.noise_pruning,
            noise_pruning_threshold: self.noise_pruning_threshold,
            use_missing_value_learning: self.use_missing_value_learning,
            // Reset cached backend - clone gets its own lazily initialized backend
            cached_backend: RefCell::new(None),
        }
    }
}

impl TreeGrower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_colsample(mut self, colsample: f32) -> Self {
        self.colsample = colsample;
        self
    }

    /// Set monotonic constraints for features
    ///
    /// The vector should have one entry per feature. Features beyond the
    /// vector length are treated as unconstrained.
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Set feature interaction constraints
    ///
    /// Features in the same group can interact (appear together in a tree path).
    /// Features in different groups cannot be used together.
    pub fn with_interaction_constraints(mut self, constraints: InteractionConstraints) -> Self {
        self.interaction_constraints = constraints;
        self
    }

    /// Set the backend type for histogram building
    ///
    /// - `Auto`: Automatically choose based on dataset size and hardware
    /// - `Wgpu`: Force GPU acceleration (falls back to Scalar if unavailable)
    /// - `Scalar`: Force CPU-only (uses AVX2/NEON SIMD where available)
    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Get the configured backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Set the GPU batch size for histogram dispatch
    ///
    /// When using GPU backend, multiple small histogram builds are batched together
    /// into a single GPU dispatch to amortize dispatch overhead.
    ///
    /// - Default: 32 (optimal for trees with max_depth 5-6)
    /// - Set to 0 to disable batching (for debugging or comparison)
    pub fn with_gpu_batch_size(mut self, batch_size: usize) -> Self {
        self.gpu_batch_size = batch_size;
        self
    }

    /// Enable or disable GPU subgroup operations for histogram building
    ///
    /// Subgroups can reduce atomic contention but show minimal benefit on
    /// modern NVIDIA GPUs (~1.0x speedup). May help on older AMD or Intel.
    ///
    /// - Default: false (disabled)
    pub fn with_gpu_subgroups(mut self, enabled: bool) -> Self {
        self.use_gpu_subgroups = enabled;
        self
    }

    /// Enable or disable era-based splitting (Directional Era Splitting / DES)
    ///
    /// When enabled, only accepts splits where ALL eras agree on the split direction.
    /// This filters out spurious correlations that work in some eras but not others.
    ///
    /// Requires the dataset to have era indices set via `BinnedDataset::set_era_indices`.
    ///
    /// - Default: false (disabled)
    pub fn with_era_splitting(mut self, enabled: bool) -> Self {
        self.era_splitting = enabled;
        self
    }

    /// Check if era splitting is enabled
    pub fn era_splitting(&self) -> bool {
        self.era_splitting
    }

    /// Enable or disable noise pruning (PaloBoost-style)
    ///
    /// When enabled with era_splitting, treats Era 0 as training and Era 1 as validation.
    /// Splits are found using Era 0 gain, but rejected if Era 1 gain <= 0.
    ///
    /// - Default: false (disabled)
    pub fn with_noise_pruning(mut self, enabled: bool) -> Self {
        self.noise_pruning = enabled;
        self
    }

    /// Check if noise pruning is enabled
    pub fn noise_pruning(&self) -> bool {
        self.noise_pruning
    }

    /// Set noise pruning threshold
    ///
    /// Splits are rejected if validation gain <= this threshold.
    /// - 0.0: Classic PaloBoost (only accept positive validation gains)
    /// - -0.1: Default (allows small negative gains, handles distribution shift)
    /// - -0.2: More permissive (for high temporal distribution shift)
    pub fn with_noise_pruning_threshold(mut self, threshold: f32) -> Self {
        self.noise_pruning_threshold = threshold;
        self
    }

    /// Enable adaptive missing value direction selection
    ///
    /// When enabled, evaluates both child directions for missing values at each split
    /// and selects the direction that maximizes gain. When disabled (default), missing
    /// values follow the left child direction.
    pub fn with_missing_value_learning(mut self, enabled: bool) -> Self {
        self.use_missing_value_learning = enabled;
        self
    }

    /// Pre-initialize backend with full dataset size
    ///
    /// CRITICAL: Call this BEFORE any training to select backend based on full dataset,
    /// not on train/validation split sizes. This ensures consistent backend usage
    /// throughout training (no mixing CPU/GPU which hurts accuracy).
    ///
    /// # Arguments
    /// * `full_dataset_rows` - Total rows in the FULL dataset (before any splits)
    pub fn init_backend(&self, full_dataset_rows: usize) -> Result<()> {
        self.ensure_backend(full_dataset_rows)
    }

    /// Initialize backend (called once at start of tree growing)
    fn ensure_backend(&self, num_rows: usize) -> Result<()> {
        let mut cached = self.cached_backend.borrow_mut();
        if cached.is_none() {
            let config = BackendConfig {
                preferred: self.backend_type,
                use_gpu_subgroups: self.use_gpu_subgroups,
                ..Default::default()
            };
            *cached = Some(BackendSelector::with_config(config).select(num_rows)?);
        }
        Ok(())
    }

    /// Get the backend for histogram operations
    fn backend(&self) -> std::cell::Ref<'_, Box<dyn HistogramBackend>> {
        std::cell::Ref::map(self.cached_backend.borrow(), |opt| {
            opt.as_ref().expect("Backend not initialized")
        })
    }

    /// Create a split finder with current configuration
    pub(crate) fn create_split_finder(&self) -> SplitFinder {
        SplitFinder::new()
            .with_lambda(self.lambda)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_hessian_leaf(self.min_hessian_leaf)
            .with_entropy_weight(self.entropy_weight)
            .with_min_gain(self.min_gain)
            .with_monotonic_constraints(self.monotonic_constraints.clone())
            .with_missing_value_learning(self.use_missing_value_learning)
            .with_noise_pruning(self.noise_pruning)
            .with_noise_pruning_threshold(self.noise_pruning_threshold)
    }

    /// Compute per-era totals (gradient, hessian, count) for given row indices
    fn compute_per_era_totals(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Vec<(f32, f32, u32)> {
        let num_eras = dataset.num_eras();
        let era_indices = dataset
            .era_indices()
            .expect("Dataset must have era indices for DES");

        let mut totals = vec![(0.0f32, 0.0f32, 0u32); num_eras];

        for &row in row_indices {
            let era = era_indices[row] as usize;
            totals[era].0 += gradients[row];
            totals[era].1 += hessians[row];
            totals[era].2 += 1;
        }

        totals
    }

    /// Build era histograms for given row indices using the backend
    fn build_era_histograms(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        grad_hess: &[(f32, f32)],
    ) -> EraHistograms {
        let era_indices = dataset
            .era_indices()
            .expect("Dataset must have era indices for DES");
        let num_eras = dataset.num_eras();

        let histograms_2d = self.backend().build_era_histograms(
            dataset,
            grad_hess,
            row_indices,
            era_indices,
            num_eras,
        );

        EraHistograms::from_vec(histograms_2d)
    }

    /// Compute child era histograms using subtraction trick
    fn compute_era_sibling(&self, parent: &EraHistograms, child: &EraHistograms) -> EraHistograms {
        EraHistograms::from_subtraction(parent, child)
    }

    /// Compute child per-era totals from parent and sibling
    fn compute_era_totals_sibling(
        parent_totals: &[(f32, f32, u32)],
        child_totals: &[(f32, f32, u32)],
    ) -> Vec<(f32, f32, u32)> {
        parent_totals
            .iter()
            .zip(child_totals.iter())
            .map(|(&(pg, ph, pc), &(cg, ch, cc))| (pg - cg, ph - ch, pc - cc))
            .collect()
    }

    /// Grow a tree using Best-First (Leaf-wise) strategy
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `gradients` - Gradient for each sample
    /// * `hessians` - Hessian for each sample
    pub fn grow(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<Tree> {
        // Use all rows
        let all_rows: Vec<usize> = (0..dataset.num_rows()).collect();
        self.grow_with_indices(dataset, gradients, hessians, &all_rows)
    }

    /// Grow a tree using only the specified row indices (for row subsampling)
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `gradients` - Gradient for each sample
    /// * `hessians` - Hessian for each sample
    /// * `row_indices` - Subset of row indices to use for training this tree
    pub fn grow_with_indices(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
    ) -> Result<Tree> {
        // Use era splitting path if enabled and dataset has era indices
        if self.era_splitting && dataset.has_eras() {
            return self.grow_with_indices_era(dataset, gradients, hessians, row_indices);
        }

        let num_features = dataset.num_features();
        let num_rows = row_indices.len();

        // Ensure backend is initialized
        self.ensure_backend(num_rows)?;
        let split_finder = self.create_split_finder();

        // Convert separate gradient/hessian arrays to interleaved format for backend
        // This is a one-time cost per tree, amortized over all histogram builds
        let grad_hess: Vec<(f32, f32)> = gradients
            .iter()
            .zip(hessians.iter())
            .map(|(&g, &h)| (g, h))
            .collect();

        // Generate column subsample mask (per tree)
        let feature_mask = super::generate_feature_mask(
            num_features,
            self.colsample,
            (num_rows as u64).wrapping_mul(31337),
        );

        // Compute initial sums for the subsampled rows only
        let total_gradient: f32 = row_indices.iter().map(|&i| gradients[i]).sum();
        let total_hessian: f32 = row_indices.iter().map(|&i| hessians[i]).sum();
        let initial_weight = Node::compute_leaf_weight(total_gradient, total_hessian, self.lambda);

        // Initialize tree with root leaf
        let mut tree = Tree::new(
            initial_weight * self.learning_rate,
            num_rows,
            total_gradient,
            total_hessian,
        );

        // Initialize priority queue with root candidate
        let mut candidates: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Initialize row partitioner with all rows (single allocation for entire tree growth)
        let mut partitioner = RowPartitioner::new(row_indices.to_vec());

        // Build root histograms using backend
        let root_histograms = NodeHistograms::from_vec(self.backend().build_histograms(
            dataset,
            &grad_hess,
            partitioner.get_rows(0, num_rows),
        ));

        // Compute effective feature mask for root (no ancestors)
        let root_feature_mask =
            self.compute_effective_feature_mask(&[], feature_mask.as_deref(), num_features);

        // Find best split for root
        let root_split = split_finder.find_best_split_with_features(
            &root_histograms,
            total_gradient,
            total_hessian,
            num_rows as u32,
            root_feature_mask.as_deref(),
        );

        candidates.push(SplitCandidate {
            node_idx: 0,
            row_start: 0,
            row_end: num_rows,
            histograms: Some(NodeHistogramStorage::Standard(root_histograms)),
            split_info: root_split,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
            ancestor_features: Vec::new(),
        });

        let mut num_leaves = 1;

        // Determine batch size for GPU acceleration
        // When batching is enabled, process multiple candidates per iteration
        let use_batching = self.gpu_batch_size > 1 && self.backend().is_tensor_tile();
        let effective_batch_size = if use_batching { self.gpu_batch_size } else { 1 };

        // Best-first growth loop with optional batching
        while !candidates.is_empty() && num_leaves < self.max_leaves {
            // Phase 1: Collect up to batch_size valid candidates
            let mut batch_states: Vec<BatchedSplitState> = Vec::with_capacity(effective_batch_size);

            while batch_states.len() < effective_batch_size && !candidates.is_empty() {
                let candidate = candidates.pop().unwrap();

                // Check if candidate has valid split
                let split_info = match &candidate.split_info {
                    Some(info) if info.is_valid() => *info,
                    _ => continue, // No valid split, try next candidate
                };

                // Validate gradient/hessian sums (catches histogram computation bugs)
                // Use relative error (1e-3 = 0.1%) to handle both large and small values
                debug_assert!(
                    approx_equal_relative(
                        candidate.sum_gradients,
                        split_info.left_gradient + split_info.right_gradient,
                        1e-3
                    ),
                    "Gradient sum mismatch in node {}: left({}) + right({}) != parent({})",
                    candidate.node_idx,
                    split_info.left_gradient,
                    split_info.right_gradient,
                    candidate.sum_gradients
                );
                debug_assert!(
                    approx_equal_relative(
                        candidate.sum_hessians,
                        split_info.left_hessian + split_info.right_hessian,
                        1e-3
                    ),
                    "Hessian sum mismatch in node {}: left({}) + right({}) != parent({})",
                    candidate.node_idx,
                    split_info.left_hessian,
                    split_info.right_hessian,
                    candidate.sum_hessians
                );

                // Check depth constraint
                let current_node = tree.get_node(candidate.node_idx);
                if current_node.depth >= self.max_depth {
                    continue;
                }

                // Check leaf limit
                if num_leaves >= self.max_leaves {
                    break;
                }

                // Perform the split: partition rows in-place (zero allocation!)
                let mid = partitioner.partition_in_place(
                    dataset,
                    candidate.row_start,
                    candidate.row_end,
                    split_info.feature_idx,
                    split_info.bin_threshold,
                );

                let left_start = candidate.row_start;
                let left_end = mid;
                let right_start = mid;
                let right_end = candidate.row_end;
                let left_count = left_end - left_start;
                let right_count = right_end - right_start;

                // Create child leaf nodes
                let left_weight = Node::compute_leaf_weight(
                    split_info.left_gradient,
                    split_info.left_hessian,
                    self.lambda,
                );
                let right_weight = Node::compute_leaf_weight(
                    split_info.right_gradient,
                    split_info.right_hessian,
                    self.lambda,
                );

                let child_depth = current_node.depth + 1;

                let left_node = Node::leaf(
                    left_weight * self.learning_rate,
                    child_depth,
                    left_count,
                    split_info.left_gradient,
                    split_info.left_hessian,
                );
                let right_node = Node::leaf(
                    right_weight * self.learning_rate,
                    child_depth,
                    right_count,
                    split_info.right_gradient,
                    split_info.right_hessian,
                );

                let left_idx = tree.add_node(left_node);
                let right_idx = tree.add_node(right_node);

                // Get the actual split value from bin boundaries
                let split_value =
                    dataset.get_split_value(split_info.feature_idx, split_info.bin_threshold);

                // Convert current leaf to internal node
                let current_node = tree.get_node_mut(candidate.node_idx);
                current_node.node_type = crate::tree::NodeType::Internal {
                    feature_idx: split_info.feature_idx,
                    bin_threshold: split_info.bin_threshold,
                    split_value,
                    left_child: left_idx,
                    right_child: right_idx,
                    default_left: split_info.default_left,
                    gain: split_info.gain,
                };

                num_leaves += 1; // One leaf becomes two (net +1)

                // Skip histogram building if we've hit the leaf limit
                if num_leaves >= self.max_leaves {
                    break;
                }

                // Determine smaller child for histogram subtraction trick
                let parent_histograms = candidate.histograms.unwrap();
                let (
                    smaller_start,
                    smaller_end,
                    smaller_idx,
                    larger_start,
                    larger_end,
                    larger_idx,
                    smaller_g,
                    smaller_h,
                    larger_g,
                    larger_h,
                ) = if left_count <= right_count {
                    (
                        left_start,
                        left_end,
                        left_idx,
                        right_start,
                        right_end,
                        right_idx,
                        split_info.left_gradient,
                        split_info.left_hessian,
                        split_info.right_gradient,
                        split_info.right_hessian,
                    )
                } else {
                    (
                        right_start,
                        right_end,
                        right_idx,
                        left_start,
                        left_end,
                        left_idx,
                        split_info.right_gradient,
                        split_info.right_hessian,
                        split_info.left_gradient,
                        split_info.left_hessian,
                    )
                };

                // Store state for batched histogram building
                batch_states.push(BatchedSplitState {
                    _node_idx: candidate.node_idx,
                    split_info,
                    parent_histograms,
                    ancestor_features: candidate.ancestor_features,
                    smaller_start,
                    smaller_end,
                    smaller_idx,
                    smaller_g,
                    smaller_h,
                    larger_start,
                    larger_end,
                    larger_idx,
                    larger_g,
                    larger_h,
                });
            }

            // Phase 2: Build histograms (batched for GPU, individual for CPU)
            if batch_states.is_empty() {
                break;
            }

            let all_smaller_histograms: Vec<Vec<Histogram>> =
                if use_batching && batch_states.len() > 1 {
                    // Batched GPU dispatch: build all histograms in one kernel launch
                    let row_slices: Vec<&[usize]> = batch_states
                        .iter()
                        .map(|state| partitioner.get_rows(state.smaller_start, state.smaller_end))
                        .collect();
                    self.backend()
                        .build_histograms_batched(dataset, &grad_hess, &row_slices)
                } else {
                    // Individual builds (CPU or single histogram)
                    batch_states
                        .iter()
                        .map(|state| {
                            self.backend().build_histograms(
                                dataset,
                                &grad_hess,
                                partitioner.get_rows(state.smaller_start, state.smaller_end),
                            )
                        })
                        .collect()
                };

            // Phase 3: Process results - compute larger children, find splits, push to heap
            for (state, smaller_hists) in batch_states.into_iter().zip(all_smaller_histograms) {
                let smaller_histograms = NodeHistograms::from_vec(smaller_hists);

                // Extract parent histograms (must be Standard variant in this path)
                let parent_hists = match state.parent_histograms {
                    NodeHistogramStorage::Standard(h) => h,
                    NodeHistogramStorage::Era { .. } => {
                        unreachable!("Era histograms in standard path")
                    }
                };

                // Compute larger child histogram using subtraction trick
                let larger_histograms =
                    NodeHistograms::from_vec(self.backend().build_histograms_sibling(
                        &parent_hists.histograms,
                        &smaller_histograms.histograms,
                    ));

                let smaller_count = state.smaller_end - state.smaller_start;
                let larger_count = state.larger_end - state.larger_start;

                // Compute child ancestor features (parent's ancestors + current split feature)
                let mut child_ancestors = state.ancestor_features;
                child_ancestors.push(state.split_info.feature_idx);

                // Compute effective feature mask for children (interaction + column subsampling)
                let child_feature_mask = self.compute_effective_feature_mask(
                    &child_ancestors,
                    feature_mask.as_deref(),
                    num_features,
                );

                // Find splits for children
                let smaller_split = split_finder.find_best_split_with_features(
                    &smaller_histograms,
                    state.smaller_g,
                    state.smaller_h,
                    smaller_count as u32,
                    child_feature_mask.as_deref(),
                );
                let larger_split = split_finder.find_best_split_with_features(
                    &larger_histograms,
                    state.larger_g,
                    state.larger_h,
                    larger_count as u32,
                    child_feature_mask.as_deref(),
                );

                candidates.push(SplitCandidate {
                    node_idx: state.smaller_idx,
                    row_start: state.smaller_start,
                    row_end: state.smaller_end,
                    histograms: Some(NodeHistogramStorage::Standard(smaller_histograms)),
                    split_info: smaller_split,
                    sum_gradients: state.smaller_g,
                    sum_hessians: state.smaller_h,
                    ancestor_features: child_ancestors.clone(),
                });

                candidates.push(SplitCandidate {
                    node_idx: state.larger_idx,
                    row_start: state.larger_start,
                    row_end: state.larger_end,
                    histograms: Some(NodeHistogramStorage::Standard(larger_histograms)),
                    split_info: larger_split,
                    sum_gradients: state.larger_g,
                    sum_hessians: state.larger_h,
                    ancestor_features: child_ancestors,
                });
            }
        }

        Ok(tree)
    }

    /// Grow a tree using era-stratified histograms (Directional Era Splitting / DES)
    ///
    /// Only accepts splits where ALL eras agree on the split direction.
    /// This filters out spurious correlations that work in some eras but not others.
    ///
    /// # Arguments
    /// * `dataset` - Binned training data (must have era indices set)
    /// * `gradients` - Gradient for each sample
    /// * `hessians` - Hessian for each sample
    /// * `row_indices` - Subset of row indices to use for training this tree
    fn grow_with_indices_era(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
    ) -> Result<Tree> {
        debug_assert!(dataset.has_eras(), "Dataset must have era indices for DES");

        let num_features = dataset.num_features();
        let num_rows = row_indices.len();

        // Ensure backend is initialized for GPU era histograms
        self.ensure_backend(num_rows)?;

        // Convert separate gradient/hessian arrays to interleaved format for backend
        let grad_hess: Vec<(f32, f32)> = gradients
            .iter()
            .zip(hessians.iter())
            .map(|(&g, &h)| (g, h))
            .collect();

        let split_finder = self.create_split_finder();

        // Generate column subsample mask (per tree)
        let feature_mask = super::generate_feature_mask(
            num_features,
            self.colsample,
            (num_rows as u64).wrapping_mul(31337),
        );

        // Compute initial sums for the subsampled rows only
        let total_gradient: f32 = row_indices.iter().map(|&i| gradients[i]).sum();
        let total_hessian: f32 = row_indices.iter().map(|&i| hessians[i]).sum();
        let initial_weight = Node::compute_leaf_weight(total_gradient, total_hessian, self.lambda);

        // Initialize tree with root leaf
        let mut tree = Tree::new(
            initial_weight * self.learning_rate,
            num_rows,
            total_gradient,
            total_hessian,
        );

        // Initialize priority queue with root candidate
        let mut candidates: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Initialize row partitioner
        let mut partitioner = RowPartitioner::new(row_indices.to_vec());

        // Build root era histograms using backend
        let root_era_histograms =
            self.build_era_histograms(dataset, partitioner.get_rows(0, num_rows), &grad_hess);

        // Compute per-era totals for root
        let root_per_era_totals = self.compute_per_era_totals(
            dataset,
            partitioner.get_rows(0, num_rows),
            gradients,
            hessians,
        );

        // Compute effective feature mask for root
        let root_feature_mask =
            self.compute_effective_feature_mask(&[], feature_mask.as_deref(), num_features);

        // Find best split for root using era-aware method
        let root_split = split_finder.find_best_split_with_eras_and_features(
            &root_era_histograms,
            &root_per_era_totals,
            root_feature_mask.as_deref(),
        );

        candidates.push(SplitCandidate {
            node_idx: 0,
            row_start: 0,
            row_end: num_rows,
            histograms: Some(NodeHistogramStorage::Era {
                histograms: root_era_histograms,
                per_era_totals: root_per_era_totals,
            }),
            split_info: root_split,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
            ancestor_features: Vec::new(),
        });

        let mut num_leaves = 1;

        // Best-first growth loop (non-batched for era splitting - GPU kernels not yet implemented)
        while let Some(candidate) = candidates.pop() {
            if num_leaves >= self.max_leaves {
                break;
            }

            // Check if candidate has valid split
            let split_info = match &candidate.split_info {
                Some(info) if info.is_valid() => *info,
                _ => continue,
            };

            // Validate gradient/hessian sums (catches histogram computation bugs)
            // Use relative error (1e-3 = 0.1%) to handle both large and small values
            debug_assert!(
                approx_equal_relative(
                    candidate.sum_gradients,
                    split_info.left_gradient + split_info.right_gradient,
                    1e-3
                ),
                "Gradient sum mismatch in node {}: left({}) + right({}) != parent({})",
                candidate.node_idx,
                split_info.left_gradient,
                split_info.right_gradient,
                candidate.sum_gradients
            );
            debug_assert!(
                approx_equal_relative(
                    candidate.sum_hessians,
                    split_info.left_hessian + split_info.right_hessian,
                    1e-3
                ),
                "Hessian sum mismatch in node {}: left({}) + right({}) != parent({})",
                candidate.node_idx,
                split_info.left_hessian,
                split_info.right_hessian,
                candidate.sum_hessians
            );

            // Check depth constraint
            let current_node = tree.get_node(candidate.node_idx);
            if current_node.depth >= self.max_depth {
                continue;
            }

            // Perform the split: partition rows in-place
            let mid = partitioner.partition_in_place(
                dataset,
                candidate.row_start,
                candidate.row_end,
                split_info.feature_idx,
                split_info.bin_threshold,
            );

            let left_start = candidate.row_start;
            let left_end = mid;
            let right_start = mid;
            let right_end = candidate.row_end;
            let left_count = left_end - left_start;
            let right_count = right_end - right_start;

            // Create child leaf nodes
            let left_weight = Node::compute_leaf_weight(
                split_info.left_gradient,
                split_info.left_hessian,
                self.lambda,
            );
            let right_weight = Node::compute_leaf_weight(
                split_info.right_gradient,
                split_info.right_hessian,
                self.lambda,
            );

            let child_depth = current_node.depth + 1;

            let left_node = Node::leaf(
                left_weight * self.learning_rate,
                child_depth,
                left_count,
                split_info.left_gradient,
                split_info.left_hessian,
            );
            let right_node = Node::leaf(
                right_weight * self.learning_rate,
                child_depth,
                right_count,
                split_info.right_gradient,
                split_info.right_hessian,
            );

            let left_idx = tree.add_node(left_node);
            let right_idx = tree.add_node(right_node);

            // Get the actual split value from bin boundaries
            let split_value =
                dataset.get_split_value(split_info.feature_idx, split_info.bin_threshold);

            // Convert current leaf to internal node
            let current_node = tree.get_node_mut(candidate.node_idx);
            current_node.node_type = crate::tree::NodeType::Internal {
                feature_idx: split_info.feature_idx,
                bin_threshold: split_info.bin_threshold,
                split_value,
                left_child: left_idx,
                right_child: right_idx,
                default_left: split_info.default_left,
                gain: split_info.gain,
            };

            num_leaves += 1;

            // Skip histogram building if we've hit the leaf limit
            if num_leaves >= self.max_leaves {
                break;
            }

            // Extract parent era histograms
            let (parent_era_histograms, parent_per_era_totals) = match candidate.histograms.unwrap()
            {
                NodeHistogramStorage::Era {
                    histograms,
                    per_era_totals,
                } => (histograms, per_era_totals),
                NodeHistogramStorage::Standard(_) => {
                    unreachable!("Standard histograms in era path")
                }
            };

            // Determine smaller child for histogram subtraction trick
            let (
                smaller_start,
                smaller_end,
                smaller_idx,
                larger_start,
                larger_end,
                larger_idx,
                smaller_g,
                smaller_h,
                larger_g,
                larger_h,
            ) = if left_count <= right_count {
                (
                    left_start,
                    left_end,
                    left_idx,
                    right_start,
                    right_end,
                    right_idx,
                    split_info.left_gradient,
                    split_info.left_hessian,
                    split_info.right_gradient,
                    split_info.right_hessian,
                )
            } else {
                (
                    right_start,
                    right_end,
                    right_idx,
                    left_start,
                    left_end,
                    left_idx,
                    split_info.right_gradient,
                    split_info.right_hessian,
                    split_info.left_gradient,
                    split_info.left_hessian,
                )
            };

            // Build smaller child era histograms using backend
            let smaller_era_histograms = self.build_era_histograms(
                dataset,
                partitioner.get_rows(smaller_start, smaller_end),
                &grad_hess,
            );

            // Compute larger child via subtraction trick
            let larger_era_histograms =
                self.compute_era_sibling(&parent_era_histograms, &smaller_era_histograms);

            // Compute per-era totals for children
            let smaller_per_era_totals = self.compute_per_era_totals(
                dataset,
                partitioner.get_rows(smaller_start, smaller_end),
                gradients,
                hessians,
            );
            let larger_per_era_totals =
                Self::compute_era_totals_sibling(&parent_per_era_totals, &smaller_per_era_totals);

            // Compute child ancestor features
            let mut child_ancestors = candidate.ancestor_features;
            child_ancestors.push(split_info.feature_idx);

            // Compute effective feature mask for children
            let child_feature_mask = self.compute_effective_feature_mask(
                &child_ancestors,
                feature_mask.as_deref(),
                num_features,
            );

            // Find splits for children using era-aware method
            let smaller_split = split_finder.find_best_split_with_eras_and_features(
                &smaller_era_histograms,
                &smaller_per_era_totals,
                child_feature_mask.as_deref(),
            );
            let larger_split = split_finder.find_best_split_with_eras_and_features(
                &larger_era_histograms,
                &larger_per_era_totals,
                child_feature_mask.as_deref(),
            );

            candidates.push(SplitCandidate {
                node_idx: smaller_idx,
                row_start: smaller_start,
                row_end: smaller_end,
                histograms: Some(NodeHistogramStorage::Era {
                    histograms: smaller_era_histograms,
                    per_era_totals: smaller_per_era_totals,
                }),
                split_info: smaller_split,
                sum_gradients: smaller_g,
                sum_hessians: smaller_h,
                ancestor_features: child_ancestors.clone(),
            });

            candidates.push(SplitCandidate {
                node_idx: larger_idx,
                row_start: larger_start,
                row_end: larger_end,
                histograms: Some(NodeHistogramStorage::Era {
                    histograms: larger_era_histograms,
                    per_era_totals: larger_per_era_totals,
                }),
                split_info: larger_split,
                sum_gradients: larger_g,
                sum_hessians: larger_h,
                ancestor_features: child_ancestors,
            });
        }

        Ok(tree)
    }

    /// Grow a tree with fused gradient+histogram computation for the root
    ///
    /// This method eliminates cache pollution by computing gradients AND building
    /// the root histogram in a single pass over the data. The gradients are then
    /// reused for child histogram building.
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `row_indices` - Subset of row indices to use for training
    /// * `targets` - Target values for all samples
    /// * `predictions` - Current predictions for all samples
    /// * `loss_fn` - Loss function for gradient/hessian computation
    /// * `gradients` - Output buffer for gradients (will be written)
    /// * `hessians` - Output buffer for hessians (will be written)
    ///
    /// # Performance
    ///
    /// This method provides ~40-80% speedup over separate gradient+histogram computation
    /// on large datasets (500k+ rows) by eliminating cache pollution.
    #[allow(clippy::too_many_arguments)]
    pub fn grow_fused(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        targets: &[f32],
        predictions: &[f32],
        loss_fn: &dyn LossFunction,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<Tree> {
        let num_features = dataset.num_features();
        let num_rows = row_indices.len();

        // Ensure backend is initialized
        self.ensure_backend(num_rows)?;
        let fused_builder = FusedHistogramBuilder::new();
        let split_finder = self.create_split_finder();

        // Generate column subsample mask (per tree)
        let feature_mask = super::generate_feature_mask(
            num_features,
            self.colsample,
            (num_rows as u64).wrapping_mul(31337),
        );

        // Initialize row partitioner
        let mut partitioner = RowPartitioner::new(row_indices.to_vec());

        // FUSED: Build root histograms AND compute gradients in single pass
        // This is the key optimization - eliminates cache pollution
        let fused_result = fused_builder.build_root(
            dataset,
            partitioner.get_rows(0, num_rows),
            targets,
            predictions,
            loss_fn,
            gradients,
            hessians,
        );

        let total_gradient = fused_result.total_gradient;
        let total_hessian = fused_result.total_hessian;
        let root_histograms = fused_result.histograms;

        // Convert pre-computed gradients/hessians to interleaved format for backend
        // This is done once after the fused root computation
        let grad_hess: Vec<(f32, f32)> = gradients
            .iter()
            .zip(hessians.iter())
            .map(|(&g, &h)| (g, h))
            .collect();

        let initial_weight = Node::compute_leaf_weight(total_gradient, total_hessian, self.lambda);

        // Initialize tree with root leaf
        let mut tree = Tree::new(
            initial_weight * self.learning_rate,
            num_rows,
            total_gradient,
            total_hessian,
        );

        // Initialize priority queue with root candidate
        let mut candidates: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Compute effective feature mask for root
        let root_feature_mask =
            self.compute_effective_feature_mask(&[], feature_mask.as_deref(), num_features);

        // Find best split for root
        let root_split = split_finder.find_best_split_with_features(
            &root_histograms,
            total_gradient,
            total_hessian,
            num_rows as u32,
            root_feature_mask.as_deref(),
        );

        candidates.push(SplitCandidate {
            node_idx: 0,
            row_start: 0,
            row_end: num_rows,
            histograms: Some(NodeHistogramStorage::Standard(root_histograms)),
            split_info: root_split,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
            ancestor_features: Vec::new(),
        });

        let mut num_leaves = 1;

        // Determine batch size for GPU acceleration
        let use_batching = self.gpu_batch_size > 1 && self.backend().is_tensor_tile();
        let effective_batch_size = if use_batching { self.gpu_batch_size } else { 1 };

        // Best-first growth loop with optional batching (uses pre-computed gradients)
        while !candidates.is_empty() && num_leaves < self.max_leaves {
            // Phase 1: Collect up to batch_size valid candidates
            let mut batch_states: Vec<BatchedSplitState> = Vec::with_capacity(effective_batch_size);

            while batch_states.len() < effective_batch_size && !candidates.is_empty() {
                let candidate = candidates.pop().unwrap();

                let split_info = match &candidate.split_info {
                    Some(info) if info.is_valid() => *info,
                    _ => continue,
                };

                // Validate gradient/hessian sums (catches histogram computation bugs)
                // Use relative error (1e-3 = 0.1%) to handle both large and small values
                debug_assert!(
                    approx_equal_relative(
                        candidate.sum_gradients,
                        split_info.left_gradient + split_info.right_gradient,
                        1e-3
                    ),
                    "Gradient sum mismatch in node {}: left({}) + right({}) != parent({})",
                    candidate.node_idx,
                    split_info.left_gradient,
                    split_info.right_gradient,
                    candidate.sum_gradients
                );
                debug_assert!(
                    approx_equal_relative(
                        candidate.sum_hessians,
                        split_info.left_hessian + split_info.right_hessian,
                        1e-3
                    ),
                    "Hessian sum mismatch in node {}: left({}) + right({}) != parent({})",
                    candidate.node_idx,
                    split_info.left_hessian,
                    split_info.right_hessian,
                    candidate.sum_hessians
                );

                let current_node = tree.get_node(candidate.node_idx);
                if current_node.depth >= self.max_depth {
                    continue;
                }

                if num_leaves >= self.max_leaves {
                    break;
                }

                // Partition rows in-place
                let mid = partitioner.partition_in_place(
                    dataset,
                    candidate.row_start,
                    candidate.row_end,
                    split_info.feature_idx,
                    split_info.bin_threshold,
                );

                let left_start = candidate.row_start;
                let left_end = mid;
                let right_start = mid;
                let right_end = candidate.row_end;
                let left_count = left_end - left_start;
                let right_count = right_end - right_start;

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    continue;
                }

                // Create child leaf nodes
                let left_weight = Node::compute_leaf_weight(
                    split_info.left_gradient,
                    split_info.left_hessian,
                    self.lambda,
                );
                let right_weight = Node::compute_leaf_weight(
                    split_info.right_gradient,
                    split_info.right_hessian,
                    self.lambda,
                );

                let child_depth = current_node.depth + 1;

                let left_node = Node::leaf(
                    left_weight * self.learning_rate,
                    child_depth,
                    left_count,
                    split_info.left_gradient,
                    split_info.left_hessian,
                );
                let right_node = Node::leaf(
                    right_weight * self.learning_rate,
                    child_depth,
                    right_count,
                    split_info.right_gradient,
                    split_info.right_hessian,
                );

                let left_idx = tree.add_node(left_node);
                let right_idx = tree.add_node(right_node);

                // Get the actual split value from bin boundaries
                let split_value =
                    dataset.get_split_value(split_info.feature_idx, split_info.bin_threshold);

                // Convert current leaf to internal node
                let current_node = tree.get_node_mut(candidate.node_idx);
                current_node.node_type = crate::tree::NodeType::Internal {
                    feature_idx: split_info.feature_idx,
                    bin_threshold: split_info.bin_threshold,
                    split_value,
                    left_child: left_idx,
                    right_child: right_idx,
                    default_left: split_info.default_left,
                    gain: split_info.gain,
                };

                num_leaves += 1;

                if num_leaves >= self.max_leaves {
                    break;
                }

                // Build child histograms using the pre-computed gradients
                let parent_histograms = candidate.histograms.unwrap();

                let (
                    smaller_start,
                    smaller_end,
                    smaller_idx,
                    larger_start,
                    larger_end,
                    larger_idx,
                    smaller_g,
                    smaller_h,
                    larger_g,
                    larger_h,
                ) = if left_count <= right_count {
                    (
                        left_start,
                        left_end,
                        left_idx,
                        right_start,
                        right_end,
                        right_idx,
                        split_info.left_gradient,
                        split_info.left_hessian,
                        split_info.right_gradient,
                        split_info.right_hessian,
                    )
                } else {
                    (
                        right_start,
                        right_end,
                        right_idx,
                        left_start,
                        left_end,
                        left_idx,
                        split_info.right_gradient,
                        split_info.right_hessian,
                        split_info.left_gradient,
                        split_info.left_hessian,
                    )
                };

                batch_states.push(BatchedSplitState {
                    _node_idx: candidate.node_idx,
                    split_info,
                    parent_histograms,
                    ancestor_features: candidate.ancestor_features,
                    smaller_start,
                    smaller_end,
                    smaller_idx,
                    smaller_g,
                    smaller_h,
                    larger_start,
                    larger_end,
                    larger_idx,
                    larger_g,
                    larger_h,
                });
            }

            // Phase 2: Build histograms (batched for GPU, individual for CPU)
            if batch_states.is_empty() {
                break;
            }

            let all_smaller_histograms: Vec<Vec<Histogram>> =
                if use_batching && batch_states.len() > 1 {
                    let row_slices: Vec<&[usize]> = batch_states
                        .iter()
                        .map(|state| partitioner.get_rows(state.smaller_start, state.smaller_end))
                        .collect();
                    self.backend()
                        .build_histograms_batched(dataset, &grad_hess, &row_slices)
                } else {
                    batch_states
                        .iter()
                        .map(|state| {
                            self.backend().build_histograms(
                                dataset,
                                &grad_hess,
                                partitioner.get_rows(state.smaller_start, state.smaller_end),
                            )
                        })
                        .collect()
                };

            // Phase 3: Process results
            for (state, smaller_hists) in batch_states.into_iter().zip(all_smaller_histograms) {
                let smaller_histograms = NodeHistograms::from_vec(smaller_hists);

                // Extract parent histograms (must be Standard variant in grow_fused path)
                let parent_hists = match state.parent_histograms {
                    NodeHistogramStorage::Standard(h) => h,
                    NodeHistogramStorage::Era { .. } => {
                        unreachable!("Era histograms in grow_fused path")
                    }
                };

                let larger_histograms =
                    NodeHistograms::from_vec(self.backend().build_histograms_sibling(
                        &parent_hists.histograms,
                        &smaller_histograms.histograms,
                    ));

                let smaller_count = state.smaller_end - state.smaller_start;
                let larger_count = state.larger_end - state.larger_start;

                let mut child_ancestors = state.ancestor_features;
                child_ancestors.push(state.split_info.feature_idx);

                let child_feature_mask = self.compute_effective_feature_mask(
                    &child_ancestors,
                    feature_mask.as_deref(),
                    num_features,
                );

                let smaller_split = split_finder.find_best_split_with_features(
                    &smaller_histograms,
                    state.smaller_g,
                    state.smaller_h,
                    smaller_count as u32,
                    child_feature_mask.as_deref(),
                );
                let larger_split = split_finder.find_best_split_with_features(
                    &larger_histograms,
                    state.larger_g,
                    state.larger_h,
                    larger_count as u32,
                    child_feature_mask.as_deref(),
                );

                candidates.push(SplitCandidate {
                    node_idx: state.smaller_idx,
                    row_start: state.smaller_start,
                    row_end: state.smaller_end,
                    histograms: Some(NodeHistogramStorage::Standard(smaller_histograms)),
                    split_info: smaller_split,
                    sum_gradients: state.smaller_g,
                    sum_hessians: state.smaller_h,
                    ancestor_features: child_ancestors.clone(),
                });

                candidates.push(SplitCandidate {
                    node_idx: state.larger_idx,
                    row_start: state.larger_start,
                    row_end: state.larger_end,
                    histograms: Some(NodeHistogramStorage::Standard(larger_histograms)),
                    split_info: larger_split,
                    sum_gradients: state.larger_g,
                    sum_hessians: state.larger_h,
                    ancestor_features: child_ancestors,
                });
            }
        }

        Ok(tree)
    }

    /// Compute effective feature mask combining interaction constraints and column subsampling
    ///
    /// Returns None if all features are allowed, Some(mask) otherwise
    fn compute_effective_feature_mask(
        &self,
        ancestor_features: &[usize],
        colsample_mask: Option<&[usize]>,
        num_features: usize,
    ) -> Option<Vec<usize>> {
        // Get interaction-allowed features
        let interaction_allowed = if self.interaction_constraints.is_empty() {
            None
        } else {
            Some(
                self.interaction_constraints
                    .allowed_features(ancestor_features, num_features),
            )
        };

        // Combine with column subsampling mask
        match (interaction_allowed, colsample_mask) {
            (None, None) => None, // No constraints
            (Some(allowed), None) => Some(allowed),
            (None, Some(mask)) => Some(mask.to_vec()),
            (Some(allowed), Some(mask)) => {
                // Intersection of both constraints
                let allowed_set: std::collections::HashSet<_> = allowed.into_iter().collect();
                let combined: Vec<usize> = mask
                    .iter()
                    .copied()
                    .filter(|f| allowed_set.contains(f))
                    .collect();
                if combined.is_empty() {
                    // Edge case: no features allowed - return the interaction allowed set
                    // to let the algorithm gracefully stop
                    Some(
                        self.interaction_constraints
                            .allowed_features(ancestor_features, num_features),
                    )
                } else {
                    Some(combined)
                }
            }
        }
    }
}

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

        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32).sin()).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_grow_single_leaf() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = vec![0.0; 100]; // No gradient = no split benefit
        let hessians: Vec<f32> = vec![1.0; 100];

        let grower = TreeGrower::new().with_max_depth(3).with_min_gain(1000.0); // Very high min gain = no splits

        let tree = grower.grow(&dataset, &gradients, &hessians).unwrap();

        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.num_nodes(), 1);
    }

    #[test]
    fn test_grow_with_splits() {
        let dataset = create_test_dataset(1000, 5);

        // Create gradients that should encourage splitting
        let gradients: Vec<f32> = (0..1000)
            .map(|i| if i < 500 { -1.0 } else { 1.0 })
            .collect();
        let hessians: Vec<f32> = vec![1.0; 1000];

        let grower = TreeGrower::new()
            .with_max_depth(4)
            .with_max_leaves(16)
            .with_min_samples_leaf(10)
            .with_learning_rate(0.1);

        let tree = grower.grow(&dataset, &gradients, &hessians).unwrap();

        // Should have multiple nodes
        assert!(tree.num_nodes() > 1);
        assert!(tree.num_leaves() > 1);
        assert!(tree.max_depth() <= 4);
    }

    #[test]
    fn test_max_depth_constraint() {
        let dataset = create_test_dataset(1000, 3);
        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let hessians: Vec<f32> = vec![1.0; 1000];

        let grower = TreeGrower::new().with_max_depth(2).with_max_leaves(100);

        let tree = grower.grow(&dataset, &gradients, &hessians).unwrap();

        assert!(tree.max_depth() <= 2);
    }

    #[test]
    fn test_max_leaves_constraint() {
        let dataset = create_test_dataset(1000, 3);
        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let hessians: Vec<f32> = vec![1.0; 1000];

        let grower = TreeGrower::new().with_max_depth(10).with_max_leaves(5);

        let tree = grower.grow(&dataset, &gradients, &hessians).unwrap();

        assert!(tree.num_leaves() <= 5);
    }
}
