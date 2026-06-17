//! CUDA partition kernel for level-wise tree building.
//!
//! Optimized with:
//! - Buffer caching (bins stay on GPU)
//! - True batched partitioning (all nodes in one kernel)
//! - GPU-resident mode: indices stay on GPU between levels
//! - Double-buffering for zero-copy level transitions

use super::device::CudaDevice;
use super::kernels::CacheKey;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// CUDA source for partition kernels.
/// Includes both single-node and batched versions.
const PARTITION_KERNEL_SOURCE: &str = r#"
// Single node partition with atomic counters
extern "C" __global__ void partition_atomic(
    const unsigned char* __restrict__ bins,
    const unsigned int* __restrict__ input_indices,
    unsigned int* __restrict__ left_indices,
    unsigned int* __restrict__ right_indices,
    unsigned int* __restrict__ counters,
    unsigned int num_indices,
    unsigned int num_features,
    unsigned int split_feature,
    unsigned int split_threshold
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    unsigned int row = input_indices[idx];
    unsigned int bin = bins[row * num_features + split_feature];

    if (bin <= split_threshold) {
        unsigned int pos = atomicAdd(&counters[0], 1);
        left_indices[pos] = row;
    } else {
        unsigned int pos = atomicAdd(&counters[1], 1);
        right_indices[pos] = row;
    }
}

// Batched partition: process multiple nodes in parallel
// Grid: (num_nodes, row_tiles)
// Each block handles one node's partition for a tile of rows
extern "C" __global__ void partition_batched(
    const unsigned char* __restrict__ bins,
    const unsigned int* __restrict__ input_indices,      // Concatenated indices for all nodes
    unsigned int* __restrict__ output_indices,           // Interleaved left/right output
    unsigned int* __restrict__ counters,                 // [node_idx * 2 + 0/1] for left/right
    const unsigned int* __restrict__ node_starts,        // Start offset for each node
    const unsigned int* __restrict__ node_counts,        // Row count for each node
    const unsigned int* __restrict__ split_features,     // Split feature for each node
    const unsigned int* __restrict__ split_thresholds,   // Split threshold for each node
    const unsigned int* __restrict__ output_starts,      // Output start offset for each node
    unsigned int num_features,
    unsigned int max_node_rows                           // Max output size per node
) {
    unsigned int node_idx = blockIdx.x;
    unsigned int tile_idx = blockIdx.y;
    unsigned int rows_per_tile = blockDim.x;

    unsigned int node_start = node_starts[node_idx];
    unsigned int node_count = node_counts[node_idx];
    unsigned int split_feature = split_features[node_idx];
    unsigned int split_threshold = split_thresholds[node_idx];
    unsigned int output_start = output_starts[node_idx];

    unsigned int local_row = tile_idx * rows_per_tile + threadIdx.x;
    if (local_row >= node_count) return;

    unsigned int row = input_indices[node_start + local_row];
    unsigned int bin = bins[row * num_features + split_feature];

    // Left indices at output_start, right indices at output_start + max_node_rows
    if (bin <= split_threshold) {
        unsigned int pos = atomicAdd(&counters[node_idx * 2], 1);
        output_indices[output_start + pos] = row;
    } else {
        unsigned int pos = atomicAdd(&counters[node_idx * 2 + 1], 1);
        output_indices[output_start + max_node_rows + pos] = row;
    }
}

extern "C" __global__ void zero_counters_n(unsigned int* counters, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        counters[idx] = 0;
    }
}

// GPU-resident partition: reads from input buffer, writes to output buffer
// Output layout: contiguous left/right indices for each node
// Each node's results are written starting at output_starts[node_idx]
// Left indices first, then right indices after left_counts[node_idx]
extern "C" __global__ void partition_gpu_resident(
    const unsigned char* __restrict__ bins,
    const unsigned int* __restrict__ input_indices,      // Input GPU buffer
    unsigned int* __restrict__ output_indices,           // Output GPU buffer
    unsigned int* __restrict__ counters,                 // [node_idx * 2 + 0/1] for left/right count
    const unsigned int* __restrict__ node_input_starts,  // Where to read from input for each node
    const unsigned int* __restrict__ node_counts,        // Number of rows for each node
    const unsigned int* __restrict__ split_features,     // Split feature for each node
    const unsigned int* __restrict__ split_thresholds,   // Split threshold for each node
    const unsigned int* __restrict__ node_output_starts, // Where to write in output for each node
    unsigned int num_features,
    unsigned int max_node_rows                           // Max rows per node (for right offset)
) {
    unsigned int node_idx = blockIdx.x;
    unsigned int tile_idx = blockIdx.y;
    unsigned int rows_per_tile = blockDim.x;

    unsigned int input_start = node_input_starts[node_idx];
    unsigned int node_count = node_counts[node_idx];
    unsigned int split_feature = split_features[node_idx];
    unsigned int split_threshold = split_thresholds[node_idx];
    unsigned int output_start = node_output_starts[node_idx];

    unsigned int local_row = tile_idx * rows_per_tile + threadIdx.x;
    if (local_row >= node_count) return;

    unsigned int row = input_indices[input_start + local_row];
    unsigned int bin = bins[row * num_features + split_feature];

    // Write left at output_start, right at output_start + max_node_rows
    if (bin <= split_threshold) {
        unsigned int pos = atomicAdd(&counters[node_idx * 2], 1);
        output_indices[output_start + pos] = row;
    } else {
        unsigned int pos = atomicAdd(&counters[node_idx * 2 + 1], 1);
        output_indices[output_start + max_node_rows + pos] = row;
    }
}

// Compact: rearrange output buffer so left/right are contiguous for next level
// Input: [node0_left(max), node0_right(max), node1_left(max), node1_right(max), ...]
// Output: [node0_left(actual), node0_right(actual), node1_left(actual), node1_right(actual), ...]
extern "C" __global__ void compact_partitions(
    const unsigned int* __restrict__ input,        // Padded output from partition
    unsigned int* __restrict__ output,             // Compacted for next level
    const unsigned int* __restrict__ input_starts, // Original output_starts from partition
    const unsigned int* __restrict__ left_counts,  // Actual left counts
    const unsigned int* __restrict__ right_counts, // Actual right counts
    const unsigned int* __restrict__ output_starts,// Where to write compacted results
    unsigned int max_node_rows,
    unsigned int num_nodes
) {
    // Each block handles one node's compaction
    unsigned int node_idx = blockIdx.x;
    if (node_idx >= num_nodes) return;

    unsigned int in_start = input_starts[node_idx];
    unsigned int left_count = left_counts[node_idx];
    unsigned int right_count = right_counts[node_idx];
    unsigned int out_start = output_starts[node_idx];

    // Copy left indices
    for (unsigned int i = threadIdx.x; i < left_count; i += blockDim.x) {
        output[out_start + i] = input[in_start + i];
    }

    // Copy right indices (right is at in_start + max_node_rows in input)
    for (unsigned int i = threadIdx.x; i < right_count; i += blockDim.x) {
        output[out_start + left_count + i] = input[in_start + max_node_rows + i];
    }
}

// Fused partition + compact: partition and write directly to contiguous output
// Uses two global counters per node: running offset for left and right
// Output layout: [node0_left, node0_right, node1_left, node1_right, ...]
// The output_offsets array stores the starting position for each node's combined output
extern "C" __global__ void partition_fused(
    const unsigned char* __restrict__ bins,
    const unsigned int* __restrict__ input_indices,
    unsigned int* __restrict__ output_indices,
    unsigned int* __restrict__ counters,           // [node_idx * 2 + 0/1] for left/right local count
    const unsigned int* __restrict__ node_input_starts,
    const unsigned int* __restrict__ node_counts,
    const unsigned int* __restrict__ split_features,
    const unsigned int* __restrict__ split_thresholds,
    const unsigned int* __restrict__ output_offsets,  // Where each node's output starts
    const unsigned int* __restrict__ left_capacities, // Max left count for each node (for right offset)
    unsigned int num_features
) {
    unsigned int node_idx = blockIdx.x;
    unsigned int tile_idx = blockIdx.y;
    unsigned int rows_per_tile = blockDim.x;

    unsigned int input_start = node_input_starts[node_idx];
    unsigned int node_count = node_counts[node_idx];
    unsigned int split_feature = split_features[node_idx];
    unsigned int split_threshold = split_thresholds[node_idx];
    unsigned int output_base = output_offsets[node_idx];
    unsigned int left_capacity = left_capacities[node_idx];

    unsigned int local_row = tile_idx * rows_per_tile + threadIdx.x;
    if (local_row >= node_count) return;

    unsigned int row = input_indices[input_start + local_row];
    unsigned int bin = bins[row * num_features + split_feature];

    // Write directly to contiguous output
    // Left at output_base, right at output_base + left_capacity
    if (bin <= split_threshold) {
        unsigned int pos = atomicAdd(&counters[node_idx * 2], 1);
        output_indices[output_base + pos] = row;
    } else {
        unsigned int pos = atomicAdd(&counters[node_idx * 2 + 1], 1);
        output_indices[output_base + left_capacity + pos] = row;
    }
}
"#;

/// Split information for one node.
#[derive(Debug, Clone, Copy)]
pub struct NodeSplit {
    pub input_start: u32,
    pub input_count: u32,
    pub split_feature: u32,
    pub split_threshold: u32,
}

/// Result of a partition operation.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    pub left_indices: Vec<u32>,
    pub right_indices: Vec<u32>,
    pub left_count: u32,
    pub right_count: u32,
}

/// Result of GPU-resident partition (only counts, indices stay on GPU).
#[derive(Debug, Clone, Copy)]
pub struct GpuPartitionResult {
    pub left_count: u32,
    pub right_count: u32,
    pub output_start: u32, // Start offset in output buffer for this node's results
}

/// CUDA partition kernel executor with buffer caching.
pub struct PartitionKernel {
    device: Arc<CudaDevice>,
    module: Option<Arc<CudaModule>>,
    partition_fn: Option<CudaFunction>,
    partition_batched_fn: Option<CudaFunction>,
    partition_gpu_resident_fn: Option<CudaFunction>,
    partition_fused_fn: Option<CudaFunction>,
    compact_partitions_fn: Option<CudaFunction>,
    zero_counters_fn: Option<CudaFunction>,

    // Cached GPU buffers
    cached_bins: Option<CudaSlice<u8>>,
    cached_bins_key: Option<CacheKey>,

    // Reusable output buffers (sized for max expected use)
    cached_left: Option<CudaSlice<u32>>,
    cached_right: Option<CudaSlice<u32>>,
    cached_counters: Option<CudaSlice<u32>>,
    cached_output_size: usize,

    // Batched partition buffers
    cached_batch_output: Option<CudaSlice<u32>>,
    cached_batch_counters: Option<CudaSlice<u32>>,
    cached_batch_size: usize,

    // Cached metadata buffers for fused partition (avoid per-level allocations)
    cached_node_input_starts: Option<CudaSlice<u32>>,
    cached_node_counts: Option<CudaSlice<u32>>,
    cached_split_features: Option<CudaSlice<u32>>,
    cached_split_thresholds: Option<CudaSlice<u32>>,
    cached_output_offsets: Option<CudaSlice<u32>>,
    cached_left_capacities: Option<CudaSlice<u32>>,
    cached_metadata_capacity: usize,
}

impl PartitionKernel {
    /// Create a new partition kernel.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            module: None,
            partition_fn: None,
            partition_batched_fn: None,
            partition_gpu_resident_fn: None,
            partition_fused_fn: None,
            compact_partitions_fn: None,
            zero_counters_fn: None,
            cached_bins: None,
            cached_bins_key: None,
            cached_left: None,
            cached_right: None,
            cached_counters: None,
            cached_output_size: 0,
            cached_batch_output: None,
            cached_batch_counters: None,
            cached_batch_size: 0,
            cached_node_input_starts: None,
            cached_node_counts: None,
            cached_split_features: None,
            cached_split_thresholds: None,
            cached_output_offsets: None,
            cached_left_capacities: None,
            cached_metadata_capacity: 0,
        }
    }

    /// Ensure the kernel is compiled and loaded.
    fn ensure_initialized(&mut self) {
        if self.module.is_some() {
            return;
        }

        let ptx = compile_ptx(PARTITION_KERNEL_SOURCE).expect("Failed to compile partition kernel");
        let module = self.device.load_module(ptx);

        self.partition_fn = Some(CudaDevice::load_function(&module, "partition_atomic"));
        self.partition_batched_fn = Some(CudaDevice::load_function(&module, "partition_batched"));
        self.partition_gpu_resident_fn =
            Some(CudaDevice::load_function(&module, "partition_gpu_resident"));
        self.partition_fused_fn = Some(CudaDevice::load_function(&module, "partition_fused"));
        self.compact_partitions_fn = Some(CudaDevice::load_function(&module, "compact_partitions"));
        self.zero_counters_fn = Some(CudaDevice::load_function(&module, "zero_counters_n"));
        self.module = Some(module);
    }

    /// Ensure metadata buffers are large enough for num_nodes.
    fn ensure_metadata_buffers(&mut self, num_nodes: usize) {
        if self.cached_metadata_capacity >= num_nodes && self.cached_node_input_starts.is_some() {
            return;
        }
        // Allocate with some headroom
        let capacity = (num_nodes * 2).max(64);
        self.cached_node_input_starts = Some(self.device.alloc_zeros(capacity));
        self.cached_node_counts = Some(self.device.alloc_zeros(capacity));
        self.cached_split_features = Some(self.device.alloc_zeros(capacity));
        self.cached_split_thresholds = Some(self.device.alloc_zeros(capacity));
        self.cached_output_offsets = Some(self.device.alloc_zeros(capacity));
        self.cached_left_capacities = Some(self.device.alloc_zeros(capacity));
        self.cached_metadata_capacity = capacity;
    }

    /// Get the device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Ensure bins are cached on GPU.
    pub fn ensure_bins_cached(&mut self, bins: &[u8]) {
        let bins_key = CacheKey::from_slice(bins);
        if self.cached_bins_key != Some(bins_key) || self.cached_bins.is_none() {
            self.cached_bins = Some(self.device.htod_copy(bins));
            self.cached_bins_key = Some(bins_key);
        }
    }

    /// Get cached bins buffer.
    pub fn cached_bins(&self) -> Option<&CudaSlice<u8>> {
        self.cached_bins.as_ref()
    }

    /// Ensure output buffers are allocated
    fn ensure_output_buffers(&mut self, size: usize) {
        if self.cached_output_size < size || self.cached_left.is_none() {
            // Allocate with some slack
            let alloc_size = (size * 3 / 2).max(size);
            self.cached_left = Some(self.device.alloc_zeros(alloc_size));
            self.cached_right = Some(self.device.alloc_zeros(alloc_size));
            self.cached_counters = Some(self.device.alloc_zeros(2));
            self.cached_output_size = alloc_size;
        }
    }

    /// Partition rows based on a split condition.
    pub fn partition(
        &mut self,
        bins: &[u8],
        input_indices: &[u32],
        split_feature: u32,
        split_threshold: u32,
        num_features: usize,
    ) -> PartitionResult {
        self.ensure_initialized();

        let num_indices = input_indices.len();
        if num_indices == 0 {
            return PartitionResult {
                left_indices: Vec::new(),
                right_indices: Vec::new(),
                left_count: 0,
                right_count: 0,
            };
        }

        // Cache bins and allocate output buffers
        self.ensure_bins_cached(bins);
        self.ensure_output_buffers(num_indices);

        // Upload input indices
        let d_input = self.device.htod_copy(input_indices);

        let stream = self.device.stream();

        // Zero counters
        let zero_config = LaunchConfig {
            block_dim: (2, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_counters = self.cached_counters.as_mut().unwrap();
            stream
                .launch_builder(self.zero_counters_fn.as_ref().unwrap())
                .arg(d_counters)
                .arg(&2u32)
                .launch(zero_config)
                .expect("Failed to launch zero_counters kernel");
        }

        // Launch partition kernel
        let config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (num_indices.div_ceil(256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_bins = self.cached_bins.as_ref().unwrap();
            let d_left = self.cached_left.as_mut().unwrap();
            let d_right = self.cached_right.as_mut().unwrap();
            let d_counters = self.cached_counters.as_mut().unwrap();

            stream
                .launch_builder(self.partition_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(&d_input)
                .arg(d_left)
                .arg(d_right)
                .arg(d_counters)
                .arg(&(num_indices as u32))
                .arg(&(num_features as u32))
                .arg(&split_feature)
                .arg(&split_threshold)
                .launch(config)
                .expect("Failed to launch partition kernel");
        }

        self.device.synchronize();

        // Read back counters
        let counters = self
            .device
            .dtoh_copy(self.cached_counters.as_ref().unwrap());
        let left_count = counters[0];
        let right_count = counters[1];

        // Read back only the needed portions
        let all_left = self.device.dtoh_copy(self.cached_left.as_ref().unwrap());
        let all_right = self.device.dtoh_copy(self.cached_right.as_ref().unwrap());

        PartitionResult {
            left_indices: all_left[..left_count as usize].to_vec(),
            right_indices: all_right[..right_count as usize].to_vec(),
            left_count,
            right_count,
        }
    }

    /// Partition multiple nodes in a single kernel launch.
    pub fn partition_batched(
        &mut self,
        bins: &[u8],
        input_indices: &[u32],
        node_splits: &[NodeSplit],
        num_features: usize,
    ) -> Vec<PartitionResult> {
        self.ensure_initialized();

        if node_splits.is_empty() {
            return Vec::new();
        }

        // For single node, use simpler path
        if node_splits.len() == 1 {
            let split = &node_splits[0];
            let start = split.input_start as usize;
            let end = start + split.input_count as usize;
            return vec![self.partition(
                bins,
                &input_indices[start..end],
                split.split_feature,
                split.split_threshold,
                num_features,
            )];
        }

        // Cache bins
        self.ensure_bins_cached(bins);

        let num_nodes = node_splits.len();
        let max_node_rows = node_splits
            .iter()
            .map(|s| s.input_count as usize)
            .max()
            .unwrap_or(0);

        // Prepare batch parameters
        let node_starts: Vec<u32> = node_splits.iter().map(|s| s.input_start).collect();
        let node_counts: Vec<u32> = node_splits.iter().map(|s| s.input_count).collect();
        let split_features: Vec<u32> = node_splits.iter().map(|s| s.split_feature).collect();
        let split_thresholds: Vec<u32> = node_splits.iter().map(|s| s.split_threshold).collect();

        // Output starts: each node gets max_node_rows * 2 (left + right)
        let output_starts: Vec<u32> = (0..num_nodes)
            .map(|i| (i * max_node_rows * 2) as u32)
            .collect();

        // Ensure batch buffers are allocated
        let total_output = num_nodes * max_node_rows * 2;
        let num_counters = num_nodes * 2;

        if self.cached_batch_size < total_output || self.cached_batch_output.is_none() {
            self.cached_batch_output = Some(self.device.alloc_zeros(total_output));
            self.cached_batch_size = total_output;
        }
        if self.cached_batch_counters.is_none()
            || self
                .cached_batch_counters
                .as_ref()
                .map(|c| c.len())
                .unwrap_or(0)
                < num_counters
        {
            self.cached_batch_counters = Some(self.device.alloc_zeros(num_counters));
        }

        // Upload batch parameters
        let d_input = self.device.htod_copy(input_indices);
        let d_node_starts = self.device.htod_copy(&node_starts);
        let d_node_counts = self.device.htod_copy(&node_counts);
        let d_split_features = self.device.htod_copy(&split_features);
        let d_split_thresholds = self.device.htod_copy(&split_thresholds);
        let d_output_starts = self.device.htod_copy(&output_starts);

        let stream = self.device.stream();

        // Zero counters
        let zero_blocks = num_counters.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks.max(1), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_counters = self.cached_batch_counters.as_mut().unwrap();
            stream
                .launch_builder(self.zero_counters_fn.as_ref().unwrap())
                .arg(d_counters)
                .arg(&(num_counters as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_counters kernel");
        }

        // Launch batched partition kernel
        // Grid: (num_nodes, row_tiles)
        let threads_per_block = 256u32;
        let row_tiles = (max_node_rows as u32).div_ceil(threads_per_block);

        let config = LaunchConfig {
            block_dim: (threads_per_block, 1, 1),
            grid_dim: (num_nodes as u32, row_tiles, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let d_bins = self.cached_bins.as_ref().unwrap();
            let d_output = self.cached_batch_output.as_mut().unwrap();
            let d_counters = self.cached_batch_counters.as_mut().unwrap();

            stream
                .launch_builder(self.partition_batched_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(&d_input)
                .arg(d_output)
                .arg(d_counters)
                .arg(&d_node_starts)
                .arg(&d_node_counts)
                .arg(&d_split_features)
                .arg(&d_split_thresholds)
                .arg(&d_output_starts)
                .arg(&(num_features as u32))
                .arg(&(max_node_rows as u32))
                .launch(config)
                .expect("Failed to launch partition_batched kernel");
        }

        self.device.synchronize();

        // Read back results
        let counters = self
            .device
            .dtoh_copy(self.cached_batch_counters.as_ref().unwrap());
        let output = self
            .device
            .dtoh_copy(self.cached_batch_output.as_ref().unwrap());

        // Extract results for each node
        (0..num_nodes)
            .map(|i| {
                let left_count = counters[i * 2];
                let right_count = counters[i * 2 + 1];
                let output_start = output_starts[i] as usize;

                let left_indices =
                    output[output_start..output_start + left_count as usize].to_vec();
                let right_indices = output[output_start + max_node_rows
                    ..output_start + max_node_rows + right_count as usize]
                    .to_vec();

                PartitionResult {
                    left_indices,
                    right_indices,
                    left_count,
                    right_count,
                }
            })
            .collect()
    }

    /// GPU-resident partition: reads from input GPU buffer, writes to output GPU buffer.
    /// Returns only counts (indices stay on GPU).
    /// Uses internally cached bins buffer.
    ///
    /// # Arguments
    /// * `d_input` - Input GPU buffer with row indices
    /// * `d_output` - Output GPU buffer (will contain partitioned indices)
    /// * `node_splits` - Split info for each node (includes input_start, input_count, feature, threshold)
    /// * `num_features` - Number of features
    ///
    /// # Returns
    /// Vec of GpuPartitionResult with counts and output offsets
    pub fn partition_gpu_resident(
        &mut self,
        d_input: &CudaSlice<u32>,
        d_output: &mut CudaSlice<u32>,
        node_splits: &[NodeSplit],
        num_features: usize,
    ) -> Vec<GpuPartitionResult> {
        self.ensure_initialized();

        if node_splits.is_empty() {
            return Vec::new();
        }

        let num_nodes = node_splits.len();
        let max_node_rows = node_splits
            .iter()
            .map(|s| s.input_count as usize)
            .max()
            .unwrap_or(0);

        // Prepare node metadata
        let node_input_starts: Vec<u32> = node_splits.iter().map(|s| s.input_start).collect();
        let node_counts: Vec<u32> = node_splits.iter().map(|s| s.input_count).collect();
        let split_features: Vec<u32> = node_splits.iter().map(|s| s.split_feature).collect();
        let split_thresholds: Vec<u32> = node_splits.iter().map(|s| s.split_threshold).collect();

        // Output layout: each node gets max_node_rows * 2 space (left then right)
        let node_output_starts: Vec<u32> = (0..num_nodes)
            .map(|i| (i * max_node_rows * 2) as u32)
            .collect();

        let d_node_input_starts = self.device.htod_copy(&node_input_starts);
        let d_node_counts = self.device.htod_copy(&node_counts);
        let d_split_features = self.device.htod_copy(&split_features);
        let d_split_thresholds = self.device.htod_copy(&split_thresholds);
        let d_node_output_starts = self.device.htod_copy(&node_output_starts);

        // Ensure counters buffer
        let num_counters = num_nodes * 2;
        if self.cached_batch_counters.is_none()
            || self
                .cached_batch_counters
                .as_ref()
                .map(|c| c.len())
                .unwrap_or(0)
                < num_counters
        {
            self.cached_batch_counters = Some(self.device.alloc_zeros(num_counters));
        }

        let stream = self.device.stream();

        // Zero counters
        let zero_blocks = num_counters.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks.max(1), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_counters = self.cached_batch_counters.as_mut().unwrap();
            stream
                .launch_builder(self.zero_counters_fn.as_ref().unwrap())
                .arg(d_counters)
                .arg(&(num_counters as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_counters kernel");
        }

        // Launch GPU-resident partition kernel
        let threads_per_block = 256u32;
        let row_tiles = (max_node_rows as u32).div_ceil(threads_per_block);

        let config = LaunchConfig {
            block_dim: (threads_per_block, 1, 1),
            grid_dim: (num_nodes as u32, row_tiles, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let d_bins = self.cached_bins.as_ref().expect("bins must be cached");
            let d_counters = self.cached_batch_counters.as_mut().unwrap();

            stream
                .launch_builder(self.partition_gpu_resident_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(d_input)
                .arg(d_output)
                .arg(d_counters)
                .arg(&d_node_input_starts)
                .arg(&d_node_counts)
                .arg(&d_split_features)
                .arg(&d_split_thresholds)
                .arg(&d_node_output_starts)
                .arg(&(num_features as u32))
                .arg(&(max_node_rows as u32))
                .launch(config)
                .expect("Failed to launch partition_gpu_resident kernel");
        }

        self.device.synchronize();

        // Read back only counters (indices stay on GPU)
        let counters = self
            .device
            .dtoh_copy(self.cached_batch_counters.as_ref().unwrap());

        // Return results with offsets
        (0..num_nodes)
            .map(|i| GpuPartitionResult {
                left_count: counters[i * 2],
                right_count: counters[i * 2 + 1],
                output_start: node_output_starts[i],
            })
            .collect()
    }

    /// Compact partition results: rearrange output so left/right are contiguous.
    /// Call this after partition_gpu_resident to prepare indices for next level.
    ///
    /// Input layout (per node): [left(max_rows), right(max_rows)]
    /// Output layout (per node): [left(actual), right(actual)] - contiguous
    pub fn compact_partitions(
        &mut self,
        d_padded: &CudaSlice<u32>,
        d_compacted: &mut CudaSlice<u32>,
        partition_results: &[GpuPartitionResult],
        max_node_rows: usize,
    ) -> Vec<(u32, u32)> {
        self.ensure_initialized();

        if partition_results.is_empty() {
            return Vec::new();
        }

        let num_nodes = partition_results.len();

        // Calculate compacted output offsets
        let mut compacted_output_starts = Vec::with_capacity(num_nodes);
        let mut offset = 0u32;
        for r in partition_results {
            compacted_output_starts.push(offset);
            offset += r.left_count + r.right_count;
        }

        let input_starts: Vec<u32> = partition_results.iter().map(|r| r.output_start).collect();
        let left_counts: Vec<u32> = partition_results.iter().map(|r| r.left_count).collect();
        let right_counts: Vec<u32> = partition_results.iter().map(|r| r.right_count).collect();

        let d_input_starts = self.device.htod_copy(&input_starts);
        let d_left_counts = self.device.htod_copy(&left_counts);
        let d_right_counts = self.device.htod_copy(&right_counts);
        let d_output_starts = self.device.htod_copy(&compacted_output_starts);

        let stream = self.device.stream();

        let config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (num_nodes as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(self.compact_partitions_fn.as_ref().unwrap())
                .arg(d_padded)
                .arg(d_compacted)
                .arg(&d_input_starts)
                .arg(&d_left_counts)
                .arg(&d_right_counts)
                .arg(&d_output_starts)
                .arg(&(max_node_rows as u32))
                .arg(&(num_nodes as u32))
                .launch(config)
                .expect("Failed to launch compact_partitions kernel");
        }

        self.device.synchronize();

        // Return (compacted_start, total_count) for each node
        partition_results
            .iter()
            .zip(compacted_output_starts.iter())
            .map(|(r, &start)| (start, r.left_count + r.right_count))
            .collect()
    }

    /// Fused partition: partition and write directly to contiguous output.
    /// Eliminates the separate compaction step for better performance.
    ///
    /// # Arguments
    /// * `d_input` - Input GPU buffer with row indices
    /// * `d_output` - Output GPU buffer (will contain partitioned indices in contiguous layout)
    /// * `node_splits` - Split info for each node
    /// * `num_features` - Number of features
    ///
    /// # Returns
    /// Vec of (output_start, left_count, right_count) for each node
    pub fn partition_fused(
        &mut self,
        d_input: &CudaSlice<u32>,
        d_output: &mut CudaSlice<u32>,
        node_splits: &[NodeSplit],
        num_features: usize,
    ) -> Vec<(u32, u32, u32)> {
        self.ensure_initialized();

        if node_splits.is_empty() {
            return Vec::new();
        }

        let num_nodes = node_splits.len();

        // Ensure metadata buffers are allocated
        self.ensure_metadata_buffers(num_nodes);

        // Calculate output layout: each node gets node_count space (left + right)
        // Pre-allocate with left_capacity = node_count (worst case all go left)
        let mut output_offsets = Vec::with_capacity(num_nodes);
        let mut left_capacities = Vec::with_capacity(num_nodes);
        let mut offset = 0u32;
        for split in node_splits {
            output_offsets.push(offset);
            left_capacities.push(split.input_count); // Use full count as capacity
            offset += split.input_count * 2; // Space for left + right (with padding)
        }

        // Prepare node metadata
        let node_input_starts: Vec<u32> = node_splits.iter().map(|s| s.input_start).collect();
        let node_counts: Vec<u32> = node_splits.iter().map(|s| s.input_count).collect();
        let split_features: Vec<u32> = node_splits.iter().map(|s| s.split_feature).collect();
        let split_thresholds: Vec<u32> = node_splits.iter().map(|s| s.split_threshold).collect();

        // Copy metadata to cached GPU buffers (htod_copy_into avoids allocation)
        self.device.htod_copy_into(
            &node_input_starts,
            self.cached_node_input_starts.as_mut().unwrap(),
        );
        self.device
            .htod_copy_into(&node_counts, self.cached_node_counts.as_mut().unwrap());
        self.device.htod_copy_into(
            &split_features,
            self.cached_split_features.as_mut().unwrap(),
        );
        self.device.htod_copy_into(
            &split_thresholds,
            self.cached_split_thresholds.as_mut().unwrap(),
        );
        self.device.htod_copy_into(
            &output_offsets,
            self.cached_output_offsets.as_mut().unwrap(),
        );
        self.device.htod_copy_into(
            &left_capacities,
            self.cached_left_capacities.as_mut().unwrap(),
        );

        // Ensure counters buffer
        let num_counters = num_nodes * 2;
        if self.cached_batch_counters.is_none()
            || self
                .cached_batch_counters
                .as_ref()
                .map(|c| c.len())
                .unwrap_or(0)
                < num_counters
        {
            self.cached_batch_counters = Some(self.device.alloc_zeros(num_counters));
        }

        let stream = self.device.stream();

        // Zero counters
        let zero_blocks = num_counters.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks.max(1), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_counters = self.cached_batch_counters.as_mut().unwrap();
            stream
                .launch_builder(self.zero_counters_fn.as_ref().unwrap())
                .arg(d_counters)
                .arg(&(num_counters as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_counters kernel");
        }

        // Launch fused partition kernel
        let max_node_rows = node_splits
            .iter()
            .map(|s| s.input_count as usize)
            .max()
            .unwrap_or(0);
        let threads_per_block = 256u32;
        let row_tiles = (max_node_rows as u32).div_ceil(threads_per_block);

        let config = LaunchConfig {
            block_dim: (threads_per_block, 1, 1),
            grid_dim: (num_nodes as u32, row_tiles, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let d_bins = self.cached_bins.as_ref().expect("bins must be cached");
            let d_counters = self.cached_batch_counters.as_mut().unwrap();

            stream
                .launch_builder(self.partition_fused_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(d_input)
                .arg(d_output)
                .arg(d_counters)
                .arg(self.cached_node_input_starts.as_ref().unwrap())
                .arg(self.cached_node_counts.as_ref().unwrap())
                .arg(self.cached_split_features.as_ref().unwrap())
                .arg(self.cached_split_thresholds.as_ref().unwrap())
                .arg(self.cached_output_offsets.as_ref().unwrap())
                .arg(self.cached_left_capacities.as_ref().unwrap())
                .arg(&(num_features as u32))
                .launch(config)
                .expect("Failed to launch partition_fused kernel");
        }

        self.device.synchronize();

        // Read back only counters (indices stay on GPU)
        let counters = self
            .device
            .dtoh_copy(self.cached_batch_counters.as_ref().unwrap());

        // Return (output_start, left_count, right_count) for each node
        (0..num_nodes)
            .map(|i| (output_offsets[i], counters[i * 2], counters[i * 2 + 1]))
            .collect()
    }
}
