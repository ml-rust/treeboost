//! GPU Row Partitioning Kernel
//!
//! Implements efficient parallel row partitioning for fully-GPU tree building.
//! Uses stream compaction with parallel prefix sums to partition rows based on
//! split conditions.
//!
//! Two implementations are provided:
//! 1. **3-pass scan**: Efficient for large arrays, uses parallel prefix sums
//! 2. **Atomic**: Simpler but slower, uses atomic counters
//!
//! The batched atomic version is used for level-wise tree building where
//! multiple nodes are partitioned simultaneously.

use super::device::GpuDevice;
use std::sync::Arc;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, ComputePipeline};

/// Parameters for single-node partitioning
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PartitionParams {
    pub num_indices: u32,
    pub split_feature: u32,
    pub split_threshold: u32,
    pub num_features: u32,
    pub num_blocks: u32,
    pub _padding: [u32; 3],
}

/// Parameters for batched partitioning
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchedPartitionParams {
    pub num_nodes: u32,
    pub num_features: u32,
    pub _padding: [u32; 2],
}

/// Split information for one node in batched partitioning
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeSplit {
    pub input_start: u32,
    pub input_count: u32,
    pub output_left_start: u32,
    pub output_right_start: u32,
    pub split_feature: u32,
    pub split_threshold: u32,
    pub _padding: [u32; 2],
}

/// Result of a partition operation
#[derive(Debug, Clone)]
pub struct PartitionResult {
    pub left_indices: Vec<u32>,
    pub right_indices: Vec<u32>,
    pub left_count: u32,
    pub right_count: u32,
}

/// Pooled buffer for reuse
struct PooledBuffer {
    buffer: Buffer,
    capacity: u64,
}

/// Buffer pool for partition operations
struct PartitionBufferPool {
    params: Option<Buffer>,
    batched_params: Option<Buffer>,
    input_indices: Option<PooledBuffer>,
    left_indices: Option<PooledBuffer>,
    right_indices: Option<PooledBuffer>,
    counters: Option<Buffer>,
    node_splits: Option<PooledBuffer>,
    node_counters: Option<PooledBuffer>,
    // Staging buffers for readback
    staging_left: Option<PooledBuffer>,
    staging_right: Option<PooledBuffer>,
    staging_counters: Option<Buffer>,
}

impl PartitionBufferPool {
    fn new() -> Self {
        Self {
            params: None,
            batched_params: None,
            input_indices: None,
            left_indices: None,
            right_indices: None,
            counters: None,
            node_splits: None,
            node_counters: None,
            staging_left: None,
            staging_right: None,
            staging_counters: None,
        }
    }
}

/// GPU partition kernel executor
pub struct PartitionKernel {
    device: Arc<GpuDevice>,

    // Atomic pipelines (simpler, for small arrays or batched)
    pipeline_zero_counters: ComputePipeline,
    pipeline_atomic: ComputePipeline,
    bind_group_layout_zero_counters: BindGroupLayout,
    bind_group_layout_atomic: BindGroupLayout,

    // Batched atomic pipelines
    pipeline_zero_node_counters: ComputePipeline,
    pipeline_batched_atomic: ComputePipeline,
    bind_group_layout_zero_node_counters: BindGroupLayout,
    bind_group_layout_batched: BindGroupLayout,

    // Buffer pool
    buffer_pool: std::sync::Mutex<PartitionBufferPool>,
}

impl PartitionKernel {
    const WORKGROUP_SIZE: u32 = 256;

    /// Create partition kernel with compiled shaders
    pub fn new(device: Arc<GpuDevice>) -> Self {
        let shader_source = include_str!("shaders/row_partition.wgsl");

        // Atomic pipelines
        let pipeline_zero_counters = device.create_compute_pipeline(
            "zero_counters_pipeline",
            shader_source,
            "zero_counters",
        );
        let pipeline_atomic = device.create_compute_pipeline(
            "partition_atomic_pipeline",
            shader_source,
            "partition_atomic",
        );

        // Batched pipelines
        let pipeline_zero_node_counters = device.create_compute_pipeline(
            "zero_node_counters_pipeline",
            shader_source,
            "zero_node_counters",
        );
        let pipeline_batched_atomic = device.create_compute_pipeline(
            "partition_batched_atomic_pipeline",
            shader_source,
            "partition_batched_atomic",
        );

        // Get bind group layouts
        let bind_group_layout_zero_counters = pipeline_zero_counters.get_bind_group_layout(0);
        let bind_group_layout_atomic = pipeline_atomic.get_bind_group_layout(0);
        let bind_group_layout_zero_node_counters =
            pipeline_zero_node_counters.get_bind_group_layout(0);
        let bind_group_layout_batched = pipeline_batched_atomic.get_bind_group_layout(0);

        Self {
            device,
            pipeline_zero_counters,
            pipeline_atomic,
            bind_group_layout_zero_counters,
            bind_group_layout_atomic,
            pipeline_zero_node_counters,
            pipeline_batched_atomic,
            bind_group_layout_zero_node_counters,
            bind_group_layout_batched,
            buffer_pool: std::sync::Mutex::new(PartitionBufferPool::new()),
        }
    }

    /// Partition rows using atomic counters (simple version)
    ///
    /// This is simpler but has atomic contention. Good for small arrays
    /// or when simplicity is preferred over maximum performance.
    pub fn partition_atomic(
        &self,
        bins_packed: &[u32],
        input_indices: &[u32],
        split_feature: u32,
        split_threshold: u32,
        num_features: usize,
    ) -> PartitionResult {
        let dev = &self.device;
        let num_indices = input_indices.len();

        if num_indices == 0 {
            return PartitionResult {
                left_indices: Vec::new(),
                right_indices: Vec::new(),
                left_count: 0,
                right_count: 0,
            };
        }

        let num_blocks = (num_indices as u32).div_ceil(Self::WORKGROUP_SIZE);

        // Calculate buffer sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let indices_size = (num_indices * 4) as u64;
        let counters_size = 8u64; // 2 * u32

        let params = PartitionParams {
            num_indices: num_indices as u32,
            split_feature,
            split_threshold,
            num_features: num_features as u32,
            num_blocks,
            _padding: [0; 3],
        };

        let mut pool = self.buffer_pool.lock().unwrap();

        // Ensure buffers
        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "partition_params",
                std::mem::size_of::<PartitionParams>() as u64,
            ));
        }

        Self::ensure_storage_buffer(dev, &mut pool.input_indices, "input_indices", indices_size);
        Self::ensure_storage_buffer(dev, &mut pool.left_indices, "left_indices", indices_size);
        Self::ensure_storage_buffer(dev, &mut pool.right_indices, "right_indices", indices_size);

        if pool.counters.is_none() {
            pool.counters = Some(dev.create_storage_buffer("counters", counters_size, true));
        }

        // We need a buffer for bins - reuse from histogram kernel if possible
        // For now, create a temporary one
        let bins_buffer = dev.create_storage_buffer("partition_bins", bins_size, false);
        dev.write_buffer(&bins_buffer, bins_packed);

        // Ensure staging buffers
        Self::ensure_staging_buffer(dev, &mut pool.staging_left, "staging_left", indices_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_right, "staging_right", indices_size);
        if pool.staging_counters.is_none() {
            pool.staging_counters =
                Some(dev.create_staging_buffer("staging_counters", counters_size));
        }

        // Upload data
        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);
        dev.write_buffer(&pool.input_indices.as_ref().unwrap().buffer, input_indices);

        // Create bind group for zero_counters (only needs counters binding)
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_counters_bind_group"),
            layout: &self.bind_group_layout_zero_counters,
            entries: &[BindGroupEntry {
                binding: 9,
                resource: pool.counters.as_ref().unwrap().as_entire_binding(),
            }],
        });

        // Create bind group for partition_atomic
        let bind_group_atomic = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("partition_atomic_bind_group"),
            layout: &self.bind_group_layout_atomic,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bins_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool
                        .input_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool
                        .left_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool
                        .right_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: pool.counters.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = dev.create_encoder("partition_atomic_encoder");

        // Zero counters
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_counters_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero_counters);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Partition
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("partition_atomic_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_atomic);
            pass.set_bind_group(0, &bind_group_atomic, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // Copy results
        encoder.copy_buffer_to_buffer(
            &pool.left_indices.as_ref().unwrap().buffer,
            0,
            &pool.staging_left.as_ref().unwrap().buffer,
            0,
            indices_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.right_indices.as_ref().unwrap().buffer,
            0,
            &pool.staging_right.as_ref().unwrap().buffer,
            0,
            indices_size,
        );
        encoder.copy_buffer_to_buffer(
            pool.counters.as_ref().unwrap(),
            0,
            pool.staging_counters.as_ref().unwrap(),
            0,
            counters_size,
        );

        dev.submit_and_wait(encoder);

        // Read back results
        let mut counters_data = [0u32; 2];
        dev.read_buffer(pool.staging_counters.as_ref().unwrap(), &mut counters_data);

        let left_count = counters_data[0];
        let right_count = counters_data[1];

        let mut left_data = vec![0u32; left_count as usize];
        let mut right_data = vec![0u32; right_count as usize];

        if left_count > 0 {
            dev.read_buffer_partial(&pool.staging_left.as_ref().unwrap().buffer, &mut left_data);
        }
        if right_count > 0 {
            dev.read_buffer_partial(
                &pool.staging_right.as_ref().unwrap().buffer,
                &mut right_data,
            );
        }

        drop(pool);

        PartitionResult {
            left_indices: left_data,
            right_indices: right_data,
            left_count,
            right_count,
        }
    }

    /// Partition multiple nodes simultaneously (for level-wise tree building)
    ///
    /// Each node has its own split condition and row indices.
    /// All nodes are partitioned in a single GPU dispatch.
    pub fn partition_batched(
        &self,
        bins_packed: &[u32],
        input_indices: &[u32],
        node_splits: &[NodeSplit],
        num_features: usize,
        total_rows: usize,
    ) -> Vec<PartitionResult> {
        let dev = &self.device;
        let num_nodes = node_splits.len();

        if num_nodes == 0 {
            return Vec::new();
        }

        // Calculate buffer sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let indices_size = (total_rows.max(1) * 4) as u64;
        let node_splits_size = std::mem::size_of_val(node_splits) as u64;
        let node_counters_size = (num_nodes * 2 * 4) as u64; // left + right per node

        let batched_params = BatchedPartitionParams {
            num_nodes: num_nodes as u32,
            num_features: num_features as u32,
            _padding: [0; 2],
        };

        let mut pool = self.buffer_pool.lock().unwrap();

        // Ensure buffers
        if pool.batched_params.is_none() {
            pool.batched_params = Some(dev.create_uniform_buffer(
                "batched_partition_params",
                std::mem::size_of::<BatchedPartitionParams>() as u64,
            ));
        }

        Self::ensure_storage_buffer(dev, &mut pool.input_indices, "input_indices", indices_size);
        Self::ensure_storage_buffer(dev, &mut pool.left_indices, "left_indices", indices_size);
        Self::ensure_storage_buffer(dev, &mut pool.right_indices, "right_indices", indices_size);
        Self::ensure_storage_buffer(dev, &mut pool.node_splits, "node_splits", node_splits_size);
        Self::ensure_storage_buffer(
            dev,
            &mut pool.node_counters,
            "node_counters",
            node_counters_size,
        );

        // Bins buffer
        let bins_buffer = dev.create_storage_buffer("partition_bins", bins_size, false);
        dev.write_buffer(&bins_buffer, bins_packed);

        // Staging buffers
        Self::ensure_staging_buffer(dev, &mut pool.staging_left, "staging_left", indices_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_right, "staging_right", indices_size);

        let staging_counters =
            dev.create_staging_buffer("staging_node_counters", node_counters_size);

        // Upload data
        dev.write_buffer(pool.batched_params.as_ref().unwrap(), &[batched_params]);
        dev.write_buffer(&pool.input_indices.as_ref().unwrap().buffer, input_indices);
        dev.write_buffer(&pool.node_splits.as_ref().unwrap().buffer, node_splits);

        // Create bind group for zero_node_counters (only needs batched_params and node_counters)
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_node_counters_bind_group"),
            layout: &self.bind_group_layout_zero_node_counters,
            entries: &[
                BindGroupEntry {
                    binding: 10,
                    resource: pool.batched_params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: pool
                        .node_counters
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
            ],
        });

        // Create bind group for partition_batched_atomic
        // Note: batched version uses bindings 1-4, 10-12 (not binding 0)
        let bind_group_batched = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("partition_batched_bind_group"),
            layout: &self.bind_group_layout_batched,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: bins_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool
                        .input_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool
                        .left_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool
                        .right_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: pool.batched_params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: pool
                        .node_splits
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: pool
                        .node_counters
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = dev.create_encoder("partition_batched_encoder");

        // Zero node counters
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_node_counters_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero_node_counters);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let workgroups = (num_nodes as u32 * 2).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Partition all nodes
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("partition_batched_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_batched_atomic);
            pass.set_bind_group(0, &bind_group_batched, &[]);
            pass.dispatch_workgroups(num_nodes as u32, 1, 1);
        }

        // Copy results
        encoder.copy_buffer_to_buffer(
            &pool.left_indices.as_ref().unwrap().buffer,
            0,
            &pool.staging_left.as_ref().unwrap().buffer,
            0,
            indices_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.right_indices.as_ref().unwrap().buffer,
            0,
            &pool.staging_right.as_ref().unwrap().buffer,
            0,
            indices_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.node_counters.as_ref().unwrap().buffer,
            0,
            &staging_counters,
            0,
            node_counters_size,
        );

        dev.submit_and_wait(encoder);

        // Read back counters
        let mut counters_data = vec![0u32; num_nodes * 2];
        dev.read_buffer(&staging_counters, &mut counters_data);

        // Read back all left and right indices
        let mut all_left = vec![0u32; total_rows];
        let mut all_right = vec![0u32; total_rows];

        if total_rows > 0 {
            dev.read_buffer(&pool.staging_left.as_ref().unwrap().buffer, &mut all_left);
            dev.read_buffer(&pool.staging_right.as_ref().unwrap().buffer, &mut all_right);
        }

        drop(pool);

        // Extract results per node
        let mut results = Vec::with_capacity(num_nodes);
        for (i, split) in node_splits.iter().enumerate() {
            let left_count = counters_data[i * 2];
            let right_count = counters_data[i * 2 + 1];

            let left_start = split.output_left_start as usize;
            let right_start = split.output_right_start as usize;

            let left_indices = all_left[left_start..left_start + left_count as usize].to_vec();
            let right_indices = all_right[right_start..right_start + right_count as usize].to_vec();

            results.push(PartitionResult {
                left_indices,
                right_indices,
                left_count,
                right_count,
            });
        }

        results
    }

    fn ensure_storage_buffer(
        dev: &GpuDevice,
        pool: &mut Option<PooledBuffer>,
        label: &str,
        required_size: u64,
    ) {
        let needs_new = match pool {
            Some(ref pb) => pb.capacity < required_size,
            None => true,
        };

        if needs_new {
            let capacity = ((required_size as f64 * 1.2) as u64 + 3) & !3;
            let buffer = dev.create_storage_buffer(label, capacity, true);
            *pool = Some(PooledBuffer { buffer, capacity });
        }
    }

    fn ensure_staging_buffer(
        dev: &GpuDevice,
        pool: &mut Option<PooledBuffer>,
        label: &str,
        required_size: u64,
    ) {
        let needs_new = match pool {
            Some(ref pb) => pb.capacity < required_size,
            None => true,
        };

        if needs_new {
            let capacity = ((required_size as f64 * 1.2) as u64 + 3) & !3;
            let buffer = dev.create_staging_buffer(label, capacity);
            *pool = Some(PooledBuffer { buffer, capacity });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<GpuDevice>> {
        GpuDevice::new().map(Arc::new)
    }

    #[test]
    fn test_partition_atomic_basic() {
        let device = match create_test_device() {
            Some(d) => d,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        let kernel = PartitionKernel::new(device);

        // Create simple test data
        // 4 rows, 2 features, row-major bins
        // Row 0: [0, 1], Row 1: [2, 3], Row 2: [4, 5], Row 3: [6, 7]
        let bins = [0u8, 1, 2, 3, 4, 5, 6, 7];
        let bins_packed: Vec<u32> = bins
            .chunks(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &b)| acc | ((b as u32) << (i * 8)))
            })
            .collect();

        let input_indices = vec![0u32, 1, 2, 3];

        // Split on feature 0, threshold 2
        // Rows with bin[feature 0] <= 2 go left
        // Row 0: bin[0]=0 <= 2 -> left
        // Row 1: bin[0]=2 <= 2 -> left
        // Row 2: bin[0]=4 > 2 -> right
        // Row 3: bin[0]=6 > 2 -> right

        let result = kernel.partition_atomic(
            &bins_packed,
            &input_indices,
            0, // feature 0
            2, // threshold
            2, // num_features
        );

        assert_eq!(result.left_count, 2);
        assert_eq!(result.right_count, 2);

        // Check that correct rows went to correct sides
        assert!(result.left_indices.contains(&0));
        assert!(result.left_indices.contains(&1));
        assert!(result.right_indices.contains(&2));
        assert!(result.right_indices.contains(&3));

        println!("Partition test passed!");
        println!("Left: {:?}", result.left_indices);
        println!("Right: {:?}", result.right_indices);
    }

    #[test]
    fn test_partition_batched() {
        let device = match create_test_device() {
            Some(d) => d,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        let kernel = PartitionKernel::new(device);

        // Create test data: 8 rows, 2 features (row-major)
        // Row 0: bins[0,1] = [0, 1]   -> feature 0 = 0
        // Row 1: bins[2,3] = [2, 3]   -> feature 0 = 2
        // Row 2: bins[4,5] = [4, 5]   -> feature 0 = 4
        // Row 3: bins[6,7] = [6, 7]   -> feature 0 = 6
        // Row 4: bins[8,9] = [8, 9]   -> feature 0 = 8
        // Row 5: bins[10,11] = [10,11] -> feature 0 = 10
        // Row 6: bins[12,13] = [12,13] -> feature 0 = 12
        // Row 7: bins[14,15] = [14,15] -> feature 0 = 14
        let num_features = 2;
        let bins: Vec<u8> = (0..16).collect();
        let bins_packed: Vec<u32> = bins
            .chunks(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &b)| acc | ((b as u32) << (i * 8)))
            })
            .collect();

        let input_indices: Vec<u32> = (0..8).collect();

        // Two nodes: node 0 gets rows 0-3, node 1 gets rows 4-7
        let node_splits = vec![
            NodeSplit {
                input_start: 0,
                input_count: 4,
                output_left_start: 0,
                output_right_start: 0,
                split_feature: 0,
                split_threshold: 4, // bins 0,2,4 <= 4 go left; 6 > 4 goes right
                _padding: [0; 2],
            },
            NodeSplit {
                input_start: 4,
                input_count: 4,
                output_left_start: 4,
                output_right_start: 4,
                split_feature: 0,
                split_threshold: 10, // bins 8,10 <= 10 go left; 12,14 > 10 go right
                _padding: [0; 2],
            },
        ];

        let results =
            kernel.partition_batched(&bins_packed, &input_indices, &node_splits, num_features, 8);

        assert_eq!(results.len(), 2);

        // Node 0: rows 0,1,2 go left (bin[0] = 0,2,4 <= 4), row 3 goes right (bin[0] = 6 > 4)
        assert_eq!(results[0].left_count, 3, "Node 0 left count");
        assert_eq!(results[0].right_count, 1, "Node 0 right count");

        // Node 1: rows 4,5 go left (bin[0] = 8,10 <= 10), rows 6,7 go right (bin[0] = 12,14 > 10)
        assert_eq!(results[1].left_count, 2, "Node 1 left count");
        assert_eq!(results[1].right_count, 2, "Node 1 right count");

        println!("Batched partition test passed!");
        println!(
            "Node 0 - Left: {:?}, Right: {:?}",
            results[0].left_indices, results[0].right_indices
        );
        println!(
            "Node 1 - Left: {:?}, Right: {:?}",
            results[1].left_indices, results[1].right_indices
        );
    }
}
