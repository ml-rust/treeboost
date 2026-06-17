//! GPU kernel dispatch for histogram building.
//!
//! Manages compute pipelines, buffer allocation, and kernel execution
//! for GPU-accelerated GBDT histogram construction.
//!
//! # Fixed-Point Gradient Accumulation
//!
//! This module uses fixed-point i32 arithmetic for gradient/hessian accumulation
//! to avoid expensive CAS loops. The optimization works as follows:
//!
//! 1. CPU quantizes f32 gradients/hessians to i32 (multiply by FIXED_POINT_SCALE)
//! 2. GPU uses native i32 atomicAdd (no CAS loops!)
//! 3. CPU dequantizes results (divide by FIXED_POINT_SCALE)
//!
//! This gives ~2-3x faster histogram building vs CAS loops for f32 atomics.
//!
//! # Sparse Feature Support
//!
//! For features with >90% sparsity, uses specialized sparse kernel:
//! - Only processes non-default entries
//! - Uses default-bin subtraction trick for correctness
//! - Up to 10x faster than dense kernel on highly sparse data

use super::device::GpuDevice;
use crate::histogram::Histogram;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, ComputePipeline};

/// Fixed-point scale factor for gradient/hessian quantization.
/// Using 2^10 (1024) allows values up to ±31.99 with ~0.001 precision.
/// This fits in i16 range while maintaining sufficient accuracy for GBDT.
/// Gradients are packed as i16 pairs in u32 for 2x bandwidth reduction.
const FIXED_POINT_SCALE: f32 = 1024.0; // 2^10
const FIXED_POINT_SCALE_INV: f32 = 1.0 / 1024.0;

/// Detailed timing breakdown for GPU histogram building.
/// All times are in seconds.
#[derive(Debug, Clone, Default)]
pub struct GpuProfileData {
    /// CPU: Convert indices to u32
    pub indices_convert: Duration,
    /// CPU: Check bins alignment / pack to u32
    pub bins_pack: Duration,
    /// GPU: Ensure buffers are allocated (may include allocation)
    pub buffer_alloc: Duration,
    /// GPU: Upload params to uniform buffer
    pub upload_params: Duration,
    /// GPU: Upload bins (may be skipped if cached)
    pub upload_bins: Duration,
    /// Whether bins upload was skipped due to caching
    pub bins_cached: bool,
    /// GPU: Upload grad/hess
    pub upload_grad_hess: Duration,
    /// GPU: Upload indices
    pub upload_indices: Duration,
    /// GPU: Create bind groups
    pub bind_group_create: Duration,
    /// GPU: Encode zero pass + histogram pass + copy commands
    pub encode_commands: Duration,
    /// GPU: Submit and wait for completion
    pub gpu_execute: Duration,
    /// GPU: Read staging buffers back to CPU
    pub download_results: Duration,
    /// CPU: Convert raw u32 data to Histogram structs
    pub unpack_histograms: Duration,
    /// Total time for the entire operation
    pub total: Duration,
    /// Number of rows processed
    pub num_rows: usize,
    /// Number of features
    pub num_features: usize,
    /// Number of indices (subset size)
    pub num_indices: usize,
}

/// Parameters passed to histogram shader via uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HistogramParams {
    pub num_rows: u32,
    pub num_features: u32,
    pub num_indices: u32, // 0 = use all rows (single batch mode)
    pub num_batches: u32, // 0 or 1 = single batch mode, >1 = batched mode
}

/// Batch descriptor for batched histogram building.
/// Describes a contiguous slice of the concatenated row_indices array.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchInfo {
    pub start: u32, // Start offset in row_indices
    pub count: u32, // Number of rows in this batch
}

/// Pooled buffer that tracks capacity for reuse.
struct PooledBuffer {
    buffer: Buffer,
    capacity: u64,
}

/// Cache key for detecting if data has changed (pointer + length).
/// Uses raw pointer address as a fingerprint - if same pointer and length,
/// data hasn't changed and we can skip the upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheKey {
    ptr: usize,
    len: usize,
}

impl CacheKey {
    fn from_slice<T>(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr() as usize,
            len: slice.len(),
        }
    }
}

/// Buffer pool for reusing GPU allocations.
struct BufferPool {
    // Input buffers
    params: Option<Buffer>,
    bins: Option<PooledBuffer>,
    bins_cache_key: Option<CacheKey>, // Track what bins are currently uploaded (stable across training)
    bins_4bit: Option<PooledBuffer>,  // 4-bit packed bins buffer
    bins_4bit_cache_key: Option<CacheKey>,
    grad_hess: Option<PooledBuffer>,
    indices: Option<PooledBuffer>,
    batch_info: Option<PooledBuffer>, // Batch descriptors for batched mode
    era_indices: Option<PooledBuffer>, // Era indices for DES
    era_indices_cache_key: Option<CacheKey>,
    // Output histogram buffers
    hist_grad: Option<PooledBuffer>,
    hist_hess: Option<PooledBuffer>,
    hist_count: Option<PooledBuffer>,
    // Staging buffers for readback
    staging_grad: Option<PooledBuffer>,
    staging_hess: Option<PooledBuffer>,
    staging_count: Option<PooledBuffer>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            params: None,
            bins: None,
            bins_cache_key: None,
            bins_4bit: None,
            bins_4bit_cache_key: None,
            grad_hess: None,
            indices: None,
            batch_info: None,
            era_indices: None,
            era_indices_cache_key: None,
            hist_grad: None,
            hist_hess: None,
            hist_count: None,
            staging_grad: None,
            staging_hess: None,
            staging_count: None,
        }
    }
}

/// GPU histogram kernel executor.
pub struct HistogramKernel {
    device: Arc<GpuDevice>,
    // Single-batch pipelines (8-bit)
    pipeline_dense: ComputePipeline,
    pipeline_zero: ComputePipeline,
    bind_group_layout_dense: BindGroupLayout,
    bind_group_layout_zero: BindGroupLayout,
    // Batched pipelines (8-bit)
    pipeline_batched: ComputePipeline,
    pipeline_zero_batched: ComputePipeline,
    bind_group_layout_batched: BindGroupLayout,
    // 4-bit bin pipelines (for datasets with max_bins <= 16)
    pipeline_dense_4bit: ComputePipeline,
    pipeline_zero_4bit: ComputePipeline,
    bind_group_layout_dense_4bit: BindGroupLayout,
    // Era histogram pipelines (for Directional Era Splitting)
    pipeline_era: ComputePipeline,
    pipeline_zero_era: ComputePipeline,
    bind_group_layout_era: BindGroupLayout,
    /// Buffer pool for reusing allocations (Mutex for thread safety)
    buffer_pool: Mutex<BufferPool>,
    /// Whether subgroup operations are available on hardware
    subgroups_supported: bool,
    /// Whether to use subgroup operations (default: false)
    /// Set via `set_use_subgroups()`. Only effective if `subgroups_supported` is true.
    use_subgroups: std::sync::atomic::AtomicBool,
    /// Subgroup-optimized pipelines (None if not supported)
    pipeline_dense_subgroups: Option<ComputePipeline>,
    pipeline_batched_subgroups: Option<ComputePipeline>,
    bind_group_layout_dense_subgroups: Option<BindGroupLayout>,
    bind_group_layout_batched_subgroups: Option<BindGroupLayout>,
}

impl HistogramKernel {
    /// Create histogram kernel with compiled shaders.
    pub fn new(device: Arc<GpuDevice>) -> Self {
        // Load shader source
        let shader_source = include_str!("shaders/histogram.wgsl");

        // Create pipelines for each entry point
        let pipeline_dense = device.create_compute_pipeline(
            "histogram_dense_pipeline",
            shader_source,
            "histogram_dense",
        );

        let pipeline_zero = device.create_compute_pipeline(
            "zero_histograms_pipeline",
            shader_source,
            "zero_histograms",
        );

        // Batched pipelines
        let pipeline_batched = device.create_compute_pipeline(
            "histogram_batched_pipeline",
            shader_source,
            "histogram_batched",
        );

        let pipeline_zero_batched = device.create_compute_pipeline(
            "zero_histograms_batched_pipeline",
            shader_source,
            "zero_histograms_batched",
        );

        // Get bind group layouts from each pipeline (they differ in which bindings are used)
        let bind_group_layout_dense = pipeline_dense.get_bind_group_layout(0);
        let bind_group_layout_zero = pipeline_zero.get_bind_group_layout(0);
        let bind_group_layout_batched = pipeline_batched.get_bind_group_layout(0);

        // Create 4-bit pipelines for datasets with small bin counts
        let shader_source_4bit = include_str!("shaders/histogram_4bit.wgsl");

        let pipeline_dense_4bit = device.create_compute_pipeline(
            "histogram_dense_4bit_pipeline",
            shader_source_4bit,
            "histogram_dense_4bit",
        );

        let pipeline_zero_4bit = device.create_compute_pipeline(
            "zero_histograms_4bit_pipeline",
            shader_source_4bit,
            "zero_histograms_4bit",
        );

        let bind_group_layout_dense_4bit = pipeline_dense_4bit.get_bind_group_layout(0);

        // Create era histogram pipelines (for Directional Era Splitting)
        let shader_source_era = include_str!("shaders/histogram_era.wgsl");

        let pipeline_era = device.create_compute_pipeline(
            "histogram_era_pipeline",
            shader_source_era,
            "histogram_era",
        );

        let pipeline_zero_era = device.create_compute_pipeline(
            "zero_histograms_era_pipeline",
            shader_source_era,
            "zero_histograms_era",
        );

        let bind_group_layout_era = pipeline_era.get_bind_group_layout(0);

        // Check for subgroup support and create optimized pipelines if available
        // Note: Even if the device reports SUBGROUP support, the WGSL compiler may not
        // support the `enable subgroups;` extension yet (depends on wgpu/naga version).
        // We try to compile and fall back gracefully if it fails.
        let subgroups_supported = device.subgroups_supported;
        let (
            pipeline_dense_subgroups,
            pipeline_batched_subgroups,
            bind_group_layout_dense_subgroups,
            bind_group_layout_batched_subgroups,
        ) = if subgroups_supported {
            // Try to create subgroup pipelines - may fail if WGSL extension not supported
            let subgroup_shader = include_str!("shaders/histogram_subgroups.wgsl");

            // Use try_create_compute_pipeline which returns Option instead of panicking
            match device.try_create_compute_pipeline(
                "histogram_dense_subgroups_pipeline",
                subgroup_shader,
                "histogram_dense_subgroups",
            ) {
                Some(dense_sg) => {
                    // Dense succeeded, try batched
                    match device.try_create_compute_pipeline(
                        "histogram_batched_subgroups_pipeline",
                        subgroup_shader,
                        "histogram_batched_subgroups",
                    ) {
                        Some(batched_sg) => {
                            let layout_dense_sg = dense_sg.get_bind_group_layout(0);
                            let layout_batched_sg = batched_sg.get_bind_group_layout(0);
                            (
                                Some(dense_sg),
                                Some(batched_sg),
                                Some(layout_dense_sg),
                                Some(layout_batched_sg),
                            )
                        }
                        None => (None, None, None, None),
                    }
                }
                None => (None, None, None, None),
            }
        } else {
            (None, None, None, None)
        };

        // Update subgroups_supported based on whether we actually got working pipelines
        let subgroups_supported = pipeline_dense_subgroups.is_some();

        Self {
            device,
            pipeline_dense,
            pipeline_zero,
            bind_group_layout_dense,
            bind_group_layout_zero,
            pipeline_batched,
            pipeline_zero_batched,
            bind_group_layout_batched,
            pipeline_dense_4bit,
            pipeline_zero_4bit,
            bind_group_layout_dense_4bit,
            pipeline_era,
            pipeline_zero_era,
            bind_group_layout_era,
            buffer_pool: Mutex::new(BufferPool::new()),
            subgroups_supported,
            use_subgroups: std::sync::atomic::AtomicBool::new(false), // Disabled by default
            pipeline_dense_subgroups,
            pipeline_batched_subgroups,
            bind_group_layout_dense_subgroups,
            bind_group_layout_batched_subgroups,
        }
    }

    /// Returns true if subgroup operations are supported by the hardware.
    pub fn subgroups_available(&self) -> bool {
        self.subgroups_supported
    }

    /// Returns true if subgroup operations are enabled and will be used.
    pub fn has_subgroups(&self) -> bool {
        self.subgroups_supported
            && self
                .use_subgroups
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Enable or disable subgroup operations.
    ///
    /// Subgroups are disabled by default. Enable them if you want to test
    /// whether they provide benefit on your hardware.
    ///
    /// Note: Has no effect if hardware doesn't support subgroups.
    pub fn set_use_subgroups(&self, enabled: bool) {
        self.use_subgroups
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Build histograms using the base shader (no subgroups).
    ///
    /// This is primarily for benchmarking to compare subgroup vs non-subgroup performance.
    /// The implementation duplicates `build_histograms` but forces the base pipeline.
    pub fn build_histograms_base_shader(
        &self,
        bins_row_major: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        num_rows: usize,
        num_features: usize,
    ) -> Vec<Histogram> {
        // This is a copy of build_histograms but forces the base pipeline
        // (Could be refactored to share code, but keeping separate for clarity)
        let dev = &self.device;

        let grad_hess_packed: Vec<u32> = grad_hess
            .iter()
            .map(|(g, h)| {
                let grad_i16 = ((*g * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                let hess_i16 = ((*h * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                (grad_i16 as u16 as u32) | ((hess_i16 as u16 as u32) << 16)
            })
            .collect();

        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();

        let bins_aligned = bins_row_major.len().is_multiple_of(4);
        let bins_packed_owned: Vec<u32>;
        let bins_packed: &[u32] = if bins_aligned {
            bytemuck::cast_slice(bins_row_major)
        } else {
            bins_packed_owned = pack_bins_u32(bins_row_major);
            &bins_packed_owned
        };

        let bins_size = (bins_packed.len() * 4) as u64;
        let grad_hess_size = (grad_hess_packed.len() * 4) as u64;
        let indices_size = if indices_u32.is_empty() {
            4u64
        } else {
            (indices_u32.len() * 4) as u64
        };
        let hist_size = (num_features * 256 * 4) as u64;

        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: indices_u32.len() as u32,
            num_batches: 0,
        };

        let mut pool = self.buffer_pool.lock().unwrap();

        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "params_buffer",
                std::mem::size_of::<HistogramParams>() as u64,
            ));
        }

        if Self::ensure_storage_buffer(dev, &mut pool.bins, "bins_buffer", bins_size, false) {
            pool.bins_cache_key = None;
        }
        Self::ensure_storage_buffer(
            dev,
            &mut pool.grad_hess,
            "grad_hess_buffer",
            grad_hess_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.indices,
            "indices_buffer",
            indices_size,
            false,
        );
        Self::ensure_storage_buffer(dev, &mut pool.hist_grad, "hist_grad", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_hess, "hist_hess", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_count, "hist_count", hist_size, true);
        Self::ensure_staging_buffer(dev, &mut pool.staging_grad, "staging_grad", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_hess, "staging_hess", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_count, "staging_count", hist_size);

        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);

        let bins_key = CacheKey::from_slice(bins_row_major);
        if pool.bins_cache_key != Some(bins_key) {
            dev.write_buffer(&pool.bins.as_ref().unwrap().buffer, bins_packed);
            pool.bins_cache_key = Some(bins_key);
        }

        dev.write_buffer(&pool.grad_hess.as_ref().unwrap().buffer, &grad_hess_packed);

        if !indices_u32.is_empty() {
            dev.write_buffer(&pool.indices.as_ref().unwrap().buffer, &indices_u32);
        }

        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group"),
            layout: &self.bind_group_layout_zero,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // FORCE base layout (not subgroup layout)
        let bind_group_dense = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("histogram_bind_group_base"),
            layout: &self.bind_group_layout_dense,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pool.bins.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool.grad_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool.indices.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = dev.create_encoder("histogram_encoder_base");

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let total_bins = (num_features * 256) as u32;
            let workgroups = total_bins.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // FORCE base pipeline (not subgroup pipeline)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_pass_base"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dense);
            pass.set_bind_group(0, &bind_group_dense, &[]);
            pass.dispatch_workgroups(num_features as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &pool.hist_grad.as_ref().unwrap().buffer,
            0,
            &pool.staging_grad.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_hess.as_ref().unwrap().buffer,
            0,
            &pool.staging_hess.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_count.as_ref().unwrap().buffer,
            0,
            &pool.staging_count.as_ref().unwrap().buffer,
            0,
            hist_size,
        );

        dev.submit_and_wait(encoder);

        let mut grad_data = vec![0i32; num_features * 256];
        let mut hess_data = vec![0i32; num_features * 256];
        let mut count_data = vec![0u32; num_features * 256];

        dev.read_buffer(&pool.staging_grad.as_ref().unwrap().buffer, &mut grad_data);
        dev.read_buffer(&pool.staging_hess.as_ref().unwrap().buffer, &mut hess_data);
        dev.read_buffer(
            &pool.staging_count.as_ref().unwrap().buffer,
            &mut count_data,
        );

        drop(pool);

        let mut histograms = Vec::with_capacity(num_features);
        for f in 0..num_features {
            let mut hist = Histogram::new();
            let offset = f * 256;

            for bin in 0..256 {
                let idx = offset + bin;
                let sum_grad = grad_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let sum_hess = hess_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let count = count_data[idx];

                if count > 0 {
                    hist.accumulate(bin as u8, sum_grad, sum_hess);
                    let entry = hist.get_mut(bin as u8);
                    entry.count = count;
                }
            }

            histograms.push(hist);
        }

        histograms
    }

    /// Execute histogram building on GPU.
    ///
    /// # Arguments
    /// * `bins_row_major` - Bin values in row-major order [row][feature], packed as u32
    /// * `grad_hess` - Interleaved gradients and hessians [(g0,h0), (g1,h1), ...]
    /// * `row_indices` - Optional row indices (empty = use all rows)
    /// * `num_rows` - Total number of rows
    /// * `num_features` - Number of features
    ///
    /// # Returns
    /// Vector of histograms, one per feature
    pub fn build_histograms(
        &self,
        bins_row_major: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        num_rows: usize,
        num_features: usize,
    ) -> Vec<Histogram> {
        let dev = &self.device;

        // Quantize gradients/hessians to fixed-point i16 and pack into u32
        // This halves upload bandwidth while using native atomicAdd
        let grad_hess_packed: Vec<u32> = grad_hess
            .iter()
            .map(|(g, h)| {
                // Scale and clamp to i16 range
                let grad_i16 = ((*g * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                let hess_i16 = ((*h * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                // Pack: gradient in low 16 bits, hessian in high 16 bits
                (grad_i16 as u16 as u32) | ((hess_i16 as u16 as u32) << 16)
            })
            .collect();

        // Convert row indices to u32 (still needs allocation, but small)
        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();

        // For bins, try zero-copy cast if aligned, else pack
        let bins_aligned = bins_row_major.len().is_multiple_of(4);
        let bins_packed_owned: Vec<u32>;
        let bins_packed: &[u32] = if bins_aligned {
            // Zero-copy: interpret bytes directly as u32s
            bytemuck::cast_slice(bins_row_major)
        } else {
            // Fallback: pack with padding (rare case)
            bins_packed_owned = pack_bins_u32(bins_row_major);
            &bins_packed_owned
        };

        // Calculate required buffer sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let grad_hess_size = (grad_hess_packed.len() * 4) as u64; // 1 u32 per row (packed i16 pair)
        let indices_size = if indices_u32.is_empty() {
            4u64
        } else {
            (indices_u32.len() * 4) as u64
        };
        let hist_size = (num_features * 256 * 4) as u64;

        // Create uniform buffer with parameters
        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: indices_u32.len() as u32,
            num_batches: 0, // Single batch mode
        };

        // Lock buffer pool and ensure all buffers exist with sufficient capacity
        let mut pool = self.buffer_pool.lock().unwrap();

        // Params buffer (fixed size, create once)
        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "params_buffer",
                std::mem::size_of::<HistogramParams>() as u64,
            ));
        }

        // Ensure input buffers have sufficient capacity (invalidate cache if reallocated)
        if Self::ensure_storage_buffer(dev, &mut pool.bins, "bins_buffer", bins_size, false) {
            pool.bins_cache_key = None; // Buffer was reallocated, must re-upload
        }
        Self::ensure_storage_buffer(
            dev,
            &mut pool.grad_hess,
            "grad_hess_buffer",
            grad_hess_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.indices,
            "indices_buffer",
            indices_size,
            false,
        );

        // Ensure output histogram buffers
        Self::ensure_storage_buffer(dev, &mut pool.hist_grad, "hist_grad", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_hess, "hist_hess", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_count, "hist_count", hist_size, true);

        // Ensure staging buffers
        Self::ensure_staging_buffer(dev, &mut pool.staging_grad, "staging_grad", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_hess, "staging_hess", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_count, "staging_count", hist_size);

        // Write data to buffers (with caching to skip redundant uploads)
        // Params always need updating (num_indices changes per call)
        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);

        // Bins: only upload if data changed (same dataset = same pointer)
        // This works because dataset bins never change during training
        let bins_key = CacheKey::from_slice(bins_row_major);
        if pool.bins_cache_key != Some(bins_key) {
            dev.write_buffer(&pool.bins.as_ref().unwrap().buffer, bins_packed);
            pool.bins_cache_key = Some(bins_key);
        }

        // Grad/Hess: always upload - values change every round even though the
        // slice pointer may stay the same (pointer-based caching would be incorrect)
        dev.write_buffer(&pool.grad_hess.as_ref().unwrap().buffer, &grad_hess_packed);

        // Indices: always upload (different subset each call)
        if !indices_u32.is_empty() {
            dev.write_buffer(&pool.indices.as_ref().unwrap().buffer, &indices_u32);
        }

        // Create bind groups using pooled buffers
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group"),
            layout: &self.bind_group_layout_zero,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // Select the appropriate layout based on whether we're using subgroups
        let dense_layout = if self.has_subgroups() {
            self.bind_group_layout_dense_subgroups.as_ref().unwrap()
        } else {
            &self.bind_group_layout_dense
        };

        let bind_group_dense = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("histogram_bind_group"),
            layout: dense_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pool.bins.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool.grad_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool.indices.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // Execute kernels
        let mut encoder = dev.create_encoder("histogram_encoder");

        // Zero histograms first
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let total_bins = (num_features * 256) as u32;
            let workgroups = total_bins.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Run histogram kernel (use subgroup-optimized version if available)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_pass"),
                timestamp_writes: None,
            });
            if self.has_subgroups() {
                pass.set_pipeline(self.pipeline_dense_subgroups.as_ref().unwrap());
            } else {
                pass.set_pipeline(&self.pipeline_dense);
            }
            pass.set_bind_group(0, &bind_group_dense, &[]);
            pass.dispatch_workgroups(num_features as u32, 1, 1);
        }

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &pool.hist_grad.as_ref().unwrap().buffer,
            0,
            &pool.staging_grad.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_hess.as_ref().unwrap().buffer,
            0,
            &pool.staging_hess.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_count.as_ref().unwrap().buffer,
            0,
            &pool.staging_count.as_ref().unwrap().buffer,
            0,
            hist_size,
        );

        // Submit and wait
        dev.submit_and_wait(encoder);

        // Read back results (as i32 fixed-point)
        let mut grad_data = vec![0i32; num_features * 256];
        let mut hess_data = vec![0i32; num_features * 256];
        let mut count_data = vec![0u32; num_features * 256];

        dev.read_buffer(&pool.staging_grad.as_ref().unwrap().buffer, &mut grad_data);
        dev.read_buffer(&pool.staging_hess.as_ref().unwrap().buffer, &mut hess_data);
        dev.read_buffer(
            &pool.staging_count.as_ref().unwrap().buffer,
            &mut count_data,
        );

        // Drop pool borrow before building histograms
        drop(pool);

        // Convert to Histogram structs with dequantization
        let mut histograms = Vec::with_capacity(num_features);
        for f in 0..num_features {
            let mut hist = Histogram::new();
            let offset = f * 256;

            for bin in 0..256 {
                let idx = offset + bin;
                // Dequantize from fixed-point i32 back to f32
                let sum_grad = grad_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let sum_hess = hess_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let count = count_data[idx];

                if count > 0 {
                    hist.accumulate(bin as u8, sum_grad, sum_hess);
                    // Adjust count (accumulate adds 1, we want the actual count)
                    let entry = hist.get_mut(bin as u8);
                    entry.count = count;
                }
            }

            histograms.push(hist);
        }

        histograms
    }

    /// Build histograms using 4-bit packed bins.
    ///
    /// This is optimized for datasets where all features have ≤16 bins.
    /// Uses nibble-packed bin data for 50% memory bandwidth reduction.
    ///
    /// # Arguments
    /// * `bins_4bit` - 4-bit packed bins in row-major order (2 features per byte)
    /// * `grad_hess` - Interleaved gradients and hessians [(g0,h0), (g1,h1), ...]
    /// * `row_indices` - Optional row indices (empty = use all rows)
    /// * `num_rows` - Total number of rows
    /// * `num_features` - Number of features
    ///
    /// # Returns
    /// Vector of histograms, one per feature
    pub fn build_histograms_4bit(
        &self,
        bins_4bit: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        num_rows: usize,
        num_features: usize,
    ) -> Vec<Histogram> {
        let dev = &self.device;

        // Quantize gradients/hessians to fixed-point i16 and pack into u32
        let grad_hess_packed: Vec<u32> = grad_hess
            .iter()
            .map(|(g, h)| {
                let grad_i16 = ((*g * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                let hess_i16 = ((*h * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                (grad_i16 as u16 as u32) | ((hess_i16 as u16 as u32) << 16)
            })
            .collect();

        // Convert row indices to u32
        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();

        // Pack 4-bit bins for GPU (already nibble-packed, just need u32 alignment)
        let bins_aligned = bins_4bit.len().is_multiple_of(4);
        let bins_packed_owned: Vec<u32>;
        let bins_packed: &[u32] = if bins_aligned {
            bytemuck::cast_slice(bins_4bit)
        } else {
            bins_packed_owned = pack_bins_u32(bins_4bit);
            &bins_packed_owned
        };

        // Calculate required buffer sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let grad_hess_size = (grad_hess_packed.len() * 4) as u64;
        let indices_size = if indices_u32.is_empty() {
            4u64
        } else {
            (indices_u32.len() * 4) as u64
        };
        let hist_size = (num_features * 256 * 4) as u64;

        // Create uniform buffer with parameters
        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: indices_u32.len() as u32,
            num_batches: 0, // Single batch mode
        };

        // Lock buffer pool and ensure all buffers exist
        let mut pool = self.buffer_pool.lock().unwrap();

        // Params buffer (fixed size, create once)
        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "params_buffer",
                std::mem::size_of::<HistogramParams>() as u64,
            ));
        }

        // Ensure 4-bit bins buffer (separate from 8-bit)
        if Self::ensure_storage_buffer(
            dev,
            &mut pool.bins_4bit,
            "bins_4bit_buffer",
            bins_size,
            false,
        ) {
            pool.bins_4bit_cache_key = None;
        }
        Self::ensure_storage_buffer(
            dev,
            &mut pool.grad_hess,
            "grad_hess_buffer",
            grad_hess_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.indices,
            "indices_buffer",
            indices_size,
            false,
        );

        // Ensure output histogram buffers
        Self::ensure_storage_buffer(dev, &mut pool.hist_grad, "hist_grad", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_hess, "hist_hess", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_count, "hist_count", hist_size, true);

        // Ensure staging buffers
        Self::ensure_staging_buffer(dev, &mut pool.staging_grad, "staging_grad", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_hess, "staging_hess", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_count, "staging_count", hist_size);

        // Write data to buffers with caching
        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);

        // 4-bit bins: check cache
        let bins_key = CacheKey::from_slice(bins_4bit);
        if pool.bins_4bit_cache_key != Some(bins_key) {
            dev.write_buffer(&pool.bins_4bit.as_ref().unwrap().buffer, bins_packed);
            pool.bins_4bit_cache_key = Some(bins_key);
        }

        dev.write_buffer(&pool.grad_hess.as_ref().unwrap().buffer, &grad_hess_packed);

        if !indices_u32.is_empty() {
            dev.write_buffer(&pool.indices.as_ref().unwrap().buffer, &indices_u32);
        }

        // Create bind groups using 4-bit layouts
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group_4bit"),
            layout: &self.pipeline_zero_4bit.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_dense = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("histogram_bind_group_4bit"),
            layout: &self.bind_group_layout_dense_4bit,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pool.bins_4bit.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool.grad_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool.indices.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // Execute kernels
        let mut encoder = dev.create_encoder("histogram_encoder_4bit");

        // Zero histograms first
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass_4bit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero_4bit);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let total_bins = (num_features * 256) as u32;
            let workgroups = total_bins.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Run 4-bit histogram kernel
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_pass_4bit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dense_4bit);
            pass.set_bind_group(0, &bind_group_dense, &[]);
            pass.dispatch_workgroups(num_features as u32, 1, 1);
        }

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &pool.hist_grad.as_ref().unwrap().buffer,
            0,
            &pool.staging_grad.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_hess.as_ref().unwrap().buffer,
            0,
            &pool.staging_hess.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_count.as_ref().unwrap().buffer,
            0,
            &pool.staging_count.as_ref().unwrap().buffer,
            0,
            hist_size,
        );

        // Submit and wait
        dev.submit_and_wait(encoder);

        // Read back results (as i32 fixed-point)
        let mut grad_data = vec![0i32; num_features * 256];
        let mut hess_data = vec![0i32; num_features * 256];
        let mut count_data = vec![0u32; num_features * 256];

        dev.read_buffer(&pool.staging_grad.as_ref().unwrap().buffer, &mut grad_data);
        dev.read_buffer(&pool.staging_hess.as_ref().unwrap().buffer, &mut hess_data);
        dev.read_buffer(
            &pool.staging_count.as_ref().unwrap().buffer,
            &mut count_data,
        );

        // Drop pool borrow
        drop(pool);

        // Convert to Histogram structs with dequantization
        let mut histograms = Vec::with_capacity(num_features);
        for f in 0..num_features {
            let mut hist = Histogram::new();
            let offset = f * 256;

            // Only check first 16 bins for 4-bit mode
            for bin in 0..16 {
                let idx = offset + bin;
                let sum_grad = grad_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let sum_hess = hess_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let count = count_data[idx];

                if count > 0 {
                    hist.accumulate(bin as u8, sum_grad, sum_hess);
                    let entry = hist.get_mut(bin as u8);
                    entry.count = count;
                }
            }

            histograms.push(hist);
        }

        histograms
    }

    /// Build histograms with detailed profiling.
    ///
    /// Returns both the histograms and detailed timing breakdown for each step.
    /// Use this to understand GPU overhead and identify optimization opportunities.
    pub fn build_histograms_profiled(
        &self,
        bins_row_major: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        num_rows: usize,
        num_features: usize,
    ) -> (Vec<Histogram>, GpuProfileData) {
        let total_start = Instant::now();
        let mut profile = GpuProfileData {
            num_rows,
            num_features,
            num_indices: row_indices.len(),
            ..Default::default()
        };

        let dev = &self.device;

        // Quantize gradients/hessians to fixed-point i16 and pack into u32
        let grad_hess_packed: Vec<u32> = grad_hess
            .iter()
            .map(|(g, h)| {
                let grad_i16 = ((*g * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                let hess_i16 = ((*h * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                (grad_i16 as u16 as u32) | ((hess_i16 as u16 as u32) << 16)
            })
            .collect();

        // Convert row indices to u32
        let t = Instant::now();
        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();
        profile.indices_convert = t.elapsed();

        // Pack bins
        let t = Instant::now();
        let bins_aligned = bins_row_major.len().is_multiple_of(4);
        let bins_packed_owned: Vec<u32>;
        let bins_packed: &[u32] = if bins_aligned {
            bytemuck::cast_slice(bins_row_major)
        } else {
            bins_packed_owned = pack_bins_u32(bins_row_major);
            &bins_packed_owned
        };
        profile.bins_pack = t.elapsed();

        // Calculate sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let grad_hess_size = (grad_hess_packed.len() * 4) as u64; // 1 u32 per row (packed i16 pair)
        let indices_size = if indices_u32.is_empty() {
            4u64
        } else {
            (indices_u32.len() * 4) as u64
        };
        let hist_size = (num_features * 256 * 4) as u64;

        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: indices_u32.len() as u32,
            num_batches: 0, // Single batch mode
        };

        // Buffer allocation
        let t = Instant::now();
        let mut pool = self.buffer_pool.lock().unwrap();

        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "params_buffer",
                std::mem::size_of::<HistogramParams>() as u64,
            ));
        }

        if Self::ensure_storage_buffer(dev, &mut pool.bins, "bins_buffer", bins_size, false) {
            pool.bins_cache_key = None;
        }
        Self::ensure_storage_buffer(
            dev,
            &mut pool.grad_hess,
            "grad_hess_buffer",
            grad_hess_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.indices,
            "indices_buffer",
            indices_size,
            false,
        );
        Self::ensure_storage_buffer(dev, &mut pool.hist_grad, "hist_grad", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_hess, "hist_hess", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_count, "hist_count", hist_size, true);
        Self::ensure_staging_buffer(dev, &mut pool.staging_grad, "staging_grad", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_hess, "staging_hess", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_count, "staging_count", hist_size);
        profile.buffer_alloc = t.elapsed();

        // Upload params
        let t = Instant::now();
        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);
        profile.upload_params = t.elapsed();

        // Upload bins (check cache)
        let t = Instant::now();
        let bins_key = CacheKey::from_slice(bins_row_major);
        if pool.bins_cache_key != Some(bins_key) {
            dev.write_buffer(&pool.bins.as_ref().unwrap().buffer, bins_packed);
            pool.bins_cache_key = Some(bins_key);
            profile.bins_cached = false;
        } else {
            profile.bins_cached = true;
        }
        profile.upload_bins = t.elapsed();

        // Upload grad/hess (packed i16 pairs)
        let t = Instant::now();
        dev.write_buffer(&pool.grad_hess.as_ref().unwrap().buffer, &grad_hess_packed);
        profile.upload_grad_hess = t.elapsed();

        // Upload indices
        let t = Instant::now();
        if !indices_u32.is_empty() {
            dev.write_buffer(&pool.indices.as_ref().unwrap().buffer, &indices_u32);
        }
        profile.upload_indices = t.elapsed();

        // Create bind groups
        let t = Instant::now();
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group"),
            layout: &self.bind_group_layout_zero,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // Select the appropriate layout based on whether we're using subgroups
        let dense_layout = if self.has_subgroups() {
            self.bind_group_layout_dense_subgroups.as_ref().unwrap()
        } else {
            &self.bind_group_layout_dense
        };

        let bind_group_dense = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("histogram_bind_group"),
            layout: dense_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pool.bins.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool.grad_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool.indices.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });
        profile.bind_group_create = t.elapsed();

        // Encode commands
        let t = Instant::now();
        let mut encoder = dev.create_encoder("histogram_encoder");

        // Zero histograms
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let total_bins = (num_features * 256) as u32;
            let workgroups = total_bins.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Histogram kernel (use subgroup-optimized version if available)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_pass"),
                timestamp_writes: None,
            });
            if self.has_subgroups() {
                pass.set_pipeline(self.pipeline_dense_subgroups.as_ref().unwrap());
            } else {
                pass.set_pipeline(&self.pipeline_dense);
            }
            pass.set_bind_group(0, &bind_group_dense, &[]);
            pass.dispatch_workgroups(num_features as u32, 1, 1);
        }

        // Copy to staging
        encoder.copy_buffer_to_buffer(
            &pool.hist_grad.as_ref().unwrap().buffer,
            0,
            &pool.staging_grad.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_hess.as_ref().unwrap().buffer,
            0,
            &pool.staging_hess.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_count.as_ref().unwrap().buffer,
            0,
            &pool.staging_count.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        profile.encode_commands = t.elapsed();

        // Submit and wait
        let t = Instant::now();
        dev.submit_and_wait(encoder);
        profile.gpu_execute = t.elapsed();

        // Download results (as i32 fixed-point)
        let t = Instant::now();
        let mut grad_data = vec![0i32; num_features * 256];
        let mut hess_data = vec![0i32; num_features * 256];
        let mut count_data = vec![0u32; num_features * 256];

        dev.read_buffer(&pool.staging_grad.as_ref().unwrap().buffer, &mut grad_data);
        dev.read_buffer(&pool.staging_hess.as_ref().unwrap().buffer, &mut hess_data);
        dev.read_buffer(
            &pool.staging_count.as_ref().unwrap().buffer,
            &mut count_data,
        );
        profile.download_results = t.elapsed();

        drop(pool);

        // Unpack histograms with dequantization
        let t = Instant::now();
        let mut histograms = Vec::with_capacity(num_features);
        for f in 0..num_features {
            let mut hist = Histogram::new();
            let offset = f * 256;

            for bin in 0..256 {
                let idx = offset + bin;
                // Dequantize from fixed-point i32 back to f32
                let sum_grad = grad_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let sum_hess = hess_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                let count = count_data[idx];

                if count > 0 {
                    hist.accumulate(bin as u8, sum_grad, sum_hess);
                    let entry = hist.get_mut(bin as u8);
                    entry.count = count;
                }
            }

            histograms.push(hist);
        }
        profile.unpack_histograms = t.elapsed();

        profile.total = total_start.elapsed();

        (histograms, profile)
    }

    /// Ensure a storage buffer exists with sufficient capacity.
    /// Returns true if buffer was (re)allocated (cache should be invalidated).
    fn ensure_storage_buffer(
        dev: &GpuDevice,
        pool: &mut Option<PooledBuffer>,
        label: &str,
        required_size: u64,
        read_write: bool,
    ) -> bool {
        let needs_new = match pool {
            Some(ref pb) => pb.capacity < required_size,
            None => true,
        };

        if needs_new {
            // Allocate with 20% headroom, aligned to 4 bytes
            let capacity = ((required_size as f64 * 1.2) as u64 + 3) & !3;
            let buffer = dev.create_storage_buffer(label, capacity, read_write);
            *pool = Some(PooledBuffer { buffer, capacity });
            true
        } else {
            false
        }
    }

    /// Ensure a staging buffer exists with sufficient capacity.
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
            // Allocate with 20% headroom, aligned to 4 bytes
            let capacity = ((required_size as f64 * 1.2) as u64 + 3) & !3;
            let buffer = dev.create_staging_buffer(label, capacity);
            *pool = Some(PooledBuffer { buffer, capacity });
        }
    }

    /// Build histograms for multiple batches in a single GPU dispatch.
    ///
    /// This is significantly more efficient than calling `build_histograms` multiple times
    /// because it amortizes GPU dispatch overhead across all batches.
    ///
    /// # Arguments
    ///
    /// * `bins_row_major` - Row-major bin data for entire dataset
    /// * `grad_hess` - Gradient/hessian pairs for entire dataset
    /// * `batches` - Slice of row index arrays, one per batch
    /// * `num_rows` - Total number of rows in dataset
    /// * `num_features` - Number of features
    ///
    /// # Returns
    ///
    /// Vector of histogram vectors, one per batch. Each inner vector contains
    /// `num_features` histograms.
    pub fn build_histograms_batched(
        &self,
        bins_row_major: &[u8],
        grad_hess: &[(f32, f32)],
        batches: &[&[usize]],
        num_rows: usize,
        num_features: usize,
    ) -> Vec<Vec<Histogram>> {
        let num_batches = batches.len();
        if num_batches == 0 {
            return Vec::new();
        }

        // For single batch, use the optimized single-batch path
        if num_batches == 1 {
            return vec![self.build_histograms(
                bins_row_major,
                grad_hess,
                batches[0],
                num_rows,
                num_features,
            )];
        }

        let dev = &self.device;

        // Quantize gradients/hessians to fixed-point i16 and pack into u32
        let grad_hess_packed: Vec<u32> = grad_hess
            .iter()
            .map(|(g, h)| {
                let grad_i16 = ((*g * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                let hess_i16 = ((*h * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                (grad_i16 as u16 as u32) | ((hess_i16 as u16 as u32) << 16)
            })
            .collect();

        // Concatenate all batch row indices and create batch info
        let mut all_indices: Vec<u32> = Vec::new();
        let mut batch_info_data: Vec<BatchInfo> = Vec::with_capacity(num_batches);

        for batch_indices in batches {
            let start = all_indices.len() as u32;
            let count = batch_indices.len() as u32;
            batch_info_data.push(BatchInfo { start, count });
            all_indices.extend(batch_indices.iter().map(|&i| i as u32));
        }

        // For bins, try zero-copy cast if aligned, else pack
        let bins_aligned = bins_row_major.len().is_multiple_of(4);
        let bins_packed_owned: Vec<u32>;
        let bins_packed: &[u32] = if bins_aligned {
            bytemuck::cast_slice(bins_row_major)
        } else {
            bins_packed_owned = pack_bins_u32(bins_row_major);
            &bins_packed_owned
        };

        // Calculate buffer sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let grad_hess_size = (grad_hess_packed.len() * 4) as u64; // 1 u32 per row (packed i16 pair)
        let indices_size = if all_indices.is_empty() {
            4u64
        } else {
            (all_indices.len() * 4) as u64
        };
        let batch_info_size = (batch_info_data.len() * std::mem::size_of::<BatchInfo>()) as u64;
        let hist_size = (num_batches * num_features * 256 * 4) as u64;

        // Create uniform buffer with parameters
        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: all_indices.len() as u32,
            num_batches: num_batches as u32,
        };

        // Lock buffer pool and ensure all buffers exist
        let mut pool = self.buffer_pool.lock().unwrap();

        // Ensure params buffer
        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "params_buffer",
                std::mem::size_of::<HistogramParams>() as u64,
            ));
        }

        // Ensure input buffers
        Self::ensure_storage_buffer(dev, &mut pool.bins, "bins_buffer", bins_size, false);
        Self::ensure_storage_buffer(
            dev,
            &mut pool.grad_hess,
            "grad_hess_buffer",
            grad_hess_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.indices,
            "indices_buffer",
            indices_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.batch_info,
            "batch_info_buffer",
            batch_info_size,
            false,
        );

        // Ensure output buffers (sized for all batches)
        Self::ensure_storage_buffer(
            dev,
            &mut pool.hist_grad,
            "hist_grad_buffer",
            hist_size,
            true,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.hist_hess,
            "hist_hess_buffer",
            hist_size,
            true,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.hist_count,
            "hist_count_buffer",
            hist_size,
            true,
        );

        // Ensure staging buffers
        Self::ensure_staging_buffer(dev, &mut pool.staging_grad, "staging_grad", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_hess, "staging_hess", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_count, "staging_count", hist_size);

        // Upload data
        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);
        dev.write_buffer(&pool.bins.as_ref().unwrap().buffer, bins_packed);
        dev.write_buffer(&pool.grad_hess.as_ref().unwrap().buffer, &grad_hess_packed);
        dev.write_buffer(&pool.indices.as_ref().unwrap().buffer, &all_indices);
        dev.write_buffer(&pool.batch_info.as_ref().unwrap().buffer, &batch_info_data);

        // Select the appropriate layout based on whether we're using subgroups
        let batched_layout = if self.pipeline_batched_subgroups.is_some() {
            self.bind_group_layout_batched_subgroups.as_ref().unwrap()
        } else {
            &self.bind_group_layout_batched
        };

        // Create bind groups for batched kernel
        let bind_group_batched = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("batched_bind_group"),
            layout: batched_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pool.bins.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool.grad_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool.indices.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: pool.batch_info.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // Zero bind group for batched kernel (use batched zero pipeline's layout)
        let bind_group_layout_zero_batched = self.pipeline_zero_batched.get_bind_group_layout(0);
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group_batched"),
            layout: &bind_group_layout_zero_batched,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        // Execute kernels
        let mut encoder = dev.create_encoder("histogram_batched_encoder");

        // Zero histograms first (for all batches)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass_batched"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero_batched);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let total_bins = (num_batches * num_features * 256) as u32;
            let workgroups = total_bins.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Run batched histogram kernel (use subgroup-optimized version if available)
        // Dispatch: (num_features, num_batches, 1)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_batched_pass"),
                timestamp_writes: None,
            });
            if let Some(ref sg_pipeline) = self.pipeline_batched_subgroups {
                pass.set_pipeline(sg_pipeline);
            } else {
                pass.set_pipeline(&self.pipeline_batched);
            }
            pass.set_bind_group(0, &bind_group_batched, &[]);
            pass.dispatch_workgroups(num_features as u32, num_batches as u32, 1);
        }

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &pool.hist_grad.as_ref().unwrap().buffer,
            0,
            &pool.staging_grad.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_hess.as_ref().unwrap().buffer,
            0,
            &pool.staging_hess.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_count.as_ref().unwrap().buffer,
            0,
            &pool.staging_count.as_ref().unwrap().buffer,
            0,
            hist_size,
        );

        // Submit and wait
        dev.submit_and_wait(encoder);

        // Read back results (as i32 fixed-point)
        let total_hist_entries = num_batches * num_features * 256;
        let mut grad_data = vec![0i32; total_hist_entries];
        let mut hess_data = vec![0i32; total_hist_entries];
        let mut count_data = vec![0u32; total_hist_entries];

        dev.read_buffer(&pool.staging_grad.as_ref().unwrap().buffer, &mut grad_data);
        dev.read_buffer(&pool.staging_hess.as_ref().unwrap().buffer, &mut hess_data);
        dev.read_buffer(
            &pool.staging_count.as_ref().unwrap().buffer,
            &mut count_data,
        );

        // Drop pool borrow
        drop(pool);

        // Convert to Histogram structs with dequantization - one Vec<Histogram> per batch
        // Output layout: [batch * num_features * 256 + feature * 256 + bin]
        let hist_stride = num_features * 256;
        let mut all_histograms = Vec::with_capacity(num_batches);

        for batch in 0..num_batches {
            let batch_offset = batch * hist_stride;
            let mut batch_histograms = Vec::with_capacity(num_features);

            for f in 0..num_features {
                let mut hist = Histogram::new();
                let feature_offset = batch_offset + f * 256;

                for bin in 0..256 {
                    let idx = feature_offset + bin;
                    // Dequantize from fixed-point i32 back to f32
                    let sum_grad = grad_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                    let sum_hess = hess_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                    let count = count_data[idx];

                    if count > 0 {
                        hist.accumulate(bin as u8, sum_grad, sum_hess);
                        let entry = hist.get_mut(bin as u8);
                        entry.count = count;
                    }
                }

                batch_histograms.push(hist);
            }

            all_histograms.push(batch_histograms);
        }

        all_histograms
    }

    /// Build era-stratified histograms for Directional Era Splitting (DES).
    ///
    /// Returns histograms indexed as `[era][feature]`.
    ///
    /// # Arguments
    /// * `bins_row_major` - Bin values in row-major order [row][feature]
    /// * `grad_hess` - Interleaved gradients and hessians [(g0,h0), (g1,h1), ...]
    /// * `row_indices` - Row indices to process (empty = use all rows)
    /// * `era_indices` - Era index for each row in the full dataset
    /// * `num_rows` - Total number of rows
    /// * `num_features` - Number of features
    /// * `num_eras` - Number of unique eras
    ///
    /// # Returns
    /// 2D vector of histograms: `[num_eras][num_features]`
    // reason: kernel/training entry point with many parameters
    #[allow(clippy::too_many_arguments)]
    pub fn build_era_histograms(
        &self,
        bins_row_major: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        era_indices: &[u16],
        num_rows: usize,
        num_features: usize,
        num_eras: usize,
    ) -> Vec<Vec<Histogram>> {
        let dev = &self.device;

        // Quantize gradients/hessians to fixed-point i16 and pack into u32
        let grad_hess_packed: Vec<u32> = grad_hess
            .iter()
            .map(|(g, h)| {
                let grad_i16 = ((*g * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                let hess_i16 = ((*h * FIXED_POINT_SCALE).clamp(-32767.0, 32767.0)) as i16;
                (grad_i16 as u16 as u32) | ((hess_i16 as u16 as u32) << 16)
            })
            .collect();

        // Convert row indices to u32
        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();

        // Pack era indices (u16) into u32 (2 per u32)
        let era_packed: Vec<u32> = era_indices
            .chunks(2)
            .map(|chunk| {
                let e0 = chunk[0] as u32;
                let e1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
                e0 | (e1 << 16)
            })
            .collect();

        // Pack bins
        let bins_aligned = bins_row_major.len().is_multiple_of(4);
        let bins_packed_owned: Vec<u32>;
        let bins_packed: &[u32] = if bins_aligned {
            bytemuck::cast_slice(bins_row_major)
        } else {
            bins_packed_owned = pack_bins_u32(bins_row_major);
            &bins_packed_owned
        };

        // Calculate buffer sizes
        let bins_size = (bins_packed.len() * 4) as u64;
        let grad_hess_size = (grad_hess_packed.len() * 4) as u64;
        let indices_size = if indices_u32.is_empty() {
            4u64
        } else {
            (indices_u32.len() * 4) as u64
        };
        let era_size = (era_packed.len() * 4) as u64;
        let hist_size = (num_eras * num_features * 256 * 4) as u64;

        // Create uniform buffer with parameters (reusing HistogramParams with num_eras in num_batches field)
        let params = HistogramParams {
            num_rows: num_rows as u32,
            num_features: num_features as u32,
            num_indices: indices_u32.len() as u32,
            num_batches: num_eras as u32, // Repurposed as num_eras for era shader
        };

        // Lock buffer pool and ensure all buffers exist
        let mut pool = self.buffer_pool.lock().unwrap();

        // Params buffer
        if pool.params.is_none() {
            pool.params = Some(dev.create_uniform_buffer(
                "params_buffer",
                std::mem::size_of::<HistogramParams>() as u64,
            ));
        }

        // Ensure input buffers
        if Self::ensure_storage_buffer(dev, &mut pool.bins, "bins_buffer", bins_size, false) {
            pool.bins_cache_key = None;
        }
        Self::ensure_storage_buffer(
            dev,
            &mut pool.grad_hess,
            "grad_hess_buffer",
            grad_hess_size,
            false,
        );
        Self::ensure_storage_buffer(
            dev,
            &mut pool.indices,
            "indices_buffer",
            indices_size,
            false,
        );
        if Self::ensure_storage_buffer(
            dev,
            &mut pool.era_indices,
            "era_indices_buffer",
            era_size,
            false,
        ) {
            pool.era_indices_cache_key = None;
        }

        // Ensure output histogram buffers (sized for all eras)
        Self::ensure_storage_buffer(dev, &mut pool.hist_grad, "hist_grad", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_hess, "hist_hess", hist_size, true);
        Self::ensure_storage_buffer(dev, &mut pool.hist_count, "hist_count", hist_size, true);

        // Ensure staging buffers
        Self::ensure_staging_buffer(dev, &mut pool.staging_grad, "staging_grad", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_hess, "staging_hess", hist_size);
        Self::ensure_staging_buffer(dev, &mut pool.staging_count, "staging_count", hist_size);

        // Upload data
        dev.write_buffer(pool.params.as_ref().unwrap(), &[params]);

        // Bins: check cache
        let bins_key = CacheKey::from_slice(bins_row_major);
        if pool.bins_cache_key != Some(bins_key) {
            dev.write_buffer(&pool.bins.as_ref().unwrap().buffer, bins_packed);
            pool.bins_cache_key = Some(bins_key);
        }

        dev.write_buffer(&pool.grad_hess.as_ref().unwrap().buffer, &grad_hess_packed);

        if !indices_u32.is_empty() {
            dev.write_buffer(&pool.indices.as_ref().unwrap().buffer, &indices_u32);
        }

        // Era indices: check cache
        let era_key = CacheKey::from_slice(era_indices);
        if pool.era_indices_cache_key != Some(era_key) {
            dev.write_buffer(&pool.era_indices.as_ref().unwrap().buffer, &era_packed);
            pool.era_indices_cache_key = Some(era_key);
        }

        // Create bind groups for era histogram kernel
        let bind_group_zero = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("zero_bind_group_era"),
            layout: &self.pipeline_zero_era.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_era = dev.device.create_bind_group(&BindGroupDescriptor {
            label: Some("histogram_bind_group_era"),
            layout: &self.bind_group_layout_era,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pool.params.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pool.bins.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pool.grad_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool.indices.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: pool.hist_grad.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: pool.hist_hess.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: pool.hist_count.as_ref().unwrap().buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: pool
                        .era_indices
                        .as_ref()
                        .unwrap()
                        .buffer
                        .as_entire_binding(),
                },
            ],
        });

        // Execute kernels
        let mut encoder = dev.create_encoder("histogram_era_encoder");

        // Zero histograms first (for all eras)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass_era"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_zero_era);
            pass.set_bind_group(0, &bind_group_zero, &[]);
            let total_bins = (num_eras * num_features * 256) as u32;
            let workgroups = total_bins.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Run era histogram kernel
        // Dispatch: (num_features, num_eras, 1)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_era_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_era);
            pass.set_bind_group(0, &bind_group_era, &[]);
            pass.dispatch_workgroups(num_features as u32, num_eras as u32, 1);
        }

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &pool.hist_grad.as_ref().unwrap().buffer,
            0,
            &pool.staging_grad.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_hess.as_ref().unwrap().buffer,
            0,
            &pool.staging_hess.as_ref().unwrap().buffer,
            0,
            hist_size,
        );
        encoder.copy_buffer_to_buffer(
            &pool.hist_count.as_ref().unwrap().buffer,
            0,
            &pool.staging_count.as_ref().unwrap().buffer,
            0,
            hist_size,
        );

        // Submit and wait
        dev.submit_and_wait(encoder);

        // Read back results (as i32 fixed-point)
        let total_hist_entries = num_eras * num_features * 256;
        let mut grad_data = vec![0i32; total_hist_entries];
        let mut hess_data = vec![0i32; total_hist_entries];
        let mut count_data = vec![0u32; total_hist_entries];

        dev.read_buffer(&pool.staging_grad.as_ref().unwrap().buffer, &mut grad_data);
        dev.read_buffer(&pool.staging_hess.as_ref().unwrap().buffer, &mut hess_data);
        dev.read_buffer(
            &pool.staging_count.as_ref().unwrap().buffer,
            &mut count_data,
        );

        // Drop pool borrow
        drop(pool);

        // Convert to Histogram structs with dequantization - [era][feature]
        // Output layout: [era * num_features * 256 + feature * 256 + bin]
        let hist_stride = num_features * 256;
        let mut all_histograms = Vec::with_capacity(num_eras);

        for era in 0..num_eras {
            let era_offset = era * hist_stride;
            let mut era_histograms = Vec::with_capacity(num_features);

            for f in 0..num_features {
                let mut hist = Histogram::new();
                let feature_offset = era_offset + f * 256;

                for bin in 0..256 {
                    let idx = feature_offset + bin;
                    // Dequantize from fixed-point i32 back to f32
                    let sum_grad = grad_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                    let sum_hess = hess_data[idx] as f32 * FIXED_POINT_SCALE_INV;
                    let count = count_data[idx];

                    if count > 0 {
                        hist.accumulate(bin as u8, sum_grad, sum_hess);
                        let entry = hist.get_mut(bin as u8);
                        entry.count = count;
                    }
                }

                era_histograms.push(hist);
            }

            all_histograms.push(era_histograms);
        }

        all_histograms
    }
}

/// Pack u8 bins into u32 array (4 bytes per u32).
fn pack_bins_u32(bins: &[u8]) -> Vec<u32> {
    let mut packed = vec![0u32; bins.len().div_ceil(4)];
    for (i, &bin) in bins.iter().enumerate() {
        let word_idx = i / 4;
        let byte_idx = i % 4;
        packed[word_idx] |= (bin as u32) << (byte_idx * 8);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_bins_u32() {
        let bins = vec![0u8, 1, 2, 3, 4, 5];
        let packed = pack_bins_u32(&bins);

        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 0x03020100); // Little-endian: [0,1,2,3]
        assert_eq!(packed[1] & 0xFFFF, 0x0504); // [4,5,0,0]
    }
}
