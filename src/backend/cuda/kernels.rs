//! CUDA histogram kernel implementation.

use super::device::CudaDevice;
use crate::histogram::Histogram;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Cache key for bins data (hash-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheKey(u64);

impl CacheKey {
    pub(crate) fn from_slice(data: &[u8]) -> Self {
        let mut hasher = DefaultHasher::new();
        data.len().hash(&mut hasher);
        // Hash first, middle, and last chunks for speed
        if !data.is_empty() {
            let chunk_size = 1024.min(data.len());
            data[..chunk_size].hash(&mut hasher);
            if data.len() > chunk_size * 2 {
                let mid = data.len() / 2;
                data[mid..mid + chunk_size.min(data.len() - mid)].hash(&mut hasher);
            }
            if data.len() > chunk_size {
                data[data.len() - chunk_size..].hash(&mut hasher);
            }
        }
        CacheKey(hasher.finish())
    }
}

/// CUDA source for histogram building kernels.
/// Loaded from kernels/histogram.cu for easier debugging.
const HISTOGRAM_KERNEL_SOURCE: &str = include_str!("kernels/histogram.cu");

/// Threads per block for histogram kernel
const THREADS_PER_BLOCK: u32 = 256;

/// Node descriptor for GPU-resident batched histogram building.
/// Describes a range of indices in the GPU indices buffer.
#[derive(Debug, Clone, Copy)]
pub struct NodeRange {
    pub start: u32,
    pub count: u32,
}

/// CUDA histogram kernel executor.
pub struct HistogramKernel {
    device: Arc<CudaDevice>,
    module: Option<Arc<CudaModule>>,
    build_histogram_fn: Option<CudaFunction>,
    build_histogram_batched_fn: Option<CudaFunction>,
    build_histogram_era_fn: Option<CudaFunction>,
    zero_histograms_fn: Option<CudaFunction>,

    // Cached GPU buffers
    cached_bins: Option<CudaSlice<u8>>,
    cached_bins_key: Option<CacheKey>,
    cached_gradients: Option<CudaSlice<f32>>,
    cached_hessians: Option<CudaSlice<f32>>,
    cached_grad_hess_key: Option<CacheKey>,
    // Era indices buffer
    cached_era_indices: Option<CudaSlice<u16>>,
    cached_era_indices_key: Option<CacheKey>,
    // Output histograms (separate for f32 atomics)
    cached_grad_hist: Option<CudaSlice<f32>>,
    cached_hess_hist: Option<CudaSlice<f32>>,
    cached_count_hist: Option<CudaSlice<u32>>,
    cached_output_len: usize,
    // Batched histogram output (larger buffer for multiple nodes)
    cached_batched_grad_hist: Option<CudaSlice<f32>>,
    cached_batched_hess_hist: Option<CudaSlice<f32>>,
    cached_batched_count_hist: Option<CudaSlice<u32>>,
    cached_batched_output_len: usize,
    // Era histogram output (for DES)
    cached_era_grad_hist: Option<CudaSlice<f32>>,
    cached_era_hess_hist: Option<CudaSlice<f32>>,
    cached_era_count_hist: Option<CudaSlice<u32>>,
    cached_era_output_len: usize,
}

impl HistogramKernel {
    /// Create a new histogram kernel.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            module: None,
            build_histogram_fn: None,
            build_histogram_batched_fn: None,
            build_histogram_era_fn: None,
            zero_histograms_fn: None,
            cached_bins: None,
            cached_bins_key: None,
            cached_gradients: None,
            cached_hessians: None,
            cached_grad_hess_key: None,
            cached_era_indices: None,
            cached_era_indices_key: None,
            cached_grad_hist: None,
            cached_hess_hist: None,
            cached_count_hist: None,
            cached_output_len: 0,
            cached_batched_grad_hist: None,
            cached_batched_hess_hist: None,
            cached_batched_count_hist: None,
            cached_batched_output_len: 0,
            cached_era_grad_hist: None,
            cached_era_hess_hist: None,
            cached_era_count_hist: None,
            cached_era_output_len: 0,
        }
    }

    /// Ensure the kernel is compiled and loaded.
    fn ensure_initialized(&mut self) {
        if self.module.is_some() {
            return;
        }

        let ptx = compile_ptx(HISTOGRAM_KERNEL_SOURCE).expect("Failed to compile CUDA kernel");
        let module = self.device.load_module(ptx);

        self.build_histogram_fn = Some(CudaDevice::load_function(&module, "build_histogram"));
        self.build_histogram_batched_fn = Some(CudaDevice::load_function(
            &module,
            "build_histogram_batched",
        ));
        self.build_histogram_era_fn =
            Some(CudaDevice::load_function(&module, "build_histogram_era"));
        self.zero_histograms_fn = Some(CudaDevice::load_function(&module, "zero_histograms"));
        self.module = Some(module);
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

    /// Ensure gradients and hessians are cached on GPU.
    pub fn ensure_grad_hess_cached(&mut self, gradients: &[f32], hessians: &[f32]) {
        let grad_key = {
            let mut hasher = DefaultHasher::new();
            gradients.len().hash(&mut hasher);
            if !gradients.is_empty() {
                gradients[0].to_bits().hash(&mut hasher);
                if gradients.len() > 1 {
                    gradients[gradients.len() / 2].to_bits().hash(&mut hasher);
                }
            }
            CacheKey(hasher.finish())
        };

        if self.cached_grad_hess_key != Some(grad_key)
            || self.cached_gradients.is_none()
            || self.cached_hessians.is_none()
        {
            self.cached_gradients = Some(self.device.htod_copy(gradients));
            self.cached_hessians = Some(self.device.htod_copy(hessians));
            self.cached_grad_hess_key = Some(grad_key);
        }
    }

    /// Get cached bins buffer (for partition kernel).
    pub fn cached_bins(&self) -> Option<&CudaSlice<u8>> {
        self.cached_bins.as_ref()
    }

    /// Build histograms for all features using 2D tiled kernel.
    pub fn build_histograms(
        &mut self,
        bins: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        num_rows: usize,
        num_features: usize,
    ) -> Vec<Histogram> {
        self.ensure_initialized();

        if row_indices.is_empty() || num_features == 0 {
            return (0..num_features).map(|_| Histogram::new()).collect();
        }

        let num_indices = row_indices.len();

        // Separate gradients and hessians for f32 atomics
        let gradients: Vec<f32> = grad_hess.iter().map(|(g, _)| *g).collect();
        let hessians: Vec<f32> = grad_hess.iter().map(|(_, h)| *h).collect();

        // Convert row indices to u32
        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();

        // Upload bins (with caching)
        let bins_key = CacheKey::from_slice(bins);
        if self.cached_bins_key != Some(bins_key) || self.cached_bins.is_none() {
            self.cached_bins = Some(self.device.htod_copy(bins));
            self.cached_bins_key = Some(bins_key);
        }

        // Upload gradients/hessians (with caching - don't change within a tree)
        // Use a simple hash of first few values as cache key
        let grad_key = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&gradients.len(), &mut hasher);
            if !gradients.is_empty() {
                std::hash::Hash::hash(&gradients[0].to_bits(), &mut hasher);
                if gradients.len() > 1 {
                    std::hash::Hash::hash(&gradients[gradients.len() / 2].to_bits(), &mut hasher);
                }
            }
            CacheKey(std::hash::Hasher::finish(&hasher))
        };

        if self.cached_grad_hess_key != Some(grad_key)
            || self.cached_gradients.is_none()
            || self.cached_hessians.is_none()
        {
            self.cached_gradients = Some(self.device.htod_copy(&gradients));
            self.cached_hessians = Some(self.device.htod_copy(&hessians));
            self.cached_grad_hess_key = Some(grad_key);
        }

        // Upload indices (always needed fresh)
        let d_indices = self.device.htod_copy(&indices_u32);

        // Allocate output buffers (reuse if same size)
        let output_bins = num_features * 256;
        if self.cached_output_len != output_bins
            || self.cached_grad_hist.is_none()
            || self.cached_hess_hist.is_none()
            || self.cached_count_hist.is_none()
        {
            self.cached_grad_hist = Some(self.device.alloc_zeros(output_bins));
            self.cached_hess_hist = Some(self.device.alloc_zeros(output_bins));
            self.cached_count_hist = Some(self.device.alloc_zeros(output_bins));
            self.cached_output_len = output_bins;
        }

        let stream = self.device.stream();

        // Zero output histograms
        let zero_blocks = output_bins.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_grad_hist = self.cached_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_count_hist.as_mut().unwrap();
            stream
                .launch_builder(self.zero_histograms_fn.as_ref().unwrap())
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(output_bins as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_histograms kernel");
        }

        // Calculate 2D grid dimensions
        // grid.x = num_features, grid.y = row_tiles
        let rows_per_tile = THREADS_PER_BLOCK;
        let row_tiles = (num_indices as u32).div_ceil(rows_per_tile);

        // Shared memory: 256 * (f32 + f32 + u32) = 256 * 12 = 3072 bytes
        let shared_mem_bytes = 256 * (4 + 4 + 4);

        let config = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (num_features as u32, row_tiles, 1),
            shared_mem_bytes,
        };

        unsafe {
            let d_bins = self.cached_bins.as_ref().unwrap();
            let d_gradients = self.cached_gradients.as_ref().unwrap();
            let d_hessians = self.cached_hessians.as_ref().unwrap();
            let d_grad_hist = self.cached_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_count_hist.as_mut().unwrap();

            stream
                .launch_builder(self.build_histogram_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(d_gradients)
                .arg(d_hessians)
                .arg(&d_indices)
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(num_rows as u32))
                .arg(&(num_features as u32))
                .arg(&(num_indices as u32))
                .launch(config)
                .expect("Failed to launch build_histogram kernel");
        }

        self.device.synchronize();

        // Read back results
        let grad_hist = self
            .device
            .dtoh_copy(self.cached_grad_hist.as_ref().unwrap());
        let hess_hist = self
            .device
            .dtoh_copy(self.cached_hess_hist.as_ref().unwrap());
        let count_hist = self
            .device
            .dtoh_copy(self.cached_count_hist.as_ref().unwrap());

        // Convert to Histogram structs
        (0..num_features)
            .map(|f| {
                let base = f * 256;
                let mut grads = [0.0f32; 256];
                let mut hess = [0.0f32; 256];
                let mut counts = [0u32; 256];

                grads.copy_from_slice(&grad_hist[base..base + 256]);
                hess.copy_from_slice(&hess_hist[base..base + 256]);
                counts.copy_from_slice(&count_hist[base..base + 256]);

                Histogram::from_raw_arrays(&grads, &hess, &counts)
            })
            .collect()
    }

    /// Build histograms for multiple batches in a single GPU dispatch.
    /// This amortizes dispatch overhead across all batches.
    pub fn build_histograms_batched(
        &mut self,
        bins: &[u8],
        grad_hess: &[(f32, f32)],
        batches: &[&[usize]],
        _num_rows: usize,
        num_features: usize,
    ) -> Vec<Vec<Histogram>> {
        self.ensure_initialized();

        let num_batches = batches.len();
        if num_batches == 0 {
            return Vec::new();
        }

        // For single batch, use the optimized single-batch path
        if num_batches == 1 {
            return vec![self.build_histograms(
                bins,
                grad_hess,
                batches[0],
                bins.len() / num_features,
                num_features,
            )];
        }

        // Concatenate all batch indices and create node metadata
        let mut all_indices: Vec<u32> = Vec::new();
        let mut node_starts: Vec<u32> = Vec::with_capacity(num_batches);
        let mut node_counts: Vec<u32> = Vec::with_capacity(num_batches);
        let mut max_count = 0usize;

        for batch in batches {
            node_starts.push(all_indices.len() as u32);
            node_counts.push(batch.len() as u32);
            max_count = max_count.max(batch.len());
            all_indices.extend(batch.iter().map(|&i| i as u32));
        }

        if all_indices.is_empty() {
            return (0..num_batches)
                .map(|_| (0..num_features).map(|_| Histogram::new()).collect())
                .collect();
        }

        // Separate gradients and hessians
        let gradients: Vec<f32> = grad_hess.iter().map(|(g, _)| *g).collect();
        let hessians: Vec<f32> = grad_hess.iter().map(|(_, h)| *h).collect();

        // Upload bins (with caching)
        let bins_key = CacheKey::from_slice(bins);
        if self.cached_bins_key != Some(bins_key) || self.cached_bins.is_none() {
            self.cached_bins = Some(self.device.htod_copy(bins));
            self.cached_bins_key = Some(bins_key);
        }

        // Upload gradients/hessians (with caching)
        let grad_key = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&gradients.len(), &mut hasher);
            if !gradients.is_empty() {
                std::hash::Hash::hash(&gradients[0].to_bits(), &mut hasher);
            }
            CacheKey(std::hash::Hasher::finish(&hasher))
        };

        if self.cached_grad_hess_key != Some(grad_key)
            || self.cached_gradients.is_none()
            || self.cached_hessians.is_none()
        {
            self.cached_gradients = Some(self.device.htod_copy(&gradients));
            self.cached_hessians = Some(self.device.htod_copy(&hessians));
            self.cached_grad_hess_key = Some(grad_key);
        }

        // Upload indices and node metadata
        let d_indices = self.device.htod_copy(&all_indices);
        let d_node_starts = self.device.htod_copy(&node_starts);
        let d_node_counts = self.device.htod_copy(&node_counts);

        // Allocate output buffers (num_batches * num_features * 256)
        let output_size = num_batches * num_features * 256;
        if self.cached_batched_output_len < output_size
            || self.cached_batched_grad_hist.is_none()
            || self.cached_batched_hess_hist.is_none()
            || self.cached_batched_count_hist.is_none()
        {
            self.cached_batched_grad_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_batched_hess_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_batched_count_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_batched_output_len = output_size;
        }

        let stream = self.device.stream();

        // Zero output histograms
        let zero_blocks = output_size.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_grad_hist = self.cached_batched_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_batched_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_batched_count_hist.as_mut().unwrap();
            stream
                .launch_builder(self.zero_histograms_fn.as_ref().unwrap())
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(output_size as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_histograms kernel");
        }

        // Grid: (num_features, num_batches * tiles_per_batch)
        let tiles_per_batch = (max_count as u32).div_ceil(THREADS_PER_BLOCK);
        let shared_mem_bytes = 256 * (4 + 4 + 4);

        let config = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (
                num_features as u32,
                (num_batches as u32) * tiles_per_batch,
                1,
            ),
            shared_mem_bytes,
        };

        unsafe {
            let d_bins = self.cached_bins.as_ref().unwrap();
            let d_gradients = self.cached_gradients.as_ref().unwrap();
            let d_hessians = self.cached_hessians.as_ref().unwrap();
            let d_grad_hist = self.cached_batched_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_batched_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_batched_count_hist.as_mut().unwrap();

            stream
                .launch_builder(self.build_histogram_batched_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(d_gradients)
                .arg(d_hessians)
                .arg(&d_indices)
                .arg(&d_node_starts)
                .arg(&d_node_counts)
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(num_features as u32))
                .arg(&(num_batches as u32))
                .arg(&tiles_per_batch)
                .launch(config)
                .expect("Failed to launch build_histogram_batched kernel");
        }

        self.device.synchronize();

        // Read back results
        let grad_hist = self
            .device
            .dtoh_copy(self.cached_batched_grad_hist.as_ref().unwrap());
        let hess_hist = self
            .device
            .dtoh_copy(self.cached_batched_hess_hist.as_ref().unwrap());
        let count_hist = self
            .device
            .dtoh_copy(self.cached_batched_count_hist.as_ref().unwrap());

        // Convert to Histogram structs - layout is [batch][feature][bin]
        (0..num_batches)
            .map(|batch_idx| {
                (0..num_features)
                    .map(|f| {
                        let base = (batch_idx * num_features + f) * 256;
                        let mut grads = [0.0f32; 256];
                        let mut hess = [0.0f32; 256];
                        let mut counts = [0u32; 256];

                        grads.copy_from_slice(&grad_hist[base..base + 256]);
                        hess.copy_from_slice(&hess_hist[base..base + 256]);
                        counts.copy_from_slice(&count_hist[base..base + 256]);

                        Histogram::from_raw_arrays(&grads, &hess, &counts)
                    })
                    .collect()
            })
            .collect()
    }

    /// Build histograms for multiple nodes using GPU-resident indices buffer.
    /// This is the key method for CUDA Full mode - no per-level PCIe transfers.
    ///
    /// # Arguments
    /// * `d_indices` - GPU buffer containing all row indices
    /// * `node_ranges` - Describes start/count in d_indices for each node (NodeRange struct)
    /// * `num_features` - Number of features
    ///
    /// # Returns
    /// Vec of histograms for each node, each node has num_features histograms
    pub fn build_histograms_gpu(
        &mut self,
        d_indices: &CudaSlice<u32>,
        node_ranges: &[NodeRange],
        num_features: usize,
    ) -> Vec<Vec<Histogram>> {
        self.ensure_initialized();

        if node_ranges.is_empty() || num_features == 0 {
            return Vec::new();
        }

        let num_nodes = node_ranges.len();
        let max_node_count = node_ranges
            .iter()
            .map(|n| n.count as usize)
            .max()
            .unwrap_or(0);

        if max_node_count == 0 {
            return (0..num_nodes)
                .map(|_| (0..num_features).map(|_| Histogram::new()).collect())
                .collect();
        }

        // Prepare node metadata
        let node_starts: Vec<u32> = node_ranges.iter().map(|n| n.start).collect();
        let node_counts: Vec<u32> = node_ranges.iter().map(|n| n.count).collect();

        let d_node_starts = self.device.htod_copy(&node_starts);
        let d_node_counts = self.device.htod_copy(&node_counts);

        // Allocate output buffers (num_nodes * num_features * 256)
        let output_size = num_nodes * num_features * 256;
        if self.cached_batched_output_len < output_size
            || self.cached_batched_grad_hist.is_none()
            || self.cached_batched_hess_hist.is_none()
            || self.cached_batched_count_hist.is_none()
        {
            self.cached_batched_grad_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_batched_hess_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_batched_count_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_batched_output_len = output_size;
        }

        let stream = self.device.stream();

        // Zero output histograms
        let zero_blocks = output_size.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_grad_hist = self.cached_batched_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_batched_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_batched_count_hist.as_mut().unwrap();
            stream
                .launch_builder(self.zero_histograms_fn.as_ref().unwrap())
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(output_size as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_histograms kernel");
        }

        // Grid: (num_features, tiles_per_node, num_nodes) - split into Y and Z to stay under 65535
        let tiles_per_node = (max_node_count as u32).div_ceil(THREADS_PER_BLOCK);
        let shared_mem_bytes = 256 * (4 + 4 + 4);

        let config = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (num_features as u32, tiles_per_node, num_nodes as u32),
            shared_mem_bytes,
        };

        unsafe {
            let d_bins = self
                .cached_bins
                .as_ref()
                .expect("bins must be cached first");
            let d_gradients = self
                .cached_gradients
                .as_ref()
                .expect("gradients must be cached first");
            let d_hessians = self
                .cached_hessians
                .as_ref()
                .expect("hessians must be cached first");
            let d_grad_hist = self.cached_batched_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_batched_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_batched_count_hist.as_mut().unwrap();

            stream
                .launch_builder(self.build_histogram_batched_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(d_gradients)
                .arg(d_hessians)
                .arg(d_indices)
                .arg(&d_node_starts)
                .arg(&d_node_counts)
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(num_features as u32))
                .arg(&(num_nodes as u32))
                .arg(&tiles_per_node)
                .launch(config)
                .expect("Failed to launch build_histogram_batched kernel");
        }

        self.device.synchronize();

        // Read back results
        let grad_hist = self
            .device
            .dtoh_copy(self.cached_batched_grad_hist.as_ref().unwrap());
        let hess_hist = self
            .device
            .dtoh_copy(self.cached_batched_hess_hist.as_ref().unwrap());
        let count_hist = self
            .device
            .dtoh_copy(self.cached_batched_count_hist.as_ref().unwrap());

        // Convert to Histogram structs - layout is [node][feature][bin]
        (0..num_nodes)
            .map(|node_idx| {
                (0..num_features)
                    .map(|f| {
                        let base = (node_idx * num_features + f) * 256;
                        let mut grads = [0.0f32; 256];
                        let mut hess = [0.0f32; 256];
                        let mut counts = [0u32; 256];

                        grads.copy_from_slice(&grad_hist[base..base + 256]);
                        hess.copy_from_slice(&hess_hist[base..base + 256]);
                        counts.copy_from_slice(&count_hist[base..base + 256]);

                        Histogram::from_raw_arrays(&grads, &hess, &counts)
                    })
                    .collect()
            })
            .collect()
    }

    /// Build era-stratified histograms for Directional Era Splitting (DES).
    ///
    /// Returns histograms indexed as `[era][feature]`, enabling directional
    /// agreement checks across eras during split finding.
    // reason: kernel/training entry point with many parameters
    #[allow(clippy::too_many_arguments)]
    pub fn build_era_histograms(
        &mut self,
        bins: &[u8],
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        era_indices: &[u16],
        num_rows: usize,
        num_features: usize,
        num_eras: usize,
    ) -> Vec<Vec<Histogram>> {
        self.ensure_initialized();

        if row_indices.is_empty() || num_eras == 0 {
            return (0..num_eras)
                .map(|_| (0..num_features).map(|_| Histogram::new()).collect())
                .collect();
        }

        let num_indices = row_indices.len();

        // Separate gradients and hessians
        let gradients: Vec<f32> = grad_hess.iter().map(|(g, _)| *g).collect();
        let hessians: Vec<f32> = grad_hess.iter().map(|(_, h)| *h).collect();

        // Convert row indices to u32
        let indices_u32: Vec<u32> = row_indices.iter().map(|&i| i as u32).collect();

        // Upload bins (with caching)
        let bins_key = CacheKey::from_slice(bins);
        if self.cached_bins_key != Some(bins_key) || self.cached_bins.is_none() {
            self.cached_bins = Some(self.device.htod_copy(bins));
            self.cached_bins_key = Some(bins_key);
        }

        // Upload gradients/hessians (with caching)
        let grad_key = {
            let mut hasher = DefaultHasher::new();
            gradients.len().hash(&mut hasher);
            if !gradients.is_empty() {
                gradients[0].to_bits().hash(&mut hasher);
                if gradients.len() > 1 {
                    gradients[gradients.len() / 2].to_bits().hash(&mut hasher);
                }
            }
            CacheKey(hasher.finish())
        };

        if self.cached_grad_hess_key != Some(grad_key)
            || self.cached_gradients.is_none()
            || self.cached_hessians.is_none()
        {
            self.cached_gradients = Some(self.device.htod_copy(&gradients));
            self.cached_hessians = Some(self.device.htod_copy(&hessians));
            self.cached_grad_hess_key = Some(grad_key);
        }

        // Upload era indices (with caching)
        let era_key = {
            let era_bytes: &[u8] = bytemuck::cast_slice(era_indices);
            CacheKey::from_slice(era_bytes)
        };

        if self.cached_era_indices_key != Some(era_key) || self.cached_era_indices.is_none() {
            self.cached_era_indices = Some(self.device.htod_copy(era_indices));
            self.cached_era_indices_key = Some(era_key);
        }

        // Upload row indices (always fresh)
        let d_indices = self.device.htod_copy(&indices_u32);

        // Allocate output buffers (num_eras * num_features * 256)
        let output_size = num_eras * num_features * 256;
        if self.cached_era_output_len < output_size
            || self.cached_era_grad_hist.is_none()
            || self.cached_era_hess_hist.is_none()
            || self.cached_era_count_hist.is_none()
        {
            self.cached_era_grad_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_era_hess_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_era_count_hist = Some(self.device.alloc_zeros(output_size));
            self.cached_era_output_len = output_size;
        }

        let stream = self.device.stream();

        // Zero output histograms
        let zero_blocks = output_size.div_ceil(256) as u32;
        let zero_config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (zero_blocks, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let d_grad_hist = self.cached_era_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_era_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_era_count_hist.as_mut().unwrap();
            stream
                .launch_builder(self.zero_histograms_fn.as_ref().unwrap())
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(output_size as u32))
                .launch(zero_config)
                .expect("Failed to launch zero_histograms kernel");
        }

        // Launch era histogram kernel
        // Grid: (num_features, num_eras)
        // Each block processes one (feature, era) pair
        let shared_mem_bytes = 256 * (4 + 4 + 4); // 256 bins * (f32 + f32 + u32)

        let config = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (num_features as u32, num_eras as u32, 1),
            shared_mem_bytes,
        };

        unsafe {
            let d_bins = self.cached_bins.as_ref().unwrap();
            let d_gradients = self.cached_gradients.as_ref().unwrap();
            let d_hessians = self.cached_hessians.as_ref().unwrap();
            let d_era_indices = self.cached_era_indices.as_ref().unwrap();
            let d_grad_hist = self.cached_era_grad_hist.as_mut().unwrap();
            let d_hess_hist = self.cached_era_hess_hist.as_mut().unwrap();
            let d_count_hist = self.cached_era_count_hist.as_mut().unwrap();

            stream
                .launch_builder(self.build_histogram_era_fn.as_ref().unwrap())
                .arg(d_bins)
                .arg(d_gradients)
                .arg(d_hessians)
                .arg(&d_indices)
                .arg(d_era_indices)
                .arg(d_grad_hist)
                .arg(d_hess_hist)
                .arg(d_count_hist)
                .arg(&(num_rows as u32))
                .arg(&(num_features as u32))
                .arg(&(num_indices as u32))
                .arg(&(num_eras as u32))
                .launch(config)
                .expect("Failed to launch build_histogram_era kernel");
        }

        self.device.synchronize();

        // Read back results
        let grad_hist = self
            .device
            .dtoh_copy(self.cached_era_grad_hist.as_ref().unwrap());
        let hess_hist = self
            .device
            .dtoh_copy(self.cached_era_hess_hist.as_ref().unwrap());
        let count_hist = self
            .device
            .dtoh_copy(self.cached_era_count_hist.as_ref().unwrap());

        // Convert to Histogram structs - layout is [era][feature][bin]
        (0..num_eras)
            .map(|era_idx| {
                (0..num_features)
                    .map(|f| {
                        let base = (era_idx * num_features + f) * 256;
                        let mut grads = [0.0f32; 256];
                        let mut hess = [0.0f32; 256];
                        let mut counts = [0u32; 256];

                        grads.copy_from_slice(&grad_hist[base..base + 256]);
                        hess.copy_from_slice(&hess_hist[base..base + 256]);
                        counts.copy_from_slice(&count_hist[base..base + 256]);

                        Histogram::from_raw_arrays(&grads, &hess, &counts)
                    })
                    .collect()
            })
            .collect()
    }
}
