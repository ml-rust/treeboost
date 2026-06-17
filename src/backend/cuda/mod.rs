//! Native CUDA backend for extreme GPU performance.
//!
//! This module provides a native CUDA implementation that bypasses WGPU overhead.
//! CUDA dispatch latency is 10-100μs vs WGPU's 1-2ms, enabling much better
//! performance for tree building where many small kernel dispatches are needed.
//!
//! # Architecture
//!
//! - `CudaDevice`: Wrapper around cudarc for device management
//! - `CudaBackend`: Implements `HistogramBackend` trait for hybrid mode
//! - `FullCudaTreeBuilder`: Standalone full-GPU tree builder for level-wise mode
//!
//! # Usage
//!
//! ```ignore
//! use treeboost::backend::{BackendConfig, BackendType, GpuMode};
//!
//! // Hybrid mode (GPU histogram + CPU partition)
//! let config = BackendConfig::new()
//!     .with_backend(BackendType::Cuda);
//!
//! // Full GPU mode (level-wise tree building)
//! let config = BackendConfig::new()
//!     .with_backend(BackendType::Cuda)
//!     .with_gpu_mode(GpuMode::Full);
//! ```

mod device;
mod kernels;

pub mod full_gpu;
pub mod partition;

pub use device::CudaDevice;
pub use full_gpu::FullCudaTreeBuilder;
pub use kernels::{HistogramKernel, NodeRange};
pub use partition::{GpuPartitionResult, NodeSplit, PartitionKernel};

use std::sync::{Arc, Mutex};

use crate::backend::traits::{HistogramBackend, SplitCandidate, SplitConfig};
use crate::histogram::Histogram;

/// Native CUDA backend for histogram building.
///
/// Uses cudarc for direct CUDA API access, bypassing WGPU overhead.
/// Dispatch latency is 10-100μs vs WGPU's 1-2ms.
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    histogram_kernel: Mutex<HistogramKernel>,
}

impl CudaBackend {
    /// Create a new CUDA backend if a compatible GPU is available.
    pub fn new() -> Option<Self> {
        let device = CudaDevice::new()?;
        let device = Arc::new(device);
        let histogram_kernel = Mutex::new(HistogramKernel::new(Arc::clone(&device)));

        Some(Self {
            device,
            histogram_kernel,
        })
    }

    /// Get the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl HistogramBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn backend_type(&self) -> crate::backend::BackendType {
        crate::backend::BackendType::Cuda
    }

    fn is_tensor_tile(&self) -> bool {
        true
    }

    fn build_histograms(
        &self,
        bins: &dyn crate::backend::traits::BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> Vec<Histogram> {
        let bins_row_major = bins.as_row_major().expect("CUDA requires row-major bins");
        self.histogram_kernel.lock().unwrap().build_histograms(
            bins_row_major,
            grad_hess,
            row_indices,
            bins.num_rows(),
            bins.num_features(),
        )
    }

    fn build_histograms_sibling(
        &self,
        parent: &[Histogram],
        smaller_child: &[Histogram],
    ) -> Vec<Histogram> {
        // Histogram subtraction is done on CPU (fast, no GPU dispatch needed)
        parent
            .iter()
            .zip(smaller_child.iter())
            .map(|(p, c)| Histogram::from_subtraction(p, c))
            .collect()
    }

    fn find_best_split(
        &self,
        histograms: &[Histogram],
        config: &SplitConfig,
    ) -> Option<SplitCandidate> {
        // Split finding on CPU (fast scan of 256 bins per feature)
        use crate::kernel;

        let mut best: Option<SplitCandidate> = None;

        for (feature_idx, hist) in histograms.iter().enumerate() {
            let grads = hist.sum_gradients();
            let hess = hist.sum_hessians();
            let counts = hist.counts();

            let total_gradient: f32 = grads.iter().sum();
            let total_hessian: f32 = hess.iter().sum();
            let total_count: u32 = counts.iter().sum();

            if let Some(candidate) = kernel::find_best_split(
                &grads,
                &hess,
                &counts,
                total_gradient,
                total_hessian,
                total_count,
                config.lambda,
                config.min_samples_leaf,
                config.min_hessian_leaf,
            ) {
                if candidate.gain > config.min_gain {
                    let split = SplitCandidate {
                        feature: feature_idx,
                        threshold: candidate.bin_threshold,
                        gain: candidate.gain,
                        left_gradient: candidate.left_gradient,
                        left_hessian: candidate.left_hessian,
                        left_count: candidate.left_count,
                        right_gradient: candidate.right_gradient,
                        right_hessian: candidate.right_hessian,
                        right_count: candidate.right_count,
                    };

                    match &best {
                        None => best = Some(split),
                        Some(b) if split.gain > b.gain => best = Some(split),
                        _ => {}
                    }
                }
            }
        }

        best
    }

    fn build_histograms_batched(
        &self,
        bins: &dyn crate::backend::traits::BinStorage,
        grad_hess: &[(f32, f32)],
        batches: &[&[usize]],
    ) -> Vec<Vec<Histogram>> {
        let bins_row_major = bins.as_row_major().expect("CUDA requires row-major bins");
        self.histogram_kernel
            .lock()
            .unwrap()
            .build_histograms_batched(
                bins_row_major,
                grad_hess,
                batches,
                bins.num_rows(),
                bins.num_features(),
            )
    }

    fn build_era_histograms(
        &self,
        bins: &dyn crate::backend::traits::BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        era_indices: &[u16],
        num_eras: usize,
    ) -> Vec<Vec<Histogram>> {
        let bins_row_major = bins.as_row_major().expect("CUDA requires row-major bins");
        self.histogram_kernel.lock().unwrap().build_era_histograms(
            bins_row_major,
            grad_hess,
            row_indices,
            era_indices,
            bins.num_rows(),
            bins.num_features(),
            num_eras,
        )
    }
}
