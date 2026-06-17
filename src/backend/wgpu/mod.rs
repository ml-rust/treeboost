//! WGPU GPU backend for histogram building.
//!
//! Provides GPU-accelerated histogram construction using WebGPU,
//! supporting all major GPU vendors through Vulkan, Metal, and DX12.
//!

pub mod device;
pub mod full_gpu;
pub mod kernels;
pub mod partition;

pub use device::GpuDevice;
pub use full_gpu::FullGpuTreeBuilder;
pub use kernels::GpuProfileData;
pub use partition::{NodeSplit, PartitionKernel, PartitionResult};

use std::sync::Arc;

use crate::backend::scalar::ScalarBackend;
use crate::backend::traits::{BinStorage, HistogramBackend, SplitCandidate, SplitConfig};
use crate::histogram::Histogram;
use crate::kernel;

use kernels::HistogramKernel;

/// WGPU GPU backend for histogram building.
///
/// Uses compute shaders for parallel histogram accumulation across all features.
/// Falls back to CPU for operations that don't benefit from GPU parallelism,
/// including small histogram builds where GPU overhead dominates.
pub struct WgpuBackend {
    device: Arc<GpuDevice>,
    kernel: HistogramKernel,
    /// CPU fallback for small row counts where GPU overhead dominates
    _cpu_fallback: ScalarBackend,
}

impl WgpuBackend {
    /// Attempt to create a WGPU backend.
    ///
    /// Returns `None` if no suitable GPU is available.
    pub fn new() -> Option<Self> {
        let device = Arc::new(GpuDevice::new()?);
        let kernel = HistogramKernel::new(device.clone());

        // WGPU backend initialized (uncomment log when log crate is available)
        // log::info!(
        //     "WGPU backend initialized: {} ({:?})",
        //     device.name(),
        //     device.backend()
        // );

        Some(Self {
            device,
            kernel,
            _cpu_fallback: ScalarBackend::new(),
        })
    }

    /// Get the GPU device name.
    pub fn device_name(&self) -> String {
        self.device.name()
    }

    /// Get the GPU backend type (Vulkan, Metal, DX12).
    pub fn backend_type(&self) -> wgpu::Backend {
        self.device.backend()
    }

    /// Returns true if subgroup operations are supported by the hardware.
    pub fn subgroups_available(&self) -> bool {
        self.kernel.subgroups_available()
    }

    /// Returns true if subgroup operations are enabled and will be used.
    ///
    /// When subgroups are active, the GPU shader uses `subgroupAdd` and
    /// `subgroupBroadcastFirst` to reduce atomic contention when multiple
    /// threads in a subgroup write to the same histogram bin.
    ///
    /// Note: Subgroups are disabled by default. Use `set_use_subgroups(true)` to enable.
    pub fn has_subgroups(&self) -> bool {
        self.kernel.has_subgroups()
    }

    /// Enable or disable subgroup operations.
    ///
    /// Subgroups are **disabled by default** because benchmarks show minimal
    /// benefit on modern NVIDIA GPUs (~1.0x speedup). They may help on older
    /// AMD or Intel GPUs with slower atomics.
    ///
    /// Has no effect if hardware doesn't support subgroups.
    pub fn set_use_subgroups(&self, enabled: bool) {
        self.kernel.set_use_subgroups(enabled);
    }

    /// Returns the subgroup size range, or (0, 0) if subgroups are not supported.
    pub fn subgroup_size(&self) -> (u32, u32) {
        (self.device.min_subgroup_size, self.device.max_subgroup_size)
    }

    /// Build histograms using the base shader (no subgroups).
    ///
    /// This is primarily for benchmarking to compare subgroup vs non-subgroup performance.
    pub fn build_histograms_base_shader(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> Vec<Histogram> {
        let num_rows = bins.num_rows();
        let num_features = bins.num_features();

        let bins_row_major: std::borrow::Cow<[u8]> = match bins.as_row_major() {
            Some(data) => std::borrow::Cow::Borrowed(data),
            None => {
                let mut row_major = vec![0u8; num_rows * num_features];
                for f in 0..num_features {
                    if let Some(col) = bins.feature_column(f) {
                        for r in 0..num_rows {
                            row_major[r * num_features + f] = col[r];
                        }
                    }
                }
                std::borrow::Cow::Owned(row_major)
            }
        };

        self.kernel.build_histograms_base_shader(
            &bins_row_major,
            grad_hess,
            row_indices,
            num_rows,
            num_features,
        )
    }

    /// Build histograms with detailed profiling.
    ///
    /// Returns histograms and detailed timing for each GPU operation.
    /// Useful for understanding GPU overhead and identifying bottlenecks.
    ///
    /// Note: This bypasses the CPU fallback for small row counts to profile
    /// the actual GPU path.
    pub fn build_histograms_profiled(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> (Vec<Histogram>, GpuProfileData) {
        let num_rows = bins.num_rows();
        let num_features = bins.num_features();

        // Get row-major bins (converting if necessary)
        let bins_row_major: std::borrow::Cow<[u8]> = match bins.as_row_major() {
            Some(data) => std::borrow::Cow::Borrowed(data),
            None => {
                let mut row_major = vec![0u8; num_rows * num_features];
                for f in 0..num_features {
                    if let Some(col) = bins.feature_column(f) {
                        for r in 0..num_rows {
                            row_major[r * num_features + f] = col[r];
                        }
                    }
                }
                std::borrow::Cow::Owned(row_major)
            }
        };

        self.kernel.build_histograms_profiled(
            &bins_row_major,
            grad_hess,
            row_indices,
            num_rows,
            num_features,
        )
    }

    /// Build histograms for multiple batches in a single GPU dispatch.
    ///
    /// This is significantly more efficient than calling `build_histograms` multiple times
    /// because it amortizes GPU dispatch overhead across all batches.
    ///
    /// # Performance
    ///
    /// For N batches of small row subsets:
    /// - Individual: N * dispatch_overhead + N * compute_time
    /// - Batched: 1 * dispatch_overhead + N * compute_time (parallel)
    ///
    /// Expected speedup: 2-30x depending on batch count and sizes.
    ///
    /// # Arguments
    ///
    /// * `bins` - Dataset with bin data
    /// * `grad_hess` - Gradient/hessian pairs for entire dataset
    /// * `batches` - Slice of row index arrays, one per batch
    ///
    /// # Returns
    ///
    /// Vector of histogram vectors, one per batch.
    pub fn build_histograms_batched(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        batches: &[&[usize]],
    ) -> Vec<Vec<Histogram>> {
        let num_rows = bins.num_rows();
        let num_features = bins.num_features();

        // Get row-major bins (converting if necessary)
        let bins_row_major: std::borrow::Cow<[u8]> = match bins.as_row_major() {
            Some(data) => std::borrow::Cow::Borrowed(data),
            None => {
                let mut row_major = vec![0u8; num_rows * num_features];
                for f in 0..num_features {
                    if let Some(col) = bins.feature_column(f) {
                        for r in 0..num_rows {
                            row_major[r * num_features + f] = col[r];
                        }
                    }
                }
                std::borrow::Cow::Owned(row_major)
            }
        };

        self.kernel.build_histograms_batched(
            &bins_row_major,
            grad_hess,
            batches,
            num_rows,
            num_features,
        )
    }
}

impl HistogramBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "WGPU"
    }

    fn backend_type(&self) -> crate::backend::BackendType {
        crate::backend::BackendType::Wgpu
    }

    fn is_tensor_tile(&self) -> bool {
        true
    }

    fn build_histograms(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> Vec<Histogram> {
        let num_rows = bins.num_rows();
        let num_features = bins.num_features();

        // Check if we can use 4-bit path (50% memory bandwidth reduction)
        if bins.supports_4bit() {
            if let Some(bins_4bit) = bins.as_row_major_4bit() {
                return self.kernel.build_histograms_4bit(
                    bins_4bit,
                    grad_hess,
                    row_indices,
                    num_rows,
                    num_features,
                );
            }
        }

        // Fall back to 8-bit path
        // Get row-major bins (converting if necessary)
        // Use Cow to avoid allocation when data is already row-major
        let bins_row_major: std::borrow::Cow<[u8]> = match bins.as_row_major() {
            Some(data) => std::borrow::Cow::Borrowed(data),
            None => {
                // Convert column-major to row-major
                let mut row_major = vec![0u8; num_rows * num_features];
                for f in 0..num_features {
                    if let Some(col) = bins.feature_column(f) {
                        for r in 0..num_rows {
                            row_major[r * num_features + f] = col[r];
                        }
                    }
                }
                std::borrow::Cow::Owned(row_major)
            }
        };

        // Use GPU kernel for histogram building
        self.kernel.build_histograms(
            &bins_row_major,
            grad_hess,
            row_indices,
            num_rows,
            num_features,
        )
    }

    fn build_histograms_sibling(
        &self,
        parent: &[Histogram],
        smaller_child: &[Histogram],
    ) -> Vec<Histogram> {
        // Sibling computation via subtraction is fast on CPU
        // No GPU benefit for this simple operation
        parent
            .iter()
            .zip(smaller_child.iter())
            .map(|(p, s)| Histogram::from_subtraction(p, s))
            .collect()
    }

    fn find_best_split(
        &self,
        histograms: &[Histogram],
        config: &SplitConfig,
    ) -> Option<SplitCandidate> {
        // Split finding is O(256 * num_features), which is cheap
        // GPU overhead not worth it; use CPU kernel
        let mut best: Option<SplitCandidate> = None;

        for (feature, hist) in histograms.iter().enumerate() {
            let (total_grad, total_hess, total_count) = hist.totals();

            if total_count < 2 * config.min_samples_leaf {
                continue;
            }

            if let Some(candidate) = kernel::find_best_split(
                &hist.sum_gradients(),
                &hist.sum_hessians(),
                &hist.counts(),
                total_grad,
                total_hess,
                total_count,
                config.lambda,
                config.min_samples_leaf,
                config.min_hessian_leaf,
            ) {
                let split = SplitCandidate {
                    feature,
                    threshold: candidate.bin_threshold,
                    gain: candidate.gain,
                    left_gradient: candidate.left_gradient,
                    left_hessian: candidate.left_hessian,
                    left_count: candidate.left_count,
                    right_gradient: candidate.right_gradient,
                    right_hessian: candidate.right_hessian,
                    right_count: candidate.right_count,
                };

                if split.gain > config.min_gain {
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
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        batches: &[&[usize]],
    ) -> Vec<Vec<Histogram>> {
        // Use the optimized GPU batched implementation
        WgpuBackend::build_histograms_batched(self, bins, grad_hess, batches)
    }

    fn build_era_histograms(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        era_indices: &[u16],
        num_eras: usize,
    ) -> Vec<Vec<Histogram>> {
        let num_rows = bins.num_rows();
        let num_features = bins.num_features();

        // Get row-major bins (converting if necessary)
        let bins_row_major: std::borrow::Cow<[u8]> = match bins.as_row_major() {
            Some(data) => std::borrow::Cow::Borrowed(data),
            None => {
                // Convert column-major to row-major
                let mut row_major = vec![0u8; num_rows * num_features];
                for f in 0..num_features {
                    if let Some(col) = bins.feature_column(f) {
                        for r in 0..num_rows {
                            row_major[r * num_features + f] = col[r];
                        }
                    }
                }
                std::borrow::Cow::Owned(row_major)
            }
        };

        // Use GPU kernel for era histogram building
        self.kernel.build_era_histograms(
            &bins_row_major,
            grad_hess,
            row_indices,
            era_indices,
            num_rows,
            num_features,
            num_eras,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ScalarBackend;
    use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};

    #[test]
    fn test_wgpu_backend_creation() {
        // This test will skip on systems without GPU
        match WgpuBackend::new() {
            Some(backend) => {
                println!("WGPU backend created: {}", backend.device_name());
                assert_eq!(backend.name(), "WGPU");
                assert!(backend.is_tensor_tile());
            }
            None => {
                println!("No GPU available, skipping WGPU backend test");
            }
        }
    }

    /// Create a test dataset with known bin values
    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        // Column-major layout: features[feature_idx * num_rows + row_idx]
        let mut features = vec![0u8; num_rows * num_features];
        for f in 0..num_features {
            for r in 0..num_rows {
                // Assign bins based on row index (0-255)
                features[f * num_rows + r] = (r % 256) as u8;
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
        let feature_info: Vec<FeatureInfo> = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("feature_{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_wgpu_matches_scalar_small() {
        let wgpu_backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU available, skipping WGPU vs Scalar comparison test");
                return;
            }
        };
        let scalar_backend = ScalarBackend::new();

        // Small dataset: 256 rows, 4 features
        let dataset = create_test_dataset(256, 4);

        // Generate gradients and hessians
        let grad_hess: Vec<(f32, f32)> = (0..256).map(|i| (i as f32 * 0.1, 1.0)).collect();

        // Use all rows
        let row_indices: Vec<usize> = (0..256).collect();

        // Build histograms with both backends
        let wgpu_hists = wgpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

        assert_eq!(wgpu_hists.len(), scalar_hists.len());

        // Compare histograms (allow small floating point differences)
        for (f, (wgpu_hist, scalar_hist)) in wgpu_hists.iter().zip(scalar_hists.iter()).enumerate()
        {
            let (wgpu_total_grad, wgpu_total_hess, wgpu_total_count) = wgpu_hist.totals();
            let (scalar_total_grad, scalar_total_hess, scalar_total_count) = scalar_hist.totals();

            assert_eq!(
                wgpu_total_count, scalar_total_count,
                "Feature {}: count mismatch ({} vs {})",
                f, wgpu_total_count, scalar_total_count
            );

            let grad_diff = (wgpu_total_grad - scalar_total_grad).abs();
            let hess_diff = (wgpu_total_hess - scalar_total_hess).abs();

            // Allow 0.1% relative error for GPU float atomics
            let grad_tolerance = scalar_total_grad.abs() * 0.001 + 0.001;
            let hess_tolerance = scalar_total_hess.abs() * 0.001 + 0.001;

            assert!(
                grad_diff < grad_tolerance,
                "Feature {}: gradient mismatch ({} vs {}, diff={})",
                f,
                wgpu_total_grad,
                scalar_total_grad,
                grad_diff
            );
            assert!(
                hess_diff < hess_tolerance,
                "Feature {}: hessian mismatch ({} vs {}, diff={})",
                f,
                wgpu_total_hess,
                scalar_total_hess,
                hess_diff
            );
        }

        println!("WGPU matches Scalar for 256 rows × 4 features");
    }

    #[test]
    fn test_wgpu_matches_scalar_medium() {
        let wgpu_backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU available, skipping WGPU vs Scalar comparison test");
                return;
            }
        };
        let scalar_backend = ScalarBackend::new();

        // Medium dataset: 10000 rows, 10 features
        let num_rows = 10000;
        let num_features = 10;
        let dataset = create_test_dataset(num_rows, num_features);

        // Generate gradients and hessians with varied values
        let grad_hess: Vec<(f32, f32)> = (0..num_rows)
            .map(|i| {
                let g = (i as f32 * 0.01).sin() * 10.0;
                let h = 1.0 + (i % 10) as f32 * 0.1;
                (g, h)
            })
            .collect();

        // Use all rows
        let row_indices: Vec<usize> = (0..num_rows).collect();

        // Build histograms with both backends
        let wgpu_hists = wgpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

        assert_eq!(wgpu_hists.len(), scalar_hists.len());

        // Compare totals for each feature
        for f in 0..num_features {
            let (wgpu_total_grad, wgpu_total_hess, wgpu_total_count) = wgpu_hists[f].totals();
            let (scalar_total_grad, scalar_total_hess, scalar_total_count) =
                scalar_hists[f].totals();

            assert_eq!(
                wgpu_total_count, scalar_total_count,
                "Feature {}: count mismatch",
                f
            );

            // Looser tolerance for larger dataset (more floating point accumulation)
            let grad_tolerance = scalar_total_grad.abs() * 0.01 + 0.1;
            let hess_tolerance = scalar_total_hess.abs() * 0.01 + 0.1;

            let grad_diff = (wgpu_total_grad - scalar_total_grad).abs();
            let hess_diff = (wgpu_total_hess - scalar_total_hess).abs();

            assert!(
                grad_diff < grad_tolerance,
                "Feature {}: gradient mismatch (GPU={}, CPU={}, diff={})",
                f,
                wgpu_total_grad,
                scalar_total_grad,
                grad_diff
            );
            assert!(
                hess_diff < hess_tolerance,
                "Feature {}: hessian mismatch (GPU={}, CPU={}, diff={})",
                f,
                wgpu_total_hess,
                scalar_total_hess,
                hess_diff
            );
        }

        println!(
            "WGPU matches Scalar for {} rows × {} features",
            num_rows, num_features
        );
    }

    #[test]
    fn test_wgpu_with_row_indices() {
        let wgpu_backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU available, skipping WGPU row indices test");
                return;
            }
        };
        let scalar_backend = ScalarBackend::new();

        // Dataset: 1000 rows, 5 features
        let num_rows = 1000;
        let dataset = create_test_dataset(num_rows, 5);

        let grad_hess: Vec<(f32, f32)> = (0..num_rows).map(|i| (i as f32, 1.0)).collect();

        // Use only even-indexed rows
        let row_indices: Vec<usize> = (0..num_rows).filter(|i| i % 2 == 0).collect();

        let wgpu_hists = wgpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

        // Compare totals
        for f in 0..5 {
            let (_, _, wgpu_count) = wgpu_hists[f].totals();
            let (_, _, scalar_count) = scalar_hists[f].totals();

            assert_eq!(
                wgpu_count, scalar_count,
                "Feature {}: count mismatch with row indices",
                f
            );
            assert_eq!(
                wgpu_count, 500,
                "Expected 500 rows (half of 1000), got {}",
                wgpu_count
            );
        }

        println!("WGPU correctly handles row indices");
    }

    /// Create a test dataset with known bin values for 4-bit testing
    fn create_test_dataset_4bit(num_rows: usize, num_features: usize) -> BinnedDataset {
        // Column-major layout: features[feature_idx * num_rows + row_idx]
        let mut features = vec![0u8; num_rows * num_features];
        for f in 0..num_features {
            for r in 0..num_rows {
                // Assign bins based on row index (0-15 for 4-bit)
                features[f * num_rows + r] = (r % 16) as u8;
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
        let feature_info: Vec<FeatureInfo> = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("feature_{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 16, // <=16 bins enables 4-bit path
                bin_boundaries: vec![],
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_wgpu_4bit_matches_scalar() {
        let wgpu_backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU available, skipping WGPU 4-bit test");
                return;
            }
        };
        let scalar_backend = ScalarBackend::new();

        // Create dataset with <=16 bins per feature (triggers 4-bit path)
        let num_rows = 1000;
        let num_features = 8;
        let dataset = create_test_dataset_4bit(num_rows, num_features);

        // Verify dataset supports 4-bit
        assert!(dataset.supports_4bit());
        assert!(dataset.max_bins() <= 16);

        // Generate gradients and hessians
        let grad_hess: Vec<(f32, f32)> = (0..num_rows).map(|i| (i as f32 * 0.01, 1.0)).collect();

        // Use all rows
        let row_indices: Vec<usize> = (0..num_rows).collect();

        // Build histograms with both backends
        // WGPU should use 4-bit path automatically
        let wgpu_hists = wgpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

        assert_eq!(wgpu_hists.len(), scalar_hists.len());

        // Compare histograms
        for (f, (wgpu_hist, scalar_hist)) in wgpu_hists.iter().zip(scalar_hists.iter()).enumerate()
        {
            let (wgpu_total_grad, wgpu_total_hess, wgpu_total_count) = wgpu_hist.totals();
            let (scalar_total_grad, scalar_total_hess, scalar_total_count) = scalar_hist.totals();

            assert_eq!(
                wgpu_total_count, scalar_total_count,
                "Feature {}: count mismatch ({} vs {})",
                f, wgpu_total_count, scalar_total_count
            );

            let grad_diff = (wgpu_total_grad - scalar_total_grad).abs();
            let hess_diff = (wgpu_total_hess - scalar_total_hess).abs();

            // Allow small tolerance for GPU float quantization
            let grad_tolerance = scalar_total_grad.abs() * 0.01 + 0.1;
            let hess_tolerance = scalar_total_hess.abs() * 0.01 + 0.1;

            assert!(
                grad_diff < grad_tolerance,
                "Feature {}: gradient mismatch ({} vs {}, diff={})",
                f,
                wgpu_total_grad,
                scalar_total_grad,
                grad_diff
            );
            assert!(
                hess_diff < hess_tolerance,
                "Feature {}: hessian mismatch ({} vs {}, diff={})",
                f,
                wgpu_total_hess,
                scalar_total_hess,
                hess_diff
            );
        }

        println!(
            "WGPU 4-bit matches Scalar for {} rows × {} features (max_bins={})",
            num_rows,
            num_features,
            dataset.max_bins()
        );
    }

    #[test]
    fn test_wgpu_4bit_with_odd_features() {
        let wgpu_backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU available, skipping WGPU 4-bit odd features test");
                return;
            }
        };
        let scalar_backend = ScalarBackend::new();

        // Odd number of features to test nibble padding
        let num_rows = 500;
        let num_features = 7;
        let dataset = create_test_dataset_4bit(num_rows, num_features);

        assert!(dataset.supports_4bit());

        let grad_hess: Vec<(f32, f32)> = (0..num_rows)
            .map(|i| ((i as f32).sin() * 5.0, 1.0))
            .collect();

        let row_indices: Vec<usize> = (0..num_rows).collect();

        let wgpu_hists = wgpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

        // Verify counts match for all features
        for f in 0..num_features {
            let (_, _, wgpu_count) = wgpu_hists[f].totals();
            let (_, _, scalar_count) = scalar_hists[f].totals();

            assert_eq!(
                wgpu_count, scalar_count,
                "Feature {}: count mismatch with odd features",
                f
            );
        }

        println!(
            "WGPU 4-bit correctly handles {} features (odd count)",
            num_features
        );
    }
}
