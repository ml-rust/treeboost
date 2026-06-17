//! x86_64 SIMD implementations for histogram operations
//!
//! # Key Optimizations
//!
//! 1. **Vectorized contiguous loads**: Use AVX2 `loadu_ps` for sequential access
//! 2. **8x unrolled scatter**: Scatter to bins with good ILP
//! 3. **Software prefetching**: Prefetch upcoming data
//!
//! # Architecture Notes
//!
//! The histogram scatter operation is inherently difficult to vectorize because
//! multiple rows may map to the same bin (conflict). However, the LOAD side can
//! be vectorized when data is contiguous (sequential row access).
//!
//! Key insight: AVX2 `loadu_ps` is much faster than `gather` for sequential data:
//! - `loadu_ps`: ~3-4 cycles latency, loads 8 contiguous floats
//! - `gather`: ~20-30 cycles latency, random access pattern

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::backend::scalar::kernel::fallback::HistogramAccumParams;

/// AVX2 histogram accumulation with indexed rows
///
/// Uses AVX2 gather to load 8 gradients/hessians at once, then scatters
/// to histogram bins. The scatter is still scalar due to potential conflicts.
///
/// # Safety
/// - Requires AVX2 support
/// - All pointers must be valid and properly sized
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn histogram_accumulate_avx2(params: HistogramAccumParams) {
    const PREFETCH_DISTANCE: usize = 64; // Prefetch 64 iterations ahead

    let HistogramAccumParams {
        feature_bins,
        row_indices,
        num_rows,
        gradients,
        hessians,
        hist_grads,
        hist_hess,
        hist_counts,
    } = params;

    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process 8 rows at a time
    for i in 0..chunks {
        let base = i * 8;

        // Prefetch upcoming data
        if base + PREFETCH_DISTANCE < num_rows {
            _mm_prefetch(
                row_indices.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load 8 row indices
        let idx0 = *row_indices.add(base);
        let idx1 = *row_indices.add(base + 1);
        let idx2 = *row_indices.add(base + 2);
        let idx3 = *row_indices.add(base + 3);
        let idx4 = *row_indices.add(base + 4);
        let idx5 = *row_indices.add(base + 5);
        let idx6 = *row_indices.add(base + 6);
        let idx7 = *row_indices.add(base + 7);

        // Load 8 bin indices (gather from u8 array)
        let bin0 = *feature_bins.add(idx0) as usize;
        let bin1 = *feature_bins.add(idx1) as usize;
        let bin2 = *feature_bins.add(idx2) as usize;
        let bin3 = *feature_bins.add(idx3) as usize;
        let bin4 = *feature_bins.add(idx4) as usize;
        let bin5 = *feature_bins.add(idx5) as usize;
        let bin6 = *feature_bins.add(idx6) as usize;
        let bin7 = *feature_bins.add(idx7) as usize;

        // Create gather indices for gradients/hessians
        let indices = _mm256_set_epi32(
            idx7 as i32,
            idx6 as i32,
            idx5 as i32,
            idx4 as i32,
            idx3 as i32,
            idx2 as i32,
            idx1 as i32,
            idx0 as i32,
        );

        // Gather 8 gradients using AVX2 gather
        let grads = _mm256_i32gather_ps(gradients, indices, 4);

        // Gather 8 hessians using AVX2 gather
        let hess = _mm256_i32gather_ps(hessians, indices, 4);

        // Extract individual values for scatter (no SIMD scatter in AVX2)
        let grad_arr = std::mem::transmute::<__m256, [f32; 8]>(grads);
        let hess_arr = std::mem::transmute::<__m256, [f32; 8]>(hess);

        // Scatter to histogram bins (must be scalar due to conflicts)
        *hist_grads.add(bin0) += grad_arr[0];
        *hist_hess.add(bin0) += hess_arr[0];
        *hist_counts.add(bin0) += 1;

        *hist_grads.add(bin1) += grad_arr[1];
        *hist_hess.add(bin1) += hess_arr[1];
        *hist_counts.add(bin1) += 1;

        *hist_grads.add(bin2) += grad_arr[2];
        *hist_hess.add(bin2) += hess_arr[2];
        *hist_counts.add(bin2) += 1;

        *hist_grads.add(bin3) += grad_arr[3];
        *hist_hess.add(bin3) += hess_arr[3];
        *hist_counts.add(bin3) += 1;

        *hist_grads.add(bin4) += grad_arr[4];
        *hist_hess.add(bin4) += hess_arr[4];
        *hist_counts.add(bin4) += 1;

        *hist_grads.add(bin5) += grad_arr[5];
        *hist_hess.add(bin5) += hess_arr[5];
        *hist_counts.add(bin5) += 1;

        *hist_grads.add(bin6) += grad_arr[6];
        *hist_hess.add(bin6) += hess_arr[6];
        *hist_counts.add(bin6) += 1;

        *hist_grads.add(bin7) += grad_arr[7];
        *hist_hess.add(bin7) += hess_arr[7];
        *hist_counts.add(bin7) += 1;
    }

    // Handle remainder (scalar)
    let base = chunks * 8;
    for i in 0..remainder {
        let idx = *row_indices.add(base + i);
        let bin = *feature_bins.add(idx) as usize;
        let grad = *gradients.add(idx);
        let hess = *hessians.add(idx);

        *hist_grads.add(bin) += grad;
        *hist_hess.add(bin) += hess;
        *hist_counts.add(bin) += 1;
    }
}

/// AVX2 histogram accumulation for contiguous rows
///
/// Optimized path when rows are 0..num_rows (no indirection needed).
/// Uses AVX2 loads directly since data is contiguous.
///
/// # Safety
/// - Requires AVX2 support
/// - All pointers must be valid and properly sized
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn histogram_accumulate_contiguous_avx2(
    feature_bins: *const u8,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
    const PREFETCH_DISTANCE: usize = 64;

    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process 8 rows at a time
    for i in 0..chunks {
        let base = i * 8;

        // Prefetch upcoming data (contiguous, so more effective)
        if base + PREFETCH_DISTANCE < num_rows {
            _mm_prefetch(
                feature_bins.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                gradients.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                hessians.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load 8 bin indices (contiguous u8 access)
        let bin0 = *feature_bins.add(base) as usize;
        let bin1 = *feature_bins.add(base + 1) as usize;
        let bin2 = *feature_bins.add(base + 2) as usize;
        let bin3 = *feature_bins.add(base + 3) as usize;
        let bin4 = *feature_bins.add(base + 4) as usize;
        let bin5 = *feature_bins.add(base + 5) as usize;
        let bin6 = *feature_bins.add(base + 6) as usize;
        let bin7 = *feature_bins.add(base + 7) as usize;

        // Load 8 gradients (contiguous, use AVX2 load)
        let grads = _mm256_loadu_ps(gradients.add(base));

        // Load 8 hessians (contiguous, use AVX2 load)
        let hess = _mm256_loadu_ps(hessians.add(base));

        // Extract for scatter
        let grad_arr = std::mem::transmute::<__m256, [f32; 8]>(grads);
        let hess_arr = std::mem::transmute::<__m256, [f32; 8]>(hess);

        // Scatter to histogram bins
        *hist_grads.add(bin0) += grad_arr[0];
        *hist_hess.add(bin0) += hess_arr[0];
        *hist_counts.add(bin0) += 1;

        *hist_grads.add(bin1) += grad_arr[1];
        *hist_hess.add(bin1) += hess_arr[1];
        *hist_counts.add(bin1) += 1;

        *hist_grads.add(bin2) += grad_arr[2];
        *hist_hess.add(bin2) += hess_arr[2];
        *hist_counts.add(bin2) += 1;

        *hist_grads.add(bin3) += grad_arr[3];
        *hist_hess.add(bin3) += hess_arr[3];
        *hist_counts.add(bin3) += 1;

        *hist_grads.add(bin4) += grad_arr[4];
        *hist_hess.add(bin4) += hess_arr[4];
        *hist_counts.add(bin4) += 1;

        *hist_grads.add(bin5) += grad_arr[5];
        *hist_hess.add(bin5) += hess_arr[5];
        *hist_counts.add(bin5) += 1;

        *hist_grads.add(bin6) += grad_arr[6];
        *hist_hess.add(bin6) += hess_arr[6];
        *hist_counts.add(bin6) += 1;

        *hist_grads.add(bin7) += grad_arr[7];
        *hist_hess.add(bin7) += hess_arr[7];
        *hist_counts.add(bin7) += 1;
    }

    // Handle remainder (scalar)
    let base = chunks * 8;
    for i in 0..remainder {
        let bin = *feature_bins.add(base + i) as usize;
        let grad = *gradients.add(base + i);
        let hess = *hessians.add(base + i);

        *hist_grads.add(bin) += grad;
        *hist_hess.add(bin) += hess;
        *hist_counts.add(bin) += 1;
    }
}

// ============================================================================
// Grad/Hess Interleaving
// ============================================================================

/// Block size for cache-blocked histogram building
pub const BLOCK_SIZE: usize = 2048;

/// AVX2 copy of gradients and hessians to interleaved cache.
///
/// Uses AVX2 loads and shuffle/permute to interleave 8 gradient/hessian pairs
/// at a time: `[g0,g1,...,g7]` + `[h0,h1,...,h7]` -> `[(g0,h0),(g1,h1),...]`
///
/// # Safety
/// - Requires AVX2 support
/// - `gradients` and `hessians` must have at least `start + len` elements
/// - `gh_cache` must have capacity for `len` elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn copy_gh_interleaved_avx2(
    gradients: &[f32],
    hessians: &[f32],
    start: usize,
    len: usize,
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    use std::arch::x86_64::*;

    let chunks = len / 8;
    let remainder = len % 8;

    let grad_ptr = gradients.as_ptr().add(start);
    let hess_ptr = hessians.as_ptr().add(start);
    let cache_ptr = gh_cache.as_mut_ptr() as *mut f32;

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 gradients and 8 hessians
        let grads = _mm256_loadu_ps(grad_ptr.add(offset));
        let hess = _mm256_loadu_ps(hess_ptr.add(offset));

        // Interleave using unpack operations
        // unpacklo: [g0,h0,g1,h1,g4,h4,g5,h5]
        // unpackhi: [g2,h2,g3,h3,g6,h6,g7,h7]
        let lo = _mm256_unpacklo_ps(grads, hess);
        let hi = _mm256_unpackhi_ps(grads, hess);

        // Permute to fix lane crossing
        // first: [g0,h0,g1,h1,g2,h2,g3,h3]
        // second: [g4,h4,g5,h5,g6,h6,g7,h7]
        let first = _mm256_permute2f128_ps(lo, hi, 0x20);
        let second = _mm256_permute2f128_ps(lo, hi, 0x31);

        // Store interleaved pairs (16 floats = 8 pairs)
        let dst = cache_ptr.add(offset * 2);
        _mm256_storeu_ps(dst, first);
        _mm256_storeu_ps(dst.add(8), second);
    }

    // Handle remainder with scalar code
    let rem_start = chunks * 8;
    for i in 0..remainder {
        let idx = rem_start + i;
        let g = *gradients.get_unchecked(start + idx);
        let h = *hessians.get_unchecked(start + idx);
        *gh_cache.get_unchecked_mut(idx) = (g, h);
    }
}

#[cfg(test)]
// reason: index loops mirror the SIMD kernel layout under test.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_accumulate_indexed() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3];
        let row_indices: Vec<usize> = (0..10).collect();
        let gradients: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let hessians: Vec<f32> = vec![1.0; 10];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_avx2(HistogramAccumParams {
                feature_bins: feature_bins.as_ptr(),
                row_indices: row_indices.as_ptr(),
                num_rows: 10,
                gradients: gradients.as_ptr(),
                hessians: hessians.as_ptr(),
                hist_grads: hist_grads.as_mut_ptr(),
                hist_hess: hist_hess.as_mut_ptr(),
                hist_counts: hist_counts.as_mut_ptr(),
            });
        }

        // Bin 0: rows 0, 3, 6 -> grads 1+4+7=12
        assert!(
            (hist_grads[0] - 12.0).abs() < 1e-5,
            "Bin 0 grad mismatch: {}",
            hist_grads[0]
        );
        assert_eq!(hist_counts[0], 3);

        // Bin 1: rows 1, 4, 7 -> grads 2+5+8=15
        assert!(
            (hist_grads[1] - 15.0).abs() < 1e-5,
            "Bin 1 grad mismatch: {}",
            hist_grads[1]
        );
        assert_eq!(hist_counts[1], 3);

        // Bin 2: rows 2, 5, 8 -> grads 3+6+9=18
        assert!(
            (hist_grads[2] - 18.0).abs() < 1e-5,
            "Bin 2 grad mismatch: {}",
            hist_grads[2]
        );
        assert_eq!(hist_counts[2], 3);

        // Bin 3: row 9 -> grad 10
        assert!(
            (hist_grads[3] - 10.0).abs() < 1e-5,
            "Bin 3 grad mismatch: {}",
            hist_grads[3]
        );
        assert_eq!(hist_counts[3], 1);
    }

    #[test]
    fn test_avx2_accumulate_contiguous() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3];
        let gradients: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let hessians: Vec<f32> = vec![1.0; 10];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_contiguous_avx2(
                feature_bins.as_ptr(),
                10,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
        }

        // Same expected results
        assert!((hist_grads[0] - 12.0).abs() < 1e-5);
        assert_eq!(hist_counts[0], 3);
        assert!((hist_grads[1] - 15.0).abs() < 1e-5);
        assert_eq!(hist_counts[1], 3);
        assert!((hist_grads[2] - 18.0).abs() < 1e-5);
        assert_eq!(hist_counts[2], 3);
        assert!((hist_grads[3] - 10.0).abs() < 1e-5);
        assert_eq!(hist_counts[3], 1);
    }

    #[test]
    fn test_avx2_large_dataset() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        // Test with 100k rows
        let num_rows = 100_000;
        let feature_bins: Vec<u8> = (0..num_rows).map(|i| (i % 256) as u8).collect();
        let row_indices: Vec<usize> = (0..num_rows).collect();
        let gradients: Vec<f32> = vec![1.0; num_rows];
        let hessians: Vec<f32> = vec![1.0; num_rows];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_avx2(HistogramAccumParams {
                feature_bins: feature_bins.as_ptr(),
                row_indices: row_indices.as_ptr(),
                num_rows,
                gradients: gradients.as_ptr(),
                hessians: hessians.as_ptr(),
                hist_grads: hist_grads.as_mut_ptr(),
                hist_hess: hist_hess.as_mut_ptr(),
                hist_counts: hist_counts.as_mut_ptr(),
            });
        }

        // Each bin should have ~390 or 391 rows (100000/256)
        let expected_per_bin = num_rows / 256;
        for bin in 0..256 {
            let count = hist_counts[bin];
            assert!(
                count >= expected_per_bin as u32 - 1 && count <= expected_per_bin as u32 + 1,
                "Bin {} has unexpected count: {}",
                bin,
                count
            );
        }
    }
}
