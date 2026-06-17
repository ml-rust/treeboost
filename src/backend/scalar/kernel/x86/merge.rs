//! SIMD-optimized histogram merge operations
//!
//! # Key Optimization
//!
//! When reducing partial histograms from parallel blocks, we need to merge
//! 256 bins worth of (gradient, hessian, count) tuples. With AVX2, we can
//! process 8 floats / 8 u32s at a time, giving ~8x speedup over scalar.
//!
//! # Memory Layout
//!
//! Histograms use Array-of-Structs: [BinEntry; 256] where BinEntry is:
//!   - sum_gradients: f32
//!   - sum_hessians: f32
//!   - count: u32
//!
//! Total: 12 bytes per bin, 3072 bytes per histogram.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD merge of two SoA histogram arrays (gradients)
///
/// Adds `other_grads` into `self_grads` using AVX2.
/// Processes 8 floats (32 bytes) at a time.
///
/// # Safety
/// - Requires AVX2 support
/// - Arrays must be exactly 256 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn merge_histogram_grads_avx2(self_grads: &mut [f32; 256], other_grads: &[f32; 256]) {
    // 256 floats = 32 AVX2 iterations (8 floats each)
    for i in (0..256).step_by(8) {
        let self_vec = _mm256_loadu_ps(self_grads.as_ptr().add(i));
        let other_vec = _mm256_loadu_ps(other_grads.as_ptr().add(i));
        let sum = _mm256_add_ps(self_vec, other_vec);
        _mm256_storeu_ps(self_grads.as_mut_ptr().add(i), sum);
    }
}

/// SIMD merge of two SoA histogram arrays (hessians)
///
/// # Safety
/// - Requires AVX2 support
/// - Arrays must be exactly 256 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn merge_histogram_hess_avx2(self_hess: &mut [f32; 256], other_hess: &[f32; 256]) {
    for i in (0..256).step_by(8) {
        let self_vec = _mm256_loadu_ps(self_hess.as_ptr().add(i));
        let other_vec = _mm256_loadu_ps(other_hess.as_ptr().add(i));
        let sum = _mm256_add_ps(self_vec, other_vec);
        _mm256_storeu_ps(self_hess.as_mut_ptr().add(i), sum);
    }
}

/// SIMD merge of two SoA histogram arrays (counts)
///
/// # Safety
/// - Requires AVX2 support
/// - Arrays must be exactly 256 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn merge_histogram_counts_avx2(self_counts: &mut [u32; 256], other_counts: &[u32; 256]) {
    for i in (0..256).step_by(8) {
        let self_vec = _mm256_loadu_si256(self_counts.as_ptr().add(i) as *const __m256i);
        let other_vec = _mm256_loadu_si256(other_counts.as_ptr().add(i) as *const __m256i);
        let sum = _mm256_add_epi32(self_vec, other_vec);
        _mm256_storeu_si256(self_counts.as_mut_ptr().add(i) as *mut __m256i, sum);
    }
}

/// SIMD subtract of two SoA histogram arrays (gradients)
///
/// Subtracts `other_grads` from `self_grads` using AVX2.
///
/// # Safety
/// - Requires AVX2 support
/// - Arrays must be exactly 256 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn subtract_histogram_grads_avx2(self_grads: &mut [f32; 256], other_grads: &[f32; 256]) {
    for i in (0..256).step_by(8) {
        let self_vec = _mm256_loadu_ps(self_grads.as_ptr().add(i));
        let other_vec = _mm256_loadu_ps(other_grads.as_ptr().add(i));
        let diff = _mm256_sub_ps(self_vec, other_vec);
        _mm256_storeu_ps(self_grads.as_mut_ptr().add(i), diff);
    }
}

/// SIMD subtract of two SoA histogram arrays (hessians)
///
/// # Safety
/// - Requires AVX2 support
/// - Arrays must be exactly 256 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn subtract_histogram_hess_avx2(self_hess: &mut [f32; 256], other_hess: &[f32; 256]) {
    for i in (0..256).step_by(8) {
        let self_vec = _mm256_loadu_ps(self_hess.as_ptr().add(i));
        let other_vec = _mm256_loadu_ps(other_hess.as_ptr().add(i));
        let diff = _mm256_sub_ps(self_vec, other_vec);
        _mm256_storeu_ps(self_hess.as_mut_ptr().add(i), diff);
    }
}

/// SIMD subtract of two SoA histogram arrays (counts)
///
/// # Safety
/// - Requires AVX2 support
/// - Arrays must be exactly 256 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn subtract_histogram_counts_avx2(
    self_counts: &mut [u32; 256],
    other_counts: &[u32; 256],
) {
    for i in (0..256).step_by(8) {
        let self_vec = _mm256_loadu_si256(self_counts.as_ptr().add(i) as *const __m256i);
        let other_vec = _mm256_loadu_si256(other_counts.as_ptr().add(i) as *const __m256i);
        let diff = _mm256_sub_epi32(self_vec, other_vec);
        _mm256_storeu_si256(self_counts.as_mut_ptr().add(i) as *mut __m256i, diff);
    }
}

#[cfg(test)]
// reason: index loops mirror the SIMD kernel layout under test.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_grads_avx2() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let mut self_grads = [0.0f32; 256];
        let mut other_grads = [0.0f32; 256];

        // Set up test data
        for i in 0..256 {
            self_grads[i] = i as f32;
            other_grads[i] = (256 - i) as f32;
        }

        unsafe {
            merge_histogram_grads_avx2(&mut self_grads, &other_grads);
        }

        // All should be 256
        for (i, &val) in self_grads.iter().enumerate() {
            assert!(
                (val - 256.0).abs() < 1e-5,
                "Mismatch at index {}: expected 256, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_merge_counts_avx2() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let mut self_counts = [0u32; 256];
        let mut other_counts = [0u32; 256];

        for i in 0..256 {
            self_counts[i] = i as u32;
            other_counts[i] = 1;
        }

        unsafe {
            merge_histogram_counts_avx2(&mut self_counts, &other_counts);
        }

        for i in 0..256 {
            assert_eq!(self_counts[i], i as u32 + 1, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_subtract_grads_avx2() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let mut self_grads = [100.0f32; 256];
        let mut other_grads = [0.0f32; 256];

        for i in 0..256 {
            other_grads[i] = i as f32;
        }

        unsafe {
            subtract_histogram_grads_avx2(&mut self_grads, &other_grads);
        }

        for i in 0..256 {
            let expected = 100.0 - i as f32;
            assert!(
                (self_grads[i] - expected).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                self_grads[i]
            );
        }
    }
}
