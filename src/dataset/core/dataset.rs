//! Core binned dataset with columnar u8 storage
//!
//! Provides memory-efficient storage for histogram-based GBDT training.
//! Features are discretized to u8 bins (256 values) for 8x memory reduction.
//!
//! # Sparsity Awareness
//!
//! For sparse features (many zeros), we store only non-zero entries and compute
//! the zero bin by subtraction: `zero_bin = total - sum(non_zero_bins)`.
//! This provides up to 20x speedup on 95% sparse data.
//!
//! # Data Layouts
//!
//! - **Column-major** (default): `bins[feature][row]` - optimal for scalar CPU
//! - **Row-major** (lazy): `bins[row][feature]` - optimal for GPU/tensor-tile
//!
//! Row-major layout is computed lazily on first GPU use and cached for reuse.

use rkyv::{Archive, Deserialize, Serialize};
use std::sync::OnceLock;

use super::feature_info::{FeatureInfo, DEFAULT_BIN};
use super::sparse::SparseColumn;

// =============================================================================
// BinnedDataset
// =============================================================================

/// Columnar binned dataset for efficient histogram construction
///
/// Memory layout:
/// - Features stored column-major as contiguous u8 arrays
/// - Each feature column is `num_rows` bytes
/// - Total feature memory: `num_rows * num_features` bytes
/// - Multi-output targets stored row-wise: `targets[row * num_target_cols + label]`
///
/// Sparse features are additionally stored in CSR-like format for efficient
/// histogram building on sparse data.
///
/// For GPU backends, a row-major layout is computed lazily and cached.
#[derive(Archive, Serialize, Deserialize)]
pub struct BinnedDataset {
    /// Number of rows (samples)
    num_rows: usize,
    /// Feature data in column-major order: features[feature_idx * num_rows + row_idx]
    features: Vec<u8>,
    /// Target values (original scale, not binned)
    /// For multi-output: row-wise layout `targets[row * num_target_cols + label]`
    targets: Vec<f32>,
    /// Number of target columns (1 for scalar targets, >1 for multi-output)
    num_target_cols: usize,
    /// Feature metadata
    feature_info: Vec<FeatureInfo>,
    /// Sparse representations for sparse features (None if dense)
    sparse_columns: Vec<Option<SparseColumn>>,
    /// Era indices for each sample (optional, for era-based splitting)
    /// Used by Directional Era Splitting (DES) to learn invariant patterns
    /// u16 supports up to 65536 eras
    era_indices: Option<Vec<u16>>,
    /// Number of unique eras (cached for efficiency)
    num_eras: usize,
    /// Cached row-major layout for GPU backends (lazily computed)
    /// Not serialized - recomputed on first GPU use after deserialization
    #[rkyv(with = rkyv::with::Skip)]
    row_major_cache: OnceLock<Vec<u8>>,
    /// Cached 4-bit packed row-major layout for GPU backends (lazily computed)
    /// Only used when all features have ≤16 bins
    #[rkyv(with = rkyv::with::Skip)]
    row_major_4bit_cache: OnceLock<Vec<u8>>,
    /// Raw (unbinned) features for LinearThenTree mode (optional)
    ///
    /// Stores the preprocessed feature values BEFORE binning. This is critical
    /// for LinearThenTree mode where the linear model needs the exact StandardScaled
    /// polynomial/interaction features, not bin center approximations.
    ///
    /// Layout: Row-major `raw_features[row * num_features + feature]`
    ///
    /// **Serialization**: This MUST be serialized for correct LinearThenTree predictions.
    /// **Memory**: Can be large (num_rows × num_features × 4 bytes), but necessary
    /// for correct linear model behavior with engineered features.
    raw_features: Option<Vec<f32>>,
}

impl std::fmt::Debug for BinnedDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinnedDataset")
            .field("num_rows", &self.num_rows)
            .field("num_features", &self.num_features())
            .field("num_target_cols", &self.num_target_cols)
            .field("features_len", &self.features.len())
            .field("sparse_features", &self.num_sparse_features())
            .field("max_bins", &self.max_bins())
            .field("supports_4bit", &self.supports_4bit())
            .field("has_eras", &self.era_indices.is_some())
            .field("num_eras", &self.num_eras)
            .field("row_major_cached", &self.row_major_cache.get().is_some())
            .field(
                "row_major_4bit_cached",
                &self.row_major_4bit_cache.get().is_some(),
            )
            .finish()
    }
}

impl Clone for BinnedDataset {
    fn clone(&self) -> Self {
        Self {
            num_rows: self.num_rows,
            features: self.features.clone(),
            targets: self.targets.clone(),
            num_target_cols: self.num_target_cols,
            feature_info: self.feature_info.clone(),
            sparse_columns: self.sparse_columns.clone(),
            era_indices: self.era_indices.clone(),
            num_eras: self.num_eras,
            // Don't clone caches - they will be recomputed if needed
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: self.raw_features.clone(),
        }
    }
}

impl BinnedDataset {
    /// Create a new binned dataset with scalar targets
    ///
    /// Automatically detects sparse features and creates sparse representations.
    /// For multi-output targets, use `new_multioutput()` instead.
    pub fn new(
        num_rows: usize,
        features: Vec<u8>,
        targets: Vec<f32>,
        feature_info: Vec<FeatureInfo>,
    ) -> Self {
        debug_assert_eq!(features.len(), num_rows * feature_info.len());
        debug_assert_eq!(targets.len(), num_rows);

        let num_features = feature_info.len();

        // Detect sparse features and create sparse representations
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * num_rows;
                let column = &features[start..start + num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        Self {
            num_rows,
            features,
            targets,
            num_target_cols: 1, // Scalar targets
            feature_info,
            sparse_columns,
            era_indices: None,
            num_eras: 0,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: None, // No raw features by default
        }
    }

    /// Create a new binned dataset with multi-output targets
    ///
    /// # Arguments
    /// * `num_rows` - Number of samples
    /// * `features` - Column-major feature data
    /// * `targets` - Row-wise flattened targets: `[row0_label0, row0_label1, ..., row1_label0, ...]`
    /// * `feature_info` - Feature metadata
    /// * `num_target_cols` - Number of target columns (labels/outputs)
    ///
    /// # Panics
    /// Panics if `targets.len() != num_rows * num_target_cols`
    pub fn new_multioutput(
        num_rows: usize,
        features: Vec<u8>,
        targets: Vec<f32>,
        feature_info: Vec<FeatureInfo>,
        num_target_cols: usize,
    ) -> Self {
        assert_eq!(
            targets.len(),
            num_rows * num_target_cols,
            "targets length ({}) must equal num_rows ({}) * num_target_cols ({})",
            targets.len(),
            num_rows,
            num_target_cols
        );
        debug_assert_eq!(features.len(), num_rows * feature_info.len());

        let num_features = feature_info.len();

        // Detect sparse features and create sparse representations
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * num_rows;
                let column = &features[start..start + num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        Self {
            num_rows,
            features,
            targets,
            num_target_cols,
            feature_info,
            sparse_columns,
            era_indices: None,
            num_eras: 0,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: None,
        }
    }

    /// Create a new binned dataset with era indices for era-based splitting
    ///
    /// Era indices must be in range [0, num_eras) where num_eras is automatically
    /// computed from the maximum era index + 1.
    ///
    /// # Arguments
    /// * `num_rows` - Number of samples
    /// * `features` - Column-major feature data
    /// * `targets` - Target values (scalar)
    /// * `feature_info` - Feature metadata
    /// * `era_indices` - Era index for each sample (u16, supports up to 65536 eras)
    pub fn new_with_eras(
        num_rows: usize,
        features: Vec<u8>,
        targets: Vec<f32>,
        feature_info: Vec<FeatureInfo>,
        era_indices: Vec<u16>,
    ) -> Self {
        debug_assert_eq!(features.len(), num_rows * feature_info.len());
        debug_assert_eq!(targets.len(), num_rows);
        debug_assert_eq!(era_indices.len(), num_rows);

        let num_features = feature_info.len();

        // Detect sparse features
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * num_rows;
                let column = &features[start..start + num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        // Compute number of eras from max era index
        let num_eras = era_indices
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);

        Self {
            num_rows,
            features,
            targets,
            num_target_cols: 1, // Scalar targets
            feature_info,
            sparse_columns,
            era_indices: Some(era_indices),
            num_eras,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: None, // No raw features by default
        }
    }

    // =========================================================================
    // Raw Features (for LinearThenTree mode)
    // =========================================================================

    /// Attach raw (unbinned) features to this dataset
    ///
    /// This is critical for LinearThenTree mode where the linear model needs
    /// exact preprocessed feature values (e.g., StandardScaled polynomials),
    /// not bin center approximations.
    ///
    /// # Arguments
    /// * `raw_features` - Row-major features: `[row0_feat0, row0_feat1, ..., row1_feat0, ...]`
    ///
    /// # Panics
    /// Panics if `raw_features.len() != num_rows * num_features`
    pub fn with_raw_features(mut self, raw_features: Vec<f32>) -> Self {
        assert_eq!(
            raw_features.len(),
            self.num_rows * self.num_features(),
            "Raw features size mismatch: expected {} ({}×{}), got {}",
            self.num_rows * self.num_features(),
            self.num_rows,
            self.num_features(),
            raw_features.len()
        );
        self.raw_features = Some(raw_features);
        self
    }

    /// Get raw features if available
    ///
    /// Returns the unbinned feature values that were attached via `with_raw_features()`.
    /// If not available, returns `None` (caller should fall back to extracting bin centers).
    pub fn raw_features(&self) -> Option<&[f32]> {
        self.raw_features.as_deref()
    }

    // =========================================================================
    // Era Methods
    // =========================================================================

    /// Set era indices on an existing dataset
    ///
    /// # Arguments
    /// * `era_indices` - Era index for each sample (must have length == num_rows)
    pub fn set_era_indices(&mut self, era_indices: Vec<u16>) {
        debug_assert_eq!(era_indices.len(), self.num_rows);
        self.num_eras = era_indices
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        self.era_indices = Some(era_indices);
    }

    /// Check if era indices are available
    #[inline]
    pub fn has_eras(&self) -> bool {
        self.era_indices.is_some()
    }

    /// Get the number of eras (0 if no era indices)
    #[inline]
    pub fn num_eras(&self) -> usize {
        self.num_eras
    }

    /// Get era index for a row (panics if no era indices set)
    #[inline]
    pub fn era(&self, row_idx: usize) -> u16 {
        self.era_indices.as_ref().expect("No era indices set")[row_idx]
    }

    /// Get era indices slice (None if not set)
    #[inline]
    pub fn era_indices(&self) -> Option<&[u16]> {
        self.era_indices.as_deref()
    }

    // =========================================================================
    // Basic Accessors
    // =========================================================================

    /// Number of samples
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of features
    #[inline]
    pub fn num_features(&self) -> usize {
        self.feature_info.len()
    }

    /// Get feature info for a feature
    #[inline]
    pub fn feature_info(&self, feature_idx: usize) -> &FeatureInfo {
        &self.feature_info[feature_idx]
    }

    /// Get all feature info
    #[inline]
    pub fn all_feature_info(&self) -> &[FeatureInfo] {
        &self.feature_info
    }

    /// Get bin value for a specific row and feature
    #[inline]
    pub fn get_bin(&self, row_idx: usize, feature_idx: usize) -> u8 {
        self.features[feature_idx * self.num_rows + row_idx]
    }

    /// Get the entire column for a feature (contiguous slice)
    #[inline]
    pub fn feature_column(&self, feature_idx: usize) -> &[u8] {
        let start = feature_idx * self.num_rows;
        &self.features[start..start + self.num_rows]
    }

    /// Get all binned features (column-major layout)
    ///
    /// Returns the raw feature array in column-major order:
    /// `features[feature_idx * num_rows + row_idx]`
    #[inline]
    pub fn features(&self) -> &[u8] {
        &self.features
    }

    // =========================================================================
    // Sparse Feature Access
    // =========================================================================

    /// Check if a feature has a sparse representation
    #[inline]
    pub fn is_sparse(&self, feature_idx: usize) -> bool {
        self.sparse_columns
            .get(feature_idx)
            .map(|s| s.is_some())
            .unwrap_or(false)
    }

    /// Get sparse column for a feature (if available)
    #[inline]
    pub fn sparse_column(&self, feature_idx: usize) -> Option<&SparseColumn> {
        self.sparse_columns
            .get(feature_idx)
            .and_then(|s| s.as_ref())
    }

    /// Get number of sparse features
    pub fn num_sparse_features(&self) -> usize {
        self.sparse_columns.iter().filter(|s| s.is_some()).count()
    }

    // =========================================================================
    // Target Access
    // =========================================================================

    /// Number of target columns (1 for scalar, >1 for multi-output)
    #[inline]
    pub fn num_target_cols(&self) -> usize {
        self.num_target_cols
    }

    /// Check if this is a multi-output dataset
    #[inline]
    pub fn is_multioutput(&self) -> bool {
        self.num_target_cols > 1
    }

    /// Get target value for a row (scalar targets only)
    ///
    /// For multi-output datasets, use `get_target(row, label_idx)` instead.
    #[inline]
    pub fn target(&self, row_idx: usize) -> f32 {
        debug_assert_eq!(self.num_target_cols, 1, "Use get_target() for multi-output");
        self.targets[row_idx]
    }

    /// Get target value for a specific row and output index
    ///
    /// Works for both scalar and multi-output datasets.
    #[inline]
    pub fn get_target(&self, row_idx: usize, label_idx: usize) -> f32 {
        debug_assert!(label_idx < self.num_target_cols);
        self.targets[row_idx * self.num_target_cols + label_idx]
    }

    /// Get all target values for a row (for multi-output)
    ///
    /// Returns a slice of length `num_target_cols`.
    #[inline]
    pub fn get_targets_row(&self, row_idx: usize) -> &[f32] {
        let start = row_idx * self.num_target_cols;
        &self.targets[start..start + self.num_target_cols]
    }

    /// Get all targets (raw buffer)
    ///
    /// For scalar targets: `targets[row_idx]`
    /// For multi-output: `targets[row * num_target_cols + label]`
    #[inline]
    pub fn targets(&self) -> &[f32] {
        &self.targets
    }

    /// Get mutable targets (for testing with outliers, etc.)
    #[inline]
    pub fn targets_mut(&mut self) -> &mut [f32] {
        &mut self.targets
    }

    /// Create a new dataset with replaced scalar targets, sharing feature data
    ///
    /// This is more efficient than `clone()` followed by target modification
    /// because it avoids cloning the targets vector only to overwrite it.
    ///
    /// # Arguments
    /// * `new_targets` - New target values (must have same length as num_rows)
    ///
    /// # Panics
    /// Panics if `new_targets.len() != self.num_rows`
    pub fn with_targets(&self, new_targets: Vec<f32>) -> Self {
        assert_eq!(
            new_targets.len(),
            self.num_rows,
            "new_targets length ({}) must match num_rows ({})",
            new_targets.len(),
            self.num_rows
        );

        Self {
            num_rows: self.num_rows,
            features: self.features.clone(),
            targets: new_targets, // Use provided targets directly, no clone
            num_target_cols: 1,   // Scalar targets
            feature_info: self.feature_info.clone(),
            sparse_columns: self.sparse_columns.clone(),
            era_indices: self.era_indices.clone(),
            num_eras: self.num_eras,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: self.raw_features.clone(),
        }
    }

    /// Create a new dataset with replaced multi-output targets, sharing feature data
    ///
    /// # Arguments
    /// * `new_targets` - New target values, row-wise layout
    /// * `num_target_cols` - Number of target columns
    ///
    /// # Panics
    /// Panics if `new_targets.len() != self.num_rows * num_target_cols`
    pub fn with_targets_multioutput(&self, new_targets: Vec<f32>, num_target_cols: usize) -> Self {
        assert_eq!(
            new_targets.len(),
            self.num_rows * num_target_cols,
            "new_targets length ({}) must match num_rows ({}) * num_target_cols ({})",
            new_targets.len(),
            self.num_rows,
            num_target_cols
        );

        Self {
            num_rows: self.num_rows,
            features: self.features.clone(),
            targets: new_targets,
            num_target_cols,
            feature_info: self.feature_info.clone(),
            sparse_columns: self.sparse_columns.clone(),
            era_indices: self.era_indices.clone(),
            num_eras: self.num_eras,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: self.raw_features.clone(),
        }
    }

    // =========================================================================
    // Bin Value Operations
    // =========================================================================

    /// Get raw bin value for original feature value using binary search
    pub fn bin_value(&self, feature_idx: usize, value: f64) -> u8 {
        let info = &self.feature_info[feature_idx];
        if info.bin_boundaries.is_empty() {
            return 0;
        }

        // Binary search for the appropriate bin
        match info
            .bin_boundaries
            .binary_search_by(|b| b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(idx) => (idx + 1).min(info.num_bins as usize - 1) as u8,
            Err(idx) => idx.min(info.num_bins as usize - 1) as u8,
        }
    }

    /// Get the actual split value for a given feature and bin threshold
    ///
    /// For raw prediction without binning, we need the actual threshold value.
    /// Samples with value <= split_value go left.
    ///
    /// # Arguments
    /// * `feature_idx` - Feature index
    /// * `bin_threshold` - Bin threshold from tree split
    ///
    /// # Returns
    /// The actual split value (f64) for raw data comparison
    #[inline]
    pub fn get_split_value(&self, feature_idx: usize, bin_threshold: u8) -> f64 {
        let info = &self.feature_info[feature_idx];

        // Edge cases
        if info.bin_boundaries.is_empty() {
            return 0.0;
        }

        // bin_threshold is the largest bin that goes left
        // bin_boundaries[i] is the upper bound of bin i
        // So split_value = bin_boundaries[bin_threshold] (if exists)
        // For bin_threshold = 0, samples in bin 0 go left, threshold is boundaries[0]
        let idx = bin_threshold as usize;
        if idx < info.bin_boundaries.len() {
            info.bin_boundaries[idx]
        } else {
            // bin_threshold >= num_bins - 1, use the last boundary
            // or max value (everything goes left)
            info.bin_boundaries.last().copied().unwrap_or(f64::MAX)
        }
    }

    /// Extract raw feature values from bins (lossy approximation)
    ///
    /// Reconstructs continuous feature values from quantized bins using bin-center
    /// approximation. This is a LOSSY operation that should only be used when:
    /// 1. Raw features are unavailable (e.g., model loaded from disk)
    /// 2. The downstream algorithm can tolerate approximate values
    ///
    /// # Accuracy Trade-offs
    ///
    /// - **Numeric features**: Uses bin split values as approximations. Precision
    ///   loss depends on binning resolution (typically 256 bins).
    /// - **Categorical features**: Returns bin indices cast to f32 (not actual categories).
    ///
    /// For LinearThenTree mode, passing actual raw features to training is **strongly
    /// recommended** for optimal linear model accuracy.
    ///
    /// # Layout
    ///
    /// Returns row-major layout: `features[row * num_features + feature]`
    pub fn extract_raw_features_from_bins(&self) -> Vec<f32> {
        let num_rows = self.num_rows();
        let num_features = self.num_features();
        let mut raw = vec![0.0f32; num_rows * num_features];

        for row in 0..num_rows {
            for feat in 0..num_features {
                let bin = self.get_bin(row, feat);
                raw[row * num_features + feat] = self.get_split_value(feat, bin) as f32;
            }
        }

        raw
    }

    // =========================================================================
    // Row-Major Layout (GPU)
    // =========================================================================

    /// Get row-major layout for GPU backends (lazy conversion).
    ///
    /// Converts column-major `bins[feature][row]` to row-major `bins[row][feature]`.
    /// Result is cached for subsequent calls.
    ///
    /// # Returns
    /// A slice of row-major bin data: `bins[row * num_features + feature]`
    pub fn as_row_major(&self) -> &[u8] {
        self.row_major_cache.get_or_init(|| {
            let num_rows = self.num_rows;
            let num_features = self.num_features();
            let mut row_major = vec![0u8; num_rows * num_features];

            // Transpose: column-major → row-major
            // Column-major: features[feature * num_rows + row]
            // Row-major: row_major[row * num_features + feature]
            for row in 0..num_rows {
                for feature in 0..num_features {
                    row_major[row * num_features + feature] =
                        self.features[feature * num_rows + row];
                }
            }

            row_major
        })
    }

    /// Get maximum number of bins across all features.
    #[inline]
    pub fn max_bins(&self) -> u8 {
        self.feature_info
            .iter()
            .map(|f| f.num_bins)
            .max()
            .unwrap_or(0)
    }

    /// Check if dataset supports 4-bit bin packing.
    ///
    /// Returns true if all features have ≤16 bins, enabling 4-bit packing
    /// for 50% memory bandwidth reduction on GPU.
    #[inline]
    pub fn supports_4bit(&self) -> bool {
        self.max_bins() <= 16
    }

    /// Get 4-bit packed row-major layout for GPU backends (lazy conversion).
    ///
    /// Packs two 4-bit bin values into each byte (nibble packing).
    /// Only valid when all features have ≤16 bins.
    ///
    /// Layout: For each row, features are packed in pairs:
    /// - `byte[i] = (feature[2i+1] << 4) | feature[2i]`
    /// - If odd number of features, last nibble is padded with 0
    ///
    /// # Panics
    /// Panics if any feature has more than 16 bins.
    ///
    /// # Returns
    /// A slice of 4-bit packed row-major bin data
    pub fn as_row_major_4bit(&self) -> &[u8] {
        self.row_major_4bit_cache.get_or_init(|| {
            assert!(
                self.supports_4bit(),
                "4-bit packing requires all features to have ≤16 bins, max is {}",
                self.max_bins()
            );

            let num_rows = self.num_rows;
            let num_features = self.num_features();
            // Each pair of features packs into 1 byte
            let bytes_per_row = num_features.div_ceil(2);
            let mut packed = vec![0u8; num_rows * bytes_per_row];

            for row in 0..num_rows {
                let row_offset = row * bytes_per_row;
                for pair in 0..bytes_per_row {
                    let f0 = pair * 2;
                    let f1 = f0 + 1;

                    // Get bin values (4-bit, clamped to 0-15)
                    let bin0 = self.features[f0 * num_rows + row] & 0x0F;
                    let bin1 = if f1 < num_features {
                        self.features[f1 * num_rows + row] & 0x0F
                    } else {
                        0 // Padding for odd number of features
                    };

                    // Pack: low nibble = even feature, high nibble = odd feature
                    packed[row_offset + pair] = bin0 | (bin1 << 4);
                }
            }

            packed
        })
    }

    /// Get the number of bytes per row in 4-bit packed format.
    #[inline]
    pub fn bytes_per_row_4bit(&self) -> usize {
        self.num_features().div_ceil(2)
    }

    // =========================================================================
    // Subset Operations
    // =========================================================================

    /// Create a subset of this dataset containing only the specified rows.
    ///
    /// This is used for proper K-fold cross-validation where training must
    /// be performed on a subset of the data (the training fold).
    ///
    /// Preserves multi-output structure (num_target_cols).
    ///
    /// # Arguments
    /// * `indices` - Row indices to include in the subset (must be valid)
    ///
    /// # Returns
    /// A new BinnedDataset containing only the specified rows
    ///
    /// # Panics
    /// Panics if any index is out of bounds.
    pub fn subset_by_indices(&self, indices: &[usize]) -> Self {
        let new_num_rows = indices.len();
        let num_features = self.num_features();
        let num_target_cols = self.num_target_cols;

        // Validate indices
        for &idx in indices {
            assert!(
                idx < self.num_rows,
                "Index {} out of bounds for dataset with {} rows",
                idx,
                self.num_rows
            );
        }

        // Extract feature data for the selected rows (column-major)
        let mut new_features = Vec::with_capacity(new_num_rows * num_features);
        for f in 0..num_features {
            let col_start = f * self.num_rows;
            for &idx in indices {
                new_features.push(self.features[col_start + idx]);
            }
        }

        // Extract targets for the selected rows (preserving multi-output layout)
        let mut new_targets = Vec::with_capacity(new_num_rows * num_target_cols);
        for &idx in indices {
            let start = idx * num_target_cols;
            new_targets.extend_from_slice(&self.targets[start..start + num_target_cols]);
        }

        // Extract era indices if present
        let new_era_indices = self
            .era_indices
            .as_ref()
            .map(|eras| indices.iter().map(|&idx| eras[idx]).collect::<Vec<u16>>());

        // Recompute sparse columns for the new dataset
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * new_num_rows;
                let column = &new_features[start..start + new_num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        // Compute new num_eras
        let new_num_eras = new_era_indices
            .as_ref()
            .and_then(|eras| eras.iter().copied().max())
            .map(|m| m as usize + 1)
            .unwrap_or(0);

        // Filter raw_features if present (row-major layout: row × feature)
        // This method only filters rows, keeps all features
        let new_raw_features = self.raw_features.as_ref().map(|raw| {
            let num_features = self.num_features();
            let mut filtered = Vec::with_capacity(new_num_rows * num_features);
            for &idx in indices {
                let row_start = idx * num_features;
                filtered.extend_from_slice(&raw[row_start..row_start + num_features]);
            }
            filtered
        });

        Self {
            num_rows: new_num_rows,
            features: new_features,
            targets: new_targets,
            num_target_cols,
            feature_info: self.feature_info.clone(),
            sparse_columns,
            era_indices: new_era_indices,
            num_eras: new_num_eras,
            raw_features: new_raw_features,
            // Caches are not copied - will be recomputed if needed
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
        }
    }

    /// Create a new BinnedDataset with only the specified features
    ///
    /// This is useful for LinearThenTree mode where you want to train the tree
    /// on a subset of features (e.g., only categorical features, excluding
    /// polynomial/interaction features that the linear model already used).
    ///
    /// # Arguments
    ///
    /// * `feature_indices` - Indices of features to keep
    ///
    /// # Returns
    ///
    /// A new BinnedDataset with only the selected features (all rows preserved)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Keep only features 0, 2, 5 (e.g., original categorical features)
    /// let tree_features = &[0, 2, 5];
    /// let filtered_dataset = dataset.subset_features(tree_features);
    /// assert_eq!(filtered_dataset.num_features(), 3);
    /// assert_eq!(filtered_dataset.num_rows(), dataset.num_rows()); // Same rows
    /// ```
    pub fn subset_features(&self, feature_indices: &[usize]) -> Self {
        let new_num_features = feature_indices.len();
        let num_features = self.num_features();
        let num_rows = self.num_rows;

        // Validate feature indices
        for &idx in feature_indices {
            assert!(
                idx < num_features,
                "Feature index {} out of bounds for dataset with {} features",
                idx,
                num_features
            );
        }

        // Extract selected feature columns (column-major layout)
        let mut new_features = Vec::with_capacity(num_rows * new_num_features);
        for &feat_idx in feature_indices {
            let col_start = feat_idx * num_rows;
            let col_end = col_start + num_rows;
            new_features.extend_from_slice(&self.features[col_start..col_end]);
        }

        // Keep all targets unchanged (no feature filtering for targets)
        let new_targets = self.targets.clone();

        // Extract selected feature_info
        let new_feature_info: Vec<FeatureInfo> = feature_indices
            .iter()
            .map(|&idx| self.feature_info[idx].clone())
            .collect();

        // Recompute sparse columns for selected features
        let sparse_columns: Vec<Option<SparseColumn>> = (0..new_num_features)
            .map(|f| {
                let start = f * num_rows;
                let column = &new_features[start..start + num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        // Filter raw_features if present (row-major layout)
        let new_raw_features = self.raw_features.as_ref().map(|raw| {
            let mut filtered = Vec::with_capacity(num_rows * new_num_features);
            for row in 0..num_rows {
                for &feat_idx in feature_indices {
                    filtered.push(raw[row * num_features + feat_idx]);
                }
            }
            filtered
        });

        Self {
            num_rows,
            features: new_features,
            targets: new_targets,
            num_target_cols: self.num_target_cols,
            feature_info: new_feature_info,
            sparse_columns,
            era_indices: self.era_indices.clone(),
            num_eras: self.num_eras,
            // Caches are not copied - will be recomputed if needed
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
            raw_features: new_raw_features,
        }
    }

    // =========================================================================
    // Concatenation (for pre-split validation)
    // =========================================================================

    /// Concatenate two BinnedDatasets row-wise
    ///
    /// This is used for pre-split validation where you have separate train and
    /// validation datasets that need to be combined for the training loop
    /// (which uses indices to distinguish train vs validation rows).
    ///
    /// # Requirements
    /// - Both datasets MUST have the same number of features
    /// - Both datasets MUST have compatible feature_info (same bin edges)
    /// - Both datasets MUST have the same num_target_cols
    ///
    /// # Returns
    /// A new BinnedDataset where:
    /// - Rows 0..self.num_rows() are from `self` (typically training data)
    /// - Rows self.num_rows()..total are from `other` (typically validation data)
    ///
    /// # Example
    /// ```ignore
    /// let train_data: BinnedDataset = ...;
    /// let val_data: BinnedDataset = ...;
    ///
    /// let combined = train_data.concat(&val_data)?;
    ///
    /// // Train indices: 0..train_data.num_rows()
    /// // Val indices: train_data.num_rows()..combined.num_rows()
    /// ```
    pub fn concat(&self, other: &Self) -> crate::Result<Self> {
        // Validate compatibility
        if self.num_features() != other.num_features() {
            return Err(crate::TreeBoostError::Config(format!(
                "Cannot concat datasets with different number of features: {} vs {}",
                self.num_features(),
                other.num_features()
            )));
        }

        if self.num_target_cols != other.num_target_cols {
            return Err(crate::TreeBoostError::Config(format!(
                "Cannot concat datasets with different num_target_cols: {} vs {}",
                self.num_target_cols, other.num_target_cols
            )));
        }

        // Validate feature_info compatibility (same bin edges)
        for (i, (a, b)) in self
            .feature_info
            .iter()
            .zip(other.feature_info.iter())
            .enumerate()
        {
            if a.num_bins != b.num_bins {
                return Err(crate::TreeBoostError::Config(format!(
                    "Feature {} has incompatible bins: {} vs {}. \
                     Both datasets must be binned with the same binner.",
                    i, a.num_bins, b.num_bins
                )));
            }
        }

        let new_num_rows = self.num_rows + other.num_rows;
        let num_features = self.num_features();
        let num_target_cols = self.num_target_cols;

        // Concatenate features (column-major layout)
        // For each feature column, append other's rows after self's rows
        let mut new_features = Vec::with_capacity(new_num_rows * num_features);
        for f in 0..num_features {
            // Self's column
            let self_start = f * self.num_rows;
            new_features.extend_from_slice(&self.features[self_start..self_start + self.num_rows]);
            // Other's column
            let other_start = f * other.num_rows;
            new_features
                .extend_from_slice(&other.features[other_start..other_start + other.num_rows]);
        }

        // Concatenate targets (row-wise layout for multi-output)
        let mut new_targets = Vec::with_capacity(new_num_rows * num_target_cols);
        new_targets.extend_from_slice(&self.targets);
        new_targets.extend_from_slice(&other.targets);

        // Concatenate era_indices - both must have them or both must not
        let new_era_indices = match (&self.era_indices, &other.era_indices) {
            (Some(self_eras), Some(other_eras)) => {
                let mut combined = self_eras.clone();
                combined.extend_from_slice(other_eras);
                Some(combined)
            }
            (None, None) => None,
            _ => {
                return Err(crate::TreeBoostError::Config(
                    "Cannot concatenate datasets: both must have era_indices or neither (inconsistent data)".to_string(),
                ))
            }
        };

        // Recompute sparse columns for the combined dataset
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * new_num_rows;
                let column = &new_features[start..start + new_num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        // Compute new num_eras
        let new_num_eras = new_era_indices
            .as_ref()
            .and_then(|eras| eras.iter().copied().max())
            .map(|m| m as usize + 1)
            .unwrap_or(0);

        // Concatenate raw_features - both must have them or both must not
        let new_raw_features = match (&self.raw_features, &other.raw_features) {
            (Some(self_raw), Some(other_raw)) => {
                let mut combined = self_raw.clone();
                combined.extend_from_slice(other_raw);
                Some(combined)
            }
            (None, None) => None,
            _ => {
                return Err(crate::TreeBoostError::Config(
                    "Cannot concatenate datasets: both must have raw_features or neither (inconsistent data)".to_string(),
                ))
            }
        };

        Ok(Self {
            num_rows: new_num_rows,
            features: new_features,
            targets: new_targets,
            num_target_cols,
            feature_info: self.feature_info.clone(), // Use self's feature_info
            sparse_columns,
            era_indices: new_era_indices,
            num_eras: new_num_eras,
            raw_features: new_raw_features,
            // Caches must be recomputed
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
        })
    }
}

// =============================================================================
// BinStorage Trait Implementation
// =============================================================================

impl crate::backend::BinStorage for BinnedDataset {
    fn get_bin(&self, row: usize, feature: usize) -> u8 {
        self.features[feature * self.num_rows + row]
    }

    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_features(&self) -> usize {
        self.feature_info.len()
    }

    fn feature_column(&self, feature: usize) -> Option<&[u8]> {
        let start = feature * self.num_rows;
        Some(&self.features[start..start + self.num_rows])
    }

    fn sparse_column(&self, feature: usize) -> Option<&SparseColumn> {
        self.sparse_columns.get(feature).and_then(|s| s.as_ref())
    }

    fn as_row_major(&self) -> Option<&[u8]> {
        // Delegate to the lazy-cached method
        Some(BinnedDataset::as_row_major(self))
    }

    fn max_bins(&self) -> u8 {
        BinnedDataset::max_bins(self)
    }

    fn supports_4bit(&self) -> bool {
        BinnedDataset::supports_4bit(self)
    }

    fn as_row_major_4bit(&self) -> Option<&[u8]> {
        if self.supports_4bit() {
            Some(BinnedDataset::as_row_major_4bit(self))
        } else {
            None
        }
    }

    fn bytes_per_row_4bit(&self) -> usize {
        BinnedDataset::bytes_per_row_4bit(self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FeatureType;

    fn create_test_feature_info(name: &str, num_bins: u8) -> FeatureInfo {
        FeatureInfo {
            name: name.to_string(),
            feature_type: FeatureType::Numeric,
            num_bins,
            bin_boundaries: vec![],
            impute_value: 0.0,
        }
    }

    fn create_test_feature_info_with_boundaries(
        name: &str,
        num_bins: u8,
        boundaries: Vec<f64>,
    ) -> FeatureInfo {
        FeatureInfo {
            name: name.to_string(),
            feature_type: FeatureType::Numeric,
            num_bins,
            bin_boundaries: boundaries,
            impute_value: 0.0,
        }
    }

    #[test]
    fn test_binned_dataset_access() {
        let num_rows = 4;

        // Column-major: feature 0 = [0,1,2,3], feature 1 = [10,11,12,13]
        let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            create_test_feature_info_with_boundaries("f0", 4, vec![0.5, 1.5, 2.5]),
            create_test_feature_info_with_boundaries("f1", 4, vec![10.5, 11.5, 12.5]),
        ];

        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        assert_eq!(dataset.num_rows(), 4);
        assert_eq!(dataset.num_features(), 2);

        // Test individual access
        assert_eq!(dataset.get_bin(0, 0), 0);
        assert_eq!(dataset.get_bin(2, 0), 2);
        assert_eq!(dataset.get_bin(1, 1), 11);

        // Test column access
        assert_eq!(dataset.feature_column(0), &[0, 1, 2, 3]);
        assert_eq!(dataset.feature_column(1), &[10, 11, 12, 13]);

        // Test targets
        assert_eq!(dataset.target(0), 1.0);
        assert_eq!(dataset.targets(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_row_major_conversion() {
        let num_rows = 4;
        let num_features = 2;

        // Column-major: feature 0 = [0,1,2,3], feature 1 = [10,11,12,13]
        let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            create_test_feature_info_with_boundaries("f0", 4, vec![0.5, 1.5, 2.5]),
            create_test_feature_info_with_boundaries("f1", 4, vec![10.5, 11.5, 12.5]),
        ];

        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        // Get row-major layout
        let row_major = dataset.as_row_major();

        // Verify row-major format: [row0_feat0, row0_feat1, row1_feat0, row1_feat1, ...]
        // Row 0: (0, 10), Row 1: (1, 11), Row 2: (2, 12), Row 3: (3, 13)
        assert_eq!(row_major.len(), num_rows * num_features);
        assert_eq!(row_major[0], 0); // row 0, feature 0
        assert_eq!(row_major[1], 10); // row 0, feature 1
        assert_eq!(row_major[num_features], 1); // row 1, feature 0
        assert_eq!(row_major[num_features + 1], 11); // row 1, feature 1
        assert_eq!(row_major[3 * num_features], 3); // row 3, feature 0
        assert_eq!(row_major[3 * num_features + 1], 13); // row 3, feature 1

        // Verify caching (second call returns same data)
        let row_major2 = dataset.as_row_major();
        assert_eq!(row_major.as_ptr(), row_major2.as_ptr());
    }

    #[test]
    fn test_row_major_via_bin_storage_trait() {
        use crate::backend::BinStorage;

        let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            create_test_feature_info("f0", 4),
            create_test_feature_info("f1", 4),
        ];

        let dataset = BinnedDataset::new(4, features, targets, feature_info);

        // Access via trait
        let storage: &dyn BinStorage = &dataset;
        let row_major = storage.as_row_major();
        assert!(row_major.is_some());

        let data = row_major.unwrap();
        assert_eq!(data.len(), 8);
        // Row 0: (0, 10)
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 10);
    }

    #[test]
    fn test_4bit_packing() {
        let num_rows = 4;

        // Column-major: feature 0 = [1,2,3,4], feature 1 = [5,6,7,8], feature 2 = [9,10,11,12]
        // All values <= 15, so 4-bit packing is supported
        let features = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            create_test_feature_info("f0", 16),
            create_test_feature_info("f1", 16),
            create_test_feature_info("f2", 16),
        ];

        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        // Verify 4-bit support
        assert!(dataset.supports_4bit());
        assert_eq!(dataset.max_bins(), 16);
        assert_eq!(dataset.bytes_per_row_4bit(), 2); // ceil(3/2) = 2 bytes per row

        // Get 4-bit packed data
        let packed = dataset.as_row_major_4bit();

        assert_eq!(packed.len(), num_rows * 2);

        // Verify row 0
        assert_eq!(packed[0], 0x51); // (5 << 4) | 1
        assert_eq!(packed[1], 0x09); // (0 << 4) | 9

        // Verify row 1
        assert_eq!(packed[2], 0x62); // (6 << 4) | 2
        assert_eq!(packed[3], 0x0A); // (0 << 4) | 10

        // Verify row 3
        assert_eq!(packed[6], 0x84); // (8 << 4) | 4
        assert_eq!(packed[7], 0x0C); // (0 << 4) | 12

        // Verify unpacking works correctly
        for row in 0..num_rows {
            let row_offset = row * 2;
            // Feature 0 (low nibble of byte 0)
            let bin0 = packed[row_offset] & 0x0F;
            assert_eq!(bin0, (row + 1) as u8);
            // Feature 1 (high nibble of byte 0)
            let bin1 = (packed[row_offset] >> 4) & 0x0F;
            assert_eq!(bin1, (row + 5) as u8);
            // Feature 2 (low nibble of byte 1)
            let bin2 = packed[row_offset + 1] & 0x0F;
            assert_eq!(bin2, (row + 9) as u8);
        }
    }

    #[test]
    fn test_4bit_not_supported_for_large_bins() {
        let features = vec![0u8, 1, 2, 3, 100, 101, 102, 103];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            create_test_feature_info("f0", 16),
            create_test_feature_info("f1", 128), // More than 16 bins
        ];

        let dataset = BinnedDataset::new(4, features, targets, feature_info);

        assert!(!dataset.supports_4bit());
        assert_eq!(dataset.max_bins(), 128);
    }

    #[test]
    fn test_4bit_via_bin_storage_trait() {
        use crate::backend::BinStorage;

        let features = vec![1u8, 2, 5, 6];
        let targets = vec![1.0f32, 2.0];
        let feature_info = vec![
            create_test_feature_info("f0", 8),
            create_test_feature_info("f1", 8),
        ];

        let dataset = BinnedDataset::new(2, features, targets, feature_info);
        let storage: &dyn BinStorage = &dataset;

        assert!(storage.supports_4bit());
        assert_eq!(storage.max_bins(), 8);

        let packed = storage.as_row_major_4bit();
        assert!(packed.is_some());

        let data = packed.unwrap();
        assert_eq!(data.len(), 2); // 2 rows, 1 byte per row (2 features)
        assert_eq!(data[0], 0x51);
        assert_eq!(data[1], 0x62);
    }

    #[test]
    fn test_subset_by_indices() {
        let num_rows = 5;

        // Column-major: feature 0 = [0,1,2,3,4], feature 1 = [10,11,12,13,14]
        let features = vec![0u8, 1, 2, 3, 4, 10, 11, 12, 13, 14];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let feature_info = vec![
            create_test_feature_info_with_boundaries("f0", 5, vec![0.5, 1.5, 2.5, 3.5]),
            create_test_feature_info_with_boundaries("f1", 5, vec![10.5, 11.5, 12.5, 13.5]),
        ];

        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        // Take indices [1, 3, 4]
        let subset = dataset.subset_by_indices(&[1, 3, 4]);

        assert_eq!(subset.num_rows(), 3);
        assert_eq!(subset.num_features(), 2);

        // Check feature values: original rows 1, 3, 4 should become rows 0, 1, 2
        assert_eq!(subset.get_bin(0, 0), 1);
        assert_eq!(subset.get_bin(1, 0), 3);
        assert_eq!(subset.get_bin(2, 0), 4);

        assert_eq!(subset.get_bin(0, 1), 11);
        assert_eq!(subset.get_bin(1, 1), 13);
        assert_eq!(subset.get_bin(2, 1), 14);

        // Check targets
        assert_eq!(subset.targets(), &[2.0, 4.0, 5.0]);

        // Check feature columns are consistent
        assert_eq!(subset.feature_column(0), &[1, 3, 4]);
        assert_eq!(subset.feature_column(1), &[11, 13, 14]);
    }

    #[test]
    fn test_subset_by_indices_with_eras() {
        let num_rows = 4;

        let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            create_test_feature_info("f0", 4),
            create_test_feature_info("f1", 14),
        ];
        let era_indices = vec![0u16, 1, 0, 2];

        let dataset =
            BinnedDataset::new_with_eras(num_rows, features, targets, feature_info, era_indices);

        // Take indices [0, 2] (both have era 0)
        let subset = dataset.subset_by_indices(&[0, 2]);

        assert_eq!(subset.num_rows(), 2);
        assert!(subset.has_eras());
        assert_eq!(subset.era_indices().unwrap(), &[0, 0]);
        assert_eq!(subset.num_eras(), 1);
    }

    #[test]
    fn test_with_targets() {
        let features = vec![0u8, 1, 2, 3];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![create_test_feature_info("f0", 4)];

        let dataset = BinnedDataset::new(4, features, targets, feature_info);
        let new_dataset = dataset.with_targets(vec![10.0, 20.0, 30.0, 40.0]);

        assert_eq!(new_dataset.targets(), &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(new_dataset.num_rows(), 4);
        // Original unchanged
        assert_eq!(dataset.targets(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_era_methods() {
        let features = vec![0u8, 1, 2, 3];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![create_test_feature_info("f0", 4)];
        let era_indices = vec![0u16, 1, 2, 1];

        let dataset = BinnedDataset::new_with_eras(4, features, targets, feature_info, era_indices);

        assert!(dataset.has_eras());
        assert_eq!(dataset.num_eras(), 3);
        assert_eq!(dataset.era(0), 0);
        assert_eq!(dataset.era(1), 1);
        assert_eq!(dataset.era(2), 2);
        assert_eq!(dataset.era(3), 1);
    }

    #[test]
    fn test_set_era_indices() {
        let features = vec![0u8, 1, 2, 3];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![create_test_feature_info("f0", 4)];

        let mut dataset = BinnedDataset::new(4, features, targets, feature_info);
        assert!(!dataset.has_eras());

        dataset.set_era_indices(vec![0, 1, 0, 2]);
        assert!(dataset.has_eras());
        assert_eq!(dataset.num_eras(), 3);
    }
}
