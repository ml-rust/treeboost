//! Preprocessing pipeline with enum dispatch pattern
//!
//! Provides a unified `Preprocessor` enum that wraps all preprocessing transformations.
//! This enables:
//! - **Serialization**: Easy to serialize/deserialize with serde/rkyv
//! - **Static dispatch**: No trait object overhead for built-in types
//! - **Pipeline composition**: Chain multiple preprocessors
//! - **Extensibility**: Custom preprocessors via the `Custom` variant
//!
//! # Example
//!
//! ```rust
//! use treeboost::preprocessing::{Preprocessor, StandardScaler, FrequencyEncoder};
//!
//! // Create a pipeline
//! let pipeline = Preprocessor::Pipeline(Box::new(vec![
//!     Preprocessor::Standard(StandardScaler::new()),
//!     // FrequencyEncoder would be added for categorical columns
//! ]));
//!
//! // Serialize the entire pipeline
//! let json = serde_json::to_string(&pipeline).unwrap();
//! ```
//!
//! # Custom Preprocessors
//!
//! You can create custom preprocessors by implementing [`PreprocessorTrait`]
//! and registering them with [`register_preprocessor`](super::registry::register_preprocessor):
//!
//! ```ignore
//! use treeboost::preprocessing::{Preprocessor, PreprocessorTrait, register_preprocessor};
//!
//! #[derive(Clone, Default)]
//! struct MyScaler { /* ... */ }
//!
//! impl PreprocessorTrait for MyScaler {
//!     fn type_id(&self) -> &'static str { "my_crate::MyScaler" }
//!     // ... implement other methods
//! }
//!
//! // Register at startup
//! register_preprocessor::<MyScaler>();
//!
//! // Use in pipeline
//! let custom = Preprocessor::custom(MyScaler::default())?;
//! ```

use crate::{Result, TreeBoostError};

use super::encoding::{FrequencyEncoder, LabelEncoder, OneHotEncoder};
use super::imputer::SimpleImputer;
use super::registry;
use super::scaler::{MinMaxScaler, RobustScaler, Scaler, StandardScaler};
use super::traits::{CustomPreprocessorDyn, PreprocessorTrait, PreprocessorWrapper};
use super::transforms::YeoJohnsonTransform;

// =============================================================================
// Preprocessor Enum (Dispatch Pattern)
// =============================================================================

/// Unified preprocessor enum for pipeline composition and serialization
///
/// This enum wraps all preprocessing transformations, enabling:
/// - Static dispatch (better performance than trait objects) for built-in types
/// - Easy serialization (no `dyn` complications for built-ins)
/// - Pipeline composition (via `Preprocessor::Pipeline`)
/// - Custom preprocessors (via `Preprocessor::Custom`)
///
/// # Design Rationale
///
/// In Rust, `Box<dyn Trait>` is problematic for serialization (especially with rkyv).
/// The enum dispatch pattern provides the same polymorphism with trivial serialization
/// for built-in types. Custom preprocessors use a registry-based approach for
/// serialization.
///
/// # Example: Built-in Preprocessor
///
/// ```rust
/// use treeboost::preprocessing::{Preprocessor, StandardScaler};
///
/// let mut scaler = Preprocessor::Standard(StandardScaler::new());
///
/// // Fit on training data
/// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
/// scaler.fit_numerical(&data, 3).unwrap();
/// scaler.transform_numerical(&mut data, 3).unwrap();
/// ```
///
/// # Example: Custom Preprocessor
///
/// ```ignore
/// use treeboost::preprocessing::{Preprocessor, PreprocessorTrait, register_preprocessor};
///
/// // Define custom type
/// #[derive(Clone, Default)]
/// struct MyScaler { fitted: bool }
///
/// impl PreprocessorTrait for MyScaler {
///     fn type_id(&self) -> &'static str { "my_crate::MyScaler" }
///     // ... implement methods
/// }
///
/// // Register and use
/// register_preprocessor::<MyScaler>();
/// let custom = Preprocessor::custom(MyScaler::default())?;
/// ```
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum Preprocessor {
    // -------------------------------------------------------------------------
    // Scalers (numerical)
    // -------------------------------------------------------------------------
    /// StandardScaler: (x - μ) / σ
    Standard(StandardScaler),
    /// MinMaxScaler: scale to [min, max] range
    MinMax(MinMaxScaler),
    /// RobustScaler: (x - median) / IQR
    Robust(RobustScaler),

    // -------------------------------------------------------------------------
    // Encoders (categorical)
    // -------------------------------------------------------------------------
    /// FrequencyEncoder: category → count (optimal for trees)
    Frequency(FrequencyEncoder),
    /// LabelEncoder: category → integer label
    Label(LabelEncoder),
    /// OneHotEncoder: category → binary columns (for linear models)
    OneHot(OneHotEncoder),

    // -------------------------------------------------------------------------
    // Imputers (missing values)
    // -------------------------------------------------------------------------
    /// SimpleImputer: Mean/Median/Mode/Constant strategies
    Imputer(SimpleImputer),

    // -------------------------------------------------------------------------
    // Power transforms
    // -------------------------------------------------------------------------
    /// YeoJohnsonTransform: Normalize skewed distributions
    YeoJohnson(YeoJohnsonTransform),

    // -------------------------------------------------------------------------
    // Composition
    // -------------------------------------------------------------------------
    /// Pipeline: chain multiple preprocessors sequentially
    Pipeline(Box<Vec<Preprocessor>>),

    // -------------------------------------------------------------------------
    // Custom (user-defined)
    // -------------------------------------------------------------------------
    /// Custom preprocessor from user code
    ///
    /// This variant stores:
    /// - `type_id`: Unique identifier for registry lookup (e.g., "my_crate::MyScaler")
    /// - `serialized_state`: The preprocessor's state as bytes (produced by `serialize_state()`)
    /// - `instance`: Cached runtime instance (not serialized, reconstructed on load)
    ///
    /// To create a custom preprocessor:
    /// 1. Implement [`PreprocessorTrait`]
    /// 2. Call [`register_preprocessor`](super::registry::register_preprocessor) at startup
    /// 3. Use [`Preprocessor::custom()`] to create the Custom variant
    Custom {
        /// Type identifier for registry lookup
        type_id: String,
        /// Serialized state (produced by `PreprocessorTrait::serialize_state()`)
        serialized_state: Vec<u8>,
        /// Cached runtime instance (reconstructed from registry on deserialize)
        #[serde(skip)]
        instance: Option<Box<dyn CustomPreprocessorDyn>>,
    },
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard(s) => f.debug_tuple("Standard").field(s).finish(),
            Self::MinMax(s) => f.debug_tuple("MinMax").field(s).finish(),
            Self::Robust(s) => f.debug_tuple("Robust").field(s).finish(),
            Self::Frequency(e) => f.debug_tuple("Frequency").field(e).finish(),
            Self::Label(e) => f.debug_tuple("Label").field(e).finish(),
            Self::OneHot(e) => f.debug_tuple("OneHot").field(e).finish(),
            Self::Imputer(i) => f.debug_tuple("Imputer").field(i).finish(),
            Self::YeoJohnson(t) => f.debug_tuple("YeoJohnson").field(t).finish(),
            Self::Pipeline(steps) => f.debug_tuple("Pipeline").field(steps).finish(),
            Self::Custom { type_id, .. } => f
                .debug_struct("Custom")
                .field("type_id", type_id)
                .finish_non_exhaustive(),
        }
    }
}

impl Preprocessor {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a Custom variant from any [`PreprocessorTrait`] implementor.
    ///
    /// This serializes the preprocessor's state and stores it in the Custom variant.
    /// The type must be registered with [`register_preprocessor`](super::registry::register_preprocessor)
    /// before models containing it can be loaded.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::preprocessing::{Preprocessor, PreprocessorTrait, register_preprocessor};
    ///
    /// #[derive(Clone, Default)]
    /// struct MyScaler { fitted: bool }
    ///
    /// impl PreprocessorTrait for MyScaler {
    ///     fn type_id(&self) -> &'static str { "my_crate::MyScaler" }
    ///     // ... other methods
    /// }
    ///
    /// // Register the type
    /// register_preprocessor::<MyScaler>();
    ///
    /// // Create a custom preprocessor
    /// let mut scaler = MyScaler::default();
    /// scaler.fit_numerical(&data, num_features)?;
    ///
    /// // Wrap in Preprocessor::Custom
    /// let custom = Preprocessor::custom(scaler)?;
    /// ```
    pub fn custom<T: PreprocessorTrait>(preprocessor: T) -> Result<Self> {
        let type_id = preprocessor.type_id().to_string();
        let serialized_state = preprocessor.serialize_state()?;

        Ok(Preprocessor::Custom {
            type_id,
            serialized_state,
            instance: Some(Box::new(PreprocessorWrapper(preprocessor))),
        })
    }

    /// Ensure Custom variant has its runtime instance populated.
    ///
    /// After deserializing a `Preprocessor` (via serde or rkyv), the Custom variant's
    /// `instance` field will be `None`. Call this method to reconstruct the instance
    /// from the registry.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The type_id is not registered (call `register_preprocessor` first)
    /// - The serialized state cannot be deserialized
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Register types before loading
    /// register_preprocessor::<MyScaler>();
    ///
    /// // Load and initialize
    /// let mut preprocessor: Preprocessor = serde_json::from_str(&json)?;
    /// preprocessor.ensure_initialized()?;
    /// ```
    pub fn ensure_initialized(&mut self) -> Result<()> {
        if let Preprocessor::Custom {
            type_id,
            serialized_state,
            instance,
        } = self
        {
            if instance.is_none() {
                *instance = Some(registry::construct_preprocessor(type_id, serialized_state)?);
            }
        }

        // Recursively initialize pipeline steps
        if let Preprocessor::Pipeline(steps) = self {
            for step in steps.iter_mut() {
                step.ensure_initialized()?;
            }
        }

        Ok(())
    }

    /// Update the serialized state after fitting/transforming a Custom preprocessor.
    ///
    /// This is called internally after operations that modify the Custom's state.
    fn sync_custom_state(&mut self) -> Result<()> {
        if let Preprocessor::Custom {
            serialized_state,
            instance: Some(inst),
            ..
        } = self
        {
            *serialized_state = inst.serialize_state()?;
        }
        Ok(())
    }

    // =========================================================================
    // Numerical Operations (for scalers)
    // =========================================================================

    /// Fit on numerical data (for scalers)
    ///
    /// This operation is valid for:
    /// - StandardScaler, MinMaxScaler, RobustScaler
    /// - Pipeline (fits all numerical preprocessors in sequence)
    /// - Custom preprocessors that support numerical operations
    ///
    /// Returns error for categorical encoders (use `fit_categorical` instead).
    pub fn fit_numerical(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        match self {
            Preprocessor::Standard(s) => s.fit(data, num_features),
            Preprocessor::MinMax(s) => s.fit(data, num_features),
            Preprocessor::Robust(s) => s.fit(data, num_features),
            Preprocessor::Imputer(i) => i.fit(data, num_features),
            Preprocessor::YeoJohnson(t) => t.fit(data, num_features),
            Preprocessor::Pipeline(steps) => {
                // For pipelines, fit each step in sequence
                // Note: This creates a copy for intermediate transformations
                let mut current_data = data.to_vec();
                for step in steps.iter_mut() {
                    step.fit_numerical(&current_data, num_features)?;
                    step.transform_numerical(&mut current_data, num_features)?;
                }
                Ok(())
            }
            Preprocessor::Custom { instance, .. } => {
                let inst = instance.as_mut().ok_or_else(|| {
                    TreeBoostError::Config(
                        "Custom preprocessor not initialized. Call ensure_initialized() first."
                            .into(),
                    )
                })?;
                inst.fit_numerical(data, num_features)?;
                self.sync_custom_state()
            }
            _ => {
                let preprocessor_name = match self {
                    Preprocessor::Frequency(_) => "FrequencyEncoder",
                    Preprocessor::Label(_) => "LabelEncoder",
                    Preprocessor::OneHot(_) => "OneHotEncoder",
                    _ => "unknown preprocessor",
                };
                Err(TreeBoostError::Config(format!(
                    "Cannot call fit_numerical() on {}. Categorical encoders require fit_categorical() instead. \
                     Use fit_categorical() for string/category data, or fit_numerical() for numerical scalers.",
                    preprocessor_name
                )))
            }
        }
    }

    /// Transform numerical data in-place (for scalers, imputers, and transforms)
    pub fn transform_numerical(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        match self {
            Preprocessor::Standard(s) => s.transform(data, num_features),
            Preprocessor::MinMax(s) => s.transform(data, num_features),
            Preprocessor::Robust(s) => s.transform(data, num_features),
            Preprocessor::Imputer(i) => i.transform(data, num_features),
            Preprocessor::YeoJohnson(t) => t.transform(data, num_features),
            Preprocessor::Pipeline(steps) => {
                for step in steps.iter() {
                    step.transform_numerical(data, num_features)?;
                }
                Ok(())
            }
            Preprocessor::Custom { instance, .. } => {
                let inst = instance.as_ref().ok_or_else(|| {
                    TreeBoostError::Config(
                        "Custom preprocessor not initialized. Call ensure_initialized() first."
                            .into(),
                    )
                })?;
                inst.transform_numerical(data, num_features)
            }
            _ => {
                let preprocessor_name = match self {
                    Preprocessor::Frequency(_) => "FrequencyEncoder",
                    Preprocessor::Label(_) => "LabelEncoder",
                    Preprocessor::OneHot(_) => "OneHotEncoder",
                    _ => "unknown preprocessor",
                };
                Err(TreeBoostError::Config(format!(
                    "Cannot call transform_numerical() on {}. Categorical encoders require transform_categorical() instead. \
                     Use transform_categorical() for string/category data, or transform_numerical() for numerical scalers.",
                    preprocessor_name
                )))
            }
        }
    }

    /// Fit and transform numerical data (convenience)
    pub fn fit_transform_numerical(&mut self, data: &mut [f32], num_features: usize) -> Result<()> {
        self.fit_numerical(data, num_features)?;
        self.transform_numerical(data, num_features)?;
        Ok(())
    }

    // =========================================================================
    // Categorical Operations (for encoders)
    // =========================================================================

    /// Fit on categorical data (for encoders)
    ///
    /// This operation is valid for:
    /// - FrequencyEncoder, LabelEncoder, OneHotEncoder
    /// - Custom preprocessors that support categorical operations
    ///
    /// Returns error for scalers (use `fit_numerical` instead).
    pub fn fit_categorical(&mut self, categories: &[impl AsRef<str>]) -> Result<()> {
        match self {
            Preprocessor::Frequency(e) => {
                e.fit(categories);
                Ok(())
            }
            Preprocessor::Label(e) => {
                e.fit(categories);
                Ok(())
            }
            Preprocessor::OneHot(e) => {
                e.fit(categories)?;
                Ok(())
            }
            Preprocessor::Custom { instance, .. } => {
                let inst = instance.as_mut().ok_or_else(|| {
                    TreeBoostError::Config(
                        "Custom preprocessor not initialized. Call ensure_initialized() first."
                            .into(),
                    )
                })?;
                // Convert to &[&str] for the trait method
                let cats: Vec<&str> = categories.iter().map(|s| s.as_ref()).collect();
                inst.fit_categorical(&cats)?;
                self.sync_custom_state()
            }
            _ => {
                let preprocessor_name = match self {
                    Preprocessor::Standard(_) => "StandardScaler",
                    Preprocessor::MinMax(_) => "MinMaxScaler",
                    Preprocessor::Robust(_) => "RobustScaler",
                    Preprocessor::Imputer(_) => "SimpleImputer",
                    Preprocessor::YeoJohnson(_) => "YeoJohnsonTransform",
                    _ => "unknown preprocessor",
                };
                Err(TreeBoostError::Config(format!(
                    "Cannot call fit_categorical() on {}. Numerical preprocessors require fit_numerical() instead. \
                     Use fit_numerical() for numerical data, or fit_categorical() for categorical encoders.",
                    preprocessor_name
                )))
            }
        }
    }

    /// Transform categorical data to f32 (for tree input)
    ///
    /// - FrequencyEncoder: category → count (f32)
    /// - LabelEncoder: category → label (f32)
    /// - OneHotEncoder: category → flattened binary columns (f32)
    pub fn transform_categorical(&self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        match self {
            Preprocessor::Frequency(e) => e.transform(categories),
            Preprocessor::Label(e) => e.transform_f32(categories),
            Preprocessor::OneHot(e) => e.transform(categories),
            Preprocessor::Custom { instance, .. } => {
                let inst = instance.as_ref().ok_or_else(|| {
                    TreeBoostError::Config(
                        "Custom preprocessor not initialized. Call ensure_initialized() first."
                            .into(),
                    )
                })?;
                // Convert to &[&str] for the trait method
                let cats: Vec<&str> = categories.iter().map(|s| s.as_ref()).collect();
                inst.transform_categorical(&cats)
            }
            _ => {
                let preprocessor_name = match self {
                    Preprocessor::Standard(_) => "StandardScaler",
                    Preprocessor::MinMax(_) => "MinMaxScaler",
                    Preprocessor::Robust(_) => "RobustScaler",
                    Preprocessor::Imputer(_) => "SimpleImputer",
                    Preprocessor::YeoJohnson(_) => "YeoJohnsonTransform",
                    _ => "unknown preprocessor",
                };
                Err(TreeBoostError::Config(format!(
                    "Cannot call transform_categorical() on {}. Numerical preprocessors require transform_numerical() instead. \
                     Use transform_numerical() for numerical data, or transform_categorical() for categorical encoders.",
                    preprocessor_name
                )))
            }
        }
    }

    /// Fit and transform categorical data (convenience)
    pub fn fit_transform_categorical(
        &mut self,
        categories: &[impl AsRef<str>],
    ) -> Result<Vec<f32>> {
        self.fit_categorical(categories)?;
        self.transform_categorical(categories)
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /// Check if the preprocessor has been fitted
    pub fn is_fitted(&self) -> bool {
        match self {
            Preprocessor::Standard(s) => s.is_fitted(),
            Preprocessor::MinMax(s) => s.is_fitted(),
            Preprocessor::Robust(s) => s.is_fitted(),
            Preprocessor::Frequency(e) => e.is_fitted(),
            Preprocessor::Label(e) => e.is_fitted(),
            Preprocessor::OneHot(e) => e.is_fitted(),
            Preprocessor::Imputer(i) => i.is_fitted(),
            Preprocessor::YeoJohnson(t) => t.is_fitted(),
            Preprocessor::Pipeline(steps) => steps.iter().all(|s| s.is_fitted()),
            Preprocessor::Custom { instance, .. } => {
                instance.as_ref().map(|i| i.is_fitted()).unwrap_or(false)
            }
        }
    }

    /// Check if this is a numerical preprocessor (scaler, imputer, or transform)
    pub fn is_numerical(&self) -> bool {
        match self {
            Preprocessor::Standard(_)
            | Preprocessor::MinMax(_)
            | Preprocessor::Robust(_)
            | Preprocessor::Imputer(_)
            | Preprocessor::YeoJohnson(_) => true,
            Preprocessor::Custom { instance, .. } => instance
                .as_ref()
                .map(|i| i.supports_numerical())
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Check if this is a categorical preprocessor (encoder)
    pub fn is_categorical(&self) -> bool {
        match self {
            Preprocessor::Frequency(_) | Preprocessor::Label(_) | Preprocessor::OneHot(_) => true,
            Preprocessor::Custom { instance, .. } => instance
                .as_ref()
                .map(|i| i.supports_categorical())
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Check if this is an imputer
    pub fn is_imputer(&self) -> bool {
        matches!(self, Preprocessor::Imputer(_))
    }

    /// Check if this is a power transform
    pub fn is_transform(&self) -> bool {
        matches!(self, Preprocessor::YeoJohnson(_))
    }

    /// Check if this is a pipeline
    pub fn is_pipeline(&self) -> bool {
        matches!(self, Preprocessor::Pipeline(_))
    }

    /// Check if this is a custom preprocessor
    pub fn is_custom(&self) -> bool {
        matches!(self, Preprocessor::Custom { .. })
    }

    /// Get the number of output columns for categorical encoders
    ///
    /// Returns `None` for scalers (output columns = input columns).
    pub fn num_output_columns(&self) -> Option<usize> {
        match self {
            Preprocessor::Frequency(_) => Some(1), // Single column output
            Preprocessor::Label(_) => Some(1),     // Single column output
            Preprocessor::OneHot(e) => Some(e.num_columns()),
            _ => None,
        }
    }
}

// =============================================================================
// Pipeline Builder
// =============================================================================

/// Builder for creating preprocessing pipelines
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{PipelineBuilder, StandardScaler, FrequencyEncoder};
///
/// let pipeline = PipelineBuilder::new()
///     .with_scaler(StandardScaler::new())
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct PipelineBuilder {
    steps: Vec<Preprocessor>,
}

impl PipelineBuilder {
    /// Create a new empty pipeline builder
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a StandardScaler to the pipeline
    pub fn with_standard_scaler(mut self) -> Self {
        self.steps
            .push(Preprocessor::Standard(StandardScaler::new()));
        self
    }

    /// Add a MinMaxScaler to the pipeline
    pub fn with_minmax_scaler(mut self) -> Self {
        self.steps.push(Preprocessor::MinMax(MinMaxScaler::new()));
        self
    }

    /// Add a RobustScaler to the pipeline
    pub fn with_robust_scaler(mut self) -> Self {
        self.steps.push(Preprocessor::Robust(RobustScaler::new()));
        self
    }

    /// Add a mean imputer to the pipeline
    pub fn with_mean_imputer(mut self) -> Self {
        self.steps
            .push(Preprocessor::Imputer(SimpleImputer::mean()));
        self
    }

    /// Add a median imputer to the pipeline
    pub fn with_median_imputer(mut self) -> Self {
        self.steps
            .push(Preprocessor::Imputer(SimpleImputer::median()));
        self
    }

    /// Add a custom imputer to the pipeline
    pub fn with_imputer(mut self, imputer: SimpleImputer) -> Self {
        self.steps.push(Preprocessor::Imputer(imputer));
        self
    }

    /// Add a Yeo-Johnson transform to the pipeline
    pub fn with_yeo_johnson(mut self) -> Self {
        self.steps
            .push(Preprocessor::YeoJohnson(YeoJohnsonTransform::new()));
        self
    }

    /// Add a custom scaler to the pipeline
    pub fn with_scaler(mut self, scaler: impl Into<Preprocessor>) -> Self {
        self.steps.push(scaler.into());
        self
    }

    /// Add a preprocessor to the pipeline
    pub fn with_preprocessor(mut self, preprocessor: Preprocessor) -> Self {
        self.steps.push(preprocessor);
        self
    }

    /// Add a custom preprocessor to the pipeline
    ///
    /// This wraps the preprocessor in a `Preprocessor::Custom` variant.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::preprocessing::{PipelineBuilder, PreprocessorTrait, register_preprocessor};
    ///
    /// #[derive(Clone, Default)]
    /// struct MyScaler { /* ... */ }
    ///
    /// impl PreprocessorTrait for MyScaler {
    ///     fn type_id(&self) -> &'static str { "my_crate::MyScaler" }
    ///     // ... other methods
    /// }
    ///
    /// register_preprocessor::<MyScaler>();
    ///
    /// let pipeline = PipelineBuilder::new()
    ///     .with_custom(MyScaler::default())?
    ///     .with_standard_scaler()
    ///     .build();
    /// ```
    pub fn with_custom<T: PreprocessorTrait>(mut self, preprocessor: T) -> Result<Self> {
        self.steps.push(Preprocessor::custom(preprocessor)?);
        Ok(self)
    }

    /// Build the pipeline
    pub fn build(self) -> Preprocessor {
        if self.steps.len() == 1 {
            self.steps.into_iter().next().unwrap()
        } else {
            Preprocessor::Pipeline(Box::new(self.steps))
        }
    }
}

// =============================================================================
// Conversion Traits
// =============================================================================

impl From<StandardScaler> for Preprocessor {
    fn from(scaler: StandardScaler) -> Self {
        Preprocessor::Standard(scaler)
    }
}

impl From<MinMaxScaler> for Preprocessor {
    fn from(scaler: MinMaxScaler) -> Self {
        Preprocessor::MinMax(scaler)
    }
}

impl From<RobustScaler> for Preprocessor {
    fn from(scaler: RobustScaler) -> Self {
        Preprocessor::Robust(scaler)
    }
}

impl From<FrequencyEncoder> for Preprocessor {
    fn from(encoder: FrequencyEncoder) -> Self {
        Preprocessor::Frequency(encoder)
    }
}

impl From<LabelEncoder> for Preprocessor {
    fn from(encoder: LabelEncoder) -> Self {
        Preprocessor::Label(encoder)
    }
}

impl From<OneHotEncoder> for Preprocessor {
    fn from(encoder: OneHotEncoder) -> Self {
        Preprocessor::OneHot(encoder)
    }
}

impl From<SimpleImputer> for Preprocessor {
    fn from(imputer: SimpleImputer) -> Self {
        Preprocessor::Imputer(imputer)
    }
}

impl From<YeoJohnsonTransform> for Preprocessor {
    fn from(transform: YeoJohnsonTransform) -> Self {
        Preprocessor::YeoJohnson(transform)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_standard_scaler() {
        let mut preprocessor = Preprocessor::Standard(StandardScaler::new());
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features

        preprocessor.fit_numerical(&data, 3).unwrap();
        assert!(preprocessor.is_fitted());
        assert!(preprocessor.is_numerical());
        assert!(!preprocessor.is_categorical());

        preprocessor.transform_numerical(&mut data, 3).unwrap();
    }

    #[test]
    fn test_preprocessor_frequency_encoder() {
        let mut preprocessor = Preprocessor::Frequency(FrequencyEncoder::new());
        let categories = vec!["A", "B", "A", "C"];

        preprocessor.fit_categorical(&categories).unwrap();
        assert!(preprocessor.is_fitted());
        assert!(preprocessor.is_categorical());
        assert!(!preprocessor.is_numerical());

        let encoded = preprocessor.transform_categorical(&categories).unwrap();
        assert_eq!(encoded, vec![2.0, 1.0, 2.0, 1.0]); // A=2, B=1, C=1
    }

    #[test]
    fn test_preprocessor_pipeline() {
        let mut pipeline = Preprocessor::Pipeline(Box::new(vec![
            Preprocessor::Standard(StandardScaler::new()),
            Preprocessor::MinMax(MinMaxScaler::new()),
        ]));

        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        pipeline.fit_numerical(&data, 3).unwrap();

        assert!(pipeline.is_fitted());
        assert!(pipeline.is_pipeline());

        pipeline.transform_numerical(&mut data, 3).unwrap();
    }

    #[test]
    fn test_preprocessor_serialization() {
        let mut preprocessor = Preprocessor::Standard(StandardScaler::new());
        let data = vec![1.0, 2.0, 3.0, 4.0];
        preprocessor.fit_numerical(&data, 2).unwrap();

        // Serialize
        let json = serde_json::to_string(&preprocessor).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let loaded: Preprocessor = serde_json::from_str(&json).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_standard_scaler()
            .with_minmax_scaler()
            .build();

        assert!(pipeline.is_pipeline());
    }

    #[test]
    fn test_from_conversions() {
        let p1: Preprocessor = StandardScaler::new().into();
        assert!(matches!(p1, Preprocessor::Standard(_)));

        let p2: Preprocessor = FrequencyEncoder::new().into();
        assert!(matches!(p2, Preprocessor::Frequency(_)));

        let p3: Preprocessor = OneHotEncoder::new().into();
        assert!(matches!(p3, Preprocessor::OneHot(_)));
    }

    #[test]
    fn test_numerical_on_categorical_error() {
        let mut preprocessor = Preprocessor::Frequency(FrequencyEncoder::new());
        let data = vec![1.0, 2.0, 3.0];

        let result = preprocessor.fit_numerical(&data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_categorical_on_numerical_error() {
        let mut preprocessor = Preprocessor::Standard(StandardScaler::new());
        let categories = vec!["A", "B"];

        let result = preprocessor.fit_categorical(&categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_custom() {
        let standard = Preprocessor::Standard(StandardScaler::new());
        assert!(!standard.is_custom());

        // Note: Testing Custom variant requires a registered type
        // See registry.rs tests for Custom variant tests
    }

    #[test]
    fn test_debug_impl() {
        let preprocessor = Preprocessor::Standard(StandardScaler::new());
        let debug_str = format!("{:?}", preprocessor);
        assert!(debug_str.contains("Standard"));
    }

    // =========================================================================
    // Custom Preprocessor Integration Tests
    // =========================================================================

    /// Test preprocessor for integration tests
    #[derive(Clone, Default)]
    struct TestCustomScaler {
        mean: f32,
        fitted: bool,
    }

    impl super::super::traits::PreprocessorTrait for TestCustomScaler {
        fn type_id(&self) -> &'static str {
            "test::TestCustomScaler"
        }

        fn supports_numerical(&self) -> bool {
            true
        }

        fn fit_numerical(&mut self, data: &[f32], _num_features: usize) -> crate::Result<()> {
            if data.is_empty() {
                return Err(crate::TreeBoostError::Config(
                    "DummyPreprocessor::fit_numerical() received empty data. Provide at least 1 data point."
                        .into(),
                ));
            }
            self.mean = data.iter().sum::<f32>() / data.len() as f32;
            self.fitted = true;
            Ok(())
        }

        fn transform_numerical(&self, data: &mut [f32], _num_features: usize) -> crate::Result<()> {
            if !self.fitted {
                return Err(crate::TreeBoostError::Config(
                    "DummyPreprocessor not fitted. Call fit_numerical() first to learn mean."
                        .into(),
                ));
            }
            for x in data.iter_mut() {
                *x -= self.mean;
            }
            Ok(())
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn serialize_state(&self) -> crate::Result<Vec<u8>> {
            Ok(format!("{}|{}", self.mean, self.fitted).into_bytes())
        }

        fn deserialize_state(&mut self, data: &[u8]) -> crate::Result<()> {
            let s = std::str::from_utf8(data).map_err(|e| {
                crate::TreeBoostError::Serialization(format!(
                    "DummyPreprocessor::deserialize_state() failed to decode UTF-8: {}",
                    e
                ))
            })?;
            let parts: Vec<&str> = s.split('|').collect();
            if parts.len() != 2 {
                return Err(crate::TreeBoostError::Serialization(format!(
                    "DummyPreprocessor::deserialize_state() invalid format: expected 'mean|fitted', got {} parts",
                    parts.len()
                )));
            }
            self.mean = parts[0].parse().map_err(|e: std::num::ParseFloatError| {
                crate::TreeBoostError::Serialization(format!(
                    "DummyPreprocessor::deserialize_state() failed to parse mean: {}",
                    e
                ))
            })?;
            self.fitted = parts[1] == "true";
            Ok(())
        }
    }

    #[test]
    fn test_custom_preprocessor_create_and_use() {
        // Register the test type
        super::super::registry::register_preprocessor::<TestCustomScaler>();

        // Create a custom preprocessor
        let mut scaler = TestCustomScaler::default();
        scaler.fit_numerical(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert!(scaler.is_fitted());

        // Wrap in Preprocessor::Custom
        let custom = Preprocessor::custom(scaler).unwrap();
        assert!(custom.is_custom());
        assert!(custom.is_fitted());
        assert!(custom.is_numerical());

        // Use for transformation
        let mut data = vec![5.0, 6.0];
        custom.transform_numerical(&mut data, 2).unwrap();
        // mean was 2.5, so 5.0 - 2.5 = 2.5, 6.0 - 2.5 = 3.5
        assert!((data[0] - 2.5).abs() < 0.001);
        assert!((data[1] - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_custom_preprocessor_serde_roundtrip() {
        // Register the test type
        super::super::registry::register_preprocessor::<TestCustomScaler>();

        // Create and fit a custom preprocessor
        let mut scaler = TestCustomScaler::default();
        scaler.fit_numerical(&[10.0, 20.0, 30.0, 40.0], 2).unwrap();
        let custom = Preprocessor::custom(scaler).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&custom).unwrap();
        assert!(json.contains("test::TestCustomScaler"));

        // Deserialize
        let mut loaded: Preprocessor = serde_json::from_str(&json).unwrap();

        // Instance is None after deserialize (due to #[serde(skip)])
        // Must call ensure_initialized() to reconstruct
        loaded.ensure_initialized().unwrap();

        // Now it should work
        assert!(loaded.is_fitted());
        assert!(loaded.is_custom());

        let mut data = vec![25.0, 35.0];
        loaded.transform_numerical(&mut data, 2).unwrap();
        // mean was 25.0, so 25.0 - 25.0 = 0.0, 35.0 - 25.0 = 10.0
        assert!((data[0] - 0.0).abs() < 0.001);
        assert!((data[1] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_custom_preprocessor_in_pipeline() {
        // Register the test type
        super::super::registry::register_preprocessor::<TestCustomScaler>();

        // Create a pipeline with custom and built-in preprocessors
        let pipeline = PipelineBuilder::new()
            .with_custom(TestCustomScaler::default())
            .unwrap()
            .with_standard_scaler()
            .build();

        assert!(pipeline.is_pipeline());

        // Fit and transform
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut pipeline = pipeline; // need mut for fit
        pipeline.fit_numerical(&data, 3).unwrap();
        assert!(pipeline.is_fitted());

        pipeline.transform_numerical(&mut data, 3).unwrap();
    }

    #[test]
    fn test_custom_preprocessor_uninitialized_error() {
        // Create a Custom variant directly without instance
        let custom = Preprocessor::Custom {
            type_id: "test::TestCustomScaler".to_string(),
            serialized_state: vec![],
            instance: None,
        };

        // Trying to use it without initialization should fail
        let mut data = vec![1.0, 2.0];
        let result = custom.transform_numerical(&mut data, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not initialized"));
    }
}
