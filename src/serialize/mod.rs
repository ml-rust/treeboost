//! Serialization module
//!
//! Provides model serialization in multiple formats:
//! - rkyv: Zero-copy deserialization, fastest loading (recommended for production)
//! - trb: Incremental format supporting model updates and crash recovery
//!
//! # Feature Flags
//!
//! - **`mmap`**: Enables `MmapTrbReader` for true zero-copy I/O. When enabled,
//!   TRB files can be memory-mapped for instant model loading with lazy page faults.
//!
//! # TRB Reader Comparison
//!
//! | Reader | Feature | I/O Model | Load Time | Memory |
//! |--------|---------|-----------|-----------|--------|
//! | [`TrbReader`] | default | read() to heap | O(model_size) | O(model_size) |
//! | `MmapTrbReader` | `mmap` | mmap (lazy) | O(1) | O(1) initial |
//!
//! # Example (with mmap feature)
//!
//! ```ignore
//! #[cfg(feature = "mmap")]
//! {
//!     use treeboost::serialize::MmapTrbReader;
//!
//!     // Instant load - no heap allocation
//!     let reader = MmapTrbReader::open("model.trb")?;
//!
//!     // Option 1: Zero-copy access to archived model
//!     let archived = reader.archived_model()?;
//!
//!     // Option 2: Deserialize (still faster than TrbReader due to mmap)
//!     let model = reader.load_model()?;
//! }
//! ```

pub mod rkyv_io;
mod trb;

pub use rkyv_io::{
    deserialize_universal_model, load_model, load_universal_model, save_model,
    save_universal_model, serialize_universal_model,
};
pub use trb::{
    open_for_append, TrbHeader, TrbReader, TrbSegment, TrbUpdateHeader, TrbWriter, UpdateType,
    FORMAT_VERSION, TRB_MAGIC,
};

// Re-export MmapTrbReader when mmap feature is enabled
#[cfg(feature = "mmap")]
pub use trb::MmapTrbReader;
