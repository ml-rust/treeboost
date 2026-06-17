//! Preprocessing pipeline for dirty data handling
//!
//! This module provides the full dirty data pipeline:
//!
//! 1. Count-Min Sketch filtering for rare categories → "unknown"
//! 2. Ordered Target Encoding with M-Estimate smoothing
//! 3. T-Digest quantile binning to u8
//!
//! Key types:
//!
//! - [`DataPipeline`] - Main pipeline orchestrator
//! - [`PipelineConfig`] - Configuration options
//! - [`PipelineState`] - Learned state for inference

// reason: renaming the inner module would be invasive; keep the conventional name.
#[allow(clippy::module_inception)]
mod pipeline;

pub use pipeline::{CategoricalEncodingState, DataPipeline, PipelineConfig, PipelineState};
