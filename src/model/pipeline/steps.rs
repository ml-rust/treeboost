//! Concrete implementations of pipeline steps
//!
//! Each step implements the PipelineStep trait, ensuring uniform interface:
//! - fit_transform: Learn state from training data
//! - transform: Apply learned state to inference data
//! - partial_fit: Update state with new data
//! - serialize/deserialize: Save/load learned state
//!
//! # Serialization
//!
//! All steps are serializable via both rkyv (binary) and serde (JSON).
//! The `PipelineStepKind` enum provides a unified serializable wrapper.
//!
//! # Custom Features
//!
//! Users can define custom features using the `CustomFeaturesStep`:
//! - `LinearFormula`: Weighted sum of columns with optional LUT mappings
//! - `TrigFeature`: Trigonometric transforms (sin/cos) with configurable period
//! - `LookupMapping`: Map categorical values to numeric using hand-crafted LUT
//!
//! Example:
//! ```ignore
//! AutoConfig::new()
//!     .with_custom_features(vec![
//!         CustomFeature::LinearFormula {
//!             name: "manual_formula".to_string(),
//!             terms: vec![
//!                 ("study_hours".to_string(), 6.0),
//!                 ("class_attendance".to_string(), 0.35),
//!             ],
//!             lut_terms: vec![
//!                 ("sleep_quality".to_string(), vec![
//!                     ("good".to_string(), 5.0),
//!                     ("average".to_string(), 0.0),
//!                     ("poor".to_string(), -5.0),
//!                 ]),
//!             ],
//!         },
//!         CustomFeature::Trig {
//!             source: "study_hours".to_string(),
//!             func: TrigFunc::Sin,
//!             period: 12.0,
//!             name: None, // Auto-generates "study_hours_sin"
//!         },
//!     ])
//! ```

use super::PipelineStep;
use crate::dataset::binning::QuantileBinner;
use crate::encoding::{CategoryFilter, CategoryMapping, EncodingMap, OrderedTargetEncoder};
use crate::features::{
    dataframe::{apply_interaction_features, apply_polynomial_features, apply_ratio_features},
    InteractionGenerator, PolynomialGenerator, RatioGenerator,
};
use crate::preprocessing::TargetTransformKind;
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// PipelineStepKind: Enum wrapper for all step types (rkyv-serializable)
// ============================================================================

/// Enum wrapping all pipeline step types for serialization.
///
/// This enables the Pipeline to be serialized with rkyv (zero-copy binary format).
/// Each variant contains the full step state, enabling save/load without loss.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub enum PipelineStepKind {
    /// Drop useless columns (e.g., id, constant columns) - MUST be first step
    DropColumns(DropColumnsStep),
    /// Custom user-defined features (expression builder)
    CustomFeatures(CustomFeaturesStep),
    /// Feature engineering (polynomials, interactions, ratios, trig, LUT)
    EngineerFeatures(EngineerFeaturesStep),
    /// Time-series feature engineering (lag, rolling, EWMA)
    EngineerTimeSeriesFeatures(EngineerTimeSeriesFeaturesStep),
    /// Categorical encoding (CMS filter + target encoding)
    EncodeCategoricals(EncodeCategoricalsState),
    /// Target transformation (logit, etc.)
    TransformTarget(TransformTargetStep),
    /// Numeric binning (quantile boundaries)
    BinNumericFeatures(BinNumericFeaturesState),
    /// Linear feature extraction (records indices for LinearThenTree)
    ExtractLinearFeatures(ExtractLinearFeaturesStep),
}

impl PipelineStepKind {
    /// Get the step name
    pub fn name(&self) -> &str {
        match self {
            Self::DropColumns(_) => "DropColumns",
            Self::CustomFeatures(_) => "CustomFeatures",
            Self::EngineerFeatures(_) => "EngineerFeatures",
            Self::EngineerTimeSeriesFeatures(_) => "EngineerTimeSeriesFeatures",
            Self::EncodeCategoricals(_) => "EncodeCategoricals",
            Self::TransformTarget(_) => "TransformTarget",
            Self::BinNumericFeatures(_) => "BinNumericFeatures",
            Self::ExtractLinearFeatures(_) => "ExtractLinearFeatures",
        }
    }

    /// Get target transform if this is a TransformTarget step
    pub fn get_target_transform(&self) -> Option<&TargetTransformKind> {
        match self {
            Self::TransformTarget(step) => Some(&step.transform),
            _ => None,
        }
    }

    /// Transform a DataFrame using this step (inference mode)
    pub fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        match self {
            Self::DropColumns(step) => step.transform(df),
            Self::CustomFeatures(step) => step.transform(df),
            Self::EngineerFeatures(step) => step.transform(df),
            Self::EngineerTimeSeriesFeatures(step) => step.transform(df),
            Self::EncodeCategoricals(state) => state.transform(df),
            Self::TransformTarget(step) => step.transform(df),
            Self::BinNumericFeatures(state) => state.transform(df),
            Self::ExtractLinearFeatures(step) => step.transform(df),
        }
    }

    /// Fit on training data and transform (returns updated step + transformed df)
    pub fn fit_transform(&mut self, df: DataFrame, targets: Option<&[f32]>) -> Result<DataFrame> {
        match self {
            Self::DropColumns(step) => step.transform(df), // No fitting needed, just drop
            Self::CustomFeatures(step) => step.fit_transform(df, targets),
            Self::EngineerFeatures(step) => step.fit_transform(df, targets),
            Self::EngineerTimeSeriesFeatures(step) => step.fit_transform(df, targets),
            Self::EncodeCategoricals(state) => state.fit_transform(df, targets),
            Self::TransformTarget(step) => step.fit_transform(df, targets),
            Self::BinNumericFeatures(state) => state.fit_transform(df, targets),
            Self::ExtractLinearFeatures(step) => step.fit_transform(df, targets),
        }
    }
}

/// Serializable state for EncodeCategoricalsStep
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct EncodeCategoricalsState {
    pub encodings: Vec<(String, CategoryEncoding)>,
}

impl EncodeCategoricalsState {
    pub fn new() -> Self {
        Self {
            encodings: Vec::new(),
        }
    }

    pub fn from_step(step: &EncodeCategoricalsStep) -> Self {
        Self {
            encodings: step
                .encodings
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    pub fn to_step(&self) -> EncodeCategoricalsStep {
        let mut step = EncodeCategoricalsStep::new();
        step.encodings = self.encodings.iter().cloned().collect();
        step
    }

    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        self.to_step().transform(df)
    }

    fn fit_transform(&mut self, df: DataFrame, targets: Option<&[f32]>) -> Result<DataFrame> {
        let mut step = self.to_step();
        let result = step.fit_transform(df, targets)?;
        *self = Self::from_step(&step);
        Ok(result)
    }
}

impl Default for EncodeCategoricalsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable state for BinNumericFeaturesStep
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct BinNumericFeaturesState {
    pub num_bins: usize,
    /// Bin boundaries for each feature (feature_name, boundaries)
    /// Skipped in JSON (too large), but serialized in rkyv for inference
    #[serde(skip)]
    pub boundaries: Vec<(String, Vec<f64>)>,
}

impl BinNumericFeaturesState {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            boundaries: Vec::new(),
        }
    }

    pub fn from_step(step: &BinNumericFeaturesStep) -> Self {
        Self {
            num_bins: step.num_bins,
            boundaries: step
                .boundaries
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    pub fn to_step(&self) -> BinNumericFeaturesStep {
        let mut step = BinNumericFeaturesStep::new(self.num_bins);
        step.boundaries = self.boundaries.iter().cloned().collect();
        step
    }

    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        self.to_step().transform(df)
    }

    fn fit_transform(&mut self, df: DataFrame, targets: Option<&[f32]>) -> Result<DataFrame> {
        let mut step = self.to_step();
        let result = step.fit_transform(df, targets)?;
        *self = Self::from_step(&step);
        Ok(result)
    }
}

// ============================================================================
// Step 0: Drop Columns (id, constants, etc.)
// ============================================================================

/// Drop columns step - removes useless columns (id, constants, etc.)
///
/// This MUST be the first step in the pipeline to ensure inference data
/// is cleaned the same way as training data.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct DropColumnsStep {
    /// Columns to drop (e.g., ["id", "constant_col"])
    pub columns: Vec<String>,
}

impl DropColumnsStep {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Transform: drop the specified columns
    pub fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        if self.columns.is_empty() {
            return Ok(df);
        }

        // Drop columns that exist in the DataFrame (ignore missing ones)
        let df_cols: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let existing_cols: Vec<String> = self
            .columns
            .iter()
            .filter(|c| df_cols.contains(c))
            .cloned()
            .collect();

        if existing_cols.is_empty() {
            return Ok(df);
        }

        Ok(df.drop_many(&existing_cols))
    }
}

// ============================================================================
// Step 1: Engineer Cross-Sectional Features (Polynomial, Interaction, Ratio)
// ============================================================================

/// Trigonometric function type for feature engineering
#[derive(
    Debug, Clone, Copy, PartialEq, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize,
)]
pub enum TrigFunc {
    Sin,
    Cos,
}

/// Trigonometric feature specification
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct TrigFeature {
    /// Source column name
    pub column: String,
    /// Trig function (sin or cos)
    pub func: TrigFunc,
    /// Period for the transform: sin(2π * x / period)
    pub period: f64,
    /// Optional custom name (defaults to "{column}_{func}")
    pub name: Option<String>,
}

impl TrigFeature {
    pub fn sin(column: impl Into<String>, period: f64) -> Self {
        Self {
            column: column.into(),
            func: TrigFunc::Sin,
            period,
            name: None,
        }
    }

    pub fn cos(column: impl Into<String>, period: f64) -> Self {
        Self {
            column: column.into(),
            func: TrigFunc::Cos,
            period,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the output column name
    pub fn output_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| {
            let func_name = match self.func {
                TrigFunc::Sin => "sin",
                TrigFunc::Cos => "cos",
            };
            format!("{}_{}", self.column, func_name)
        })
    }
}

/// Lookup table mapping for categorical columns
/// Maps categorical values to numeric values using a hand-crafted LUT
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct LutMapping {
    /// Source categorical column
    pub column: String,
    /// Mapping from category string to numeric value
    pub mapping: Vec<(String, f64)>,
    /// Default value for unknown categories (defaults to 0.0)
    pub default: f64,
    /// Optional custom output column name (defaults to "{column}_lut")
    pub name: Option<String>,
}

impl LutMapping {
    pub fn new(column: impl Into<String>, mapping: Vec<(impl Into<String>, f64)>) -> Self {
        Self {
            column: column.into(),
            mapping: mapping.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            default: 0.0,
            name: None,
        }
    }

    pub fn with_default(mut self, default: f64) -> Self {
        self.default = default;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the output column name
    pub fn output_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("{}_lut", self.column))
    }

    /// Get the numeric value for a category
    pub fn get(&self, category: &str) -> f64 {
        self.mapping
            .iter()
            .find(|(k, _)| k == category)
            .map(|(_, v)| *v)
            .unwrap_or(self.default)
    }
}

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct EngineerFeaturesStep {
    pub polynomial_features: Vec<String>,
    pub interaction_pairs: Vec<(String, String)>,
    pub ratio_pairs: Vec<(String, String)>,
    /// Trigonometric features (sin/cos with period)
    #[serde(default)]
    pub trig_features: Vec<TrigFeature>,
    /// Lookup table mappings for categoricals
    #[serde(default)]
    pub lut_mappings: Vec<LutMapping>,
}

impl EngineerFeaturesStep {
    pub fn new(
        polynomial_features: Vec<String>,
        interaction_pairs: Vec<(String, String)>,
        ratio_pairs: Vec<(String, String)>,
    ) -> Self {
        Self {
            polynomial_features,
            interaction_pairs,
            ratio_pairs,
            trig_features: Vec::new(),
            lut_mappings: Vec::new(),
        }
    }

    /// Add trigonometric features
    pub fn with_trig_features(mut self, trig_features: Vec<TrigFeature>) -> Self {
        self.trig_features = trig_features;
        self
    }

    /// Add LUT mappings for categoricals
    pub fn with_lut_mappings(mut self, lut_mappings: Vec<LutMapping>) -> Self {
        self.lut_mappings = lut_mappings;
        self
    }
}

impl PipelineStep for EngineerFeaturesStep {
    fn name(&self) -> &str {
        "EngineerFeatures"
    }

    fn fit_transform(&mut self, df: DataFrame, _targets: Option<&[f32]>) -> Result<DataFrame> {
        // This step is deterministic (no state to learn)
        self.transform(df)
    }

    fn transform(&self, mut df: DataFrame) -> Result<DataFrame> {
        // Apply LUT mappings for categoricals FIRST (before encoding removes them)
        for lut in &self.lut_mappings {
            if let Ok(series) = df.column(&lut.column) {
                // Convert to string and apply LUT
                let str_series = series.cast(&DataType::String)?;
                let str_chunked = str_series.str()?;

                let mapped: Vec<f64> = str_chunked
                    .iter()
                    .map(|opt| lut.get(opt.unwrap_or("")))
                    .collect();

                let new_series = Series::new(lut.output_name().as_str().into(), mapped);
                df = df.with_column(new_series.into())?.clone();
            }
        }

        // Apply trigonometric features
        for trig in &self.trig_features {
            if let Ok(series) = df.column(&trig.column) {
                let casted = series.cast(&DataType::Float64)?;
                let values = casted.f64()?;
                let two_pi = 2.0 * std::f64::consts::PI;

                let transformed: Vec<f64> = values
                    .iter()
                    .map(|opt| {
                        let x = opt.unwrap_or(0.0);
                        let angle = two_pi * x / trig.period;
                        match trig.func {
                            TrigFunc::Sin => angle.sin(),
                            TrigFunc::Cos => angle.cos(),
                        }
                    })
                    .collect();

                let new_series = Series::new(trig.output_name().as_str().into(), transformed);
                df = df.with_column(new_series.into())?.clone();
            }
        }

        // Apply polynomial features (x², √x, log(x+1))
        if !self.polynomial_features.is_empty() {
            let poly_gen = PolynomialGenerator::all();
            df = apply_polynomial_features(df, &poly_gen, Some(self.polynomial_features.clone()))?;
        }

        // Apply interaction features (x_i × x_j)
        if !self.interaction_pairs.is_empty() {
            // Convert named pairs to index pairs
            let numeric_cols: Vec<String> = df
                .columns()
                .iter()
                .filter(|col| col.dtype().is_numeric())
                .map(|col| col.name().to_string())
                .collect();

            let mut index_pairs = Vec::new();
            for (col_a, col_b) in &self.interaction_pairs {
                if let (Some(idx_a), Some(idx_b)) = (
                    numeric_cols.iter().position(|c| c == col_a),
                    numeric_cols.iter().position(|c| c == col_b),
                ) {
                    index_pairs.push((idx_a, idx_b));
                }
            }

            if !index_pairs.is_empty() {
                let interaction_gen = InteractionGenerator::from_pairs(index_pairs);
                df = apply_interaction_features(df, &interaction_gen, None)?;
            }
        }

        // Apply ratio features (x_i / x_j)
        if !self.ratio_pairs.is_empty() {
            // Convert named pairs to index pairs
            let numeric_cols: Vec<String> = df
                .columns()
                .iter()
                .filter(|col| col.dtype().is_numeric())
                .map(|col| col.name().to_string())
                .collect();

            let mut index_pairs = Vec::new();
            for (col_a, col_b) in &self.ratio_pairs {
                if let (Some(idx_a), Some(idx_b)) = (
                    numeric_cols.iter().position(|c| c == col_a),
                    numeric_cols.iter().position(|c| c == col_b),
                ) {
                    index_pairs.push((idx_a, idx_b));
                }
            }

            if !index_pairs.is_empty() {
                let ratio_gen = RatioGenerator::from_pairs(index_pairs);
                df = apply_ratio_features(df, &ratio_gen, None)?;
            }
        }

        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        // No state to update (deterministic transformation)
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to serialize EngineerFeaturesStep: {}",
                e
            ))
        })
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        let deserialized: Self = serde_json::from_value(state).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to deserialize EngineerFeaturesStep: {}",
                e
            ))
        })?;
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Custom Feature Expression Builder (Composable, rkyv-Serializable)
// ============================================================================

// Uses a flattened arena-style representation to avoid recursive Box<T> which
// rkyv cannot handle. Expressions are stored as a Vec of operations, with
// binary ops referencing operands by index.

/// Index into the expression arena
pub type OpIdx = u16;

/// A single operation in the flattened expression tree
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub enum FlatOp {
    /// Reference a column by name
    Col(String),
    /// Constant value
    Const(f64),
    /// Add two sub-expressions (by index)
    Add(OpIdx, OpIdx),
    /// Subtract: left - right
    Sub(OpIdx, OpIdx),
    /// Multiply two sub-expressions
    Mul(OpIdx, OpIdx),
    /// Divide: left / right
    Div(OpIdx, OpIdx),
    /// LUT mapping for categorical column
    Lut {
        column: String,
        mapping: Vec<(String, f64)>,
        default: f64,
    },
    /// Sin: sin(2π * col / period)
    Sin { column: String, period: f64 },
    /// Cos: cos(2π * col / period)
    Cos { column: String, period: f64 },
    /// Log1p: log(1 + col)
    Log1p(String),
    /// Square: col^2
    Square(String),
    /// Sqrt: sqrt(col)
    Sqrt(String),
    /// Power: col^power
    Pow(String, f64),
    /// Absolute value of sub-expression
    Abs(OpIdx),
    /// Clamp sub-expression to [min, max]
    Clamp { expr: OpIdx, min: f64, max: f64 },
}

/// Flattened expression tree stored as arena
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct FeatureExpr {
    /// All operations in the expression tree
    ops: Vec<FlatOp>,
    /// Index of the root operation
    root: OpIdx,
}

impl FeatureExpr {
    /// Create from a single leaf operation
    fn leaf(op: FlatOp) -> Self {
        Self {
            ops: vec![op],
            root: 0,
        }
    }

    /// Add an operation and return its index
    fn push(&mut self, op: FlatOp) -> OpIdx {
        let idx = self.ops.len() as OpIdx;
        self.ops.push(op);
        idx
    }

    /// Merge another expression into this one, returning the new root index
    fn merge(&mut self, other: Self) -> OpIdx {
        let offset = self.ops.len() as OpIdx;
        for op in other.ops {
            let remapped = match op {
                FlatOp::Add(l, r) => FlatOp::Add(l + offset, r + offset),
                FlatOp::Sub(l, r) => FlatOp::Sub(l + offset, r + offset),
                FlatOp::Mul(l, r) => FlatOp::Mul(l + offset, r + offset),
                FlatOp::Div(l, r) => FlatOp::Div(l + offset, r + offset),
                FlatOp::Abs(e) => FlatOp::Abs(e + offset),
                FlatOp::Clamp { expr, min, max } => FlatOp::Clamp {
                    expr: expr + offset,
                    min,
                    max,
                },
                other => other, // Leaf ops don't have indices
            };
            self.ops.push(remapped);
        }
        other.root + offset
    }

    /// Evaluate the expression on a DataFrame
    pub fn evaluate(&self, df: &DataFrame) -> Result<Vec<f64>> {
        // Evaluate all ops bottom-up, caching results
        let n = df.height();
        let mut cache: Vec<Option<Vec<f64>>> = vec![None; self.ops.len()];

        fn eval_op(
            idx: OpIdx,
            ops: &[FlatOp],
            cache: &mut [Option<Vec<f64>>],
            df: &DataFrame,
            n: usize,
        ) -> Result<Vec<f64>> {
            if let Some(ref cached) = cache[idx as usize] {
                return Ok(cached.clone());
            }

            let result = match &ops[idx as usize] {
                FlatOp::Col(name) => {
                    let series = df.column(name)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    values.iter().map(|v| v.unwrap_or(0.0)).collect()
                }
                FlatOp::Const(v) => vec![*v; n],
                FlatOp::Add(l, r) => {
                    let lv = eval_op(*l, ops, cache, df, n)?;
                    let rv = eval_op(*r, ops, cache, df, n)?;
                    lv.iter().zip(rv.iter()).map(|(a, b)| a + b).collect()
                }
                FlatOp::Sub(l, r) => {
                    let lv = eval_op(*l, ops, cache, df, n)?;
                    let rv = eval_op(*r, ops, cache, df, n)?;
                    lv.iter().zip(rv.iter()).map(|(a, b)| a - b).collect()
                }
                FlatOp::Mul(l, r) => {
                    let lv = eval_op(*l, ops, cache, df, n)?;
                    let rv = eval_op(*r, ops, cache, df, n)?;
                    lv.iter().zip(rv.iter()).map(|(a, b)| a * b).collect()
                }
                FlatOp::Div(l, r) => {
                    let lv = eval_op(*l, ops, cache, df, n)?;
                    let rv = eval_op(*r, ops, cache, df, n)?;
                    lv.iter()
                        .zip(rv.iter())
                        .map(|(a, b)| if b.abs() < 1e-10 { 0.0 } else { a / b })
                        .collect()
                }
                FlatOp::Lut {
                    column,
                    mapping,
                    default,
                } => {
                    let series = df.column(column)?;
                    let str_series = series.cast(&DataType::String)?;
                    let str_chunked = str_series.str()?;
                    str_chunked
                        .iter()
                        .map(|opt| {
                            let cat = opt.unwrap_or("");
                            mapping
                                .iter()
                                .find(|(k, _)| k == cat)
                                .map(|(_, v)| *v)
                                .unwrap_or(*default)
                        })
                        .collect()
                }
                FlatOp::Sin { column, period } => {
                    let series = df.column(column)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    let two_pi = 2.0 * std::f64::consts::PI;
                    values
                        .iter()
                        .map(|v| (two_pi * v.unwrap_or(0.0) / period).sin())
                        .collect()
                }
                FlatOp::Cos { column, period } => {
                    let series = df.column(column)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    let two_pi = 2.0 * std::f64::consts::PI;
                    values
                        .iter()
                        .map(|v| (two_pi * v.unwrap_or(0.0) / period).cos())
                        .collect()
                }
                FlatOp::Log1p(column) => {
                    let series = df.column(column)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    values
                        .iter()
                        .map(|v| (1.0 + v.unwrap_or(0.0)).ln())
                        .collect()
                }
                FlatOp::Square(column) => {
                    let series = df.column(column)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    values
                        .iter()
                        .map(|v| {
                            let x = v.unwrap_or(0.0);
                            x * x
                        })
                        .collect()
                }
                FlatOp::Sqrt(column) => {
                    let series = df.column(column)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    values
                        .iter()
                        .map(|v| v.unwrap_or(0.0).max(0.0).sqrt())
                        .collect()
                }
                FlatOp::Pow(column, power) => {
                    let series = df.column(column)?;
                    let casted = series.cast(&DataType::Float64)?;
                    let values = casted.f64()?;
                    values
                        .iter()
                        .map(|v| v.unwrap_or(0.0).powf(*power))
                        .collect()
                }
                FlatOp::Abs(e) => {
                    let ev = eval_op(*e, ops, cache, df, n)?;
                    ev.into_iter().map(|v| v.abs()).collect()
                }
                FlatOp::Clamp { expr, min, max } => {
                    let ev = eval_op(*expr, ops, cache, df, n)?;
                    ev.into_iter().map(|v| v.clamp(*min, *max)).collect()
                }
            };

            cache[idx as usize] = Some(result.clone());
            Ok(result)
        }

        eval_op(self.root, &self.ops, &mut cache, df, n)
    }
}

/// Builder for creating feature expressions with ergonomic API
///
/// Provides a fluent interface that internally builds a flattened expression tree.
#[derive(Debug, Clone)]
pub struct FeatureOp {
    expr: FeatureExpr,
}

impl FeatureOp {
    /// Create a column reference
    pub fn col(name: impl Into<String>) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Col(name.into())),
        }
    }

    /// Create a constant
    pub fn constant(value: f64) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Const(value)),
        }
    }

    /// Create a LUT mapping
    pub fn lut(column: impl Into<String>, mapping: &[(&str, f64)]) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Lut {
                column: column.into(),
                mapping: mapping.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                default: 0.0,
            }),
        }
    }

    /// Create a LUT with custom default
    pub fn lut_with_default(
        column: impl Into<String>,
        mapping: &[(&str, f64)],
        default: f64,
    ) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Lut {
                column: column.into(),
                mapping: mapping.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                default,
            }),
        }
    }

    /// Sin: sin(2π * col / period)
    pub fn sin(column: impl Into<String>, period: f64) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Sin {
                column: column.into(),
                period,
            }),
        }
    }

    /// Cos: cos(2π * col / period)
    pub fn cos(column: impl Into<String>, period: f64) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Cos {
                column: column.into(),
                period,
            }),
        }
    }

    /// Log1p: log(1 + col)
    pub fn log1p(column: impl Into<String>) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Log1p(column.into())),
        }
    }

    /// Square: col^2
    pub fn square(column: impl Into<String>) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Square(column.into())),
        }
    }

    /// Sqrt: sqrt(col)
    pub fn sqrt(column: impl Into<String>) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Sqrt(column.into())),
        }
    }

    /// Power: col^power
    pub fn pow(column: impl Into<String>, power: f64) -> Self {
        Self {
            expr: FeatureExpr::leaf(FlatOp::Pow(column.into(), power)),
        }
    }

    /// Multiply column by constant (convenience)
    pub fn scale(column: impl Into<String>, factor: f64) -> Self {
        let mut expr = FeatureExpr::leaf(FlatOp::Col(column.into()));
        let const_idx = expr.push(FlatOp::Const(factor));
        let root = expr.push(FlatOp::Mul(0, const_idx));
        expr.root = root;
        Self { expr }
    }

    /// Add two expressions
    #[allow(clippy::should_implement_trait)] // reason: intentional inherent API, not the std trait
    pub fn add(mut self, other: Self) -> Self {
        let other_root = self.expr.merge(other.expr);
        let new_root = self.expr.push(FlatOp::Add(self.expr.root, other_root));
        self.expr.root = new_root;
        self
    }

    /// Subtract: self - other
    #[allow(clippy::should_implement_trait)] // reason: intentional inherent API, not the std trait
    pub fn sub(mut self, other: Self) -> Self {
        let other_root = self.expr.merge(other.expr);
        let new_root = self.expr.push(FlatOp::Sub(self.expr.root, other_root));
        self.expr.root = new_root;
        self
    }

    /// Multiply two expressions
    #[allow(clippy::should_implement_trait)] // reason: intentional inherent API, not the std trait
    pub fn mul(mut self, other: Self) -> Self {
        let other_root = self.expr.merge(other.expr);
        let new_root = self.expr.push(FlatOp::Mul(self.expr.root, other_root));
        self.expr.root = new_root;
        self
    }

    /// Divide: self / other
    #[allow(clippy::should_implement_trait)] // reason: intentional inherent API, not the std trait
    pub fn div(mut self, other: Self) -> Self {
        let other_root = self.expr.merge(other.expr);
        let new_root = self.expr.push(FlatOp::Div(self.expr.root, other_root));
        self.expr.root = new_root;
        self
    }

    /// Absolute value
    pub fn abs(mut self) -> Self {
        let new_root = self.expr.push(FlatOp::Abs(self.expr.root));
        self.expr.root = new_root;
        self
    }

    /// Clamp to [min, max]
    pub fn clamp(mut self, min: f64, max: f64) -> Self {
        let new_root = self.expr.push(FlatOp::Clamp {
            expr: self.expr.root,
            min,
            max,
        });
        self.expr.root = new_root;
        self
    }

    /// Build the final expression
    pub fn build(self) -> FeatureExpr {
        self.expr
    }

    /// Evaluate directly (for convenience)
    pub fn evaluate(&self, df: &DataFrame) -> Result<Vec<f64>> {
        self.expr.evaluate(df)
    }
}

/// A custom feature definition with name and expression
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct CustomFeature {
    /// Output column name
    pub name: String,
    /// Flattened expression tree
    pub expr: FeatureExpr,
}

impl CustomFeature {
    /// Create a new custom feature from a FeatureOp builder
    pub fn new(name: impl Into<String>, op: FeatureOp) -> Self {
        Self {
            name: name.into(),
            expr: op.build(),
        }
    }

    /// Create from a pre-built expression
    pub fn from_expr(name: impl Into<String>, expr: FeatureExpr) -> Self {
        Self {
            name: name.into(),
            expr,
        }
    }

    /// Evaluate this feature on a DataFrame, returning the computed values
    pub fn evaluate(&self, df: &DataFrame) -> Result<Vec<f64>> {
        self.expr.evaluate(df)
    }

    /// Create a manual formula feature (weighted sum with optional LUT terms)
    ///
    /// Example (replicating the winning notebook's formula):
    /// ```ignore
    /// CustomFeature::formula("manual_formula")
    ///     .add_term("study_hours", 6.0)
    ///     .add_term("class_attendance", 0.35)
    ///     .add_term("sleep_hours", 1.5)
    ///     .add_lut("sleep_quality", &[("good", 5.0), ("average", 0.0), ("poor", -5.0)])
    ///     .add_lut("study_method", &[("coaching", 10.0), ("mixed", 5.0), ...])
    ///     .build()
    /// ```
    pub fn formula(name: impl Into<String>) -> FormulaBuilder {
        FormulaBuilder::new(name)
    }
}

/// Builder for creating linear formula features
pub struct FormulaBuilder {
    name: String,
    terms: Vec<FeatureOp>,
}

impl FormulaBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            terms: Vec::new(),
        }
    }

    /// Add a weighted column term: weight * column
    pub fn add_term(mut self, column: impl Into<String>, weight: f64) -> Self {
        self.terms.push(FeatureOp::scale(column, weight));
        self
    }

    /// Add a constant (intercept) term
    pub fn add_const(mut self, value: f64) -> Self {
        self.terms.push(FeatureOp::constant(value));
        self
    }

    /// Add a LUT term for a categorical column
    pub fn add_lut(mut self, column: impl Into<String>, mapping: &[(&str, f64)]) -> Self {
        self.terms.push(FeatureOp::lut(column, mapping));
        self
    }

    /// Add a weighted LUT term: weight * lut(column)
    pub fn add_weighted_lut(
        mut self,
        column: impl Into<String>,
        mapping: &[(&str, f64)],
        default: f64,
        weight: f64,
    ) -> Self {
        let lut_op = FeatureOp::lut_with_default(column, mapping, default);
        self.terms.push(lut_op.mul(FeatureOp::constant(weight)));
        self
    }

    /// Add a raw expression term
    pub fn add_expr(mut self, expr: FeatureOp) -> Self {
        self.terms.push(expr);
        self
    }

    /// Build the final CustomFeature
    pub fn build(self) -> CustomFeature {
        let expr = if self.terms.is_empty() {
            FeatureOp::constant(0.0)
        } else {
            let mut iter = self.terms.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, term| acc.add(term))
        };

        CustomFeature {
            name: self.name,
            expr: expr.build(),
        }
    }
}

/// Custom features step - applies user-defined feature expressions
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct CustomFeaturesStep {
    pub features: Vec<CustomFeature>,
}

impl CustomFeaturesStep {
    pub fn new(features: Vec<CustomFeature>) -> Self {
        Self { features }
    }
}

impl PipelineStep for CustomFeaturesStep {
    fn name(&self) -> &str {
        "CustomFeatures"
    }

    fn fit_transform(&mut self, df: DataFrame, _targets: Option<&[f32]>) -> Result<DataFrame> {
        // Custom features are deterministic (no state to learn)
        self.transform(df)
    }

    fn transform(&self, mut df: DataFrame) -> Result<DataFrame> {
        for feature in &self.features {
            let values = feature.expr.evaluate(&df)?;
            let series = Series::new(feature.name.as_str().into(), values);
            df = df.with_column(series.into())?.clone();
        }
        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize CustomFeaturesStep: {}", e))
        })
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        let deserialized: Self = serde_json::from_value(state).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to deserialize CustomFeaturesStep: {}",
                e
            ))
        })?;
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 2: Engineer Time-Series Features (Lag, Rolling, EWMA)
// ============================================================================

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct EngineerTimeSeriesFeaturesStep {
    pub group_column: String,
    pub date_column: String,
    pub lag_periods: Vec<usize>,
    pub rolling_windows: Vec<usize>,
    pub ewma_alphas: Vec<f64>,
}

impl EngineerTimeSeriesFeaturesStep {
    pub fn new(
        group_column: String,
        date_column: String,
        lag_periods: Vec<usize>,
        rolling_windows: Vec<usize>,
        ewma_alphas: Vec<f64>,
    ) -> Self {
        Self {
            group_column,
            date_column,
            lag_periods,
            rolling_windows,
            ewma_alphas,
        }
    }
}

impl PipelineStep for EngineerTimeSeriesFeaturesStep {
    fn name(&self) -> &str {
        "EngineerTimeSeriesFeatures"
    }

    fn fit_transform(&mut self, df: DataFrame, _targets: Option<&[f32]>) -> Result<DataFrame> {
        // This step is deterministic (no state to learn)
        self.transform(df)
    }

    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // TODO: Implement time-series feature generation using apply_timeseries_features
        // For now, return unchanged (will implement when panel data detection is added)
        // This requires:
        // 1. PanelDataInfo (group_column, date_column)
        // 2. TimeSeriesFeaturePlan (lag_periods, rolling_windows, ewma_alphas)
        // 3. apply_timeseries_features from features/dataframe.rs
        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        // No state to update (deterministic transformation)
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to serialize EngineerTimeSeriesFeaturesStep: {}",
                e
            ))
        })
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        let deserialized: Self = serde_json::from_value(state).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to deserialize EngineerTimeSeriesFeaturesStep: {}",
                e
            ))
        })?;
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 3: Encode Categorical Columns (with learned state)
// ============================================================================

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct CategoryEncoding {
    pub method: String, // "frequency", "target", "onehot"
    pub category_mapping: CategoryMapping,
    pub encoding_map: EncodingMap,
    /// Bin boundaries for the encoded values
    /// Skipped in JSON (too large), but serialized in rkyv for inference
    #[serde(skip)]
    pub bin_boundaries: Vec<f64>,
}

#[derive(Clone)]
pub struct EncodeCategoricalsStep {
    pub encodings: HashMap<String, CategoryEncoding>,
}

impl Default for EncodeCategoricalsStep {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeCategoricalsStep {
    pub fn new() -> Self {
        Self {
            encodings: HashMap::new(),
        }
    }

    /// Convert series to string vec
    fn series_to_strings(series: &Series) -> Result<Vec<String>> {
        // Cast to String handles both String and Categorical types
        let str_series = series.cast(&DataType::String)?;
        let str_chunked = str_series.str()?;
        Ok(str_chunked
            .iter()
            .map(|opt| opt.unwrap_or("").to_string())
            .collect())
    }
}

impl PipelineStep for EncodeCategoricalsStep {
    fn name(&self) -> &str {
        "EncodeCategoricals"
    }

    fn fit_transform(&mut self, mut df: DataFrame, targets: Option<&[f32]>) -> Result<DataFrame> {
        let targets = targets.ok_or_else(|| {
            TreeBoostError::Data("EncodeCategoricalsStep requires targets for training".to_string())
        })?;

        // Convert f32 targets to f64 for target encoding
        let targets_f64: Vec<f64> = targets.iter().map(|&t| t as f64).collect();

        // Identify categorical columns
        let categorical_columns: Vec<String> = df
            .columns()
            .iter()
            .filter(|col| matches!(col.dtype(), DataType::String | DataType::Categorical(_, _)))
            .map(|col| col.name().to_string())
            .collect();

        // Learn encodings for each categorical column
        for col_name in &categorical_columns {
            let series = df.column(col_name)?;
            let categories = Self::series_to_strings(series.as_materialized_series())?;

            // Step 1: CMS filter for rare categories
            let mut filter = CategoryFilter::new(0.001, 0.99, 5);
            for cat in &categories {
                filter.count(cat);
            }

            let unique: Vec<String> = categories
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            filter.finalize(unique);

            let filtered: Vec<String> = categories
                .iter()
                .map(|c| filter.filter(c).to_string())
                .collect();

            let category_mapping = CategoryMapping::from_filter(&filter);

            // Step 2: Ordered Target Encoding
            let mut encoder = OrderedTargetEncoder::new(10.0); // Smoothing = 10.0
            let encoded = encoder.encode_column(&filtered, &targets_f64);
            let encoding_map = encoder.get_encoding_map();

            // Store encoding for this column
            self.encodings.insert(
                col_name.clone(),
                CategoryEncoding {
                    method: "target".to_string(),
                    category_mapping,
                    encoding_map,
                    bin_boundaries: Vec::new(), // Not needed for Float64 output
                },
            );

            // Replace categorical column with encoded Float64 column
            let encoded_series = Series::new(col_name.as_str().into(), encoded);
            df.replace(col_name.as_str(), encoded_series.into())?;
        }

        Ok(df)
    }

    fn transform(&self, mut df: DataFrame) -> Result<DataFrame> {
        // Apply learned encodings to each categorical column
        for (col_name, encoding) in &self.encodings {
            if let Ok(series) = df.column(col_name) {
                let categories = Self::series_to_strings(series.as_materialized_series())?;

                // Map categories to encoded values
                let encoded: Vec<f64> = categories
                    .iter()
                    .map(|cat| {
                        let idx = encoding.category_mapping.get_index(cat);
                        if idx == encoding.category_mapping.unknown_idx {
                            encoding.encoding_map.default_value
                        } else {
                            encoding.encoding_map.encode(cat)
                        }
                    })
                    .collect();

                // Replace with encoded Float64 column
                let encoded_series = Series::new(col_name.as_str().into(), encoded);
                df.replace(col_name.as_str(), encoded_series.into())?;
            }
        }

        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        // TODO: Update encodings with new categories
        // For now, this is a no-op (would require updating OrderedTargetEncoder incrementally)
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        // Serialize the HashMap of encodings
        let mut encodings_json = serde_json::Map::new();
        for (col_name, encoding) in &self.encodings {
            encodings_json.insert(
                col_name.clone(),
                serde_json::to_value(encoding).map_err(|e| {
                    TreeBoostError::Serialization(format!(
                        "Failed to serialize encoding for {}: {}",
                        col_name, e
                    ))
                })?,
            );
        }
        Ok(serde_json::Value::Object(encodings_json))
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        if let serde_json::Value::Object(map) = state {
            self.encodings.clear();
            for (col_name, encoding_json) in map {
                let encoding: CategoryEncoding =
                    serde_json::from_value(encoding_json).map_err(|e| {
                        TreeBoostError::Serialization(format!(
                            "Failed to deserialize encoding for {}: {}",
                            col_name, e
                        ))
                    })?;
                self.encodings.insert(col_name, encoding);
            }
            Ok(())
        } else {
            Err(TreeBoostError::Serialization(
                "Expected JSON object for encodings".to_string(),
            ))
        }
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 4: Transform Target (training only, inverse on prediction)
// ============================================================================

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct TransformTargetStep {
    pub transform: TargetTransformKind,
}

impl TransformTargetStep {
    pub fn new(transform: TargetTransformKind) -> Self {
        Self { transform }
    }
}

impl PipelineStep for TransformTargetStep {
    fn name(&self) -> &str {
        "TransformTarget"
    }

    fn get_target_transform(&self) -> Option<&crate::preprocessing::TargetTransformKind> {
        Some(&self.transform)
    }

    fn fit_transform(&mut self, df: DataFrame, _targets: Option<&[f32]>) -> Result<DataFrame> {
        // Target transform happens outside the DataFrame (on targets vector)
        // This step just records the transform for later inverse
        Ok(df)
    }

    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // Skip during inference (no target column)
        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        // No state to update (transform is fixed)
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(&self.transform).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize TransformTargetStep: {}", e))
        })
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        self.transform = serde_json::from_value(state).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to deserialize TransformTargetStep: {}",
                e
            ))
        })?;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 5: Bin Numeric Features (with learned quantile boundaries)
// ============================================================================

#[derive(Clone)]
pub struct BinNumericFeaturesStep {
    pub num_bins: usize,
    pub boundaries: HashMap<String, Vec<f64>>,
}

impl BinNumericFeaturesStep {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            boundaries: HashMap::new(),
        }
    }
}

impl BinNumericFeaturesStep {
    /// Convert series to f64 vec
    fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
        series
            .cast(&DataType::Float64)?
            .f64()?
            .iter()
            .map(|opt| Ok(opt.unwrap_or(f64::NAN)))
            .collect()
    }
}

impl PipelineStep for BinNumericFeaturesStep {
    fn name(&self) -> &str {
        "BinNumericFeatures"
    }

    fn fit_transform(&mut self, df: DataFrame, _targets: Option<&[f32]>) -> Result<DataFrame> {
        // Learn quantile boundaries for all numeric columns
        // Note: We DO NOT bin the DataFrame here - it stays as Float64
        // Actual binning to u8 happens later in DataPipeline
        let binner = QuantileBinner::new(self.num_bins);

        for col in df.columns() {
            if col.dtype().is_numeric() {
                let col_name = col.name().to_string();
                let values = Self::series_to_f64(col.as_materialized_series())?;

                // Compute boundaries using T-Digest
                let boundaries = binner.compute_boundaries(&values);
                self.boundaries.insert(col_name, boundaries);
            }
        }

        // Return DataFrame unchanged (no binning applied)
        Ok(df)
    }

    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // Inference: Do NOT bin DataFrame
        // Boundaries are used later by DataPipeline for binning to u8
        // Here we just pass through the Float64 DataFrame
        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        // TODO: Update boundaries with new quantiles (incremental learning)
        // For now, this requires re-computing boundaries with combined data
        // which is complex for T-Digest - implement when needed
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_bins": self.num_bins,
            "boundaries": self.boundaries,
        }))
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        self.num_bins = state["num_bins"].as_u64().unwrap_or(255) as usize;
        if let Some(boundaries) = state["boundaries"].as_object() {
            self.boundaries.clear();
            for (col_name, boundaries_json) in boundaries {
                if let Some(arr) = boundaries_json.as_array() {
                    let bounds: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    self.boundaries.insert(col_name.clone(), bounds);
                }
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Step 6: Extract Linear Features (records indices for LinearThenTree)
// ============================================================================

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct ExtractLinearFeaturesStep {
    pub linear_feature_indices: Vec<usize>,
    pub all_feature_names: Vec<String>,
}

impl ExtractLinearFeaturesStep {
    pub fn new(linear_feature_indices: Vec<usize>) -> Self {
        Self {
            linear_feature_indices,
            all_feature_names: Vec::new(),
        }
    }
}

impl PipelineStep for ExtractLinearFeaturesStep {
    fn name(&self) -> &str {
        "ExtractLinearFeatures"
    }

    fn fit_transform(&mut self, df: DataFrame, _targets: Option<&[f32]>) -> Result<DataFrame> {
        // Record all feature names for debugging
        self.all_feature_names = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // No transformation needed (just records indices)
        Ok(df)
    }

    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // No transformation needed
        Ok(df)
    }

    fn partial_fit(&mut self, _df: DataFrame, _targets: Option<&[f32]>) -> Result<()> {
        // No state to update
        Ok(())
    }

    fn serialize_state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to serialize ExtractLinearFeaturesStep: {}",
                e
            ))
        })
    }

    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()> {
        let deserialized: Self = serde_json::from_value(state).map_err(|e| {
            TreeBoostError::Serialization(format!(
                "Failed to deserialize ExtractLinearFeaturesStep: {}",
                e
            ))
        })?;
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}
