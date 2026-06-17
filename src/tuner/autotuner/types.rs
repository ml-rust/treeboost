//! Helper types for AutoTuner evaluation
//!
//! Contains shared type definitions used across evaluation modules.

use crate::dataset::BinnedDataset;
use crate::Result;
use polars::prelude::DataFrame;

use crate::tuner::realistic::RealisticModeConfig;

// =============================================================================
// Constants
// =============================================================================

/// Maximum consecutive zone switches before abandoning search
pub(super) const MAX_ZONE_SWITCH_FAILS: usize = 3;

/// Binary classification threshold for F1 score computation
pub(super) const BINARY_CLASSIFICATION_THRESHOLD: f32 = 0.5;

// =============================================================================
// Thread-safe pointer wrapper with panic safety
// =============================================================================

/// Thread-safe wrapper for raw pointer to enable Send/Sync
///
/// SAFETY: Pointer is read-only and valid during tune_with_validation call.
/// We accept the limitation that panics during evaluation may leave stale pointers,
/// but this is acceptable because:
/// 1. Panics are exceptional - normal code paths always clean up
/// 2. After a panic, the AutoTuner is in an undefined state anyway
/// 3. The pointers are only accessed through methods that check custom_validation first
pub(crate) struct SendPtr<T>(*const T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub(super) fn new(ptr: *const T) -> Self {
        SendPtr(ptr)
    }

    pub(super) unsafe fn as_ref<'a>(&self) -> &'a T {
        &*self.0
    }
}

// =============================================================================
// Evaluation Data Types
// =============================================================================

/// Input data for evaluation (unifies optimistic and realistic modes)
pub(super) enum EvalInput<'a> {
    /// Pre-binned dataset (optimistic mode - faster, may have target leakage)
    Optimistic(&'a BinnedDataset),
    /// Raw DataFrame with encoding config (realistic mode - no target leakage)
    Realistic {
        raw_data: &'a DataFrame,
        config: &'a RealisticModeConfig,
    },
}

/// Evaluation metrics for model performance assessment
///
/// Captures comprehensive evaluation metrics including primary validation/train metrics,
/// model complexity (num_trees), and task-specific metrics (F1, ROC-AUC, Rank IC).
#[derive(Debug, Clone)]
pub(super) struct EvalMetrics {
    /// Primary metric computed on validation set (e.g., MSE, LogLoss)
    pub(super) val_metric: f32,
    /// Primary metric computed on training set (for overfitting detection)
    pub(super) train_metric: f32,
    /// Number of trees/rounds in the trained model (complexity indicator)
    pub(super) num_trees: usize,
    /// F1 score for classification tasks (None for regression)
    pub(super) f1_score: Option<f32>,
    /// ROC-AUC for binary classification tasks (None for multi-class or regression)
    pub(super) roc_auc: Option<f64>,
    /// Rank Information Coefficient for regression with panel data (None otherwise)
    pub(super) rank_ic: Option<f64>,
}

/// Result of model evaluation
pub(super) type EvalResult = Result<EvalMetrics>;

// =============================================================================
// Validation Helpers
// =============================================================================

/// Check if the model supports conformal prediction
///
/// Returns an error if conformal evaluation is requested but not supported by the model.
pub(super) fn check_conformal_support<M: crate::tuner::traits::TunableModel>() -> Result<()> {
    if !M::supports_conformal() {
        return Err(crate::TreeBoostError::Config(
            "Conformal evaluation is not supported for this model type. \
             Use EvalStrategy::Holdout for generic model tuning."
                .to_string(),
        ));
    }
    Ok(())
}

/// Aggregate results from multiple folds by averaging metrics
///
/// Computes mean validation/train metrics, mean number of trees, and averages
/// optional metrics (F1, ROC-AUC, Rank IC) where available.
///
/// Returns worst-case metrics (f32::MAX) if no results provided.
pub(super) fn aggregate_fold_results(results: &[EvalMetrics]) -> EvalMetrics {
    let k = results.len();
    if k == 0 {
        return EvalMetrics {
            val_metric: f32::MAX,
            train_metric: f32::MAX,
            num_trees: 0,
            f1_score: None,
            roc_auc: None,
            rank_ic: None,
        };
    }

    let avg_val = results.iter().map(|r| r.val_metric).sum::<f32>() / k as f32;
    let avg_train = results.iter().map(|r| r.train_metric).sum::<f32>() / k as f32;
    let avg_trees = results.iter().map(|r| r.num_trees).sum::<usize>() / k;

    let f1_scores: Vec<f32> = results.iter().filter_map(|r| r.f1_score).collect();
    let avg_f1 = if f1_scores.is_empty() {
        None
    } else {
        Some(f1_scores.iter().sum::<f32>() / f1_scores.len() as f32)
    };

    let roc_aucs: Vec<f64> = results.iter().filter_map(|r| r.roc_auc).collect();
    let avg_roc_auc = if roc_aucs.is_empty() {
        None
    } else {
        Some(roc_aucs.iter().sum::<f64>() / roc_aucs.len() as f64)
    };

    let rank_ics: Vec<f64> = results.iter().filter_map(|r| r.rank_ic).collect();
    let avg_rank_ic = if rank_ics.is_empty() {
        None
    } else {
        Some(rank_ics.iter().sum::<f64>() / rank_ics.len() as f64)
    };

    EvalMetrics {
        val_metric: avg_val,
        train_metric: avg_train,
        num_trees: avg_trees,
        f1_score: avg_f1,
        roc_auc: avg_roc_auc,
        rank_ic: avg_rank_ic,
    }
}
