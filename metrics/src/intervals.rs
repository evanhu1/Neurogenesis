//! Derive [`IntervalMetrics`] (the per-reporting-interval timeseries rows
//! consumed by reports and the live CLI) from raw fact rows.
//!
//! Every behavioral metric is derived from action-time
//! [`BehaviorIntervalRow`]s. Intervals are closed-open windows
//! `(start_tick, end_tick]`; the last may be partial. Lifetime rows are kept
//! separately for lifecycle/cohort analysis and never stand in for tail
//! behavior.
//!
//! This is the pure, storage-agnostic computation: callers pass row slices
//! (the eval harness from a loaded Parquet dataset, the CLI from its live
//! in-memory ledger output).

use crate::schema::BehaviorIntervalRow;
use serde::Serialize;

/// One row per reporting interval — the primary timeseries format consumed by
/// reports, comparisons, and the CSV export. Every rate uses actions taken
/// during that interval as the denominator, including actions from organisms
/// still alive at its boundary.
#[derive(Debug, Clone, Serialize)]
pub struct IntervalMetrics {
    /// Exclusive start boundary of the raw action-time interval.
    pub start_tick: u64,
    pub tick: u64,
    /// Living population (viability context, not a competence score).
    pub pop: u32,
    /// Successful contingent actions / total actions. Idling and spinning
    /// self-penalise because they inflate the denominator without succeeding.
    pub action_effectiveness: Option<f64>,
    /// Successful attack-energy transfers / total actions.
    pub successful_attack_rate: Option<f64>,
    /// Action-time success-vs-age slope within this interval. This remains an
    /// observational learning diagnostic, not a causal plasticity assay.
    pub learning_slope: Option<f64>,
}

pub fn derive_interval_metrics(rows: &[BehaviorIntervalRow]) -> Vec<IntervalMetrics> {
    rows.iter().map(IntervalAccumulator::from_row).collect()
}

struct IntervalAccumulator {
    start_tick: u64,
    end_tick: u64,
    pop: u32,
    total_actions: u64,
    contingent_actions: u64,
    failed_actions: u64,
    successful_attacks: u64,
    learning_samples: u64,
    learning_within_numerator: f64,
    learning_within_denominator: f64,
}

impl IntervalAccumulator {
    fn from_row(row: &BehaviorIntervalRow) -> IntervalMetrics {
        Self {
            start_tick: row.start_tick,
            end_tick: row.end_tick,
            pop: row.population,
            total_actions: row.total_actions,
            contingent_actions: row.contingent_actions,
            failed_actions: row.failed_actions,
            successful_attacks: row.successful_attacks,
            learning_samples: row.learning_samples,
            learning_within_numerator: row.learning_within_numerator,
            learning_within_denominator: row.learning_within_denominator,
        }
        .finalize()
    }

    fn finalize(self) -> IntervalMetrics {
        let total = self.total_actions;
        let rate = |num: u64| (total > 0).then(|| num as f64 / total as f64);

        let action_effectiveness =
            rate(self.contingent_actions.saturating_sub(self.failed_actions));
        let successful_attack_rate = rate(self.successful_attacks);
        let learning_slope = action_time_learning_slope(
            self.learning_samples,
            self.learning_within_numerator,
            self.learning_within_denominator,
        );

        IntervalMetrics {
            start_tick: self.start_tick,
            tick: self.end_tick,
            pop: self.pop,
            action_effectiveness,
            successful_attack_rate,
            learning_slope,
        }
    }
}

fn action_time_learning_slope(
    samples: u64,
    within_numerator: f64,
    within_denominator: f64,
) -> Option<f64> {
    if samples < 30 || within_denominator <= 0.0 {
        return None;
    }
    Some(within_numerator / within_denominator)
}
