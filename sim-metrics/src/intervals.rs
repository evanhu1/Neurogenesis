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

use crate::schema::{BehaviorIntervalRow, ACTION_COUNT, JOINT_LEN, SENSORY_BIN_COUNT};
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
    /// Plant consumptions / total actions — foraging "consume success".
    pub plant_consumption_rate: Option<f64>,
    /// Prey/corpse consumptions / total actions — predation competence.
    pub prey_consumption_rate: Option<f64>,
    /// Miller-Madow MI between food-visibility context and selected action.
    pub mi_sa: Option<f64>,
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
    plant_consumptions: u64,
    prey_consumptions: u64,
    pooled_joint: [u64; JOINT_LEN],
    learning_samples: u64,
    learning_within_numerator: f64,
    learning_within_denominator: f64,
}

impl IntervalAccumulator {
    fn from_row(row: &BehaviorIntervalRow) -> IntervalMetrics {
        let mut pooled_joint = [0; JOINT_LEN];
        pool_joint(&mut pooled_joint, &row.joint_sensory_action);
        Self {
            start_tick: row.start_tick,
            end_tick: row.end_tick,
            pop: row.population,
            total_actions: row.total_actions,
            contingent_actions: row.contingent_actions,
            failed_actions: row.failed_actions,
            plant_consumptions: row.plant_consumptions,
            prey_consumptions: row.prey_consumptions,
            pooled_joint,
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
        let plant_consumption_rate = rate(self.plant_consumptions);
        let prey_consumption_rate = rate(self.prey_consumptions);

        let mi_sa = mi_from_joint(&self.pooled_joint);
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
            plant_consumption_rate,
            prey_consumption_rate,
            mi_sa,
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

fn pool_joint(into: &mut [u64; JOINT_LEN], from: &[u64]) {
    for (idx, value) in from.iter().take(JOINT_LEN).enumerate() {
        into[idx] = into[idx].saturating_add(*value);
    }
}

/// Miller-Madow-corrected mutual information I(S;A) from a pooled joint
/// histogram. Returns `None` when the joint has no observations.
fn mi_from_joint(joint: &[u64; JOINT_LEN]) -> Option<f64> {
    let total_obs: u64 = joint.iter().sum();
    if total_obs == 0 {
        return None;
    }

    let mut p_s = [0_u64; SENSORY_BIN_COUNT];
    let mut p_a = [0_u64; ACTION_COUNT];
    let mut nonzero_cells: u64 = 0;
    for sensory_idx in 0..SENSORY_BIN_COUNT {
        for action_idx in 0..ACTION_COUNT {
            let count = joint[sensory_idx * ACTION_COUNT + action_idx];
            p_s[sensory_idx] = p_s[sensory_idx].saturating_add(count);
            p_a[action_idx] = p_a[action_idx].saturating_add(count);
            if count > 0 {
                nonzero_cells = nonzero_cells.saturating_add(1);
            }
        }
    }
    let nonzero_s = p_s.iter().filter(|&&count| count > 0).count();
    let nonzero_a = p_a.iter().filter(|&&count| count > 0).count();

    let n = total_obs as f64;
    let mut mi = 0.0;
    for sensory_idx in 0..SENSORY_BIN_COUNT {
        for action_idx in 0..ACTION_COUNT {
            let joint_count = joint[sensory_idx * ACTION_COUNT + action_idx];
            if joint_count == 0 {
                continue;
            }
            let p_sa = joint_count as f64 / n;
            let p_s = p_s[sensory_idx] as f64 / n;
            let p_a = p_a[action_idx] as f64 / n;
            mi += p_sa * (p_sa / (p_s * p_a)).log2();
        }
    }

    // Miller-Madow bias correction for MI = H(S) + H(A) - H(S,A): the net
    // ML-estimate bias to subtract is (K_joint - K_S - K_A + 1)/(2N ln 2).
    let correction = (nonzero_cells as f64 - nonzero_s as f64 - nonzero_a as f64 + 1.0)
        / (2.0 * n * std::f64::consts::LN_2);

    Some((mi - correction).max(0.0))
}
