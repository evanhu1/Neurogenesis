//! Derive [`IntervalMetrics`] (the per-reporting-interval timeseries rows
//! consumed by reports and the live CLI) from raw fact rows.
//!
//! Every behavioural metric is pooled from descendant [`OrganismLifetimeRow`]s
//! bucketed by `death_tick`. Intervals are closed-open windows
//! `(prev_end, this_end]`; the last may be shorter than `report_every` if
//! `total_ticks` isn't an exact multiple.
//!
//! This is the pure, storage-agnostic computation: callers pass row slices
//! (the eval harness from a loaded Parquet dataset, the CLI from its live
//! in-memory ledger output).

use crate::schema::{
    OrganismLifetimeRow, TickSummaryRow, ACTION_COUNT, DESCENDANT_CODE, JOINT_LEN, SENSORY_BIN_COUNT,
};
use serde::Serialize;
use std::collections::BTreeMap;

/// One row per reporting interval — the primary timeseries format consumed by
/// reports, comparisons, and the CSV export. Derived by pooling descendant
/// lifetime rows per death-tick interval. Every rate uses total actions taken
/// as the denominator.
#[derive(Debug, Clone, Serialize)]
pub struct IntervalMetrics {
    pub tick: u64,
    /// Descendant population (viability context, not a competence score).
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
    /// Mean within-life success-vs-age slope over descendants — in-life learning.
    pub learning_slope: Option<f64>,
}

pub fn derive_interval_metrics(
    tick_summary: &[TickSummaryRow],
    organism_lifetimes: &[OrganismLifetimeRow],
    report_every: u64,
    total_ticks: u64,
) -> Vec<IntervalMetrics> {
    if report_every == 0 || total_ticks == 0 {
        return Vec::new();
    }

    let boundaries: Vec<u64> = {
        let mut b = Vec::new();
        let mut cursor = report_every;
        while cursor < total_ticks {
            b.push(cursor);
            cursor = cursor.saturating_add(report_every);
        }
        b.push(total_ticks);
        b
    };

    let mut accs: Vec<IntervalAccumulator> = boundaries
        .iter()
        .copied()
        .map(IntervalAccumulator::new)
        .collect();

    // `pop` context: the descendant population reported nearest each interval
    // end (the per-tick line).
    let mut pop_by_tick: BTreeMap<u64, u32> = BTreeMap::new();
    for row in tick_summary {
        pop_by_tick.insert(row.tick, row.descendant_population);
    }
    for acc in accs.iter_mut() {
        acc.pop = pop_by_tick
            .range(..=acc.end_tick)
            .next_back()
            .map(|(_, &pop)| pop)
            .unwrap_or(0);
    }

    for row in organism_lifetimes {
        if row.origin != DESCENDANT_CODE {
            continue;
        }
        let Some(death_tick) = row.death_tick else {
            continue;
        };
        if let Some(idx) = interval_index(&boundaries, death_tick) {
            accs[idx].add_lifetime(row);
        }
    }

    accs.into_iter().map(|acc| acc.finalize()).collect()
}

fn interval_index(boundaries: &[u64], tick: u64) -> Option<usize> {
    let idx = boundaries.partition_point(|&end| end < tick);
    (idx < boundaries.len()).then_some(idx)
}

struct IntervalAccumulator {
    end_tick: u64,
    pop: u32,
    total_actions: u64,
    contingent_actions: u64,
    failed_actions: u64,
    plant_consumptions: u64,
    prey_consumptions: u64,
    pooled_joint: [u64; JOINT_LEN],
    learning_slope_sum: f64,
    learning_slope_count: u64,
}

impl IntervalAccumulator {
    fn new(end_tick: u64) -> Self {
        Self {
            end_tick,
            pop: 0,
            total_actions: 0,
            contingent_actions: 0,
            failed_actions: 0,
            plant_consumptions: 0,
            prey_consumptions: 0,
            pooled_joint: [0; JOINT_LEN],
            learning_slope_sum: 0.0,
            learning_slope_count: 0,
        }
    }

    fn add_lifetime(&mut self, row: &OrganismLifetimeRow) {
        self.total_actions = self.total_actions.saturating_add(row.total_actions);
        self.contingent_actions = self
            .contingent_actions
            .saturating_add(row.contingent_actions);
        self.failed_actions = self.failed_actions.saturating_add(row.failed_actions);
        self.plant_consumptions = self
            .plant_consumptions
            .saturating_add(row.plant_consumptions);
        self.prey_consumptions = self.prey_consumptions.saturating_add(row.prey_consumptions);
        pool_joint(&mut self.pooled_joint, &row.joint_sensory_action);
        if let Some(slope) = row.learning_slope {
            self.learning_slope_sum += f64::from(slope);
            self.learning_slope_count += 1;
        }
    }

    fn finalize(self) -> IntervalMetrics {
        let total = self.total_actions;
        let rate = |num: u64| (total > 0).then(|| num as f64 / total as f64);

        let action_effectiveness =
            rate(self.contingent_actions.saturating_sub(self.failed_actions));
        let plant_consumption_rate = rate(self.plant_consumptions);
        let prey_consumption_rate = rate(self.prey_consumptions);

        let mi_sa = mi_from_joint(&self.pooled_joint);
        let learning_slope = (self.learning_slope_count > 0)
            .then(|| self.learning_slope_sum / self.learning_slope_count as f64);

        IntervalMetrics {
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
