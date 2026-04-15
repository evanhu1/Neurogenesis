//! Derive `IntervalMetrics` (the per-reporting-interval timeseries rows
//! consumed by reports) from the raw dataset tables.
//!
//! Intervals are closed-open windows `(prev_end, this_end]`. The last
//! interval may be shorter than `report_every` if `total_ticks` isn't an
//! exact multiple.

use crate::dataset::{DatasetReader, ACTION_COUNT, JOINT_LEN, SENSORY_BIN_COUNT};
use crate::types::IntervalMetrics;

const IDLE: usize = 0;
const FORWARD: usize = 3;
const EAT: usize = 4;
const ATTACK: usize = 5;
const REPRODUCE: usize = 6;

/// Actions whose failure is a meaningful signal (non-no-op, non-turn).
const CONTINGENT_ACTIONS: [usize; 4] = [FORWARD, EAT, ATTACK, REPRODUCE];

pub fn derive_interval_metrics(
    dataset: &DatasetReader,
    report_every: u64,
    total_ticks: u64,
    min_lifetime: u64,
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

    // Each table is visited exactly once. `partition_point` finds the owning
    // interval in O(log N) for each row, so the whole derivation is
    // O((rows_total) * log(intervals)) rather than O(rows_total * intervals).
    for row in &dataset.tick_summary {
        if let Some(idx) = interval_index(&boundaries, row.tick) {
            accs[idx].add_tick_summary(row);
        }
    }
    for row in &dataset.action_counts {
        if let Some(idx) = interval_index(&boundaries, row.tick) {
            accs[idx].add_action_count(row);
        }
    }
    for row in &dataset.organism_lifetimes {
        let Some(death_tick) = row.death_tick else {
            continue;
        };
        let lifetime = death_tick.saturating_sub(row.birth_tick);
        if lifetime < min_lifetime {
            continue;
        }
        if let Some(idx) = interval_index(&boundaries, death_tick) {
            accs[idx].add_lifetime(row, lifetime);
        }
    }
    for event in &dataset.reproduction_events {
        if event.outcome != 0 {
            continue;
        }
        if let Some(idx) = interval_index(&boundaries, event.tick) {
            accs[idx].add_reproduction(event.parent_age_turns);
        }
    }

    // Population snapshots are sparse (one per flush). For each interval, use
    // the most recent snapshot at or before the interval end.
    let snapshot_cursor = &dataset.population_snapshots;
    accs.iter_mut().for_each(|acc| {
        acc.population_snapshot = snapshot_cursor
            .iter()
            .filter(|row| row.tick <= acc.end_tick)
            .max_by_key(|row| row.tick)
            .cloned();
    });

    accs.into_iter().map(|acc| acc.finalize()).collect()
}

fn interval_index(boundaries: &[u64], tick: u64) -> Option<usize> {
    let idx = boundaries.partition_point(|&end| end < tick);
    (idx < boundaries.len()).then_some(idx)
}

struct IntervalAccumulator {
    end_tick: u64,
    // tick_summary
    births: u64,
    deaths: u64,
    consumptions: u64,
    predations: u64,
    population_exposure: u64,
    last_pop: u32,
    last_food: u32,
    last_max_generation: Option<u64>,
    // action_counts
    action_counts: [u64; ACTION_COUNT],
    action_failed: [u64; ACTION_COUNT],
    // deceased
    deceased_count: u64,
    lifetime_sum: u64,
    ate_count: u64,
    consumptions_sum: u64,
    utilization_sum: f64,
    food_ahead_ticks_sum: u64,
    fwd_when_food_ahead_sum: u64,
    pooled_juvenile: [u64; JOINT_LEN],
    pooled_adult: [u64; JOINT_LEN],
    // reproduction_events
    age_sum: f64,
    age_count: u64,
    // population snapshot (filled post-walk)
    population_snapshot: Option<crate::dataset::PopulationSnapshotRow>,
}

impl IntervalAccumulator {
    fn new(end_tick: u64) -> Self {
        Self {
            end_tick,
            births: 0,
            deaths: 0,
            consumptions: 0,
            predations: 0,
            population_exposure: 0,
            last_pop: 0,
            last_food: 0,
            last_max_generation: None,
            action_counts: [0; ACTION_COUNT],
            action_failed: [0; ACTION_COUNT],
            deceased_count: 0,
            lifetime_sum: 0,
            ate_count: 0,
            consumptions_sum: 0,
            utilization_sum: 0.0,
            food_ahead_ticks_sum: 0,
            fwd_when_food_ahead_sum: 0,
            pooled_juvenile: [0; JOINT_LEN],
            pooled_adult: [0; JOINT_LEN],
            age_sum: 0.0,
            age_count: 0,
            population_snapshot: None,
        }
    }

    fn add_tick_summary(&mut self, row: &crate::dataset::TickSummaryRow) {
        self.births = self.births.saturating_add(u64::from(row.births));
        self.deaths = self.deaths.saturating_add(u64::from(row.deaths));
        self.consumptions = self.consumptions.saturating_add(u64::from(row.consumptions));
        self.predations = self.predations.saturating_add(u64::from(row.predations));
        self.population_exposure = self
            .population_exposure
            .saturating_add(u64::from(row.population));
        self.last_pop = row.population;
        self.last_food = row.food_count;
        self.last_max_generation = row.max_generation.or(self.last_max_generation);
    }

    fn add_action_count(&mut self, row: &crate::dataset::ActionCountRow) {
        let idx = row.action_type as usize;
        if idx >= ACTION_COUNT {
            return;
        }
        self.action_counts[idx] = self.action_counts[idx].saturating_add(row.count);
        self.action_failed[idx] = self.action_failed[idx].saturating_add(row.failed_count);
    }

    fn add_lifetime(&mut self, row: &crate::dataset::OrganismLifetimeRow, lifetime: u64) {
        self.deceased_count = self.deceased_count.saturating_add(1);
        self.lifetime_sum = self.lifetime_sum.saturating_add(lifetime);
        if row.total_consumptions > 0 {
            self.ate_count = self.ate_count.saturating_add(1);
        }
        self.consumptions_sum = self.consumptions_sum.saturating_add(row.total_consumptions);
        self.utilization_sum += f64::from(row.utilization.clamp(0.0, 1.0));
        self.food_ahead_ticks_sum = self
            .food_ahead_ticks_sum
            .saturating_add(u64::from(row.food_ahead_ticks));
        self.fwd_when_food_ahead_sum = self
            .fwd_when_food_ahead_sum
            .saturating_add(u64::from(row.fwd_when_food_ahead));
        pool_joint(&mut self.pooled_juvenile, &row.joint_juvenile);
        pool_joint(&mut self.pooled_adult, &row.joint_adult);
    }

    fn add_reproduction(&mut self, parent_age_turns: u64) {
        self.age_sum += parent_age_turns as f64;
        self.age_count = self.age_count.saturating_add(1);
    }

    fn finalize(self) -> IntervalMetrics {
        let total_actions: u64 = self.action_counts.iter().sum();
        let attack_attempts = self.action_counts[ATTACK];

        let attack_attempt_rate = event_rate(attack_attempts, self.population_exposure);
        let foraging_rate = event_rate(self.consumptions, self.population_exposure);
        let predation_rate = event_rate(self.predations, self.population_exposure);
        let attack_success_rate = if attack_attempts == 0 {
            None
        } else {
            Some(self.predations as f64 / attack_attempts as f64)
        };
        let mut contingent_total = 0_u64;
        let mut contingent_failed = 0_u64;
        for idx in CONTINGENT_ACTIONS {
            contingent_total = contingent_total.saturating_add(self.action_counts[idx]);
            contingent_failed = contingent_failed.saturating_add(self.action_failed[idx]);
        }
        let failed_action_rate = if contingent_total == 0 {
            None
        } else {
            Some(contingent_failed as f64 / contingent_total as f64)
        };

        let mut action_histogram = [0.0_f64; ACTION_COUNT];
        if total_actions > 0 {
            for (idx, slot) in action_histogram.iter_mut().enumerate() {
                *slot = self.action_counts[idx] as f64 / total_actions as f64;
            }
        }
        let idle_fraction = if total_actions == 0 {
            None
        } else {
            Some(action_histogram[IDLE])
        };

        let pooled_total = sum_joints(&self.pooled_juvenile, &self.pooled_adult);
        let (life_mean, ate_pct, cons_mean, util) = if self.deceased_count == 0 {
            (None, None, None, None)
        } else {
            let count_f = self.deceased_count as f64;
            (
                Some(self.lifetime_sum as f64 / count_f),
                Some(100.0 * self.ate_count as f64 / count_f),
                Some(self.consumptions_sum as f64 / count_f),
                Some(self.utilization_sum / count_f),
            )
        };
        let p_fwd_food = if self.food_ahead_ticks_sum == 0 {
            None
        } else {
            Some(self.fwd_when_food_ahead_sum as f64 / self.food_ahead_ticks_sum as f64)
        };
        let mi_sa = mi_from_joint(&pooled_total);
        let mi_sa_juvenile = mi_from_joint(&self.pooled_juvenile);
        let mi_sa_adult = mi_from_joint(&self.pooled_adult);

        let generation_time = if self.age_count == 0 {
            None
        } else {
            Some(self.age_sum / self.age_count as f64)
        };

        let snapshot = self.population_snapshot.as_ref();

        IntervalMetrics {
            tick: self.end_tick,
            pop: self.last_pop,
            births: self.births,
            deaths: self.deaths,
            food: u64::from(self.last_food),
            max_generation: self.last_max_generation,
            life_mean,
            predation_rate,
            foraging_rate,
            attack_attempt_rate,
            attack_success_rate,
            failed_action_rate,
            ate_pct,
            cons_mean,
            brain_size: snapshot.and_then(|r| r.brain_size_mean),
            brain_size_stddev: snapshot.and_then(|r| r.brain_size_stddev),
            brain_size_p10: snapshot.and_then(|r| r.brain_size_p10),
            brain_size_p50: snapshot.and_then(|r| r.brain_size_p50),
            brain_size_p90: snapshot.and_then(|r| r.brain_size_p90),
            lineage_diversity: snapshot.and_then(|r| r.lineage_diversity),
            p_fwd_food,
            mi_sa,
            mi_sa_juvenile,
            mi_sa_adult,
            idle_fraction,
            generation_time,
            util,
            action_histogram,
        }
    }
}

fn pool_joint(into: &mut [u64; JOINT_LEN], from: &[u64]) {
    for (idx, value) in from.iter().take(JOINT_LEN).enumerate() {
        into[idx] = into[idx].saturating_add(*value);
    }
}

fn sum_joints(a: &[u64; JOINT_LEN], b: &[u64; JOINT_LEN]) -> [u64; JOINT_LEN] {
    let mut out = [0_u64; JOINT_LEN];
    for idx in 0..JOINT_LEN {
        out[idx] = a[idx].saturating_add(b[idx]);
    }
    out
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

    let correction = if nonzero_cells > 1 {
        (nonzero_cells as f64 - 1.0) / (2.0 * n * std::f64::consts::LN_2)
    } else {
        0.0
    };

    Some((mi - correction).max(0.0))
}

fn event_rate(events: u64, population_exposure: u64) -> Option<f64> {
    if population_exposure == 0 {
        return None;
    }
    Some(events as f64 / population_exposure as f64)
}

pub fn action_baseline_probability() -> f64 {
    1.0 / ACTION_COUNT as f64
}
