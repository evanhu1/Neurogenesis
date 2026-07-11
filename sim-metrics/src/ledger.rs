//! Per-organism sidecar maintained during a sim run. Produces
//! [`OrganismLifetimeRow`]s at death (or at end-of-run for survivors).
//! Interval- and pillar-level metrics are NOT computed here — they live in the
//! analysis layer ([`crate::intervals`], [`crate::pillars`]) which reads pooled
//! lifetime rows.
//!
//! The sidecar also tracks lifetime consumptions per organism so genome
//! snapshots can pick the top forager at each flush boundary.

use crate::schema::{OrganismLifetimeRow, ACTION_COUNT, JOINT_LEN};
use serde::{Deserialize, Serialize};
use sim_types::{ActionRecord, OrganismId};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Minimum number of contingent actions an organism must take before we trust a
/// within-life learning slope; below this the regression is too noisy to mean
/// anything.
const MIN_LEARNING_SAMPLES: u64 = 30;

/// Identity hasher for integer-keyed HashMaps. OrganismId values are unique
/// monotonic u64s, so the identity function is a perfect hash.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdHasher only supports write_u64");
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

type IdHashMap<K, V> = HashMap<K, V, BuildHasherDefault<IdHasher>>;

/// Online accumulator for the regression slope of action success vs age.
/// Tracks the five running sums needed for an ordinary-least-squares slope
/// without storing per-action samples or needing to know the lifespan up front.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct LearningAccumulator {
    n: u64,
    sum_age: f64,
    sum_age_sq: f64,
    sum_succ: f64,
    sum_age_succ: f64,
}

impl LearningAccumulator {
    fn observe(&mut self, age: f64, success: f64) {
        self.n += 1;
        self.sum_age += age;
        self.sum_age_sq += age * age;
        self.sum_succ += success;
        self.sum_age_succ += age * success;
    }

    /// OLS slope `Cov(age, success) / Var(age)`. `None` below the sample gate
    /// or when age has no spread (all actions at one age).
    fn slope(&self) -> Option<f32> {
        if self.n < MIN_LEARNING_SAMPLES {
            return None;
        }
        let n = self.n as f64;
        let denom = n * self.sum_age_sq - self.sum_age * self.sum_age;
        if denom <= 0.0 {
            return None;
        }
        let numer = n * self.sum_age_succ - self.sum_age * self.sum_succ;
        Some((numer / denom) as f32)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrganismEntry {
    pub id: u64,
    total_actions: u64,
    contingent_actions: u64,
    failed_actions: u64,
    plant_consumptions: u64,
    prey_consumptions: u64,
    /// Row-major `[SENSORY_BIN_COUNT][ACTION_COUNT]` flattened across the whole
    /// lifetime.
    joint_sensory_action: Vec<u64>,
    learning: LearningAccumulator,
}

impl OrganismEntry {
    fn new(id: u64) -> Self {
        Self {
            id,
            total_actions: 0,
            contingent_actions: 0,
            failed_actions: 0,
            plant_consumptions: 0,
            prey_consumptions: 0,
            joint_sensory_action: vec![0; JOINT_LEN],
            learning: LearningAccumulator::default(),
        }
    }

    /// Total lifetime consumptions (foraging + predation) — the survival-arena
    /// analog of reproductive success, used to pick a representative genome.
    pub fn total_consumptions(&self) -> u64 {
        self.plant_consumptions
            .saturating_add(self.prey_consumptions)
    }

    fn into_lifetime_row(self, death_tick: Option<u64>) -> OrganismLifetimeRow {
        OrganismLifetimeRow {
            id: self.id,
            death_tick,
            total_actions: self.total_actions,
            contingent_actions: self.contingent_actions,
            failed_actions: self.failed_actions,
            plant_consumptions: self.plant_consumptions,
            prey_consumptions: self.prey_consumptions,
            joint_sensory_action: self.joint_sensory_action,
            learning_slope: self.learning.slope(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Ledger {
    sidecar: IdHashMap<OrganismId, OrganismEntry>,
    population: u32,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            sidecar: IdHashMap::default(),
            population: 0,
        }
    }

    pub fn birth(&mut self, id: OrganismId) {
        self.population = self.population.saturating_add(1);
        self.sidecar.insert(id, OrganismEntry::new(id.0));
    }

    /// Register an organism that is *already alive* when recording begins
    /// mid-run. Lifetime counters therefore start partway through the
    /// organism's life; callers should label such windows partial.
    pub fn register_existing(&mut self, id: OrganismId) {
        self.population = self.population.saturating_add(1);
        self.sidecar.insert(id, OrganismEntry::new(id.0));
    }

    /// Ingest one tick's worth of one organism's action record.
    pub fn record_action(&mut self, record: &ActionRecord) {
        let action_idx = record.selected_action.index();
        let sensory_bin = sensory_bin(record);
        // Absent sidecar entry means we never saw this organism's birth — skip.
        let Some(entry) = self.sidecar.get_mut(&record.organism_id) else {
            return;
        };

        entry.total_actions = entry.total_actions.saturating_add(1);
        let joint_idx = sensory_bin * ACTION_COUNT + action_idx;
        entry.joint_sensory_action[joint_idx] =
            entry.joint_sensory_action[joint_idx].saturating_add(1);
        entry.plant_consumptions = record.plant_consumptions_count;
        entry.prey_consumptions = record.prey_consumptions_count;

        if record.selected_action.can_fail() {
            entry.contingent_actions = entry.contingent_actions.saturating_add(1);
            if record.action_failed {
                entry.failed_actions = entry.failed_actions.saturating_add(1);
            }
            // In-life learning is measured on contingent-action competence
            // (Forward/Eat/Attack success vs age).
            let success = if record.action_failed { 0.0 } else { 1.0 };
            entry.learning.observe(record.age_turns as f64, success);
        }
    }

    pub fn death(&mut self, id: OrganismId, tick: u64) -> Option<OrganismLifetimeRow> {
        let entry = self.sidecar.remove(&id)?;
        self.population = self.population.saturating_sub(1);
        Some(entry.into_lifetime_row(Some(tick)))
    }

    /// Drain remaining live entries at run end. Called once after the last
    /// tick completes so every organism contributes a lifetime row.
    pub fn drain_survivors(&mut self) -> Vec<OrganismLifetimeRow> {
        let survivors = std::mem::take(&mut self.sidecar);
        self.population = 0;
        survivors
            .into_values()
            .map(|entry| entry.into_lifetime_row(None))
            .collect()
    }

    pub fn population(&self) -> u32 {
        self.population
    }

    /// Select the living organism with the most lifetime consumptions — the
    /// survival-arena analog of the top reproducer, used to pick a
    /// representative genome. Ties are broken by lowest id to keep snapshot
    /// selection deterministic. Returns `None` when no living organism has
    /// consumed anything yet.
    pub fn top_forager(&self) -> Option<&OrganismEntry> {
        let mut best: Option<&OrganismEntry> = None;
        for entry in self.sidecar.values() {
            if entry.total_consumptions() == 0 {
                continue;
            }
            match best {
                None => best = Some(entry),
                Some(current)
                    if entry.total_consumptions() > current.total_consumptions()
                        || (entry.total_consumptions() == current.total_consumptions()
                            && entry.id < current.id) =>
                {
                    best = Some(entry)
                }
                _ => {}
            }
        }
        best
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

fn sensory_bin(record: &ActionRecord) -> usize {
    for (idx, visible) in record.food_visible.iter().copied().enumerate() {
        if visible {
            return idx + 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{ActionType, OrganismId, SensoryReceptor};

    fn record(action: ActionType, failed: bool, age: u64) -> ActionRecord {
        ActionRecord {
            organism_id: OrganismId(1),
            selected_action: action,
            action_failed: failed,
            food_visible: [false; SensoryReceptor::VISION_RAY_OFFSETS.len()],
            age_turns: age,
            utilization: 0.0,
            consumptions_count: 0,
            plant_consumptions_count: 0,
            prey_consumptions_count: 0,
        }
    }

    #[test]
    fn lifetime_row_counts_contingent_failures_and_consumptions() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1));
        ledger.record_action(&record(ActionType::TurnLeft, true, 0));
        ledger.record_action(&record(ActionType::Eat, true, 0));
        let mut ate = record(ActionType::Eat, false, 1);
        ate.plant_consumptions_count = 1;
        ledger.record_action(&ate);

        let row = ledger.death(OrganismId(1), 10).unwrap();
        assert_eq!(row.total_actions, 3);
        // Turns never fail; both Eats are contingent, one failed.
        assert_eq!(row.contingent_actions, 2);
        assert_eq!(row.failed_actions, 1);
        assert_eq!(row.plant_consumptions, 1);
    }

    #[test]
    fn learning_slope_none_below_sample_gate() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1));
        ledger.record_action(&record(ActionType::Forward, false, 0));
        let row = ledger.death(OrganismId(1), 10).unwrap();
        assert!(row.learning_slope.is_none());
    }
}
