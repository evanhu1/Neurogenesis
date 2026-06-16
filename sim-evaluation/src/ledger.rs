//! Per-organism sidecar maintained during a sim run. Produces
//! [`OrganismLifetimeRow`]s at death (or at end-of-run for survivors).
//! Interval- and pillar-level metrics are NOT computed here — they live in the
//! analysis layer which reads the persisted dataset.
//!
//! The sidecar also tracks `num_offspring` per organism so genome snapshots
//! can pick the top reproducer at each flush boundary.

use crate::dataset::{OrganismLifetimeRow, OrganismOrigin, ACTION_COUNT, JOINT_LEN};
use sim_types::{ActionRecord, ActionType, OrganismId, ReproductionEvent};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Minimum number of non-Reproduce contingent actions an organism must take
/// before we trust a within-life learning slope; below this the regression is
/// too noisy to mean anything.
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
#[derive(Debug, Clone, Default)]
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

#[derive(Debug)]
pub struct OrganismEntry {
    pub id: u64,
    origin: OrganismOrigin,
    pub num_offspring: u32,
    total_actions: u64,
    contingent_actions: u64,
    failed_actions: u64,
    plant_consumptions: u64,
    prey_consumptions: u64,
    action_histogram: [u64; ACTION_COUNT],
    /// Row-major `[SENSORY_BIN_COUNT][ACTION_COUNT]` flattened across the whole
    /// lifetime.
    joint_sensory_action: Vec<u64>,
    learning: LearningAccumulator,
}

impl OrganismEntry {
    fn new(id: u64, origin: OrganismOrigin) -> Self {
        Self {
            id,
            origin,
            num_offspring: 0,
            total_actions: 0,
            contingent_actions: 0,
            failed_actions: 0,
            plant_consumptions: 0,
            prey_consumptions: 0,
            action_histogram: [0; ACTION_COUNT],
            joint_sensory_action: vec![0; JOINT_LEN],
            learning: LearningAccumulator::default(),
        }
    }

    fn into_lifetime_row(self, death_tick: Option<u64>) -> OrganismLifetimeRow {
        OrganismLifetimeRow {
            id: self.id,
            origin: self.origin.code(),
            death_tick,
            total_actions: self.total_actions,
            contingent_actions: self.contingent_actions,
            failed_actions: self.failed_actions,
            plant_consumptions: self.plant_consumptions,
            prey_consumptions: self.prey_consumptions,
            action_histogram: self.action_histogram.to_vec(),
            joint_sensory_action: self.joint_sensory_action,
            learning_slope: self.learning.slope(),
        }
    }
}

#[derive(Debug)]
pub struct Ledger {
    sidecar: IdHashMap<OrganismId, OrganismEntry>,
    /// Child id → parent id, captured when a successful `ReproductionEvent`
    /// is observed and consumed on the corresponding `birth()` call. Founders
    /// and periodic injections never appear here, so their `parent_id` stays
    /// `None` — that's the signal used to distinguish descendants from seeded
    /// organisms.
    pending_parents: IdHashMap<OrganismId, OrganismId>,
    descendant_population: u32,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            sidecar: IdHashMap::default(),
            pending_parents: IdHashMap::default(),
            descendant_population: 0,
        }
    }

    pub fn birth(&mut self, id: OrganismId, birth_tick: u64) {
        let parent_id = self.pending_parents.remove(&id).map(|p| p.0);
        let origin = if parent_id.is_some() {
            OrganismOrigin::Descendant
        } else if birth_tick == 0 {
            OrganismOrigin::InitialFounder
        } else {
            OrganismOrigin::PeriodicInjection
        };
        if origin == OrganismOrigin::Descendant {
            self.descendant_population = self.descendant_population.saturating_add(1);
        }
        self.sidecar.insert(id, OrganismEntry::new(id.0, origin));
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
        entry.action_histogram[action_idx] = entry.action_histogram[action_idx].saturating_add(1);
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
            // In-life learning is measured on maturation-neutral competence:
            // Forward/Eat/Attack success vs age. Reproduce is excluded because
            // its availability is gated on maturity, which would confound the
            // slope with the onset of reproduction rather than learning.
            if record.selected_action != ActionType::Reproduce {
                let success = if record.action_failed { 0.0 } else { 1.0 };
                entry.learning.observe(record.age_turns as f64, success);
            }
        }
    }

    pub fn record_reproduction(&mut self, event: &ReproductionEvent) {
        // Only successful events advance lineage state; failures (blocked
        // birth, parent died) carry no parent→child edge.
        if event.failure_cause.is_some() {
            return;
        }
        if let Some(entry) = self.sidecar.get_mut(&event.parent_id) {
            entry.num_offspring = entry.num_offspring.saturating_add(1);
        }
        if let Some(child_id) = event.child_id {
            self.pending_parents.insert(child_id, event.parent_id);
        }
    }

    pub fn death(&mut self, id: OrganismId, tick: u64) -> Option<OrganismLifetimeRow> {
        let entry = self.sidecar.remove(&id)?;
        if entry.origin == OrganismOrigin::Descendant {
            debug_assert!(
                self.descendant_population > 0,
                "descendant_population underflow: death of {id:?} with zero running count"
            );
            self.descendant_population = self.descendant_population.saturating_sub(1);
        }
        Some(entry.into_lifetime_row(Some(tick)))
    }

    /// Drain remaining live entries at run end. Called once after the last
    /// tick completes so every organism contributes a lifetime row.
    pub fn drain_survivors(&mut self) -> Vec<OrganismLifetimeRow> {
        let survivors = std::mem::take(&mut self.sidecar);
        self.descendant_population = 0;
        survivors
            .into_values()
            .map(|entry| entry.into_lifetime_row(None))
            .collect()
    }

    pub fn descendant_population(&self) -> u32 {
        self.descendant_population
    }

    /// Select the living organism with the most offspring. Ties are broken by
    /// lowest id to keep snapshot selection deterministic. Returns `None` when
    /// no living organism has reproduced yet.
    pub fn top_reproducer(&self) -> Option<&OrganismEntry> {
        let mut best: Option<&OrganismEntry> = None;
        for entry in self.sidecar.values() {
            if entry.num_offspring == 0 {
                continue;
            }
            match best {
                None => best = Some(entry),
                Some(current)
                    if entry.num_offspring > current.num_offspring
                        || (entry.num_offspring == current.num_offspring
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
    use sim_types::{OrganismId, SensoryReceptor};

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
        ledger.birth(OrganismId(1), 0);
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
        ledger.birth(OrganismId(1), 0);
        ledger.record_action(&record(ActionType::Forward, false, 0));
        let row = ledger.death(OrganismId(1), 10).unwrap();
        assert!(row.learning_slope.is_none());
    }

    #[test]
    fn top_reproducer_picks_max_offspring_breaking_ties_low_id() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1), 0);
        ledger.birth(OrganismId(2), 0);
        ledger.birth(OrganismId(3), 0);
        ledger
            .sidecar
            .get_mut(&OrganismId(1))
            .unwrap()
            .num_offspring = 3;
        ledger
            .sidecar
            .get_mut(&OrganismId(2))
            .unwrap()
            .num_offspring = 5;
        ledger
            .sidecar
            .get_mut(&OrganismId(3))
            .unwrap()
            .num_offspring = 5;
        assert_eq!(ledger.top_reproducer().unwrap().id, 2);
    }
}
