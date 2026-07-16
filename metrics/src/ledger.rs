//! Per-organism sidecar maintained during a sim run. Produces
//! [`OrganismLifetimeRow`]s at death (or at end-of-run for survivors).
//! Interval- and pillar-level metrics are NOT computed here — they live in the
//! analysis layer ([`crate::intervals`], [`crate::pillars`]) which reads pooled
//! lifetime rows.
//!
//! The sidecar also tracks lifetime successful attacks per organism.

use crate::schema::{BehaviorIntervalRow, OrganismLifetimeRow};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use types::{ActionRecord, ActionType, OrganismId};

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct BehaviorAccumulator {
    total_actions: u64,
    contingent_actions: u64,
    failed_actions: u64,
    successful_attacks: u64,
    learning_by_organism: BTreeMap<OrganismId, LearningAccumulator>,
}

impl BehaviorAccumulator {
    fn observe(&mut self, record: &ActionRecord, successful_attack_delta: u64) {
        self.total_actions = self.total_actions.saturating_add(1);
        self.successful_attacks = self
            .successful_attacks
            .saturating_add(successful_attack_delta);

        let contingent_mask = record.selected_action_mask
            & (ActionType::Forward.command_bit() | ActionType::Attack.command_bit());
        let contingent_count = u64::from(contingent_mask.count_ones());
        let failed_count = u64::from(record.failed_action_mask.count_ones());
        if contingent_count > 0 {
            self.contingent_actions = self.contingent_actions.saturating_add(contingent_count);
            self.failed_actions = self.failed_actions.saturating_add(failed_count);
            self.learning_by_organism
                .entry(record.organism_id)
                .or_default()
                .observe(
                    record.age_turns as f64,
                    (contingent_count - failed_count) as f64 / contingent_count as f64,
                );
        }
    }

    fn into_row(self, start_tick: u64, end_tick: u64, population: u32) -> BehaviorIntervalRow {
        let (learning_samples, learning_within_numerator, learning_within_denominator) = self
            .learning_by_organism
            .values()
            .fold((0_u64, 0.0_f64, 0.0_f64), |acc, learning| {
                let (samples, numerator, denominator) = learning.within_centered_terms();
                (
                    acc.0.saturating_add(samples),
                    acc.1 + numerator,
                    acc.2 + denominator,
                )
            });
        BehaviorIntervalRow {
            start_tick,
            end_tick,
            population,
            total_actions: self.total_actions,
            contingent_actions: self.contingent_actions,
            failed_actions: self.failed_actions,
            successful_attacks: self.successful_attacks,
            learning_samples,
            learning_within_numerator,
            learning_within_denominator,
        }
    }
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

    fn within_centered_terms(&self) -> (u64, f64, f64) {
        if self.n == 0 {
            return (0, 0.0, 0.0);
        }
        let n = self.n as f64;
        let numerator = self.sum_age_succ - self.sum_age * self.sum_succ / n;
        let denominator = self.sum_age_sq - self.sum_age * self.sum_age / n;
        (self.n, numerator, denominator.max(0.0))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrganismEntry {
    pub id: u64,
    total_actions: u64,
    contingent_actions: u64,
    failed_actions: u64,
    successful_attacks: u64,
    /// Last cumulative engine counter observed; kept separate from the
    /// recorder-local total above so mid-run recording starts at zero.
    last_seen_successful_attacks: u64,
    learning: LearningAccumulator,
}

impl OrganismEntry {
    fn new(id: u64) -> Self {
        Self {
            id,
            total_actions: 0,
            contingent_actions: 0,
            failed_actions: 0,
            successful_attacks: 0,
            last_seen_successful_attacks: 0,
            learning: LearningAccumulator::default(),
        }
    }

    pub fn successful_attacks(&self) -> u64 {
        self.successful_attacks
    }

    fn into_lifetime_row(self, death_tick: Option<u64>) -> OrganismLifetimeRow {
        OrganismLifetimeRow {
            id: self.id,
            death_tick,
            total_actions: self.total_actions,
            contingent_actions: self.contingent_actions,
            failed_actions: self.failed_actions,
            successful_attacks: self.successful_attacks,
            learning_slope: self.learning.slope(),
        }
    }
}

#[derive(Debug)]
pub struct Ledger {
    sidecar: IdHashMap<OrganismId, OrganismEntry>,
    population: u32,
    behavior_interval_start_tick: u64,
    behavior: BehaviorAccumulator,
}

#[derive(Serialize)]
struct LedgerWireRef<'a> {
    sidecar: Vec<(&'a OrganismId, &'a OrganismEntry)>,
    population: u32,
    behavior_interval_start_tick: u64,
    behavior: &'a BehaviorAccumulator,
}

#[derive(Deserialize)]
struct LedgerWireOwned {
    sidecar: Vec<(OrganismId, OrganismEntry)>,
    population: u32,
    behavior_interval_start_tick: u64,
    behavior: BehaviorAccumulator,
}

impl Serialize for Ledger {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sidecar: Vec<_> = self.sidecar.iter().collect();
        sidecar.sort_unstable_by_key(|(id, _)| id.0);
        LedgerWireRef {
            sidecar,
            population: self.population,
            behavior_interval_start_tick: self.behavior_interval_start_tick,
            behavior: &self.behavior,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Ledger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LedgerWireOwned::deserialize(deserializer)?;
        let mut sidecar = IdHashMap::default();
        for (id, entry) in wire.sidecar {
            sidecar.insert(id, entry);
        }
        Ok(Self {
            sidecar,
            population: wire.population,
            behavior_interval_start_tick: wire.behavior_interval_start_tick,
            behavior: wire.behavior,
        })
    }
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            sidecar: IdHashMap::default(),
            population: 0,
            behavior_interval_start_tick: 0,
            behavior: BehaviorAccumulator::default(),
        }
    }

    /// Set the exclusive start boundary when recording begins partway through
    /// a world. Fresh runs already start at zero.
    pub fn begin_behavior_recording(&mut self, start_tick: u64) {
        self.behavior_interval_start_tick = start_tick;
        self.behavior = BehaviorAccumulator::default();
    }

    pub fn birth(&mut self, id: OrganismId) {
        self.population = self.population.saturating_add(1);
        self.sidecar.insert(id, OrganismEntry::new(id.0));
    }

    /// Register an organism that is *already alive* when recording begins
    /// mid-run. Lifetime counters therefore start partway through the
    /// organism's life; callers should label such windows partial.
    pub fn register_existing(&mut self, organism: &types::OrganismState) {
        self.population = self.population.saturating_add(1);
        let mut entry = OrganismEntry::new(organism.id.0);
        // Establish a cumulative-counter baseline so the first recorded action
        // cannot attribute pre-recording attacks to the new interval.
        entry.last_seen_successful_attacks = organism.successful_attacks_count;
        self.sidecar.insert(organism.id, entry);
    }

    /// Ingest one tick's worth of one organism's action record.
    pub fn record_action(&mut self, record: &ActionRecord) {
        // Absent sidecar entry means we never saw this organism's birth — skip.
        let Some(entry) = self.sidecar.get_mut(&record.organism_id) else {
            return;
        };

        let successful_attack_delta = record
            .successful_attacks_count
            .saturating_sub(entry.last_seen_successful_attacks);
        self.behavior.observe(record, successful_attack_delta);

        entry.total_actions = entry.total_actions.saturating_add(1);
        entry.successful_attacks = entry
            .successful_attacks
            .saturating_add(successful_attack_delta);
        entry.last_seen_successful_attacks = record.successful_attacks_count;

        let contingent_mask = record.selected_action_mask
            & (ActionType::Forward.command_bit() | ActionType::Attack.command_bit());
        let contingent_count = u64::from(contingent_mask.count_ones());
        let failed_count = u64::from(record.failed_action_mask.count_ones());
        if contingent_count > 0 {
            entry.contingent_actions = entry.contingent_actions.saturating_add(contingent_count);
            entry.failed_actions = entry.failed_actions.saturating_add(failed_count);
            // In-life learning is measured on contingent-action competence
            // (Forward/Attack success vs age).
            let success = (contingent_count - failed_count) as f64 / contingent_count as f64;
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
        let mut rows: Vec<_> = survivors
            .into_values()
            .map(|entry| entry.into_lifetime_row(None))
            .collect();
        rows.sort_unstable_by_key(|row| row.id);
        rows
    }

    pub fn population(&self) -> u32 {
        self.population
    }

    /// Close the current action-time interval and begin a fresh one at
    /// `end_tick`. The row contains only facts observed since the preceding
    /// boundary, including actions by organisms that remain alive.
    pub fn take_behavior_interval(&mut self, end_tick: u64) -> BehaviorIntervalRow {
        let start_tick = self.behavior_interval_start_tick;
        self.behavior_interval_start_tick = end_tick;
        std::mem::take(&mut self.behavior).into_row(start_tick, end_tick, self.population)
    }

    /// Snapshot the open interval without draining it, for live read commands
    /// issued between reporting boundaries.
    pub fn behavior_interval_snapshot(&self, end_tick: u64) -> BehaviorIntervalRow {
        self.behavior
            .clone()
            .into_row(self.behavior_interval_start_tick, end_tick, self.population)
    }

    /// Select the living organism with the most successful attacks. Ties are
    /// broken by lowest id to keep snapshot selection deterministic.
    pub fn top_attacker(&self) -> Option<&OrganismEntry> {
        let mut best: Option<&OrganismEntry> = None;
        for entry in self.sidecar.values() {
            if entry.successful_attacks() == 0 {
                continue;
            }
            match best {
                None => best = Some(entry),
                Some(current)
                    if entry.successful_attacks() > current.successful_attacks()
                        || (entry.successful_attacks() == current.successful_attacks()
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

#[cfg(test)]
mod tests {
    use super::*;
    use types::{ActionType, OrganismId};

    fn record(action: ActionType, failed: bool, age: u64) -> ActionRecord {
        ActionRecord {
            organism_id: OrganismId(1),
            selected_action: action,
            selected_action_mask: action.command_bit(),
            failed_action_mask: if failed { action.command_bit() } else { 0 },
            action_failed: failed,
            age_turns: age,
            utilization: 0.0,
            successful_attacks_count: 0,
        }
    }

    #[test]
    fn lifetime_row_counts_contingent_failures_and_attacks() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1));
        ledger.record_action(&record(ActionType::TurnLeft, true, 0));
        ledger.record_action(&record(ActionType::Attack, true, 0));
        let mut attacked = record(ActionType::Attack, false, 1);
        attacked.successful_attacks_count = 1;
        ledger.record_action(&attacked);

        let row = ledger.death(OrganismId(1), 10).unwrap();
        assert_eq!(row.total_actions, 3);
        // Turns never fail; both attacks are contingent, one failed.
        assert_eq!(row.contingent_actions, 2);
        assert_eq!(row.failed_actions, 1);
        assert_eq!(row.successful_attacks, 1);
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
