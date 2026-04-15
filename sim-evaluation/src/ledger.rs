//! Per-organism sidecar maintained during a sim run. Produces
//! [`OrganismLifetimeRow`]s at death (or at end-of-run for survivors), and
//! per-tick action aggregates for the `action_counts` table. Interval- and
//! pillar-level metrics are NOT computed here — they live in the analysis
//! layer which reads the persisted dataset.
//!
//! The sidecar also tracks `num_offspring` per organism so genome snapshots
//! can pick the top reproducer at each flush boundary.

use crate::dataset::{
    ActionCountRow, OrganismLifetimeRow, ReproductionEventRow, ReproductionOutcome, ACTION_COUNT,
    JOINT_LEN,
};
use sim_types::{ActionRecord, ActionType, OrganismId, ReproductionEvent};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

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

/// Per-tick action aggregates. One row per action type will be emitted to
/// `action_counts` on every tick.
#[derive(Debug, Clone)]
pub struct TickActionAggregates {
    pub counts: [u64; ACTION_COUNT],
    pub failed: [u64; ACTION_COUNT],
    pub juvenile: [u64; ACTION_COUNT],
    pub adult: [u64; ACTION_COUNT],
}

impl TickActionAggregates {
    fn new() -> Self {
        Self {
            counts: [0; ACTION_COUNT],
            failed: [0; ACTION_COUNT],
            juvenile: [0; ACTION_COUNT],
            adult: [0; ACTION_COUNT],
        }
    }

    pub fn into_rows(self, tick: u64) -> Vec<ActionCountRow> {
        (0..ACTION_COUNT as u8)
            .map(|action_type| ActionCountRow {
                tick,
                action_type,
                count: self.counts[action_type as usize],
                failed_count: self.failed[action_type as usize],
                juvenile_count: self.juvenile[action_type as usize],
                adult_count: self.adult[action_type as usize],
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct OrganismEntry {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub species_id: u64,
    pub generation: u64,
    pub birth_tick: u64,
    pub age_of_maturity: u32,
    pub num_offspring: u32,
    pub total_actions: u64,
    /// Lifetime count of consumptions. Tracked from `ActionRecord::consumptions_count`
    /// (a cumulative counter produced by the sim), so `record_action` replaces
    /// the stored value each tick.
    pub total_consumptions: u64,
    pub action_histogram: [u64; ACTION_COUNT],
    pub utilization: f32,
    pub food_ahead_ticks: u32,
    pub fwd_when_food_ahead: u32,
    /// Row-major `[SENSORY_BIN_COUNT][ACTION_COUNT]` flattened, juvenile-only.
    pub joint_juvenile: Vec<u64>,
    /// Row-major `[SENSORY_BIN_COUNT][ACTION_COUNT]` flattened, adult-only.
    pub joint_adult: Vec<u64>,
}

impl OrganismEntry {
    fn new(
        id: u64,
        parent_id: Option<u64>,
        species_id: u64,
        generation: u64,
        birth_tick: u64,
        age_of_maturity: u32,
    ) -> Self {
        Self {
            id,
            parent_id,
            species_id,
            generation,
            birth_tick,
            age_of_maturity,
            num_offspring: 0,
            total_actions: 0,
            total_consumptions: 0,
            action_histogram: [0; ACTION_COUNT],
            utilization: 0.0,
            food_ahead_ticks: 0,
            fwd_when_food_ahead: 0,
            joint_juvenile: vec![0; JOINT_LEN],
            joint_adult: vec![0; JOINT_LEN],
        }
    }

    fn into_lifetime_row(self, death_tick: Option<u64>) -> OrganismLifetimeRow {
        OrganismLifetimeRow {
            id: self.id,
            parent_id: self.parent_id,
            species_id: self.species_id,
            birth_tick: self.birth_tick,
            death_tick,
            generation: self.generation,
            age_of_maturity: self.age_of_maturity,
            num_offspring: self.num_offspring,
            total_consumptions: self.total_consumptions,
            total_actions: self.total_actions,
            action_histogram: self.action_histogram.to_vec(),
            utilization: self.utilization,
            food_ahead_ticks: self.food_ahead_ticks,
            fwd_when_food_ahead: self.fwd_when_food_ahead,
            joint_juvenile: self.joint_juvenile,
            joint_adult: self.joint_adult,
        }
    }
}

#[derive(Debug)]
pub struct Ledger {
    sidecar: IdHashMap<OrganismId, OrganismEntry>,
    tick_aggregates: TickActionAggregates,
    /// Child id → parent id, captured when a successful `ReproductionEvent`
    /// is observed and consumed on the corresponding `birth()` call. Founders
    /// and periodic injections never appear here, so their `parent_id` stays
    /// `None` — that's the signal used downstream to distinguish descendants
    /// from seeded organisms when computing lineage-survival metrics.
    pending_parents: IdHashMap<OrganismId, OrganismId>,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            sidecar: IdHashMap::default(),
            tick_aggregates: TickActionAggregates::new(),
            pending_parents: IdHashMap::default(),
        }
    }

    pub fn birth(
        &mut self,
        id: OrganismId,
        species_id: u64,
        generation: u64,
        birth_tick: u64,
        age_of_maturity: u32,
    ) {
        let parent_id = self.pending_parents.remove(&id).map(|p| p.0);
        self.sidecar.insert(
            id,
            OrganismEntry::new(
                id.0,
                parent_id,
                species_id,
                generation,
                birth_tick,
                age_of_maturity,
            ),
        );
    }

    /// Ingest one tick's worth of per-organism action records. Updates the
    /// per-organism sidecar and the per-tick action aggregates.
    pub fn record_action(&mut self, record: &ActionRecord) {
        let action_idx = action_index(record.selected_action);
        let sensory_bin = sensory_bin(record);
        let is_juvenile_record = self
            .sidecar
            .get(&record.organism_id)
            .map(|entry| record.age_turns < u64::from(entry.age_of_maturity))
            .unwrap_or(false);

        self.tick_aggregates.counts[action_idx] =
            self.tick_aggregates.counts[action_idx].saturating_add(1);
        if action_can_fail(record.selected_action) && record.action_failed {
            self.tick_aggregates.failed[action_idx] =
                self.tick_aggregates.failed[action_idx].saturating_add(1);
        }
        if is_juvenile_record {
            self.tick_aggregates.juvenile[action_idx] =
                self.tick_aggregates.juvenile[action_idx].saturating_add(1);
        } else {
            self.tick_aggregates.adult[action_idx] =
                self.tick_aggregates.adult[action_idx].saturating_add(1);
        }

        let Some(entry) = self.sidecar.get_mut(&record.organism_id) else {
            return;
        };

        entry.total_actions = entry.total_actions.saturating_add(1);
        entry.total_consumptions = record.consumptions_count;
        entry.action_histogram[action_idx] = entry.action_histogram[action_idx].saturating_add(1);
        entry.utilization = record.utilization.clamp(0.0, 1.0);
        let joint_idx = sensory_bin * ACTION_COUNT + action_idx;
        if is_juvenile_record {
            entry.joint_juvenile[joint_idx] = entry.joint_juvenile[joint_idx].saturating_add(1);
        } else {
            entry.joint_adult[joint_idx] = entry.joint_adult[joint_idx].saturating_add(1);
        }
        if record.food_visible_at_offset(0) {
            entry.food_ahead_ticks = entry.food_ahead_ticks.saturating_add(1);
            if record.selected_action == ActionType::Forward {
                entry.fwd_when_food_ahead = entry.fwd_when_food_ahead.saturating_add(1);
            }
        }
    }

    pub fn record_reproduction(
        &mut self,
        tick: u64,
        event: ReproductionEvent,
    ) -> ReproductionEventRow {
        let outcome = match event.failure_cause {
            None => {
                if let Some(entry) = self.sidecar.get_mut(&event.parent_id) {
                    entry.num_offspring = entry.num_offspring.saturating_add(1);
                }
                if let Some(child_id) = event.child_id {
                    self.pending_parents.insert(child_id, event.parent_id);
                }
                ReproductionOutcome::Success
            }
            Some(sim_types::ReproductionFailureCause::BlockedBirth) => {
                ReproductionOutcome::BlockedBirth
            }
            Some(sim_types::ReproductionFailureCause::ParentDied) => {
                ReproductionOutcome::ParentDied
            }
        };
        ReproductionEventRow {
            tick,
            parent_id: event.parent_id.0,
            parent_species_id: event.parent_species_id.0,
            parent_generation: event.parent_generation,
            parent_age_turns: event.parent_age_turns,
            child_id: event.child_id.map(|c| c.0),
            investment_energy: event.investment_energy,
            parent_energy_after: event.parent_energy_after_event,
            outcome: outcome.code(),
        }
    }

    pub fn death(&mut self, id: OrganismId, tick: u64) -> Option<OrganismLifetimeRow> {
        self.sidecar
            .remove(&id)
            .map(|entry| entry.into_lifetime_row(Some(tick)))
    }

    /// Drain remaining live entries at run end. Called once after the last
    /// tick completes so every organism contributes a lifetime row.
    pub fn drain_survivors(&mut self) -> Vec<OrganismLifetimeRow> {
        let survivors = std::mem::take(&mut self.sidecar);
        survivors
            .into_values()
            .map(|entry| entry.into_lifetime_row(None))
            .collect()
    }

    /// Take ownership of the buffered tick aggregates and reset the buffer
    /// for the next tick. Called exactly once per tick.
    pub fn take_tick_aggregates(&mut self) -> TickActionAggregates {
        std::mem::replace(&mut self.tick_aggregates, TickActionAggregates::new())
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

fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::Idle => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
        ActionType::Forward => 3,
        ActionType::Eat => 4,
        ActionType::Attack => 5,
        ActionType::Reproduce => 6,
    }
}

fn action_can_fail(action: ActionType) -> bool {
    matches!(
        action,
        ActionType::Forward | ActionType::Eat | ActionType::Attack | ActionType::Reproduce
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{OrganismId, SensoryReceptor, SpeciesId};

    #[test]
    fn tick_aggregates_count_only_contingent_action_failures() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1), SpeciesId(0).0, 0, 0, 10);
        ledger.record_action(&ActionRecord {
            organism_id: OrganismId(1),
            selected_action: ActionType::TurnLeft,
            action_failed: true,
            food_visible: [false; SensoryReceptor::VISION_RAY_OFFSETS.len()],
            damage_taken_last_turn: 0.0,
            age_turns: 0,
            utilization: 0.0,
            consumptions_count: 0,
        });
        ledger.record_action(&ActionRecord {
            organism_id: OrganismId(1),
            selected_action: ActionType::Eat,
            action_failed: true,
            food_visible: [false; SensoryReceptor::VISION_RAY_OFFSETS.len()],
            damage_taken_last_turn: 0.0,
            age_turns: 0,
            utilization: 0.0,
            consumptions_count: 0,
        });
        ledger.record_action(&ActionRecord {
            organism_id: OrganismId(1),
            selected_action: ActionType::Attack,
            action_failed: false,
            food_visible: [false; SensoryReceptor::VISION_RAY_OFFSETS.len()],
            damage_taken_last_turn: 0.0,
            age_turns: 0,
            utilization: 0.0,
            consumptions_count: 0,
        });

        let aggregates = ledger.take_tick_aggregates();
        assert_eq!(aggregates.counts[action_index(ActionType::TurnLeft)], 1);
        assert_eq!(aggregates.failed[action_index(ActionType::TurnLeft)], 0);
        assert_eq!(aggregates.failed[action_index(ActionType::Eat)], 1);
        assert_eq!(aggregates.failed[action_index(ActionType::Attack)], 0);
    }

    #[test]
    fn top_reproducer_picks_max_offspring_breaking_ties_low_id() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1), 0, 0, 0, 10);
        ledger.birth(OrganismId(2), 0, 0, 0, 10);
        ledger.birth(OrganismId(3), 0, 0, 0, 10);

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

        let top = ledger.top_reproducer().unwrap();
        assert_eq!(top.id, 2);
    }

    #[test]
    fn top_reproducer_none_when_nothing_has_reproduced() {
        let mut ledger = Ledger::new();
        ledger.birth(OrganismId(1), 0, 0, 0, 10);
        assert!(ledger.top_reproducer().is_none());
    }
}
