//! Per-organism sidecar maintained during a sim run. Produces
//! [`OrganismLifetimeRow`]s at death (or at end-of-run for survivors), and
//! per-tick action aggregates for the `action_counts` table. Interval- and
//! pillar-level metrics are NOT computed here — they live in the analysis
//! layer which reads the persisted dataset.
//!
//! The sidecar also tracks `num_offspring` per organism so genome snapshots
//! can pick the top reproducer at each flush boundary.

use crate::dataset::{
    ActionCountRow, OrganismLifetimeRow, OrganismOrigin, PopulationSnapshotRow,
    ReproductionEventRow, ReproductionOutcome, ACTION_COUNT, JOINT_LEN, ORIGIN_COUNT,
};
use sim_types::{ActionRecord, ActionType, OrganismId, OrganismState, ReproductionEvent};
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

/// Per-tick action aggregates split by origin class. One row per
/// `(origin, action_type)` combination is emitted each tick so the analysis
/// layer can filter to descendants without losing the founder/injection
/// buckets.
#[derive(Debug, Clone)]
pub struct TickActionAggregates {
    per_origin: [OriginBucket; ORIGIN_COUNT],
}

#[derive(Debug, Clone)]
struct OriginBucket {
    counts: [u64; ACTION_COUNT],
    failed: [u64; ACTION_COUNT],
    pre_maturity: [u64; ACTION_COUNT],
    post_maturity: [u64; ACTION_COUNT],
}

impl OriginBucket {
    fn new() -> Self {
        Self {
            counts: [0; ACTION_COUNT],
            failed: [0; ACTION_COUNT],
            pre_maturity: [0; ACTION_COUNT],
            post_maturity: [0; ACTION_COUNT],
        }
    }
}

impl TickActionAggregates {
    fn new() -> Self {
        Self {
            per_origin: [
                OriginBucket::new(),
                OriginBucket::new(),
                OriginBucket::new(),
            ],
        }
    }

    pub fn into_rows(self, tick: u64) -> Vec<ActionCountRow> {
        let mut rows = Vec::with_capacity(ORIGIN_COUNT * ACTION_COUNT);
        for (origin_idx, bucket) in self.per_origin.into_iter().enumerate() {
            if bucket.counts == [0; ACTION_COUNT] {
                continue;
            }
            let origin = origin_idx as u8;
            for action_type in 0..ACTION_COUNT as u8 {
                let idx = action_type as usize;
                rows.push(ActionCountRow {
                    tick,
                    origin,
                    action_type,
                    count: bucket.counts[idx],
                    failed_count: bucket.failed[idx],
                    pre_maturity_count: bucket.pre_maturity[idx],
                    post_maturity_count: bucket.post_maturity[idx],
                });
            }
        }
        rows
    }
}

#[derive(Debug, Clone)]
struct MaturityWindowSummary {
    action_count: u64,
    consumptions: u64,
    action_histogram: [u64; ACTION_COUNT],
    food_ahead_ticks: u32,
    fwd_when_food_ahead: u32,
}

impl MaturityWindowSummary {
    fn new() -> Self {
        Self {
            action_count: 0,
            consumptions: 0,
            action_histogram: [0; ACTION_COUNT],
            food_ahead_ticks: 0,
            fwd_when_food_ahead: 0,
        }
    }
}

#[derive(Debug)]
pub struct OrganismEntry {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub origin: OrganismOrigin,
    pub species_id: u64,
    pub generation: u64,
    pub birth_tick: u64,
    pub age_of_maturity: u32,
    maturity_tick: u64,
    pub num_offspring: u32,
    pub total_actions: u64,
    /// Lifetime count of consumptions. Tracked from `ActionRecord::consumptions_count`
    /// (a cumulative counter produced by the sim and patched post-commit so a
    /// tick's consumption is visible in that tick's record), so `record_action`
    /// replaces the stored value each tick.
    pub total_consumptions: u64,
    pub contingent_actions: u64,
    pub failed_actions: u64,
    pub action_histogram: [u64; ACTION_COUNT],
    pub utilization: f32,
    pub food_ahead_ticks: u32,
    pub fwd_when_food_ahead: u32,
    /// Row-major `[SENSORY_BIN_COUNT][ACTION_COUNT]` flattened across the
    /// whole lifetime.
    joint_sensory_action: Vec<u64>,
    pre_maturity: MaturityWindowSummary,
    post_maturity: MaturityWindowSummary,
}

impl OrganismEntry {
    fn new(
        id: u64,
        parent_id: Option<u64>,
        origin: OrganismOrigin,
        species_id: u64,
        generation: u64,
        birth_tick: u64,
        age_of_maturity: u32,
    ) -> Self {
        Self {
            id,
            parent_id,
            origin,
            species_id,
            generation,
            birth_tick,
            age_of_maturity,
            maturity_tick: birth_tick.saturating_add(u64::from(age_of_maturity)),
            num_offspring: 0,
            total_actions: 0,
            total_consumptions: 0,
            contingent_actions: 0,
            failed_actions: 0,
            action_histogram: [0; ACTION_COUNT],
            utilization: 0.0,
            food_ahead_ticks: 0,
            fwd_when_food_ahead: 0,
            joint_sensory_action: vec![0; JOINT_LEN],
            pre_maturity: MaturityWindowSummary::new(),
            post_maturity: MaturityWindowSummary::new(),
        }
    }

    fn into_lifetime_row(self, death_tick: Option<u64>) -> OrganismLifetimeRow {
        OrganismLifetimeRow {
            id: self.id,
            parent_id: self.parent_id,
            origin: self.origin.code(),
            species_id: self.species_id,
            birth_tick: self.birth_tick,
            death_tick,
            generation: self.generation,
            age_of_maturity: self.age_of_maturity,
            maturity_tick: self.maturity_tick,
            num_offspring: self.num_offspring,
            total_consumptions: self.total_consumptions,
            total_actions: self.total_actions,
            action_histogram: self.action_histogram.to_vec(),
            utilization: self.utilization,
            food_ahead_ticks: self.food_ahead_ticks,
            fwd_when_food_ahead: self.fwd_when_food_ahead,
            joint_sensory_action: self.joint_sensory_action,
            pre_maturity_actions: self.pre_maturity.action_count,
            post_maturity_actions: self.post_maturity.action_count,
            pre_maturity_action_histogram: self.pre_maturity.action_histogram.to_vec(),
            post_maturity_action_histogram: self.post_maturity.action_histogram.to_vec(),
            pre_maturity_consumptions: self.pre_maturity.consumptions,
            post_maturity_consumptions: self.post_maturity.consumptions,
            pre_maturity_food_ahead_ticks: self.pre_maturity.food_ahead_ticks,
            post_maturity_food_ahead_ticks: self.post_maturity.food_ahead_ticks,
            pre_maturity_fwd_when_food_ahead: self.pre_maturity.fwd_when_food_ahead,
            post_maturity_fwd_when_food_ahead: self.post_maturity.fwd_when_food_ahead,
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
    descendant_population: u32,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            sidecar: IdHashMap::default(),
            tick_aggregates: TickActionAggregates::new(),
            pending_parents: IdHashMap::default(),
            descendant_population: 0,
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
        self.sidecar.insert(
            id,
            OrganismEntry::new(
                id.0,
                parent_id,
                origin,
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
        let action_idx = record.selected_action.index();
        let sensory_bin = sensory_bin(record);
        // Absent sidecar entry means we never saw this organism's birth —
        // skip so origin-bucketed aggregates stay consistent.
        let Some(entry) = self.sidecar.get_mut(&record.organism_id) else {
            return;
        };
        let origin_idx = entry.origin as usize;
        let is_pre_maturity_record = record.age_turns < u64::from(entry.age_of_maturity);
        let bucket = &mut self.tick_aggregates.per_origin[origin_idx];

        bucket.counts[action_idx] = bucket.counts[action_idx].saturating_add(1);
        if record.selected_action.can_fail() {
            entry.contingent_actions = entry.contingent_actions.saturating_add(1);
            if record.action_failed {
                bucket.failed[action_idx] = bucket.failed[action_idx].saturating_add(1);
                entry.failed_actions = entry.failed_actions.saturating_add(1);
            }
        }
        if is_pre_maturity_record {
            bucket.pre_maturity[action_idx] = bucket.pre_maturity[action_idx].saturating_add(1);
        } else {
            bucket.post_maturity[action_idx] = bucket.post_maturity[action_idx].saturating_add(1);
        }

        let consumption_delta = record
            .consumptions_count
            .saturating_sub(entry.total_consumptions);
        entry.total_actions = entry.total_actions.saturating_add(1);
        entry.total_consumptions = record.consumptions_count;
        entry.action_histogram[action_idx] = entry.action_histogram[action_idx].saturating_add(1);
        entry.utilization = record.utilization.clamp(0.0, 1.0);
        let joint_idx = sensory_bin * ACTION_COUNT + action_idx;
        entry.joint_sensory_action[joint_idx] =
            entry.joint_sensory_action[joint_idx].saturating_add(1);
        let stage_summary = if is_pre_maturity_record {
            &mut entry.pre_maturity
        } else {
            &mut entry.post_maturity
        };
        stage_summary.action_count = stage_summary.action_count.saturating_add(1);
        stage_summary.consumptions = stage_summary.consumptions.saturating_add(consumption_delta);
        stage_summary.action_histogram[action_idx] =
            stage_summary.action_histogram[action_idx].saturating_add(1);
        if record.food_visible_at_offset(0) {
            entry.food_ahead_ticks = entry.food_ahead_ticks.saturating_add(1);
            stage_summary.food_ahead_ticks = stage_summary.food_ahead_ticks.saturating_add(1);
            if record.selected_action == ActionType::Forward {
                entry.fwd_when_food_ahead = entry.fwd_when_food_ahead.saturating_add(1);
                stage_summary.fwd_when_food_ahead =
                    stage_summary.fwd_when_food_ahead.saturating_add(1);
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

    pub fn descendant_abs_dopamine(&self, organisms: &[OrganismState]) -> (f64, u32) {
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for organism in organisms {
            if let Some(entry) = self.sidecar.get(&organism.id) {
                if entry.origin == OrganismOrigin::Descendant {
                    sum += f64::from(organism.dopamine.abs());
                    count += 1;
                }
            }
        }
        (sum, count)
    }

    pub fn population_snapshot_rows(
        &self,
        tick: u64,
        organisms: &[OrganismState],
    ) -> Vec<PopulationSnapshotRow> {
        organisms
            .iter()
            .filter_map(|organism| {
                let entry = self.sidecar.get(&organism.id)?;
                Some(PopulationSnapshotRow {
                    tick,
                    organism_id: organism.id.0,
                    parent_id: entry.parent_id,
                    origin: entry.origin.code(),
                    species_id: organism.species_id.0,
                    generation: organism.generation,
                    birth_tick: entry.birth_tick,
                    age_turns: organism.age_turns,
                    age_of_maturity: entry.age_of_maturity,
                    max_organism_age: organism.genome.lifecycle.max_organism_age,
                    num_neurons: organism.genome.topology.num_neurons,
                    synapse_count: organism.brain.synapse_count,
                    contingent_action_count: entry.contingent_actions,
                    failed_action_count: entry.failed_actions,
                })
            })
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
            age_turns: 0,
            utilization: 0.0,
            consumptions_count: 0,
        });
        ledger.record_action(&ActionRecord {
            organism_id: OrganismId(1),
            selected_action: ActionType::Eat,
            action_failed: true,
            food_visible: [false; SensoryReceptor::VISION_RAY_OFFSETS.len()],
            age_turns: 0,
            utilization: 0.0,
            consumptions_count: 0,
        });
        ledger.record_action(&ActionRecord {
            organism_id: OrganismId(1),
            selected_action: ActionType::Attack,
            action_failed: false,
            food_visible: [false; SensoryReceptor::VISION_RAY_OFFSETS.len()],
            age_turns: 0,
            utilization: 0.0,
            consumptions_count: 0,
        });

        let aggregates = ledger.take_tick_aggregates();
        let founder = &aggregates.per_origin[OrganismOrigin::InitialFounder as usize];
        assert_eq!(founder.counts[ActionType::TurnLeft.index()], 1);
        assert_eq!(founder.failed[ActionType::TurnLeft.index()], 0);
        assert_eq!(founder.failed[ActionType::Eat.index()], 1);
        assert_eq!(founder.failed[ActionType::Attack.index()], 0);
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
