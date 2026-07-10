use super::*;
use crate::PendingReproductionState;
use sim_types::{offspring_transfer_energy, ReproductionEvent, ReproductionFailureCause};

#[derive(Clone, Copy)]
struct PendingBirthEvent {
    event: ReproductionEvent,
}

pub(super) struct ReproductionPhaseState {
    spawn_requests: Vec<SpawnRequest>,
    successful_births: Vec<PendingBirthEvent>,
    reproduction_events: Vec<ReproductionEvent>,
    gestation_started_this_tick: Vec<bool>,
}

impl ReproductionPhaseState {
    /// `gestation_started_scratch` is a recycled buffer (see `TurnScratch`);
    /// it is cleared + resized here and handed back by
    /// `finalize_reproduction_events`.
    pub(super) fn new(organism_count: usize, mut gestation_started_scratch: Vec<bool>) -> Self {
        gestation_started_scratch.clear();
        gestation_started_scratch.resize(organism_count, false);
        Self {
            spawn_requests: Vec::new(),
            successful_births: Vec::new(),
            reproduction_events: Vec::new(),
            gestation_started_this_tick: gestation_started_scratch,
        }
    }

    pub(super) fn gestation_started_this_tick_mut(&mut self) -> &mut Vec<bool> {
        &mut self.gestation_started_this_tick
    }

    pub(super) fn spawn_requests_mut(&mut self) -> &mut Vec<SpawnRequest> {
        &mut self.spawn_requests
    }

    pub(super) fn extend_reproduction_events(&mut self, events: Vec<ReproductionEvent>) {
        self.reproduction_events.extend(events);
    }

    /// Returns the finalized events plus the recycled gestation scratch buffer
    /// so the caller can hand it back to `TurnScratch`.
    pub(super) fn finalize_reproduction_events(
        mut self,
        reproduction_spawned: &[OrganismState],
    ) -> (Vec<ReproductionEvent>, Vec<bool>) {
        debug_assert_eq!(self.successful_births.len(), reproduction_spawned.len());
        for (birth, child) in self.successful_births.drain(..).zip(reproduction_spawned) {
            let mut event = birth.event;
            event.child_id = Some(child.id);
            self.reproduction_events.push(event);
        }
        (self.reproduction_events, self.gestation_started_this_tick)
    }

    // Clonal reproduction: an initiator produces an exact copy of its own
    // genome. There is no mate selection, crossover, or in-world mutation —
    // all genetic variation is owned by the generational NEAT outer loop
    // (`crate::evolution`); the world only evaluates fixed genomes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_triggers(
        &mut self,
        organisms: &mut [OrganismState],
        pending_actions: &mut [PendingActionState],
        pending_reproductions: &mut [Option<PendingReproductionState>],
        intents: &[OrganismIntent],
        occupancy: &[Option<Occupant>],
        world_width: i32,
        #[cfg(feature = "instrumentation")] action_records: &mut [Option<
            sim_types::ActionRecord,
        >],
    ) {
        debug_assert_eq!(pending_reproductions.len(), organisms.len());

        for intent in intents {
            let org_idx = intent.idx;
            if !intent.wants_reproduce {
                continue;
            }
            let organism = &organisms[org_idx];
            // Invariant: a reproduce intent can never coincide with an active
            // Reproduce pending. wants_reproduce requires turns_remaining == 0
            // at intent time (locked_intent otherwise), and queue_completions
            // always clears a Reproduce pending within the same tick its
            // counter reaches 0, so one never survives to the next intent
            // phase with turns_remaining == 0.
            debug_assert_ne!(pending_actions[org_idx].kind, PendingActionKind::Reproduce);

            if let Some(cause) = local_reproduction_failure(organism, occupancy, world_width) {
                self.reproduction_events
                    .push(rejected_reproduction_event(organism, cause));
                continue;
            }
            let transfer_energy =
                offspring_transfer_energy(organism.genome.lifecycle.gestation_ticks);

            let base_genome = organism.genome.clone();
            let offspring_generation = organism.generation.saturating_add(1);

            let organism = &mut organisms[org_idx];
            organism.energy -= transfer_energy;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            organism.is_gestating = true;
            pending_actions[org_idx] = PendingActionState {
                kind: PendingActionKind::Reproduce,
                turns_remaining: organism.genome.lifecycle.gestation_ticks,
                reproduction_energy_bits: transfer_energy.to_bits(),
            };
            debug_assert!(pending_reproductions[org_idx].is_none());
            pending_reproductions[org_idx] = Some(PendingReproductionState {
                base_genome,
                offspring_generation,
                parent_age_turns_at_conception: organism.age_turns,
            });
            self.gestation_started_this_tick[org_idx] =
                organism.genome.lifecycle.gestation_ticks > 0;
            #[cfg(feature = "instrumentation")]
            {
                if let Some(Some(record)) = action_records.get_mut(org_idx) {
                    record.action_failed = false;
                }
            }
        }
    }

    pub(super) fn queue_completions(&mut self, sim: &mut Simulation, world_width: i32) {
        // Few completions per tick: a linear scan over a small vec beats hashing.
        let mut reserved_spawn_cells: Vec<(i32, i32)> = Vec::new();

        for idx in 0..sim.pending_actions.len() {
            if sim.pending_actions[idx].kind != PendingActionKind::Reproduce {
                continue;
            }
            if self.gestation_started_this_tick[idx] {
                continue;
            }

            if sim.pending_actions[idx].turns_remaining > 0 {
                sim.pending_actions[idx].turns_remaining =
                    sim.pending_actions[idx].turns_remaining.saturating_sub(1);
            }
            if sim.pending_actions[idx].turns_remaining > 0 {
                continue;
            }

            let pending_action = sim.pending_actions[idx];
            let pending_reproduction = sim.pending_reproductions[idx]
                .take()
                .expect("Reproduce pending action must carry reproduction metadata");
            let parent = &sim.organisms[idx];
            let (q, r) = reproduction_target(world_width, parent.q, parent.r, parent.facing);
            let event = pending_reproduction.event(
                parent,
                pending_action.reproduction_energy(),
                None,
                None,
            );
            if occupancy_snapshot_cell(&sim.occupancy, world_width, q, r).is_none()
                && !reserved_spawn_cells.contains(&(q, r))
            {
                reserved_spawn_cells.push((q, r));
                self.spawn_requests
                    .push(SpawnRequest::Reproduction(Box::new(ReproductionSpawn {
                        offspring_genome: pending_reproduction.base_genome,
                        offspring_generation: pending_reproduction.offspring_generation,
                        parent_species_id: parent.species_id,
                        parent_facing: parent.facing,
                        offspring_starting_energy: pending_action.reproduction_energy(),
                        q,
                        r,
                    })));
                self.successful_births.push(PendingBirthEvent { event });
            } else {
                let mut event = event;
                event.failure_cause = Some(ReproductionFailureCause::BlockedBirth);
                self.reproduction_events.push(event);
            }

            sim.organisms[idx].is_gestating = false;
            sim.pending_actions[idx] = PendingActionState::default();
        }
    }
}

fn local_reproduction_failure(
    organism: &OrganismState,
    occupancy: &[Option<Occupant>],
    world_width: i32,
) -> Option<ReproductionFailureCause> {
    let transfer_energy = offspring_transfer_energy(organism.genome.lifecycle.gestation_ticks);
    if organism.energy < transfer_energy {
        return Some(ReproductionFailureCause::InsufficientEnergy);
    }
    if organism.age_turns < u64::from(organism.genome.lifecycle.age_of_maturity) {
        return Some(ReproductionFailureCause::Immature);
    }
    let (q, r) = reproduction_target(world_width, organism.q, organism.r, organism.facing);
    matches!(
        occupancy_snapshot_cell(occupancy, world_width, q, r),
        Some(Occupant::Wall)
    )
    .then_some(ReproductionFailureCause::BirthTargetBlockedByWall)
}

fn rejected_reproduction_event(
    organism: &OrganismState,
    cause: ReproductionFailureCause,
) -> ReproductionEvent {
    ReproductionEvent {
        parent_id: organism.id,
        parent_species_id: organism.species_id,
        parent_age_turns: organism.age_turns,
        parent_generation: organism.generation,
        investment_energy: 0.0,
        parent_energy_after_event: organism.energy,
        child_id: None,
        failure_cause: Some(cause),
    }
}
