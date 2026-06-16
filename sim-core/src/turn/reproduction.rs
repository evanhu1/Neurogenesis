use super::*;
use sim_types::{offspring_transfer_energy, ReproductionEvent, ReproductionFailureCause};

#[derive(Clone, Copy)]
struct PendingBirthEvent {
    parent_id: OrganismId,
    parent_species_id: sim_types::SpeciesId,
    parent_age_turns: u64,
    parent_generation: u64,
    investment_energy: f32,
    parent_energy_after_event: f32,
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
            self.reproduction_events.push(ReproductionEvent {
                parent_id: birth.parent_id,
                parent_species_id: birth.parent_species_id,
                parent_age_turns: birth.parent_age_turns,
                parent_generation: birth.parent_generation,
                investment_energy: birth.investment_energy,
                parent_energy_after_event: birth.parent_energy_after_event,
                child_id: Some(child.id),
                failure_cause: None,
            });
        }
        (self.reproduction_events, self.gestation_started_this_tick)
    }

    pub(super) fn apply_triggers(
        &mut self,
        organisms: &mut [OrganismState],
        pending_actions: &mut [PendingActionState],
        intents: &[OrganismIntent],
        occupancy: &[Option<Occupant>],
        world_width: i32,
        #[cfg(feature = "instrumentation")] action_records: &mut [Option<
            sim_types::ActionRecord,
        >],
    ) {
        for intent in intents {
            let org_idx = intent.idx;
            let organism = &mut organisms[org_idx];
            if !intent.wants_reproduce {
                continue;
            }
            // Invariant: a reproduce intent can never coincide with an active
            // Reproduce pending. wants_reproduce requires turns_remaining == 0
            // at intent time (locked_intent otherwise), and queue_completions
            // always clears a Reproduce pending within the same tick its
            // counter reaches 0, so one never survives to the next intent
            // phase with turns_remaining == 0.
            debug_assert_ne!(pending_actions[org_idx].kind, PendingActionKind::Reproduce);

            let transfer_energy =
                offspring_transfer_energy(organism.genome.lifecycle.gestation_ticks);
            let parent_energy = organism.energy;
            if parent_energy < transfer_energy {
                continue;
            }
            let maturity_age = u64::from(organism.genome.lifecycle.age_of_maturity);
            if organism.age_turns < maturity_age {
                continue;
            }
            let (q, r) = reproduction_target(world_width, organism.q, organism.r, organism.facing);
            if matches!(
                occupancy_snapshot_cell(occupancy, world_width, q, r),
                Some(Occupant::Wall)
            ) {
                continue;
            }

            organism.energy -= transfer_energy;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            organism.is_gestating = true;
            pending_actions[org_idx] = PendingActionState {
                kind: PendingActionKind::Reproduce,
                turns_remaining: organism.genome.lifecycle.gestation_ticks,
                reproduction_energy_bits: transfer_energy.to_bits(),
            };
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

        for (idx, pending_action) in sim.pending_actions.iter_mut().enumerate() {
            if pending_action.kind != PendingActionKind::Reproduce {
                continue;
            }
            if self.gestation_started_this_tick[idx] {
                continue;
            }

            if pending_action.turns_remaining > 0 {
                pending_action.turns_remaining = pending_action.turns_remaining.saturating_sub(1);
            }
            if pending_action.turns_remaining > 0 {
                continue;
            }

            let parent = &sim.organisms[idx];
            let (q, r) = reproduction_target(world_width, parent.q, parent.r, parent.facing);
            if occupancy_snapshot_cell(&sim.occupancy, world_width, q, r).is_none()
                && !reserved_spawn_cells.contains(&(q, r))
            {
                reserved_spawn_cells.push((q, r));
                self.spawn_requests
                    .push(SpawnRequest::Reproduction(Box::new(ReproductionSpawn {
                        parent_genome: parent.genome.clone(),
                        parent_generation: parent.generation,
                        parent_species_id: parent.species_id,
                        parent_facing: parent.facing,
                        offspring_starting_energy: pending_action.reproduction_energy(),
                        q,
                        r,
                    })));
                self.successful_births.push(PendingBirthEvent {
                    parent_id: parent.id,
                    parent_species_id: parent.species_id,
                    parent_age_turns: parent.age_turns,
                    parent_generation: parent.generation,
                    investment_energy: pending_action.reproduction_energy(),
                    parent_energy_after_event: parent.energy,
                });
            } else {
                self.reproduction_events.push(ReproductionEvent {
                    parent_id: parent.id,
                    parent_species_id: parent.species_id,
                    parent_age_turns: parent.age_turns,
                    parent_generation: parent.generation,
                    investment_energy: pending_action.reproduction_energy(),
                    parent_energy_after_event: parent.energy,
                    child_id: None,
                    failure_cause: Some(ReproductionFailureCause::BlockedBirth),
                });
            }

            sim.organisms[idx].is_gestating = false;
            *pending_action = PendingActionState::default();
        }
    }
}
