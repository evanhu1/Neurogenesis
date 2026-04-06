use super::*;
use sim_types::{ReproductionEvent, ReproductionFailureCause};

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
    skip_pending_action_decrement: Vec<bool>,
}

impl ReproductionPhaseState {
    pub(super) fn new(organism_count: usize) -> Self {
        Self {
            spawn_requests: Vec::new(),
            successful_births: Vec::new(),
            reproduction_events: Vec::new(),
            skip_pending_action_decrement: vec![false; organism_count],
        }
    }

    pub(super) fn skip_pending_action_decrement_mut(&mut self) -> &mut Vec<bool> {
        &mut self.skip_pending_action_decrement
    }

    pub(super) fn spawn_requests_mut(&mut self) -> &mut Vec<SpawnRequest> {
        &mut self.spawn_requests
    }

    pub(super) fn extend_reproduction_events(&mut self, events: Vec<ReproductionEvent>) {
        self.reproduction_events.extend(events);
    }

    pub(super) fn finalize_reproduction_events(
        mut self,
        reproduction_spawned: &[OrganismState],
    ) -> Vec<ReproductionEvent> {
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
        self.reproduction_events
    }

    pub(super) fn apply_triggers(
        &mut self,
        organisms: &mut [OrganismState],
        pending_actions: &mut [PendingActionState],
        reward_ledgers: &mut [crate::RewardLedger],
        intents: &[OrganismIntent],
        occupancy: &[Option<Occupant>],
        world_width: i32,
        reproduction_investment_energy: f32,
    ) {
        for intent in intents {
            let org_idx = intent.idx;
            let organism = &mut organisms[org_idx];
            if !intent.wants_reproduce
                || pending_actions[org_idx].kind == PendingActionKind::Reproduce
            {
                continue;
            }

            let parent_energy = organism.energy;
            if parent_energy < reproduction_investment_energy {
                continue;
            }
            let maturity_age = u64::from(organism.genome.age_of_maturity);
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

            organism.energy -= reproduction_investment_energy;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            reward_ledgers[org_idx].record(RewardEvent::ReproductionInvested {
                energy: reproduction_investment_energy,
            });
            pending_actions[org_idx] = PendingActionState {
                kind: PendingActionKind::Reproduce,
                turns_remaining: REPRODUCE_LOCK_DURATION_TURNS,
                reproduction_energy_bits: reproduction_investment_energy.to_bits(),
            };
            self.skip_pending_action_decrement[org_idx] = true;
        }
    }

    pub(super) fn queue_completions(&mut self, sim: &mut Simulation, world_width: i32) {
        let mut reserved_spawn_cells = HashSet::new();

        for (idx, pending_action) in sim.pending_actions.iter_mut().enumerate() {
            if pending_action.kind != PendingActionKind::Reproduce {
                pending_action.kind = PendingActionKind::None;
                continue;
            }
            if self.skip_pending_action_decrement[idx] {
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
                && reserved_spawn_cells.insert((q, r))
            {
                sim.reward_ledgers[idx].record(RewardEvent::OffspringSpawned {
                    reward: sim.config.food_energy * PLANT_CONSUMPTION_ENERGY_FRACTION,
                });
                self.spawn_requests.push(SpawnRequest {
                    kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
                        parent_genome: parent.genome.clone(),
                        parent_generation: parent.generation,
                        parent_species_id: parent.species_id,
                        parent_facing: parent.facing,
                        q,
                        r,
                    }),
                });
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

            *pending_action = PendingActionState::default();
        }
    }
}
