use crate::brain::{action_index, evaluate_brain, ActionSelectionPolicy, BrainScratch};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right, wrap_position};
use crate::plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
#[cfg(feature = "profiling")]
use crate::{profiling, profiling::TurnPhase};
use crate::{PendingActionKind, PendingActionState, Simulation};
use rayon::prelude::*;
use sim_types::{
    ActionType, EntityId, FacingDirection, FoodState, Occupant, OrganismFacing, OrganismId,
    OrganismMove, OrganismState, RemovedEntityPosition, SpeciesId, TickDelta, WorldConfig,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
#[cfg(feature = "profiling")]
use std::time::Instant;

const FOOD_ENERGY_METABOLISM_DIVISOR: f32 = 10000.0;
const RNG_TURN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const RNG_ORGANISM_MIX: u64 = 0xBF58_476D_1CE4_E5B9;
const REPRODUCE_LOCK_DURATION_TURNS: u8 = 2;
const CONSUMPTION_ENERGY_FRACTION: f32 = 0.10;
const FAILED_PREDATION_ACTION_COST_MULTIPLIER: f32 = 10.0;
const RNG_PREY_MIX: u64 = 0xD6E8_FF3A_5A9C_31F1;

#[derive(Clone, Copy)]
struct SnapshotOrganismState {
    q: i32,
    r: i32,
    facing: FacingDirection,
}

#[derive(Clone)]
struct TurnSnapshot {
    world_width: i32,
    organism_count: usize,
    organism_ids: Vec<OrganismId>,
    organism_states: Vec<SnapshotOrganismState>,
}

#[derive(Clone, Copy)]
struct OrganismIntent {
    idx: usize,
    id: OrganismId,
    from: (i32, i32),
    facing_after_actions: FacingDirection,
    wants_move: bool,
    wants_consume: bool,
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
    consume_target: Option<(i32, i32)>,
    move_confidence: f32,
    action_cost_count: u8,
    synapse_ops: u64,
}

#[derive(Clone, Copy)]
struct MoveCandidate {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    target: (i32, i32),
    confidence: f32,
}

#[derive(Clone, Copy)]
struct MoveResolution {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    to: (i32, i32),
}

#[derive(Default)]
struct CommitResult {
    moves: Vec<OrganismMove>,
    facing_updates: Vec<OrganismFacing>,
    removed_positions: Vec<RemovedEntityPosition>,
    food_spawned: Vec<FoodState>,
    consumptions: u64,
    predations: u64,
    actions_applied: u64,
}

impl Simulation {
    pub(crate) fn tick(&mut self) -> TickDelta {
        self.reconcile_pending_actions();

        #[cfg(feature = "profiling")]
        let tick_started = Instant::now();

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let (starvations, starved_removed_positions) = self.lifecycle_phase();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Lifecycle, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let snapshot = self.build_turn_snapshot();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Snapshot, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let intents = self.build_intents(&snapshot);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Intents, phase_started.elapsed());

        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let mut spawn_requests = Vec::new();
        let mut skip_pending_action_decrement = vec![false; snapshot.organism_count];
        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        Self::reproduction_phase(
            &mut self.organisms,
            &mut self.pending_actions,
            &intents,
            &self.occupancy,
            snapshot.world_width,
            self.config.seed_genome_config.starting_energy,
            &mut skip_pending_action_decrement,
        );
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Reproduction, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let resolutions = self.resolve_moves(&snapshot, &self.occupancy, &intents);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::MoveResolution, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let commit = self.commit_phase(
            &snapshot,
            &intents,
            &resolutions,
            &mut skip_pending_action_decrement,
        );
        self.queue_reproduction_completions(
            snapshot.world_width,
            &mut spawn_requests,
            &skip_pending_action_decrement,
        );
        self.apply_post_commit_runtime_weight_updates();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Commit, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.increment_age_for_survivors();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Age, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.enqueue_periodic_injections(&mut spawn_requests);
        let spawned = self.resolve_spawn_requests(&spawn_requests);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Spawn, phase_started.elapsed());
        let reproductions = spawned.len() as u64;

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.prune_extinct_species();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::PruneSpecies, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.debug_assert_consistent_state();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::ConsistencyCheck, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.turn = self.turn.saturating_add(1);
        self.metrics.turns = self.turn;
        self.metrics.synapse_ops_last_turn = synapse_ops;
        self.metrics.actions_applied_last_turn = commit.actions_applied + reproductions;
        self.metrics.consumptions_last_turn = commit.consumptions;
        self.metrics.predations_last_turn = commit.predations;
        self.metrics.total_consumptions += commit.consumptions;
        self.metrics.reproductions_last_turn = reproductions;
        self.metrics.starvations_last_turn = starvations;
        self.refresh_population_metrics();

        let mut removed_positions = commit.removed_positions;
        removed_positions.extend(starved_removed_positions);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::MetricsAndDelta, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        profiling::record_tick_total(tick_started.elapsed());

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            facing_updates: commit.facing_updates,
            removed_positions,
            spawned,
            food_spawned: commit.food_spawned,
            metrics: self.metrics.clone(),
        }
    }

    fn build_turn_snapshot(&self) -> TurnSnapshot {
        let len = self.organisms.len();
        let mut organism_ids = Vec::with_capacity(len);
        let mut organism_states = Vec::with_capacity(len);

        for organism in &self.organisms {
            organism_ids.push(organism.id);
            organism_states.push(SnapshotOrganismState {
                q: organism.q,
                r: organism.r,
                facing: organism.facing,
            });
        }

        TurnSnapshot {
            world_width: self.config.world_width as i32,
            organism_count: len,
            organism_ids,
            organism_states,
        }
    }

    fn reconcile_pending_actions(&mut self) {
        if self.pending_actions.len() != self.organisms.len() {
            self.pending_actions
                .resize(self.organisms.len(), PendingActionState::default());
        }
    }

    fn build_intents(&mut self, snapshot: &TurnSnapshot) -> Vec<OrganismIntent> {
        let occupancy = &self.occupancy;
        let pending_actions = &self.pending_actions;
        let runtime_plasticity_enabled = self.config.runtime_plasticity_enabled;
        let action_selection = ActionSelectionPolicy {
            temperature: self.config.action_temperature,
        };
        let sim_seed = self.seed;
        let tick = self.turn;
        #[cfg(feature = "profiling")]
        let brain_eval_started = Instant::now();
        let intents = self
            .organisms
            .par_iter_mut()
            .enumerate()
            .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                build_intent_for_organism(
                    idx,
                    organism,
                    pending_actions[idx],
                    snapshot.world_width,
                    occupancy,
                    snapshot.organism_states[idx],
                    snapshot.organism_ids[idx],
                    sim_seed,
                    tick,
                    action_selection,
                    runtime_plasticity_enabled,
                    scratch,
                )
            })
            .collect();
        #[cfg(feature = "profiling")]
        profiling::record_brain_eval_total(brain_eval_started.elapsed());
        intents
    }

    fn apply_post_commit_runtime_weight_updates(&mut self) {
        if !self.config.runtime_plasticity_enabled {
            return;
        }

        let food_energy = self.config.food_energy;
        #[cfg(feature = "profiling")]
        let plasticity_started = Instant::now();
        self.organisms.par_iter_mut().for_each(|organism| {
            let passive_energy_baseline =
                organism_metabolism_energy_cost_from_food_energy(food_energy, organism);
            apply_runtime_weight_updates(organism, passive_energy_baseline);
        });
        #[cfg(feature = "profiling")]
        profiling::record_brain_plasticity_total(plasticity_started.elapsed());
    }

    fn resolve_moves(
        &self,
        snapshot: &TurnSnapshot,
        occupancy: &[Option<Occupant>],
        intents: &[OrganismIntent],
    ) -> Vec<MoveResolution> {
        let w = snapshot.world_width as usize;
        let world_cells = occupancy.len();
        let mut best_by_cell: Vec<Option<MoveCandidate>> = vec![None; world_cells];

        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * w + target.0 as usize;
            if occupancy[cell_idx].is_some() {
                continue;
            }
            let candidate = MoveCandidate {
                actor_idx: intent.idx,
                actor_id: intent.id,
                from: intent.from,
                target,
                confidence: intent.move_confidence,
            };
            match &best_by_cell[cell_idx] {
                Some(current)
                    if compare_move_candidates(&candidate, current) != Ordering::Greater => {}
                _ => best_by_cell[cell_idx] = Some(candidate),
            }
        }

        let mut winners: Vec<MoveCandidate> = best_by_cell.into_iter().flatten().collect();
        winners.sort_by_key(|w| w.actor_idx);

        winners
            .into_iter()
            .map(|winner| MoveResolution {
                actor_idx: winner.actor_idx,
                actor_id: winner.actor_id,
                from: winner.from,
                to: winner.target,
            })
            .collect()
    }

    fn commit_phase(
        &mut self,
        snapshot: &TurnSnapshot,
        intents: &[OrganismIntent],
        resolutions: &[MoveResolution],
        skip_pending_action_decrement: &mut Vec<bool>,
    ) -> CommitResult {
        let world_width = snapshot.world_width;
        let world_width_usize = world_width as usize;

        let mut facing_updates = Vec::new();
        let mut actions_applied = 0_u64;
        let action_energy_cost = self.config.move_action_energy_cost;
        for (idx, intent) in intents.iter().enumerate() {
            debug_assert_eq!(intent.idx, idx);
            let organism = &mut self.organisms[idx];
            if organism.facing != intent.facing_after_actions {
                facing_updates.push(OrganismFacing {
                    id: organism.id,
                    facing: intent.facing_after_actions,
                });
            }
            organism.facing = intent.facing_after_actions;
            let action_count = u64::from(intent.action_cost_count);
            actions_applied += action_count;
            if action_count > 0 {
                organism.energy -= action_energy_cost * intent.action_cost_count as f32;
            }
        }

        let org_count = self.organisms.len();
        let food_count = self.foods.len();
        let mut removed_food = vec![false; food_count];
        let mut removed_positions = Vec::new();
        let mut consumptions = 0_u64;
        let mut predations = 0_u64;
        let mut dead_organisms = vec![false; org_count];
        let plant_threshold = self.plant_biomass_threshold();
        for resolution in resolutions {
            let from_idx =
                resolution.from.1 as usize * world_width_usize + resolution.from.0 as usize;
            let to_idx = resolution.to.1 as usize * world_width_usize + resolution.to.0 as usize;
            debug_assert_eq!(
                self.occupancy[from_idx],
                Some(Occupant::Organism(resolution.actor_id))
            );
            debug_assert_eq!(self.occupancy[to_idx], None);

            self.occupancy[from_idx] = None;
            self.occupancy[to_idx] = Some(Occupant::Organism(resolution.actor_id));
            let organism = &mut self.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
        }

        for (idx, intent) in intents.iter().enumerate() {
            if !intent.wants_consume || dead_organisms[idx] {
                continue;
            }
            let Some((target_q, target_r)) = intent.consume_target else {
                continue;
            };
            let target_idx = target_r as usize * world_width_usize + target_q as usize;

            match self.occupancy[target_idx] {
                Some(Occupant::Food(food_id)) => {
                    let Some(food_idx) = food_index_by_id(&self.foods, food_id) else {
                        continue;
                    };
                    if removed_food[food_idx] {
                        continue;
                    }
                    if self.biomass[target_idx] <= plant_threshold {
                        removed_food[food_idx] = true;
                        let food = &self.foods[food_idx];
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Food(food_id),
                            q: food.q,
                            r: food.r,
                        });
                        self.occupancy[target_idx] = None;
                        continue;
                    }

                    let consumed_biomass = self.biomass[target_idx]
                        .max(0.0)
                        .min(self.config.food_energy);
                    if consumed_biomass <= 0.0 {
                        continue;
                    }

                    self.biomass[target_idx] =
                        (self.biomass[target_idx] - consumed_biomass).max(0.0);
                    let predator = &mut self.organisms[idx];
                    predator.energy += consumed_biomass * CONSUMPTION_ENERGY_FRACTION;
                    predator.consumptions_count = predator.consumptions_count.saturating_add(1);
                    consumptions += 1;

                    if self.biomass[target_idx] <= plant_threshold {
                        removed_food[food_idx] = true;
                        let food = &self.foods[food_idx];
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Food(food_id),
                            q: food.q,
                            r: food.r,
                        });
                        self.occupancy[target_idx] = None;
                    }
                }
                Some(Occupant::Organism(prey_id)) => {
                    let Some(prey_idx) = organism_index_by_id(&self.organisms, prey_id) else {
                        continue;
                    };
                    if idx == prey_idx || dead_organisms[prey_idx] {
                        continue;
                    }

                    let success_probability =
                        prey_probability(&self.organisms[idx], &self.organisms[prey_idx]);
                    let success_sample = deterministic_predation_sample(
                        self.seed,
                        self.turn,
                        self.organisms[idx].id,
                        prey_id,
                    );
                    if success_sample <= success_probability {
                        let prey_energy = self.organisms[prey_idx].energy.max(0.0);
                        let gained_energy = prey_energy * CONSUMPTION_ENERGY_FRACTION;
                        let (prey_q, prey_r) =
                            (self.organisms[prey_idx].q, self.organisms[prey_idx].r);

                        if idx < prey_idx {
                            let (left, _) = self.organisms.split_at_mut(prey_idx);
                            let predator = &mut left[idx];
                            predator.energy += gained_energy;
                            predator.consumptions_count =
                                predator.consumptions_count.saturating_add(1);
                        } else {
                            let (left, right) = self.organisms.split_at_mut(idx);
                            let predator = &mut right[0];
                            debug_assert!(prey_idx < left.len());
                            predator.energy += gained_energy;
                            predator.consumptions_count =
                                predator.consumptions_count.saturating_add(1);
                        }

                        dead_organisms[prey_idx] = true;
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Organism(prey_id),
                            q: prey_q,
                            r: prey_r,
                        });
                        self.occupancy[target_idx] = None;
                        consumptions += 1;
                        predations += 1;
                    } else {
                        self.organisms[idx].energy -=
                            action_energy_cost * (FAILED_PREDATION_ACTION_COST_MULTIPLIER - 1.0);
                    }
                }
                None => {}
                Some(Occupant::Wall) => {}
            }
        }

        if dead_organisms.iter().any(|dead| *dead) {
            let mut survivors = Vec::with_capacity(self.organisms.len());
            let mut survivor_pending_actions = Vec::with_capacity(self.pending_actions.len());
            let mut survivor_skip = Vec::with_capacity(skip_pending_action_decrement.len());
            for (idx, (organism, pending_action)) in self
                .organisms
                .drain(..)
                .zip(self.pending_actions.drain(..))
                .enumerate()
            {
                if !dead_organisms[idx] {
                    survivors.push(organism);
                    survivor_pending_actions.push(pending_action);
                    survivor_skip.push(skip_pending_action_decrement[idx]);
                }
            }
            self.organisms = survivors;
            self.pending_actions = survivor_pending_actions;
            *skip_pending_action_decrement = survivor_skip;
        }

        let mut new_foods = Vec::with_capacity(food_count);
        for (idx, food) in self.foods.drain(..).enumerate() {
            if !removed_food[idx] {
                new_foods.push(food);
            }
        }
        self.foods = new_foods;

        let food_spawned = self.replenish_food_supply();

        let moves = resolutions
            .iter()
            .map(|resolution| OrganismMove {
                id: resolution.actor_id,
                from: resolution.from,
                to: resolution.to,
            })
            .collect();

        CommitResult {
            moves,
            facing_updates,
            removed_positions,
            food_spawned,
            consumptions,
            predations,
            actions_applied,
        }
    }

    fn reproduction_phase(
        organisms: &mut [OrganismState],
        pending_actions: &mut [PendingActionState],
        intents: &[OrganismIntent],
        occupancy: &[Option<Occupant>],
        world_width: i32,
        reproduction_investment_energy: f32,
        skip_pending_action_decrement: &mut [bool],
    ) {
        for intent in intents {
            let org_idx = intent.idx;
            let organism = &mut organisms[org_idx];
            if !intent.wants_reproduce {
                continue;
            }
            if pending_actions[org_idx].turns_remaining > 0 {
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
            pending_actions[org_idx] = PendingActionState {
                kind: PendingActionKind::Reproduce,
                turns_remaining: REPRODUCE_LOCK_DURATION_TURNS,
            };
            skip_pending_action_decrement[org_idx] = true;
        }
    }

    fn queue_reproduction_completions(
        &mut self,
        world_width: i32,
        spawn_requests: &mut Vec<SpawnRequest>,
        skip_pending_action_decrement: &[bool],
    ) {
        let mut reserved_spawn_cells = HashSet::new();

        for (idx, pending_action) in self.pending_actions.iter_mut().enumerate() {
            if pending_action.turns_remaining == 0 {
                pending_action.kind = PendingActionKind::None;
                continue;
            }
            if skip_pending_action_decrement[idx] {
                continue;
            }

            pending_action.turns_remaining = pending_action.turns_remaining.saturating_sub(1);
            if pending_action.turns_remaining > 0 {
                continue;
            }

            if pending_action.kind == PendingActionKind::Reproduce {
                let parent = &self.organisms[idx];
                let (q, r) = reproduction_target(world_width, parent.q, parent.r, parent.facing);
                if occupancy_snapshot_cell(&self.occupancy, world_width, q, r).is_none()
                    && reserved_spawn_cells.insert((q, r))
                {
                    spawn_requests.push(SpawnRequest {
                        kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
                            parent_genome: parent.genome.clone(),
                            parent_species_id: parent.species_id,
                            parent_generation: parent.generation,
                            parent_facing: parent.facing,
                            q,
                            r,
                        }),
                    });
                }
            }

            pending_action.kind = PendingActionKind::None;
        }
    }

    fn lifecycle_phase(&mut self) -> (u64, Vec<RemovedEntityPosition>) {
        let max_age = self.config.max_organism_age as u64;
        let world_width = self.config.world_width as usize;
        let mut dead = vec![false; self.organisms.len()];
        let mut starved_positions = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let metabolism_energy_cost = organism_metabolism_energy_cost(&self.config, organism);
            organism.energy -= metabolism_energy_cost;
            if organism.energy <= 0.0 || organism.age_turns >= max_age {
                dead[idx] = true;
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
                self.occupancy[cell_idx] = None;
                starved_positions.push(RemovedEntityPosition {
                    entity_id: EntityId::Organism(organism.id),
                    q: organism.q,
                    r: organism.r,
                });
            }
        }

        let starvation_count = starved_positions.len() as u64;
        if starvation_count == 0 {
            return (0, starved_positions);
        }

        let mut new_organisms = Vec::with_capacity(self.organisms.len());
        let mut new_pending_actions = Vec::with_capacity(self.pending_actions.len());
        for (idx, (organism, pending_action)) in self
            .organisms
            .drain(..)
            .zip(self.pending_actions.drain(..))
            .enumerate()
        {
            if !dead[idx] {
                new_organisms.push(organism);
                new_pending_actions.push(pending_action);
            }
        }
        self.organisms = new_organisms;
        self.pending_actions = new_pending_actions;

        (starvation_count, starved_positions)
    }
}

fn organism_metabolism_energy_cost(config: &WorldConfig, organism: &OrganismState) -> f32 {
    organism_metabolism_energy_cost_from_food_energy(config.food_energy, organism)
}

fn organism_metabolism_energy_cost_from_food_energy(
    food_energy: f32,
    organism: &OrganismState,
) -> f32 {
    // `num_neurons` tracks enabled interneurons. Sensory neurons are concrete runtime nodes.
    let neuron_count = organism.genome.num_neurons as f32 + organism.brain.sensory.len() as f32;
    let vision_distance_cost = organism.genome.vision_distance as f32 / 3.0;
    let neuron_energy_cost = food_energy / FOOD_ENERGY_METABOLISM_DIVISOR;
    neuron_energy_cost * (neuron_count + vision_distance_cost)
}

impl Simulation {
    fn increment_age_for_survivors(&mut self) {
        for organism in &mut self.organisms {
            organism.age_turns = organism.age_turns.saturating_add(1);
        }
    }

    pub(crate) fn refresh_population_metrics(&mut self) {
        self.metrics.organisms = self.organisms.len() as u32;
        self.metrics.total_species_created = self.next_species_id;
        self.metrics.species_counts = self.compute_species_counts();
    }

    fn compute_species_counts(&self) -> BTreeMap<SpeciesId, u32> {
        let mut species_counts = BTreeMap::new();
        for organism in &self.organisms {
            let count = species_counts.entry(organism.species_id).or_insert(0_u32);
            *count = count.saturating_add(1);
        }
        species_counts
    }
}

fn compare_move_candidates(a: &MoveCandidate, b: &MoveCandidate) -> Ordering {
    a.confidence
        .total_cmp(&b.confidence)
        .then_with(|| b.actor_id.cmp(&a.actor_id))
}

fn build_intent_for_organism(
    idx: usize,
    organism: &mut OrganismState,
    pending_action: PendingActionState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    snapshot_state: SnapshotOrganismState,
    organism_id: OrganismId,
    sim_seed: u64,
    tick: u64,
    action_selection: ActionSelectionPolicy,
    runtime_plasticity_enabled: bool,
    scratch: &mut BrainScratch,
) -> OrganismIntent {
    if pending_action.turns_remaining > 0 {
        return OrganismIntent {
            idx,
            id: organism_id,
            from: (snapshot_state.q, snapshot_state.r),
            facing_after_actions: snapshot_state.facing,
            wants_move: false,
            wants_consume: false,
            wants_reproduce: false,
            move_target: None,
            consume_target: None,
            move_confidence: 0.0,
            action_cost_count: 0,
            synapse_ops: 0,
        };
    }

    let vision_distance = organism.genome.vision_distance;
    let action_sample = deterministic_action_sample(sim_seed, tick, organism_id);
    let evaluation = evaluate_brain(
        organism,
        world_width,
        occupancy,
        vision_distance,
        action_selection,
        action_sample,
        scratch,
    );
    if runtime_plasticity_enabled {
        compute_pending_coactivations(organism, scratch);
    }

    let selected_action = evaluation.resolved_actions.selected_action;
    organism.last_action_taken = selected_action;
    let selected_action_activation = evaluation.action_activations[action_index(selected_action)];
    let (
        facing_after_actions,
        wants_move,
        wants_consume,
        wants_reproduce,
        move_target,
        consume_target,
    ) = intent_from_selected_action(selected_action, snapshot_state, world_width);
    let move_confidence = if wants_move {
        selected_action_activation
    } else {
        0.0
    };

    OrganismIntent {
        idx,
        id: organism_id,
        from: (snapshot_state.q, snapshot_state.r),
        facing_after_actions,
        wants_move,
        wants_consume,
        wants_reproduce,
        move_target,
        consume_target,
        move_confidence,
        action_cost_count: u8::from(selected_action != ActionType::Idle),
        synapse_ops: evaluation.synapse_ops,
    }
}

fn intent_from_selected_action(
    selected_action: ActionType,
    snapshot_state: SnapshotOrganismState,
    world_width: i32,
) -> (
    FacingDirection,
    bool,
    bool,
    bool,
    Option<(i32, i32)>,
    Option<(i32, i32)>,
) {
    let from = (snapshot_state.q, snapshot_state.r);
    let current_facing = snapshot_state.facing;

    match selected_action {
        ActionType::Idle => (current_facing, false, false, false, None, None),
        ActionType::TurnLeft => (rotate_left(current_facing), false, false, false, None, None),
        ActionType::TurnRight => (
            rotate_right(current_facing),
            false,
            false,
            false,
            None,
            None,
        ),
        ActionType::Forward => (
            current_facing,
            true,
            false,
            false,
            Some(hex_neighbor(from, current_facing, world_width)),
            None,
        ),
        ActionType::Consume => (
            current_facing,
            false,
            true,
            false,
            None,
            Some(hex_neighbor(from, current_facing, world_width)),
        ),
        ActionType::Reproduce => (current_facing, false, false, true, None, None),
    }
}

fn occupancy_snapshot_cell(
    occupancy: &[Option<Occupant>],
    world_width: i32,
    q: i32,
    r: i32,
) -> Option<Occupant> {
    let (q, r) = wrap_position((q, r), world_width);
    let idx = r as usize * world_width as usize + q as usize;
    occupancy[idx]
}

fn organism_index_by_id(organisms: &[OrganismState], id: OrganismId) -> Option<usize> {
    organisms.binary_search_by_key(&id, |o| o.id).ok()
}

fn food_index_by_id(foods: &[FoodState], id: sim_types::FoodId) -> Option<usize> {
    foods.binary_search_by_key(&id, |food| food.id).ok()
}

fn reproduction_target(
    world_width: i32,
    parent_q: i32,
    parent_r: i32,
    parent_facing: FacingDirection,
) -> (i32, i32) {
    let opposite_facing = opposite_direction(parent_facing);
    hex_neighbor((parent_q, parent_r), opposite_facing, world_width)
}

fn action_rng_seed(sim_seed: u64, tick: u64, organism_id: OrganismId) -> u64 {
    let mixed =
        sim_seed ^ tick.wrapping_mul(RNG_TURN_MIX) ^ organism_id.0.wrapping_mul(RNG_ORGANISM_MIX);
    mix_u64(mixed)
}

fn deterministic_action_sample(sim_seed: u64, tick: u64, organism_id: OrganismId) -> f32 {
    let sample = (action_rng_seed(sim_seed, tick, organism_id) >> 40) as u32;
    sample as f32 / ((1_u32 << 24) - 1) as f32
}

fn deterministic_predation_sample(
    sim_seed: u64,
    tick: u64,
    predator_id: OrganismId,
    prey_id: OrganismId,
) -> f32 {
    let mixed = sim_seed
        ^ tick.wrapping_mul(RNG_TURN_MIX)
        ^ predator_id.0.wrapping_mul(RNG_ORGANISM_MIX)
        ^ prey_id.0.wrapping_mul(RNG_PREY_MIX);
    let sample = (mix_u64(mixed) >> 40) as u32;
    sample as f32 / ((1_u32 << 24) - 1) as f32
}

fn prey_probability(predator: &OrganismState, prey: &OrganismState) -> f32 {
    let predator_energy = predator.energy.max(0.0);
    let prey_energy = prey.energy.max(0.0);
    let total_energy = predator_energy + prey_energy;
    if total_energy <= f32::EPSILON {
        return 0.0;
    }
    (predator_energy / total_energy).clamp(0.0, 1.0)
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value
}
