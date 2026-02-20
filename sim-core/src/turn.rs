use crate::brain::{action_index, apply_runtime_plasticity, evaluate_brain, BrainScratch};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right, wrap_position};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
use crate::Simulation;
#[cfg(feature = "profiling")]
use crate::{profiling, profiling::TurnPhase};
use rayon::prelude::*;
use sim_types::{
    ActionType, EntityId, FacingDirection, FoodState, Occupant, OrganismFacing, OrganismId,
    OrganismMove, OrganismState, RemovedEntityPosition, SpeciesId, TickDelta, WorldConfig,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
#[cfg(feature = "profiling")]
use std::time::Instant;

const FOOD_ENERGY_METABOLISM_DIVISOR: f32 = 100.0;

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
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
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
        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let successful_reproduction = Self::reproduction_phase(
            &mut self.organisms,
            &intents,
            &self.occupancy,
            snapshot.world_width,
            self.config.reproduction_energy_cost,
            &mut spawn_requests,
        );
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Reproduction, phase_started.elapsed());

        let reproductions = successful_reproduction
            .iter()
            .filter(|reproduced| **reproduced)
            .count() as u64;
        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let resolutions = self.resolve_moves(
            &snapshot,
            &self.occupancy,
            &intents,
            &successful_reproduction,
        );
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::MoveResolution, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let commit = self.commit_phase(&snapshot, &intents, &resolutions, &successful_reproduction);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Commit, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.increment_age_for_survivors();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Age, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let spawned = self.resolve_spawn_requests(&spawn_requests);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Spawn, phase_started.elapsed());

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

    fn build_intents(&mut self, snapshot: &TurnSnapshot) -> Vec<OrganismIntent> {
        let occupancy = &self.occupancy;

        if self.should_parallelize_intents(snapshot.organism_count) {
            let intent_threads = self.intent_parallelism();
            let world_width = snapshot.world_width;
            let organism_ids = &snapshot.organism_ids;
            let organism_states = &snapshot.organism_states;
            return crate::install_with_intent_pool(intent_threads, || {
                self.organisms
                    .par_iter_mut()
                    .enumerate()
                    .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                        build_intent_for_organism(
                            idx,
                            organism,
                            world_width,
                            occupancy,
                            organism_states[idx],
                            organism_ids[idx],
                            scratch,
                        )
                    })
                    .collect()
            });
        }

        let mut intents = Vec::with_capacity(snapshot.organism_count);
        let mut scratch = BrainScratch::new();
        for idx in 0..snapshot.organism_count {
            intents.push(build_intent_for_organism(
                idx,
                &mut self.organisms[idx],
                snapshot.world_width,
                occupancy,
                snapshot.organism_states[idx],
                snapshot.organism_ids[idx],
                &mut scratch,
            ));
        }
        intents
    }

    fn resolve_moves(
        &self,
        snapshot: &TurnSnapshot,
        occupancy: &[Option<Occupant>],
        intents: &[OrganismIntent],
        successful_reproduction: &[bool],
    ) -> Vec<MoveResolution> {
        let w = snapshot.world_width as usize;
        let world_cells = occupancy.len();
        let mut best_by_cell: Vec<Option<MoveCandidate>> = vec![None; world_cells];

        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            if successful_reproduction[intent.idx] {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * w + target.0 as usize;
            if matches!(
                occupancy[cell_idx],
                Some(Occupant::Organism(_)) | Some(Occupant::Wall)
            ) {
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
        successful_reproduction: &[bool],
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
        let mut bite_targets = vec![None; org_count];
        for (idx, intent) in intents.iter().enumerate() {
            if !intent.wants_move || successful_reproduction[idx] {
                continue;
            }
            let Some((target_q, target_r)) = intent.move_target else {
                continue;
            };
            let target_idx = target_r as usize * world_width_usize + target_q as usize;
            if let Some(Occupant::Organism(prey_id)) = self.occupancy[target_idx] {
                bite_targets[idx] = Some(prey_id);
            }
        }

        let mut move_to: Vec<Option<(i32, i32)>> = vec![None; org_count];
        let mut consumed_food = vec![false; food_count];
        let mut removed_positions = Vec::new();
        let mut consumptions = 0_u64;
        let mut predations = 0_u64;
        let mut dead_organisms = vec![false; org_count];
        for resolution in resolutions {
            move_to[resolution.actor_idx] = Some(resolution.to);
            let from_idx =
                resolution.from.1 as usize * world_width_usize + resolution.from.0 as usize;
            let to_idx = resolution.to.1 as usize * world_width_usize + resolution.to.0 as usize;
            debug_assert_eq!(
                self.occupancy[from_idx],
                Some(Occupant::Organism(resolution.actor_id))
            );
            debug_assert!(!matches!(
                self.occupancy[to_idx],
                Some(Occupant::Organism(_)) | Some(Occupant::Wall)
            ));

            if let Some(Occupant::Food(food_id)) = self.occupancy[to_idx] {
                if let Some(food_idx) = food_index_by_id(&self.foods, food_id) {
                    consumed_food[food_idx] = true;
                    let food = &self.foods[food_idx];
                    removed_positions.push(RemovedEntityPosition {
                        entity_id: EntityId::Food(food_id),
                        q: food.q,
                        r: food.r,
                    });
                    self.organisms[resolution.actor_idx].energy += food.energy;
                    self.organisms[resolution.actor_idx].consumptions_count = self.organisms
                        [resolution.actor_idx]
                        .consumptions_count
                        .saturating_add(1);
                    consumptions += 1;
                }
            }

            self.occupancy[from_idx] = None;
            self.occupancy[to_idx] = Some(Occupant::Organism(resolution.actor_id));
            let organism = &mut self.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
        }

        for (idx, intent) in intents.iter().enumerate() {
            if !intent.wants_move || dead_organisms[idx] {
                continue;
            }
            if successful_reproduction[idx] {
                continue;
            }
            if move_to[idx].is_some() {
                continue;
            }
            let Some((target_q, target_r)) = intent.move_target else {
                continue;
            };
            let Some(prey_id) = bite_targets[idx] else {
                continue;
            };
            let target_idx = target_r as usize * world_width_usize + target_q as usize;

            match self.occupancy[target_idx] {
                Some(Occupant::Organism(current_prey_id)) if current_prey_id == prey_id => {
                    let Some(prey_idx) = organism_index_by_id(&self.organisms, prey_id) else {
                        continue;
                    };
                    if idx == prey_idx || dead_organisms[prey_idx] {
                        continue;
                    }

                    let drain = self.organisms[prey_idx]
                        .energy
                        .min(self.config.food_energy * 2.0)
                        .max(0.0);
                    if drain <= 0.0 {
                        continue;
                    }

                    if idx < prey_idx {
                        let (left, right) = self.organisms.split_at_mut(prey_idx);
                        let predator = &mut left[idx];
                        let prey = &mut right[0];
                        prey.energy -= drain;
                        predator.energy += drain;
                        predator.consumptions_count = predator.consumptions_count.saturating_add(1);
                    } else if idx > prey_idx {
                        let (left, right) = self.organisms.split_at_mut(idx);
                        let prey = &mut left[prey_idx];
                        let predator = &mut right[0];
                        prey.energy -= drain;
                        predator.energy += drain;
                        predator.consumptions_count = predator.consumptions_count.saturating_add(1);
                    } else {
                        continue;
                    }

                    consumptions += 1;
                    predations += 1;

                    if self.organisms[prey_idx].energy <= 0.0 && !dead_organisms[prey_idx] {
                        dead_organisms[prey_idx] = true;
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Organism(prey_id),
                            q: self.organisms[prey_idx].q,
                            r: self.organisms[prey_idx].r,
                        });
                        self.occupancy[target_idx] = None;
                    }
                }
                None => {}
                Some(Occupant::Food(_)) => {}
                Some(Occupant::Organism(_)) => {}
                Some(Occupant::Wall) => {}
            }
        }

        if dead_organisms.iter().any(|dead| *dead) {
            let mut survivors = Vec::with_capacity(self.organisms.len());
            for (idx, organism) in self.organisms.drain(..).enumerate() {
                if !dead_organisms[idx] {
                    survivors.push(organism);
                }
            }
            self.organisms = survivors;
        }

        let mut new_foods = Vec::with_capacity(food_count);
        for (idx, food) in self.foods.drain(..).enumerate() {
            if !consumed_food[idx] {
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
        intents: &[OrganismIntent],
        occupancy: &[Option<Occupant>],
        world_width: i32,
        reproduction_energy_cost: f32,
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> Vec<bool> {
        let mut reserved_spawn_cells = HashSet::new();
        let mut successful_reproduction = vec![false; organisms.len()];

        for intent in intents {
            let org_idx = intent.idx;
            let organism = &mut organisms[org_idx];
            if !intent.wants_reproduce {
                continue;
            }

            let parent_energy = organism.energy;
            if parent_energy < reproduction_energy_cost {
                continue;
            }
            let maturity_age = u64::from(organism.genome.age_of_maturity);
            if organism.age_turns < maturity_age {
                continue;
            }
            let parent_q = organism.q;
            let parent_r = organism.r;
            let parent_facing = organism.facing;
            let parent_species_id = organism.species_id;
            let parent_genome = organism.genome.clone();

            let (q, r) = reproduction_target(world_width, parent_q, parent_r, parent_facing);
            if occupancy_snapshot_cell(occupancy, world_width, q, r).is_some()
                || reserved_spawn_cells.contains(&(q, r))
            {
                continue;
            }

            spawn_requests.push(SpawnRequest {
                kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
                    parent_genome,
                    parent_species_id,
                    parent_facing,
                    q,
                    r,
                }),
            });
            reserved_spawn_cells.insert((q, r));
            organism.energy -= reproduction_energy_cost;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            successful_reproduction[org_idx] = true;
        }

        successful_reproduction
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
        for (idx, organism) in self.organisms.drain(..).enumerate() {
            if !dead[idx] {
                new_organisms.push(organism);
            }
        }
        self.organisms = new_organisms;

        (starvation_count, starved_positions)
    }
}

fn organism_metabolism_energy_cost(config: &WorldConfig, organism: &OrganismState) -> f32 {
    // `num_neurons` tracks enabled interneurons. Sensory neurons are concrete runtime nodes.
    let neuron_count = organism.genome.num_neurons as f32 + organism.brain.sensory.len() as f32;
    let vision_distance_cost = organism.genome.vision_distance as f32;
    let neuron_energy_cost = config.food_energy / FOOD_ENERGY_METABOLISM_DIVISOR;
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
    world_width: i32,
    occupancy: &[Option<Occupant>],
    snapshot_state: SnapshotOrganismState,
    organism_id: OrganismId,
    scratch: &mut BrainScratch,
) -> OrganismIntent {
    let vision_distance = organism.genome.vision_distance;
    #[cfg(feature = "profiling")]
    let brain_eval_started = Instant::now();
    let evaluation = evaluate_brain(organism, world_width, occupancy, vision_distance, scratch);
    #[cfg(feature = "profiling")]
    profiling::record_brain_eval_total(brain_eval_started.elapsed());

    #[cfg(feature = "profiling")]
    let plasticity_started = Instant::now();
    apply_runtime_plasticity(organism, scratch);
    #[cfg(feature = "profiling")]
    profiling::record_brain_plasticity_total(plasticity_started.elapsed());

    let selected_action = evaluation.resolved_actions.selected_action;
    let selected_action_activation = evaluation.action_activations[action_index(selected_action)];
    let (facing_after_actions, wants_move, wants_reproduce, move_target) =
        intent_from_selected_action(selected_action, snapshot_state, world_width);
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
        wants_reproduce,
        move_target,
        move_confidence,
        action_cost_count: u8::from(selected_action != ActionType::Idle),
        synapse_ops: evaluation.synapse_ops,
    }
}

fn intent_from_selected_action(
    selected_action: ActionType,
    snapshot_state: SnapshotOrganismState,
    world_width: i32,
) -> (FacingDirection, bool, bool, Option<(i32, i32)>) {
    let from = (snapshot_state.q, snapshot_state.r);
    let current_facing = snapshot_state.facing;

    match selected_action {
        ActionType::Idle => (current_facing, false, false, None),
        ActionType::TurnLeft => (rotate_left(current_facing), false, false, None),
        ActionType::TurnRight => (rotate_right(current_facing), false, false, None),
        ActionType::Forward => (
            current_facing,
            true,
            false,
            Some(hex_neighbor(from, current_facing, world_width)),
        ),
        ActionType::TurnLeftForward => {
            let facing = rotate_left(current_facing);
            (
                facing,
                true,
                false,
                Some(hex_neighbor(from, facing, world_width)),
            )
        }
        ActionType::TurnRightForward => {
            let facing = rotate_right(current_facing);
            (
                facing,
                true,
                false,
                Some(hex_neighbor(from, facing, world_width)),
            )
        }
        ActionType::Consume => (current_facing, false, false, None),
        ActionType::Reproduce => (current_facing, false, true, None),
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
