use crate::brain::{action_index, evaluate_brain, BrainScratch, TurnChoice};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
use crate::Simulation;
use rayon::prelude::*;
use sim_types::{
    ActionType, EntityId, FacingDirection, FoodState, Occupant, OrganismFacing, OrganismId,
    OrganismMove, OrganismState, RemovedEntityPosition, SpeciesId, TickDelta,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Copy)]
struct SnapshotOrganismState {
    q: i32,
    r: i32,
    facing: FacingDirection,
}

#[derive(Clone)]
struct TurnSnapshot {
    world_width: i32,
    occupancy: Vec<Option<Occupant>>,
    organism_count: usize,
    organism_ids: Vec<OrganismId>,
    organism_states: Vec<SnapshotOrganismState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IntentActionKind {
    Turn,
    Move,
    Consume,
}

#[derive(Clone, Copy)]
struct RankedIntentAction {
    kind: IntentActionKind,
    strength: f32,
}

#[derive(Clone, Copy)]
struct OrganismIntent {
    id: OrganismId,
    from: (i32, i32),
    facing_before_actions: FacingDirection,
    facing_after_actions: FacingDirection,
    turn_choice: TurnChoice,
    wants_move: bool,
    wants_consume: bool,
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
    move_confidence: f32,
    ordered_actions: [IntentActionKind; 3],
    ordered_action_count: u8,
    action_cost_count: u8,
    synapse_ops: u64,
}

#[derive(Clone, Copy)]
struct MoveCandidate {
    actor: OrganismId,
    from: (i32, i32),
    target: (i32, i32),
    confidence: f32,
}

#[derive(Clone, Copy)]
struct MoveResolution {
    actor: OrganismId,
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
        let (starvations, starved_removed_positions) = self.lifecycle_phase();
        let snapshot = self.build_turn_snapshot();
        let intents = self.build_intents(&snapshot);
        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let resolutions = self.resolve_moves(&snapshot, &intents);
        let mut spawn_requests = Vec::new();
        let commit = self.commit_phase(&intents, &resolutions);
        let reproductions = self.reproduction_phase(&intents, &mut spawn_requests);
        self.increment_age_for_survivors();
        let spawned = self.resolve_spawn_requests(&spawn_requests);
        self.prune_extinct_species();
        self.debug_assert_consistent_state();

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
            occupancy: self.occupancy.clone(),
            organism_count: len,
            organism_ids,
            organism_states,
        }
    }

    fn build_intents(&mut self, snapshot: &TurnSnapshot) -> Vec<OrganismIntent> {
        if self.should_parallelize_intents(snapshot.organism_count) {
            let intent_threads = self.intent_parallelism();
            let world_width = snapshot.world_width;
            let occupancy = &snapshot.occupancy;
            let organism_ids = &snapshot.organism_ids;
            let organism_states = &snapshot.organism_states;
            return crate::install_with_intent_pool(intent_threads, || {
                self.organisms
                    .par_iter_mut()
                    .enumerate()
                    .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                        build_intent_for_organism(
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
                &mut self.organisms[idx],
                snapshot.world_width,
                &snapshot.occupancy,
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
        intents: &[OrganismIntent],
    ) -> Vec<MoveResolution> {
        let w = snapshot.world_width as usize;
        let world_cells = w * w;
        let mut best_by_cell: Vec<Option<MoveCandidate>> = vec![None; world_cells];

        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * w + target.0 as usize;
            if snapshot.occupancy[cell_idx].is_some() {
                continue;
            }
            let candidate = MoveCandidate {
                actor: intent.id,
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
        winners.sort_by_key(|w| w.actor);

        winners
            .into_iter()
            .map(|winner| MoveResolution {
                actor: winner.actor,
                from: winner.from,
                to: winner.target,
            })
            .collect()
    }

    fn commit_phase(
        &mut self,
        intents: &[OrganismIntent],
        resolutions: &[MoveResolution],
    ) -> CommitResult {
        // intents[i] aligns with organisms[i] (both built in sorted ID order)
        let mut facing_updates = Vec::new();
        let mut actions_applied = 0_u64;
        let action_energy_cost = self.config.move_action_energy_cost;
        for (idx, intent) in intents.iter().enumerate() {
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
        let mut move_to: Vec<Option<(i32, i32)>> = vec![None; org_count];
        for resolution in resolutions {
            let actor_idx = self.organism_index(resolution.actor);
            move_to[actor_idx] = Some(resolution.to);
        }

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            if let Some((next_q, next_r)) = move_to[idx] {
                organism.q = next_q;
                organism.r = next_r;
            }
        }

        self.rebuild_occupancy();

        let mut consumed_food = vec![false; food_count];
        let mut removed_positions = Vec::new();
        let mut consumptions = 0_u64;
        let mut predations = 0_u64;
        for (idx, intent) in intents.iter().enumerate() {
            if !intent.wants_consume {
                continue;
            }

            let Some((target_q, target_r)) = consume_target_for_intent(intent, move_to[idx]) else {
                continue;
            };
            let Some(target_idx) = self.cell_index(target_q, target_r) else {
                continue;
            };

            match self.occupancy[target_idx] {
                Some(Occupant::Food(food_id)) => {
                    let food_idx = self.food_index(food_id);
                    if consumed_food[food_idx] {
                        continue;
                    }
                    consumed_food[food_idx] = true;
                    let food = &self.foods[food_idx];
                    removed_positions.push(RemovedEntityPosition {
                        entity_id: EntityId::Food(food_id),
                        q: food.q,
                        r: food.r,
                    });
                    self.organisms[idx].energy += food.energy;
                    self.organisms[idx].consumptions_count =
                        self.organisms[idx].consumptions_count.saturating_add(1);
                    self.occupancy[target_idx] = None;
                    consumptions += 1;
                }
                Some(Occupant::Organism(prey_id)) => {
                    let prey_idx = self.organism_index(prey_id);
                    let drain = self.organisms[prey_idx]
                        .energy
                        .min(self.config.food_energy)
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
                }
                None => {}
            }
        }

        let mut new_foods = Vec::with_capacity(food_count);
        for (idx, food) in self.foods.drain(..).enumerate() {
            if !consumed_food[idx] {
                new_foods.push(food);
            }
        }
        self.foods = new_foods;

        self.rebuild_occupancy();
        let food_spawned = self.replenish_food_supply();

        let moves = resolutions
            .iter()
            .map(|resolution| OrganismMove {
                id: resolution.actor,
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
        &mut self,
        intents: &[OrganismIntent],
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> u64 {
        let mut reserved_spawn_cells = HashSet::new();
        let mut successful_reproductions = 0_u64;
        let world_width = self.config.world_width as i32;
        let reproduction_energy_cost = self.config.reproduction_energy_cost;
        let occupancy_snapshot = self.occupancy.clone();

        // Merge-iterate: both intents and organisms are sorted by ID.
        // Advance intent_idx to find matching intents.
        let mut intent_idx = 0;
        for org_idx in 0..self.organisms.len() {
            let organism_id = self.organisms[org_idx].id;
            while intent_idx < intents.len() && intents[intent_idx].id < organism_id {
                intent_idx += 1;
            }
            if intent_idx >= intents.len() || intents[intent_idx].id != organism_id {
                continue;
            }
            let intent = &intents[intent_idx];
            if !intent.wants_reproduce {
                continue;
            }

            let parent_energy = self.organisms[org_idx].energy;
            if parent_energy < reproduction_energy_cost {
                continue;
            }
            let parent_q = self.organisms[org_idx].q;
            let parent_r = self.organisms[org_idx].r;
            let parent_facing = self.organisms[org_idx].facing;
            let parent_species_id = self.organisms[org_idx].species_id;
            let parent_genome = self.organisms[org_idx].genome.clone();

            let Some((q, r)) = reproduction_target(world_width, parent_q, parent_r, parent_facing)
            else {
                continue;
            };
            if occupancy_snapshot_cell(&occupancy_snapshot, world_width, q, r).is_some()
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
            let organism = &mut self.organisms[org_idx];
            organism.energy -= reproduction_energy_cost;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            successful_reproductions += 1;
        }

        successful_reproductions
    }

    fn lifecycle_phase(&mut self) -> (u64, Vec<RemovedEntityPosition>) {
        let max_age = self.config.max_organism_age as u64;
        let mut dead = vec![false; self.organisms.len()];
        let mut starved_positions = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            organism.energy -= self.config.turn_energy_cost;
            if organism.energy <= 0.0 || organism.age_turns >= max_age {
                dead[idx] = true;
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
        self.rebuild_occupancy();

        (starvation_count, starved_positions)
    }
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
        .then_with(|| b.actor.cmp(&a.actor))
}

fn intent_action_priority(action: IntentActionKind) -> u8 {
    match action {
        IntentActionKind::Turn => 0,
        IntentActionKind::Move => 1,
        IntentActionKind::Consume => 2,
    }
}

fn build_intent_for_organism(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    snapshot_state: SnapshotOrganismState,
    organism_id: OrganismId,
    scratch: &mut BrainScratch,
) -> OrganismIntent {
    let vision_distance = organism.genome.vision_distance;
    let evaluation = evaluate_brain(organism, world_width, occupancy, vision_distance, scratch);

    let turn_choice = evaluation.resolved_actions.turn;
    let wants_move = evaluation.resolved_actions.wants_move;
    let wants_consume = evaluation.resolved_actions.wants_consume;
    let wants_reproduce = evaluation.resolved_actions.wants_reproduce;
    let move_confidence = evaluation.action_activations[action_index(ActionType::MoveForward)];
    let turn_strength = evaluation.action_activations[action_index(ActionType::Turn)].abs();
    let consume_strength = evaluation.action_activations[action_index(ActionType::Consume)];

    let mut ordered_actions = [IntentActionKind::Turn; 3];
    let mut ordered_action_count = 0_usize;
    let mut move_target = None;

    if !wants_reproduce {
        let mut ranked_actions = [
            RankedIntentAction {
                kind: IntentActionKind::Turn,
                strength: 0.0,
            },
            RankedIntentAction {
                kind: IntentActionKind::Move,
                strength: 0.0,
            },
            RankedIntentAction {
                kind: IntentActionKind::Consume,
                strength: 0.0,
            },
        ];

        if turn_choice != TurnChoice::None {
            ranked_actions[ordered_action_count] = RankedIntentAction {
                kind: IntentActionKind::Turn,
                strength: turn_strength,
            };
            ordered_action_count += 1;
        }
        if wants_move {
            ranked_actions[ordered_action_count] = RankedIntentAction {
                kind: IntentActionKind::Move,
                strength: move_confidence,
            };
            ordered_action_count += 1;
        }
        if wants_consume {
            ranked_actions[ordered_action_count] = RankedIntentAction {
                kind: IntentActionKind::Consume,
                strength: consume_strength,
            };
            ordered_action_count += 1;
        }

        ranked_actions[..ordered_action_count].sort_by(|a, b| {
            b.strength
                .total_cmp(&a.strength)
                .then_with(|| intent_action_priority(a.kind).cmp(&intent_action_priority(b.kind)))
        });

        for idx in 0..ordered_action_count {
            ordered_actions[idx] = ranked_actions[idx].kind;
        }

        let mut virtual_facing = snapshot_state.facing;
        for action in &ordered_actions[..ordered_action_count] {
            match action {
                IntentActionKind::Turn => {
                    virtual_facing = facing_after_turn(virtual_facing, turn_choice);
                }
                IntentActionKind::Move => {
                    let target = hex_neighbor((snapshot_state.q, snapshot_state.r), virtual_facing);
                    move_target = (target.0 >= 0
                        && target.1 >= 0
                        && target.0 < world_width
                        && target.1 < world_width)
                        .then_some(target);
                    break;
                }
                IntentActionKind::Consume => {}
            }
        }
    }

    let facing_after_actions = facing_after_turn(snapshot_state.facing, turn_choice);

    OrganismIntent {
        id: organism_id,
        from: (snapshot_state.q, snapshot_state.r),
        facing_before_actions: snapshot_state.facing,
        facing_after_actions,
        turn_choice,
        wants_move,
        wants_consume,
        wants_reproduce,
        move_target,
        move_confidence,
        ordered_actions,
        ordered_action_count: ordered_action_count as u8,
        action_cost_count: ordered_action_count as u8,
        synapse_ops: evaluation.synapse_ops,
    }
}

fn consume_target_for_intent(
    intent: &OrganismIntent,
    move_to: Option<(i32, i32)>,
) -> Option<(i32, i32)> {
    if !intent.wants_consume {
        return None;
    }

    let mut current_pos = intent.from;
    let mut current_facing = intent.facing_before_actions;

    for action in &intent.ordered_actions[..intent.ordered_action_count as usize] {
        match action {
            IntentActionKind::Turn => {
                current_facing = facing_after_turn(current_facing, intent.turn_choice);
            }
            IntentActionKind::Move => {
                if let Some(to) = move_to {
                    current_pos = to;
                }
            }
            IntentActionKind::Consume => {
                return Some(hex_neighbor(current_pos, current_facing));
            }
        }
    }

    None
}

fn occupancy_snapshot_cell(
    occupancy: &[Option<Occupant>],
    world_width: i32,
    q: i32,
    r: i32,
) -> Option<Occupant> {
    if q < 0 || r < 0 || q >= world_width || r >= world_width {
        return None;
    }
    let idx = r as usize * world_width as usize + q as usize;
    occupancy[idx]
}

fn reproduction_target(
    world_width: i32,
    parent_q: i32,
    parent_r: i32,
    parent_facing: FacingDirection,
) -> Option<(i32, i32)> {
    let opposite_facing = opposite_direction(parent_facing);
    let (q, r) = hex_neighbor((parent_q, parent_r), opposite_facing);
    (q >= 0 && r >= 0 && q < world_width && r < world_width).then_some((q, r))
}

pub(crate) fn facing_after_turn(
    current: FacingDirection,
    turn_choice: TurnChoice,
) -> FacingDirection {
    match turn_choice {
        TurnChoice::None => current,
        TurnChoice::Left => rotate_left(current),
        TurnChoice::Right => rotate_right(current),
    }
}
