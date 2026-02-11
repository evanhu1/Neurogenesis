use crate::brain::{action_index, evaluate_brain, BrainScratch};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
use crate::{CellEntity, Simulation};
use sim_protocol::{
    ActionType, FacingDirection, FoodId, FoodState, OrganismId, OrganismMove, OrganismState,
    RemovedFoodPosition, RemovedOrganismPosition, SpeciesId, TickDelta,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy)]
struct SnapshotOrganismState {
    q: i32,
    r: i32,
    facing: FacingDirection,
}

#[derive(Clone)]
struct TurnSnapshot {
    world_width: i32,
    occupancy: Vec<Option<CellEntity>>,
    ordered_ids: Vec<OrganismId>,
    organism_states: Vec<SnapshotOrganismState>,
    id_to_index: HashMap<OrganismId, usize>,
}

impl TurnSnapshot {
    fn organism(&self, id: OrganismId) -> Option<SnapshotOrganismState> {
        let idx = self.id_to_index.get(&id)?;
        self.organism_states.get(*idx).copied()
    }

    fn in_bounds(&self, q: i32, r: i32) -> bool {
        q >= 0 && r >= 0 && q < self.world_width && r < self.world_width
    }

    fn cell_index(&self, q: i32, r: i32) -> Option<usize> {
        if !self.in_bounds(q, r) {
            return None;
        }
        Some(r as usize * self.world_width as usize + q as usize)
    }

    fn occupant_at(&self, q: i32, r: i32) -> Option<CellEntity> {
        let idx = self.cell_index(q, r)?;
        self.occupancy[idx]
    }
}

#[derive(Clone, Copy)]
struct OrganismIntent {
    id: OrganismId,
    from: (i32, i32),
    facing_after_turn: FacingDirection,
    wants_move: bool,
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
    move_confidence: f32,
    synapse_ops: u64,
}

#[derive(Clone, Copy)]
struct MoveCandidate {
    actor: OrganismId,
    from: (i32, i32),
    target: (i32, i32),
    confidence: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MoveResolutionKind {
    MoveOnly,
    ConsumeOrganism { consumed_organism: OrganismId },
    ConsumeFood { consumed_food: FoodId },
}

#[derive(Clone, Copy)]
struct MoveResolution {
    actor: OrganismId,
    from: (i32, i32),
    to: (i32, i32),
    kind: MoveResolutionKind,
}

#[derive(Default)]
struct CommitResult {
    moves: Vec<OrganismMove>,
    removed_positions: Vec<RemovedOrganismPosition>,
    food_removed_positions: Vec<RemovedFoodPosition>,
    food_spawned: Vec<FoodState>,
    consumptions: u64,
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
        self.debug_assert_consistent_state();

        self.turn = self.turn.saturating_add(1);
        self.metrics.turns = self.turn;
        self.metrics.synapse_ops_last_turn = synapse_ops;
        self.metrics.actions_applied_last_turn = commit.moves.len() as u64 + reproductions;
        self.metrics.consumptions_last_turn = commit.consumptions;
        self.metrics.total_consumptions += commit.consumptions;
        self.metrics.reproductions_last_turn = reproductions;
        self.metrics.starvations_last_turn = starvations;
        self.refresh_population_metrics();

        let mut removed_positions = commit.removed_positions;
        removed_positions.extend(starved_removed_positions);

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            removed_positions,
            spawned,
            food_removed_positions: commit.food_removed_positions,
            food_spawned: commit.food_spawned,
            metrics: self.metrics.clone(),
        }
    }

    fn build_turn_snapshot(&self) -> TurnSnapshot {
        let mut ordered: Vec<&OrganismState> = self.organisms.iter().collect();
        ordered.sort_by_key(|organism| organism.id);

        let mut ordered_ids = Vec::with_capacity(ordered.len());
        let mut organism_states = Vec::with_capacity(ordered.len());
        let mut id_to_index = HashMap::with_capacity(ordered.len());

        for (idx, organism) in ordered.into_iter().enumerate() {
            ordered_ids.push(organism.id);
            organism_states.push(SnapshotOrganismState {
                q: organism.q,
                r: organism.r,
                facing: organism.facing,
            });
            id_to_index.insert(organism.id, idx);
        }

        TurnSnapshot {
            world_width: self.config.world_width as i32,
            occupancy: self.occupancy.clone(),
            ordered_ids,
            organism_states,
            id_to_index,
        }
    }

    fn build_intents(&mut self, snapshot: &TurnSnapshot) -> Vec<OrganismIntent> {
        let index_by_id: HashMap<OrganismId, usize> = self
            .organisms
            .iter()
            .enumerate()
            .map(|(idx, organism)| (organism.id, idx))
            .collect();

        let mut intents = Vec::with_capacity(snapshot.ordered_ids.len());
        let mut scratch = BrainScratch::new();
        for organism_id in &snapshot.ordered_ids {
            let Some(snapshot_state) = snapshot.organism(*organism_id) else {
                continue;
            };
            let Some(organism_idx) = index_by_id.get(organism_id).copied() else {
                continue;
            };

            let vision_distance = self
                .species_config(self.organisms[organism_idx].species_id)
                .map(|c| c.vision_distance)
                .unwrap_or(1);
            let evaluation = evaluate_brain(
                &mut self.organisms[organism_idx],
                snapshot.world_width,
                &snapshot.occupancy,
                vision_distance,
                &mut scratch,
            );

            let turn_left_active = evaluation.actions[action_index(ActionType::TurnLeft)];
            let turn_right_active = evaluation.actions[action_index(ActionType::TurnRight)];
            let facing_after_turn =
                facing_after_turn(snapshot_state.facing, turn_left_active, turn_right_active);
            let wants_move = evaluation.actions[action_index(ActionType::MoveForward)];
            let wants_reproduce = evaluation.actions[action_index(ActionType::Reproduce)];
            let move_confidence =
                evaluation.action_activations[action_index(ActionType::MoveForward)];
            let move_target = if wants_move {
                let target = hex_neighbor((snapshot_state.q, snapshot_state.r), facing_after_turn);
                snapshot.in_bounds(target.0, target.1).then_some(target)
            } else {
                None
            };

            intents.push(OrganismIntent {
                id: *organism_id,
                from: (snapshot_state.q, snapshot_state.r),
                facing_after_turn,
                wants_move,
                wants_reproduce,
                move_target,
                move_confidence,
                synapse_ops: evaluation.synapse_ops,
            });
        }
        intents
    }

    fn resolve_moves(
        &self,
        snapshot: &TurnSnapshot,
        intents: &[OrganismIntent],
    ) -> Vec<MoveResolution> {
        let mut contenders: HashMap<(i32, i32), Vec<MoveCandidate>> = HashMap::new();
        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            contenders.entry(target).or_default().push(MoveCandidate {
                actor: intent.id,
                from: intent.from,
                target,
                confidence: intent.move_confidence,
            });
        }

        let mut winners = Vec::with_capacity(contenders.len());
        for contenders_for_target in contenders.into_values() {
            if let Some(winner) = contenders_for_target
                .into_iter()
                .max_by(compare_move_candidates)
            {
                winners.push(winner);
            }
        }
        winners.sort_by_key(|winner| winner.actor);

        let moving_ids: HashSet<OrganismId> = winners.iter().map(|winner| winner.actor).collect();
        let mut resolutions = Vec::with_capacity(winners.len());
        for winner in winners {
            let kind = match snapshot.occupant_at(winner.target.0, winner.target.1) {
                Some(CellEntity::Organism(occupant)) if !moving_ids.contains(&occupant) => {
                    MoveResolutionKind::ConsumeOrganism {
                        consumed_organism: occupant,
                    }
                }
                Some(CellEntity::Food(food_id)) => MoveResolutionKind::ConsumeFood {
                    consumed_food: food_id,
                },
                _ => MoveResolutionKind::MoveOnly,
            };
            resolutions.push(MoveResolution {
                actor: winner.actor,
                from: winner.from,
                to: winner.target,
                kind,
            });
        }

        resolutions
    }

    fn commit_phase(
        &mut self,
        intents: &[OrganismIntent],
        resolutions: &[MoveResolution],
    ) -> CommitResult {
        let intent_by_id: HashMap<OrganismId, OrganismIntent> =
            intents.iter().map(|intent| (intent.id, *intent)).collect();
        for organism in &mut self.organisms {
            if let Some(intent) = intent_by_id.get(&organism.id) {
                organism.facing = intent.facing_after_turn;
            }
        }

        let mut move_by_actor: HashMap<OrganismId, (i32, i32)> = HashMap::new();
        let mut consumed_organism_ids = HashSet::new();
        let positions_by_id: HashMap<OrganismId, (i32, i32)> = self
            .organisms
            .iter()
            .map(|organism| (organism.id, (organism.q, organism.r)))
            .collect();
        let energy_by_id: HashMap<OrganismId, f32> = self
            .organisms
            .iter()
            .map(|organism| (organism.id, organism.energy))
            .collect();
        let positions_by_food_id: HashMap<FoodId, (i32, i32)> = self
            .foods
            .iter()
            .map(|food| (food.id, (food.q, food.r)))
            .collect();
        let energy_by_food_id: HashMap<FoodId, f32> = self
            .foods
            .iter()
            .map(|food| (food.id, food.energy))
            .collect();
        let mut removed_positions = Vec::new();
        let mut food_removed_positions = Vec::new();
        let mut consumed_energy_by_actor: HashMap<OrganismId, f32> = HashMap::new();
        let mut consumed_food_ids = HashSet::new();
        let mut consumptions = 0_u64;

        for resolution in resolutions {
            move_by_actor.insert(resolution.actor, resolution.to);
            match resolution.kind {
                MoveResolutionKind::ConsumeOrganism { consumed_organism } => {
                    if consumed_organism_ids.insert(consumed_organism) {
                        if let Some((q, r)) = positions_by_id.get(&consumed_organism).copied() {
                            removed_positions.push(RemovedOrganismPosition {
                                id: consumed_organism,
                                q,
                                r,
                            });
                        }
                        let consumed_energy =
                            energy_by_id.get(&consumed_organism).copied().unwrap_or(0.0);
                        *consumed_energy_by_actor
                            .entry(resolution.actor)
                            .or_insert(0.0) += consumed_energy;
                    }
                    consumptions += 1;
                }
                MoveResolutionKind::ConsumeFood { consumed_food } => {
                    if consumed_food_ids.insert(consumed_food) {
                        if let Some((q, r)) = positions_by_food_id.get(&consumed_food).copied() {
                            food_removed_positions.push(RemovedFoodPosition {
                                id: consumed_food,
                                q,
                                r,
                            });
                        }
                        let consumed_energy = energy_by_food_id
                            .get(&consumed_food)
                            .copied()
                            .unwrap_or(0.0);
                        *consumed_energy_by_actor
                            .entry(resolution.actor)
                            .or_insert(0.0) += consumed_energy;
                    }
                    consumptions += 1;
                }
                MoveResolutionKind::MoveOnly => {}
            }
        }

        self.organisms
            .retain(|organism| !consumed_organism_ids.contains(&organism.id));
        self.foods
            .retain(|food| !consumed_food_ids.contains(&food.id));

        for organism in &mut self.organisms {
            if let Some((next_q, next_r)) = move_by_actor.get(&organism.id).copied() {
                organism.q = next_q;
                organism.r = next_r;
                organism.energy -= self.config.move_action_energy_cost;
            }
            if let Some(consumed_energy) = consumed_energy_by_actor.get(&organism.id).copied() {
                organism.energy += consumed_energy;
                organism.consumptions_count = organism.consumptions_count.saturating_add(1);
            }
        }

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
            removed_positions,
            food_removed_positions,
            food_spawned,
            consumptions,
        }
    }

    fn reproduction_phase(
        &mut self,
        intents: &[OrganismIntent],
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> u64 {
        let intent_by_id: HashMap<OrganismId, OrganismIntent> =
            intents.iter().map(|intent| (intent.id, *intent)).collect();
        let mut reserved_spawn_cells = HashSet::new();
        let mut successful_reproductions = 0_u64;
        let world_width = self.config.world_width as i32;
        let reproduction_energy_cost = self.config.reproduction_energy_cost;
        let occupancy_snapshot = self.occupancy.clone();

        self.organisms.sort_by_key(|organism| organism.id);
        for idx in 0..self.organisms.len() {
            let organism_id = self.organisms[idx].id;
            let Some(intent) = intent_by_id.get(&organism_id) else {
                continue;
            };
            if !intent.wants_reproduce {
                continue;
            }

            let parent_energy = self.organisms[idx].energy;
            if parent_energy < reproduction_energy_cost {
                continue;
            }
            let parent_q = self.organisms[idx].q;
            let parent_r = self.organisms[idx].r;
            let parent_facing = self.organisms[idx].facing;
            let parent_species_id = self.organisms[idx].species_id;

            let Some((q, r)) = reproduction_target(world_width, parent_q, parent_r, parent_facing)
            else {
                continue;
            };
            if occupancy_snapshot_cell(&occupancy_snapshot, world_width, q, r).is_some()
                || reserved_spawn_cells.contains(&(q, r))
            {
                continue;
            }

            let offspring_species_id = self.species_id_for_reproduction(parent_species_id);
            spawn_requests.push(SpawnRequest {
                kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
                    species_id: offspring_species_id,
                    parent_facing,
                    q,
                    r,
                }),
            });
            reserved_spawn_cells.insert((q, r));
            let organism = &mut self.organisms[idx];
            organism.energy -= reproduction_energy_cost;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            successful_reproductions += 1;
        }

        successful_reproductions
    }

    fn lifecycle_phase(&mut self) -> (u64, Vec<RemovedOrganismPosition>) {
        self.organisms.sort_by_key(|organism| organism.id);

        let max_age = self.config.max_organism_age as u64;
        let mut starved_ids = Vec::new();
        for organism in &mut self.organisms {
            organism.energy -= self.config.turn_energy_cost;
            if organism.energy <= 0.0 || organism.age_turns >= max_age {
                starved_ids.push(organism.id);
            }
        }

        let starved_set: HashSet<OrganismId> = starved_ids.iter().copied().collect();
        let starved_positions = self
            .organisms
            .iter()
            .filter(|organism| starved_set.contains(&organism.id))
            .map(|organism| RemovedOrganismPosition {
                id: organism.id,
                q: organism.q,
                r: organism.r,
            })
            .collect::<Vec<_>>();

        if starved_ids.is_empty() {
            return (0, starved_positions);
        }

        self.organisms
            .retain(|organism| !starved_set.contains(&organism.id));
        self.rebuild_occupancy();

        (starved_ids.len() as u64, starved_positions)
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

fn occupancy_snapshot_cell(
    occupancy: &[Option<CellEntity>],
    world_width: i32,
    q: i32,
    r: i32,
) -> Option<CellEntity> {
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
    turn_left_active: bool,
    turn_right_active: bool,
) -> FacingDirection {
    if turn_left_active ^ turn_right_active {
        if turn_left_active {
            rotate_left(current)
        } else {
            rotate_right(current)
        }
    } else {
        current
    }
}
