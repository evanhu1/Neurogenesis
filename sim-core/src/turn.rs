use crate::brain::{action_index, evaluate_brain, BrainScratch};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
use crate::Simulation;
use sim_types::{
    ActionType, EntityId, FacingDirection, FoodId, FoodState, Occupant, OrganismId, OrganismMove,
    RemovedEntityPosition, SpeciesId, TickDelta,
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

impl TurnSnapshot {
    fn in_bounds(&self, q: i32, r: i32) -> bool {
        q >= 0 && r >= 0 && q < self.world_width && r < self.world_width
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
    removed_positions: Vec<RemovedEntityPosition>,
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
        self.prune_extinct_species();
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
        let mut intents = Vec::with_capacity(snapshot.organism_count);
        let mut scratch = BrainScratch::new();
        for idx in 0..snapshot.organism_count {
            let snapshot_state = snapshot.organism_states[idx];

            let vision_distance = self.organisms[idx].genome.vision_distance;
            let evaluation = evaluate_brain(
                &mut self.organisms[idx],
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
                id: snapshot.organism_ids[idx],
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

        // Track which cells have a winner moving FROM them
        let mut moving_from = vec![false; world_cells];
        for winner in &winners {
            let from_idx = winner.from.1 as usize * w + winner.from.0 as usize;
            moving_from[from_idx] = true;
        }

        let mut resolutions = Vec::with_capacity(winners.len());
        for winner in winners {
            let target_idx = winner.target.1 as usize * w + winner.target.0 as usize;
            let kind = match snapshot.occupancy[target_idx] {
                Some(Occupant::Organism(_)) if moving_from[target_idx] => {
                    // Target cell's occupant is itself moving away
                    MoveResolutionKind::MoveOnly
                }
                Some(Occupant::Organism(occupant)) => MoveResolutionKind::ConsumeOrganism {
                    consumed_organism: occupant,
                },
                Some(Occupant::Food(food_id)) => MoveResolutionKind::ConsumeFood {
                    consumed_food: food_id,
                },
                None => MoveResolutionKind::MoveOnly,
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
        // intents[i] aligns with organisms[i] (both built in sorted ID order)
        for (idx, intent) in intents.iter().enumerate() {
            self.organisms[idx].facing = intent.facing_after_turn;
        }

        let org_count = self.organisms.len();
        let food_count = self.foods.len();
        let mut move_to: Vec<Option<(i32, i32)>> = vec![None; org_count];
        let mut consumed_org = vec![false; org_count];
        let mut consumed_food = vec![false; food_count];
        let mut consumed_energy = vec![0.0_f32; org_count];
        let mut removed_positions = Vec::new();
        let mut consumptions = 0_u64;

        for resolution in resolutions {
            let actor_idx = self.organism_index(resolution.actor);
            move_to[actor_idx] = Some(resolution.to);
            match resolution.kind {
                MoveResolutionKind::ConsumeOrganism { consumed_organism } => {
                    let victim_idx = self.organism_index(consumed_organism);
                    if !consumed_org[victim_idx] {
                        consumed_org[victim_idx] = true;
                        let victim = &self.organisms[victim_idx];
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Organism(consumed_organism),
                            q: victim.q,
                            r: victim.r,
                        });
                        consumed_energy[actor_idx] += self.config.food_energy * 2.0;
                    }
                    consumptions += 1;
                }
                MoveResolutionKind::ConsumeFood {
                    consumed_food: food_id,
                } => {
                    let food_idx = self.food_index(food_id);
                    if !consumed_food[food_idx] {
                        consumed_food[food_idx] = true;
                        let food = &self.foods[food_idx];
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Food(food_id),
                            q: food.q,
                            r: food.r,
                        });
                        consumed_energy[actor_idx] += food.energy;
                    }
                    consumptions += 1;
                }
                MoveResolutionKind::MoveOnly => {}
            }
        }

        let move_energy_cost = self.config.move_action_energy_cost;
        let mut new_organisms = Vec::with_capacity(org_count);
        for (idx, mut organism) in self.organisms.drain(..).enumerate() {
            if consumed_org[idx] {
                continue;
            }
            if let Some((next_q, next_r)) = move_to[idx] {
                organism.q = next_q;
                organism.r = next_r;
                organism.energy -= move_energy_cost;
            }
            if consumed_energy[idx] > 0.0 {
                organism.energy += consumed_energy[idx];
                organism.consumptions_count = organism.consumptions_count.saturating_add(1);
            }
            new_organisms.push(organism);
        }
        self.organisms = new_organisms;

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
            removed_positions,
            food_spawned,
            consumptions,
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
        // Some organisms may have been consumed in commit_phase, so organisms
        // is a subset. Advance intent_idx to find matching intents.
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
