use crate::brain::{action_index, evaluate_brain, move_confidence_signal};
use crate::grid::{hex_neighbor, rotate_left, rotate_right};
use crate::Simulation;
use crate::{SpawnRequest, SpawnRequestKind};
use sim_protocol::{
    ActionType, EvolutionStats, FacingDirection, OrganismId, OrganismMove, OrganismState,
    RemovedOrganismPosition, TickDelta,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct SnapshotOrganismState {
    q: i32,
    r: i32,
    facing: FacingDirection,
    turns_since_last_meal: u32,
    move_confidence: f32,
}

#[derive(Clone)]
struct TurnSnapshot {
    world_width: i32,
    occupancy: Vec<Option<OrganismId>>,
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

    fn occupant_at(&self, q: i32, r: i32) -> Option<OrganismId> {
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
    EatAndReplace { prey: OrganismId },
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
    meals: u64,
    eaters: HashSet<OrganismId>,
}

impl Simulation {
    pub(crate) fn tick(&mut self) -> TickDelta {
        let snapshot = self.build_turn_snapshot();
        let intents = self.build_intents(&snapshot);
        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let resolutions = self.resolve_moves(&snapshot, &intents);
        let mut spawn_requests = Vec::new();
        let commit = self.commit_phase(&intents, &resolutions, &mut spawn_requests);
        let (starvations, starved_removed_positions) =
            self.lifecycle_phase(&commit.eaters, &mut spawn_requests);
        self.increment_age_for_survivors();
        let spawned = self.resolve_spawn_requests(&spawn_requests);
        let births = spawned.len() as u64;
        self.debug_assert_consistent_state();

        self.turn = self.turn.saturating_add(1);
        self.metrics.turns = self.turn;
        self.metrics.synapse_ops_last_turn = synapse_ops;
        self.metrics.actions_applied_last_turn = commit.moves.len() as u64;
        self.metrics.meals_last_turn = commit.meals;
        self.metrics.starvations_last_turn = starvations;
        self.metrics.births_last_turn = births;
        self.refresh_population_metrics();

        let mut removed_positions = commit.removed_positions;
        removed_positions.extend(starved_removed_positions);

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            removed_positions,
            spawned,
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
                turns_since_last_meal: organism.turns_since_last_meal,
                move_confidence: move_confidence_signal(&organism.brain),
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
        for organism_id in &snapshot.ordered_ids {
            let Some(snapshot_state) = snapshot.organism(*organism_id) else {
                continue;
            };
            let Some(organism_idx) = index_by_id.get(organism_id).copied() else {
                continue;
            };
            let _turns_since_last_meal = snapshot_state.turns_since_last_meal;

            let evaluation = {
                let brain = &mut self.organisms[organism_idx].brain;
                evaluate_brain(
                    brain,
                    (snapshot_state.q, snapshot_state.r),
                    snapshot_state.facing,
                    *organism_id,
                    snapshot.world_width,
                    &snapshot.occupancy,
                )
            };

            let turn_left_active = evaluation.actions[action_index(ActionType::TurnLeft)];
            let turn_right_active = evaluation.actions[action_index(ActionType::TurnRight)];
            let facing_after_turn =
                facing_after_turn(snapshot_state.facing, turn_left_active, turn_right_active);
            let wants_move = evaluation.actions[action_index(ActionType::MoveForward)];
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
                move_target,
                move_confidence: snapshot_state.move_confidence,
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
                Some(occupant) if !moving_ids.contains(&occupant) => {
                    MoveResolutionKind::EatAndReplace { prey: occupant }
                }
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
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> CommitResult {
        let intent_by_id: HashMap<OrganismId, OrganismIntent> =
            intents.iter().map(|intent| (intent.id, *intent)).collect();
        for organism in &mut self.organisms {
            if let Some(intent) = intent_by_id.get(&organism.id) {
                organism.facing = intent.facing_after_turn;
            }
        }

        let mut move_by_actor: HashMap<OrganismId, (i32, i32)> = HashMap::new();
        let mut prey_kills = HashSet::new();
        let positions_by_id: HashMap<OrganismId, (i32, i32)> = self
            .organisms
            .iter()
            .map(|organism| (organism.id, (organism.q, organism.r)))
            .collect();
        let mut removed_positions = Vec::new();
        let mut eaters = HashSet::new();
        let mut meals = 0_u64;

        for resolution in resolutions {
            move_by_actor.insert(resolution.actor, resolution.to);
            if let MoveResolutionKind::EatAndReplace { prey } = resolution.kind {
                if prey_kills.insert(prey) {
                    if let Some((q, r)) = positions_by_id.get(&prey).copied() {
                        removed_positions.push(RemovedOrganismPosition { id: prey, q, r });
                    }
                }
                eaters.insert(resolution.actor);
                meals += 1;
                spawn_requests.push(SpawnRequest {
                    kind: SpawnRequestKind::Reproduction {
                        parent: resolution.actor,
                    },
                });
            }
        }

        self.organisms
            .retain(|organism| !prey_kills.contains(&organism.id));

        for organism in &mut self.organisms {
            if let Some((next_q, next_r)) = move_by_actor.get(&organism.id).copied() {
                organism.q = next_q;
                organism.r = next_r;
                if eaters.contains(&organism.id) {
                    organism.turns_since_last_meal = 0;
                    organism.meals_eaten = organism.meals_eaten.saturating_add(1);
                }
            }
        }

        self.rebuild_occupancy();
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
            meals,
            eaters,
        }
    }

    fn lifecycle_phase(
        &mut self,
        eaters: &HashSet<OrganismId>,
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> (u64, Vec<RemovedOrganismPosition>) {
        self.organisms.sort_by_key(|organism| organism.id);

        let mut starved_ids = Vec::new();
        for organism in &mut self.organisms {
            if eaters.contains(&organism.id) {
                continue;
            }
            organism.turns_since_last_meal = organism.turns_since_last_meal.saturating_add(1);
            if organism.turns_since_last_meal >= self.config.turns_to_starve {
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

        for _ in &starved_ids {
            spawn_requests.push(SpawnRequest {
                kind: SpawnRequestKind::StarvationReplacement,
            });
        }
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
        self.metrics.evolution = self.compute_evolution_stats();
    }

    fn compute_evolution_stats(&self) -> EvolutionStats {
        if self.organisms.is_empty() {
            return EvolutionStats::default();
        }

        let mut ages: Vec<u64> = self
            .organisms
            .iter()
            .map(|organism| organism.age_turns)
            .collect();
        ages.sort_unstable();

        let count = ages.len();
        let total_age: u64 = ages.iter().sum();
        let mean_age_turns = total_age as f64 / count as f64;
        let median_age_turns = if count % 2 == 1 {
            ages[count / 2] as f64
        } else {
            (ages[count / 2 - 1] as f64 + ages[count / 2] as f64) / 2.0
        };
        let max_age_turns = ages[count - 1];

        EvolutionStats {
            mean_age_turns,
            median_age_turns,
            max_age_turns,
        }
    }
}

fn compare_move_candidates(a: &MoveCandidate, b: &MoveCandidate) -> Ordering {
    a.confidence
        .total_cmp(&b.confidence)
        .then_with(|| b.actor.cmp(&a.actor))
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
