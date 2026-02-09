use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, FacingDirection, InterNeuronState, MetricsSnapshot,
    NeuronId, NeuronState, NeuronType, OccupancyCell, OrganismId, OrganismMove, OrganismState,
    SensoryNeuronState, SensoryReceptorType, SynapseEdge, TickDelta, WorldConfig, WorldSnapshot,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use thiserror::Error;

const SYNAPSE_STRENGTH_MAX: f32 = 8.0;
const DEFAULT_BIAS: f32 = 0.0;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct Simulation {
    config: WorldConfig,
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    next_organism_id: u64,
    organisms: Vec<OrganismState>,
    occupancy: Vec<Option<OrganismId>>,
    metrics: MetricsSnapshot,
}

#[derive(Default)]
struct BrainEvaluation {
    actions: [bool; 3],
    synapse_ops: u64,
}

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum SpawnRequestKind {
    StarvationReplacement,
    Reproduction { parent: OrganismId },
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SpawnRequest {
    kind: SpawnRequestKind,
}

#[derive(Default)]
struct CommitResult {
    moves: Vec<OrganismMove>,
    removed: Vec<OrganismId>,
    meals: u64,
    eaters: HashSet<OrganismId>,
}

impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        validate_config(&config)?;

        let capacity = world_capacity(config.world_width);
        let mut sim = Self {
            config,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_organism_id: 0,
            organisms: Vec::new(),
            occupancy: vec![None; capacity],
            metrics: MetricsSnapshot::default(),
        };

        sim.spawn_initial_population();
        sim.metrics.organisms = sim.organisms.len() as u32;
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let mut organisms = self.organisms.clone();
        organisms.sort_by_key(|o| o.id);

        let width = self.config.world_width as usize;
        let mut occupancy = Vec::with_capacity(self.organisms.len());
        for (idx, maybe_id) in self.occupancy.iter().enumerate() {
            if let Some(id) = maybe_id {
                let q = (idx % width) as i32;
                let r = (idx / width) as i32;
                occupancy.push(OccupancyCell {
                    q,
                    r,
                    organism_ids: vec![*id],
                });
            }
        }
        occupancy.sort_by_key(|c| (c.q, c.r));

        WorldSnapshot {
            turn: self.turn,
            rng_seed: self.seed,
            config: self.config.clone(),
            organisms,
            occupancy,
            metrics: self.metrics.clone(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.seed = seed.unwrap_or(self.seed);
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.turn = 0;
        self.next_organism_id = 0;
        self.organisms.clear();
        self.occupancy.fill(None);
        self.metrics = MetricsSnapshot::default();
        self.spawn_initial_population();
        self.metrics.organisms = self.organisms.len() as u32;
    }

    pub fn step_n(&mut self, count: u32) -> Vec<TickDelta> {
        let mut deltas = Vec::with_capacity(count as usize);
        for _ in 0..count {
            deltas.push(self.tick());
        }
        deltas
    }

    pub fn focused_organism(&self, id: OrganismId) -> Option<OrganismState> {
        self.organisms.iter().find(|o| o.id == id).cloned()
    }

    pub fn export_trace_jsonl(&mut self, turns: u32) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(
            serde_json::to_string(&self.snapshot())
                .expect("serialize initial snapshot for trace export"),
        );
        for _ in 0..turns {
            self.tick();
            lines.push(
                serde_json::to_string(&self.snapshot())
                    .expect("serialize turn snapshot for trace export"),
            );
        }
        lines
    }

    pub fn metrics(&self) -> &MetricsSnapshot {
        &self.metrics
    }

    fn tick(&mut self) -> TickDelta {
        let snapshot = self.build_turn_snapshot();
        let intents = self.build_intents(&snapshot);
        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let resolutions = self.resolve_moves(&snapshot, &intents);
        let mut spawn_requests = Vec::new();
        let commit = self.commit_phase(&intents, &resolutions, &mut spawn_requests);
        let (starvations, starved_removed) =
            self.lifecycle_phase(&commit.eaters, &mut spawn_requests);
        let spawned = self.resolve_spawn_requests(&spawn_requests);
        let births = spawned.len() as u64;
        self.debug_assert_consistent_state();

        self.turn = self.turn.saturating_add(1);
        self.metrics.turns = self.turn;
        self.metrics.organisms = self.organisms.len() as u32;
        self.metrics.synapse_ops_last_turn = synapse_ops;
        self.metrics.actions_applied_last_turn = commit.moves.len() as u64;
        self.metrics.meals_last_turn = commit.meals;
        self.metrics.starvations_last_turn = starvations;
        self.metrics.births_last_turn = births;

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            removed: commit.removed.into_iter().chain(starved_removed).collect(),
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

            let facing_after_turn = facing_after_turn(
                snapshot_state.facing,
                evaluation.actions[1],
                evaluation.actions[2],
            );
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
        let mut removed = Vec::new();
        let mut eaters = HashSet::new();
        let mut meals = 0_u64;
        for resolution in resolutions {
            move_by_actor.insert(resolution.actor, resolution.to);
            if let MoveResolutionKind::EatAndReplace { prey } = resolution.kind {
                if prey_kills.insert(prey) {
                    removed.push(prey);
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
            removed,
            meals,
            eaters,
        }
    }

    fn lifecycle_phase(
        &mut self,
        eaters: &HashSet<OrganismId>,
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> (u64, Vec<OrganismId>) {
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

        if starved_ids.is_empty() {
            return (0, Vec::new());
        }

        let starved_set: HashSet<OrganismId> = starved_ids.iter().copied().collect();
        self.organisms
            .retain(|organism| !starved_set.contains(&organism.id));
        self.rebuild_occupancy();

        for _ in &starved_ids {
            spawn_requests.push(SpawnRequest {
                kind: SpawnRequestKind::StarvationReplacement,
            });
        }
        (starved_ids.len() as u64, starved_ids)
    }

    fn resolve_spawn_requests(&mut self, queue: &[SpawnRequest]) -> Vec<OrganismState> {
        let mut spawned = Vec::new();
        for request in queue {
            let Some((q, r)) = self.sample_center_weighted_spawn_position() else {
                continue;
            };

            let id = self.alloc_organism_id();
            let organism = match request.kind {
                SpawnRequestKind::StarvationReplacement => OrganismState {
                    id,
                    q,
                    r,
                    facing: self.random_facing(),
                    turns_since_last_meal: 0,
                    meals_eaten: 0,
                    brain: self.generate_brain(),
                },
                SpawnRequestKind::Reproduction { parent } => {
                    let Some(parent_state) = self
                        .organisms
                        .iter()
                        .find(|organism| organism.id == parent)
                        .cloned()
                    else {
                        continue;
                    };

                    let mut brain = parent_state.brain;
                    self.mutate_brain(&mut brain);
                    OrganismState {
                        id,
                        q,
                        r,
                        facing: parent_state.facing,
                        turns_since_last_meal: 0,
                        meals_eaten: 0,
                        brain,
                    }
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
        }

        self.organisms.sort_by_key(|organism| organism.id);
        spawned
    }

    fn sample_center_weighted_spawn_position(&mut self) -> Option<(i32, i32)> {
        if self.organisms.len() >= world_capacity(self.config.world_width) {
            return None;
        }

        let width = self.config.world_width as i32;
        let attempts = (world_capacity(self.config.world_width) * 4).max(64);
        let center = (width as f64 - 1.0) / 2.0;
        let spread = (self.config.center_spawn_max_fraction
            - self.config.center_spawn_min_fraction)
            .abs()
            .max(0.05);
        let sigma = (width as f64 * f64::from(spread) / 2.0).max(0.5);

        for _ in 0..attempts {
            let (z_q, z_r) = self.sample_standard_normal_pair();
            let q = (center + z_q * sigma).round() as i32;
            let r = (center + z_r * sigma).round() as i32;
            if !self.in_bounds(q, r) {
                continue;
            }
            if self.occupant_at(q, r).is_none() {
                return Some((q, r));
            }
        }

        self.nearest_empty_to_center()
    }

    fn sample_standard_normal_pair(&mut self) -> (f64, f64) {
        let u1 = loop {
            let sample = self.rng.random::<f64>();
            if sample > f64::EPSILON {
                break sample;
            }
        };
        let u2 = self.rng.random::<f64>();

        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }

    fn nearest_empty_to_center(&self) -> Option<(i32, i32)> {
        let width = self.config.world_width as i32;
        let center = (width as f64 - 1.0) / 2.0;

        let mut best: Option<((i32, i32), f64)> = None;
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_some() {
                    continue;
                }
                let distance = (q as f64 - center).powi(2) + (r as f64 - center).powi(2);
                match best {
                    None => best = Some(((q, r), distance)),
                    Some(((best_q, best_r), best_distance))
                        if distance < best_distance
                            || (distance == best_distance && (r, q) < (best_r, best_q)) =>
                    {
                        best = Some(((q, r), distance));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(position, _)| position)
    }

    fn debug_assert_consistent_state(&self) {
        if cfg!(debug_assertions) {
            debug_assert_eq!(
                self.organisms.len(),
                self.occupancy.iter().flatten().count(),
                "occupancy vector count should match organism count",
            );
            for organism in &self.organisms {
                let idx = self
                    .cell_index(organism.q, organism.r)
                    .expect("organism position must remain in bounds");
                debug_assert_eq!(
                    self.occupancy[idx],
                    Some(organism.id),
                    "occupancy must point at organism occupying that cell",
                );
            }
        }
    }

    fn spawn_initial_population(&mut self) {
        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);

        for _ in 0..self.target_population() {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let brain = self.generate_brain();
            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                q,
                r,
                facing,
                turns_since_last_meal: 0,
                meals_eaten: 0,
                brain,
            };
            let added = self.add_organism(organism);
            debug_assert!(added);
        }

        self.organisms.sort_by_key(|o| o.id);
    }

    fn mutate_brain(&mut self, brain: &mut BrainState) {
        if self.rng.random::<f32>() > self.config.mutation_chance {
            return;
        }

        let operations = self.config.mutation_magnitude.max(1.0).round() as usize;
        for _ in 0..operations {
            if self.rng.random::<f32>() < 0.5 {
                self.apply_topology_mutation(brain);
            } else {
                self.apply_synapse_mutation(brain);
            }
        }

        brain.synapse_count = count_synapses(brain) as u32;
    }

    fn apply_topology_mutation(&mut self, brain: &mut BrainState) {
        match self.rng.random_range(0..3) {
            0 => {
                if brain.inter.len() as u32 >= self.config.max_num_neurons {
                    return;
                }

                let next_id = next_inter_neuron_id(brain);
                brain.inter.push(InterNeuronState {
                    neuron: make_neuron(NeuronId(next_id), NeuronType::Inter, self.random_bias()),
                    synapses: Vec::new(),
                });

                let pre_candidates = output_neuron_ids(brain);
                let post_candidates = post_neuron_ids(brain);
                if pre_candidates.is_empty() || post_candidates.is_empty() {
                    return;
                }

                let pre = pre_candidates[self.rng.random_range(0..pre_candidates.len())];
                let post = post_candidates[self.rng.random_range(0..post_candidates.len())];
                let weight = self
                    .rng
                    .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
                let _ = create_synapse(brain, pre, post, weight);
            }
            1 => {
                if brain.inter.is_empty() {
                    return;
                }
                let idx = self.rng.random_range(0..brain.inter.len());
                let removed_id = brain.inter[idx].neuron.neuron_id;
                brain.inter.remove(idx);
                remove_neuron_references(brain, removed_id);
            }
            _ => {
                let mutate_action = self.rng.random::<f32>() < 0.5;
                if mutate_action && !brain.action.is_empty() {
                    let idx = self.rng.random_range(0..brain.action.len());
                    let bias = &mut brain.action[idx].neuron.bias;
                    *bias = (*bias + self.rng.random_range(-1.0..1.0)).clamp(-8.0, 8.0);
                } else if !brain.inter.is_empty() {
                    let idx = self.rng.random_range(0..brain.inter.len());
                    let bias = &mut brain.inter[idx].neuron.bias;
                    *bias = (*bias + self.rng.random_range(-1.0..1.0)).clamp(-8.0, 8.0);
                }
            }
        }
    }

    fn apply_synapse_mutation(&mut self, brain: &mut BrainState) {
        match self.rng.random_range(0..3) {
            0 => {
                let pre_candidates = output_neuron_ids(brain);
                let post_candidates = post_neuron_ids(brain);
                if pre_candidates.is_empty() || post_candidates.is_empty() {
                    return;
                }

                let pre = pre_candidates[self.rng.random_range(0..pre_candidates.len())];
                let post = post_candidates[self.rng.random_range(0..post_candidates.len())];
                let weight = self
                    .rng
                    .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
                let _ = create_synapse(brain, pre, post, weight);
            }
            1 => {
                let outputs = output_neuron_ids(brain);
                if outputs.is_empty() {
                    return;
                }
                let pre = outputs[self.rng.random_range(0..outputs.len())];
                remove_random_synapse(brain, pre, &mut self.rng);
            }
            _ => {
                let outputs = output_neuron_ids(brain);
                if outputs.is_empty() {
                    return;
                }
                let pre = outputs[self.rng.random_range(0..outputs.len())];
                perturb_random_synapse(brain, pre, self.config.mutation_magnitude, &mut self.rng);
            }
        }
    }

    fn random_bias(&mut self) -> f32 {
        self.rng.random_range(-1.0..1.0)
    }

    fn random_facing(&mut self) -> FacingDirection {
        FacingDirection::ALL[self.rng.random_range(0..FacingDirection::ALL.len())]
    }

    fn add_organism(&mut self, organism: OrganismState) -> bool {
        let Some(cell_idx) = self.cell_index(organism.q, organism.r) else {
            return false;
        };
        if self.occupancy[cell_idx].is_some() {
            return false;
        }

        self.occupancy[cell_idx] = Some(organism.id);
        self.organisms.push(organism);
        true
    }

    fn rebuild_occupancy(&mut self) {
        self.occupancy.fill(None);
        for organism in &self.organisms {
            let idx = self
                .cell_index(organism.q, organism.r)
                .expect("organism must remain in bounds");
            debug_assert!(self.occupancy[idx].is_none());
            self.occupancy[idx] = Some(organism.id);
        }
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn occupant_at(&self, q: i32, r: i32) -> Option<OrganismId> {
        let idx = self.cell_index(q, r)?;
        self.occupancy[idx]
    }

    fn in_bounds(&self, q: i32, r: i32) -> bool {
        let w = self.config.world_width as i32;
        q >= 0 && r >= 0 && q < w && r < w
    }

    fn cell_index(&self, q: i32, r: i32) -> Option<usize> {
        if !self.in_bounds(q, r) {
            return None;
        }
        let width = self.config.world_width as usize;
        Some(r as usize * width + q as usize)
    }

    fn target_population(&self) -> usize {
        (self.config.num_organisms as usize).min(world_capacity(self.config.world_width))
    }

    fn empty_positions(&self) -> Vec<(i32, i32)> {
        let width = self.config.world_width as i32;
        let mut out = Vec::new();
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_none() {
                    out.push((q, r));
                }
            }
        }
        out
    }

    fn generate_brain(&mut self) -> BrainState {
        let mut sensory = Vec::new();
        for (idx, receptor_type) in SensoryReceptorType::ALL.into_iter().enumerate() {
            sensory.push(make_sensory_neuron(idx as u32, receptor_type));
        }

        let mut inter = Vec::new();
        for i in 0..self.config.num_neurons {
            inter.push(InterNeuronState {
                neuron: make_neuron(NeuronId(1000 + i), NeuronType::Inter, self.random_bias()),
                synapses: Vec::new(),
            });
        }

        let mut action = Vec::new();
        for (idx, action_type) in ActionType::ALL.into_iter().enumerate() {
            action.push(make_action_neuron(
                2000 + idx as u32,
                action_type,
                self.random_bias(),
            ));
        }

        let mut brain = BrainState {
            sensory,
            inter,
            action,
            synapse_count: 0,
        };

        for _ in 0..self.config.num_synapses {
            let pre_candidates = output_neuron_ids(&brain);
            let post_candidates = post_neuron_ids(&brain);
            if pre_candidates.is_empty() || post_candidates.is_empty() {
                break;
            }

            let pre = pre_candidates[self.rng.random_range(0..pre_candidates.len())];
            let post = post_candidates[self.rng.random_range(0..post_candidates.len())];
            let weight = self
                .rng
                .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
            let _ = create_synapse(&mut brain, pre, post, weight);
        }

        brain.synapse_count = count_synapses(&brain) as u32;
        brain
    }
}

fn world_capacity(width: u32) -> usize {
    width as usize * width as usize
}

fn validate_config(config: &WorldConfig) -> Result<(), SimError> {
    if config.world_width == 0 {
        return Err(SimError::InvalidConfig(
            "world_width must be greater than zero".to_owned(),
        ));
    }
    if config.num_organisms == 0 {
        return Err(SimError::InvalidConfig(
            "num_organisms must be greater than zero".to_owned(),
        ));
    }
    if config.turns_to_starve == 0 {
        return Err(SimError::InvalidConfig(
            "turns_to_starve must be >= 1".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.mutation_chance) {
        return Err(SimError::InvalidConfig(
            "mutation_chance must be within [0, 1]".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.center_spawn_min_fraction)
        || !(0.0..=1.0).contains(&config.center_spawn_max_fraction)
    {
        return Err(SimError::InvalidConfig(
            "center spawn fractions must be within [0, 1]".to_owned(),
        ));
    }
    if config.center_spawn_min_fraction >= config.center_spawn_max_fraction {
        return Err(SimError::InvalidConfig(
            "center_spawn_min_fraction must be less than center_spawn_max_fraction".to_owned(),
        ));
    }
    Ok(())
}

fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::MoveForward => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
    }
}

fn compare_move_candidates(a: &MoveCandidate, b: &MoveCandidate) -> Ordering {
    a.confidence
        .total_cmp(&b.confidence)
        .then_with(|| b.actor.cmp(&a.actor))
}

fn facing_after_turn(
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

fn move_confidence_signal(brain: &BrainState) -> f32 {
    brain
        .inter
        .iter()
        .map(|inter| inter.neuron.activation)
        .max_by(f32::total_cmp)
        .unwrap_or(0.0)
}

fn make_neuron(id: NeuronId, neuron_type: NeuronType, bias: f32) -> NeuronState {
    NeuronState {
        neuron_id: id,
        neuron_type,
        bias,
        activation: 0.0,
        parent_ids: Vec::new(),
    }
}

fn make_sensory_neuron(id: u32, receptor_type: SensoryReceptorType) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, DEFAULT_BIAS),
        receptor_type,
        synapses: Vec::new(),
    }
}

fn make_action_neuron(id: u32, action_type: ActionType, bias: f32) -> ActionNeuronState {
    ActionNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Action, bias),
        action_type,
        is_active: false,
    }
}

fn next_inter_neuron_id(brain: &BrainState) -> u32 {
    brain
        .inter
        .iter()
        .map(|n| n.neuron.neuron_id.0)
        .max()
        .map_or(1000, |id| id + 1)
}

fn evaluate_brain(
    brain: &mut BrainState,
    position: (i32, i32),
    facing: FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<OrganismId>],
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    for action in &mut brain.action {
        action.is_active = false;
    }

    let look = look_sensor_value(position, facing, organism_id, world_width, occupancy);
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = match sensory.receptor_type {
            SensoryReceptorType::Look => look,
        };
    }

    let inter_index: HashMap<NeuronId, usize> = brain
        .inter
        .iter()
        .enumerate()
        .map(|(idx, neuron)| (neuron.neuron.neuron_id, idx))
        .collect();
    let action_index_map: HashMap<NeuronId, usize> = brain
        .action
        .iter()
        .enumerate()
        .map(|(idx, neuron)| (neuron.neuron.neuron_id, idx))
        .collect();

    let mut inter_inputs: Vec<f32> = brain.inter.iter().map(|n| n.neuron.bias).collect();
    let prev_inter: Vec<f32> = brain.inter.iter().map(|n| n.neuron.activation).collect();

    for sensory in &brain.sensory {
        for edge in &sensory.synapses {
            if let Some(idx) = inter_index.get(&edge.post_neuron_id) {
                inter_inputs[*idx] += sensory.neuron.activation * edge.weight;
                result.synapse_ops += 1;
            }
        }
    }

    for (source_idx, inter) in brain.inter.iter().enumerate() {
        let source_activation = prev_inter[source_idx];
        for edge in &inter.synapses {
            if let Some(idx) = inter_index.get(&edge.post_neuron_id) {
                inter_inputs[*idx] += source_activation * edge.weight;
                result.synapse_ops += 1;
            }
        }
    }

    for (idx, neuron) in brain.inter.iter_mut().enumerate() {
        neuron.neuron.activation = inter_inputs[idx].tanh();
    }

    let mut action_inputs: Vec<f32> = brain.action.iter().map(|n| n.neuron.bias).collect();

    for sensory in &brain.sensory {
        for edge in &sensory.synapses {
            if let Some(idx) = action_index_map.get(&edge.post_neuron_id) {
                action_inputs[*idx] += sensory.neuron.activation * edge.weight;
                result.synapse_ops += 1;
            }
        }
    }

    for inter in &brain.inter {
        for edge in &inter.synapses {
            if let Some(idx) = action_index_map.get(&edge.post_neuron_id) {
                action_inputs[*idx] += inter.neuron.activation * edge.weight;
                result.synapse_ops += 1;
            }
        }
    }

    for action in &mut brain.action {
        if let Some(idx) = action_index_map.get(&action.neuron.neuron_id) {
            action.neuron.activation = action_inputs[*idx].tanh();
            action.is_active = action.neuron.activation > 0.0;
            result.actions[action_index(action.action_type)] = action.is_active;
        }
    }

    result
}

fn look_sensor_value(
    position: (i32, i32),
    facing: FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<OrganismId>],
) -> f32 {
    let target = hex_neighbor(position, facing);
    if target.0 < 0 || target.1 < 0 || target.0 >= world_width || target.1 >= world_width {
        return 0.0;
    }

    let idx = target.1 as usize * world_width as usize + target.0 as usize;
    match occupancy[idx] {
        Some(id) if id != organism_id => 1.0,
        _ => 0.0,
    }
}

fn rotate_left(direction: FacingDirection) -> FacingDirection {
    match direction {
        FacingDirection::East => FacingDirection::NorthEast,
        FacingDirection::NorthEast => FacingDirection::NorthWest,
        FacingDirection::NorthWest => FacingDirection::West,
        FacingDirection::West => FacingDirection::SouthWest,
        FacingDirection::SouthWest => FacingDirection::SouthEast,
        FacingDirection::SouthEast => FacingDirection::East,
    }
}

fn rotate_right(direction: FacingDirection) -> FacingDirection {
    match direction {
        FacingDirection::East => FacingDirection::SouthEast,
        FacingDirection::SouthEast => FacingDirection::SouthWest,
        FacingDirection::SouthWest => FacingDirection::West,
        FacingDirection::West => FacingDirection::NorthWest,
        FacingDirection::NorthWest => FacingDirection::NorthEast,
        FacingDirection::NorthEast => FacingDirection::East,
    }
}

fn hex_neighbor(position: (i32, i32), facing: FacingDirection) -> (i32, i32) {
    let (q, r) = position;
    match facing {
        FacingDirection::East => (q + 1, r),
        FacingDirection::NorthEast => (q + 1, r - 1),
        FacingDirection::NorthWest => (q, r - 1),
        FacingDirection::West => (q - 1, r),
        FacingDirection::SouthWest => (q - 1, r + 1),
        FacingDirection::SouthEast => (q, r + 1),
    }
}

fn output_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    brain
        .sensory
        .iter()
        .map(|n| n.neuron.neuron_id)
        .chain(brain.inter.iter().map(|n| n.neuron.neuron_id))
        .collect()
}

fn post_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    brain
        .inter
        .iter()
        .map(|n| n.neuron.neuron_id)
        .chain(brain.action.iter().map(|n| n.neuron.neuron_id))
        .collect()
}

fn count_synapses(brain: &BrainState) -> usize {
    brain
        .sensory
        .iter()
        .map(|s| s.synapses.len())
        .sum::<usize>()
        + brain.inter.iter().map(|i| i.synapses.len()).sum::<usize>()
}

fn get_neuron_mut(brain: &mut BrainState, id: NeuronId) -> Option<&mut NeuronState> {
    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == id) {
        return Some(&mut sensory.neuron);
    }
    if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == id) {
        return Some(&mut inter.neuron);
    }
    if let Some(action) = brain.action.iter_mut().find(|n| n.neuron.neuron_id == id) {
        return Some(&mut action.neuron);
    }
    None
}

fn create_synapse(brain: &mut BrainState, pre: NeuronId, post: NeuronId, weight: f32) -> bool {
    if pre == post {
        return false;
    }

    let duplicate = brain
        .sensory
        .iter()
        .find(|n| n.neuron.neuron_id == pre)
        .map(|n| n.synapses.iter().any(|edge| edge.post_neuron_id == post))
        .or_else(|| {
            brain
                .inter
                .iter()
                .find(|n| n.neuron.neuron_id == pre)
                .map(|n| n.synapses.iter().any(|edge| edge.post_neuron_id == post))
        })
        .unwrap_or(true);

    if duplicate {
        return false;
    }

    let clamped = weight.clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);

    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        sensory.synapses.push(SynapseEdge {
            post_neuron_id: post,
            weight: clamped,
        });
        sensory
            .synapses
            .sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
    } else if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        inter.synapses.push(SynapseEdge {
            post_neuron_id: post,
            weight: clamped,
        });
        inter
            .synapses
            .sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
    } else {
        return false;
    }

    if let Some(post_neuron) = get_neuron_mut(brain, post) {
        if !post_neuron.parent_ids.contains(&pre) {
            post_neuron.parent_ids.push(pre);
            post_neuron.parent_ids.sort();
        }
    }

    true
}

fn remove_neuron_references(brain: &mut BrainState, target: NeuronId) {
    for sensory in &mut brain.sensory {
        sensory
            .synapses
            .retain(|edge| edge.post_neuron_id != target);
        sensory.neuron.parent_ids.retain(|id| *id != target);
    }

    for inter in &mut brain.inter {
        inter.synapses.retain(|edge| edge.post_neuron_id != target);
        inter.neuron.parent_ids.retain(|id| *id != target);
    }

    for action in &mut brain.action {
        action.neuron.parent_ids.retain(|id| *id != target);
    }
}

fn remove_random_synapse<R: Rng + ?Sized>(brain: &mut BrainState, pre: NeuronId, rng: &mut R) {
    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if sensory.synapses.is_empty() {
            return;
        }

        let idx = rng.random_range(0..sensory.synapses.len());
        let post = sensory.synapses[idx].post_neuron_id;
        sensory.synapses.remove(idx);
        if let Some(post_neuron) = get_neuron_mut(brain, post) {
            post_neuron.parent_ids.retain(|id| *id != pre);
        }
        return;
    }

    if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if inter.synapses.is_empty() {
            return;
        }

        let idx = rng.random_range(0..inter.synapses.len());
        let post = inter.synapses[idx].post_neuron_id;
        inter.synapses.remove(idx);
        if let Some(post_neuron) = get_neuron_mut(brain, post) {
            post_neuron.parent_ids.retain(|id| *id != pre);
        }
    }
}

fn perturb_random_synapse<R: Rng + ?Sized>(
    brain: &mut BrainState,
    pre: NeuronId,
    mutation_magnitude: f32,
    rng: &mut R,
) {
    let magnitude = mutation_magnitude.clamp(0.1, 8.0);
    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if sensory.synapses.is_empty() {
            return;
        }

        let idx = rng.random_range(0..sensory.synapses.len());
        let delta = rng.random_range(-magnitude..magnitude);
        sensory.synapses[idx].weight = (sensory.synapses[idx].weight + delta)
            .clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
        return;
    }

    if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if inter.synapses.is_empty() {
            return;
        }

        let idx = rng.random_range(0..inter.synapses.len());
        let delta = rng.random_range(-magnitude..magnitude);
        inter.synapses[idx].weight =
            (inter.synapses[idx].weight + delta).clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
    }
}

pub fn compare_snapshots(a: &WorldSnapshot, b: &WorldSnapshot) -> Ordering {
    let sa = serde_json::to_string(a).expect("serialize snapshot A");
    let sb = serde_json::to_string(b).expect("serialize snapshot B");
    sa.cmp(&sb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn test_config(world_width: u32, num_organisms: u32) -> WorldConfig {
        WorldConfig {
            world_width,
            num_organisms,
            num_neurons: 1,
            num_synapses: 0,
            turns_to_starve: 10,
            mutation_chance: 0.0,
            ..WorldConfig::default()
        }
    }

    fn forced_brain(
        wants_move: bool,
        turn_left: bool,
        turn_right: bool,
        confidence: f32,
    ) -> BrainState {
        let sensory = vec![make_sensory_neuron(0, SensoryReceptorType::Look)];
        let inter = vec![InterNeuronState {
            neuron: NeuronState {
                neuron_id: NeuronId(1000),
                neuron_type: NeuronType::Inter,
                bias: 0.0,
                activation: confidence,
                parent_ids: Vec::new(),
            },
            synapses: Vec::new(),
        }];
        let action = vec![
            make_action_neuron(
                2000,
                ActionType::MoveForward,
                if wants_move { 1.0 } else { -1.0 },
            ),
            make_action_neuron(
                2001,
                ActionType::TurnLeft,
                if turn_left { 1.0 } else { -1.0 },
            ),
            make_action_neuron(
                2002,
                ActionType::TurnRight,
                if turn_right { 1.0 } else { -1.0 },
            ),
        ];

        BrainState {
            sensory,
            inter,
            action,
            synapse_count: 0,
        }
    }

    fn make_organism(
        id: u64,
        q: i32,
        r: i32,
        facing: FacingDirection,
        wants_move: bool,
        turn_left: bool,
        turn_right: bool,
        confidence: f32,
        turns_since_last_meal: u32,
    ) -> OrganismState {
        OrganismState {
            id: OrganismId(id),
            q,
            r,
            facing,
            turns_since_last_meal,
            meals_eaten: 0,
            brain: forced_brain(wants_move, turn_left, turn_right, confidence),
        }
    }

    fn configure_sim(sim: &mut Simulation, mut organisms: Vec<OrganismState>) {
        organisms.sort_by_key(|organism| organism.id);
        sim.organisms = organisms;
        sim.next_organism_id = sim
            .organisms
            .iter()
            .map(|organism| organism.id.0)
            .max()
            .map_or(0, |max_id| max_id + 1);
        sim.occupancy = vec![None; world_capacity(sim.config.world_width)];
        for organism in &sim.organisms {
            let idx = sim
                .cell_index(organism.q, organism.r)
                .expect("test organism should be in bounds");
            assert!(
                sim.occupancy[idx].is_none(),
                "test setup should not overlap"
            );
            sim.occupancy[idx] = Some(organism.id);
        }
        sim.turn = 0;
        sim.metrics = MetricsSnapshot::default();
        sim.metrics.organisms = sim.organisms.len() as u32;
    }

    fn tick_once(sim: &mut Simulation) -> TickDelta {
        sim.step_n(1).into_iter().next().expect("exactly one delta")
    }

    fn move_map(delta: &TickDelta) -> HashMap<OrganismId, ((i32, i32), (i32, i32))> {
        delta
            .moves
            .iter()
            .map(|movement| (movement.id, (movement.from, movement.to)))
            .collect()
    }

    fn assert_no_overlap(sim: &Simulation) {
        let mut seen = HashSet::new();
        for organism in &sim.organisms {
            assert!(
                seen.insert((organism.q, organism.r)),
                "organisms should not overlap",
            );
            let idx = sim
                .cell_index(organism.q, organism.r)
                .expect("organism should remain in bounds");
            assert_eq!(sim.occupancy[idx], Some(organism.id));
        }
        assert_eq!(sim.organisms.len(), sim.occupancy.iter().flatten().count());
    }

    #[test]
    fn deterministic_seed() {
        let cfg = WorldConfig::default();
        let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
        let mut b = Simulation::new(cfg, 42).expect("simulation B should initialize");
        a.step_n(30);
        b.step_n(30);
        assert_eq!(
            compare_snapshots(&a.snapshot(), &b.snapshot()),
            Ordering::Equal
        );
    }

    #[test]
    fn different_seed_changes_state() {
        let cfg = WorldConfig::default();
        let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
        let mut b = Simulation::new(cfg, 43).expect("simulation B should initialize");
        a.step_n(10);
        b.step_n(10);
        assert_ne!(
            compare_snapshots(&a.snapshot(), &b.snapshot()),
            Ordering::Equal
        );
    }

    #[test]
    fn config_validation_rejects_zero_world_width() {
        let cfg = WorldConfig {
            world_width: 0,
            ..WorldConfig::default()
        };
        let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
        assert!(err.to_string().contains("world_width"));
    }

    #[test]
    fn population_is_capped_by_world_capacity_without_overlap() {
        let cfg = WorldConfig {
            world_width: 3,
            num_organisms: 20,
            num_neurons: 0,
            num_synapses: 0,
            ..WorldConfig::default()
        };
        let sim = Simulation::new(cfg, 3).expect("simulation should initialize");
        assert_eq!(sim.organisms.len(), 9);
        assert_eq!(
            sim.occupancy.iter().filter(|cell| cell.is_some()).count(),
            9
        );
    }

    #[test]
    fn look_sensor_returns_binary_occupancy() {
        let cfg = test_config(5, 2);
        let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

        sim.organisms[0].q = 2;
        sim.organisms[0].r = 2;
        sim.organisms[0].facing = FacingDirection::East;
        sim.organisms[1].q = 3;
        sim.organisms[1].r = 2;

        sim.occupancy.fill(None);
        for org in &sim.organisms {
            let idx = sim.cell_index(org.q, org.r).expect("in-bounds test setup");
            sim.occupancy[idx] = Some(org.id);
        }

        let signal = look_sensor_value(
            (2, 2),
            FacingDirection::East,
            sim.organisms[0].id,
            sim.config.world_width as i32,
            &sim.occupancy,
        );
        assert_eq!(signal, 1.0);

        let empty_signal = look_sensor_value(
            (2, 2),
            FacingDirection::NorthWest,
            sim.organisms[0].id,
            sim.config.world_width as i32,
            &sim.occupancy,
        );
        assert_eq!(empty_signal, 0.0);
    }

    #[test]
    fn turn_actions_rotate_facing() {
        assert_eq!(
            facing_after_turn(FacingDirection::East, true, false),
            FacingDirection::NorthEast
        );
        assert_eq!(
            facing_after_turn(FacingDirection::East, false, true),
            FacingDirection::SouthEast
        );
        assert_eq!(
            facing_after_turn(FacingDirection::East, true, true),
            FacingDirection::East
        );
    }

    #[test]
    fn move_into_cell_vacated_same_turn_succeeds() {
        let cfg = test_config(5, 2);
        let mut sim = Simulation::new(cfg, 11).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.8, 0),
                make_organism(
                    1,
                    2,
                    1,
                    FacingDirection::SouthEast,
                    true,
                    false,
                    false,
                    0.7,
                    0,
                ),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.len(), 2);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
        assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (2, 2))));
        assert_eq!(delta.metrics.meals_last_turn, 0);
    }

    #[test]
    fn two_organism_swap_resolves_deterministically() {
        let cfg = test_config(5, 2);
        let mut sim = Simulation::new(cfg, 12).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.4, 0),
                make_organism(1, 2, 1, FacingDirection::West, true, false, false, 0.3, 0),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
        assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (1, 1))));
        assert_eq!(delta.metrics.meals_last_turn, 0);
    }

    #[test]
    fn multi_attacker_single_target_uses_confidence_winner() {
        let cfg = test_config(5, 2);
        let mut sim = Simulation::new(cfg, 13).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 0, 1, FacingDirection::East, true, false, false, 0.9, 0),
                make_organism(
                    1,
                    1,
                    0,
                    FacingDirection::SouthEast,
                    true,
                    false,
                    false,
                    0.1,
                    0,
                ),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.len(), 1);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((0, 1), (1, 1))));
    }

    #[test]
    fn multi_attacker_single_target_tie_breaks_by_id() {
        let cfg = test_config(5, 2);
        let mut sim = Simulation::new(cfg, 14).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 0, 1, FacingDirection::East, true, false, false, 0.5, 0),
                make_organism(
                    1,
                    1,
                    0,
                    FacingDirection::SouthEast,
                    true,
                    false,
                    false,
                    0.5,
                    0,
                ),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.len(), 1);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((0, 1), (1, 1))));
    }

    #[test]
    fn attacker_vs_escaping_prey_has_no_eat_when_prey_escapes() {
        let cfg = test_config(6, 2);
        let mut sim = Simulation::new(cfg, 15).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.7, 0),
                make_organism(1, 2, 1, FacingDirection::East, true, false, false, 0.6, 0),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
        assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (3, 1))));
        assert_eq!(delta.metrics.meals_last_turn, 0);
        assert!(sim
            .organisms
            .iter()
            .any(|organism| organism.id == OrganismId(1)));
    }

    #[test]
    fn multi_node_cycle_resolves_without_conflict() {
        let cfg = test_config(6, 3);
        let mut sim = Simulation::new(cfg, 16).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(
                    0,
                    1,
                    1,
                    FacingDirection::SouthEast,
                    true,
                    false,
                    false,
                    0.7,
                    0,
                ),
                make_organism(1, 2, 1, FacingDirection::West, true, false, false, 0.6, 0),
                make_organism(
                    2,
                    1,
                    2,
                    FacingDirection::NorthEast,
                    true,
                    false,
                    false,
                    0.5,
                    0,
                ),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.len(), 3);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (1, 2))));
        assert_eq!(moves.get(&OrganismId(2)), Some(&((1, 2), (2, 1))));
        assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (1, 1))));
        assert_eq!(delta.metrics.meals_last_turn, 0);
    }

    #[test]
    fn contested_occupied_target_where_occupant_remains_uses_eat_path() {
        let cfg = test_config(5, 3);
        let mut sim = Simulation::new(cfg, 17).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 0),
                make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
                make_organism(
                    2,
                    1,
                    2,
                    FacingDirection::NorthEast,
                    true,
                    false,
                    false,
                    0.2,
                    0,
                ),
            ],
        );

        let delta = tick_once(&mut sim);
        let moves = move_map(&delta);
        assert_eq!(moves.len(), 1);
        assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
        assert_eq!(delta.metrics.meals_last_turn, 1);
        assert!(sim
            .organisms
            .iter()
            .all(|organism| organism.id != OrganismId(1)));
    }

    #[test]
    fn starvation_and_reproduction_interact_in_same_turn() {
        let mut cfg = test_config(6, 4);
        cfg.turns_to_starve = 2;

        let mut sim = Simulation::new(cfg, 18).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
                make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
                make_organism(2, 0, 0, FacingDirection::East, false, false, false, 0.2, 1),
                make_organism(3, 4, 4, FacingDirection::West, false, false, false, 0.2, 0),
            ],
        );

        let delta = tick_once(&mut sim);
        assert_eq!(delta.metrics.meals_last_turn, 1);
        assert_eq!(delta.metrics.starvations_last_turn, 1);
        assert_eq!(delta.metrics.births_last_turn, 2);
        assert_eq!(delta.removed, vec![OrganismId(1), OrganismId(2)]);
        assert_eq!(delta.spawned.len(), 2);
        assert_eq!(sim.organisms.len(), 4);
        let predator = sim
            .organisms
            .iter()
            .find(|organism| organism.id == OrganismId(0))
            .expect("predator should survive");
        assert_eq!(predator.turns_since_last_meal, 0);
    }

    #[test]
    fn spawn_queue_order_is_deterministic_under_limited_space() {
        let cfg = test_config(2, 3);
        let mut sim = Simulation::new(cfg, 19).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(
                    0,
                    0,
                    0,
                    FacingDirection::NorthEast,
                    true,
                    false,
                    false,
                    0.9,
                    0,
                ),
                make_organism(1, 1, 0, FacingDirection::West, false, false, false, 0.1, 0),
                make_organism(2, 0, 1, FacingDirection::East, false, false, false, 0.1, 0),
            ],
        );
        let parent_brain = sim.organisms[0].brain.clone();
        let parent_facing = sim.organisms[0].facing;

        let spawned = sim.resolve_spawn_requests(&[
            SpawnRequest {
                kind: SpawnRequestKind::Reproduction {
                    parent: OrganismId(0),
                },
            },
            SpawnRequest {
                kind: SpawnRequestKind::StarvationReplacement,
            },
        ]);

        assert_eq!(spawned.len(), 1);
        let child = sim
            .organisms
            .iter()
            .find(|organism| organism.id == OrganismId(3))
            .expect("first spawn request should consume final empty slot");
        assert_eq!(child.brain, parent_brain);
        assert_eq!(child.facing, parent_facing);
    }

    #[test]
    fn no_overlap_invariant_holds_after_mixed_turn() {
        let mut cfg = test_config(6, 4);
        cfg.turns_to_starve = 2;

        let mut sim = Simulation::new(cfg, 20).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![
                make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
                make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
                make_organism(2, 0, 0, FacingDirection::East, false, false, false, 0.2, 1),
                make_organism(3, 4, 4, FacingDirection::West, true, false, false, 0.4, 0),
            ],
        );

        let _ = tick_once(&mut sim);
        assert_no_overlap(&sim);
    }

    #[test]
    fn targeted_complex_resolution_snapshot_is_deterministic() {
        let mut cfg = test_config(6, 4);
        cfg.turns_to_starve = 3;

        let scenario = vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
            make_organism(
                1,
                2,
                1,
                FacingDirection::SouthEast,
                true,
                false,
                false,
                0.7,
                0,
            ),
            make_organism(2, 2, 2, FacingDirection::West, true, false, false, 0.6, 1),
            make_organism(
                3,
                1,
                2,
                FacingDirection::NorthEast,
                true,
                false,
                false,
                0.8,
                0,
            ),
        ];

        let mut a = Simulation::new(cfg.clone(), 21).expect("simulation should initialize");
        configure_sim(&mut a, scenario.clone());
        a.step_n(3);
        let a_snapshot = serde_json::to_string(&a.snapshot()).expect("serialize snapshot");

        let mut b = Simulation::new(cfg, 21).expect("simulation should initialize");
        configure_sim(&mut b, scenario);
        b.step_n(3);
        let b_snapshot = serde_json::to_string(&b.snapshot()).expect("serialize snapshot");

        assert_eq!(a_snapshot, b_snapshot);
    }
}
