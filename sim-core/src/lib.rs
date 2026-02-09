use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, FacingDirection, InterNeuronState, MetricsSnapshot,
    NeuronId, NeuronState, NeuronType, OccupancyCell, OrganismId, OrganismMove, OrganismState,
    SensoryNeuronState, SensoryReceptorType, SynapseEdge, TickDelta, WorldConfig, WorldSnapshot,
};
use std::cmp::Ordering;
use std::collections::HashMap;
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
struct ActionResolution {
    moved: Option<((i32, i32), (i32, i32))>,
    meal: bool,
    birth: bool,
}

#[derive(Default)]
struct BrainEvaluation {
    actions: [bool; 3],
    synapse_ops: u64,
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
        let mut moves = Vec::new();
        let mut synapse_ops = 0_u64;
        let mut actions_applied = 0_u64;
        let mut meals = 0_u64;
        let mut starvations = 0_u64;
        let mut births = 0_u64;

        let actor_ids: Vec<OrganismId> = self.organisms.iter().map(|o| o.id).collect();

        for organism_id in actor_ids {
            let Some(idx) = self.organism_index(organism_id) else {
                continue;
            };

            self.organisms[idx].turns_since_last_meal =
                self.organisms[idx].turns_since_last_meal.saturating_add(1);

            if self.organisms[idx].turns_since_last_meal >= self.config.turns_to_starve {
                if self.remove_organism(organism_id).is_some() {
                    starvations += 1;
                    if self.spawn_replacement_in_center().is_some() {
                        births += 1;
                    }
                }
                continue;
            }

            let (position, facing) = {
                let org = &self.organisms[idx];
                ((org.q, org.r), org.facing)
            };

            let occupancy_snapshot = self.occupancy.clone();
            let evaluation = {
                let brain = &mut self.organisms[idx].brain;
                evaluate_brain(
                    brain,
                    position,
                    facing,
                    organism_id,
                    self.config.world_width as i32,
                    &occupancy_snapshot,
                )
            };
            synapse_ops += evaluation.synapse_ops;

            let resolution = self.apply_actions(organism_id, evaluation.actions);
            if let Some((from, to)) = resolution.moved {
                actions_applied += 1;
                moves.push(OrganismMove {
                    id: organism_id,
                    from,
                    to,
                });
            }
            if resolution.meal {
                meals += 1;
            }
            if resolution.birth {
                births += 1;
            }
        }

        self.turn = self.turn.saturating_add(1);
        self.metrics.turns = self.turn;
        self.metrics.organisms = self.organisms.len() as u32;
        self.metrics.synapse_ops_last_turn = synapse_ops;
        self.metrics.actions_applied_last_turn = actions_applied;
        self.metrics.meals_last_turn = meals;
        self.metrics.starvations_last_turn = starvations;
        self.metrics.births_last_turn = births;

        TickDelta {
            turn: self.turn,
            moves,
            metrics: self.metrics.clone(),
        }
    }

    fn apply_actions(
        &mut self,
        organism_id: OrganismId,
        action_outcomes: [bool; 3],
    ) -> ActionResolution {
        let mut result = ActionResolution::default();
        let Some(mut idx) = self.organism_index(organism_id) else {
            return result;
        };

        let turn_left = action_outcomes[action_index(ActionType::TurnLeft)];
        let turn_right = action_outcomes[action_index(ActionType::TurnRight)];
        if turn_left ^ turn_right {
            let facing = self.organisms[idx].facing;
            self.organisms[idx].facing = if turn_left {
                rotate_left(facing)
            } else {
                rotate_right(facing)
            };
        }

        if !action_outcomes[action_index(ActionType::MoveForward)] {
            return result;
        }

        let from = (self.organisms[idx].q, self.organisms[idx].r);
        let to = hex_neighbor(from, self.organisms[idx].facing);
        if !self.in_bounds(to.0, to.1) {
            return result;
        }

        match self.occupant_at(to.0, to.1) {
            None => {
                if self.move_organism_to(organism_id, to.0, to.1) {
                    result.moved = Some((from, to));
                }
            }
            Some(prey_id) if prey_id != organism_id => {
                if self.remove_organism(prey_id).is_none() {
                    return result;
                }

                if !self.move_organism_to(organism_id, to.0, to.1) {
                    return result;
                }

                idx = match self.organism_index(organism_id) {
                    Some(value) => value,
                    None => return result,
                };

                self.organisms[idx].turns_since_last_meal = 0;
                self.organisms[idx].meals_eaten = self.organisms[idx].meals_eaten.saturating_add(1);
                result.meal = true;
                result.moved = Some((from, to));

                if self.spawn_offspring_from_parent(organism_id).is_some() {
                    result.birth = true;
                }
            }
            _ => {}
        }

        result
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

    fn spawn_replacement_in_center(&mut self) -> Option<OrganismId> {
        let (q, r) = self.random_spawn_position_in_center()?;
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
        if self.add_organism(organism) {
            Some(id)
        } else {
            None
        }
    }

    fn spawn_offspring_from_parent(&mut self, parent_id: OrganismId) -> Option<OrganismId> {
        let parent = self.organisms.iter().find(|o| o.id == parent_id)?.clone();
        let (q, r) = self.random_spawn_position_in_center()?;
        let id = self.alloc_organism_id();
        let mut brain = parent.brain;
        self.mutate_brain(&mut brain);

        let child = OrganismState {
            id,
            q,
            r,
            facing: parent.facing,
            turns_since_last_meal: 0,
            meals_eaten: 0,
            brain,
        };

        if self.add_organism(child) {
            Some(id)
        } else {
            None
        }
    }

    fn random_spawn_position_in_center(&mut self) -> Option<(i32, i32)> {
        let mut candidates = self.empty_positions_in_center();
        if candidates.is_empty() {
            candidates = self.empty_positions();
        }
        if candidates.is_empty() {
            return None;
        }
        let idx = self.rng.random_range(0..candidates.len());
        Some(candidates.swap_remove(idx))
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

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn move_organism_to(&mut self, id: OrganismId, q: i32, r: i32) -> bool {
        let Some(org_idx) = self.organism_index(id) else {
            return false;
        };

        let from = (self.organisms[org_idx].q, self.organisms[org_idx].r);
        if from == (q, r) {
            return false;
        }

        let Some(to_idx) = self.cell_index(q, r) else {
            return false;
        };
        if self.occupancy[to_idx].is_some() {
            return false;
        }

        let from_idx = self.cell_index(from.0, from.1).expect("organism position in bounds");
        self.occupancy[from_idx] = None;
        self.occupancy[to_idx] = Some(id);

        self.organisms[org_idx].q = q;
        self.organisms[org_idx].r = r;
        true
    }

    fn remove_organism(&mut self, id: OrganismId) -> Option<OrganismState> {
        let idx = self.organism_index(id)?;
        let organism = self.organisms.swap_remove(idx);
        let cell_idx = self
            .cell_index(organism.q, organism.r)
            .expect("organism position in bounds");
        if self.occupancy[cell_idx] == Some(id) {
            self.occupancy[cell_idx] = None;
        }
        Some(organism)
    }

    fn organism_index(&self, id: OrganismId) -> Option<usize> {
        self.organisms.iter().position(|o| o.id == id)
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

    fn empty_positions_in_center(&self) -> Vec<(i32, i32)> {
        let width = self.config.world_width as i32;
        let min = (self.config.world_width as f32 * self.config.center_spawn_min_fraction) as i32;
        let max = (self.config.world_width as f32 * self.config.center_spawn_max_fraction) as i32;

        let mut out = Vec::new();
        for r in 0..width {
            for q in 0..width {
                if q < min || q >= max || r < min || r >= max {
                    continue;
                }
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
        sensory.synapses.retain(|edge| edge.post_neuron_id != target);
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
        sensory.synapses[idx].weight =
            (sensory.synapses[idx].weight + delta).clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
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
        assert_eq!(sim.occupancy.iter().filter(|cell| cell.is_some()).count(), 9);
    }

    #[test]
    fn look_sensor_returns_binary_occupancy() {
        let cfg = WorldConfig {
            world_width: 5,
            num_organisms: 2,
            num_neurons: 0,
            num_synapses: 0,
            ..WorldConfig::default()
        };
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
        let cfg = WorldConfig {
            world_width: 5,
            num_organisms: 1,
            num_neurons: 0,
            num_synapses: 0,
            ..WorldConfig::default()
        };
        let mut sim = Simulation::new(cfg, 2).expect("simulation should initialize");
        let id = sim.organisms[0].id;
        sim.organisms[0].facing = FacingDirection::East;

        let left = sim.apply_actions(id, [false, true, false]);
        assert!(left.moved.is_none());
        assert_eq!(sim.organisms[0].facing, FacingDirection::NorthEast);

        let right = sim.apply_actions(id, [false, false, true]);
        assert!(right.moved.is_none());
        assert_eq!(sim.organisms[0].facing, FacingDirection::East);
    }

    #[test]
    fn move_forward_into_occupied_cell_eats_and_reproduces() {
        let cfg = WorldConfig {
            world_width: 6,
            num_organisms: 2,
            num_neurons: 0,
            num_synapses: 0,
            mutation_chance: 0.0,
            ..WorldConfig::default()
        };
        let mut sim = Simulation::new(cfg, 11).expect("simulation should initialize");

        let predator_id = sim.organisms[0].id;
        let prey_id = sim.organisms[1].id;

        sim.organisms[0].q = 1;
        sim.organisms[0].r = 2;
        sim.organisms[0].facing = FacingDirection::East;
        sim.organisms[0].turns_since_last_meal = 5;

        sim.organisms[1].q = 2;
        sim.organisms[1].r = 2;

        sim.occupancy.fill(None);
        for org in &sim.organisms {
            let idx = sim.cell_index(org.q, org.r).expect("in-bounds test setup");
            sim.occupancy[idx] = Some(org.id);
        }

        let resolution = sim.apply_actions(predator_id, [true, false, false]);
        assert!(resolution.meal);
        assert!(resolution.birth);
        assert_eq!(sim.organisms.len(), 2);
        assert!(sim.organism_index(prey_id).is_none());

        let predator = sim
            .organisms
            .iter()
            .find(|org| org.id == predator_id)
            .expect("predator should remain");
        assert_eq!((predator.q, predator.r), (2, 2));
        assert_eq!(predator.turns_since_last_meal, 0);
        assert_eq!(predator.meals_eaten, 1);
    }

    #[test]
    fn starvation_spawns_replacement_in_center_region() {
        let cfg = WorldConfig {
            world_width: 8,
            num_organisms: 1,
            num_neurons: 0,
            num_synapses: 0,
            turns_to_starve: 1,
            ..WorldConfig::default()
        };
        let mut sim = Simulation::new(cfg, 19).expect("simulation should initialize");
        let original_id = sim.organisms[0].id;

        sim.step_n(1);

        assert_eq!(sim.organisms.len(), 1);
        assert_ne!(sim.organisms[0].id, original_id);
        assert_eq!(sim.metrics.starvations_last_turn, 1);
        assert_eq!(sim.metrics.births_last_turn, 1);

        let min = (sim.config.world_width as f32 * sim.config.center_spawn_min_fraction) as i32;
        let max = (sim.config.world_width as f32 * sim.config.center_spawn_max_fraction) as i32;
        let spawned = &sim.organisms[0];
        assert!(spawned.q >= min && spawned.q < max);
        assert!(spawned.r >= min && spawned.r < max);
    }
}
