use crate::grid::hex_neighbor;
use crate::{BrainEvaluation, Simulation, DEFAULT_BIAS, SYNAPSE_STRENGTH_MAX};
use rand::Rng;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, InterNeuronState, NeuronId, NeuronState, NeuronType,
    OrganismId, SensoryNeuronState, SensoryReceptorType, SynapseEdge,
};
use std::collections::HashMap;

const ACTION_ACTIVATION_THRESHOLD: f32 = 0.5;

impl Simulation {
    pub(crate) fn mutate_brain(&mut self, brain: &mut BrainState) {
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

                let _ = create_random_synapse(brain, &mut self.rng);
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
                if !brain.inter.is_empty() {
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
                let _ = create_random_synapse(brain, &mut self.rng);
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

    pub(crate) fn generate_brain(&mut self) -> BrainState {
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
            action.push(make_action_neuron(2000 + idx as u32, action_type));
        }

        let mut brain = BrainState {
            sensory,
            inter,
            action,
            synapse_count: 0,
        };

        for _ in 0..self.config.num_synapses {
            let Some((pre, post)) = random_synapse_endpoints(&brain, &mut self.rng) else {
                break;
            };
            let weight = self
                .rng
                .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
            let _ = create_synapse(&mut brain, pre, post, weight);
        }

        brain.synapse_count = count_synapses(&brain) as u32;
        brain
    }

    fn random_bias(&mut self) -> f32 {
        self.rng.random_range(-1.0..1.0)
    }
}

pub(crate) fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::MoveForward => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
    }
}

pub(crate) fn move_confidence_signal(brain: &BrainState) -> f32 {
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
        is_active: false,
        parent_ids: Vec::new(),
    }
}

pub(crate) fn make_sensory_neuron(
    id: u32,
    receptor_type: SensoryReceptorType,
) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, DEFAULT_BIAS),
        receptor_type,
        synapses: Vec::new(),
    }
}

pub(crate) fn make_action_neuron(id: u32, action_type: ActionType) -> ActionNeuronState {
    ActionNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Action, DEFAULT_BIAS),
        action_type,
    }
}

fn next_inter_neuron_id(brain: &BrainState) -> u32 {
    brain
        .inter
        .iter()
        .map(|neuron| neuron.neuron.neuron_id.0)
        .max()
        .map_or(1000, |id| id + 1)
}

pub(crate) fn evaluate_brain(
    brain: &mut BrainState,
    position: (i32, i32),
    facing: sim_protocol::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<OrganismId>],
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    for action in &mut brain.action {
        action.neuron.is_active = false;
    }

    let look = look_sensor_value(position, facing, organism_id, world_width, occupancy);
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = match sensory.receptor_type {
            SensoryReceptorType::Look => look,
        };
        sensory.neuron.is_active = sensory.neuron.activation > 0.0;
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

    let mut inter_inputs: Vec<f32> = brain
        .inter
        .iter()
        .map(|neuron| neuron.neuron.bias)
        .collect();
    let prev_inter: Vec<f32> = brain
        .inter
        .iter()
        .map(|neuron| neuron.neuron.activation)
        .collect();

    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            &inter_index,
            &mut inter_inputs,
        );
    }

    for (source_idx, inter) in brain.inter.iter().enumerate() {
        let source_activation = prev_inter[source_idx];
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            source_activation,
            &inter_index,
            &mut inter_inputs,
        );
    }

    for (idx, neuron) in brain.inter.iter_mut().enumerate() {
        neuron.neuron.activation = inter_inputs[idx].tanh();
        neuron.neuron.is_active = neuron.neuron.activation > 0.0;
    }

    let mut action_inputs: Vec<f32> = vec![0.0; brain.action.len()];

    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            &action_index_map,
            &mut action_inputs,
        );
    }

    for inter in &brain.inter {
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            inter.neuron.activation,
            &action_index_map,
            &mut action_inputs,
        );
    }

    for action in &mut brain.action {
        if let Some(idx) = action_index_map.get(&action.neuron.neuron_id) {
            action.neuron.activation = sigmoid(action_inputs[*idx]);
            action.neuron.is_active = action.neuron.activation > ACTION_ACTIVATION_THRESHOLD;
            result.actions[action_index(action.action_type)] = action.neuron.is_active;
        }
    }

    result
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn accumulate_weighted_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    index_map: &HashMap<NeuronId, usize>,
    inputs: &mut [f32],
) -> u64 {
    let mut synapse_ops = 0;
    for edge in edges {
        if let Some(idx) = index_map.get(&edge.post_neuron_id) {
            inputs[*idx] += source_activation * edge.weight;
            synapse_ops += 1;
        }
    }
    synapse_ops
}

pub(crate) fn look_sensor_value(
    position: (i32, i32),
    facing: sim_protocol::FacingDirection,
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

fn output_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    brain
        .sensory
        .iter()
        .map(|neuron| neuron.neuron.neuron_id)
        .chain(brain.inter.iter().map(|neuron| neuron.neuron.neuron_id))
        .collect()
}

fn random_neuron_id<R: Rng + ?Sized>(ids: &[NeuronId], rng: &mut R) -> Option<NeuronId> {
    if ids.is_empty() {
        return None;
    }
    Some(ids[rng.random_range(0..ids.len())])
}

fn random_synapse_endpoints<R: Rng + ?Sized>(
    brain: &BrainState,
    rng: &mut R,
) -> Option<(NeuronId, NeuronId)> {
    let pre_candidates = output_neuron_ids(brain);
    let post_candidates = post_neuron_ids(brain);

    let pre = random_neuron_id(&pre_candidates, rng)?;
    let post = random_neuron_id(&post_candidates, rng)?;
    Some((pre, post))
}

fn create_random_synapse<R: Rng + ?Sized>(brain: &mut BrainState, rng: &mut R) -> bool {
    let Some((pre, post)) = random_synapse_endpoints(brain, rng) else {
        return false;
    };
    let weight = rng.random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
    create_synapse(brain, pre, post, weight)
}

fn post_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    brain
        .inter
        .iter()
        .map(|neuron| neuron.neuron.neuron_id)
        .chain(brain.action.iter().map(|neuron| neuron.neuron.neuron_id))
        .collect()
}

fn count_synapses(brain: &BrainState) -> usize {
    brain
        .sensory
        .iter()
        .map(|sensory| sensory.synapses.len())
        .sum::<usize>()
        + brain
            .inter
            .iter()
            .map(|inter| inter.synapses.len())
            .sum::<usize>()
}

fn get_neuron_mut(brain: &mut BrainState, id: NeuronId) -> Option<&mut NeuronState> {
    if let Some(sensory) = brain
        .sensory
        .iter_mut()
        .find(|neuron| neuron.neuron.neuron_id == id)
    {
        return Some(&mut sensory.neuron);
    }
    if let Some(inter) = brain
        .inter
        .iter_mut()
        .find(|neuron| neuron.neuron.neuron_id == id)
    {
        return Some(&mut inter.neuron);
    }
    if let Some(action) = brain
        .action
        .iter_mut()
        .find(|neuron| neuron.neuron.neuron_id == id)
    {
        return Some(&mut action.neuron);
    }
    None
}

fn synapses_from_pre(brain: &BrainState, pre: NeuronId) -> Option<&[SynapseEdge]> {
    if let Some(sensory) = brain
        .sensory
        .iter()
        .find(|neuron| neuron.neuron.neuron_id == pre)
    {
        return Some(&sensory.synapses);
    }
    if let Some(inter) = brain
        .inter
        .iter()
        .find(|neuron| neuron.neuron.neuron_id == pre)
    {
        return Some(&inter.synapses);
    }
    None
}

fn synapses_from_pre_mut(brain: &mut BrainState, pre: NeuronId) -> Option<&mut Vec<SynapseEdge>> {
    if let Some(idx) = brain
        .sensory
        .iter()
        .position(|neuron| neuron.neuron.neuron_id == pre)
    {
        return Some(&mut brain.sensory[idx].synapses);
    }
    if let Some(idx) = brain
        .inter
        .iter()
        .position(|neuron| neuron.neuron.neuron_id == pre)
    {
        return Some(&mut brain.inter[idx].synapses);
    }
    None
}

fn create_synapse(brain: &mut BrainState, pre: NeuronId, post: NeuronId, weight: f32) -> bool {
    if pre == post {
        return false;
    }

    let Some(existing_synapses) = synapses_from_pre(brain, pre) else {
        return false;
    };
    let duplicate = existing_synapses
        .iter()
        .any(|edge| edge.post_neuron_id == post);

    if duplicate {
        return false;
    }

    let clamped = weight.clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);

    {
        let Some(synapses) = synapses_from_pre_mut(brain, pre) else {
            return false;
        };
        synapses.push(SynapseEdge {
            post_neuron_id: post,
            weight: clamped,
        });
        synapses.sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
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
    let post = {
        let Some(synapses) = synapses_from_pre_mut(brain, pre) else {
            return;
        };
        if synapses.is_empty() {
            return;
        }

        let idx = rng.random_range(0..synapses.len());
        let post = synapses[idx].post_neuron_id;
        synapses.remove(idx);
        post
    };

    if let Some(post_neuron) = get_neuron_mut(brain, post) {
        post_neuron.parent_ids.retain(|id| *id != pre);
    }
}

fn perturb_random_synapse<R: Rng + ?Sized>(
    brain: &mut BrainState,
    pre: NeuronId,
    mutation_magnitude: f32,
    rng: &mut R,
) {
    let magnitude = mutation_magnitude.clamp(0.1, 8.0);
    let Some(synapses) = synapses_from_pre_mut(brain, pre) else {
        return;
    };
    if synapses.is_empty() {
        return;
    }

    let idx = rng.random_range(0..synapses.len());
    let delta = rng.random_range(-magnitude..magnitude);
    synapses[idx].weight =
        (synapses[idx].weight + delta).clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
}
