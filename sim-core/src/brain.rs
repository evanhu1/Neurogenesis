use crate::grid::hex_neighbor;
use crate::{BrainEvaluation, Simulation, DEFAULT_BIAS, SYNAPSE_STRENGTH_MAX};
use rand::Rng;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, InterNeuronState, NeuronId, NeuronState, NeuronType,
    OrganismId, SensoryNeuronState, SensoryReceptorType, SpeciesConfig, SynapseEdge,
};
use std::cmp::Ordering;
use std::collections::HashMap;

const ACTION_ACTIVATION_THRESHOLD: f32 = 0.5;

impl Simulation {
    pub(crate) fn generate_brain(&mut self, species_config: &SpeciesConfig) -> BrainState {
        let mut sensory = Vec::new();
        for (idx, receptor_type) in SensoryReceptorType::ALL.into_iter().enumerate() {
            sensory.push(make_sensory_neuron(idx as u32, receptor_type));
        }

        let mut inter = Vec::new();
        for i in 0..species_config.num_neurons {
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

        for _ in 0..species_config.num_synapses {
            if !create_random_synapse(&mut brain, &mut self.rng) {
                break;
            }
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
        ActionType::Reproduce => 3,
    }
}

pub(crate) fn reset_brain_runtime_state(brain: &mut BrainState) {
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = 0.0;
    }
    for inter in &mut brain.inter {
        inter.neuron.activation = 0.0;
    }
    for action in &mut brain.action {
        action.neuron.activation = 0.0;
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

pub(crate) fn evaluate_brain(
    brain: &mut BrainState,
    position: (i32, i32),
    facing: sim_protocol::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<OrganismId>],
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

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
            result.action_activations[action_index(action.action_type)] = action.neuron.activation;
        }
    }

    if let Some(selected_idx) = select_action(result.action_activations) {
        result.actions[selected_idx] = true;
    }

    result
}

/// Derives the set of active neuron IDs from a brain's current activation state.
/// Sensory/Inter neurons are active when activation > 0.0.
/// Action neurons use winner-take-all with ACTION_ACTIVATION_THRESHOLD.
pub fn derive_active_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    let mut active = Vec::new();

    for sensory in &brain.sensory {
        if sensory.neuron.activation > 0.0 {
            active.push(sensory.neuron.neuron_id);
        }
    }

    for inter in &brain.inter {
        if inter.neuron.activation > 0.0 {
            active.push(inter.neuron.neuron_id);
        }
    }

    let action_activations: [f32; 4] = std::array::from_fn(|i| brain.action[i].neuron.activation);
    if let Some(winner_idx) = select_action(action_activations) {
        active.push(brain.action[winner_idx].neuron.neuron_id);
    }

    active
}

fn select_action(activations: [f32; 4]) -> Option<usize> {
    let mut best_idx = 0;
    let mut best_activation = activations[0];

    for (idx, &activation) in activations.iter().enumerate().skip(1) {
        if activation.total_cmp(&best_activation) == Ordering::Greater {
            best_idx = idx;
            best_activation = activation;
        }
    }

    (best_activation > ACTION_ACTIVATION_THRESHOLD).then_some(best_idx)
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

fn create_random_synapse<R: Rng + ?Sized>(brain: &mut BrainState, rng: &mut R) -> bool {
    let pre_candidates = output_neuron_ids(brain);
    let post_candidates = post_neuron_ids(brain);
    let mut available_endpoints = Vec::new();

    for pre in pre_candidates {
        let Some(existing_synapses) = synapses_from_pre(brain, pre) else {
            continue;
        };
        for &post in &post_candidates {
            if pre == post {
                continue;
            }
            let duplicate = existing_synapses
                .iter()
                .any(|edge| edge.post_neuron_id == post);
            if !duplicate {
                available_endpoints.push((pre, post));
            }
        }
    }

    if available_endpoints.is_empty() {
        return false;
    }

    let (pre, post) = available_endpoints[rng.random_range(0..available_endpoints.len())];
    let weight = rng.random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
    let created = create_synapse(brain, pre, post, weight);
    debug_assert!(
        created,
        "available endpoint selection should only choose valid synapses"
    );
    created
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
