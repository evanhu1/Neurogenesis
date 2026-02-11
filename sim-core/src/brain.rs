use crate::grid::hex_neighbor;
use crate::{CellEntity, Simulation};
use rand::Rng;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, InterNeuronState, LookTarget, NeuronId, NeuronState,
    NeuronType, OrganismId, SensoryNeuronState, SensoryReceptor, SpeciesConfig, SynapseEdge,
};
use std::cmp::Ordering;

const ACTION_ACTIVATION_THRESHOLD: f32 = 0.5;
const SYNAPSE_STRENGTH_MAX: f32 = 3.0;
const DEFAULT_BIAS: f32 = 0.0;
const ACTION_COUNT: usize = 4;
const INTER_ID_BASE: u32 = 1000;
const ACTION_ID_BASE: u32 = 2000;

#[derive(Default)]
pub(crate) struct BrainEvaluation {
    pub(crate) actions: [bool; ACTION_COUNT],
    pub(crate) action_activations: [f32; ACTION_COUNT],
    pub(crate) synapse_ops: u64,
}

/// Reusable scratch buffers for brain evaluation, avoiding per-tick allocations.
pub(crate) struct BrainScratch {
    inter_inputs: Vec<f32>,
    prev_inter: Vec<f32>,
}

impl BrainScratch {
    pub(crate) fn new() -> Self {
        Self {
            inter_inputs: Vec::new(),
            prev_inter: Vec::new(),
        }
    }
}

impl Simulation {
    pub(crate) fn generate_brain(&mut self, species_config: &SpeciesConfig) -> BrainState {
        let sensory = vec![
            make_sensory_neuron(
                0,
                SensoryReceptor::Look {
                    look_target: LookTarget::Food,
                },
            ),
            make_sensory_neuron(
                1,
                SensoryReceptor::Look {
                    look_target: LookTarget::Organism,
                },
            ),
            make_sensory_neuron(
                2,
                SensoryReceptor::Look {
                    look_target: LookTarget::OutOfBounds,
                },
            ),
            make_sensory_neuron(3, SensoryReceptor::Energy),
        ];

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

pub(crate) fn make_sensory_neuron(id: u32, receptor: SensoryReceptor) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, DEFAULT_BIAS),
        receptor,
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
    organism: &mut sim_protocol::OrganismState,
    world_width: i32,
    occupancy: &[Option<CellEntity>],
    vision_distance: u32,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    let scan = scan_ahead(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        vision_distance,
    );

    for sensory in &mut organism.brain.sensory {
        sensory.neuron.activation = match &sensory.receptor {
            SensoryReceptor::Look { look_target } => match &scan {
                Some(ref r) if r.target == *look_target => r.signal,
                _ => 0.0,
            },
            SensoryReceptor::Energy => energy_sensor_value(organism.energy),
        };
    }

    let brain = &mut organism.brain;

    // Reuse scratch buffers: clear + fill avoids reallocation after first organism
    scratch.inter_inputs.clear();
    scratch.inter_inputs.extend(brain.inter.iter().map(|n| n.neuron.bias));
    scratch.prev_inter.clear();
    scratch.prev_inter.extend(brain.inter.iter().map(|n| n.neuron.activation));

    // Accumulate sensory → inter
    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            INTER_ID_BASE,
            &mut scratch.inter_inputs,
        );
    }

    // Accumulate inter → inter (using previous tick's activations)
    for (i, inter) in brain.inter.iter().enumerate() {
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            scratch.prev_inter[i],
            INTER_ID_BASE,
            &mut scratch.inter_inputs,
        );
    }

    for (idx, neuron) in brain.inter.iter_mut().enumerate() {
        neuron.neuron.activation = scratch.inter_inputs[idx].tanh();
    }

    // Accumulate into action neurons (fixed-size array, no allocation)
    let mut action_inputs = [0.0f32; ACTION_COUNT];

    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            ACTION_ID_BASE,
            &mut action_inputs,
        );
    }

    for inter in &brain.inter {
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            inter.neuron.activation,
            ACTION_ID_BASE,
            &mut action_inputs,
        );
    }

    for (idx, action) in brain.action.iter_mut().enumerate() {
        action.neuron.activation = sigmoid(action_inputs[idx]);
        result.action_activations[action_index(action.action_type)] = action.neuron.activation;
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

    let action_activations: [f32; ACTION_COUNT] =
        std::array::from_fn(|i| brain.action[i].neuron.activation);
    if let Some(winner_idx) = select_action(action_activations) {
        active.push(brain.action[winner_idx].neuron.neuron_id);
    }

    active
}

fn select_action(activations: [f32; ACTION_COUNT]) -> Option<usize> {
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

/// Accumulates weighted inputs using arithmetic index resolution.
/// `id_base` is the neuron ID offset for the target layer (e.g. 1000 for inter, 2000 for action).
/// Edges targeting neurons outside the range are skipped via bounds check.
fn accumulate_weighted_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    id_base: u32,
    inputs: &mut [f32],
) -> u64 {
    let num_slots = inputs.len();
    let mut synapse_ops = 0;
    for edge in edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(id_base) as usize;
        if idx < num_slots {
            inputs[idx] += source_activation * edge.weight;
            synapse_ops += 1;
        }
    }
    synapse_ops
}

/// Normalizes energy to a [0, 1] range using a logarithmic curve.
/// Uses ln(1 + energy) / ln(1 + scale) where scale controls the saturation point.
fn energy_sensor_value(energy: f32) -> f32 {
    const ENERGY_SCALE: f32 = 100.0;
    (1.0 + energy.max(0.0)).ln() / (1.0 + ENERGY_SCALE).ln()
}

pub(crate) struct ScanResult {
    pub(crate) target: LookTarget,
    pub(crate) signal: f32,
}

/// Scans forward along the facing direction up to `vision_distance` hexes.
/// Returns the closest entity found (with occlusion) and a distance-encoded signal
/// strength: `(max_dist - dist + 1) / max_dist`. Returns `None` if all cells are empty.
pub(crate) fn scan_ahead(
    position: (i32, i32),
    facing: sim_protocol::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<CellEntity>],
    vision_distance: u32,
) -> Option<ScanResult> {
    let max_dist = vision_distance.max(1);
    let mut current = position;
    for d in 1..=max_dist {
        current = hex_neighbor(current, facing);
        if current.0 < 0 || current.1 < 0 || current.0 >= world_width || current.1 >= world_width {
            let signal = (max_dist - d + 1) as f32 / max_dist as f32;
            return Some(ScanResult {
                target: LookTarget::OutOfBounds,
                signal,
            });
        }
        let idx = current.1 as usize * world_width as usize + current.0 as usize;
        match occupancy[idx] {
            Some(CellEntity::Organism(id)) if id == organism_id => {}
            Some(CellEntity::Food(_)) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: LookTarget::Food,
                    signal,
                });
            }
            Some(CellEntity::Organism(_)) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: LookTarget::Organism,
                    signal,
                });
            }
            None => {}
        }
    }
    None
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
