use crate::grid::hex_neighbor;
use crate::CellEntity;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, InterNeuronState, LookTarget, NeuronId, NeuronState,
    NeuronType, OrganismGenome, OrganismId, SensoryNeuronState, SensoryReceptor, SynapseEdge,
};
use std::cmp::Ordering;

const ACTION_ACTIVATION_THRESHOLD: f32 = 0.5;
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

/// Build a BrainState deterministically from a genome.
pub(crate) fn express_genome(genome: &OrganismGenome) -> BrainState {
    let mut sensory = vec![
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

    let mut inter = Vec::with_capacity(genome.num_neurons as usize);
    for i in 0..genome.num_neurons {
        let bias = genome.inter_biases.get(i as usize).copied().unwrap_or(0.0);
        inter.push(InterNeuronState {
            neuron: make_neuron(NeuronId(INTER_ID_BASE + i), NeuronType::Inter, bias),
            synapses: Vec::new(),
        });
    }

    let mut action = Vec::with_capacity(ACTION_COUNT);
    for (idx, action_type) in ActionType::ALL.into_iter().enumerate() {
        action.push(make_action_neuron(ACTION_ID_BASE + idx as u32, action_type));
    }

    // Valid neuron IDs for this brain
    let sensory_max = sensory.len() as u32; // 0..4
    let inter_max = INTER_ID_BASE + genome.num_neurons; // 1000..1000+n
    let action_max = ACTION_ID_BASE + ACTION_COUNT as u32; // 2000..2004

    let is_valid_pre = |id: NeuronId| -> bool {
        // pre can be sensory or inter
        id.0 < sensory_max || (id.0 >= INTER_ID_BASE && id.0 < inter_max)
    };
    let is_valid_post = |id: NeuronId| -> bool {
        // post can be inter or action
        (id.0 >= INTER_ID_BASE && id.0 < inter_max) || (id.0 >= ACTION_ID_BASE && id.0 < action_max)
    };

    // Track seen (pre, post) pairs to skip duplicates
    let mut seen_edges = std::collections::HashSet::new();

    for edge in &genome.edges {
        if edge.pre == edge.post {
            continue;
        }
        if !is_valid_pre(edge.pre) || !is_valid_post(edge.post) {
            continue;
        }
        if !seen_edges.insert((edge.pre, edge.post)) {
            continue;
        }

        let synapse = SynapseEdge {
            post_neuron_id: edge.post,
            weight: edge.weight,
        };

        // Add to the correct pre-neuron's synapse list
        if edge.pre.0 < sensory_max {
            sensory[edge.pre.0 as usize].synapses.push(synapse);
        } else if edge.pre.0 >= INTER_ID_BASE && edge.pre.0 < inter_max {
            let idx = (edge.pre.0 - INTER_ID_BASE) as usize;
            inter[idx].synapses.push(synapse);
        }
    }

    // Sort synapse lists by post_neuron_id
    for s in &mut sensory {
        s.synapses.sort_by_key(|e| e.post_neuron_id);
    }
    for i in &mut inter {
        i.synapses.sort_by_key(|e| e.post_neuron_id);
    }

    // Build parent_ids in a single pass over all synapse lists
    // parent_ids[post] = list of pre neurons that connect to post
    let mut parent_map: std::collections::HashMap<NeuronId, Vec<NeuronId>> =
        std::collections::HashMap::new();
    for s in &sensory {
        for edge in &s.synapses {
            parent_map
                .entry(edge.post_neuron_id)
                .or_default()
                .push(s.neuron.neuron_id);
        }
    }
    for i in &inter {
        for edge in &i.synapses {
            parent_map
                .entry(edge.post_neuron_id)
                .or_default()
                .push(i.neuron.neuron_id);
        }
    }

    // Assign parent_ids to inter and action neurons
    for i in &mut inter {
        if let Some(parents) = parent_map.get_mut(&i.neuron.neuron_id) {
            parents.sort();
            parents.dedup();
            i.neuron.parent_ids = parents.clone();
        }
    }
    for a in &mut action {
        if let Some(parents) = parent_map.get_mut(&a.neuron.neuron_id) {
            parents.sort();
            parents.dedup();
            a.neuron.parent_ids = parents.clone();
        }
    }

    let synapse_count = sensory.iter().map(|s| s.synapses.len()).sum::<usize>()
        + inter.iter().map(|i| i.synapses.len()).sum::<usize>();

    BrainState {
        sensory,
        inter,
        action,
        synapse_count: synapse_count as u32,
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
    scratch
        .inter_inputs
        .extend(brain.inter.iter().map(|n| n.neuron.bias));
    scratch.prev_inter.clear();
    scratch
        .prev_inter
        .extend(brain.inter.iter().map(|n| n.neuron.activation));

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
