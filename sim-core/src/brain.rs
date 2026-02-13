use crate::genome::{inter_alpha_from_log_tau, DEFAULT_INTER_LOG_TAU};
use crate::grid::hex_neighbor;
use sim_types::{
    ActionNeuronState, ActionType, BrainState, EntityType, InterNeuronState, InterNeuronType,
    NeuronId, NeuronState, NeuronType, Occupant, OrganismGenome, OrganismId, SensoryNeuronState,
    SensoryReceptor, SynapseEdge,
};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const ACTION_ACTIVATION_THRESHOLD: f32 = 0.5;
const TURN_ACTION_DEADZONE: f32 = 0.3;
const DEFAULT_BIAS: f32 = 0.0;
const NEUROMODULATOR_SIGNAL: f32 = 0.0;
const OJA_WEIGHT_CLAMP_ENABLED: bool = true;
const OJA_WEIGHT_MAGNITUDE_MIN: f32 = 0.001;
const OJA_WEIGHT_MAGNITUDE_MAX: f32 = 4.0;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const SYNAPSE_PRUNE_LIFESPAN_PERCENT: u64 = 20;
pub(crate) const SENSORY_COUNT: u32 = SensoryReceptor::LOOK_NEURON_COUNT + 1;
pub(crate) const ACTION_COUNT: usize = ActionType::ALL.len();
pub(crate) const ACTION_COUNT_U32: u32 = ACTION_COUNT as u32;
pub(crate) const INTER_ID_BASE: u32 = 1000;
pub(crate) const ACTION_ID_BASE: u32 = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum TurnChoice {
    #[default]
    None,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct ResolvedActions {
    pub(crate) turn: TurnChoice,
    pub(crate) wants_move: bool,
    pub(crate) wants_consume: bool,
    pub(crate) wants_reproduce: bool,
}

#[derive(Default)]
pub(crate) struct BrainEvaluation {
    pub(crate) resolved_actions: ResolvedActions,
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
                look_target: EntityType::Food,
            },
        ),
        make_sensory_neuron(
            1,
            SensoryReceptor::Look {
                look_target: EntityType::Organism,
            },
        ),
        make_sensory_neuron(
            2,
            SensoryReceptor::Look {
                look_target: EntityType::OutOfBounds,
            },
        ),
        make_sensory_neuron(3, SensoryReceptor::Energy),
    ];

    let mut inter = Vec::with_capacity(genome.num_neurons as usize);
    for i in 0..genome.num_neurons {
        let bias = genome.inter_biases.get(i as usize).copied().unwrap_or(0.0);
        let log_tau = genome
            .inter_log_taus
            .get(i as usize)
            .copied()
            .unwrap_or(DEFAULT_INTER_LOG_TAU);
        let alpha = inter_alpha_from_log_tau(log_tau);
        let interneuron_type = genome
            .interneuron_types
            .get(i as usize)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        inter.push(InterNeuronState {
            neuron: make_neuron(NeuronId(INTER_ID_BASE + i), NeuronType::Inter, bias),
            interneuron_type,
            alpha,
            synapses: Vec::new(),
        });
    }

    let mut action = Vec::with_capacity(ACTION_COUNT);
    for (idx, action_type) in ActionType::ALL.into_iter().enumerate() {
        let bias = genome.action_biases.get(idx).copied().unwrap_or(0.0);
        action.push(make_action_neuron(
            ACTION_ID_BASE + idx as u32,
            action_type,
            bias,
        ));
    }

    // Valid neuron IDs for this brain
    let sensory_max = sensory.len() as u32; // 0..4
    let inter_count = genome.num_neurons as usize;
    let inter_max = INTER_ID_BASE + genome.num_neurons; // 1000..1000+n
    let action_max = ACTION_ID_BASE + ACTION_COUNT_U32; // 2000..2004

    let is_valid_pre = |id: NeuronId| -> bool {
        id.0 < sensory_max || (id.0 >= INTER_ID_BASE && id.0 < inter_max)
    };
    let is_valid_post = |id: NeuronId| -> bool {
        (id.0 >= INTER_ID_BASE && id.0 < inter_max) || (id.0 >= ACTION_ID_BASE && id.0 < action_max)
    };

    // Parent ID tracking: Vec-indexed instead of HashMap
    let mut inter_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); inter_count];
    let mut action_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); ACTION_COUNT];

    // Edges are sorted by (pre, post) — use adjacent-duplicate detection
    let mut prev_key: Option<(NeuronId, NeuronId)> = None;

    for edge in &genome.edges {
        let key = (edge.pre_neuron_id, edge.post_neuron_id);
        if prev_key == Some(key) {
            continue;
        }
        prev_key = Some(key);

        if !is_valid_pre(edge.pre_neuron_id) || !is_valid_post(edge.post_neuron_id) {
            continue;
        }

        if edge.pre_neuron_id.0 < sensory_max {
            debug_assert!(
                edge.weight > 0.0,
                "sensory outgoing synapse must be positive"
            );
        } else if edge.pre_neuron_id.0 >= INTER_ID_BASE && edge.pre_neuron_id.0 < inter_max {
            let idx = (edge.pre_neuron_id.0 - INTER_ID_BASE) as usize;
            let required_positive =
                matches!(inter[idx].interneuron_type, InterNeuronType::Excitatory);
            if required_positive {
                debug_assert!(
                    edge.weight > 0.0,
                    "excitatory interneuron outgoing synapse must be positive"
                );
            } else {
                debug_assert!(
                    edge.weight < 0.0,
                    "inhibitory interneuron outgoing synapse must be negative"
                );
            }
        }

        // Add to the correct pre-neuron's synapse list
        if edge.pre_neuron_id.0 < sensory_max {
            sensory[edge.pre_neuron_id.0 as usize]
                .synapses
                .push(edge.clone());
        } else if edge.pre_neuron_id.0 >= INTER_ID_BASE && edge.pre_neuron_id.0 < inter_max {
            let idx = (edge.pre_neuron_id.0 - INTER_ID_BASE) as usize;
            inter[idx].synapses.push(edge.clone());
        }

        // Track parent relationships
        if edge.post_neuron_id.0 >= INTER_ID_BASE && edge.post_neuron_id.0 < inter_max {
            let idx = (edge.post_neuron_id.0 - INTER_ID_BASE) as usize;
            inter_parent_ids[idx].push(edge.pre_neuron_id);
        } else if edge.post_neuron_id.0 >= ACTION_ID_BASE && edge.post_neuron_id.0 < action_max {
            let idx = (edge.post_neuron_id.0 - ACTION_ID_BASE) as usize;
            action_parent_ids[idx].push(edge.pre_neuron_id);
        }
    }

    // Sort synapse lists by post_neuron_id
    for s in &mut sensory {
        s.synapses.sort_by_key(|e| e.post_neuron_id);
    }
    for i in &mut inter {
        i.synapses.sort_by_key(|e| e.post_neuron_id);
    }

    // Assign parent_ids to inter and action neurons (take instead of clone)
    for (idx, i) in inter.iter_mut().enumerate() {
        let parents = &mut inter_parent_ids[idx];
        parents.sort();
        parents.dedup();
        i.neuron.parent_ids = std::mem::take(parents);
    }
    for (idx, a) in action.iter_mut().enumerate() {
        let parents = &mut action_parent_ids[idx];
        parents.sort();
        parents.dedup();
        a.neuron.parent_ids = std::mem::take(parents);
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
        ActionType::Turn => 1,
        ActionType::Consume => 2,
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

pub(crate) fn make_action_neuron(id: u32, action_type: ActionType, bias: f32) -> ActionNeuronState {
    ActionNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Action, bias),
        action_type,
    }
}

pub(crate) fn evaluate_brain(
    organism: &mut sim_types::OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
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
        let alpha = neuron.alpha;
        let previous = scratch.prev_inter[idx];
        let target = scratch.inter_inputs[idx].tanh();
        neuron.neuron.activation = (1.0 - alpha) * previous + alpha * target;
    }

    // Accumulate into action neurons (fixed-size array, no allocation).
    // Start with per-action bias terms.
    let mut action_inputs = [0.0f32; ACTION_COUNT];
    for (idx, action) in brain.action.iter().enumerate() {
        action_inputs[idx] = action.neuron.bias;
    }

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
        action.neuron.activation = match action.action_type {
            ActionType::Turn => action_inputs[idx].tanh(),
            _ => sigmoid(action_inputs[idx]),
        };
        result.action_activations[action_index(action.action_type)] = action.neuron.activation;
    }

    result.resolved_actions = resolve_actions(result.action_activations);

    result
}

pub(crate) fn apply_runtime_plasticity(
    organism: &mut sim_types::OrganismState,
    max_organism_age: u32,
) {
    let eta = (organism.genome.hebb_eta_baseline
        + organism.genome.hebb_eta_gain * NEUROMODULATOR_SIGNAL)
        .max(0.0);
    let lambda = organism.genome.eligibility_decay_lambda.clamp(0.0, 1.0);
    let prune_threshold = organism.genome.synapse_prune_threshold.max(0.0);
    let should_prune = should_prune_synapses(organism.age_turns, max_organism_age);

    let brain = &mut organism.brain;
    let inter_activations: Vec<f32> = brain
        .inter
        .iter()
        .map(|inter| inter.neuron.activation)
        .collect();
    let action_activations: [f32; ACTION_COUNT] =
        std::array::from_fn(|idx| brain.action[idx].neuron.activation);

    for sensory in &mut brain.sensory {
        tune_synapses(
            &mut sensory.synapses,
            sensory.neuron.activation,
            1.0,
            eta,
            lambda,
            &inter_activations,
            &action_activations,
        );
    }

    for inter in &mut brain.inter {
        let required_sign = match inter.interneuron_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        };
        tune_synapses(
            &mut inter.synapses,
            inter.neuron.activation,
            required_sign,
            eta,
            lambda,
            &inter_activations,
            &action_activations,
        );
    }

    if should_prune {
        prune_low_eligibility_synapses(brain, prune_threshold);
    }
}

fn should_prune_synapses(age_turns: u64, max_organism_age: u32) -> bool {
    if max_organism_age == 0 {
        return false;
    }
    let minimum_age = (u64::from(max_organism_age) * SYNAPSE_PRUNE_LIFESPAN_PERCENT) / 100;
    age_turns >= minimum_age && age_turns % SYNAPSE_PRUNE_INTERVAL_TICKS == 0
}

fn tune_synapses(
    edges: &mut [SynapseEdge],
    pre_activation: f32,
    required_sign: f32,
    eta: f32,
    lambda: f32,
    inter_activations: &[f32],
    action_activations: &[f32; ACTION_COUNT],
) {
    for edge in edges {
        let post_activation =
            match post_activation(edge.post_neuron_id, inter_activations, action_activations) {
                Some(value) => value,
                None => continue,
            };
        edge.eligibility = (1.0 - lambda) * edge.eligibility + (pre_activation * post_activation);
        let updated_weight =
            edge.weight + eta * post_activation * (pre_activation - post_activation * edge.weight);
        edge.weight = constrain_weight(updated_weight, required_sign);
    }
}

fn post_activation(
    neuron_id: NeuronId,
    inter_activations: &[f32],
    action_activations: &[f32; ACTION_COUNT],
) -> Option<f32> {
    if neuron_id.0 >= ACTION_ID_BASE {
        let action_idx = (neuron_id.0 - ACTION_ID_BASE) as usize;
        return action_activations.get(action_idx).copied();
    }
    if neuron_id.0 >= INTER_ID_BASE {
        let inter_idx = (neuron_id.0 - INTER_ID_BASE) as usize;
        return inter_activations.get(inter_idx).copied();
    }
    None
}

fn constrain_weight(weight: f32, required_sign: f32) -> f32 {
    let magnitude = if OJA_WEIGHT_CLAMP_ENABLED {
        weight
            .abs()
            .clamp(OJA_WEIGHT_MAGNITUDE_MIN, OJA_WEIGHT_MAGNITUDE_MAX)
    } else {
        weight.abs().max(OJA_WEIGHT_MAGNITUDE_MIN)
    };
    if required_sign.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

fn prune_low_eligibility_synapses(brain: &mut BrainState, threshold: f32) {
    let mut pruned_any = false;

    for sensory in &mut brain.sensory {
        let before = sensory.synapses.len();
        sensory
            .synapses
            .retain(|synapse| synapse.eligibility.abs() >= threshold);
        pruned_any |= sensory.synapses.len() != before;
    }
    for inter in &mut brain.inter {
        let before = inter.synapses.len();
        inter
            .synapses
            .retain(|synapse| synapse.eligibility.abs() >= threshold);
        pruned_any |= inter.synapses.len() != before;
    }

    if pruned_any {
        refresh_parent_ids_and_synapse_count(brain);
    }
}

fn refresh_parent_ids_and_synapse_count(brain: &mut BrainState) {
    let inter_len = brain.inter.len();
    let action_len = brain.action.len();
    let mut inter_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); inter_len];
    let mut action_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); action_len];

    for sensory in &brain.sensory {
        let pre_id = sensory.neuron.neuron_id;
        for synapse in &sensory.synapses {
            if synapse.post_neuron_id.0 >= INTER_ID_BASE {
                let inter_idx = synapse.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
                if inter_idx < inter_parent_ids.len() {
                    inter_parent_ids[inter_idx].push(pre_id);
                    continue;
                }
            }
            if synapse.post_neuron_id.0 >= ACTION_ID_BASE {
                let action_idx = synapse.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
                if action_idx < action_parent_ids.len() {
                    action_parent_ids[action_idx].push(pre_id);
                }
            }
        }
    }

    for inter in &brain.inter {
        let pre_id = inter.neuron.neuron_id;
        for synapse in &inter.synapses {
            if synapse.post_neuron_id.0 >= INTER_ID_BASE {
                let inter_idx = synapse.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
                if inter_idx < inter_parent_ids.len() {
                    inter_parent_ids[inter_idx].push(pre_id);
                    continue;
                }
            }
            if synapse.post_neuron_id.0 >= ACTION_ID_BASE {
                let action_idx = synapse.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
                if action_idx < action_parent_ids.len() {
                    action_parent_ids[action_idx].push(pre_id);
                }
            }
        }
    }

    for (idx, inter) in brain.inter.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut inter_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        inter.neuron.parent_ids = parents;
    }
    for (idx, action) in brain.action.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut action_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        action.neuron.parent_ids = parents;
    }

    let synapse_count = brain
        .sensory
        .iter()
        .map(|n| n.synapses.len())
        .sum::<usize>()
        + brain.inter.iter().map(|n| n.synapses.len()).sum::<usize>();
    brain.synapse_count = synapse_count as u32;
}

/// Derives the set of active neuron IDs from a brain's current activation state.
/// Sensory/Inter neurons are active when activation > 0.0.
/// Action neurons use policy-based resolution with ACTION_ACTIVATION_THRESHOLD.
pub fn derive_active_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    let mut active = Vec::new();

    for sensory in &brain.sensory {
        if sensory.neuron.activation > 0.0 {
            active.push(sensory.neuron.neuron_id);
        }
    }

    let action_activations: [f32; ACTION_COUNT] =
        std::array::from_fn(|i| brain.action[i].neuron.activation);

    let resolved = resolve_actions(action_activations);
    if resolved.wants_move {
        active.push(
            brain.action[action_index(ActionType::MoveForward)]
                .neuron
                .neuron_id,
        );
    }
    match resolved.turn {
        TurnChoice::None => {}
        TurnChoice::Left | TurnChoice::Right => active.push(
            brain.action[action_index(ActionType::Turn)]
                .neuron
                .neuron_id,
        ),
    }
    if resolved.wants_reproduce {
        active.push(
            brain.action[action_index(ActionType::Reproduce)]
                .neuron
                .neuron_id,
        );
    } else if resolved.wants_consume {
        active.push(
            brain.action[action_index(ActionType::Consume)]
                .neuron
                .neuron_id,
        );
    }

    active
}

fn resolve_actions(activations: [f32; ACTION_COUNT]) -> ResolvedActions {
    let move_activation = activations[action_index(ActionType::MoveForward)];
    let turn_signal = activations[action_index(ActionType::Turn)];
    let consume_activation = activations[action_index(ActionType::Consume)];
    let reproduce_activation = activations[action_index(ActionType::Reproduce)];

    let turn = if turn_signal > TURN_ACTION_DEADZONE {
        TurnChoice::Right
    } else if turn_signal < -TURN_ACTION_DEADZONE {
        TurnChoice::Left
    } else {
        TurnChoice::None
    };

    let wants_move = move_activation > ACTION_ACTIVATION_THRESHOLD;
    let wants_consume = consume_activation > ACTION_ACTIVATION_THRESHOLD;
    let wants_reproduce = reproduce_activation > ACTION_ACTIVATION_THRESHOLD;
    ResolvedActions {
        turn,
        wants_move,
        wants_consume,
        wants_reproduce,
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
    pub(crate) target: EntityType,
    pub(crate) signal: f32,
}

/// Scans forward along the facing direction up to `vision_distance` hexes.
/// Returns the closest entity found (with occlusion) and a distance-encoded signal
/// strength: `(max_dist - dist + 1) / max_dist`. Returns `None` if all cells are empty.
pub(crate) fn scan_ahead(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> Option<ScanResult> {
    let max_dist = vision_distance.max(1);
    let mut current = position;
    for d in 1..=max_dist {
        current = hex_neighbor(current, facing);
        if current.0 < 0 || current.1 < 0 || current.0 >= world_width || current.1 >= world_width {
            let signal = (max_dist - d + 1) as f32 / max_dist as f32;
            return Some(ScanResult {
                target: EntityType::OutOfBounds,
                signal,
            });
        }
        let idx = current.1 as usize * world_width as usize + current.0 as usize;
        match occupancy[idx] {
            Some(Occupant::Organism(id)) if id == organism_id => {}
            Some(Occupant::Food(_)) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: EntityType::Food,
                    signal,
                });
            }
            Some(Occupant::Organism(_)) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: EntityType::Organism,
                    signal,
                });
            }
            None => {}
        }
    }
    None
}
