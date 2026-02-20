use crate::genome::{
    inter_alpha_from_log_time_constant, BRAIN_SPACE_MAX, BRAIN_SPACE_MIN,
    DEFAULT_INTER_LOG_TIME_CONSTANT, SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN,
};
use crate::grid::{hex_neighbor, rotate_left, rotate_right};
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use rand::Rng;
use sim_types::{
    ActionNeuronState, ActionType, BrainLocation, BrainState, EntityType, InterNeuronState,
    InterNeuronType, NeuronId, NeuronState, NeuronType, Occupant, OrganismGenome, OrganismId,
    SensoryNeuronState, SensoryReceptor, SynapseEdge,
};
#[cfg(feature = "profiling")]
use std::time::Instant;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const DEFAULT_BIAS: f32 = 0.0;
const HEBB_WEIGHT_CLAMP_ENABLED: bool = true;
const DOPAMINE_ENERGY_DELTA_SCALE: f32 = 10.0;
const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const MIN_ENERGY_SENSOR_SCALE: f32 = 1.0;
const ENERGY_SENSOR_CURVE_EXPONENT: f32 = 2.0;
const MIN_ACTION_TEMPERATURE: f32 = 1.0e-6;
const LOOK_TARGETS: [EntityType; 3] = [EntityType::Food, EntityType::Organism, EntityType::Wall];
const LOOK_RAY_COUNT: usize = SensoryReceptor::LOOK_RAY_OFFSETS.len();
pub(crate) const SENSORY_COUNT: u32 = SensoryReceptor::LOOK_NEURON_COUNT + 1;
pub(crate) const ENERGY_SENSORY_ID: u32 = SensoryReceptor::LOOK_NEURON_COUNT;
pub(crate) const ACTION_COUNT: usize = ActionType::ALL.len();
pub(crate) const ACTION_COUNT_U32: u32 = ACTION_COUNT as u32;
pub(crate) const INTER_ID_BASE: u32 = 1000;
pub(crate) const ACTION_ID_BASE: u32 = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedActions {
    pub(crate) selected_action: ActionType,
}

impl Default for ResolvedActions {
    fn default() -> Self {
        Self {
            selected_action: ActionType::Idle,
        }
    }
}

#[derive(Default)]
pub(crate) struct BrainEvaluation {
    pub(crate) resolved_actions: ResolvedActions,
    pub(crate) action_logits: Vec<f32>,
    pub(crate) action_activations: [f32; ACTION_COUNT],
    pub(crate) synapse_ops: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActionSelectionPolicy {
    pub(crate) temperature: f32,
    pub(crate) argmax_margin: Option<f32>,
}

/// Reusable scratch buffers for brain evaluation, avoiding per-tick allocations.
pub(crate) struct BrainScratch {
    inter_inputs: Vec<f32>,
    prev_inter: Vec<f32>,
    inter_activations: Vec<f32>,
    action_post_signals: [f32; ACTION_COUNT],
}

impl BrainScratch {
    pub(crate) fn new() -> Self {
        Self {
            inter_inputs: Vec::new(),
            prev_inter: Vec::new(),
            inter_activations: Vec::new(),
            action_post_signals: [0.0; ACTION_COUNT],
        }
    }
}

/// Build a BrainState from inherited neuron genes and stored synapse topology.
pub(crate) fn express_genome<R: Rng + ?Sized>(genome: &OrganismGenome, _rng: &mut R) -> BrainState {
    let mut sensory = Vec::with_capacity(SENSORY_COUNT as usize);
    let mut sensory_id = 0_u32;
    for ray_offset in SensoryReceptor::LOOK_RAY_OFFSETS {
        for look_target in LOOK_TARGETS {
            sensory.push(make_sensory_neuron(
                sensory_id,
                SensoryReceptor::LookRay {
                    ray_offset,
                    look_target,
                },
                location_or_default(&genome.sensory_locations, sensory_id as usize),
            ));
            sensory_id = sensory_id.saturating_add(1);
        }
    }
    debug_assert_eq!(sensory_id, ENERGY_SENSORY_ID);
    sensory.push(make_sensory_neuron(
        ENERGY_SENSORY_ID,
        SensoryReceptor::Energy,
        location_or_default(&genome.sensory_locations, ENERGY_SENSORY_ID as usize),
    ));

    let mut inter = Vec::with_capacity(genome.num_neurons as usize);
    for i in 0..genome.num_neurons {
        let idx = i as usize;
        let bias = genome.inter_biases.get(idx).copied().unwrap_or(0.0);
        let log_time_constant = genome
            .inter_log_time_constants
            .get(idx)
            .copied()
            .unwrap_or(DEFAULT_INTER_LOG_TIME_CONSTANT);
        let alpha = inter_alpha_from_log_time_constant(log_time_constant);
        let interneuron_type = genome
            .interneuron_types
            .get(idx)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        inter.push(InterNeuronState {
            neuron: make_neuron(
                NeuronId(INTER_ID_BASE + i),
                NeuronType::Inter,
                bias,
                location_or_default(&genome.inter_locations, idx),
            ),
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
            location_or_default(&genome.action_locations, idx),
        ));
    }

    wire_birth_synapses_from_genome(genome, &mut sensory, &mut inter);

    let mut brain = BrainState {
        sensory,
        inter,
        action,
        synapse_count: 0,
    };
    refresh_parent_ids_and_synapse_count(&mut brain);
    brain
}

pub(crate) fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::Idle => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
        ActionType::Forward => 3,
        ActionType::TurnLeftForward => 4,
        ActionType::TurnRightForward => 5,
        ActionType::Reproduce => 6,
    }
}

fn location_or_default(locations: &[BrainLocation], index: usize) -> BrainLocation {
    locations.get(index).copied().unwrap_or(BrainLocation {
        x: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
        y: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
    })
}

fn wire_birth_synapses_from_genome(
    genome: &OrganismGenome,
    sensory: &mut [SensoryNeuronState],
    inter: &mut [InterNeuronState],
) {
    let max_inter_id = INTER_ID_BASE + inter.len() as u32;
    let max_action_id = ACTION_ID_BASE + ACTION_COUNT_U32;

    for edge in &genome.edges {
        let post_is_inter = (INTER_ID_BASE..max_inter_id).contains(&edge.post_neuron_id.0);
        let post_is_action = (ACTION_ID_BASE..max_action_id).contains(&edge.post_neuron_id.0);
        if !(post_is_inter || post_is_action) {
            continue;
        }

        if edge.pre_neuron_id.0 < SENSORY_COUNT {
            let Some(pre) = sensory.get_mut(edge.pre_neuron_id.0 as usize) else {
                continue;
            };
            pre.synapses.push(SynapseEdge {
                pre_neuron_id: edge.pre_neuron_id,
                post_neuron_id: edge.post_neuron_id,
                weight: constrain_weight(edge.weight, 1.0),
                eligibility: 0.0,
            });
            continue;
        }

        if !(INTER_ID_BASE..max_inter_id).contains(&edge.pre_neuron_id.0) {
            continue;
        }
        if edge.pre_neuron_id == edge.post_neuron_id {
            continue;
        }
        let pre_idx = (edge.pre_neuron_id.0 - INTER_ID_BASE) as usize;
        let Some(pre) = inter.get_mut(pre_idx) else {
            continue;
        };
        let required_sign = match pre.interneuron_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        };
        pre.synapses.push(SynapseEdge {
            pre_neuron_id: edge.pre_neuron_id,
            post_neuron_id: edge.post_neuron_id,
            weight: constrain_weight(edge.weight, required_sign),
            eligibility: 0.0,
        });
    }

    for sensory_neuron in sensory.iter_mut() {
        sensory_neuron.synapses.sort_by(|a, b| {
            a.post_neuron_id
                .cmp(&b.post_neuron_id)
                .then_with(|| a.weight.total_cmp(&b.weight))
        });
    }
    for inter_neuron in inter.iter_mut() {
        inter_neuron.synapses.sort_by(|a, b| {
            a.post_neuron_id
                .cmp(&b.post_neuron_id)
                .then_with(|| a.weight.total_cmp(&b.weight))
        });
    }
}

fn make_neuron(
    id: NeuronId,
    neuron_type: NeuronType,
    bias: f32,
    location: BrainLocation,
) -> NeuronState {
    NeuronState {
        neuron_id: id,
        neuron_type,
        bias,
        x: location.x,
        y: location.y,
        activation: 0.0,
        parent_ids: Vec::new(),
    }
}

pub(crate) fn make_sensory_neuron(
    id: u32,
    receptor: SensoryReceptor,
    location: BrainLocation,
) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, DEFAULT_BIAS, location),
        receptor,
        synapses: Vec::new(),
    }
}

pub(crate) fn make_action_neuron(
    id: u32,
    action_type: ActionType,
    bias: f32,
    location: BrainLocation,
) -> ActionNeuronState {
    ActionNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Action, bias, location),
        action_type,
    }
}

pub(crate) fn evaluate_brain(
    organism: &mut sim_types::OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
    action_selection: ActionSelectionPolicy,
    action_rng: &mut impl Rng,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        vision_distance,
    );
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ScanAhead, stage_started.elapsed());

    let energy_signal = energy_sensor_value(
        organism.energy,
        organism.genome.starting_energy.max(MIN_ENERGY_SENSOR_SCALE),
    );

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &mut organism.brain.sensory {
        sensory.neuron.activation = match &sensory.receptor {
            SensoryReceptor::LookRay {
                ray_offset,
                look_target,
            } => look_ray_signal(&ray_scans, *ray_offset, *look_target),
            SensoryReceptor::Energy => energy_signal,
        };
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::SensoryEncoding, stage_started.elapsed());

    let brain = &mut organism.brain;

    // Reuse scratch buffers: clear + fill avoids reallocation after first organism
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    scratch.inter_inputs.clear();
    scratch
        .inter_inputs
        .extend(brain.inter.iter().map(|n| n.neuron.bias));
    scratch.prev_inter.clear();
    scratch
        .prev_inter
        .extend(brain.inter.iter().map(|n| n.neuron.activation));
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterSetup, stage_started.elapsed());

    let mut action_inputs = [0.0f32; ACTION_COUNT];
    for (idx, action) in brain.action.iter().enumerate() {
        action_inputs[idx] = action.neuron.bias;
    }

    // Accumulate sensory → inter.
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            INTER_ID_BASE,
            &mut scratch.inter_inputs,
        );
    }

    // Recurrent inter → inter uses previous tick's inter activations.
    for (i, inter) in brain.inter.iter().enumerate() {
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            scratch.prev_inter[i],
            INTER_ID_BASE,
            &mut scratch.inter_inputs,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (idx, neuron) in brain.inter.iter_mut().enumerate() {
        let alpha = neuron.alpha;
        let previous = scratch.prev_inter[idx];
        let target = scratch.inter_inputs[idx].tanh();
        neuron.neuron.activation = (1.0 - alpha) * previous + alpha * target;
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterActivation, stage_started.elapsed());

    // Inter → action uses this tick's freshly updated inter activations.
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
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
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    result.action_logits = action_inputs.to_vec();
    let logit_mean = action_inputs.iter().sum::<f32>() / ACTION_COUNT as f32;
    for (idx, logit) in action_inputs.iter().copied().enumerate() {
        scratch.action_post_signals[idx] = logit - logit_mean;
    }
    for (idx, action) in brain.action.iter_mut().enumerate() {
        action.neuron.activation = sigmoid(action_inputs[idx]);
        result.action_activations[action_index(action.action_type)] = action.neuron.activation;
    }

    result.resolved_actions = ResolvedActions {
        selected_action: select_action_from_logits(
            &result.action_logits,
            action_selection,
            action_rng,
        ),
    };
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionActivationResolve, stage_started.elapsed());

    result
}

pub(crate) fn update_runtime_eligibility_traces(
    organism: &mut sim_types::OrganismState,
    scratch: &mut BrainScratch,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let eligibility_retention = organism.genome.eligibility_retention.clamp(0.0, 1.0);

    let brain = &mut organism.brain;
    scratch.inter_activations.clear();
    scratch
        .inter_activations
        .extend(brain.inter.iter().map(|inter| inter.neuron.activation));
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &mut brain.sensory {
        update_edge_eligibility(
            &mut sensory.synapses,
            sensory.neuron.activation,
            eligibility_retention,
            &scratch.inter_activations,
            &scratch.action_post_signals,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (pre_idx, inter) in brain.inter.iter_mut().enumerate() {
        update_inter_edge_eligibility(
            &mut inter.synapses,
            pre_idx,
            eligibility_retention,
            &scratch.prev_inter,
            &scratch.inter_activations,
            &scratch.action_post_signals,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
}

fn should_prune_synapses(age_turns: u64, age_of_maturity: u32) -> bool {
    let maturity_ticks = u64::from(age_of_maturity);
    age_turns >= maturity_ticks && age_turns % SYNAPSE_PRUNE_INTERVAL_TICKS == 0
}

pub(crate) fn apply_runtime_weight_updates(
    organism: &mut sim_types::OrganismState,
    passive_energy_baseline: f32,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let weight_prune_threshold = organism.genome.synapse_prune_threshold.max(0.0);
    let should_prune = should_prune_synapses(organism.age_turns, organism.genome.age_of_maturity);
    let is_mature = organism.age_turns >= u64::from(organism.genome.age_of_maturity);
    let energy_delta = organism.energy - organism.energy_prev;
    // Baseline-correct the reward signal so passive metabolism alone is neutral.
    let corrected_energy_delta = energy_delta + passive_energy_baseline.max(0.0);
    let dopamine_signal = (corrected_energy_delta / DOPAMINE_ENERGY_DELTA_SCALE).tanh();
    organism.dopamine = dopamine_signal;
    organism.energy_prev = organism.energy;
    let eta = organism.genome.hebb_eta_gain.max(0.0);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    if is_mature {
        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        for sensory in &mut organism.brain.sensory {
            apply_edge_weight_update(&mut sensory.synapses, 1.0, eta, dopamine_signal);
        }
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        for inter in &mut organism.brain.inter {
            let required_sign = match inter.interneuron_type {
                InterNeuronType::Excitatory => 1.0,
                InterNeuronType::Inhibitory => -1.0,
            };
            apply_edge_weight_update(&mut inter.synapses, required_sign, eta, dopamine_signal);
        }
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
    }

    if should_prune {
        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        prune_low_weight_synapses(&mut organism.brain, weight_prune_threshold);
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityPrune, stage_started.elapsed());
    }
}

fn update_edge_eligibility(
    edges: &mut [SynapseEdge],
    pre_activation: f32,
    eligibility_retention: f32,
    inter_activations: &[f32],
    action_post_signals: &[f32; ACTION_COUNT],
) {
    for edge in edges {
        let post_activation =
            match post_signal(edge.post_neuron_id, inter_activations, action_post_signals) {
                Some(value) => value,
                None => continue,
            };

        edge.eligibility =
            eligibility_retention * edge.eligibility + pre_activation * post_activation;
    }
}

fn update_inter_edge_eligibility(
    edges: &mut [SynapseEdge],
    pre_idx: usize,
    eligibility_retention: f32,
    prev_inter_activations: &[f32],
    inter_activations: &[f32],
    action_post_signals: &[f32; ACTION_COUNT],
) {
    let Some(pre_prev) = prev_inter_activations.get(pre_idx).copied() else {
        return;
    };
    let Some(pre_current) = inter_activations.get(pre_idx).copied() else {
        return;
    };

    for edge in edges {
        let post_activation =
            match post_signal(edge.post_neuron_id, inter_activations, action_post_signals) {
                Some(value) => value,
                None => continue,
            };
        let pre_activation = if edge.post_neuron_id.0 >= ACTION_ID_BASE {
            pre_current
        } else if edge.post_neuron_id.0 >= INTER_ID_BASE {
            pre_prev
        } else {
            continue;
        };

        edge.eligibility =
            eligibility_retention * edge.eligibility + pre_activation * post_activation;
    }
}

fn apply_edge_weight_update(
    edges: &mut [SynapseEdge],
    required_sign: f32,
    eta: f32,
    dopamine: f32,
) {
    for edge in edges {
        let updated_weight =
            edge.weight + eta * dopamine * edge.eligibility - PLASTIC_WEIGHT_DECAY * edge.weight;
        edge.weight = constrain_weight(updated_weight, required_sign);
    }
}

fn post_signal(
    neuron_id: NeuronId,
    inter_activations: &[f32],
    action_post_signals: &[f32; ACTION_COUNT],
) -> Option<f32> {
    if neuron_id.0 >= ACTION_ID_BASE {
        let action_idx = (neuron_id.0 - ACTION_ID_BASE) as usize;
        return action_post_signals.get(action_idx).copied();
    }
    if neuron_id.0 >= INTER_ID_BASE {
        let inter_idx = (neuron_id.0 - INTER_ID_BASE) as usize;
        return inter_activations.get(inter_idx).copied();
    }
    None
}

fn constrain_weight(weight: f32, required_sign: f32) -> f32 {
    if required_sign.is_sign_negative() {
        if weight >= 0.0 {
            return -SYNAPSE_STRENGTH_MIN;
        }
        let magnitude = if HEBB_WEIGHT_CLAMP_ENABLED {
            (-weight).clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
        } else {
            (-weight).max(SYNAPSE_STRENGTH_MIN)
        };
        -magnitude
    } else {
        if weight <= 0.0 {
            return SYNAPSE_STRENGTH_MIN;
        }
        let magnitude = if HEBB_WEIGHT_CLAMP_ENABLED {
            weight.clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
        } else {
            weight.max(SYNAPSE_STRENGTH_MIN)
        };
        magnitude
    }
}

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) {
    let mut pruned_any = false;

    for sensory in &mut brain.sensory {
        let before = sensory.synapses.len();
        sensory.synapses.retain(|synapse| {
            synapse.weight.abs() >= threshold || synapse.eligibility.abs() >= (2.0f32 * threshold)
        });
        pruned_any |= sensory.synapses.len() != before;
    }
    for inter in &mut brain.inter {
        let before = inter.synapses.len();
        inter.synapses.retain(|synapse| {
            synapse.weight.abs() >= threshold || synapse.eligibility.abs() >= (2.0f32 * threshold)
        });
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
/// Action neurons use policy-based categorical argmax resolution.
pub fn derive_active_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    let mut active = Vec::new();

    for sensory in &brain.sensory {
        if sensory.neuron.activation > 0.0 {
            active.push(sensory.neuron.neuron_id);
        }
    }

    let action_activations: [f32; ACTION_COUNT] =
        std::array::from_fn(|i| brain.action.get(i).map_or(0.0, |n| n.neuron.activation));

    let resolved = resolve_actions(action_activations);
    active.push(
        brain.action[action_index(resolved.selected_action)]
            .neuron
            .neuron_id,
    );

    active
}

fn resolve_actions(activations: [f32; ACTION_COUNT]) -> ResolvedActions {
    ResolvedActions {
        selected_action: ActionType::ALL[argmax_index(&activations)],
    }
}

fn select_action_from_logits(
    action_logits: &[f32],
    action_selection: ActionSelectionPolicy,
    action_rng: &mut impl Rng,
) -> ActionType {
    if action_logits.len() != ACTION_COUNT {
        return ActionType::Idle;
    }

    let best_idx = argmax_index(action_logits);
    let best_logit = action_logits[best_idx];
    let second_logit = second_largest(action_logits, best_idx);
    if let Some(margin) = action_selection.argmax_margin {
        if best_logit - second_logit > margin {
            return ActionType::ALL[best_idx];
        }
    }

    let temperature = action_selection.temperature.max(MIN_ACTION_TEMPERATURE);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = 0.0_f32;
    for (idx, logit) in action_logits.iter().copied().enumerate() {
        let scaled = (logit - best_logit) / temperature;
        let weight = scaled.exp();
        if weight.is_finite() {
            weights[idx] = weight;
            weight_sum += weight;
        }
    }

    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return ActionType::ALL[best_idx];
    }

    let sample = action_rng.random::<f32>() * weight_sum;
    let mut cumulative = 0.0_f32;
    for (idx, weight) in weights.iter().copied().enumerate() {
        cumulative += weight;
        if sample < cumulative {
            return ActionType::ALL[idx];
        }
    }
    ActionType::ALL[ACTION_COUNT - 1]
}

fn argmax_index(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = values[0];
    for (idx, value) in values.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }
    best_idx
}

fn second_largest(values: &[f32], best_idx: usize) -> f32 {
    let mut second = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if idx != best_idx && value > second {
            second = value;
        }
    }
    second
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

/// Maps energy to [0, 1) with a midpoint of 0.5 at `scale`.
/// Uses a Hill-style curve: v = (r^n) / (1 + r^n), where r = energy / scale.
fn energy_sensor_value(energy: f32, scale: f32) -> f32 {
    let safe_scale = scale.max(MIN_ENERGY_SENSOR_SCALE);
    let ratio = energy.max(0.0) / safe_scale;
    let curved = ratio.powf(ENERGY_SENSOR_CURVE_EXPONENT);
    curved / (1.0 + curved)
}

pub(crate) struct ScanResult {
    pub(crate) target: EntityType,
    pub(crate) signal: f32,
}

type RayScans = [Option<ScanResult>; LOOK_RAY_COUNT];

fn ray_offset_index(ray_offset: i8) -> Option<usize> {
    SensoryReceptor::LOOK_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == ray_offset)
}

fn look_ray_signal(ray_scans: &RayScans, ray_offset: i8, look_target: EntityType) -> f32 {
    let Some(ray_idx) = ray_offset_index(ray_offset) else {
        return 0.0;
    };
    match ray_scans[ray_idx].as_ref() {
        Some(hit) if hit.target == look_target => hit.signal,
        _ => 0.0,
    }
}

fn rotate_facing_by_offset(
    mut facing: sim_types::FacingDirection,
    ray_offset: i8,
) -> sim_types::FacingDirection {
    if ray_offset >= 0 {
        for _ in 0..u8::try_from(ray_offset).unwrap_or(0) {
            facing = rotate_right(facing);
        }
        return facing;
    }

    for _ in 0..ray_offset.unsigned_abs() {
        facing = rotate_left(facing);
    }
    facing
}

/// Scans all fixed look rays relative to `facing`, using occlusion per-ray.
pub(crate) fn scan_rays(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> RayScans {
    std::array::from_fn(|idx| {
        scan_ray(
            position,
            facing,
            SensoryReceptor::LOOK_RAY_OFFSETS[idx],
            organism_id,
            world_width,
            occupancy,
            vision_distance,
        )
    })
}

/// Scans one ray up to `vision_distance` hexes.
/// Returns the closest entity found (with occlusion) and a distance-encoded signal
/// strength: `(max_dist - dist + 1) / max_dist`. Returns `None` if all cells are empty.
fn scan_ray(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    ray_offset: i8,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> Option<ScanResult> {
    let ray_facing = rotate_facing_by_offset(facing, ray_offset);
    let max_dist = vision_distance.max(1);
    let mut current = position;
    for d in 1..=max_dist {
        current = hex_neighbor(current, ray_facing, world_width);
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
            Some(Occupant::Wall) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: EntityType::Wall,
                    signal,
                });
            }
            None => {}
        }
    }
    None
}
