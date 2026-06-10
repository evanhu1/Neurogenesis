use super::*;
use crate::brain::sensing::encode_sensory_inputs;

#[derive(Clone, Copy)]
pub(crate) struct BrainEvalContext<'a> {
    pub(crate) world_width: i32,
    pub(crate) occupancy: &'a [Option<Occupant>],
    pub(crate) spike_map: &'a [bool],
    pub(crate) spike_visual_map: &'a [VisualProperties],
    pub(crate) visual_map: &'a [VisualProperties],
    pub(crate) vision_distance: u32,
    pub(crate) action_temperature: f32,
    pub(crate) action_sample: f32,
}

pub(crate) fn evaluate_brain(
    organism: &mut sim_types::OrganismState,
    context: BrainEvalContext<'_>,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    #[cfg_attr(not(feature = "instrumentation"), allow(unused_variables))]
    let ray_scans = encode_sensory_inputs(
        organism,
        context.world_width,
        context.occupancy,
        context.spike_map,
        context.spike_visual_map,
        context.visual_map,
        context.vision_distance,
    );
    #[cfg(feature = "instrumentation")]
    {
        result.food_visible = ray_scans.map(|scan| scan.food_visible);
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ScanAhead, stage_started.elapsed());

    // `align_genome_vectors` forces `action_biases` to ACTION_COUNT on every
    // mutation and at external-genome intake, and seed genomes are built with
    // exactly ACTION_COUNT entries — fail fast instead of silently zeroing
    // evolved biases.
    debug_assert_eq!(organism.genome.brain.action_biases.len(), ACTION_COUNT);
    let mut action_bias_values = [0.0_f32; ACTION_COUNT];
    action_bias_values.copy_from_slice(&organism.genome.brain.action_biases[..ACTION_COUNT]);
    let brain = &mut organism.brain;

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    scratch.prepare_inter_buffers(brain);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterSetup, stage_started.elapsed());

    let mut action_inputs = [0.0; ACTION_COUNT];

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &brain.sensory {
        // Inactive receptors (vision rays over empty terrain, unset
        // last-action flags, ...) contribute nothing: both accumulators are
        // no-ops at zero activation, so skip the partitioning outright.
        if sensory.neuron.activation == 0.0 {
            continue;
        }
        // Outgoing edges are sorted by post ID, so inter-targeting edges form
        // a prefix and action-targeting edges a suffix; the split index is
        // cached by `refresh_action_synapse_starts_and_count`.
        let (inter_edges, action_edges) = sensory.synapses.split_at(sensory.action_synapse_start);
        result.synapse_ops += accumulate_inter_inputs(
            inter_edges,
            sensory.neuron.activation,
            &mut scratch.inter_inputs,
        );
        result.synapse_ops +=
            accumulate_action_inputs(action_edges, sensory.neuron.activation, &mut action_inputs);
    }
    for inter in &brain.inter {
        let (inter_edges, _) = inter.synapses.split_at(inter.action_synapse_start);
        result.synapse_ops += accumulate_inter_inputs(
            inter_edges,
            inter.neuron.activation,
            &mut scratch.inter_inputs,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (idx, neuron) in brain.inter.iter_mut().enumerate() {
        let alpha = neuron.alpha;
        // `neuron.state` still holds the previous-tick value: nothing between
        // the accumulation loops and here writes inter state.
        neuron.state = (1.0 - alpha) * neuron.state + alpha * scratch.inter_inputs[idx];
        neuron.neuron.activation = super::fast_tanh(neuron.state);
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterActivation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for inter in &brain.inter {
        let (_, action_edges) = inter.synapses.split_at(inter.action_synapse_start);
        result.synapse_ops +=
            accumulate_action_inputs(action_edges, inter.neuron.activation, &mut action_inputs);
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (idx, input) in action_inputs.iter_mut().enumerate() {
        *input += action_bias_values[idx];
    }
    result.action_logits = action_inputs;
    for (idx, action) in brain.action.iter_mut().enumerate() {
        debug_assert_eq!(action_index(action.action_type), idx);
        action.logit = action_inputs[idx];
    }
    let sampled_action = sample_action_from_logits(
        result.action_logits,
        EXPLICIT_IDLE_LOGIT_BIAS,
        context.action_temperature,
        context.action_sample,
        &mut scratch.action_probabilities,
    );
    scratch.selected_action_index = match sampled_action {
        ActionType::Idle => None,
        selected_action => Some(action_index(selected_action)),
    };
    result.selected_action = sampled_action;
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionActivationResolve, stage_started.elapsed());

    result
}

fn sample_action_from_logits(
    action_logits: [f32; ACTION_COUNT],
    idle_bias: f32,
    action_temperature: f32,
    action_sample: f32,
    action_probabilities: &mut [f32; ACTION_COUNT],
) -> ActionType {
    let temperature = action_temperature.max(MIN_ACTION_TEMPERATURE);
    let max_logit = action_logits.iter().copied().fold(idle_bias, f32::max);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = 0.0_f32;
    for (idx, logit) in action_logits.iter().copied().enumerate() {
        let scaled = (logit - max_logit) / temperature;
        let weight = scaled.exp();
        weights[idx] = weight;
        weight_sum += weight;
    }
    let idle_weight = ((idle_bias - max_logit) / temperature).exp();
    weight_sum += idle_weight;

    for (idx, weight) in weights.iter().copied().enumerate() {
        action_probabilities[idx] = weight / weight_sum;
    }

    let sample = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * weight_sum;
    let mut cumulative = 0.0_f32;
    for (idx, weight) in weights.iter().copied().enumerate() {
        cumulative += weight;
        if sample < cumulative {
            return ActionType::ALL[idx];
        }
    }
    ActionType::Idle
}

fn accumulate_inter_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    inter_inputs: &mut [f32],
) -> u64 {
    if source_activation == 0.0 {
        return 0;
    }
    let mut synapse_ops = 0;
    for edge in edges {
        let Some(idx) = inter_index(edge.post_neuron_id, inter_inputs.len()) else {
            continue;
        };
        // `inter_index` already proved `idx < inter_inputs.len()`.
        inter_inputs[idx] += source_activation * edge.weight;
        synapse_ops += 1;
    }
    synapse_ops
}

fn accumulate_action_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    action_inputs: &mut [f32; ACTION_COUNT],
) -> u64 {
    if source_activation == 0.0 {
        return 0;
    }
    let mut synapse_ops = 0;
    for edge in edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
        let Some(input_slot) = action_inputs.get_mut(idx) else {
            continue;
        };
        *input_slot += source_activation * edge.weight;
        synapse_ops += 1;
    }
    synapse_ops
}
