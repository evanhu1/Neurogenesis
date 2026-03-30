use super::*;
use crate::brain::sensing::{encode_sensory_inputs, look_ray_signal};

#[derive(Clone, Copy)]
struct SampledAction {
    action: ActionType,
    confidence: f32,
}

pub(crate) fn evaluate_brain(
    organism: &mut sim_types::OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    vision_distance: u32,
    action_temperature: f32,
    explicit_idle_softmax: bool,
    split_attack_actions: bool,
    action_sample: f32,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let ray_scans = encode_sensory_inputs(
        organism,
        world_width,
        occupancy,
        spike_map,
        vision_distance,
        split_attack_actions,
    );
    #[cfg(feature = "instrumentation")]
    {
        result.food_ahead = look_ray_signal(&ray_scans, 0, EntityType::Food) > 0.0;
        result.food_left = look_ray_signal(&ray_scans, -1, EntityType::Food) > 0.0;
        result.food_right = look_ray_signal(&ray_scans, 1, EntityType::Food) > 0.0;
        result.food_behind = look_ray_signal(&ray_scans, 3, EntityType::Food) > 0.0;
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ScanAhead, stage_started.elapsed());

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
        result.synapse_ops += accumulate_mixed_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            &mut scratch.inter_inputs,
            &mut action_inputs,
        );
    }
    for inter in &brain.inter {
        let (inter_edges, _) = split_inter_and_action_edges(&inter.synapses);
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
        let previous = scratch.prev_inter_states[idx];
        neuron.state = (1.0 - alpha) * previous + alpha * scratch.inter_inputs[idx];
        neuron.neuron.activation = neuron.state.tanh();
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterActivation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for inter in &brain.inter {
        let (_, action_edges) = split_inter_and_action_edges(&inter.synapses);
        result.synapse_ops +=
            accumulate_action_inputs(action_edges, inter.neuron.activation, &mut action_inputs);
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    result.action_logits = action_inputs;
    for (idx, action) in brain.action.iter_mut().enumerate() {
        debug_assert_eq!(action_index(action.action_type), idx);
        action.logit = action_inputs[idx];
    }
    scratch.update_action_post_signals(&action_inputs);

    let sampled_action = if explicit_idle_softmax {
        sample_action_from_logits(
            result.action_logits,
            EXPLICIT_IDLE_LOGIT_BIAS,
            action_temperature,
            action_sample,
        )
    } else {
        SampledAction {
            action: select_action_from_positive_logits(
                result.action_logits,
                action_temperature,
                action_sample,
            ),
            confidence: 1.0,
        }
    };
    scratch.selected_action_index = match sampled_action.action {
        ActionType::Idle => None,
        selected_action => Some(action_index(selected_action)),
    };
    scratch.selected_action_confidence = sampled_action.confidence;
    result.selected_action = sampled_action.action;
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionActivationResolve, stage_started.elapsed());

    result
}

pub fn derive_active_action_neuron_id(organism: &OrganismState) -> Option<NeuronId> {
    organism
        .brain
        .action
        .iter()
        .find(|action| action.action_type == organism.last_action_taken)
        .map(|action_neuron| action_neuron.neuron_id)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn select_action_from_logits(
    action_logits: [f32; ACTION_COUNT],
    idle_bias: f32,
    action_temperature: f32,
    action_sample: f32,
) -> ActionType {
    sample_action_from_logits(action_logits, idle_bias, action_temperature, action_sample).action
}

fn sample_action_from_logits(
    action_logits: [f32; ACTION_COUNT],
    idle_bias: f32,
    action_temperature: f32,
    action_sample: f32,
) -> SampledAction {
    let temperature = action_temperature.max(MIN_ACTION_TEMPERATURE);
    let max_logit = action_logits.iter().copied().fold(idle_bias, f32::max);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = 0.0_f32;
    for (idx, logit) in action_logits.iter().copied().enumerate() {
        let scaled = (logit - max_logit) / temperature;
        let weight = scaled.exp();
        if weight.is_finite() {
            weights[idx] = weight;
            weight_sum += weight;
        }
    }
    let idle_weight = ((idle_bias - max_logit) / temperature).exp();
    if idle_weight.is_finite() {
        weight_sum += idle_weight;
    }

    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        let best_idx = argmax_index(&action_logits);
        return if action_logits[best_idx] >= idle_bias {
            SampledAction {
                action: ActionType::ALL[best_idx],
                confidence: 1.0,
            }
        } else {
            SampledAction {
                action: ActionType::Idle,
                confidence: 1.0,
            }
        };
    }

    let sample = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * weight_sum;
    let mut cumulative = 0.0_f32;
    for (idx, weight) in weights.iter().copied().enumerate() {
        cumulative += weight;
        if sample < cumulative {
            return SampledAction {
                action: ActionType::ALL[idx],
                confidence: weight / weight_sum,
            };
        }
    }
    if idle_weight.is_finite() && sample < cumulative + idle_weight {
        SampledAction {
            action: ActionType::Idle,
            confidence: idle_weight / weight_sum,
        }
    } else {
        let final_weight = weights[ACTION_COUNT - 1];
        SampledAction {
            action: ActionType::ALL[ACTION_COUNT - 1],
            confidence: final_weight / weight_sum,
        }
    }
}

fn select_action_from_positive_logits(
    action_logits: [f32; ACTION_COUNT],
    action_temperature: f32,
    action_sample: f32,
) -> ActionType {
    let temperature = action_temperature.max(MIN_ACTION_TEMPERATURE);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = 0.0_f32;
    let mut best_positive_logit = f32::NEG_INFINITY;
    for (idx, logit) in action_logits.iter().copied().enumerate() {
        if logit > 0.0 {
            best_positive_logit = best_positive_logit.max(logit);
            weights[idx] = logit;
        }
    }

    if !best_positive_logit.is_finite() {
        return ActionType::Idle;
    }

    for weight in &mut weights {
        if *weight <= 0.0 {
            continue;
        }
        let scaled = (*weight - best_positive_logit) / temperature;
        let softmax_weight = scaled.exp();
        if softmax_weight.is_finite() {
            *weight = softmax_weight;
            weight_sum += softmax_weight;
        } else {
            *weight = 0.0;
        }
    }

    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        let best_idx = argmax_index(&action_logits);
        return if action_logits[best_idx] > 0.0 {
            ActionType::ALL[best_idx]
        } else {
            ActionType::Idle
        };
    }

    let sample = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * weight_sum;
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
        let Some(input_slot) = inter_inputs.get_mut(idx) else {
            continue;
        };
        *input_slot += source_activation * edge.weight;
        synapse_ops += 1;
    }
    synapse_ops
}

fn accumulate_mixed_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    inter_inputs: &mut [f32],
    action_inputs: &mut [f32; ACTION_COUNT],
) -> u64 {
    if source_activation == 0.0 {
        return 0;
    }
    let mut synapse_ops = 0;
    for edge in edges {
        if edge.post_neuron_id.0 >= ACTION_ID_BASE {
            let idx = edge.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
            let Some(input_slot) = action_inputs.get_mut(idx) else {
                continue;
            };
            *input_slot += source_activation * edge.weight;
            synapse_ops += 1;
            continue;
        }
        if edge.post_neuron_id.0 >= INTER_ID_BASE {
            let idx = edge.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
            let Some(input_slot) = inter_inputs.get_mut(idx) else {
                continue;
            };
            *input_slot += source_activation * edge.weight;
            synapse_ops += 1;
        }
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
