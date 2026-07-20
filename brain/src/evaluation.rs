use super::*;

#[derive(Clone, Copy)]
pub struct BrainEvalContext {
    pub leaky_neurons_enabled: bool,
    pub action_temperature: f32,
    /// `None` evaluates the recurrent network without sampling an output.
    pub action_sample: Option<f32>,
}

pub fn evaluate_brain(
    organism: &mut types::OrganismState,
    context: BrainEvalContext,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    evaluate_brain_state(&mut organism.brain, &organism.genome, context, scratch)
}

/// Evaluate one symbol step without requiring ecology/world state.
///
/// The evolutionary symbol-copy task owns a `BrainState` directly and uses
/// this entry point. The organism wrapper above remains for the optional world
/// simulator.
pub fn evaluate_brain_state(
    brain: &mut BrainState,
    genome: &OrganismGenome,
    context: BrainEvalContext,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();
    debug_assert_eq!(genome.brain.action_biases.len(), ACTION_COUNT);
    let mut action_biases = [0.0_f32; ACTION_COUNT];
    action_biases.copy_from_slice(&genome.brain.action_biases);

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    scratch.prepare_inter_buffers(brain);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterSetup, stage_started.elapsed());

    let mut action_inputs = [0.0; ACTION_COUNT];
    let mut value_input = 0.0_f32;

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    debug_assert_eq!(brain.previous_inter_activations.len(), brain.inter.len());
    for edge in &brain.recurrent_synapses {
        let pre_index = edge
            .pre_inter_index
            .expect("expressed recurrent edge has a dense presynaptic index")
            as usize;
        let post_index = edge
            .post_inter_index
            .expect("expressed recurrent edge has a dense postsynaptic index")
            as usize;
        scratch.inter_inputs[post_index] +=
            edge.weight * brain.previous_inter_activations[pre_index];
        result.synapse_ops += 1;
    }
    for edge in &brain.action_feedback_synapses {
        let pre_index = edge
            .pre_action_index
            .expect("expressed efference-copy edge has a dense action index")
            as usize;
        let post_index = edge
            .post_inter_index
            .expect("expressed efference-copy edge has a dense hidden index")
            as usize;
        scratch.inter_inputs[post_index] +=
            edge.weight * brain.previous_action_activations[pre_index];
        result.synapse_ops += 1;
    }
    if brain.previous_prediction_error != 0.0 {
        for (input, hidden) in scratch.inter_inputs.iter_mut().zip(&brain.inter) {
            *input += hidden.neuromodulatory_receptor * brain.previous_prediction_error;
        }
    }
    for sensory in &brain.sensory {
        if sensory.neuron.activation == 0.0 {
            continue;
        }
        let (inter_edges, output_edges) = sensory.synapses.split_at(sensory.output_synapse_start);
        result.synapse_ops += accumulate_inter_inputs(
            inter_edges,
            sensory.neuron.activation,
            &mut scratch.inter_inputs,
        );
        result.synapse_ops += accumulate_output_inputs(
            output_edges,
            sensory.neuron.activation,
            &mut action_inputs,
            &mut value_input,
        );
    }
    for idx in 0..brain.inter.len() {
        let activation = {
            let neuron = &mut brain.inter[idx];
            neuron.state = if context.leaky_neurons_enabled {
                (1.0 - neuron.alpha) * neuron.state + neuron.alpha * scratch.inter_inputs[idx]
            } else {
                scratch.inter_inputs[idx]
            };
            neuron.neuron.activation = fast_tanh(neuron.state);
            neuron.neuron.activation
        };
        let inter = &brain.inter[idx];
        let (inter_edges, output_edges) = inter.synapses.split_at(inter.output_synapse_start);
        debug_assert!(inter_edges.iter().all(|edge| {
            crate::topology::inter_index(edge.post_neuron_id, brain.inter.len())
                .is_some_and(|post_index| post_index > idx)
        }));
        result.synapse_ops +=
            accumulate_inter_inputs(inter_edges, activation, &mut scratch.inter_inputs);
        result.synapse_ops += accumulate_output_inputs(
            output_edges,
            activation,
            &mut action_inputs,
            &mut value_input,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (input, bias) in action_inputs.iter_mut().zip(action_biases) {
        *input += bias;
    }
    result.action_logits = action_inputs;
    result.value_prediction = fast_tanh(value_input + brain.value_bias);
    for (idx, action) in brain.action.iter_mut().enumerate() {
        debug_assert_eq!(action.symbol.index(), idx);
        action.logit = action_inputs[idx];
    }
    if let Some(action_sample) = context.action_sample {
        let selected_symbol = sample_action_symbol(
            action_inputs,
            context.action_temperature,
            action_sample,
            &mut scratch.action_probabilities,
        );
        result.selected_symbol = selected_symbol;
    } else {
        scratch.action_probabilities.fill(0.0);
    }

    for (saved, inter) in brain
        .previous_inter_activations
        .iter_mut()
        .zip(brain.inter.iter())
    {
        *saved = inter.neuron.activation;
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionActivationResolve, stage_started.elapsed());

    result
}

fn sample_action_symbol(
    logits: [f32; ACTION_COUNT],
    temperature: f32,
    sample: f32,
    probabilities: &mut [f32; ACTION_COUNT],
) -> Symbol {
    let temperature = temperature.max(MIN_ACTION_TEMPERATURE);
    let max_logit = Symbol::ALL
        .into_iter()
        .map(|symbol| logits[symbol.index()])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = 0.0;
    for symbol in Symbol::ALL {
        let weight = ((logits[symbol.index()] - max_logit) / temperature).exp();
        weights[symbol.index()] = weight;
        weight_sum += weight;
    }
    for symbol in Symbol::ALL {
        probabilities[symbol.index()] = weights[symbol.index()] / weight_sum;
    }
    let draw = sample.clamp(0.0, 1.0 - f32::EPSILON) * weight_sum;
    let mut cumulative = 0.0;
    for symbol in Symbol::ALL {
        cumulative += weights[symbol.index()];
        if draw < cumulative {
            return symbol;
        }
    }
    Symbol::End
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
        inter_inputs[idx] += source_activation * edge.weight;
        synapse_ops += 1;
    }
    synapse_ops
}

fn accumulate_output_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    action_inputs: &mut [f32; ACTION_COUNT],
    value_input: &mut f32,
) -> u64 {
    if source_activation == 0.0 {
        return 0;
    }
    let mut synapse_ops = 0;
    for edge in edges {
        if let Some(idx) = crate::topology::action_array_index(edge.post_neuron_id) {
            action_inputs[idx] += source_activation * edge.weight;
            synapse_ops += 1;
        } else if crate::topology::is_value_neuron_id(edge.post_neuron_id) {
            *value_input += source_activation * edge.weight;
            synapse_ops += 1;
        }
    }
    synapse_ops
}
