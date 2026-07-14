use super::*;

#[derive(Clone, Copy)]
pub struct BrainEvalContext {
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
    pub action_temperature: f32,
    pub action_sample: f32,
}

pub fn evaluate_brain(
    organism: &mut types::OrganismState,
    context: BrainEvalContext,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

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
        // Outgoing edges are explicitly partitioned by target role, so inter
        // targets form a prefix and action targets a suffix even when a hidden
        // runtime ID lies numerically above the stable action-ID island.
        let (inter_edges, action_edges) = sensory.synapses.split_at(sensory.action_synapse_start);
        result.synapse_ops += accumulate_inter_inputs(
            inter_edges,
            sensory.neuron.activation,
            &mut scratch.inter_inputs,
        );
        result.synapse_ops +=
            accumulate_action_inputs(action_edges, sensory.neuron.activation, &mut action_inputs);
    }
    // Hidden neurons are stored in deterministic topological order. Compute
    // each activation, then immediately propagate it to later hidden neurons
    // and actions. No previous-tick activation participates unless the explicit
    // leaky-state option is enabled.
    for idx in 0..brain.inter.len() {
        let activation = {
            let neuron = &mut brain.inter[idx];
            neuron.state = if context.leaky_neurons_enabled {
                let alpha = neuron.alpha;
                (1.0 - alpha) * neuron.state + alpha * scratch.inter_inputs[idx]
            } else {
                scratch.inter_inputs[idx]
            };
            neuron.neuron.activation = super::fast_tanh(neuron.state);
            neuron.neuron.activation
        };
        let inter = &brain.inter[idx];
        let (inter_edges, action_edges) = inter.synapses.split_at(inter.action_synapse_start);
        debug_assert!(inter_edges.iter().all(|edge| {
            crate::topology::inter_index(edge.post_neuron_id, brain.inter.len())
                .is_some_and(|post_index| post_index > idx)
        }));
        result.synapse_ops +=
            accumulate_inter_inputs(inter_edges, activation, &mut scratch.inter_inputs);
        result.synapse_ops +=
            accumulate_action_inputs(action_edges, activation, &mut action_inputs);
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterAccumulation, stage_started.elapsed());

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
        context.predation_enabled,
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
    predation_enabled: bool,
    action_probabilities: &mut [f32; ACTION_COUNT],
) -> ActionType {
    let temperature = action_temperature.max(MIN_ACTION_TEMPERATURE);
    let max_logit = action_logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, _)| ActionType::ALL[*idx].is_enabled(predation_enabled))
        .map(|(_, logit)| logit)
        .fold(idle_bias, f32::max);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = 0.0_f32;
    for (idx, logit) in action_logits.iter().copied().enumerate() {
        if !ActionType::ALL[idx].is_enabled(predation_enabled) {
            action_probabilities[idx] = 0.0;
            continue;
        }
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
