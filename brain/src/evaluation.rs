use super::*;

#[derive(Clone, Copy)]
struct ActionAvailability {
    predation_enabled: bool,
}

#[derive(Clone, Copy)]
pub struct BrainEvalContext {
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
    pub action_temperature: f32,
    pub action_samples: [f32; 3],
    pub compositional_actions_enabled: bool,
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
    // Every recurrent edge reads the same frozen previous-tick activation
    // vector before any current hidden activation is evaluated.
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
        debug_assert!(pre_index < brain.inter.len() && post_index < brain.inter.len());
        scratch.inter_inputs[post_index] +=
            edge.weight * brain.previous_inter_activations[pre_index];
        result.synapse_ops += 1;
    }
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
    // and actions. Previous-tick signals have already been accumulated from the
    // recurrent edge array above; leaky neuron state remains orthogonal.
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
    let (sampled_action, sampled_action_mask) = if context.compositional_actions_enabled {
        sample_compositional_actions(
            result.action_logits,
            EXPLICIT_IDLE_LOGIT_BIAS,
            context.action_temperature,
            context.action_samples,
            context.predation_enabled,
            &mut scratch.action_probabilities,
        )
    } else {
        let action = sample_action_from_logits(
            result.action_logits,
            EXPLICIT_IDLE_LOGIT_BIAS,
            context.action_temperature,
            context.action_samples[0],
            context.predation_enabled,
            &mut scratch.action_probabilities,
        );
        (action, action.command_bit())
    };
    scratch.selected_action_index = match sampled_action {
        ActionType::Idle => None,
        selected_action => Some(action_index(selected_action)),
    };
    result.selected_action = sampled_action;
    result.selected_action_mask = sampled_action_mask;
    // Publish current hidden activations only after all logits and actions are
    // resolved. The next tick's recurrent pass therefore observes one coherent
    // snapshot, independent of current DAG evaluation order.
    if !brain.recurrent_synapses.is_empty() {
        for (saved, inter) in brain
            .previous_inter_activations
            .iter_mut()
            .zip(brain.inter.iter())
        {
            *saved = inter.neuron.activation;
        }
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionActivationResolve, stage_started.elapsed());

    result
}

fn sample_compositional_actions(
    action_logits: [f32; ACTION_COUNT],
    idle_bias: f32,
    action_temperature: f32,
    action_samples: [f32; 3],
    predation_enabled: bool,
    action_probabilities: &mut [f32; ACTION_COUNT],
) -> (ActionType, u8) {
    action_probabilities.fill(0.0);
    let groups: [&[usize]; 3] = [&[0, 1], &[2], &[3]];
    let mut mask = 0_u8;
    for (group, sample) in groups.into_iter().zip(action_samples) {
        let selected = sample_action_group(
            action_logits,
            group,
            idle_bias,
            action_temperature,
            sample,
            ActionAvailability { predation_enabled },
            action_probabilities,
        );
        mask |= selected.command_bit();
    }

    // Preserve a single observer/plasticity projection without affecting the
    // emitted command set. Highest raw logit wins; declaration order breaks
    // ties deterministically.
    let primary = ActionType::ALL
        .iter()
        .copied()
        .filter(|action| mask & action.command_bit() != 0)
        .max_by(|left, right| {
            action_logits[action_index(*left)]
                .total_cmp(&action_logits[action_index(*right)])
                .then_with(|| right.index().cmp(&left.index()))
        })
        .unwrap_or(ActionType::Idle);
    (primary, mask)
}

fn sample_action_group(
    action_logits: [f32; ACTION_COUNT],
    indices: &[usize],
    idle_bias: f32,
    action_temperature: f32,
    action_sample: f32,
    availability: ActionAvailability,
    action_probabilities: &mut [f32; ACTION_COUNT],
) -> ActionType {
    let temperature = action_temperature.max(MIN_ACTION_TEMPERATURE);
    let max_logit = indices
        .iter()
        .copied()
        .filter(|idx| ActionType::ALL[*idx].is_enabled(availability.predation_enabled))
        .map(|idx| action_logits[idx])
        .fold(idle_bias, f32::max);
    let mut weights = [0.0_f32; ACTION_COUNT];
    let mut weight_sum = ((idle_bias - max_logit) / temperature).exp();
    for idx in indices.iter().copied() {
        if !ActionType::ALL[idx].is_enabled(availability.predation_enabled) {
            continue;
        }
        let weight = ((action_logits[idx] - max_logit) / temperature).exp();
        weights[idx] = weight;
        weight_sum += weight;
    }
    for idx in indices.iter().copied() {
        action_probabilities[idx] = weights[idx] / weight_sum;
    }

    let sample = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * weight_sum;
    let mut cumulative = 0.0_f32;
    for idx in indices.iter().copied() {
        cumulative += weights[idx];
        if sample < cumulative {
            return ActionType::ALL[idx];
        }
    }
    ActionType::Idle
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
