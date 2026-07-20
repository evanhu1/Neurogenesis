use crate::topology::{constrain_weight, ACTION_COUNT};
use types::{BrainState, Symbol};

const NLMS_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ImmediateLearningNormalization {
    #[default]
    None,
    NormalizedLeastMeanSquares,
}

#[derive(Debug, Clone, Copy)]
pub struct ImmediateLearningRequest {
    pub selected: Symbol,
    pub action_probabilities: [f32; ACTION_COUNT],
    pub reward: f32,
    pub learning_rate: f32,
    pub fast_weight_retention: f32,
    pub max_weight_delta: f32,
    pub normalization: ImmediateLearningNormalization,
}

#[derive(Debug, Clone, Copy)]
pub struct TargetPredictionLearningRequest {
    pub target: Symbol,
    pub action_probabilities: [f32; ACTION_COUNT],
    pub learning_rate: f32,
    pub fast_weight_retention: f32,
    pub max_weight_delta: f32,
    pub normalization: ImmediateLearningNormalization,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ImmediateLearningReport {
    pub edge_update_count: u64,
    pub clipped_update_count: u64,
    pub requested_absolute_delta: f64,
    pub applied_absolute_delta: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalLearningRequest {
    pub selected: Symbol,
    pub reward: f32,
    pub value_prediction: f32,
    pub learning_rate: f32,
    pub eligibility_retention: f32,
    pub fast_weight_retention: f32,
    pub max_weight_delta: f32,
    pub normalization: ImmediateLearningNormalization,
    pub plasticity_enabled: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TemporalLearningReport {
    pub edge_update_count: u64,
    pub clipped_update_count: u64,
    pub requested_absolute_delta: f64,
    pub applied_absolute_delta: f64,
    pub prediction_error: f32,
    pub reward_prediction: f32,
}

/// Reset transient recurrent/output dynamics while preserving runtime synaptic
/// weights learned earlier in the same lifetime.
pub fn reset_dynamics_preserving_weights(brain: &mut BrainState) {
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = 0.0;
    }
    for hidden in &mut brain.inter {
        hidden.neuron.activation = 0.0;
        hidden.state = 0.0;
    }
    for action in &mut brain.action {
        action.logit = 0.0;
    }
    brain.previous_inter_activations.fill(0.0);
    brain.previous_action_activations.fill(0.0);
    brain.previous_prediction_error = 0.0;
}

/// Apply an immediate policy-eligibility update to hidden→action runtime
/// weights. This v0 rule intentionally has no cross-step eligibility
/// trace: dense reward for the current output updates only the action that just
/// occurred.
pub fn apply_immediate_action_reward(
    brain: &mut BrainState,
    request: ImmediateLearningRequest,
    update_deltas: &mut Vec<f32>,
) -> ImmediateLearningReport {
    update_deltas.clear();
    let mut report = ImmediateLearningReport::default();
    let eta = request.learning_rate.max(0.0);
    let fast_retention = request.fast_weight_retention.clamp(0.0, 1.0);
    let max_delta = request.max_weight_delta.max(0.0);

    let eligibility_scale = match request.normalization {
        ImmediateLearningNormalization::None => 1.0,
        ImmediateLearningNormalization::NormalizedLeastMeanSquares => {
            let hidden_state_energy = brain
                .inter
                .iter()
                .map(|hidden| {
                    let activation = hidden.neuron.activation;
                    activation * activation
                })
                .sum::<f32>();
            1.0 / (NLMS_EPSILON + hidden_state_energy)
        }
    };

    for hidden in &mut brain.inter {
        let pre = hidden.neuron.activation;
        for edge in &mut hidden.synapses {
            let Some(action) = Symbol::from_action_neuron_id(edge.post_neuron_id) else {
                continue;
            };
            retain_fast_weight(edge, fast_retention);
            let policy_error = f32::from(action == request.selected)
                - request.action_probabilities[action.index()];
            let requested_delta = eta
                * edge.plasticity_coefficient.max(0.0)
                * request.reward
                * pre
                * eligibility_scale
                * policy_error;
            let capped_delta = requested_delta.clamp(-max_delta, max_delta);
            let previous_weight = edge.weight;
            let proposed_weight = edge.weight + capped_delta;
            edge.weight = constrain_weight(proposed_weight);
            let applied_delta = edge.weight - previous_weight;
            update_deltas.push(applied_delta);
            report.edge_update_count += 1;
            report.clipped_update_count +=
                u64::from(requested_delta.abs() > max_delta || edge.weight != proposed_weight);
            report.requested_absolute_delta += f64::from(requested_delta.abs());
            report.applied_absolute_delta += f64::from(applied_delta.abs());
        }
    }
    report
}

/// Apply the exact softmax cross-entropy output error for an observed target.
///
/// This is the supervised counterpart to `apply_immediate_action_reward`: the
/// environment supplies the correct categorical label, so every enabled
/// hidden-to-action synapse receives the local `(target - probability) * pre`
/// update instead of estimating that vector from a sampled action and reward.
pub fn apply_target_prediction_error(
    brain: &mut BrainState,
    request: TargetPredictionLearningRequest,
) -> ImmediateLearningReport {
    let mut report = ImmediateLearningReport::default();
    let eta = request.learning_rate.max(0.0);
    let fast_retention = request.fast_weight_retention.clamp(0.0, 1.0);
    let max_delta = request.max_weight_delta.max(0.0);

    let eligibility_scale = match request.normalization {
        ImmediateLearningNormalization::None => 1.0,
        ImmediateLearningNormalization::NormalizedLeastMeanSquares => {
            let hidden_state_energy = brain
                .sensory
                .iter()
                .map(|sensory| {
                    let activation = sensory.neuron.activation;
                    activation * activation
                })
                .chain(brain.inter.iter().map(|hidden| {
                    let activation = hidden.neuron.activation;
                    activation * activation
                }))
                .sum::<f32>();
            1.0 / (NLMS_EPSILON + hidden_state_energy)
        }
    };

    for sensory in &mut brain.sensory {
        let pre = sensory.neuron.activation;
        if pre == 0.0 {
            continue;
        }
        apply_target_output_updates(
            &mut sensory.synapses,
            pre,
            request,
            eligibility_scale,
            eta,
            fast_retention,
            max_delta,
            &mut report,
        );
    }
    for hidden in &mut brain.inter {
        let pre = hidden.neuron.activation;
        apply_target_output_updates(
            &mut hidden.synapses,
            pre,
            request,
            eligibility_scale,
            eta,
            fast_retention,
            max_delta,
            &mut report,
        );
    }

    report
}

#[allow(clippy::too_many_arguments)]
fn apply_target_output_updates(
    edges: &mut [types::SynapseEdge],
    pre: f32,
    request: TargetPredictionLearningRequest,
    eligibility_scale: f32,
    eta: f32,
    fast_retention: f32,
    max_delta: f32,
    report: &mut ImmediateLearningReport,
) {
    for edge in edges {
        let Some(action) = Symbol::from_action_neuron_id(edge.post_neuron_id) else {
            continue;
        };
        retain_fast_weight(edge, fast_retention);
        let output_error =
            f32::from(action == request.target) - request.action_probabilities[action.index()];
        let requested_delta =
            eta * edge.plasticity_coefficient.max(0.0) * pre * eligibility_scale * output_error;
        let capped_delta = requested_delta.clamp(-max_delta, max_delta);
        let previous_weight = edge.weight;
        let proposed_weight = edge.weight + capped_delta;
        edge.weight = constrain_weight(proposed_weight);
        let applied_delta = edge.weight - previous_weight;
        report.edge_update_count += 1;
        report.clipped_update_count +=
            u64::from(requested_delta.abs() > max_delta || edge.weight != proposed_weight);
        report.requested_absolute_delta += f64::from(requested_delta.abs());
        report.applied_absolute_delta += f64::from(applied_delta.abs());
    }
}

/// Three-factor temporal-credit update for a sampled action.
///
/// The synaptic eligibility trace records selected-action activity and carries
/// it forward across ticks. A generic value output predicts immediate reward
/// from recurrent state and centers the global signal, producing signed reward
/// surprise. This is not a policy-gradient or REINFORCE score-function update.
/// The selected action and signed surprise are persisted as next-tick
/// efference-copy and neuromodulatory feedback.
pub fn apply_temporal_action_reward(
    brain: &mut BrainState,
    request: TemporalLearningRequest,
) -> TemporalLearningReport {
    let eta = request.learning_rate.clamp(0.0, 1.0);
    let fast_retention = request.fast_weight_retention.clamp(0.0, 1.0);
    let max_delta = request.max_weight_delta.max(0.0);
    let prediction_error = request.reward - request.value_prediction;

    let eligibility_scale = match request.normalization {
        ImmediateLearningNormalization::None => 1.0,
        ImmediateLearningNormalization::NormalizedLeastMeanSquares => {
            let hidden_state_energy = brain
                .inter
                .iter()
                .map(|hidden| {
                    let activation = hidden.neuron.activation;
                    activation * activation
                })
                .sum::<f32>();
            1.0 / (NLMS_EPSILON + hidden_state_energy)
        }
    };
    let retention = request.eligibility_retention.clamp(0.0, 1.0);
    let mut report = TemporalLearningReport {
        prediction_error,
        reward_prediction: request.value_prediction,
        ..TemporalLearningReport::default()
    };

    for hidden in &mut brain.inter {
        let pre = hidden.neuron.activation;
        for edge in &mut hidden.synapses {
            let update_signal =
                if let Some(action) = Symbol::from_action_neuron_id(edge.post_neuron_id) {
                    if request.plasticity_enabled {
                        retain_fast_weight(edge, fast_retention);
                    }
                    // Reward-modulated chosen-action eligibility. Unlike the
                    // softmax score term, this does not vanish when a formerly
                    // correct action is chosen with high confidence after a
                    // reversal. Positive surprise strengthens its active
                    // pathway; negative surprise depresses it immediately.
                    pre * eligibility_scale * f32::from(action == request.selected)
                } else if crate::topology::is_value_neuron_id(edge.post_neuron_id) {
                    if request.plasticity_enabled {
                        retain_fast_weight(edge, fast_retention);
                    }
                    pre * (1.0 - request.value_prediction * request.value_prediction)
                } else {
                    continue;
                };
            edge.eligibility = retention * edge.eligibility + update_signal;
            if !request.plasticity_enabled || eta == 0.0 || max_delta == 0.0 {
                continue;
            }
            let requested_delta =
                eta * edge.plasticity_coefficient.max(0.0) * prediction_error * edge.eligibility;
            let capped_delta = requested_delta.clamp(-max_delta, max_delta);
            let previous_weight = edge.weight;
            let proposed_weight = previous_weight + capped_delta;
            edge.weight = constrain_weight(proposed_weight);
            let applied_delta = edge.weight - previous_weight;
            report.edge_update_count += 1;
            report.clipped_update_count +=
                u64::from(requested_delta.abs() > max_delta || edge.weight != proposed_weight);
            report.requested_absolute_delta += f64::from(requested_delta.abs());
            report.applied_absolute_delta += f64::from(applied_delta.abs());
        }
    }

    let value_bias_signal = 1.0 - request.value_prediction * request.value_prediction;
    brain.value_bias_eligibility = retention * brain.value_bias_eligibility + value_bias_signal;
    if request.plasticity_enabled && eta > 0.0 && max_delta > 0.0 {
        brain.value_bias = brain.inherited_value_bias
            + fast_retention * (brain.value_bias - brain.inherited_value_bias);
        let requested_delta = eta * prediction_error * brain.value_bias_eligibility;
        let capped_delta = requested_delta.clamp(-max_delta, max_delta);
        let previous_bias = brain.value_bias;
        let proposed_bias = (previous_bias + capped_delta).clamp(-1.0, 1.0);
        brain.value_bias = proposed_bias;
        let applied_delta = proposed_bias - previous_bias;
        report.edge_update_count += 1;
        report.clipped_update_count += u64::from(
            requested_delta.abs() > max_delta || proposed_bias != previous_bias + capped_delta,
        );
        report.requested_absolute_delta += f64::from(requested_delta.abs());
        report.applied_absolute_delta += f64::from(applied_delta.abs());
    }

    brain.previous_action_activations.fill(0.0);
    brain.previous_action_activations[request.selected.index()] = 1.0;
    brain.previous_prediction_error = prediction_error;
    report
}

#[inline]
fn retain_fast_weight(edge: &mut types::SynapseEdge, retention: f32) {
    edge.weight = edge.inherited_weight + retention * (edge.weight - edge.inherited_weight);
}
