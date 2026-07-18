use crate::topology::{constrain_weight, ACTION_COUNT};
use types::{BrainState, Symbol};

#[derive(Debug, Default, Clone, Copy)]
pub struct ImmediateLearningReport {
    pub edge_update_count: u64,
    pub clipped_update_count: u64,
    pub requested_absolute_delta: f64,
    pub applied_absolute_delta: f64,
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
}

/// Apply an immediate, local policy-eligibility update to hidden→action
/// runtime weights. This v0 rule intentionally has no cross-step eligibility
/// trace: dense reward for the current output updates only the action that just
/// occurred.
pub fn apply_immediate_action_reward(
    brain: &mut BrainState,
    selected: Symbol,
    action_probabilities: [f32; ACTION_COUNT],
    reward: f32,
    learning_rate: f32,
    max_weight_delta: f32,
    update_deltas: &mut Vec<f32>,
) -> ImmediateLearningReport {
    update_deltas.clear();
    let mut report = ImmediateLearningReport::default();
    let eta = learning_rate.max(0.0);
    let max_delta = max_weight_delta.max(0.0);
    if eta == 0.0 || max_delta == 0.0 || reward == 0.0 {
        return report;
    }

    for hidden in &mut brain.inter {
        let pre = hidden.neuron.activation;
        for edge in &mut hidden.synapses {
            let Some(action) = Symbol::from_action_neuron_id(edge.post_neuron_id) else {
                continue;
            };
            let policy_error = f32::from(action == selected) - action_probabilities[action.index()];
            let requested_delta = eta
                * edge.plasticity_coefficient.max(0.0)
                * reward
                * pre
                * policy_error;
            let delta = requested_delta.clamp(-max_delta, max_delta);
            edge.weight = constrain_weight(edge.weight + delta);
            update_deltas.push(delta);
            report.edge_update_count += 1;
            report.clipped_update_count += u64::from(delta != requested_delta);
            report.requested_absolute_delta += f64::from(requested_delta.abs());
            report.applied_absolute_delta += f64::from(delta.abs());
        }
    }
    report
}
