use crate::brain::fast_tanh;
use crate::plasticity::learning_rate_scale;
use sim_types::{BrainState, OrganismState};

const VALUE_DISCOUNT_GAMMA: f32 = 0.99;
const VALUE_LEARNING_RATE: f32 = 0.01;
const VALUE_WEIGHT_CLAMP: f32 = 5.0;

fn value_feature_count(brain: &BrainState) -> usize {
    brain.sensory.len() + brain.inter.len()
}

/// Canonical value-head feature layout: sensory activations first, then inter.
fn feature_activations(brain: &BrainState) -> impl Iterator<Item = f32> + '_ {
    brain
        .sensory
        .iter()
        .map(|sensory| sensory.neuron.activation)
        .chain(brain.inter.iter().map(|inter| inter.neuron.activation))
}

fn write_value_features(brain: &BrainState, dst: &mut Vec<f32>) {
    dst.clear();
    dst.extend(feature_activations(brain));
}

// V(s) = Σ w_i · feature_i over current sensory + inter activations. Every
// live brain comes from `express_genome`, which sizes `value_weights` to the
// feature count; neuron counts never change after birth, so the lengths
// always agree.
fn compute_value_estimate(brain: &BrainState) -> f32 {
    debug_assert_eq!(brain.value_weights.len(), value_feature_count(brain));
    brain
        .value_weights
        .iter()
        .zip(feature_activations(brain))
        .map(|(weight, feature)| weight * feature)
        .sum()
}

// Actor-critic TD-error dopamine: δ = r_{t-1} + γ·V(s_t) − V(s_{t-1}).
// The reward stashed from the previous tick is the result of the action chosen
// in s_{t-1}, so the TD error, the critic regression target, and the freshest
// eligibility component (the previous tick's policy gradient) all describe the
// same transition. A state-conditional V (not an EMA baseline) keeps
// plasticity reinforcing learned behaviors even as reward becomes routine.
//
// Semi-gradient TD(0) evaluates both value terms under the same (current)
// weight vector, so V(s_{t-1}) is recomputed here from the stashed feature
// vector rather than reusing the estimate cached before the previous tick's
// weight update — caching would mix two weight generations and re-inject a
// correlated fraction of the previous TD error every tick.
//
// Also runs one semi-gradient descent step on V(s_{t-1}) toward
// r_{t-1} + γ·V(s_t); ∂V/∂w_i is the previous value-head feature activation.
// Skipped on the organism's first tick, when no previous-state stash exists
// yet. Once written, the stash length always equals the weight length: both
// derive from the same brain, whose neuron counts are fixed at birth.
//
// Rolls the per-organism stashes (features, reward) forward so next tick can
// form the t → t+1 transition.
//
// Returns the tanh-squashed dopamine signal for downstream synapse plasticity.
pub(crate) fn step_actor_critic(organism: &mut OrganismState, raw_reward: f32) -> f32 {
    let v_current = compute_value_estimate(&organism.brain);
    // Empty stash (first tick) sums to 0.0, matching the pre-transition
    // baseline.
    let v_prev: f32 = organism
        .brain
        .value_weights
        .iter()
        .zip(organism.value_prev_feature_activations.iter())
        .map(|(weight, feature)| weight * feature)
        .sum();
    let td_error = organism.reward_prev + VALUE_DISCOUNT_GAMMA * v_current - v_prev;
    let dopamine_signal = fast_tanh(td_error);

    if organism.value_prev_feature_activations.is_empty() {
        // First tick: no transition to learn from yet; just seed the stash.
        write_value_features(
            &organism.brain,
            &mut organism.value_prev_feature_activations,
        );
    } else {
        debug_assert_eq!(
            organism.value_prev_feature_activations.len(),
            organism.brain.value_weights.len()
        );
        let lr = VALUE_LEARNING_RATE * learning_rate_scale(organism);
        // Fused pass: gradient step on V(s_{t-1}) using the stashed previous
        // feature, then roll the same stash slot forward to the current
        // feature — no separate feature-chain traversal for the stash.
        let BrainState {
            sensory,
            inter,
            value_weights,
            ..
        } = &mut organism.brain;
        let features = sensory
            .iter()
            .map(|sensory| sensory.neuron.activation)
            .chain(inter.iter().map(|inter| inter.neuron.activation));
        for ((weight, stashed), feature) in value_weights
            .iter_mut()
            .zip(organism.value_prev_feature_activations.iter_mut())
            .zip(features)
        {
            *weight =
                (*weight + lr * td_error * *stashed).clamp(-VALUE_WEIGHT_CLAMP, VALUE_WEIGHT_CLAMP);
            *stashed = feature;
        }
    }

    // No longer a TD-error input (V(s_{t-1}) is recomputed above); kept as
    // the wire-visible V(s) readout for the inspector.
    organism.value_prev = v_current;
    organism.reward_prev = raw_reward;

    dopamine_signal
}
