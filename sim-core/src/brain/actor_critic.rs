use crate::brain::fast_tanh;
use sim_types::{BrainState, OrganismState};

const VALUE_DISCOUNT_GAMMA: f32 = 0.99;
const VALUE_LEARNING_RATE: f32 = 0.01;
const VALUE_WEIGHT_CLAMP: f32 = 5.0;

fn value_feature_count(brain: &BrainState) -> usize {
    brain.sensory.len() + brain.inter.len()
}

fn write_value_features(brain: &BrainState, dst: &mut Vec<f32>) {
    let sensory_len = brain.sensory.len();
    dst.resize(value_feature_count(brain), 0.0);
    for (slot, sensory) in dst.iter_mut().take(sensory_len).zip(brain.sensory.iter()) {
        *slot = sensory.neuron.activation;
    }
    for (slot, inter) in dst.iter_mut().skip(sensory_len).zip(brain.inter.iter()) {
        *slot = inter.neuron.activation;
    }
}

// V(s) = Σ w_i · feature_i over current sensory + inter activations. Returns 0
// when the weight vector is out of sync with the current feature vector — can
// happen transiently after inter-layer mutations between ticks.
fn compute_value_estimate(brain: &BrainState) -> f32 {
    if brain.value_weights.len() != value_feature_count(brain) {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    let sensory_len = brain.sensory.len();
    for (weight, sensory) in brain
        .value_weights
        .iter()
        .take(sensory_len)
        .zip(brain.sensory.iter())
    {
        sum += weight * sensory.neuron.activation;
    }
    for (weight, inter) in brain
        .value_weights
        .iter()
        .skip(sensory_len)
        .zip(brain.inter.iter())
    {
        sum += weight * inter.neuron.activation;
    }
    sum
}

// Actor-critic TD-error dopamine: δ = r_t + γ·V(s_t) − V(s_{t-1}).
// A state-conditional V (not an EMA baseline) keeps plasticity reinforcing
// learned behaviors even as reward becomes routine.
//
// Also runs one semi-gradient descent step on V(s_{t-1}) toward r + γ·V(s_t);
// ∂V/∂w_i is the previous value-head feature activation. Skipped when the
// stash is missing or its size disagrees with the current weight vector
// (post-mutation transient).
//
// Rolls the per-organism stash forward so next tick can form V(s_{t+1}) − V(s_t),
// and resizes `value_weights` to match the current feature vector if a mutation
// resized it between ticks.
//
// Returns the tanh-squashed dopamine signal for downstream synapse plasticity.
pub(crate) fn step_actor_critic(organism: &mut OrganismState, raw_reward: f32) -> f32 {
    let v_current = compute_value_estimate(&organism.brain);
    let td_error = raw_reward + VALUE_DISCOUNT_GAMMA * v_current - organism.value_prev;
    let dopamine_signal = fast_tanh(td_error);

    if !organism.value_prev_feature_activations.is_empty()
        && organism.value_prev_feature_activations.len() == organism.brain.value_weights.len()
    {
        let is_mature = organism.age_turns >= u64::from(organism.genome.lifecycle.age_of_maturity);
        let lr = VALUE_LEARNING_RATE
            * if is_mature {
                1.0
            } else {
                organism.genome.plasticity.juvenile_eta_scale.max(0.0)
            };
        for (weight, pre_activation) in organism
            .brain
            .value_weights
            .iter_mut()
            .zip(organism.value_prev_feature_activations.iter().copied())
        {
            let updated = (*weight + lr * td_error * pre_activation)
                .clamp(-VALUE_WEIGHT_CLAMP, VALUE_WEIGHT_CLAMP);
            *weight = updated;
        }
    }

    if organism.brain.value_weights.len() != value_feature_count(&organism.brain) {
        organism
            .brain
            .value_weights
            .resize(value_feature_count(&organism.brain), 0.0);
    }

    organism.value_prev = v_current;
    write_value_features(&organism.brain, &mut organism.value_prev_feature_activations);

    dopamine_signal
}
