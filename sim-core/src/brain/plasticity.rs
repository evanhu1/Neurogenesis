use crate::brain::{fast_tanh, BrainScratch};
use crate::metabolism::refresh_organism_base_metabolic_cost;
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use crate::topology::{
    action_array_index, constrain_weight, inter_index, refresh_parent_ids_and_synapse_count,
    split_inter_and_action_edges_mut,
};
use crate::RewardLedger;
use sim_types::{BrainState, OrganismState, SynapseEdge};
#[cfg(feature = "profiling")]
use std::time::Instant;

const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const PRUNE_ELIGIBILITY_MULTIPLIER: f32 = 2.0;
/// Discount factor for TD-error target: `δ = r + γ·V(s_{t+1}) − V(s_t)`.
const VALUE_DISCOUNT_GAMMA: f32 = 0.95;
/// Learning rate for the value head's local gradient step on TD-error.
const VALUE_LEARNING_RATE: f32 = 0.01;
/// Max absolute magnitude for any single value-head weight.
const VALUE_WEIGHT_CLAMP: f32 = 5.0;

struct PlasticityStepParams {
    dopamine_signal: f32,
    eta: f32,
    eligibility_retention: f32,
    max_weight_delta_per_tick: f32,
    should_prune: bool,
    weight_prune_threshold: f32,
}

pub(crate) fn compute_pending_coactivations(
    organism: &mut OrganismState,
    scratch: &mut BrainScratch,
) {
    if organism.genome.plasticity.hebb_eta_gain <= 0.0 {
        return;
    }

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
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
        compute_pending_edge_coactivations(
            &mut sensory.synapses,
            sensory.neuron.activation,
            sensory.neuron.activation,
            &scratch.inter_activations,
            scratch,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (pre_idx, inter) in brain.inter.iter_mut().enumerate() {
        let Some(pre_prev) = scratch.prev_inter.get(pre_idx).copied() else {
            continue;
        };
        let Some(pre_current) = scratch.inter_activations.get(pre_idx).copied() else {
            continue;
        };
        compute_pending_edge_coactivations(
            &mut inter.synapses,
            pre_prev,
            pre_current,
            &scratch.inter_activations,
            scratch,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
}

fn should_prune_synapses(age_turns: u64, age_of_maturity: u32) -> bool {
    let maturity_ticks = u64::from(age_of_maturity);
    age_turns >= maturity_ticks && age_turns.is_multiple_of(SYNAPSE_PRUNE_INTERVAL_TICKS)
}

#[cfg(test)]
pub(crate) fn apply_runtime_weight_updates(
    organism: &mut OrganismState,
    reward_ledger: RewardLedger,
) {
    apply_runtime_weight_updates_with_multiplier(organism, reward_ledger, 1.0);
}

/// Linear value estimate V(s) = Σ w_i · activation_i over current inter neurons.
/// Returns 0 when the weight vector size disagrees with the inter layer
/// (which can happen transiently if mutations resize the layer between ticks).
fn compute_value_estimate(brain: &BrainState) -> f32 {
    if brain.value_weights.len() != brain.inter.len() {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for (weight, inter) in brain.value_weights.iter().zip(brain.inter.iter()) {
        sum += weight * inter.neuron.activation;
    }
    sum
}

pub(crate) fn apply_runtime_weight_updates_with_multiplier(
    organism: &mut OrganismState,
    reward_ledger: RewardLedger,
    reward_signal_multiplier: f32,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();

    // Actor-critic TD-error dopamine:
    //     δ = r_t + γ·V(s_t) − V(s_{t-1})
    // V(s) is a learned linear projection of current inter activations via
    // `brain.value_weights`. V(s_{t-1}) is stashed in `organism.value_prev`.
    // The TD-error replaces the raw reward signal as the dopamine input, then
    // goes through the same tanh squash to keep dopamine magnitude comparable
    // to the baseline.
    //
    // A *state-conditional* value (unlike an EMA baseline) lets plasticity
    // keep reinforcing learned behaviors even as reward becomes routine,
    // because V adapts its prediction to the current state rather than to
    // the running mean.
    let v_current = compute_value_estimate(&organism.brain);
    let raw_reward = reward_ledger.weighted_reward_signal(&organism.genome.reward_weights)
        * reward_signal_multiplier;
    let td_error = raw_reward + VALUE_DISCOUNT_GAMMA * v_current - organism.value_prev;
    // Unit-normalized tonic ledger signals give td_error in roughly [-2, +2];
    // feeding directly into tanh gives good dynamic range without saturating
    // on tonic baseline or collapsing phasic pulses.
    let dopamine_signal = fast_tanh(td_error);

    let mut params =
        PlasticityStepParams::from_organism(organism, reward_ledger, reward_signal_multiplier);
    params.dopamine_signal = dopamine_signal;
    organism.dopamine = dopamine_signal;
    organism.energy_prev = organism.energy;
    organism.health_prev = organism.health;

    // Local semi-gradient descent step on V(s_{t-1}) toward `r + γ·V(s_t)`.
    // ∂V/∂w_i at the previous state equals the previous inter activation.
    // Size must match; skip on the first tick when no stash exists or after
    // inter-layer mutations changed the layer size.
    if !organism.value_prev_inter_activations.is_empty()
        && organism.value_prev_inter_activations.len() == organism.brain.value_weights.len()
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
            .zip(organism.value_prev_inter_activations.iter().copied())
        {
            let updated = (*weight + lr * td_error * pre_activation)
                .clamp(-VALUE_WEIGHT_CLAMP, VALUE_WEIGHT_CLAMP);
            *weight = updated;
        }
    }

    // Keep value_weights aligned with the inter layer in case mutations
    // resized it between ticks.
    if organism.brain.value_weights.len() != organism.brain.inter.len() {
        organism
            .brain
            .value_weights
            .resize(organism.brain.inter.len(), 0.0);
    }

    // Stash current-tick state so next tick can form V(s_{t+1}) - V(s_t).
    organism.value_prev = v_current;
    organism.value_prev_inter_activations.clear();
    organism
        .value_prev_inter_activations
        .extend(organism.brain.inter.iter().map(|i| i.neuron.activation));

    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    if params.eta == 0.0 {
        return;
    }

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for_each_sensory_synapse_group_mut(&mut organism.brain, |edges| {
        apply_edge_weight_update_and_fold_pending(edges, &params);
    });
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for_each_inter_synapse_group_mut(&mut organism.brain, |edges| {
        apply_edge_weight_update_and_fold_pending(edges, &params);
    });
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());

    if params.should_prune {
        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        let pruned_any =
            prune_low_weight_synapses(&mut organism.brain, params.weight_prune_threshold);
        if pruned_any {
            refresh_organism_base_metabolic_cost(organism);
        }
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityPrune, stage_started.elapsed());
    }
}

impl PlasticityStepParams {
    fn from_organism(
        organism: &OrganismState,
        reward_ledger: RewardLedger,
        reward_signal_multiplier: f32,
    ) -> Self {
        let is_mature = organism.age_turns >= u64::from(organism.genome.lifecycle.age_of_maturity);
        let juvenile_eta_scale = organism.genome.plasticity.juvenile_eta_scale.max(0.0);
        let eta = if is_mature {
            organism.genome.plasticity.hebb_eta_gain.max(0.0)
        } else {
            organism.genome.plasticity.hebb_eta_gain.max(0.0) * juvenile_eta_scale
        };

        Self {
            dopamine_signal: fast_tanh(
                reward_ledger.weighted_reward_signal(&organism.genome.reward_weights)
                    * reward_signal_multiplier,
            ),
            eta,
            eligibility_retention: organism.genome.plasticity.eligibility_retention.clamp(0.0, 1.0),
            max_weight_delta_per_tick: organism.genome.plasticity.max_weight_delta_per_tick.max(0.0),
            should_prune: should_prune_synapses(
                organism.age_turns,
                organism.genome.lifecycle.age_of_maturity,
            ),
            weight_prune_threshold: organism.genome.plasticity.synapse_prune_threshold.max(0.0),
        }
    }
}

fn compute_pending_edge_coactivations(
    edges: &mut [SynapseEdge],
    inter_pre_signal: f32,
    action_pre_signal: f32,
    inter_activations: &[f32],
    scratch: &BrainScratch,
) {
    if inter_pre_signal == 0.0 && action_pre_signal == 0.0 {
        return;
    }

    let (inter_edges, action_edges) = split_inter_and_action_edges_mut(edges);

    if inter_pre_signal != 0.0 {
        for edge in inter_edges {
            let Some(idx) = inter_index(edge.post_neuron_id, inter_activations.len()) else {
                continue;
            };
            let Some(post_activation) = inter_activations.get(idx).copied() else {
                continue;
            };
            edge.pending_coactivation = inter_pre_signal * post_activation;
        }
    }

    if action_pre_signal != 0.0 {
        for edge in action_edges {
            let Some(idx) = action_array_index(edge.post_neuron_id) else {
                continue;
            };
            edge.pending_coactivation = if Some(idx) == scratch.selected_action_index {
                action_pre_signal * scratch.selected_action_confidence
            } else {
                0.0
            };
        }
    }
}

fn apply_edge_weight_update_and_fold_pending(
    edges: &mut [SynapseEdge],
    params: &PlasticityStepParams,
) {
    let instantaneous_scale = 1.0 - params.eligibility_retention;
    for edge in edges {
        let uncapped_delta = params.eta * params.dopamine_signal * edge.eligibility
            - PLASTIC_WEIGHT_DECAY * edge.weight;
        let capped_delta = uncapped_delta.clamp(
            -params.max_weight_delta_per_tick,
            params.max_weight_delta_per_tick,
        );
        let updated_weight = edge.weight + capped_delta;
        edge.weight = constrain_weight(updated_weight);
        edge.eligibility = params.eligibility_retention * edge.eligibility
            + instantaneous_scale * edge.pending_coactivation;
        edge.pending_coactivation = 0.0;
    }
}

fn for_each_sensory_synapse_group_mut(
    brain: &mut BrainState,
    mut visitor: impl FnMut(&mut [SynapseEdge]),
) {
    for sensory in &mut brain.sensory {
        visitor(&mut sensory.synapses);
    }
}

fn for_each_inter_synapse_group_mut(
    brain: &mut BrainState,
    mut visitor: impl FnMut(&mut [SynapseEdge]),
) {
    for inter in &mut brain.inter {
        visitor(&mut inter.synapses);
    }
}

fn for_each_synapse_group_vec_mut(
    brain: &mut BrainState,
    mut visitor: impl FnMut(&mut Vec<SynapseEdge>),
) {
    for sensory in &mut brain.sensory {
        visitor(&mut sensory.synapses);
    }
    for inter in &mut brain.inter {
        visitor(&mut inter.synapses);
    }
}

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) -> bool {
    let mut pruned_any = false;

    for_each_synapse_group_vec_mut(brain, |edges| {
        let before = edges.len();
        edges.retain(|synapse| {
            synapse.weight.abs() >= threshold
                || synapse.eligibility.abs() >= (PRUNE_ELIGIBILITY_MULTIPLIER * threshold)
        });
        pruned_any |= edges.len() != before;
    });

    if pruned_any {
        refresh_parent_ids_and_synapse_count(brain);
    }

    pruned_any
}
