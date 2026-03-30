use crate::brain::{
    constrain_weight, refresh_parent_ids_and_synapse_count, BrainScratch, ACTION_ID_BASE,
    INTER_ID_BASE,
};
use crate::RewardLedger;
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use sim_types::{BrainState, OrganismState, SynapseEdge};
#[cfg(feature = "profiling")]
use std::time::Instant;

const DOPAMINE_ENERGY_DELTA_SCALE: f32 = 20.0;
const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;

pub(crate) fn compute_pending_coactivations(
    organism: &mut OrganismState,
    scratch: &mut BrainScratch,
    executed_action_credit: bool,
) {
    if organism.genome.hebb_eta_gain <= 0.0 {
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
        compute_pending_sensory_edge_coactivations(
            &mut sensory.synapses,
            sensory.neuron.activation,
            &scratch.inter_activations,
            scratch,
            executed_action_credit,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (pre_idx, inter) in brain.inter.iter_mut().enumerate() {
        compute_pending_inter_edge_coactivations(
            &mut inter.synapses,
            pre_idx,
            &scratch.prev_inter,
            &scratch.inter_activations,
            scratch,
            executed_action_credit,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
}

fn should_prune_synapses(age_turns: u64, age_of_maturity: u32) -> bool {
    let maturity_ticks = u64::from(age_of_maturity);
    age_turns >= maturity_ticks && age_turns % SYNAPSE_PRUNE_INTERVAL_TICKS == 0
}

#[allow(dead_code)]
pub(crate) fn apply_runtime_weight_updates(organism: &mut OrganismState, reward_ledger: RewardLedger) {
    apply_runtime_weight_updates_with_mode(organism, reward_ledger, true, 1.0);
}

pub(crate) fn apply_runtime_weight_updates_with_mode(
    organism: &mut OrganismState,
    reward_ledger: RewardLedger,
    juvenile_plasticity_enabled: bool,
    reward_signal_multiplier: f32,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let weight_prune_threshold = organism.genome.synapse_prune_threshold.max(0.0);
    let should_prune = should_prune_synapses(organism.age_turns, organism.genome.age_of_maturity);
    let is_mature = organism.age_turns >= u64::from(organism.genome.age_of_maturity);
    let plasticity_started = organism.age_turns >= u64::from(organism.genome.plasticity_start_age);
    let eligibility_retention = organism.genome.eligibility_retention.clamp(0.0, 1.0);
    let dopamine_signal = ((reward_ledger.reward_signal() * reward_signal_multiplier)
        / DOPAMINE_ENERGY_DELTA_SCALE)
        .tanh();
    organism.dopamine = dopamine_signal;
    organism.energy_prev = organism.energy;
    let juvenile_eta_scale = organism.genome.juvenile_eta_scale.max(0.0);
    let eta = if !juvenile_plasticity_enabled {
        if is_mature {
            organism.genome.hebb_eta_gain.max(0.0)
        } else {
            0.0
        }
    } else if is_mature {
        organism.genome.hebb_eta_gain.max(0.0)
    } else {
        organism.genome.hebb_eta_gain.max(0.0) * juvenile_eta_scale
    };
    let max_weight_delta_per_tick = organism.genome.max_weight_delta_per_tick.max(0.0);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    if eta == 0.0 {
        return;
    }

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &mut organism.brain.sensory {
        apply_edge_weight_update_and_fold_pending(
            &mut sensory.synapses,
            eta,
            dopamine_signal,
            plasticity_started && (juvenile_plasticity_enabled || is_mature),
            eligibility_retention,
            max_weight_delta_per_tick,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for inter in &mut organism.brain.inter {
        apply_edge_weight_update_and_fold_pending(
            &mut inter.synapses,
            eta,
            dopamine_signal,
            plasticity_started && (juvenile_plasticity_enabled || is_mature),
            eligibility_retention,
            max_weight_delta_per_tick,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());

    if should_prune {
        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        prune_low_weight_synapses(&mut organism.brain, weight_prune_threshold);
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityPrune, stage_started.elapsed());
    }
}

fn compute_pending_sensory_edge_coactivations(
    edges: &mut [SynapseEdge],
    pre_activation: f32,
    inter_activations: &[f32],
    scratch: &BrainScratch,
    executed_action_credit: bool,
) {
    let (inter_edges, action_edges) = split_inter_and_action_edges_mut(edges);

    for edge in inter_edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
        let Some(post_activation) = inter_activations.get(idx).copied() else {
            continue;
        };
        edge.pending_coactivation = pre_activation * post_activation;
    }

    for edge in action_edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
        edge.pending_coactivation = if executed_action_credit {
            if Some(idx) == scratch.selected_action_index {
                pre_activation * scratch.selected_action_confidence
            } else {
                0.0
            }
        } else {
            pre_activation * scratch.centered_action_post_signals[idx]
        };
    }
}

fn compute_pending_inter_edge_coactivations(
    edges: &mut [SynapseEdge],
    pre_idx: usize,
    prev_inter_activations: &[f32],
    inter_activations: &[f32],
    scratch: &BrainScratch,
    executed_action_credit: bool,
) {
    let Some(pre_prev) = prev_inter_activations.get(pre_idx).copied() else {
        return;
    };
    let Some(pre_current) = inter_activations.get(pre_idx).copied() else {
        return;
    };

    let (inter_edges, action_edges) = split_inter_and_action_edges_mut(edges);

    for edge in inter_edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
        let Some(post_activation) = inter_activations.get(idx).copied() else {
            continue;
        };
        edge.pending_coactivation = pre_prev * post_activation;
    }

    for edge in action_edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
        edge.pending_coactivation = if executed_action_credit {
            if Some(idx) == scratch.selected_action_index {
                pre_current * scratch.selected_action_confidence
            } else {
                0.0
            }
        } else {
            pre_current * scratch.centered_action_post_signals[idx]
        };
    }
}

fn apply_edge_weight_update_and_fold_pending(
    edges: &mut [SynapseEdge],
    eta: f32,
    dopamine: f32,
    apply_weight_update: bool,
    eligibility_retention: f32,
    max_weight_delta_per_tick: f32,
) {
    let instantaneous_scale = 1.0 - eligibility_retention;
    for edge in edges {
        if apply_weight_update {
            let uncapped_delta =
                eta * dopamine * edge.eligibility - PLASTIC_WEIGHT_DECAY * edge.weight;
            let capped_delta =
                uncapped_delta.clamp(-max_weight_delta_per_tick, max_weight_delta_per_tick);
            let updated_weight = edge.weight + capped_delta;
            edge.weight = constrain_weight(updated_weight);
        }
        edge.eligibility = eligibility_retention * edge.eligibility
            + instantaneous_scale * edge.pending_coactivation;
        edge.pending_coactivation = 0.0;
    }
}

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) {
    let mut pruned_any = false;

    for sensory in &mut brain.sensory {
        let before = sensory.synapses.len();
        sensory.synapses.retain(|synapse| {
            synapse.weight.abs() >= threshold || synapse.eligibility.abs() >= (2.0f32 * threshold)
        });
        pruned_any |= sensory.synapses.len() != before;
    }
    for inter in &mut brain.inter {
        let before = inter.synapses.len();
        inter.synapses.retain(|synapse| {
            synapse.weight.abs() >= threshold || synapse.eligibility.abs() >= (2.0f32 * threshold)
        });
        pruned_any |= inter.synapses.len() != before;
    }

    if pruned_any {
        refresh_parent_ids_and_synapse_count(brain);
    }
}

fn split_inter_and_action_edges_mut(
    edges: &mut [SynapseEdge],
) -> (&mut [SynapseEdge], &mut [SynapseEdge]) {
    let split_idx = edges.partition_point(|edge| edge.post_neuron_id.0 < ACTION_ID_BASE);
    edges.split_at_mut(split_idx)
}
