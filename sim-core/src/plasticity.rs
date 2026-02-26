use crate::brain::{BrainScratch, ACTION_COUNT, ACTION_ID_BASE, INTER_ID_BASE};
use crate::genome::{SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN};
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use sim_types::{BrainState, NeuronId, OrganismState, SynapseEdge};
#[cfg(feature = "profiling")]
use std::time::Instant;

const HEBB_WEIGHT_CLAMP_ENABLED: bool = true;
const DOPAMINE_ENERGY_DELTA_SCALE: f32 = 10.0;
const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;

pub(crate) fn compute_pending_coactivations(
    organism: &mut OrganismState,
    scratch: &mut BrainScratch,
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
            &scratch.action_post_signals,
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
            &scratch.action_post_signals,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
}

fn should_prune_synapses(age_turns: u64, age_of_maturity: u32) -> bool {
    let maturity_ticks = u64::from(age_of_maturity);
    age_turns >= maturity_ticks && age_turns % SYNAPSE_PRUNE_INTERVAL_TICKS == 0
}

pub(crate) fn apply_runtime_weight_updates(
    organism: &mut OrganismState,
    passive_energy_baseline: f32,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let weight_prune_threshold = organism.genome.synapse_prune_threshold.max(0.0);
    let should_prune = should_prune_synapses(organism.age_turns, organism.genome.age_of_maturity);
    let is_mature = organism.age_turns >= u64::from(organism.genome.age_of_maturity);
    let eligibility_retention = organism.genome.eligibility_retention.clamp(0.0, 1.0);
    let energy_delta = organism.energy - organism.energy_prev;
    // Baseline-correct the reward signal so passive metabolism alone is neutral.
    let corrected_energy_delta = energy_delta + passive_energy_baseline.max(0.0);
    let dopamine_signal = (corrected_energy_delta / DOPAMINE_ENERGY_DELTA_SCALE).tanh();
    organism.dopamine = dopamine_signal;
    organism.energy_prev = organism.energy;
    let eta = organism.genome.hebb_eta_gain.max(0.0);
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
            is_mature,
            eligibility_retention,
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
            is_mature,
            eligibility_retention,
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
    action_post_signals: &[f32; ACTION_COUNT],
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
        let Some(post_activation) = action_post_signals.get(idx).copied() else {
            continue;
        };
        edge.pending_coactivation = pre_activation * post_activation;
    }
}

fn compute_pending_inter_edge_coactivations(
    edges: &mut [SynapseEdge],
    pre_idx: usize,
    prev_inter_activations: &[f32],
    inter_activations: &[f32],
    action_post_signals: &[f32; ACTION_COUNT],
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
        let Some(post_activation) = action_post_signals.get(idx).copied() else {
            continue;
        };
        edge.pending_coactivation = pre_current * post_activation;
    }
}

fn apply_edge_weight_update_and_fold_pending(
    edges: &mut [SynapseEdge],
    eta: f32,
    dopamine: f32,
    apply_weight_update: bool,
    eligibility_retention: f32,
) {
    let instantaneous_scale = 1.0 - eligibility_retention;
    for edge in edges {
        if apply_weight_update {
            let updated_weight = edge.weight + eta * dopamine * edge.eligibility
                - PLASTIC_WEIGHT_DECAY * edge.weight;
            edge.weight = constrain_weight(updated_weight);
        }
        edge.eligibility = eligibility_retention * edge.eligibility
            + instantaneous_scale * edge.pending_coactivation;
        edge.pending_coactivation = 0.0;
    }
}

fn constrain_weight(weight: f32) -> f32 {
    if weight == 0.0 {
        return SYNAPSE_STRENGTH_MIN;
    }
    let magnitude = if HEBB_WEIGHT_CLAMP_ENABLED {
        weight
            .abs()
            .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
    } else {
        weight.abs().max(SYNAPSE_STRENGTH_MIN)
    };
    weight.signum() * magnitude
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

fn refresh_parent_ids_and_synapse_count(brain: &mut BrainState) {
    let inter_len = brain.inter.len();
    let action_len = brain.action.len();
    let mut inter_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); inter_len];
    let mut action_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); action_len];

    for sensory in &brain.sensory {
        let pre_id = sensory.neuron.neuron_id;
        for synapse in &sensory.synapses {
            if synapse.post_neuron_id.0 >= INTER_ID_BASE {
                let inter_idx = synapse.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
                if inter_idx < inter_parent_ids.len() {
                    inter_parent_ids[inter_idx].push(pre_id);
                    continue;
                }
            }
            if synapse.post_neuron_id.0 >= ACTION_ID_BASE {
                let action_idx = synapse.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
                if action_idx < action_parent_ids.len() {
                    action_parent_ids[action_idx].push(pre_id);
                }
            }
        }
    }

    for inter in &brain.inter {
        let pre_id = inter.neuron.neuron_id;
        for synapse in &inter.synapses {
            if synapse.post_neuron_id.0 >= INTER_ID_BASE {
                let inter_idx = synapse.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
                if inter_idx < inter_parent_ids.len() {
                    inter_parent_ids[inter_idx].push(pre_id);
                    continue;
                }
            }
            if synapse.post_neuron_id.0 >= ACTION_ID_BASE {
                let action_idx = synapse.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
                if action_idx < action_parent_ids.len() {
                    action_parent_ids[action_idx].push(pre_id);
                }
            }
        }
    }

    for (idx, inter) in brain.inter.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut inter_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        inter.neuron.parent_ids = parents;
    }
    for (idx, action) in brain.action.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut action_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        action.neuron.parent_ids = parents;
    }

    let synapse_count = brain
        .sensory
        .iter()
        .map(|n| n.synapses.len())
        .sum::<usize>()
        + brain.inter.iter().map(|n| n.synapses.len()).sum::<usize>();
    brain.synapse_count = synapse_count as u32;
}

fn split_inter_and_action_edges_mut(
    edges: &mut [SynapseEdge],
) -> (&mut [SynapseEdge], &mut [SynapseEdge]) {
    let split_idx = edges.partition_point(|edge| edge.post_neuron_id.0 < ACTION_ID_BASE);
    edges.split_at_mut(split_idx)
}
