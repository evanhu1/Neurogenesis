use crate::brain::BrainScratch;
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

const DOPAMINE_ENERGY_DELTA_SCALE: f32 = 20.0;
const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const PRUNE_ELIGIBILITY_MULTIPLIER: f32 = 2.0;

struct PlasticityStepParams {
    dopamine_signal: f32,
    eta: f32,
    apply_weight_update: bool,
    eligibility_retention: f32,
    max_weight_delta_per_tick: f32,
    should_prune: bool,
    weight_prune_threshold: f32,
}

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
        compute_pending_edge_coactivations(
            &mut sensory.synapses,
            sensory.neuron.activation,
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
pub(crate) fn apply_runtime_weight_updates(
    organism: &mut OrganismState,
    reward_ledger: RewardLedger,
) {
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
    let params = PlasticityStepParams::from_organism(
        organism,
        reward_ledger,
        juvenile_plasticity_enabled,
        reward_signal_multiplier,
    );
    organism.dopamine = params.dopamine_signal;
    organism.energy_prev = organism.energy;
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
        prune_low_weight_synapses(&mut organism.brain, params.weight_prune_threshold);
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityPrune, stage_started.elapsed());
    }
}

impl PlasticityStepParams {
    fn from_organism(
        organism: &OrganismState,
        reward_ledger: RewardLedger,
        juvenile_plasticity_enabled: bool,
        reward_signal_multiplier: f32,
    ) -> Self {
        let is_mature = organism.age_turns >= u64::from(organism.genome.age_of_maturity);
        let plasticity_started =
            organism.age_turns >= u64::from(organism.genome.plasticity_start_age);
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

        Self {
            dopamine_signal: ((reward_ledger.reward_signal() * reward_signal_multiplier)
                / DOPAMINE_ENERGY_DELTA_SCALE)
                .tanh(),
            eta,
            apply_weight_update: plasticity_started && (juvenile_plasticity_enabled || is_mature),
            eligibility_retention: organism.genome.eligibility_retention.clamp(0.0, 1.0),
            max_weight_delta_per_tick: organism.genome.max_weight_delta_per_tick.max(0.0),
            should_prune: should_prune_synapses(
                organism.age_turns,
                organism.genome.age_of_maturity,
            ),
            weight_prune_threshold: organism.genome.synapse_prune_threshold.max(0.0),
        }
    }
}

fn compute_pending_edge_coactivations(
    edges: &mut [SynapseEdge],
    inter_pre_signal: f32,
    action_pre_signal: f32,
    inter_activations: &[f32],
    scratch: &BrainScratch,
    executed_action_credit: bool,
) {
    let (inter_edges, action_edges) = split_inter_and_action_edges_mut(edges);

    for edge in inter_edges {
        let Some(idx) = inter_index(edge.post_neuron_id, inter_activations.len()) else {
            continue;
        };
        let Some(post_activation) = inter_activations.get(idx).copied() else {
            continue;
        };
        edge.pending_coactivation = inter_pre_signal * post_activation;
    }

    for edge in action_edges {
        let Some(idx) = action_array_index(edge.post_neuron_id) else {
            continue;
        };
        edge.pending_coactivation = if executed_action_credit {
            if Some(idx) == scratch.selected_action_index {
                action_pre_signal * scratch.selected_action_confidence
            } else {
                0.0
            }
        } else {
            action_pre_signal * scratch.centered_action_post_signals[idx]
        };
    }
}

fn apply_edge_weight_update_and_fold_pending(
    edges: &mut [SynapseEdge],
    params: &PlasticityStepParams,
) {
    let instantaneous_scale = 1.0 - params.eligibility_retention;
    for edge in edges {
        if params.apply_weight_update {
            let uncapped_delta = params.eta * params.dopamine_signal * edge.eligibility
                - PLASTIC_WEIGHT_DECAY * edge.weight;
            let capped_delta = uncapped_delta.clamp(
                -params.max_weight_delta_per_tick,
                params.max_weight_delta_per_tick,
            );
            let updated_weight = edge.weight + capped_delta;
            edge.weight = constrain_weight(updated_weight);
        }
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

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) {
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
}
