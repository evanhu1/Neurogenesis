use crate::actor_critic::step_actor_critic;
use crate::brain::BrainScratch;
use crate::metabolism::refresh_organism_base_metabolic_cost;
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use crate::topology::{
    action_array_index, constrain_weight, inter_index, refresh_parent_ids_and_synapse_count,
    split_inter_and_action_edges_mut, ACTION_COUNT,
};
use crate::RewardLedger;
use sim_types::{BrainState, OrganismState, SynapseEdge};
#[cfg(feature = "profiling")]
use std::time::Instant;

const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const PRUNE_ELIGIBILITY_MULTIPLIER: f32 = 2.0;
/// EMA rate for the per-neuron mean activation used to center pending
/// coactivations (covariance rule). ~20-tick window; tight enough that the
/// mean tracks real drift between biomes, loose enough to smooth
/// tick-to-tick jitter.
const ACTIVATION_MEAN_ALPHA: f32 = 0.05;

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

    // Keep per-neuron mean buffers sized to the current layer layout —
    // add-neuron / add-sensor mutations can change these between ticks.
    sync_mean_buffers(brain);

    // Bootstrap uninitialized means to the current activation so the
    // covariance term `(activation - mean) * (post - post_mean)` starts at
    // zero instead of pretending the neuron spent its entire juvenile period
    // firing at zero. Critical when juvenile_eta_scale > 1 and the critical
    // period is doing most of the learning.
    bootstrap_means(brain);

    // Snapshot current-tick activations and pre-update means. Using the
    // pre-update mean gives "surprise relative to prior expectation" — the
    // EMA absorbs this tick's activations only after pending is computed.
    scratch.inter_activations.clear();
    scratch
        .inter_activations
        .extend(brain.inter.iter().map(|inter| inter.neuron.activation));
    scratch.inter_means.clear();
    scratch
        .inter_means
        .extend_from_slice(&brain.inter_mean_activation);
    scratch.sensory_means.clear();
    scratch
        .sensory_means
        .extend_from_slice(&brain.sensory_mean_activation);
    let selected_action_index = scratch.selected_action_index;
    let action_probabilities = scratch.action_probabilities;
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (sensory_idx, sensory) in brain.sensory.iter_mut().enumerate() {
        let pre_signal = sensory.neuron.activation;
        let pre_mean = scratch
            .sensory_means
            .get(sensory_idx)
            .copied()
            .unwrap_or(0.0);
        compute_pending_edge_coactivations(
            &mut sensory.synapses,
            pre_signal,
            pre_mean,
            pre_signal,
            &scratch.inter_activations,
            &scratch.inter_means,
            selected_action_index,
            &action_probabilities,
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
        let pre_mean = scratch.inter_means.get(pre_idx).copied().unwrap_or(0.0);
        compute_pending_edge_coactivations(
            &mut inter.synapses,
            pre_prev,
            pre_mean,
            pre_current,
            &scratch.inter_activations,
            &scratch.inter_means,
            selected_action_index,
            &action_probabilities,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());

    // Fold this tick's activations into the running means. Deferred until
    // after pending is computed so pending reflects the prior expectation.
    update_activation_means(brain);
}

fn sync_mean_buffers(brain: &mut BrainState) {
    if brain.sensory_mean_activation.len() != brain.sensory.len() {
        brain
            .sensory_mean_activation
            .resize(brain.sensory.len(), 0.0);
    }
    if brain.sensory_mean_initialized.len() != brain.sensory.len() {
        brain
            .sensory_mean_initialized
            .resize(brain.sensory.len(), false);
    }
    if brain.inter_mean_activation.len() != brain.inter.len() {
        brain.inter_mean_activation.resize(brain.inter.len(), 0.0);
    }
    if brain.inter_mean_initialized.len() != brain.inter.len() {
        brain
            .inter_mean_initialized
            .resize(brain.inter.len(), false);
    }
}

fn bootstrap_means(brain: &mut BrainState) {
    for (idx, initialized) in brain.sensory_mean_initialized.iter_mut().enumerate() {
        if !*initialized {
            if let (Some(mean_slot), Some(sensory)) = (
                brain.sensory_mean_activation.get_mut(idx),
                brain.sensory.get(idx),
            ) {
                *mean_slot = sensory.neuron.activation;
                *initialized = true;
            }
        }
    }
    for (idx, initialized) in brain.inter_mean_initialized.iter_mut().enumerate() {
        if !*initialized {
            if let (Some(mean_slot), Some(inter)) = (
                brain.inter_mean_activation.get_mut(idx),
                brain.inter.get(idx),
            ) {
                *mean_slot = inter.neuron.activation;
                *initialized = true;
            }
        }
    }
}

fn update_activation_means(brain: &mut BrainState) {
    let retention = 1.0 - ACTIVATION_MEAN_ALPHA;
    for (mean, sensory) in brain
        .sensory_mean_activation
        .iter_mut()
        .zip(brain.sensory.iter())
    {
        *mean = retention * *mean + ACTIVATION_MEAN_ALPHA * sensory.neuron.activation;
    }
    for (mean, inter) in brain
        .inter_mean_activation
        .iter_mut()
        .zip(brain.inter.iter())
    {
        *mean = retention * *mean + ACTIVATION_MEAN_ALPHA * inter.neuron.activation;
    }
}

fn should_prune_synapses(age_turns: u64, age_of_maturity: u32) -> bool {
    let maturity_ticks = u64::from(age_of_maturity);
    age_turns >= maturity_ticks && age_turns.is_multiple_of(SYNAPSE_PRUNE_INTERVAL_TICKS)
}

pub(crate) fn apply_runtime_weight_updates(
    organism: &mut OrganismState,
    reward_ledger: RewardLedger,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();

    let raw_reward = reward_ledger.weighted_reward_signal(&organism.genome.reward_weights);
    let dopamine_signal = step_actor_critic(organism, raw_reward);

    let params = PlasticityStepParams::from_organism(organism, dopamine_signal);
    organism.dopamine = dopamine_signal;
    organism.energy_prev = organism.energy;
    organism.health_prev = organism.health;

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
    fn from_organism(organism: &OrganismState, dopamine_signal: f32) -> Self {
        let is_mature = organism.age_turns >= u64::from(organism.genome.lifecycle.age_of_maturity);
        let juvenile_eta_scale = organism.genome.plasticity.juvenile_eta_scale.max(0.0);
        let eta = if is_mature {
            organism.genome.plasticity.hebb_eta_gain.max(0.0)
        } else {
            organism.genome.plasticity.hebb_eta_gain.max(0.0) * juvenile_eta_scale
        };

        Self {
            dopamine_signal,
            eta,
            eligibility_retention: organism
                .genome
                .plasticity
                .eligibility_retention
                .clamp(0.0, 1.0),
            max_weight_delta_per_tick: organism
                .genome
                .plasticity
                .max_weight_delta_per_tick
                .max(0.0),
            should_prune: should_prune_synapses(
                organism.age_turns,
                organism.genome.lifecycle.age_of_maturity,
            ),
            weight_prune_threshold: organism.genome.plasticity.synapse_prune_threshold.max(0.0),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_pending_edge_coactivations(
    edges: &mut [SynapseEdge],
    inter_pre_signal: f32,
    inter_pre_mean: f32,
    action_pre_signal: f32,
    inter_activations: &[f32],
    inter_means: &[f32],
    selected_action_index: Option<usize>,
    action_probabilities: &[f32; ACTION_COUNT],
) {
    let (inter_edges, action_edges) = split_inter_and_action_edges_mut(edges);

    // Covariance rule on inter-targeting edges:
    //     pending = (pre - pre̅) * (post - post̅)
    // Centering both sides prevents neurons that are always firing at a
    // non-zero baseline from hoarding eligibility on every tick — the signal
    // is the deviation from recent average, not the raw activation.
    let pre_dev = inter_pre_signal - inter_pre_mean;
    let inter_len = inter_activations.len();
    let inter_means_slice = inter_means;
    for edge in inter_edges {
        let Some(idx) = inter_index(edge.post_neuron_id, inter_len) else {
            continue;
        };
        // Safety: `inter_index` already proved `idx < inter_len`, and
        // `inter_means_slice` is the same logical buffer (same length when in
        // sync; falls back to 0 when shorter, matching the unwrap_or below).
        let post_activation = unsafe { *inter_activations.get_unchecked(idx) };
        let post_mean = inter_means_slice.get(idx).copied().unwrap_or(0.0);
        edge.pending_coactivation = pre_dev * (post_activation - post_mean);
    }

    // Action edges keep the boundary-layer `(A_i - P_i)` structure — the
    // policy is discrete and mutually exclusive, so the advantage-style term
    // already centers post-side signal around the softmax-probability
    // baseline.
    if action_pre_signal != 0.0 {
        for edge in action_edges {
            let Some(idx) = action_array_index(edge.post_neuron_id) else {
                continue;
            };
            let a_i = if Some(idx) == selected_action_index {
                1.0
            } else {
                0.0
            };
            let p_i = action_probabilities[idx];
            edge.pending_coactivation = action_pre_signal * (a_i - p_i);
        }
    } else {
        for edge in action_edges {
            edge.pending_coactivation = 0.0;
        }
    }
}

fn apply_edge_weight_update_and_fold_pending(
    edges: &mut [SynapseEdge],
    params: &PlasticityStepParams,
) {
    for edge in edges {
        let uncapped_delta = params.eta * params.dopamine_signal * edge.eligibility
            - PLASTIC_WEIGHT_DECAY * edge.weight;
        let capped_delta = uncapped_delta.clamp(
            -params.max_weight_delta_per_tick,
            params.max_weight_delta_per_tick,
        );
        let updated_weight = edge.weight + capped_delta;
        edge.weight = constrain_weight(updated_weight);
        // Additive accumulation (decaying sum) instead of EMA — preserves
        // transient signal from zero-mean coactivation/policy gradients.
        edge.eligibility = params.eligibility_retention * edge.eligibility
            + edge.pending_coactivation;
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
