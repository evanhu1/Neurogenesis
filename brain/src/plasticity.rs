use crate::fast_tanh;
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use crate::topology::{
    action_array_index, constrain_weight, inter_index, refresh_action_synapse_starts_and_count,
    ACTION_COUNT,
};
use crate::BrainScratch;
#[cfg(feature = "profiling")]
use std::time::Instant;
use types::{BrainState, OrganismState, SynapseEdge};

const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const PRUNE_ELIGIBILITY_MULTIPLIER: f32 = 2.0;

/// Three-factor (neuromodulated) gating of the Hebbian learning term.
///
/// The pure covariance rule consolidates whatever coactivations happened,
/// regardless of whether the behavior they encode actually paid off — so a
/// brain cannot LEARN within its lifetime that a behavior was rewarded. These
/// constants add a scalar neuromodulator `m`, read from the organism's
/// within-tick energy change (`energy - energy_at_last_sensing`, already
/// persisted in world bytes), that gently scales ONLY the eligibility→weight
/// term: coactivations that preceded an energy GAIN (ate prey/plant) get
/// consolidated a little harder; those before a LOSS get damped a little.
///
/// This is NOT a value function / TD / actor-critic — `m` is a bounded read of
/// an already-existing homeostatic quantity, computed per-organism, so it is
/// order-independent and adds no RNG (determinism preserved).
///
/// GENTLE band: `m = clamp(1 + GAIN * clamp(delta / SCALE, -1, 1), MIN, MAX)`.
/// `GAIN` is small so the modulator never dominates the covariance signal — it
/// biases consolidation toward rewarded coactivations rather than overriding
/// the unsupervised structure. SCALE normalizes the energy delta to roughly
/// [-1, 1] (a typical food/prey intake is on the order of a few energy units);
/// the inner clamp then caps the contribution of any single large gain/loss.
///
/// GAIN was tuned to 0.04 (down from 0.08) by a plasticity-only cross-seed
/// screen: a gentler credit signal lets the unsupervised covariance rule build
/// far richer sensory→action structure (cross-seed mi_sa ~0.13 → ~0.20) without
/// diluting action precision (action_effectiveness held) and while keeping the
/// predator niche (prey rate held). With GAIN this small the modulator stays in
/// [0.96, 1.04], so the MIN/MAX rails are an inert safety bound, not a constraint
/// that bites — the levers that matter are GAIN (strength) and SCALE (saturation).
const NEUROMOD_GAIN: f32 = 0.04;
const NEUROMOD_SCALE: f32 = 5.0;
const NEUROMOD_MIN: f32 = 0.85;
const NEUROMOD_MAX: f32 = 1.15;

/// Bounded energy-delta neuromodulator for the three-factor learning rule.
/// `delta` is the organism's within-tick energy change (post-action energy
/// minus the energy stashed at sensing time this tick).
fn energy_delta_neuromodulator(delta: f32) -> f32 {
    let normalized = (delta / NEUROMOD_SCALE).clamp(-1.0, 1.0);
    (1.0 + NEUROMOD_GAIN * normalized).clamp(NEUROMOD_MIN, NEUROMOD_MAX)
}
/// EMA rate for the per-neuron mean activation used to center pending
/// coactivations (covariance rule). ~20-tick window; tight enough that the
/// mean tracks real drift between biomes, loose enough to smooth
/// tick-to-tick jitter.
const ACTIVATION_MEAN_ALPHA: f32 = 0.05;

struct PlasticityStepParams {
    eta: f32,
    /// Bounded three-factor neuromodulator gating ONLY the learning term
    /// (eligibility→weight). The passive decay term is left un-modulated. See
    /// `energy_delta_neuromodulator`.
    learning_modulator: f32,
    eligibility_retention: f32,
    max_weight_delta_per_tick: f32,
    should_prune: bool,
    weight_prune_threshold: f32,
}

/// Squashed action-neuron logit used as the post-side "activation" for the
/// covariance rule on inter→action edges. Bounding it with the same tanh the
/// inter layer uses keeps action-edge pending magnitudes comparable to
/// inter-edge pending instead of scaling with an unbounded logit.
fn action_activation(logit: f32) -> f32 {
    fast_tanh(logit)
}

pub fn compute_pending_coactivations(organism: &mut OrganismState, scratch: &mut BrainScratch) {
    // Effective learning rate this tick — the same product the consumer
    // (`apply_runtime_weight_updates`) gates on. Age increments between this
    // producer (intents phase) and the consumer (post-commit), so the maturity
    // gate is evaluated at the post-increment age `age_turns + 1` the consumer
    // will observe; both sides of one tick then agree, including on the
    // maturity-boundary tick. When the product is zero (e.g. a juvenile
    // lineage that evolved juvenile_eta_scale == 0) every pending coactivation
    // would be discarded unconditionally, so skip the covariance pass; the
    // activation means keep updating so their semantics are unchanged across
    // the juvenile/mature boundary.
    let effective_eta = organism.genome.plasticity.hebb_eta_gain.max(0.0)
        * learning_rate_scale_at_age(&organism.genome, organism.age_turns + 1);

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let brain = &mut organism.brain;

    // The mean buffers are sized once at expression time; neurons are never
    // added or removed after birth (runtime pruning removes only synapses).
    debug_assert_eq!(brain.sensory_mean_activation.len(), brain.sensory.len());
    debug_assert_eq!(brain.inter_mean_activation.len(), brain.inter.len());
    debug_assert_eq!(brain.action_mean_activation.len(), brain.action.len());

    // Bootstrap uninitialized means to the current activation so the
    // covariance term `(activation - mean) * (post - post_mean)` starts at
    // zero instead of pretending the neuron spent its entire juvenile period
    // firing at zero. Critical when juvenile_eta_scale > 1 and the critical
    // period is doing most of the learning.
    bootstrap_means(brain);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    if effective_eta > 0.0 {
        // Snapshot current-tick inter activations: the inter loop mutates
        // `brain.inter` while reading other inter activations. The means need
        // no snapshot — they are disjoint struct fields, and they are not
        // mutated until `update_activation_means` runs after both loops, so
        // direct borrows yield the pre-update values ("surprise relative to
        // prior expectation").
        scratch.inter_activations.clear();
        scratch
            .inter_activations
            .extend(brain.inter.iter().map(|inter| inter.neuron.activation));
        // Snapshot squashed action logits the same way: the action edges read
        // them while the sensory/inter loops below mutate their synapses.
        let mut action_activations = [0.0_f32; ACTION_COUNT];
        for (slot, action) in action_activations.iter_mut().zip(brain.action.iter()) {
            *slot = action_activation(action.logit);
        }
        let sensory_means = &brain.sensory_mean_activation;
        let inter_means = &brain.inter_mean_activation;
        let action_means = &brain.action_mean_activation;

        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        for (sensory_idx, sensory) in brain.sensory.iter_mut().enumerate() {
            let pre_signal = sensory.neuron.activation;
            // Length equality with the sensory layer is asserted above.
            let pre_mean = sensory_means[sensory_idx];
            compute_pending_edge_coactivations(
                &mut sensory.synapses,
                sensory.action_synapse_start,
                pre_signal,
                pre_mean,
                pre_signal,
                &scratch.inter_activations,
                inter_means,
                &action_activations,
                action_means,
            );
        }
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        // The scratch is always prepared by `evaluate_brain` immediately
        // before this pass: `prepare_inter_buffers` fills `prev_inter` to the
        // inter layer length, and `inter_activations` was just filled above
        // from the same layer. Fail fast instead of silently skipping the
        // inter→inter pass on a misprepared scratch.
        debug_assert_eq!(scratch.prev_inter.len(), brain.inter.len());
        for (pre_idx, inter) in brain.inter.iter_mut().enumerate() {
            let pre_prev = scratch.prev_inter[pre_idx];
            let pre_current = scratch.inter_activations[pre_idx];
            let pre_mean = inter_means[pre_idx];
            compute_pending_edge_coactivations(
                &mut inter.synapses,
                inter.action_synapse_start,
                pre_prev,
                pre_mean,
                pre_current,
                &scratch.inter_activations,
                inter_means,
                &action_activations,
                action_means,
            );
        }
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
    }

    // Fold this tick's activations into the running means. Deferred until
    // after pending is computed so pending reflects the prior expectation.
    update_activation_means(brain);
}

// The mean buffers are allocated at the layer lengths in `express_genome`, so
// the zips below cover every neuron. All means bootstrap together on the
// brain's first compute pass (neurons are never added or removed after
// birth), so a single flag covers the whole brain.
fn bootstrap_means(brain: &mut BrainState) {
    if brain.means_initialized {
        return;
    }
    for (mean, sensory) in brain
        .sensory_mean_activation
        .iter_mut()
        .zip(brain.sensory.iter())
    {
        *mean = sensory.neuron.activation;
    }
    for (mean, inter) in brain
        .inter_mean_activation
        .iter_mut()
        .zip(brain.inter.iter())
    {
        *mean = inter.neuron.activation;
    }
    for (mean, action) in brain
        .action_mean_activation
        .iter_mut()
        .zip(brain.action.iter())
    {
        *mean = action_activation(action.logit);
    }
    brain.means_initialized = true;
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
    for (mean, action) in brain
        .action_mean_activation
        .iter_mut()
        .zip(brain.action.iter())
    {
        *mean = retention * *mean + ACTIVATION_MEAN_ALPHA * action_activation(action.logit);
    }
}

fn should_prune_synapses(age_turns: u64, plasticity_maturity_ticks: u32) -> bool {
    let maturity_ticks = u64::from(plasticity_maturity_ticks);
    age_turns >= maturity_ticks && age_turns.is_multiple_of(SYNAPSE_PRUNE_INTERVAL_TICKS)
}

pub fn apply_runtime_weight_updates(organism: &mut OrganismState) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();

    let params = PlasticityStepParams::from_organism(organism);

    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    if params.eta == 0.0 {
        return;
    }

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &mut organism.brain.sensory {
        apply_edge_weight_update_and_fold_pending(&mut sensory.synapses, &params);
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for inter in &mut organism.brain.inter {
        apply_edge_weight_update_and_fold_pending(&mut inter.synapses, &params);
    }
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

/// Maturity-dependent learning-rate scale at the organism's current age: 1
/// once mature, otherwise the genome's (non-negative) juvenile eta scale.
/// Consumers run after `increment_age_for_survivors`, while the producer
/// (`compute_pending_coactivations`) calls `learning_rate_scale_at_age` with
/// `age_turns + 1` to evaluate the same post-increment age.
pub fn learning_rate_scale(organism: &OrganismState) -> f32 {
    learning_rate_scale_at_age(&organism.genome, organism.age_turns)
}

fn learning_rate_scale_at_age(genome: &types::OrganismGenome, age_turns: u64) -> f32 {
    let is_mature = age_turns >= u64::from(genome.lifecycle.plasticity_maturity_ticks);
    if is_mature {
        1.0
    } else {
        genome.plasticity.juvenile_eta_scale.max(0.0)
    }
}

impl PlasticityStepParams {
    fn from_organism(organism: &OrganismState) -> Self {
        let eta = organism.genome.plasticity.hebb_eta_gain.max(0.0) * learning_rate_scale(organism);

        // Within-tick energy change: post-action energy (this is the post-commit
        // plasticity pass) minus the energy stashed during this tick's sensing
        // pass. Reuses the already-persisted `energy_at_last_sensing` (only ever
        // written at sensing time), so reading it here neither allocates new
        // state used by the sensing pass.
        let energy_delta = organism.energy as f32 - organism.energy_at_last_sensing as f32;
        let learning_modulator = energy_delta_neuromodulator(energy_delta);

        Self {
            eta,
            learning_modulator,
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
                organism.genome.lifecycle.plasticity_maturity_ticks,
            ),
            weight_prune_threshold: organism.genome.plasticity.synapse_prune_threshold.max(0.0),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_pending_edge_coactivations(
    edges: &mut [SynapseEdge],
    action_synapse_start: usize,
    inter_pre_signal: f32,
    inter_pre_mean: f32,
    action_pre_signal: f32,
    inter_activations: &[f32],
    inter_means: &[f32],
    action_activations: &[f32],
    action_means: &[f32],
) {
    // Edges stay partitioned by target role (asserted where the cache is
    // refreshed in `refresh_action_synapse_starts_and_count`), so the cached
    // split separates the inter-targeting prefix from the action-targeting suffix.
    let (inter_edges, action_edges) = edges.split_at_mut(action_synapse_start);

    // Covariance rule on inter-targeting edges:
    //     pending = (pre - pre̅) * (post - post̅)
    // Centering both sides prevents neurons that are always firing at a
    // non-zero baseline from hoarding eligibility on every tick — the signal
    // is the deviation from recent average, not the raw activation.
    let pre_dev = inter_pre_signal - inter_pre_mean;
    let inter_len = inter_activations.len();
    debug_assert_eq!(inter_means.len(), inter_len);
    for edge in inter_edges {
        let Some(idx) = inter_index(edge.post_neuron_id, inter_len) else {
            continue;
        };
        // Safety: `inter_index` already proved `idx < inter_len`, and
        // `express_genome` allocates `inter_means` at the inter layer length
        // (asserted in `compute_pending_coactivations`), matching
        // `inter_activations`.
        let (post_activation, post_mean) = unsafe {
            (
                *inter_activations.get_unchecked(idx),
                *inter_means.get_unchecked(idx),
            )
        };
        edge.pending_coactivation = pre_dev * (post_activation - post_mean);
    }

    // Same centered covariance rule on inter→action edges: the action neuron
    // has no recurrent state, so its squashed logit (`action_activations`)
    // stands in for the post activation, centered by its own running mean.
    // The pre side uses the current activation (`action_pre_signal`) because
    // action logits are accumulated from current-tick inter activations.
    let action_pre_dev = action_pre_signal - inter_pre_mean;
    for edge in action_edges {
        let Some(idx) = action_array_index(edge.post_neuron_id) else {
            continue;
        };
        edge.pending_coactivation = action_pre_dev * (action_activations[idx] - action_means[idx]);
    }
}

fn apply_edge_weight_update_and_fold_pending(
    edges: &mut [SynapseEdge],
    params: &PlasticityStepParams,
) {
    for edge in edges {
        // Three-factor covariance-rule update: the eligibility trace is a
        // decaying sum of centered coactivations, scaled by the maturity-gated
        // learning rate AND a bounded energy-delta neuromodulator so that
        // coactivations preceding an energy gain consolidate harder and those
        // preceding a loss are damped — within-life reward-learning. The
        // passive decay term toward zero is left un-modulated.
        let uncapped_delta = params.learning_modulator * params.eta * edge.eligibility
            - PLASTIC_WEIGHT_DECAY * edge.weight;
        let capped_delta = uncapped_delta.clamp(
            -params.max_weight_delta_per_tick,
            params.max_weight_delta_per_tick,
        );
        let updated_weight = edge.weight + capped_delta;
        edge.weight = constrain_weight(updated_weight);
        // Additive accumulation (decaying sum) instead of EMA — preserves
        // transient signal from zero-mean coactivations.
        edge.eligibility =
            params.eligibility_retention * edge.eligibility + edge.pending_coactivation;
        edge.pending_coactivation = 0.0;
    }
}

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) -> bool {
    let mut pruned_any = false;

    let mut prune_group = |edges: &mut Vec<SynapseEdge>| {
        let before = edges.len();
        edges.retain(|synapse| {
            synapse.weight.abs() >= threshold
                || synapse.eligibility.abs() >= (PRUNE_ELIGIBILITY_MULTIPLIER * threshold)
        });
        pruned_any |= edges.len() != before;
    };
    for sensory in &mut brain.sensory {
        prune_group(&mut sensory.synapses);
    }
    for inter in &mut brain.inter {
        prune_group(&mut inter.synapses);
    }

    if pruned_any {
        refresh_action_synapse_starts_and_count(brain);
    }

    pruned_any
}
