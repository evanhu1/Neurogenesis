//! Within-lifetime Hebbian-covariance plasticity, lifted from
//! `sim-core/src/brain/plasticity.rs` and generalized to `BrainNet`'s flat edge
//! list. The three-factor neuromodulator, centered-covariance eligibility
//! trace, maturity gate, and periodic pruning are preserved; the one addition
//! is the per-edge `plasticity_scale` (CPPN `PLR`) multiplying the learning
//! term (hybrid adaptive-HyperNEAT). `plasticity_scale == 1.0` on every edge
//! reproduces the original rule.
//!
//! Learned weights are runtime-only and are **discarded at reproduction**
//! (non-Lamarckian) — the genome, not the trained brain, is what breeds.

use crate::brain::{constrain_weight, fast_tanh, BrainNet, NeuronKind};
use crate::genome::HeaderGenes;

const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
pub const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const PRUNE_ELIGIBILITY_MULTIPLIER: f32 = 2.0;
const NEUROMOD_GAIN: f32 = 0.04;
const NEUROMOD_SCALE: f32 = 5.0;
const NEUROMOD_MIN: f32 = 0.85;
const NEUROMOD_MAX: f32 = 1.15;
const ACTIVATION_MEAN_ALPHA: f32 = 0.05;

/// Bounded energy-delta neuromodulator — identical to `sim-core`.
pub fn energy_delta_neuromodulator(delta: f32) -> f32 {
    let normalized = (delta / NEUROMOD_SCALE).clamp(-1.0, 1.0);
    (1.0 + NEUROMOD_GAIN * normalized).clamp(NEUROMOD_MIN, NEUROMOD_MAX)
}

/// Maturity-gated learning-rate scale: 1 once mature, else the juvenile scale.
pub fn learning_rate_scale(header: &HeaderGenes, age_turns: u64) -> f32 {
    if age_turns >= u64::from(header.lifecycle.age_of_maturity) {
        1.0
    } else {
        header.plasticity.juvenile_eta_scale.max(0.0)
    }
}

pub struct PlasticityParams {
    pub eta: f32,
    pub learning_modulator: f32,
    pub eligibility_retention: f32,
    pub max_weight_delta_per_tick: f32,
    pub should_prune: bool,
    pub weight_prune_threshold: f32,
}

impl PlasticityParams {
    /// Derive the per-tick parameters from the header, the organism's age, and
    /// its within-tick energy delta (post-action minus energy at sensing).
    pub fn derive(header: &HeaderGenes, age_turns: u64, energy_delta: f32) -> Self {
        let eta = header.plasticity.hebb_eta_gain.max(0.0) * learning_rate_scale(header, age_turns);
        let should_prune = age_turns >= u64::from(header.lifecycle.age_of_maturity)
            && age_turns.is_multiple_of(SYNAPSE_PRUNE_INTERVAL_TICKS);
        PlasticityParams {
            eta,
            learning_modulator: energy_delta_neuromodulator(energy_delta),
            eligibility_retention: header.plasticity.eligibility_retention.clamp(0.0, 1.0),
            max_weight_delta_per_tick: header.plasticity.max_weight_delta_per_tick.max(0.0),
            should_prune,
            weight_prune_threshold: header.plasticity.synapse_prune_threshold.max(0.0),
        }
    }
}

/// Post-side "activation" for the covariance rule: outputs use their squashed
/// logit (they have no recurrent state), hidden/input use their activation.
#[inline]
fn post_activation(brain: &BrainNet, idx: usize) -> f32 {
    let neuron = &brain.neurons[idx];
    match neuron.kind {
        NeuronKind::Output => fast_tanh(neuron.activation),
        _ => neuron.activation,
    }
}

/// One combined plasticity step: fold this tick's centered coactivations into
/// each edge's eligibility trace, apply the neuromodulated weight update, then
/// (periodically) prune. Deterministic and order-independent per organism.
pub fn plasticity_step(brain: &mut BrainNet, params: &PlasticityParams) {
    if !brain.means_initialized {
        for i in 0..brain.neurons.len() {
            brain.neurons[i].mean_activation = post_activation(brain, i);
        }
        brain.means_initialized = true;
    }

    if params.eta > 0.0 {
        for e in 0..brain.edges.len() {
            let from = brain.edges[e].from as usize;
            let to = brain.edges[e].to as usize;
            let pre = brain.neurons[from].activation - brain.neurons[from].mean_activation;
            let post = post_activation(brain, to) - brain.neurons[to].mean_activation;
            brain.edges[e].pending = pre * post;
        }
        for edge in &mut brain.edges {
            let uncapped = edge.plasticity_scale * params.learning_modulator * params.eta
                * edge.eligibility
                - PLASTIC_WEIGHT_DECAY * edge.weight;
            let capped = uncapped.clamp(
                -params.max_weight_delta_per_tick,
                params.max_weight_delta_per_tick,
            );
            edge.weight = constrain_weight(edge.weight + capped);
            edge.eligibility = params.eligibility_retention * edge.eligibility + edge.pending;
            edge.pending = 0.0;
        }
    }

    // Fold this tick's activations into the running means (deferred, so pending
    // reflected the prior expectation).
    for i in 0..brain.neurons.len() {
        let a = post_activation(brain, i);
        let neuron = &mut brain.neurons[i];
        neuron.mean_activation =
            (1.0 - ACTIVATION_MEAN_ALPHA) * neuron.mean_activation + ACTIVATION_MEAN_ALPHA * a;
    }

    if params.should_prune {
        let threshold = params.weight_prune_threshold;
        brain.edges.retain(|e| {
            e.weight.abs() >= threshold
                || e.eligibility.abs() >= PRUNE_ELIGIBILITY_MULTIPLIER * threshold
        });
    }
}
