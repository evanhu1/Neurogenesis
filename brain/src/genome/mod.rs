use crate::topology::{
    constrain_weight, ACTION_COUNT, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT,
};

/// Every runtime ID from `INTER_ID_BASE` onward is available to a hidden node
/// except the small stable action-ID island. `inter_neuron_id` skips that
/// island, so action IDs no longer impose the old 1,000-hidden-node ceiling.
pub const MAX_INTER_NEURONS: u32 = u32::MAX - INTER_ID_BASE + 1 - ACTION_COUNT as u32;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::cmp::Ordering;
use std::f32::consts::LN_10;
use types::{
    action_gene_node_id, action_gene_node_index, connection_innovation_id, is_hidden_gene_node_id,
    seed_hidden_gene_node_id, sensory_gene_node_id, sensory_gene_node_index, ActionType,
    BrainTopology, GeneNodeId, HiddenNodeGene, InnovationId, LifecycleGenes, OrganismGenome,
    PlasticityGenes, SeedGenomeConfig, SensoryReceptor, SynapseGene, SynapseTiming,
};

mod sanitization;
mod scalar;
mod seed;
mod synapse_creation;

use sanitization::debug_assert_genome_well_formed;
pub use sanitization::{
    align_genome_vectors, connection_would_create_cycle, enforce_feed_forward_edges,
};
pub use scalar::inter_alpha_from_log_time_constant;
pub use seed::generate_seed_genome;

fn align_vec_to<T>(values: &mut Vec<T>, target_len: usize, mut fill: impl FnMut() -> T) {
    while values.len() < target_len {
        values.push(fill());
    }
    values.truncate(target_len);
}

pub const SYNAPSE_STRENGTH_MAX: f32 = 1.5;
pub const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
pub const INTER_TIME_CONSTANT_MIN: f32 = 0.1;
pub const INTER_TIME_CONSTANT_MAX: f32 = 10.0;
pub const INTER_LOG_TIME_CONSTANT_MIN: f32 = -LN_10;
pub const INTER_LOG_TIME_CONSTANT_MAX: f32 = LN_10;
pub const DEFAULT_INTER_LOG_TIME_CONSTANT: f32 = -1.203_972_8;
const SYNAPSE_WEIGHT_LOG_NORMAL_MU: f32 = -0.5;
const SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA: f32 = 0.8;
const INITIAL_SYNAPSE_EXCITATORY_PROBABILITY: f32 = 0.8;

fn sample_initial_log_time_constant<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    perturb_clamped(
        DEFAULT_INTER_LOG_TIME_CONSTANT,
        0.5,
        INTER_LOG_TIME_CONSTANT_MIN,
        INTER_LOG_TIME_CONSTANT_MAX,
        rng,
    )
}

fn sample_initial_bias<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    perturb_clamped(
        0.0,
        BIAS_PERTURBATION_STDDEV * 2.0,
        -BIAS_MAX,
        BIAS_MAX,
        rng,
    )
}

fn perturb_clamped<R: Rng + ?Sized>(
    value: f32,
    stddev: f32,
    min: f32,
    max: f32,
    rng: &mut R,
) -> f32 {
    let normal = standard_normal(rng);
    (value + normal * stddev).clamp(min, max)
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    StandardNormal.sample(rng)
}

fn max_possible_synapses(num_neurons: usize, predation_enabled: bool) -> usize {
    let sensory = SensoryReceptor::active(predation_enabled).count();
    let actions = ActionType::active(predation_enabled).count();
    sensory
        .saturating_mul(num_neurons.saturating_add(actions))
        .saturating_add(num_neurons.saturating_mul(actions))
        .saturating_add(num_neurons.saturating_mul(num_neurons.saturating_sub(1)) / 2)
        .saturating_add(num_neurons.saturating_mul(num_neurons))
}

pub fn restrict_action_genes(genome: &mut OrganismGenome, predation_enabled: bool) {
    genome.brain.edges.retain(|edge| {
        !gene_node_is_disabled_action(edge.pre_node_id, predation_enabled)
            && !gene_node_is_disabled_action(edge.post_node_id, predation_enabled)
    });
    for (index, action) in ActionType::ALL.iter().copied().enumerate() {
        if !action.is_enabled(predation_enabled) {
            if let Some(bias) = genome.brain.action_biases.get_mut(index) {
                *bias = 0.0;
            }
        }
    }
}

fn gene_node_is_disabled_action(node_id: GeneNodeId, predation_enabled: bool) -> bool {
    action_gene_node_index(node_id)
        .and_then(|index| ActionType::ALL.get(index).copied())
        .is_some_and(|action| !action.is_enabled(predation_enabled))
}
