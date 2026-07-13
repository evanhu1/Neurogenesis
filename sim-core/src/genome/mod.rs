use crate::topology::{
    constrain_weight, ACTION_COUNT, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT,
};

/// Every runtime ID from `INTER_ID_BASE` onward is available to a hidden node
/// except the small stable action-ID island. `inter_neuron_id` skips that
/// island, so action IDs no longer impose the old 1,000-hidden-node ceiling.
pub(crate) const MAX_INTER_NEURONS: u32 = u32::MAX - INTER_ID_BASE + 1 - ACTION_COUNT as u32;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use sim_types::{
    action_gene_node_id, action_gene_node_index, connection_innovation_id, is_hidden_gene_node_id,
    seed_hidden_gene_node_id, sensory_gene_node_id, sensory_gene_node_index, ActionType,
    BrainTopology, GeneNodeId, HiddenNodeGene, InnovationId, LifecycleGenes, NeuronId,
    OrganismGenome, PlasticityGenes, SeedGenomeConfig, SensoryReceptor, SynapseGene, TopologyGenes,
};
use std::cmp::Ordering;
use std::f32::consts::LN_10;

mod sanitization;
mod scalar;
mod seed;
mod synapse_creation;

pub(crate) use sanitization::align_genome_vectors;
use sanitization::debug_assert_genome_well_formed;
pub(crate) use scalar::inter_alpha_from_log_time_constant;
pub(crate) use seed::generate_seed_genome;

fn align_vec_to<T>(values: &mut Vec<T>, target_len: usize, mut fill: impl FnMut() -> T) {
    while values.len() < target_len {
        values.push(fill());
    }
    values.truncate(target_len);
}

pub(crate) const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
pub(crate) const MAX_MUTATED_VISION_DISTANCE: u32 = 10;
pub(crate) const SYNAPSE_STRENGTH_MAX: f32 = 1.5;
pub(crate) const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
pub(crate) const INTER_TIME_CONSTANT_MIN: f32 = 0.1;
pub(crate) const INTER_TIME_CONSTANT_MAX: f32 = 10.0;
pub(crate) const INTER_LOG_TIME_CONSTANT_MIN: f32 = -LN_10;
pub(crate) const INTER_LOG_TIME_CONSTANT_MAX: f32 = LN_10;
pub(crate) const DEFAULT_INTER_LOG_TIME_CONSTANT: f32 = -1.203_972_8;
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
    // Inter-neuron self-edges are valid (see `is_valid_synapse_pair`), so the
    // full (pre, post) cross product is reachable.
    let pre_count = SensoryReceptor::active(predation_enabled).count() + num_neurons;
    let post_count = num_neurons + ActionType::active(predation_enabled).count();
    pre_count.saturating_mul(post_count)
}

pub(crate) fn restrict_predation_genes(genome: &mut OrganismGenome, predation_enabled: bool) {
    if predation_enabled {
        return;
    }
    genome.brain.edges.retain(|edge| {
        !gene_node_is_predation_only(edge.pre_node_id)
            && !gene_node_is_predation_only(edge.post_node_id)
    });
    let attack_index = ActionType::ALL
        .iter()
        .position(|action| *action == ActionType::Attack)
        .expect("Attack is a canonical action");
    if let Some(bias) = genome.brain.action_biases.get_mut(attack_index) {
        *bias = 0.0;
    }
}

fn gene_node_is_predation_only(node_id: GeneNodeId) -> bool {
    if let Some(index) = sensory_gene_node_index(node_id) {
        return SensoryReceptor::from_neuron_id(NeuronId(index))
            .is_some_and(SensoryReceptor::is_predation_only);
    }
    action_gene_node_index(node_id)
        .and_then(|index| ActionType::ALL.get(index).copied())
        .is_some_and(|action| action == ActionType::Attack)
}
