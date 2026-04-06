use crate::topology::{
    action_array_index, action_neuron_id, constrain_weight, inter_index, inter_neuron_id,
    is_action_id, is_inter_id, is_sensory_id, ACTION_COUNT, ACTION_COUNT_U32, INTER_ID_BASE,
    SENSORY_COUNT,
};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use sim_types::{BrainLocation, NeuronId, OrganismGenome, SeedGenomeConfig, SynapseEdge};
use std::cmp::Ordering;
use std::collections::HashSet;

mod mutation_rates;
mod sanitization;
mod scalar;
mod seed;
mod spatial_prior;
mod topology;

use mutation_rates::{effective_mutation_rates, mutate_mutation_rate_genes};
use sanitization::{align_genome_vectors, sync_synapse_genes_to_target};
pub(crate) use scalar::inter_alpha_from_log_time_constant;
use scalar::{
    mutate_inter_biases, mutate_inter_update_rates, mutate_random_neuron_location,
    mutate_synapse_weights,
};
pub(crate) use seed::generate_seed_genome;
pub(crate) use topology::{
    mutate_add_neuron_split_edge, mutate_add_synapse, mutate_remove_neuron, mutate_remove_synapse,
};

const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const MIN_MUTATED_AGE_OF_MATURITY: u32 = 0;
const MAX_MUTATED_AGE_OF_MATURITY: u32 = 10_000;
const MIN_MUTATED_GESTATION_TICKS: u8 = 0;
const MAX_MUTATED_GESTATION_TICKS: u8 = 4;
const MIN_MUTATED_MAX_ORGANISM_AGE: u32 = 1;
const MAX_MUTATED_MAX_ORGANISM_AGE: u32 = 100_000;
const MIN_MUTATED_MAX_HEALTH: f32 = 1.0;
const MAX_MUTATED_MAX_HEALTH: f32 = 1_000_000_000.0;
pub(crate) const SYNAPSE_STRENGTH_MAX: f32 = 1.5;
pub(crate) const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;
const ELIGIBILITY_RETENTION_MIN: f32 = 0.0;
const ELIGIBILITY_RETENTION_MAX: f32 = 1.0;
const SYNAPSE_PRUNE_THRESHOLD_MIN: f32 = 0.0;
const SYNAPSE_PRUNE_THRESHOLD_MAX: f32 = 1.0;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
const INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV: f32 = 0.05;
const ELIGIBILITY_RETENTION_PERTURBATION_STDDEV: f32 = 0.05;
const SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV: f32 = 0.02;
const MAX_HEALTH_PERTURBATION_STDDEV: f32 = 25.0;
const INTER_BIAS_PERTURB_NEURON_RATE: f32 = 0.8;
const INTER_UPDATE_RATE_PERTURB_NEURON_RATE: f32 = 0.8;
const SYNAPSE_WEIGHT_PERTURBATION_STDDEV: f32 = 0.15;
const SYNAPSE_WEIGHT_PERTURB_EDGE_RATE: f32 = 0.8;
const SYNAPSE_WEIGHT_REPLACEMENT_RATE: f32 = 0.1;
const LOCATION_PERTURBATION_STDDEV: f32 = 0.75;
pub(crate) const INTER_TIME_CONSTANT_MIN: f32 = 0.1;
pub(crate) const INTER_TIME_CONSTANT_MAX: f32 = 10.0;
pub(crate) const INTER_LOG_TIME_CONSTANT_MIN: f32 = -2.302_585_1;
pub(crate) const INTER_LOG_TIME_CONSTANT_MAX: f32 = 2.302_585_1;
pub(crate) const DEFAULT_INTER_LOG_TIME_CONSTANT: f32 = -1.203_972_8;
pub(crate) const BRAIN_SPACE_MIN: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MAX: f32 = 10.0;
const SPATIAL_PRIOR_LONG_RANGE_FLOOR: f32 = 0.01;
const SYNAPSE_WEIGHT_LOG_NORMAL_MU: f32 = -0.5;
const SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA: f32 = 0.8;
const INITIAL_SYNAPSE_EXCITATORY_PROBABILITY: f32 = 0.8;

pub(crate) fn mutate_genome<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    global_mutation_rate_modifier: f32,
    meta_mutation_enabled: bool,
    rng: &mut R,
) {
    align_genome_vectors(genome, rng);
    if meta_mutation_enabled {
        mutate_mutation_rate_genes(genome, rng);
    }
    let rates = effective_mutation_rates(genome, global_mutation_rate_modifier);

    if rng.random::<f32>() < rates.age_of_maturity {
        genome.age_of_maturity = step_u32(
            genome.age_of_maturity,
            MIN_MUTATED_AGE_OF_MATURITY,
            MAX_MUTATED_AGE_OF_MATURITY,
            rng,
        );
    }
    if rng.random::<f32>() < rates.gestation_ticks {
        genome.gestation_ticks = step_u8(
            genome.gestation_ticks,
            MIN_MUTATED_GESTATION_TICKS,
            MAX_MUTATED_GESTATION_TICKS,
            rng,
        );
    }
    if rng.random::<f32>() < rates.max_organism_age {
        genome.max_organism_age = step_u32(
            genome.max_organism_age,
            MIN_MUTATED_MAX_ORGANISM_AGE,
            MAX_MUTATED_MAX_ORGANISM_AGE,
            rng,
        );
    }
    if rng.random::<f32>() < rates.vision_distance {
        genome.vision_distance = step_u32(
            genome.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }
    if rng.random::<f32>() < rates.max_health {
        genome.max_health = perturb_clamped(
            genome.max_health,
            MAX_HEALTH_PERTURBATION_STDDEV,
            MIN_MUTATED_MAX_HEALTH,
            MAX_MUTATED_MAX_HEALTH,
            rng,
        );
    }
    if rng.random::<f32>() < rates.inter_bias {
        mutate_inter_biases(genome, rng);
    }
    if rng.random::<f32>() < rates.inter_update_rate {
        mutate_inter_update_rates(genome, rng);
    }
    if rng.random::<f32>() < rates.eligibility_retention {
        genome.eligibility_retention = perturb_clamped(
            genome.eligibility_retention,
            ELIGIBILITY_RETENTION_PERTURBATION_STDDEV,
            ELIGIBILITY_RETENTION_MIN,
            ELIGIBILITY_RETENTION_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < rates.synapse_prune_threshold {
        genome.synapse_prune_threshold = perturb_clamped(
            genome.synapse_prune_threshold,
            SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV,
            SYNAPSE_PRUNE_THRESHOLD_MIN,
            SYNAPSE_PRUNE_THRESHOLD_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < rates.neuron_location {
        mutate_random_neuron_location(genome, rng);
    }
    if rng.random::<f32>() < rates.synapse_weight_perturbation {
        mutate_synapse_weights(genome, rng);
    }
    if rng.random::<f32>() < rates.add_synapse {
        mutate_add_synapse(genome, rng);
    }
    if rng.random::<f32>() < rates.remove_synapse {
        mutate_remove_synapse(genome, rng);
    }
    if rng.random::<f32>() < rates.remove_neuron {
        mutate_remove_neuron(genome, rng);
    }
    if rng.random::<f32>() < rates.add_neuron_split_edge {
        mutate_add_neuron_split_edge(genome, rng);
    }

    sync_synapse_genes_to_target(genome, rng);
}

fn align_vec_to<T>(values: &mut Vec<T>, target_len: usize, mut fill: impl FnMut() -> T) {
    while values.len() < target_len {
        values.push(fill());
    }
    values.truncate(target_len);
}

fn mutate_many_or_one<T, R: Rng + ?Sized>(
    values: &mut [T],
    per_item_rate: f32,
    rng: &mut R,
    mut mutate_one: impl FnMut(&mut T, &mut R),
) {
    if values.is_empty() {
        return;
    }

    let len = values.len();
    let mut mutated_any = false;
    for value in values.iter_mut() {
        if rng.random::<f32>() >= per_item_rate {
            continue;
        }
        mutated_any = true;
        mutate_one(value, rng);
    }

    if !mutated_any {
        let idx = rng.random_range(0..len);
        mutate_one(&mut values[idx], rng);
    }
}

fn step_u32<R: Rng + ?Sized>(value: u32, min: u32, max: u32, rng: &mut R) -> u32 {
    if min >= max {
        return min;
    }
    if value <= min {
        return min.saturating_add(1).min(max);
    }
    if value >= max {
        return max.saturating_sub(1).max(min);
    }
    if rng.random::<bool>() {
        value.saturating_add(1).min(max)
    } else {
        value.saturating_sub(1).max(min)
    }
}

fn step_u8<R: Rng + ?Sized>(value: u8, min: u8, max: u8, rng: &mut R) -> u8 {
    if min >= max {
        return min;
    }
    if value <= min {
        return min.saturating_add(1).min(max);
    }
    if value >= max {
        return max.saturating_sub(1).max(min);
    }
    if rng.random::<bool>() {
        value.saturating_add(1).min(max)
    } else {
        value.saturating_sub(1).max(min)
    }
}

fn sample_initial_log_time_constant<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    perturb_clamped(
        DEFAULT_INTER_LOG_TIME_CONSTANT,
        0.5,
        INTER_LOG_TIME_CONSTANT_MIN,
        INTER_LOG_TIME_CONSTANT_MAX,
        rng,
    )
}

fn sample_uniform_location<R: Rng + ?Sized>(rng: &mut R) -> BrainLocation {
    BrainLocation {
        x: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
        y: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
    }
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

fn max_possible_synapses(num_neurons: u32) -> u32 {
    let pre_count = u64::from(SENSORY_COUNT + num_neurons);
    let post_count = u64::from(num_neurons + ACTION_COUNT_U32);
    let all_pairs = pre_count.saturating_mul(post_count);
    let max = all_pairs.saturating_sub(u64::from(num_neurons));
    max.min(u64::from(u32::MAX)) as u32
}
