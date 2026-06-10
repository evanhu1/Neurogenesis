use crate::topology::{
    action_array_index, action_neuron_id, constrain_weight, inter_index, inter_neuron_id,
    is_action_id, is_inter_id, is_sensory_id, ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE,
    INTER_ID_BASE, SENSORY_COUNT,
};

/// Inter-neuron IDs occupy `INTER_ID_BASE..ACTION_ID_BASE`; growing past this
/// bound would collide inter IDs with the action ID space.
const MAX_INTER_NEURONS: u32 = ACTION_ID_BASE - INTER_ID_BASE;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use sim_types::{
    BrainLocation, BrainTopology, LifecycleGenes, MutationRateGenes, NeuronId, OrganismGenome,
    PlasticityGenes, SeedGenomeConfig, SynapseGene, TopologyGenes,
};
use std::cmp::Ordering;
use std::f32::consts::LN_10;

mod mutation_rates;
mod sanitization;
mod scalar;
mod seed;
mod spatial_prior;
mod topology;

use mutation_rates::{
    effective_mutation_rate, effective_mutation_rates, mutate_mutation_rate_genes,
};
pub(crate) use sanitization::align_genome_vectors;
use sanitization::{debug_assert_genome_well_formed, reconcile_synapse_count};
pub(crate) use scalar::inter_alpha_from_log_time_constant;
use scalar::{
    mutate_action_biases, mutate_inter_biases, mutate_inter_update_rates,
    mutate_random_neuron_location, mutate_reward_weights, mutate_synapse_weights,
};
pub(crate) use seed::generate_seed_genome;
pub(crate) use topology::{
    mutate_add_neuron_split_edge, mutate_add_synapse, mutate_remove_neuron, mutate_remove_synapse,
};

pub(crate) const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
pub(crate) const MAX_MUTATED_VISION_DISTANCE: u32 = 10;
const MIN_MUTATED_AGE_OF_MATURITY: u32 = 0;
const MAX_MUTATED_AGE_OF_MATURITY: u32 = 10_000;
const MIN_MUTATED_GESTATION_TICKS: u8 = 0;
const MAX_MUTATED_GESTATION_TICKS: u8 = 10;
const MIN_MUTATED_MAX_ORGANISM_AGE: u32 = 1;
const MAX_MUTATED_MAX_ORGANISM_AGE: u32 = 100_000;
/// Clamp range for the evolvable `spatial_prior_sigma` gene. The lower bound
/// matches the `.max(0.05)`-style floor sanitization expects (>= 0.01); the
/// upper bound is the full brain-space width (`BRAIN_SPACE_MAX -
/// BRAIN_SPACE_MIN` = 10), beyond which the spatial prior is effectively
/// flat. The seed default (~3.0–3.5) sits comfortably inside.
const MIN_MUTATED_SPATIAL_PRIOR_SIGMA: f32 = 0.05;
const MAX_MUTATED_SPATIAL_PRIOR_SIGMA: f32 = 10.0;
/// Clamp range for the evolvable `max_weight_delta_per_tick` gene, a sane
/// positive band around its 0.05 default (one order of magnitude either way,
/// capped at 0.5 so a single tick cannot swing a weight by a third of
/// `SYNAPSE_STRENGTH_MAX`).
const MIN_MUTATED_MAX_WEIGHT_DELTA_PER_TICK: f32 = 0.005;
const MAX_MUTATED_MAX_WEIGHT_DELTA_PER_TICK: f32 = 0.5;
/// Log-space stddev for multiplicative mutation of strictly-positive
/// traits spanning wide ranges (e.g. `max_organism_age`, `age_of_maturity`,
/// `spatial_prior_sigma`, `max_weight_delta_per_tick`).
/// `σ = 0.1` corresponds to roughly ±10% per mutation, scale-invariant
/// across orders of magnitude.
const LARGE_UNBOUNDED_LOG_STDDEV: f32 = 0.1;
pub(crate) const SYNAPSE_STRENGTH_MAX: f32 = 1.5;
pub(crate) const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;
const ELIGIBILITY_RETENTION_MIN: f32 = 0.0;
const ELIGIBILITY_RETENTION_MAX: f32 = 1.0;
const HEBB_ETA_GAIN_MIN: f32 = 0.0;
const HEBB_ETA_GAIN_MAX: f32 = 0.2;
const JUVENILE_ETA_SCALE_MIN: f32 = 0.0;
const JUVENILE_ETA_SCALE_MAX: f32 = 4.0;
const SYNAPSE_PRUNE_THRESHOLD_MIN: f32 = 0.0;
const SYNAPSE_PRUNE_THRESHOLD_MAX: f32 = 1.0;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
const INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV: f32 = 0.05;
const HEBB_ETA_GAIN_PERTURBATION_STDDEV: f32 = 0.005;
const JUVENILE_ETA_SCALE_PERTURBATION_STDDEV: f32 = 0.25;
const ELIGIBILITY_RETENTION_PERTURBATION_STDDEV: f32 = 0.05;
const SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV: f32 = 0.02;
const INTER_BIAS_PERTURB_NEURON_RATE: f32 = 0.8;
const INTER_UPDATE_RATE_PERTURB_NEURON_RATE: f32 = 0.8;
/// Per-coefficient rate for the reward-weight mutator. Lower than the bias
/// rate because reward weights scale dopamine directly; the outer
/// `mutation_rate_inter_bias` gate restricts how often the operator fires.
const REWARD_WEIGHT_PERTURB_RATE: f32 = 0.5;
const REWARD_WEIGHT_PERTURBATION_STDDEV: f32 = 0.15;
const SYNAPSE_WEIGHT_PERTURBATION_STDDEV: f32 = 0.15;
const SYNAPSE_WEIGHT_PERTURB_EDGE_RATE: f32 = 0.8;
const SYNAPSE_WEIGHT_REPLACEMENT_RATE: f32 = 0.1;
const LOCATION_PERTURBATION_STDDEV: f32 = 0.75;
const BODY_COLOR_PERTURBATION_STDDEV: f32 = 0.12;
/// Baseline probability of mutating `body_color` per offspring, matching the
/// typical per-gene rates in the seed genome. Body color is a sensed phenotype,
/// so its drift is gated like every other gene and scales with
/// `global_mutation_rate_modifier`. There is no heritable
/// `MutationRateGenes` field for it (yet), so this baseline is fixed.
const BODY_COLOR_MUTATION_RATE: f32 = 0.1;
pub(crate) const INTER_TIME_CONSTANT_MIN: f32 = 0.1;
pub(crate) const INTER_TIME_CONSTANT_MAX: f32 = 10.0;
pub(crate) const INTER_LOG_TIME_CONSTANT_MIN: f32 = -LN_10;
pub(crate) const INTER_LOG_TIME_CONSTANT_MAX: f32 = LN_10;
pub(crate) const DEFAULT_INTER_LOG_TIME_CONSTANT: f32 = -1.203_972_8;
pub(crate) const BRAIN_SPACE_MIN: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MAX: f32 = 10.0;
/// Center of the [BRAIN_SPACE_MIN, BRAIN_SPACE_MAX] brain coordinate space.
const DEFAULT_BRAIN_LOCATION: BrainLocation = BrainLocation { x: 5.0, y: 5.0 };
/// Perturbation stddev is halved when deriving a new neuron from an edge split,
/// so the child neuron stays near the parent edge midpoint.
const NEW_NEURON_PERTURBATION_SCALE: f32 = 0.5;
const SPATIAL_PRIOR_LONG_RANGE_FLOOR: f32 = 0.01;
const SYNAPSE_WEIGHT_LOG_NORMAL_MU: f32 = -0.5;
const SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA: f32 = 0.8;
const INITIAL_SYNAPSE_EXCITATORY_PROBABILITY: f32 = 0.8;

pub(crate) fn distance_sq_between_locations(a: BrainLocation, b: BrainLocation) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

pub(crate) fn mutate_genome<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    seed_genome_config: &SeedGenomeConfig,
    global_mutation_rate_modifier: f32,
    meta_mutation_enabled: bool,
    rng: &mut R,
) {
    // Callers always pass well-formed genomes (seed generation, a previous
    // mutate_genome pass, and champion-pool intake all end in a full
    // sanitize), and `align_genome_vectors` draws RNG only for malformed
    // genomes, so demoting the entry pass to debug asserts is bit-for-bit
    // identical. The authoritative sanitize runs once at exit via
    // `reconcile_synapse_count`.
    debug_assert_genome_well_formed(genome);
    let inherited_rates = effective_mutation_rates(genome, global_mutation_rate_modifier);

    if rng.random::<f32>() < inherited_rates.age_of_maturity {
        genome.lifecycle.age_of_maturity = perturb_multiplicative_u32(
            genome.lifecycle.age_of_maturity,
            LARGE_UNBOUNDED_LOG_STDDEV,
            MIN_MUTATED_AGE_OF_MATURITY,
            MAX_MUTATED_AGE_OF_MATURITY,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.gestation_ticks {
        genome.lifecycle.gestation_ticks = step_u32(
            u32::from(genome.lifecycle.gestation_ticks),
            u32::from(MIN_MUTATED_GESTATION_TICKS),
            u32::from(MAX_MUTATED_GESTATION_TICKS),
            rng,
        ) as u8;
    }
    if rng.random::<f32>() < inherited_rates.max_organism_age {
        genome.lifecycle.max_organism_age = perturb_multiplicative_u32(
            genome.lifecycle.max_organism_age,
            LARGE_UNBOUNDED_LOG_STDDEV,
            MIN_MUTATED_MAX_ORGANISM_AGE,
            MAX_MUTATED_MAX_ORGANISM_AGE,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.vision_distance {
        genome.topology.vision_distance = step_u32(
            genome.topology.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.hebb_eta_gain {
        genome.plasticity.hebb_eta_gain = perturb_clamped(
            genome.plasticity.hebb_eta_gain,
            HEBB_ETA_GAIN_PERTURBATION_STDDEV,
            HEBB_ETA_GAIN_MIN,
            HEBB_ETA_GAIN_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.juvenile_eta_scale {
        genome.plasticity.juvenile_eta_scale = perturb_clamped(
            genome.plasticity.juvenile_eta_scale,
            JUVENILE_ETA_SCALE_PERTURBATION_STDDEV,
            JUVENILE_ETA_SCALE_MIN,
            JUVENILE_ETA_SCALE_MAX,
            rng,
        );
    }
    if rng.random::<f32>()
        < effective_mutation_rate(BODY_COLOR_MUTATION_RATE, global_mutation_rate_modifier)
    {
        genome.lifecycle.body_color = mutate_body_color(genome.lifecycle.body_color, rng);
    }
    if rng.random::<f32>() < inherited_rates.inter_bias {
        mutate_inter_biases(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.inter_bias {
        mutate_action_biases(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.inter_bias {
        mutate_reward_weights(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.inter_update_rate {
        mutate_inter_update_rates(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.eligibility_retention {
        genome.plasticity.eligibility_retention = perturb_clamped(
            genome.plasticity.eligibility_retention,
            ELIGIBILITY_RETENTION_PERTURBATION_STDDEV,
            ELIGIBILITY_RETENTION_MIN,
            ELIGIBILITY_RETENTION_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.synapse_prune_threshold {
        genome.plasticity.synapse_prune_threshold = perturb_clamped(
            genome.plasticity.synapse_prune_threshold,
            SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV,
            SYNAPSE_PRUNE_THRESHOLD_MIN,
            SYNAPSE_PRUNE_THRESHOLD_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.neuron_location {
        mutate_random_neuron_location(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.synapse_weight_perturbation {
        mutate_synapse_weights(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.add_synapse {
        mutate_add_synapse(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.remove_synapse {
        mutate_remove_synapse(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.remove_neuron {
        mutate_remove_neuron(genome, rng);
    }
    if rng.random::<f32>() < inherited_rates.add_neuron_split_edge {
        mutate_add_neuron_split_edge(genome, rng);
    }
    // The two gates below were appended AFTER all pre-existing gates so the
    // RNG draw prefix consumed by the older operators is unchanged; only the
    // draws from this point on shift (a sanctioned change to evolution
    // outcomes). Keep any future gates appended here, before
    // `reconcile_synapse_count`.
    if rng.random::<f32>() < inherited_rates.spatial_prior_sigma {
        genome.topology.spatial_prior_sigma = perturb_multiplicative_f32(
            genome.topology.spatial_prior_sigma,
            LARGE_UNBOUNDED_LOG_STDDEV,
            MIN_MUTATED_SPATIAL_PRIOR_SIGMA,
            MAX_MUTATED_SPATIAL_PRIOR_SIGMA,
            rng,
        );
    }
    if rng.random::<f32>() < inherited_rates.max_weight_delta_per_tick {
        genome.plasticity.max_weight_delta_per_tick = perturb_multiplicative_f32(
            genome.plasticity.max_weight_delta_per_tick,
            LARGE_UNBOUNDED_LOG_STDDEV,
            MIN_MUTATED_MAX_WEIGHT_DELTA_PER_TICK,
            MAX_MUTATED_MAX_WEIGHT_DELTA_PER_TICK,
            rng,
        );
    }

    reconcile_synapse_count(genome, rng);

    if meta_mutation_enabled {
        // Mutation-rate genes are inherited strategy parameters. Update them after
        // offspring mutation so they affect the next generation rather than
        // immediately rewarding anti-mutator lineages in the current child.
        mutate_mutation_rate_genes(genome, seed_genome_config, rng);
    }
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

fn perturb_multiplicative_u32<R: Rng + ?Sized>(
    value: u32,
    log_stddev: f32,
    min: u32,
    max: u32,
    rng: &mut R,
) -> u32 {
    if min >= max {
        return min;
    }
    let scale = (log_stddev * standard_normal(rng)).exp();
    let scaled = ((value as f32) * scale)
        .round()
        .clamp(min as f32, max as f32) as u32;
    if scaled != value {
        return scaled;
    }
    // Rounding collapsed the mutation (common for small values or when
    // `value == 0`). Fall back to a ±1 step so the operator never silently
    // no-ops; `step_u32` steps inward at the range boundaries.
    step_u32(value, min, max, rng)
}

/// Multiplicative log-normal perturbation for strictly-positive f32 traits:
/// scales `value` by `exp(log_stddev * N(0, 1))` and clamps into `[min, max]`.
/// Draws exactly one standard normal, mirroring `perturb_multiplicative_u32`
/// without the integer rounding fallback (f32 scaling never collapses to a
/// no-op for nonzero values).
fn perturb_multiplicative_f32<R: Rng + ?Sized>(
    value: f32,
    log_stddev: f32,
    min: f32,
    max: f32,
    rng: &mut R,
) -> f32 {
    let scale = (log_stddev * standard_normal(rng)).exp();
    (value * scale).clamp(min, max)
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

fn sample_body_color<R: Rng + ?Sized>(rng: &mut R) -> sim_types::RgbColor {
    sim_types::RgbColor {
        r: rng.random::<f32>(),
        g: rng.random::<f32>(),
        b: rng.random::<f32>(),
    }
    .clamped()
}

fn mutate_body_color<R: Rng + ?Sized>(
    color: sim_types::RgbColor,
    rng: &mut R,
) -> sim_types::RgbColor {
    sim_types::RgbColor {
        r: color.r + BODY_COLOR_PERTURBATION_STDDEV * standard_normal(rng),
        g: color.g + BODY_COLOR_PERTURBATION_STDDEV * standard_normal(rng),
        b: color.b + BODY_COLOR_PERTURBATION_STDDEV * standard_normal(rng),
    }
    .clamped()
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
    // Inter-neuron self-edges are valid (see `is_valid_synapse_pair`), so the
    // full (pre, post) cross product is reachable.
    let pre_count = u64::from(SENSORY_COUNT + num_neurons);
    let post_count = u64::from(num_neurons + ACTION_COUNT_U32);
    let all_pairs = pre_count.saturating_mul(post_count);
    all_pairs.min(u64::from(u32::MAX)) as u32
}
