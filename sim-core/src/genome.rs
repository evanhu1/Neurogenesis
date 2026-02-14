use crate::brain::{ACTION_COUNT, ACTION_COUNT_U32, SENSORY_COUNT};
use crate::SimError;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use sim_types::{ActionType, BrainLocation, InterNeuronType, OrganismGenome, SeedGenomeConfig};

const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const MIN_MUTATED_AGE_OF_MATURITY: u32 = 0;
const MAX_MUTATED_AGE_OF_MATURITY: u32 = 10_000;
pub(crate) const SYNAPSE_STRENGTH_MAX: f32 = 1.0;
pub(crate) const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;
const ETA_BASELINE_MIN: f32 = 0.0;
const ETA_BASELINE_MAX: f32 = 0.2;
const ETA_GAIN_MIN: f32 = -1.0;
const ETA_GAIN_MAX: f32 = 1.0;
const ELIGIBILITY_DECAY_LAMBDA_MIN: f32 = 0.0;
const ELIGIBILITY_DECAY_LAMBDA_MAX: f32 = 1.0;
const SYNAPSE_PRUNE_THRESHOLD_MIN: f32 = 0.0;
const SYNAPSE_PRUNE_THRESHOLD_MAX: f32 = 1.0;

const INTER_TYPE_EXCITATORY_PRIOR: f32 = 0.8;
const MUTATION_RATE_ADAPTATION_TAU: f32 = 0.25;
const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 1.0 - MUTATION_RATE_MIN;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
const INTER_LOG_TAU_PERTURBATION_STDDEV: f32 = 0.05;
const ELIGIBILITY_DECAY_LAMBDA_PERTURBATION_STDDEV: f32 = 0.05;
const SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV: f32 = 0.02;
const LOCATION_PERTURBATION_STDDEV: f32 = 0.75;
pub(crate) const INTER_TAU_MIN: f32 = 0.1;
pub(crate) const INTER_TAU_MAX: f32 = 15.0;
pub(crate) const INTER_LOG_TAU_MIN: f32 = -2.302_585_1;
pub(crate) const INTER_LOG_TAU_MAX: f32 = 2.995_732_3;
pub(crate) const DEFAULT_INTER_LOG_TAU: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MIN: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MAX: f32 = 10.0;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    world_max_num_neurons: u32,
    rng: &mut R,
) -> OrganismGenome {
    let clamped_num_neurons = config.num_neurons.min(world_max_num_neurons);
    let max_synapses = max_possible_synapses(clamped_num_neurons);
    let inter_biases: Vec<f32> = (0..world_max_num_neurons)
        .map(|_| sample_initial_bias(rng))
        .collect();
    let inter_log_taus: Vec<f32> = (0..world_max_num_neurons)
        .map(|_| sample_uniform_log_tau(rng))
        .collect();
    let interneuron_types: Vec<InterNeuronType> = (0..world_max_num_neurons)
        .map(|_| sample_interneuron_type(rng))
        .collect();
    let inter_locations: Vec<BrainLocation> = (0..world_max_num_neurons)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_biases: Vec<f32> = ActionType::ALL
        .into_iter()
        .map(|action_type| {
            if matches!(action_type, ActionType::Dopamine) {
                0.0
            } else {
                sample_initial_bias(rng)
            }
        })
        .collect();
    let sensory_locations: Vec<BrainLocation> = (0..SENSORY_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_locations: Vec<BrainLocation> = (0..ACTION_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();

    OrganismGenome {
        num_neurons: clamped_num_neurons,
        num_synapses: config.num_synapses.min(max_synapses),
        spatial_prior_sigma: config.spatial_prior_sigma.max(0.01),
        vision_distance: config.vision_distance,
        age_of_maturity: config.age_of_maturity,
        hebb_eta_baseline: config.hebb_eta_baseline,
        hebb_eta_gain: config.hebb_eta_gain,
        eligibility_decay_lambda: config.eligibility_decay_lambda,
        synapse_prune_threshold: config.synapse_prune_threshold,
        mutation_rate_age_of_maturity: config.mutation_rate_age_of_maturity,
        mutation_rate_vision_distance: config.mutation_rate_vision_distance,
        mutation_rate_num_synapses: config.mutation_rate_num_synapses,
        mutation_rate_inter_bias: config.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: config.mutation_rate_inter_update_rate,
        mutation_rate_action_bias: config.mutation_rate_action_bias,
        mutation_rate_eligibility_decay_lambda: config.mutation_rate_eligibility_decay_lambda,
        mutation_rate_synapse_prune_threshold: config.mutation_rate_synapse_prune_threshold,
        mutation_rate_neuron_location: config.mutation_rate_neuron_location,
        inter_biases,
        inter_log_taus,
        interneuron_types,
        action_biases,
        sensory_locations,
        inter_locations,
        action_locations,
    }
}

fn sample_interneuron_type<R: Rng + ?Sized>(rng: &mut R) -> InterNeuronType {
    if rng.random::<f32>() < INTER_TYPE_EXCITATORY_PRIOR {
        InterNeuronType::Excitatory
    } else {
        InterNeuronType::Inhibitory
    }
}

fn mutate_mutation_rate_genes<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let mut rates = [
        genome.mutation_rate_age_of_maturity,
        genome.mutation_rate_vision_distance,
        genome.mutation_rate_num_synapses,
        genome.mutation_rate_inter_bias,
        genome.mutation_rate_inter_update_rate,
        genome.mutation_rate_action_bias,
        genome.mutation_rate_eligibility_decay_lambda,
        genome.mutation_rate_synapse_prune_threshold,
        genome.mutation_rate_neuron_location,
    ];
    let shared_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TAU;

    for rate in &mut rates {
        let gene_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TAU;
        let adapted = *rate * (shared_normal + gene_normal).exp();
        *rate = adapted.clamp(MUTATION_RATE_MIN, MUTATION_RATE_MAX);
    }

    genome.mutation_rate_age_of_maturity = rates[0];
    genome.mutation_rate_vision_distance = rates[1];
    genome.mutation_rate_num_synapses = rates[2];
    genome.mutation_rate_inter_bias = rates[3];
    genome.mutation_rate_inter_update_rate = rates[4];
    genome.mutation_rate_action_bias = rates[5];
    genome.mutation_rate_eligibility_decay_lambda = rates[6];
    genome.mutation_rate_synapse_prune_threshold = rates[7];
    genome.mutation_rate_neuron_location = rates[8];
}

fn align_genome_vectors<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    world_max_num_neurons: u32,
    rng: &mut R,
) {
    genome.num_neurons = genome.num_neurons.min(world_max_num_neurons);
    genome.num_synapses = genome
        .num_synapses
        .min(max_possible_synapses(genome.num_neurons));
    genome.spatial_prior_sigma = genome.spatial_prior_sigma.max(0.01);

    let target_inter_len = world_max_num_neurons as usize;

    while genome.inter_biases.len() < target_inter_len {
        genome.inter_biases.push(sample_initial_bias(rng));
    }
    genome.inter_biases.truncate(target_inter_len);

    while genome.inter_log_taus.len() < target_inter_len {
        genome.inter_log_taus.push(sample_uniform_log_tau(rng));
    }
    genome.inter_log_taus.truncate(target_inter_len);

    while genome.interneuron_types.len() < target_inter_len {
        genome.interneuron_types.push(sample_interneuron_type(rng));
    }
    genome.interneuron_types.truncate(target_inter_len);

    while genome.inter_locations.len() < target_inter_len {
        genome.inter_locations.push(sample_uniform_location(rng));
    }
    genome.inter_locations.truncate(target_inter_len);

    if genome.action_biases.len() < ACTION_COUNT {
        genome.action_biases.resize(ACTION_COUNT, 0.0);
    } else if genome.action_biases.len() > ACTION_COUNT {
        genome.action_biases.truncate(ACTION_COUNT);
    }

    while genome.sensory_locations.len() < SENSORY_COUNT as usize {
        genome.sensory_locations.push(sample_uniform_location(rng));
    }
    genome.sensory_locations.truncate(SENSORY_COUNT as usize);

    while genome.action_locations.len() < ACTION_COUNT {
        genome.action_locations.push(sample_uniform_location(rng));
    }
    genome.action_locations.truncate(ACTION_COUNT);
}

pub(crate) fn mutate_genome<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    world_max_num_neurons: u32,
    rng: &mut R,
) {
    align_genome_vectors(genome, world_max_num_neurons, rng);
    mutate_mutation_rate_genes(genome, rng);

    if rng.random::<f32>() < genome.mutation_rate_age_of_maturity {
        genome.age_of_maturity = step_u32(
            genome.age_of_maturity,
            MIN_MUTATED_AGE_OF_MATURITY,
            MAX_MUTATED_AGE_OF_MATURITY,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_vision_distance {
        genome.vision_distance = step_u32(
            genome.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_num_synapses {
        let max_synapses = max_possible_synapses(genome.num_neurons);
        genome.num_synapses = step_u32(genome.num_synapses, 0, max_synapses, rng);
    }

    if rng.random::<f32>() < genome.mutation_rate_inter_bias && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_biases[idx] = perturb_clamped(
            genome.inter_biases[idx],
            BIAS_PERTURBATION_STDDEV,
            -BIAS_MAX,
            BIAS_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_inter_update_rate && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_log_taus[idx] = perturb_clamped(
            genome.inter_log_taus[idx],
            INTER_LOG_TAU_PERTURBATION_STDDEV,
            INTER_LOG_TAU_MIN,
            INTER_LOG_TAU_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_action_bias && !genome.action_biases.is_empty() {
        let idx = rng.random_range(0..genome.action_biases.len());
        genome.action_biases[idx] = perturb_clamped(
            genome.action_biases[idx],
            BIAS_PERTURBATION_STDDEV,
            -BIAS_MAX,
            BIAS_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_eligibility_decay_lambda {
        genome.eligibility_decay_lambda = perturb_clamped(
            genome.eligibility_decay_lambda,
            ELIGIBILITY_DECAY_LAMBDA_PERTURBATION_STDDEV,
            ELIGIBILITY_DECAY_LAMBDA_MIN,
            ELIGIBILITY_DECAY_LAMBDA_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_synapse_prune_threshold {
        genome.synapse_prune_threshold = perturb_clamped(
            genome.synapse_prune_threshold,
            SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV,
            SYNAPSE_PRUNE_THRESHOLD_MIN,
            SYNAPSE_PRUNE_THRESHOLD_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_neuron_location {
        mutate_random_neuron_location(genome, rng);
    }
}

fn mutate_random_neuron_location<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let enabled_inter = genome.num_neurons as usize;
    let total = SENSORY_COUNT as usize + enabled_inter + ACTION_COUNT;
    if total == 0 {
        return;
    }

    let idx = rng.random_range(0..total);
    let location = if idx < SENSORY_COUNT as usize {
        genome.sensory_locations.get_mut(idx)
    } else if idx < SENSORY_COUNT as usize + enabled_inter {
        genome
            .inter_locations
            .get_mut(idx.saturating_sub(SENSORY_COUNT as usize))
    } else {
        genome
            .action_locations
            .get_mut(idx.saturating_sub(SENSORY_COUNT as usize + enabled_inter))
    };

    if let Some(location) = location {
        location.x = perturb_clamped(
            location.x,
            LOCATION_PERTURBATION_STDDEV,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        );
        location.y = perturb_clamped(
            location.y,
            LOCATION_PERTURBATION_STDDEV,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        );
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

fn sample_uniform_log_tau<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    rng.random_range(INTER_LOG_TAU_MIN..=INTER_LOG_TAU_MAX)
}

fn sample_uniform_location<R: Rng + ?Sized>(rng: &mut R) -> BrainLocation {
    BrainLocation {
        x: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
        y: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
    }
}

pub(crate) fn inter_alpha_from_log_tau(log_tau: f32) -> f32 {
    let clamped_log_tau = log_tau.clamp(INTER_LOG_TAU_MIN, INTER_LOG_TAU_MAX);
    let tau = clamped_log_tau.exp().clamp(INTER_TAU_MIN, INTER_TAU_MAX);
    1.0 - (-1.0 / tau).exp()
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

pub(crate) fn prune_disconnected_inter_neurons(_genome: &mut OrganismGenome) {}

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

fn average_location_distance(a: &[BrainLocation], b: &[BrainLocation], len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for i in 0..len {
        let la = a
            .get(i)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
        let lb = b
            .get(i)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
        total += (la.x - lb.x).abs() + (la.y - lb.y).abs();
    }

    total / len as f32
}

fn centroid(locations: &[BrainLocation], len: usize) -> BrainLocation {
    if len == 0 {
        return BrainLocation { x: 5.0, y: 5.0 };
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for i in 0..len {
        let location = locations
            .get(i)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
        sum_x += location.x;
        sum_y += location.y;
    }

    BrainLocation {
        x: sum_x / len as f32,
        y: sum_y / len as f32,
    }
}

/// L1 genome distance: scalar traits + mutation-rate genes + neuron types + brain geometry.
pub(crate) fn genome_distance(a: &OrganismGenome, b: &OrganismGenome) -> f32 {
    let mut dist = (a.num_neurons as f32 - b.num_neurons as f32).abs()
        + (a.num_synapses as f32 - b.num_synapses as f32).abs()
        + (a.spatial_prior_sigma - b.spatial_prior_sigma).abs()
        + (a.vision_distance as f32 - b.vision_distance as f32).abs()
        + (a.age_of_maturity as f32 - b.age_of_maturity as f32).abs()
        + (a.hebb_eta_baseline - b.hebb_eta_baseline).abs()
        + (a.hebb_eta_gain - b.hebb_eta_gain).abs()
        + (a.eligibility_decay_lambda - b.eligibility_decay_lambda).abs()
        + (a.synapse_prune_threshold - b.synapse_prune_threshold).abs();

    let a_rates = [
        a.mutation_rate_age_of_maturity,
        a.mutation_rate_vision_distance,
        a.mutation_rate_num_synapses,
        a.mutation_rate_inter_bias,
        a.mutation_rate_inter_update_rate,
        a.mutation_rate_action_bias,
        a.mutation_rate_eligibility_decay_lambda,
        a.mutation_rate_synapse_prune_threshold,
        a.mutation_rate_neuron_location,
    ];
    let b_rates = [
        b.mutation_rate_age_of_maturity,
        b.mutation_rate_vision_distance,
        b.mutation_rate_num_synapses,
        b.mutation_rate_inter_bias,
        b.mutation_rate_inter_update_rate,
        b.mutation_rate_action_bias,
        b.mutation_rate_eligibility_decay_lambda,
        b.mutation_rate_synapse_prune_threshold,
        b.mutation_rate_neuron_location,
    ];
    for i in 0..a_rates.len() {
        dist += (a_rates[i] - b_rates[i]).abs();
    }

    let max_enabled = a.num_neurons.max(b.num_neurons) as usize;
    for i in 0..max_enabled {
        let ta = a
            .interneuron_types
            .get(i)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        let tb = b
            .interneuron_types
            .get(i)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        if ta != tb {
            dist += 1.0;
        }
    }

    dist += average_location_distance(
        &a.sensory_locations,
        &b.sensory_locations,
        SENSORY_COUNT as usize,
    );
    dist += average_location_distance(&a.action_locations, &b.action_locations, ACTION_COUNT);
    dist += average_location_distance(&a.inter_locations, &b.inter_locations, max_enabled.max(1));

    let a_inter_centroid = centroid(&a.inter_locations, a.num_neurons as usize);
    let b_inter_centroid = centroid(&b.inter_locations, b.num_neurons as usize);
    dist += (a_inter_centroid.x - b_inter_centroid.x).abs()
        + (a_inter_centroid.y - b_inter_centroid.y).abs();

    dist
}

fn validate_rate(name: &str, rate: f32) -> Result<(), SimError> {
    if (0.0..=1.0).contains(&rate) {
        Ok(())
    } else {
        Err(SimError::InvalidConfig(format!(
            "{name} must be within [0, 1]"
        )))
    }
}

fn max_possible_synapses(num_neurons: u32) -> u32 {
    let pre_count = u64::from(SENSORY_COUNT + num_neurons);
    let post_count = u64::from(num_neurons + ACTION_COUNT_U32);
    let max = pre_count.saturating_mul(post_count);
    max.min(u64::from(u32::MAX)) as u32
}

pub(crate) fn validate_seed_genome_config(config: &SeedGenomeConfig) -> Result<(), SimError> {
    if !(ETA_BASELINE_MIN..=ETA_BASELINE_MAX).contains(&config.hebb_eta_baseline) {
        return Err(SimError::InvalidConfig(format!(
            "hebb_eta_baseline must be within [{ETA_BASELINE_MIN}, {ETA_BASELINE_MAX}]"
        )));
    }
    if !(ETA_GAIN_MIN..=ETA_GAIN_MAX).contains(&config.hebb_eta_gain) {
        return Err(SimError::InvalidConfig(format!(
            "hebb_eta_gain must be within [{ETA_GAIN_MIN}, {ETA_GAIN_MAX}]"
        )));
    }
    if !(ELIGIBILITY_DECAY_LAMBDA_MIN..=ELIGIBILITY_DECAY_LAMBDA_MAX)
        .contains(&config.eligibility_decay_lambda)
    {
        return Err(SimError::InvalidConfig(format!(
            "eligibility_decay_lambda must be within [{ELIGIBILITY_DECAY_LAMBDA_MIN}, {ELIGIBILITY_DECAY_LAMBDA_MAX}]"
        )));
    }
    if !(SYNAPSE_PRUNE_THRESHOLD_MIN..=SYNAPSE_PRUNE_THRESHOLD_MAX)
        .contains(&config.synapse_prune_threshold)
    {
        return Err(SimError::InvalidConfig(format!(
            "synapse_prune_threshold must be within [{SYNAPSE_PRUNE_THRESHOLD_MIN}, {SYNAPSE_PRUNE_THRESHOLD_MAX}]"
        )));
    }

    if config.age_of_maturity > MAX_MUTATED_AGE_OF_MATURITY {
        return Err(SimError::InvalidConfig(format!(
            "age_of_maturity must be <= {MAX_MUTATED_AGE_OF_MATURITY}"
        )));
    }

    validate_rate(
        "mutation_rate_age_of_maturity",
        config.mutation_rate_age_of_maturity,
    )?;
    validate_rate(
        "mutation_rate_vision_distance",
        config.mutation_rate_vision_distance,
    )?;
    validate_rate(
        "mutation_rate_num_synapses",
        config.mutation_rate_num_synapses,
    )?;
    validate_rate("mutation_rate_inter_bias", config.mutation_rate_inter_bias)?;
    validate_rate(
        "mutation_rate_inter_update_rate",
        config.mutation_rate_inter_update_rate,
    )?;
    validate_rate(
        "mutation_rate_action_bias",
        config.mutation_rate_action_bias,
    )?;
    validate_rate(
        "mutation_rate_eligibility_decay_lambda",
        config.mutation_rate_eligibility_decay_lambda,
    )?;
    validate_rate(
        "mutation_rate_synapse_prune_threshold",
        config.mutation_rate_synapse_prune_threshold,
    )?;
    validate_rate(
        "mutation_rate_neuron_location",
        config.mutation_rate_neuron_location,
    )?;

    if config.vision_distance < MIN_MUTATED_VISION_DISTANCE {
        return Err(SimError::InvalidConfig(
            "vision_distance must be >= 1".to_owned(),
        ));
    }
    if config.vision_distance > MAX_MUTATED_VISION_DISTANCE {
        return Err(SimError::InvalidConfig(format!(
            "vision_distance must be <= {}",
            MAX_MUTATED_VISION_DISTANCE
        )));
    }

    if !config.spatial_prior_sigma.is_finite() || config.spatial_prior_sigma <= 0.0 {
        return Err(SimError::InvalidConfig(
            "spatial_prior_sigma must be > 0".to_owned(),
        ));
    }

    let max_synapses = max_possible_synapses(config.num_neurons);
    if config.num_synapses > max_synapses {
        return Err(SimError::InvalidConfig(format!(
            "num_synapses must be <= {max_synapses} for num_neurons={}",
            config.num_neurons
        )));
    }

    Ok(())
}
