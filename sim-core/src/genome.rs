use crate::brain::{ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT};
use crate::SimError;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use sim_types::{
    ActionType, BrainLocation, InterNeuronType, NeuronId, OrganismGenome, SeedGenomeConfig,
    SynapseEdge,
};
use std::cmp::Ordering;
use std::collections::HashSet;

const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const MIN_MUTATED_AGE_OF_MATURITY: u32 = 0;
const MAX_MUTATED_AGE_OF_MATURITY: u32 = 10_000;
pub(crate) const SYNAPSE_STRENGTH_MAX: f32 = 1.0;
pub(crate) const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;
const ETA_GAIN_MIN: f32 = -1.0;
const ETA_GAIN_MAX: f32 = 1.0;
const ELIGIBILITY_RETENTION_MIN: f32 = 0.0;
const ELIGIBILITY_RETENTION_MAX: f32 = 1.0;
const SYNAPSE_PRUNE_THRESHOLD_MIN: f32 = 0.0;
const SYNAPSE_PRUNE_THRESHOLD_MAX: f32 = 1.0;

const INTER_TYPE_EXCITATORY_PRIOR: f32 = 0.8;
const MUTATION_RATE_ADAPTATION_TIME_CONSTANT: f32 = 0.25;
const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 0.5;
const MUTATION_RATE_LATENT_MIN: f32 = -8.0;
const MUTATION_RATE_LATENT_MAX: f32 = 8.0;
const MUTATION_RATE_LOGIT_EPSILON: f32 = 1.0e-6;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
const INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV: f32 = 0.05;
const ELIGIBILITY_RETENTION_PERTURBATION_STDDEV: f32 = 0.05;
const SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV: f32 = 0.02;
const SYNAPSE_WEIGHT_PERTURBATION_STDDEV: f32 = 0.15;
const LOCATION_PERTURBATION_STDDEV: f32 = 0.75;
pub(crate) const INTER_TIME_CONSTANT_MIN: f32 = 0.1;
pub(crate) const INTER_TIME_CONSTANT_MAX: f32 = 15.0;
pub(crate) const INTER_LOG_TIME_CONSTANT_MIN: f32 = -2.302_585_1;
pub(crate) const INTER_LOG_TIME_CONSTANT_MAX: f32 = 2.995_732_3;
pub(crate) const DEFAULT_INTER_LOG_TIME_CONSTANT: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MIN: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MAX: f32 = 10.0;
const SPATIAL_PRIOR_LONG_RANGE_FLOOR: f32 = 0.01;
const SYNAPSE_WEIGHT_LOG_NORMAL_MU: f32 = -0.5;
const SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA: f32 = 0.8;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let num_neurons = config.num_neurons;
    let max_synapses = max_possible_synapses(num_neurons);
    let inter_biases: Vec<f32> = (0..num_neurons).map(|_| sample_initial_bias(rng)).collect();
    let inter_log_time_constants: Vec<f32> = (0..num_neurons)
        .map(|_| sample_uniform_log_time_constant(rng))
        .collect();
    let interneuron_types: Vec<InterNeuronType> = (0..num_neurons)
        .map(|_| sample_interneuron_type(rng))
        .collect();
    let inter_locations: Vec<BrainLocation> = (0..num_neurons)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_biases: Vec<f32> = ActionType::ALL
        .into_iter()
        .map(|_| sample_initial_bias(rng))
        .collect();
    let sensory_locations: Vec<BrainLocation> = (0..SENSORY_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_locations: Vec<BrainLocation> = (0..ACTION_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();

    let mut genome = OrganismGenome {
        num_neurons,
        num_synapses: config.num_synapses.min(max_synapses),
        spatial_prior_sigma: config.spatial_prior_sigma.max(0.01),
        vision_distance: config.vision_distance,
        starting_energy: config.starting_energy,
        age_of_maturity: config.age_of_maturity,
        hebb_eta_gain: config.hebb_eta_gain,
        eligibility_retention: config.eligibility_retention,
        synapse_prune_threshold: config.synapse_prune_threshold,
        mutation_rate_age_of_maturity: config.mutation_rate_age_of_maturity,
        mutation_rate_vision_distance: config.mutation_rate_vision_distance,
        mutation_rate_inter_bias: config.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: config.mutation_rate_inter_update_rate,
        mutation_rate_action_bias: config.mutation_rate_action_bias,
        mutation_rate_eligibility_retention: config.mutation_rate_eligibility_retention,
        mutation_rate_synapse_prune_threshold: config.mutation_rate_synapse_prune_threshold,
        mutation_rate_neuron_location: config.mutation_rate_neuron_location,
        mutation_rate_synapse_weight_perturbation: config.mutation_rate_synapse_weight_perturbation,
        mutation_rate_add_neuron_split_edge: config.mutation_rate_add_neuron_split_edge,
        inter_biases,
        inter_log_time_constants,
        interneuron_types,
        action_biases,
        sensory_locations,
        inter_locations,
        action_locations,
        edges: Vec::new(),
    };
    sync_synapse_genes_to_target(&mut genome, rng);
    genome
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
        genome.mutation_rate_inter_bias,
        genome.mutation_rate_inter_update_rate,
        genome.mutation_rate_action_bias,
        genome.mutation_rate_eligibility_retention,
        genome.mutation_rate_synapse_prune_threshold,
        genome.mutation_rate_neuron_location,
        genome.mutation_rate_synapse_weight_perturbation,
        genome.mutation_rate_add_neuron_split_edge,
    ];
    let shared_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TIME_CONSTANT;

    for rate in &mut rates {
        let mut latent = mutation_rate_to_latent(*rate);
        let gene_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TIME_CONSTANT;
        latent = (latent + shared_normal + gene_normal)
            .clamp(MUTATION_RATE_LATENT_MIN, MUTATION_RATE_LATENT_MAX);
        *rate = mutation_rate_from_latent(latent);
    }

    genome.mutation_rate_age_of_maturity = rates[0];
    genome.mutation_rate_vision_distance = rates[1];
    genome.mutation_rate_inter_bias = rates[2];
    genome.mutation_rate_inter_update_rate = rates[3];
    genome.mutation_rate_action_bias = rates[4];
    genome.mutation_rate_eligibility_retention = rates[5];
    genome.mutation_rate_synapse_prune_threshold = rates[6];
    genome.mutation_rate_neuron_location = rates[7];
    genome.mutation_rate_synapse_weight_perturbation = rates[8];
    genome.mutation_rate_add_neuron_split_edge = rates[9];
}

fn align_genome_vectors<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    genome.num_synapses = genome
        .num_synapses
        .min(max_possible_synapses(genome.num_neurons));
    genome.spatial_prior_sigma = genome.spatial_prior_sigma.max(0.01);

    let target_inter_len = genome.num_neurons as usize;

    while genome.inter_biases.len() < target_inter_len {
        genome.inter_biases.push(sample_initial_bias(rng));
    }
    genome.inter_biases.truncate(target_inter_len);

    while genome.inter_log_time_constants.len() < target_inter_len {
        genome
            .inter_log_time_constants
            .push(sample_uniform_log_time_constant(rng));
    }
    genome.inter_log_time_constants.truncate(target_inter_len);

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

    sanitize_synapse_genes(genome);
}

fn clamp_mutation_rate(rate: f32) -> f32 {
    rate.clamp(MUTATION_RATE_MIN, MUTATION_RATE_MAX)
}

fn mutation_rate_to_latent(rate: f32) -> f32 {
    let clamped_rate = clamp_mutation_rate(rate);
    let span = (MUTATION_RATE_MAX - MUTATION_RATE_MIN).max(f32::MIN_POSITIVE);
    let normalized = ((clamped_rate - MUTATION_RATE_MIN) / span).clamp(
        MUTATION_RATE_LOGIT_EPSILON,
        1.0 - MUTATION_RATE_LOGIT_EPSILON,
    );
    (normalized / (1.0 - normalized)).ln()
}

fn mutation_rate_from_latent(latent: f32) -> f32 {
    let clamped_latent = latent.clamp(MUTATION_RATE_LATENT_MIN, MUTATION_RATE_LATENT_MAX);
    let sigmoid = 1.0 / (1.0 + (-clamped_latent).exp());
    let rate = MUTATION_RATE_MIN + sigmoid * (MUTATION_RATE_MAX - MUTATION_RATE_MIN);
    clamp_mutation_rate(rate)
}

fn effective_mutation_rate(rate: f32, global_mutation_rate_modifier: f32) -> f32 {
    (rate * global_mutation_rate_modifier).clamp(0.0, MUTATION_RATE_MAX)
}

pub(crate) fn mutate_genome<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    global_mutation_rate_modifier: f32,
    rng: &mut R,
) {
    align_genome_vectors(genome, rng);
    mutate_mutation_rate_genes(genome, rng);

    let mutation_rate_age_of_maturity = effective_mutation_rate(
        genome.mutation_rate_age_of_maturity,
        global_mutation_rate_modifier,
    );
    let mutation_rate_vision_distance = effective_mutation_rate(
        genome.mutation_rate_vision_distance,
        global_mutation_rate_modifier,
    );
    let mutation_rate_inter_bias = effective_mutation_rate(
        genome.mutation_rate_inter_bias,
        global_mutation_rate_modifier,
    );
    let mutation_rate_inter_update_rate = effective_mutation_rate(
        genome.mutation_rate_inter_update_rate,
        global_mutation_rate_modifier,
    );
    let mutation_rate_action_bias = effective_mutation_rate(
        genome.mutation_rate_action_bias,
        global_mutation_rate_modifier,
    );
    let mutation_rate_eligibility_retention = effective_mutation_rate(
        genome.mutation_rate_eligibility_retention,
        global_mutation_rate_modifier,
    );
    let mutation_rate_synapse_prune_threshold = effective_mutation_rate(
        genome.mutation_rate_synapse_prune_threshold,
        global_mutation_rate_modifier,
    );
    let mutation_rate_neuron_location = effective_mutation_rate(
        genome.mutation_rate_neuron_location,
        global_mutation_rate_modifier,
    );
    let mutation_rate_synapse_weight_perturbation = effective_mutation_rate(
        genome.mutation_rate_synapse_weight_perturbation,
        global_mutation_rate_modifier,
    );
    let mutation_rate_add_neuron_split_edge = effective_mutation_rate(
        genome.mutation_rate_add_neuron_split_edge,
        global_mutation_rate_modifier,
    );

    if rng.random::<f32>() < mutation_rate_age_of_maturity {
        genome.age_of_maturity = step_u32(
            genome.age_of_maturity,
            MIN_MUTATED_AGE_OF_MATURITY,
            MAX_MUTATED_AGE_OF_MATURITY,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_vision_distance {
        genome.vision_distance = step_u32(
            genome.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_inter_bias && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_biases[idx] = perturb_clamped(
            genome.inter_biases[idx],
            BIAS_PERTURBATION_STDDEV,
            -BIAS_MAX,
            BIAS_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_inter_update_rate && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_log_time_constants[idx] = perturb_clamped(
            genome.inter_log_time_constants[idx],
            INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV,
            INTER_LOG_TIME_CONSTANT_MIN,
            INTER_LOG_TIME_CONSTANT_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_action_bias && !genome.action_biases.is_empty() {
        let idx = rng.random_range(0..genome.action_biases.len());
        genome.action_biases[idx] = perturb_clamped(
            genome.action_biases[idx],
            BIAS_PERTURBATION_STDDEV,
            -BIAS_MAX,
            BIAS_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_eligibility_retention {
        genome.eligibility_retention = perturb_clamped(
            genome.eligibility_retention,
            ELIGIBILITY_RETENTION_PERTURBATION_STDDEV,
            ELIGIBILITY_RETENTION_MIN,
            ELIGIBILITY_RETENTION_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_synapse_prune_threshold {
        genome.synapse_prune_threshold = perturb_clamped(
            genome.synapse_prune_threshold,
            SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV,
            SYNAPSE_PRUNE_THRESHOLD_MIN,
            SYNAPSE_PRUNE_THRESHOLD_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < mutation_rate_neuron_location {
        mutate_random_neuron_location(genome, rng);
    }

    if rng.random::<f32>() < mutation_rate_synapse_weight_perturbation {
        mutate_random_synapse_weight(genome, rng);
    }

    if rng.random::<f32>() < mutation_rate_add_neuron_split_edge {
        mutate_add_neuron_split_edge(genome, rng);
    }

    sync_synapse_genes_to_target(genome, rng);
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

fn mutate_random_synapse_weight<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    if genome.edges.is_empty() {
        return;
    }

    let idx = rng.random_range(0..genome.edges.len());
    let (pre_neuron_id, weight) = {
        let edge = &genome.edges[idx];
        (edge.pre_neuron_id, edge.weight)
    };
    let magnitude_scale = (SYNAPSE_WEIGHT_PERTURBATION_STDDEV * standard_normal(rng)).exp();
    let perturbed_weight = weight * magnitude_scale;
    let required_sign =
        required_pre_sign(pre_neuron_id, genome.num_neurons, &genome.interneuron_types)
            .unwrap_or(1.0);
    genome.edges[idx].weight = constrain_weight_to_sign(perturbed_weight, required_sign);
}

pub(crate) fn mutate_add_neuron_split_edge<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    if genome.edges.is_empty() || genome.num_neurons >= u32::MAX.saturating_sub(INTER_ID_BASE) {
        return;
    }

    let target_inter_len = genome.num_neurons as usize;
    genome.inter_biases.resize(target_inter_len, 0.0);
    genome
        .inter_log_time_constants
        .resize(target_inter_len, DEFAULT_INTER_LOG_TIME_CONSTANT);
    genome
        .interneuron_types
        .resize(target_inter_len, InterNeuronType::Excitatory);
    genome
        .inter_locations
        .resize(target_inter_len, BrainLocation { x: 5.0, y: 5.0 });
    genome
        .sensory_locations
        .resize(SENSORY_COUNT as usize, BrainLocation { x: 5.0, y: 5.0 });
    genome
        .action_locations
        .resize(ACTION_COUNT, BrainLocation { x: 5.0, y: 5.0 });

    let selected_idx = select_weighted_edge_index(&genome.edges, rng);
    let selected_edge = genome.edges.swap_remove(selected_idx);

    let new_inter_idx = genome.num_neurons;
    let new_inter_id = NeuronId(INTER_ID_BASE + new_inter_idx);

    let pre_tau = inter_log_time_constant_for_neuron(selected_edge.pre_neuron_id, genome);
    let post_tau = inter_log_time_constant_for_neuron(selected_edge.post_neuron_id, genome);
    let base_tau = match (pre_tau, post_tau) {
        (Some(pre), Some(post)) => 0.5 * (pre + post),
        (Some(pre), None) => pre,
        (None, Some(post)) => post,
        (None, None) => DEFAULT_INTER_LOG_TIME_CONSTANT,
    };
    let new_log_tau = perturb_clamped(
        base_tau,
        INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV * 0.5,
        INTER_LOG_TIME_CONSTANT_MIN,
        INTER_LOG_TIME_CONSTANT_MAX,
        rng,
    );
    let new_bias = perturb_clamped(
        0.0,
        BIAS_PERTURBATION_STDDEV * 0.5,
        -BIAS_MAX,
        BIAS_MAX,
        rng,
    );

    let pre_location = location_for_neuron(selected_edge.pre_neuron_id, genome);
    let post_location = location_for_neuron(selected_edge.post_neuron_id, genome);
    let midpoint = BrainLocation {
        x: 0.5 * (pre_location.x + post_location.x),
        y: 0.5 * (pre_location.y + post_location.y),
    };
    let new_location = BrainLocation {
        x: perturb_clamped(
            midpoint.x,
            LOCATION_PERTURBATION_STDDEV * 0.5,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        ),
        y: perturb_clamped(
            midpoint.y,
            LOCATION_PERTURBATION_STDDEV * 0.5,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        ),
    };

    let new_type = if selected_edge.weight.is_sign_negative() {
        InterNeuronType::Inhibitory
    } else {
        InterNeuronType::Excitatory
    };

    genome.num_neurons = genome.num_neurons.saturating_add(1);
    genome.inter_biases.push(new_bias);
    genome.inter_log_time_constants.push(new_log_tau);
    genome.interneuron_types.push(new_type);
    genome.inter_locations.push(new_location);

    let pre_to_new_required_sign = required_pre_sign(
        selected_edge.pre_neuron_id,
        genome.num_neurons,
        &genome.interneuron_types,
    )
    .unwrap_or(1.0);
    let new_to_post_required_sign =
        required_pre_sign(new_inter_id, genome.num_neurons, &genome.interneuron_types)
            .unwrap_or(1.0);

    genome.edges.push(SynapseEdge {
        pre_neuron_id: selected_edge.pre_neuron_id,
        post_neuron_id: new_inter_id,
        weight: constrain_weight_to_sign(selected_edge.weight, pre_to_new_required_sign),
        eligibility: 0.0,
    });
    genome.edges.push(SynapseEdge {
        pre_neuron_id: new_inter_id,
        post_neuron_id: selected_edge.post_neuron_id,
        weight: constrain_weight_to_sign(
            if selected_edge.weight.is_sign_negative() {
                -1.0
            } else {
                1.0
            },
            new_to_post_required_sign,
        ),
        eligibility: 0.0,
    });

    genome.num_synapses = genome.num_synapses.saturating_add(1);
    sync_synapse_genes_to_target(genome, rng);
}

fn select_weighted_edge_index<R: Rng + ?Sized>(edges: &[SynapseEdge], rng: &mut R) -> usize {
    if edges.len() <= 1 {
        return 0;
    }

    let total_weight: f32 = edges
        .iter()
        .map(|edge| edge.weight.abs().max(SYNAPSE_STRENGTH_MIN))
        .sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return rng.random_range(0..edges.len());
    }

    let mut sample = rng.random_range(0.0..total_weight);
    for (idx, edge) in edges.iter().enumerate() {
        sample -= edge.weight.abs().max(SYNAPSE_STRENGTH_MIN);
        if sample <= 0.0 {
            return idx;
        }
    }

    edges.len() - 1
}

fn inter_log_time_constant_for_neuron(neuron_id: NeuronId, genome: &OrganismGenome) -> Option<f32> {
    if !is_inter_id(neuron_id, genome.num_neurons) {
        return None;
    }
    let inter_idx = (neuron_id.0 - INTER_ID_BASE) as usize;
    genome.inter_log_time_constants.get(inter_idx).copied()
}

fn location_for_neuron(neuron_id: NeuronId, genome: &OrganismGenome) -> BrainLocation {
    if neuron_id.0 < SENSORY_COUNT {
        return genome
            .sensory_locations
            .get(neuron_id.0 as usize)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
    }

    if is_inter_id(neuron_id, genome.num_neurons) {
        let inter_idx = (neuron_id.0 - INTER_ID_BASE) as usize;
        return genome
            .inter_locations
            .get(inter_idx)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
    }

    if (ACTION_ID_BASE..ACTION_ID_BASE + ACTION_COUNT_U32).contains(&neuron_id.0) {
        let action_idx = (neuron_id.0 - ACTION_ID_BASE) as usize;
        return genome
            .action_locations
            .get(action_idx)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
    }

    BrainLocation { x: 5.0, y: 5.0 }
}

fn sync_synapse_genes_to_target<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    sanitize_synapse_genes(genome);

    let target = genome.num_synapses as usize;
    if genome.edges.len() < target {
        add_synapse_genes_with_spatial_prior(genome, target - genome.edges.len(), rng);
    }

    sort_synapse_genes(&mut genome.edges);
    genome.num_synapses = genome.edges.len() as u32;
}

fn sanitize_synapse_genes(genome: &mut OrganismGenome) {
    let num_neurons = genome.num_neurons;
    genome
        .edges
        .retain(|edge| is_valid_synapse_pair(edge.pre_neuron_id, edge.post_neuron_id, num_neurons));

    for edge in &mut genome.edges {
        let required_sign =
            required_pre_sign(edge.pre_neuron_id, num_neurons, &genome.interneuron_types)
                .unwrap_or(1.0);
        edge.weight = constrain_weight_to_sign(edge.weight, required_sign);
        edge.eligibility = 0.0;
    }

    sort_synapse_genes(&mut genome.edges);
    genome.edges.dedup_by(|a, b| {
        a.pre_neuron_id == b.pre_neuron_id && a.post_neuron_id == b.post_neuron_id
    });
}

fn add_synapse_genes_with_spatial_prior<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    add_count: usize,
    rng: &mut R,
) {
    if add_count == 0 {
        return;
    }

    let mut existing_pairs: HashSet<(u32, u32)> = HashSet::with_capacity(genome.edges.len());
    for edge in &genome.edges {
        existing_pairs.insert((edge.pre_neuron_id.0, edge.post_neuron_id.0));
    }

    let mut weighted_candidates: Vec<(f32, NeuronId, NeuronId)> = Vec::new();

    let num_neurons = genome.num_neurons;
    for sensory_idx in 0..SENSORY_COUNT {
        let pre_id = NeuronId(sensory_idx);
        for post_id in post_ids(num_neurons) {
            if existing_pairs.contains(&(pre_id.0, post_id.0)) {
                continue;
            }
            let probability = connection_probability(genome, pre_id, post_id);
            let priority = weighted_without_replacement_priority(probability, rng);
            weighted_candidates.push((priority, pre_id, post_id));
        }
    }

    for inter_idx in 0..num_neurons {
        let pre_id = NeuronId(INTER_ID_BASE + inter_idx);
        for post_id in post_ids(num_neurons) {
            if !is_valid_synapse_pair(pre_id, post_id, num_neurons) {
                continue;
            }
            if existing_pairs.contains(&(pre_id.0, post_id.0)) {
                continue;
            }
            let probability = connection_probability(genome, pre_id, post_id);
            let priority = weighted_without_replacement_priority(probability, rng);
            weighted_candidates.push((priority, pre_id, post_id));
        }
    }

    weighted_candidates.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    for &(_, pre_id, post_id) in weighted_candidates.iter().take(add_count) {
        let required_sign =
            required_pre_sign(pre_id, num_neurons, &genome.interneuron_types).unwrap_or(1.0);
        genome.edges.push(SynapseEdge {
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            weight: sample_signed_lognormal_weight(required_sign, rng),
            eligibility: 0.0,
        });
    }
}

fn post_ids(num_neurons: u32) -> impl Iterator<Item = NeuronId> {
    let inter = (0..num_neurons).map(|idx| NeuronId(INTER_ID_BASE + idx));
    let actions = (0..ACTION_COUNT_U32).map(|idx| NeuronId(ACTION_ID_BASE + idx));
    inter.chain(actions)
}

fn connection_probability(genome: &OrganismGenome, pre: NeuronId, post: NeuronId) -> f32 {
    let pre_location = location_for_neuron(pre, genome);
    let post_location = location_for_neuron(post, genome);
    let dx = pre_location.x - post_location.x;
    let dy = pre_location.y - post_location.y;
    let distance_sq = dx * dx + dy * dy;
    let sigma = genome.spatial_prior_sigma.max(0.01);
    let sigma_sq = sigma * sigma;
    let local_bias = (-0.5 * distance_sq / sigma_sq).exp();
    (SPATIAL_PRIOR_LONG_RANGE_FLOOR + (1.0 - SPATIAL_PRIOR_LONG_RANGE_FLOOR) * local_bias)
        .clamp(0.0, 1.0)
}

fn weighted_without_replacement_priority<R: Rng + ?Sized>(weight: f32, rng: &mut R) -> f32 {
    let clamped_weight = weight.max(f32::MIN_POSITIVE);
    let u = rng.random::<f32>().max(f32::MIN_POSITIVE);
    -u.ln() / clamped_weight
}

fn sample_signed_lognormal_weight<R: Rng + ?Sized>(required_sign: f32, rng: &mut R) -> f32 {
    let z = standard_normal(rng);
    let magnitude = (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    if required_sign.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

fn is_valid_synapse_pair(pre: NeuronId, post: NeuronId, num_neurons: u32) -> bool {
    if !is_valid_pre_id(pre, num_neurons) || !is_valid_post_id(post, num_neurons) {
        return false;
    }

    if is_inter_id(pre, num_neurons) && is_inter_id(post, num_neurons) && pre == post {
        return false;
    }

    true
}

fn is_valid_pre_id(id: NeuronId, num_neurons: u32) -> bool {
    id.0 < SENSORY_COUNT || is_inter_id(id, num_neurons)
}

fn is_valid_post_id(id: NeuronId, num_neurons: u32) -> bool {
    is_inter_id(id, num_neurons)
        || (ACTION_ID_BASE..ACTION_ID_BASE + ACTION_COUNT_U32).contains(&id.0)
}

fn is_inter_id(id: NeuronId, num_neurons: u32) -> bool {
    (INTER_ID_BASE..INTER_ID_BASE + num_neurons).contains(&id.0)
}

fn required_pre_sign(
    pre: NeuronId,
    num_neurons: u32,
    interneuron_types: &[InterNeuronType],
) -> Option<f32> {
    if pre.0 < SENSORY_COUNT {
        return Some(1.0);
    }
    if is_inter_id(pre, num_neurons) {
        let idx = (pre.0 - INTER_ID_BASE) as usize;
        let inter_type = interneuron_types
            .get(idx)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        return Some(match inter_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        });
    }
    None
}

fn sort_synapse_genes(edges: &mut [SynapseEdge]) {
    edges.sort_unstable_by(|a, b| {
        synapse_key_cmp(a, b)
            .then_with(|| a.weight.total_cmp(&b.weight))
            .then_with(|| a.eligibility.total_cmp(&b.eligibility))
    });
}

fn synapse_key_cmp(a: &SynapseEdge, b: &SynapseEdge) -> Ordering {
    a.pre_neuron_id
        .cmp(&b.pre_neuron_id)
        .then_with(|| a.post_neuron_id.cmp(&b.post_neuron_id))
}

fn constrain_weight_to_sign(weight: f32, required_sign: f32) -> f32 {
    if required_sign.is_sign_negative() {
        if weight >= 0.0 {
            return -SYNAPSE_STRENGTH_MIN;
        }
        -(-weight).clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
    } else {
        if weight <= 0.0 {
            return SYNAPSE_STRENGTH_MIN;
        }
        weight.clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
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

fn sample_uniform_log_time_constant<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    rng.random_range(INTER_LOG_TIME_CONSTANT_MIN..=INTER_LOG_TIME_CONSTANT_MAX)
}

fn sample_uniform_location<R: Rng + ?Sized>(rng: &mut R) -> BrainLocation {
    BrainLocation {
        x: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
        y: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
    }
}

pub(crate) fn inter_alpha_from_log_time_constant(log_time_constant: f32) -> f32 {
    let clamped_log_time_constant =
        log_time_constant.clamp(INTER_LOG_TIME_CONSTANT_MIN, INTER_LOG_TIME_CONSTANT_MAX);
    let time_constant = clamped_log_time_constant
        .exp()
        .clamp(INTER_TIME_CONSTANT_MIN, INTER_TIME_CONSTANT_MAX);
    1.0 - (-1.0 / time_constant).exp()
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

fn synapse_gene_distance(a: &[SynapseEdge], b: &[SynapseEdge]) -> f32 {
    let mut distance = 0.0;

    let mut a_idx = 0usize;
    let mut b_idx = 0usize;

    while a_idx < a.len() && b_idx < b.len() {
        match synapse_key_cmp(&a[a_idx], &b[b_idx]) {
            Ordering::Equal => {
                distance += (a[a_idx].weight - b[b_idx].weight).abs();
                a_idx += 1;
                b_idx += 1;
            }
            Ordering::Less => {
                distance += 1.0 + a[a_idx].weight.abs();
                a_idx += 1;
            }
            Ordering::Greater => {
                distance += 1.0 + b[b_idx].weight.abs();
                b_idx += 1;
            }
        }
    }

    while a_idx < a.len() {
        distance += 1.0 + a[a_idx].weight.abs();
        a_idx += 1;
    }
    while b_idx < b.len() {
        distance += 1.0 + b[b_idx].weight.abs();
        b_idx += 1;
    }

    distance
}

/// L1 genome distance: scalar traits + topology + brain geometry.
pub(crate) fn genome_distance(a: &OrganismGenome, b: &OrganismGenome) -> f32 {
    let mut dist = (a.num_neurons as f32 - b.num_neurons as f32).abs()
        + (a.num_synapses as f32 - b.num_synapses as f32).abs()
        + (a.spatial_prior_sigma - b.spatial_prior_sigma).abs()
        + (a.vision_distance as f32 - b.vision_distance as f32).abs()
        + (a.starting_energy - b.starting_energy).abs()
        + (a.age_of_maturity as f32 - b.age_of_maturity as f32).abs()
        + (a.hebb_eta_gain - b.hebb_eta_gain).abs()
        + (a.eligibility_retention - b.eligibility_retention).abs()
        + (a.synapse_prune_threshold - b.synapse_prune_threshold).abs();

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

    dist += synapse_gene_distance(&a.edges, &b.edges);

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
    let all_pairs = pre_count.saturating_mul(post_count);
    let max = all_pairs.saturating_sub(u64::from(num_neurons));
    max.min(u64::from(u32::MAX)) as u32
}

pub(crate) fn validate_seed_genome_config(config: &SeedGenomeConfig) -> Result<(), SimError> {
    if !(ETA_GAIN_MIN..=ETA_GAIN_MAX).contains(&config.hebb_eta_gain) {
        return Err(SimError::InvalidConfig(format!(
            "hebb_eta_gain must be within [{ETA_GAIN_MIN}, {ETA_GAIN_MAX}]"
        )));
    }
    if !(ELIGIBILITY_RETENTION_MIN..=ELIGIBILITY_RETENTION_MAX)
        .contains(&config.eligibility_retention)
    {
        return Err(SimError::InvalidConfig(format!(
            "eligibility_retention must be within [{ELIGIBILITY_RETENTION_MIN}, {ELIGIBILITY_RETENTION_MAX}]"
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
    if !config.starting_energy.is_finite() || config.starting_energy <= 0.0 {
        return Err(SimError::InvalidConfig(
            "starting_energy must be greater than zero".to_owned(),
        ));
    }
    validate_rate(
        "mutation_rate_age_of_maturity",
        config.mutation_rate_age_of_maturity,
    )?;
    validate_rate(
        "mutation_rate_vision_distance",
        config.mutation_rate_vision_distance,
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
        "mutation_rate_eligibility_retention",
        config.mutation_rate_eligibility_retention,
    )?;
    validate_rate(
        "mutation_rate_synapse_prune_threshold",
        config.mutation_rate_synapse_prune_threshold,
    )?;
    validate_rate(
        "mutation_rate_neuron_location",
        config.mutation_rate_neuron_location,
    )?;
    validate_rate(
        "mutation_rate_synapse_weight_perturbation",
        config.mutation_rate_synapse_weight_perturbation,
    )?;
    validate_rate(
        "mutation_rate_add_neuron_split_edge",
        config.mutation_rate_add_neuron_split_edge,
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
