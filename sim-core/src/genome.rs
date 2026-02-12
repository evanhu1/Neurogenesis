use crate::brain::{ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT};
use crate::SimError;
use rand::Rng;
use sim_types::{InterNeuronType, NeuronId, OrganismGenome, SeedGenomeConfig, SynapseEdge};
use std::cmp::Ordering;

const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const SYNAPSE_STRENGTH_MAX: f32 = 4.0;
const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;

const SYNAPSE_WEIGHT_LOG_NORMAL_MU: f32 = -0.5;
const SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA: f32 = 0.8;
const INTER_TYPE_EXCITATORY_PRIOR: f32 = 0.8;

const WEIGHT_PERTURBATION_STDDEV: f32 = 0.3;
const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
const INTER_LOG_TAU_PERTURBATION_STDDEV: f32 = 0.05;
pub(crate) const INTER_TAU_MIN: f32 = 0.1;
pub(crate) const INTER_TAU_MAX: f32 = 15.0;
pub(crate) const INTER_LOG_TAU_MIN: f32 = -2.302_585_1;
pub(crate) const INTER_LOG_TAU_MAX: f32 = 2.995_732_3;
pub(crate) const DEFAULT_INTER_LOG_TAU: f32 = 0.0;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    world_max_num_neurons: u32,
    rng: &mut R,
) -> OrganismGenome {
    let inter_biases: Vec<f32> = (0..world_max_num_neurons)
        .map(|_| sample_initial_bias(rng))
        .collect();
    let inter_log_taus: Vec<f32> = (0..world_max_num_neurons)
        .map(|_| sample_uniform_log_tau(rng))
        .collect();
    let interneuron_types: Vec<InterNeuronType> = (0..world_max_num_neurons)
        .map(|_| sample_interneuron_type(rng))
        .collect();
    let action_biases: Vec<f32> = (0..ACTION_COUNT)
        .map(|_| sample_initial_bias(rng))
        .collect();

    let mut edges = Vec::new();
    for _ in 0..config.num_synapses {
        if let Some(edge) = random_edge(config.num_neurons, &interneuron_types, &edges, rng) {
            edges.push(edge);
        }
    }

    sort_edges(&mut edges);

    OrganismGenome {
        num_neurons: config.num_neurons.min(world_max_num_neurons),
        vision_distance: config.vision_distance,
        mutation_rate_vision_distance: config.mutation_rate_vision_distance,
        mutation_rate_weight: config.mutation_rate_weight,
        mutation_rate_add_edge: config.mutation_rate_add_edge,
        mutation_rate_remove_edge: config.mutation_rate_remove_edge,
        mutation_rate_split_edge: config.mutation_rate_split_edge,
        mutation_rate_inter_bias: config.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: config.mutation_rate_inter_update_rate,
        mutation_rate_action_bias: config.mutation_rate_action_bias,
        inter_biases,
        inter_log_taus,
        interneuron_types,
        action_biases,
        edges,
    }
}

fn edge_key(e: &SynapseEdge) -> (NeuronId, NeuronId) {
    (e.pre_neuron_id, e.post_neuron_id)
}

fn sort_edges(edges: &mut [SynapseEdge]) {
    edges.sort_by_key(edge_key);
}

fn debug_assert_edges_sorted(edges: &[SynapseEdge]) {
    debug_assert!(
        edges.windows(2).all(|w| edge_key(&w[0]) <= edge_key(&w[1])),
        "genome edges must be sorted by (pre, post)"
    );
}

fn sample_interneuron_type<R: Rng + ?Sized>(rng: &mut R) -> InterNeuronType {
    if rng.random::<f32>() < INTER_TYPE_EXCITATORY_PRIOR {
        InterNeuronType::Excitatory
    } else {
        InterNeuronType::Inhibitory
    }
}

fn source_weight_sign(
    pre: NeuronId,
    num_neurons: u32,
    inter_types: &[InterNeuronType],
) -> Option<f32> {
    if pre.0 < SENSORY_COUNT {
        return Some(1.0);
    }
    if pre.0 >= INTER_ID_BASE && pre.0 < INTER_ID_BASE + num_neurons {
        let idx = (pre.0 - INTER_ID_BASE) as usize;
        let neuron_type = inter_types
            .get(idx)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        return Some(match neuron_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        });
    }
    None
}

fn clamp_signed_weight(weight: f32, required_sign: f32) -> f32 {
    let magnitude = weight
        .abs()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    if required_sign.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

fn sample_log_normal_magnitude<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let z = normal_sample(rng);
    (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
}

fn sample_signed_lognormal_weight<R: Rng + ?Sized>(required_sign: f32, rng: &mut R) -> f32 {
    let magnitude = sample_log_normal_magnitude(rng);
    if required_sign.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

fn random_edge<R: Rng + ?Sized>(
    num_neurons: u32,
    inter_types: &[InterNeuronType],
    existing: &[SynapseEdge],
    rng: &mut R,
) -> Option<SynapseEdge> {
    // Pre neurons: sensory (0..SENSORY_COUNT) + enabled inter (INTER_ID_BASE..INTER_ID_BASE+num_neurons)
    // Post neurons: enabled inter (INTER_ID_BASE..INTER_ID_BASE+num_neurons) + action (ACTION_ID_BASE..ACTION_ID_BASE+ACTION_COUNT)
    let pre_count = SENSORY_COUNT + num_neurons;
    let post_count = num_neurons + ACTION_COUNT_U32;

    if pre_count == 0 || post_count == 0 {
        return None;
    }

    // Try up to 20 times to find a non-duplicate edge.
    // Self-recurrence is allowed for enabled inter neurons.
    for _ in 0..20 {
        let pre_idx = rng.random_range(0..pre_count);
        let pre = if pre_idx < SENSORY_COUNT {
            NeuronId(pre_idx)
        } else {
            NeuronId(INTER_ID_BASE + (pre_idx - SENSORY_COUNT))
        };

        let post_idx = rng.random_range(0..post_count);
        let post = if post_idx < num_neurons {
            NeuronId(INTER_ID_BASE + post_idx)
        } else {
            NeuronId(ACTION_ID_BASE + (post_idx - num_neurons))
        };

        if pre == post && !is_enabled_inter_neuron(pre, num_neurons) {
            continue;
        }

        let is_dup = existing
            .iter()
            .any(|e| e.pre_neuron_id == pre && e.post_neuron_id == post);
        if is_dup {
            continue;
        }

        let sign = source_weight_sign(pre, num_neurons, inter_types)?;
        let weight = sample_signed_lognormal_weight(sign, rng);
        return Some(SynapseEdge {
            pre_neuron_id: pre,
            post_neuron_id: post,
            weight,
        });
    }
    None
}

fn mutation_rate_genes_mut(genome: &mut OrganismGenome) -> [&mut f32; 8] {
    [
        &mut genome.mutation_rate_vision_distance,
        &mut genome.mutation_rate_weight,
        &mut genome.mutation_rate_add_edge,
        &mut genome.mutation_rate_remove_edge,
        &mut genome.mutation_rate_split_edge,
        &mut genome.mutation_rate_inter_bias,
        &mut genome.mutation_rate_inter_update_rate,
        &mut genome.mutation_rate_action_bias,
    ]
}

fn mutation_rate_genes(genome: &OrganismGenome) -> [f32; 8] {
    [
        genome.mutation_rate_vision_distance,
        genome.mutation_rate_weight,
        genome.mutation_rate_add_edge,
        genome.mutation_rate_remove_edge,
        genome.mutation_rate_split_edge,
        genome.mutation_rate_inter_bias,
        genome.mutation_rate_inter_update_rate,
        genome.mutation_rate_action_bias,
    ]
}

fn mutate_mutation_rate_genes<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let mut rates = mutation_rate_genes_mut(genome);
    let shared_normal = normal_sample(rng);

    for rate in &mut rates {
        if **rate <= 0.0 {
            continue;
        }
        let gene_normal = normal_sample(rng);
        **rate = (**rate * (shared_normal - gene_normal).exp()).clamp(0.0, 1.0);
    }
}

fn align_genome_vectors<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    world_max_num_neurons: u32,
    rng: &mut R,
) {
    genome.num_neurons = genome.num_neurons.min(world_max_num_neurons);

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

    if genome.action_biases.len() < ACTION_COUNT {
        genome.action_biases.resize(ACTION_COUNT, 0.0);
    } else if genome.action_biases.len() > ACTION_COUNT {
        genome.action_biases.truncate(ACTION_COUNT);
    }
}

fn mutate_split_edge<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    world_max_num_neurons: u32,
    rng: &mut R,
) {
    if genome.edges.is_empty() || genome.num_neurons >= world_max_num_neurons {
        return;
    }

    let split_idx = rng.random_range(0..genome.edges.len());
    let split_edge = genome.edges.swap_remove(split_idx);

    let new_idx = genome.num_neurons as usize;
    if new_idx >= genome.inter_biases.len()
        || new_idx >= genome.inter_log_taus.len()
        || new_idx >= genome.interneuron_types.len()
    {
        genome.edges.push(split_edge);
        return;
    }

    let new_type = sample_interneuron_type(rng);
    genome.interneuron_types[new_idx] = new_type;
    genome.inter_biases[new_idx] = sample_initial_bias(rng);
    genome.inter_log_taus[new_idx] = sample_uniform_log_tau(rng);

    let new_inter_id = NeuronId(INTER_ID_BASE + genome.num_neurons);
    genome.num_neurons += 1;

    let pre_sign = source_weight_sign(
        split_edge.pre_neuron_id,
        genome.num_neurons.saturating_sub(1),
        &genome.interneuron_types,
    )
    .unwrap_or(1.0);
    let edge_to_new = SynapseEdge {
        pre_neuron_id: split_edge.pre_neuron_id,
        post_neuron_id: new_inter_id,
        weight: clamp_signed_weight(split_edge.weight, pre_sign),
    };

    let new_sign = match new_type {
        InterNeuronType::Excitatory => 1.0,
        InterNeuronType::Inhibitory => -1.0,
    };
    let edge_from_new = SynapseEdge {
        pre_neuron_id: new_inter_id,
        post_neuron_id: split_edge.post_neuron_id,
        weight: sample_signed_lognormal_weight(new_sign, rng),
    };

    genome.edges.push(edge_to_new);
    genome.edges.push(edge_from_new);
}

pub(crate) fn mutate_genome<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    world_max_num_neurons: u32,
    rng: &mut R,
) {
    align_genome_vectors(genome, world_max_num_neurons, rng);
    mutate_mutation_rate_genes(genome, rng);

    if rng.random::<f32>() < genome.mutation_rate_vision_distance {
        genome.vision_distance = step_u32(
            genome.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_weight && !genome.edges.is_empty() {
        let idx = rng.random_range(0..genome.edges.len());
        let edge = &mut genome.edges[idx];
        if let Some(required_sign) = source_weight_sign(
            edge.pre_neuron_id,
            genome.num_neurons,
            &genome.interneuron_types,
        ) {
            let mutated = edge.weight + normal_sample(rng) * WEIGHT_PERTURBATION_STDDEV;
            edge.weight = clamp_signed_weight(mutated, required_sign);
        }
    }

    if rng.random::<f32>() < genome.mutation_rate_add_edge {
        if let Some(edge) = random_edge(
            genome.num_neurons,
            &genome.interneuron_types,
            &genome.edges,
            rng,
        ) {
            genome.edges.push(edge);
        }
    }

    if rng.random::<f32>() < genome.mutation_rate_remove_edge && !genome.edges.is_empty() {
        let idx = rng.random_range(0..genome.edges.len());
        genome.edges.swap_remove(idx);
    }

    if rng.random::<f32>() < genome.mutation_rate_split_edge {
        mutate_split_edge(genome, world_max_num_neurons, rng);
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

    sort_edges(&mut genome.edges);
    debug_assert_edges_sorted(&genome.edges);
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

fn is_enabled_inter_neuron(id: NeuronId, num_neurons: u32) -> bool {
    id.0 >= INTER_ID_BASE && id.0 < INTER_ID_BASE + num_neurons
}

fn perturb_clamped<R: Rng + ?Sized>(
    value: f32,
    stddev: f32,
    min: f32,
    max: f32,
    rng: &mut R,
) -> f32 {
    let normal = normal_sample(rng);
    (value + normal * stddev).clamp(min, max)
}

fn normal_sample<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// L1 genome distance: scalar traits + mutation-rate genes + vectors.
pub(crate) fn genome_distance(a: &OrganismGenome, b: &OrganismGenome) -> f32 {
    let mut dist = (a.num_neurons as f32 - b.num_neurons as f32).abs()
        + (a.vision_distance as f32 - b.vision_distance as f32).abs();

    let a_rates = mutation_rate_genes(a);
    let b_rates = mutation_rate_genes(b);
    for i in 0..a_rates.len() {
        dist += (a_rates[i] - b_rates[i]).abs();
    }

    let max_type_len = a.interneuron_types.len().max(b.interneuron_types.len());
    for i in 0..max_type_len {
        let ta = a
            .interneuron_types
            .get(i)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        let tb: InterNeuronType = b
            .interneuron_types
            .get(i)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        if ta != tb {
            dist += 1.0;
        }
    }

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

pub(crate) fn validate_seed_genome_config(config: &SeedGenomeConfig) -> Result<(), SimError> {
    validate_rate(
        "mutation_rate_vision_distance",
        config.mutation_rate_vision_distance,
    )?;
    validate_rate("mutation_rate_weight", config.mutation_rate_weight)?;
    validate_rate("mutation_rate_add_edge", config.mutation_rate_add_edge)?;
    validate_rate(
        "mutation_rate_remove_edge",
        config.mutation_rate_remove_edge,
    )?;
    validate_rate("mutation_rate_split_edge", config.mutation_rate_split_edge)?;
    validate_rate("mutation_rate_inter_bias", config.mutation_rate_inter_bias)?;
    validate_rate(
        "mutation_rate_inter_update_rate",
        config.mutation_rate_inter_update_rate,
    )?;
    validate_rate(
        "mutation_rate_action_bias",
        config.mutation_rate_action_bias,
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
    Ok(())
}
