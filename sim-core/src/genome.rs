use crate::neuron_constants::{
    ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, INTER_UPDATE_RATE_MAX,
    SENSORY_COUNT, INTER_UPDATE_RATE_MIN,
};
use crate::SimError;
use rand::Rng;
use sim_types::{NeuronId, OrganismGenome, SeedGenomeConfig, SynapseEdge};
use std::cmp::Ordering;

const MAX_MUTATED_INTER_NEURONS: u32 = 256;
const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const SYNAPSE_STRENGTH_MAX: f32 = 2.0;
const BIAS_MAX: f32 = 1.0;
const INTER_UPDATE_RATE_PERTURBATION_STDDEV: f32 = 0.05;
const WEIGHT_PERTURBATION_STDDEV: f32 = 0.3;
const MUTATION_RATE_STEP: f32 = 0.01;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let inter_biases: Vec<f32> = (0..config.max_num_neurons)
        .map(|_| rng.random_range(-BIAS_MAX..BIAS_MAX))
        .collect();
    let inter_update_rates: Vec<f32> = (0..config.max_num_neurons)
        .map(|_| sample_log_uniform_update_rate(rng))
        .collect();
    let action_biases = vec![0.0; ACTION_COUNT];

    let mut edges = Vec::new();
    for _ in 0..config.num_synapses {
        if let Some(edge) = random_edge(config.num_neurons, &edges, rng) {
            edges.push(edge);
        }
    }

    sort_edges(&mut edges);

    OrganismGenome {
        num_neurons: config.num_neurons,
        max_num_neurons: config.max_num_neurons,
        vision_distance: config.vision_distance,
        mutation_rate: config.mutation_rate,
        inter_biases,
        inter_update_rates,
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

fn random_edge<R: Rng + ?Sized>(
    num_neurons: u32,
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

        let weight = rng.random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
        return Some(SynapseEdge {
            pre_neuron_id: pre,
            post_neuron_id: post,
            weight,
        });
    }
    None
}

pub(crate) fn mutate_genome<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let rate = genome.mutation_rate;

    // Trait mutations, each gated by mutation_rate
    if rng.random::<f32>() < rate {
        mutate_num_neurons(genome, rng);
    }
    if rng.random::<f32>() < rate {
        genome.max_num_neurons = step_u32(
            genome.max_num_neurons,
            genome.num_neurons,
            MAX_MUTATED_INTER_NEURONS,
            rng,
        );
    }
    if rng.random::<f32>() < rate {
        genome.vision_distance = step_u32(
            genome.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }
    if rng.random::<f32>() < rate {
        let delta = if rng.random::<bool>() {
            MUTATION_RATE_STEP
        } else {
            -MUTATION_RATE_STEP
        };
        genome.mutation_rate = (genome.mutation_rate + delta).clamp(0.0, 1.0);
    }

    // Keep genome vectors aligned with current shape.
    let target_inter_len = genome.max_num_neurons as usize;
    if genome.inter_biases.len() < target_inter_len {
        genome.inter_biases.resize(target_inter_len, 0.0);
    }
    if genome.inter_update_rates.len() < target_inter_len {
        genome
            .inter_update_rates
            .resize(target_inter_len, INTER_UPDATE_RATE_MAX);
    }
    if genome.action_biases.len() < ACTION_COUNT {
        genome.action_biases.resize(ACTION_COUNT, 0.0);
    }

    // Graph mutations
    // Weight perturbation (2x rate — most common)
    if rng.random::<f32>() < rate * 2.0 && !genome.edges.is_empty() {
        let idx = rng.random_range(0..genome.edges.len());
        genome.edges[idx].weight = perturb(
            genome.edges[idx].weight,
            WEIGHT_PERTURBATION_STDDEV,
            SYNAPSE_STRENGTH_MAX,
            rng,
        );
    }

    // Add edge
    if rng.random::<f32>() < rate {
        if let Some(edge) = random_edge(genome.num_neurons, &genome.edges, rng) {
            genome.edges.push(edge);
        }
    }

    // Remove edge
    if rng.random::<f32>() < rate && !genome.edges.is_empty() {
        let idx = rng.random_range(0..genome.edges.len());
        genome.edges.swap_remove(idx);
    }

    // Bias perturbation
    if rng.random::<f32>() < rate && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_biases[idx] = perturb(
            genome.inter_biases[idx],
            WEIGHT_PERTURBATION_STDDEV,
            BIAS_MAX,
            rng,
        );
    }

    // Inter update-rate perturbation
    if rng.random::<f32>() < rate && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_update_rates[idx] = perturb_range(
            genome.inter_update_rates[idx],
            INTER_UPDATE_RATE_PERTURBATION_STDDEV,
            INTER_UPDATE_RATE_MIN,
            INTER_UPDATE_RATE_MAX,
            rng,
        );
    }

    // Action bias perturbation
    if rng.random::<f32>() < rate && !genome.action_biases.is_empty() {
        let idx = rng.random_range(0..genome.action_biases.len());
        genome.action_biases[idx] = perturb(
            genome.action_biases[idx],
            WEIGHT_PERTURBATION_STDDEV,
            BIAS_MAX,
            rng,
        );
    }

    sort_edges(&mut genome.edges);
    debug_assert_edges_sorted(&genome.edges);
}

fn mutate_num_neurons<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let old = genome.num_neurons;
    genome.num_neurons = step_u32(old, 0, genome.max_num_neurons, rng);

    if genome.num_neurons > old {
        // Increase: init new neuron bias randomly, add 1-2 random edges to/from it
        let new_idx = old;
        if (new_idx as usize) < genome.inter_biases.len() {
            genome.inter_biases[new_idx as usize] = rng.random_range(-BIAS_MAX..BIAS_MAX);
        }
        if (new_idx as usize) < genome.inter_update_rates.len() {
            genome.inter_update_rates[new_idx as usize] = sample_log_uniform_update_rate(rng);
        }
        let edge_count = rng.random_range(1..=2u32);
        for _ in 0..edge_count {
            if let Some(edge) = random_edge(genome.num_neurons, &genome.edges, rng) {
                genome.edges.push(edge);
            }
        }
    } else if genome.num_neurons < old {
        // Decrease: remove edges incident to the disabled neuron
        let disabled_id = NeuronId(INTER_ID_BASE + genome.num_neurons);
        genome
            .edges
            .retain(|e| e.pre_neuron_id != disabled_id && e.post_neuron_id != disabled_id);
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

fn sample_log_uniform_update_rate<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u: f32 = rng.random();
    let ratio = INTER_UPDATE_RATE_MAX / INTER_UPDATE_RATE_MIN;
    INTER_UPDATE_RATE_MIN * ratio.powf(u)
}

fn is_enabled_inter_neuron(id: NeuronId, num_neurons: u32) -> bool {
    id.0 >= INTER_ID_BASE && id.0 < INTER_ID_BASE + num_neurons
}

/// Sample from a normal-ish distribution using Box-Muller, then clamp.
fn perturb<R: Rng + ?Sized>(value: f32, stddev: f32, clamp_abs: f32, rng: &mut R) -> f32 {
    let normal = normal_sample(rng);
    (value + normal * stddev).clamp(-clamp_abs, clamp_abs)
}

/// Sample from a normal-ish distribution using Box-Muller, then clamp to [min, max].
fn perturb_range<R: Rng + ?Sized>(value: f32, stddev: f32, min: f32, max: f32, rng: &mut R) -> f32 {
    let normal = normal_sample(rng);
    (value + normal * stddev).clamp(min, max)
}

fn normal_sample<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// L1 genome distance: scalar traits + bias vectors + merge-join over sorted edges.
pub(crate) fn genome_distance(a: &OrganismGenome, b: &OrganismGenome) -> f32 {
    debug_assert_edges_sorted(&a.edges);
    debug_assert_edges_sorted(&b.edges);

    let mut dist = (a.num_neurons as f32 - b.num_neurons as f32).abs()
        + (a.max_num_neurons as f32 - b.max_num_neurons as f32).abs()
        + (a.vision_distance as f32 - b.vision_distance as f32).abs()
        + (a.mutation_rate - b.mutation_rate).abs();

    let max_bias_len = a.inter_biases.len().max(b.inter_biases.len());
    for i in 0..max_bias_len {
        let ba = a.inter_biases.get(i).copied().unwrap_or(0.0);
        let bb = b.inter_biases.get(i).copied().unwrap_or(0.0);
        dist += (ba - bb).abs();
    }

    let max_update_rate_len = a.inter_update_rates.len().max(b.inter_update_rates.len());
    for i in 0..max_update_rate_len {
        let aa = a
            .inter_update_rates
            .get(i)
            .copied()
            .unwrap_or(INTER_UPDATE_RATE_MAX);
        let bb = b
            .inter_update_rates
            .get(i)
            .copied()
            .unwrap_or(INTER_UPDATE_RATE_MAX);
        dist += (aa - bb).abs();
    }

    let max_action_bias_len = a.action_biases.len().max(b.action_biases.len());
    for i in 0..max_action_bias_len {
        let ba = a.action_biases.get(i).copied().unwrap_or(0.0);
        let bb = b.action_biases.get(i).copied().unwrap_or(0.0);
        dist += (ba - bb).abs();
    }

    // Merge-join over sorted edges
    let (mut ai, mut bi) = (0, 0);
    while ai < a.edges.len() && bi < b.edges.len() {
        match edge_key(&a.edges[ai]).cmp(&edge_key(&b.edges[bi])) {
            Ordering::Equal => {
                dist += (a.edges[ai].weight - b.edges[bi].weight).abs();
                ai += 1;
                bi += 1;
            }
            Ordering::Less => {
                dist += a.edges[ai].weight.abs();
                ai += 1;
            }
            Ordering::Greater => {
                dist += b.edges[bi].weight.abs();
                bi += 1;
            }
        }
    }
    for edge in &a.edges[ai..] {
        dist += edge.weight.abs();
    }
    for edge in &b.edges[bi..] {
        dist += edge.weight.abs();
    }

    dist
}

pub(crate) fn validate_seed_genome_config(config: &SeedGenomeConfig) -> Result<(), SimError> {
    if !(0.0..=1.0).contains(&config.mutation_rate) {
        return Err(SimError::InvalidConfig(
            "mutation_rate must be within [0, 1]".to_owned(),
        ));
    }
    if config.max_num_neurons < config.num_neurons {
        return Err(SimError::InvalidConfig(
            "max_num_neurons must be >= num_neurons".to_owned(),
        ));
    }
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
