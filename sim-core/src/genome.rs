use crate::SimError;
use rand::Rng;
use sim_protocol::{
    ActionType, GenomeEdge, NeuronId, OrganismGenome, SeedGenomeConfig, SensoryReceptor, SpeciesId,
};
use std::collections::{BTreeMap, HashMap};

const MAX_MUTATED_INTER_NEURONS: u32 = 256;
const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const SYNAPSE_STRENGTH_MAX: f32 = 3.0;
const BIAS_MAX: f32 = 1.0;
const WEIGHT_PERTURBATION_STDDEV: f32 = 0.3;
const MUTATION_RATE_STEP: f32 = 0.01;

const SENSORY_COUNT: u32 = SensoryReceptor::LOOK_NEURON_COUNT + 1;
const ACTION_COUNT: u32 = ActionType::ALL.len() as u32;
const INTER_ID_BASE: u32 = 1000;
const ACTION_ID_BASE: u32 = 2000;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let inter_biases: Vec<f32> = (0..config.max_num_neurons)
        .map(|_| rng.random_range(-BIAS_MAX..BIAS_MAX))
        .collect();

    let mut edges = Vec::new();
    for _ in 0..config.num_synapses {
        if let Some(edge) = random_edge(config.num_neurons, &edges, rng) {
            edges.push(edge);
        }
    }

    OrganismGenome {
        num_neurons: config.num_neurons,
        max_num_neurons: config.max_num_neurons,
        vision_distance: config.vision_distance,
        mutation_rate: config.mutation_rate,
        inter_biases,
        edges,
    }
}

fn random_edge<R: Rng + ?Sized>(
    num_neurons: u32,
    existing: &[GenomeEdge],
    rng: &mut R,
) -> Option<GenomeEdge> {
    // Pre neurons: sensory (0..SENSORY_COUNT) + enabled inter (INTER_ID_BASE..INTER_ID_BASE+num_neurons)
    // Post neurons: enabled inter (INTER_ID_BASE..INTER_ID_BASE+num_neurons) + action (ACTION_ID_BASE..ACTION_ID_BASE+ACTION_COUNT)
    let pre_count = SENSORY_COUNT + num_neurons;
    let post_count = num_neurons + ACTION_COUNT;

    if pre_count == 0 || post_count == 0 {
        return None;
    }

    // Try up to 20 times to find a non-duplicate, non-self edge
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

        if pre == post {
            continue;
        }

        let is_dup = existing.iter().any(|e| e.pre == pre && e.post == post);
        if is_dup {
            continue;
        }

        let weight = rng.random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
        return Some(GenomeEdge { pre, post, weight });
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
            .retain(|e| e.pre != disabled_id && e.post != disabled_id);
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

/// Sample from a normal-ish distribution using Box-Muller, then clamp.
fn perturb<R: Rng + ?Sized>(value: f32, stddev: f32, clamp_abs: f32, rng: &mut R) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>();
    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    (value + normal * stddev).clamp(-clamp_abs, clamp_abs)
}

pub(crate) fn genome_distance(a: &OrganismGenome, b: &OrganismGenome) -> f32 {
    let mut dist = 0.0f32;

    dist += (a.num_neurons as f32 - b.num_neurons as f32).abs();
    dist += (a.max_num_neurons as f32 - b.max_num_neurons as f32).abs();
    dist += (a.vision_distance as f32 - b.vision_distance as f32).abs();
    dist += (a.mutation_rate - b.mutation_rate).abs();

    // Bias distance
    let max_bias_len = a.inter_biases.len().max(b.inter_biases.len());
    for i in 0..max_bias_len {
        let ba = a.inter_biases.get(i).copied().unwrap_or(0.0);
        let bb = b.inter_biases.get(i).copied().unwrap_or(0.0);
        dist += (ba - bb).abs();
    }

    // Edge distance: build HashMap<(pre,post), weight> for each
    let map_a: HashMap<(NeuronId, NeuronId), f32> = a
        .edges
        .iter()
        .map(|e| ((e.pre, e.post), e.weight))
        .collect();
    let map_b: HashMap<(NeuronId, NeuronId), f32> = b
        .edges
        .iter()
        .map(|e| ((e.pre, e.post), e.weight))
        .collect();

    for (key, &wa) in &map_a {
        match map_b.get(key) {
            Some(&wb) => dist += (wa - wb).abs(),
            None => dist += wa.abs(),
        }
    }
    for (key, &wb) in &map_b {
        if !map_a.contains_key(key) {
            dist += wb.abs();
        }
    }

    dist
}

pub(crate) fn assign_species(
    child_genome: &OrganismGenome,
    registry: &BTreeMap<SpeciesId, OrganismGenome>,
    threshold: f32,
) -> Option<SpeciesId> {
    let mut best_id = None;
    let mut best_dist = f32::MAX;

    for (&species_id, founder_genome) in registry {
        let dist = genome_distance(child_genome, founder_genome);
        if dist < best_dist {
            best_dist = dist;
            best_id = Some(species_id);
        }
    }

    if best_dist <= threshold {
        best_id
    } else {
        None
    }
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
