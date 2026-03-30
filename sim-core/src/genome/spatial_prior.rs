use super::sanitization::is_valid_synapse_pair;
use super::topology::location_for_neuron;
use super::*;

pub(super) fn add_synapse_genes_with_spatial_prior<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    add_count: usize,
    rng: &mut R,
) {
    if add_count == 0 {
        return;
    }

    let mut existing_pairs = HashSet::with_capacity(genome.edges.len());
    for edge in &genome.edges {
        existing_pairs.insert((edge.pre_neuron_id.0, edge.post_neuron_id.0));
    }

    let mut weighted_candidates = Vec::new();
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
        let pre_id = inter_neuron_id(inter_idx);
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
        genome.edges.push(SynapseEdge {
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            weight: sample_initial_synapse_weight(rng),
            eligibility: 0.0,
            pending_coactivation: 0.0,
        });
    }
}

pub(super) fn sample_lognormal_weight<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let z = standard_normal(rng);
    let magnitude = (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    sample_signed_weight(magnitude, 0.5, rng)
}

fn post_ids(num_neurons: u32) -> impl Iterator<Item = NeuronId> {
    let inter = (0..num_neurons).map(inter_neuron_id);
    let actions = (0..ACTION_COUNT).map(action_neuron_id);
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

fn sample_initial_synapse_weight<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let z = standard_normal(rng);
    let magnitude = (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    sample_signed_weight(magnitude, INITIAL_SYNAPSE_EXCITATORY_PROBABILITY, rng)
}

fn sample_signed_weight<R: Rng + ?Sized>(
    magnitude: f32,
    excitatory_probability: f32,
    rng: &mut R,
) -> f32 {
    if rng.random::<f32>() < excitatory_probability {
        magnitude
    } else {
        -magnitude
    }
}
