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

    // Both call sites (mutate_add_synapse, reconcile_synapse_count) maintain
    // the edge list sorted by (pre, post) with unique keys, so existing-pair
    // membership is a binary search instead of building a HashSet.
    debug_assert!(genome.brain.edges.windows(2).all(|w| {
        (w[0].pre_neuron_id, w[0].post_neuron_id) < (w[1].pre_neuron_id, w[1].post_neuron_id)
    }));

    if add_count == 1 {
        // Common case (one add-synapse mutation per offspring): single-pass
        // minimum selection, no candidate Vec and no sort.
        let mut best: Option<(f32, NeuronId, NeuronId)> = None;
        for_each_candidate(genome, rng, |candidate| {
            let replace = match best {
                None => true,
                Some(current) => candidate_cmp(&candidate, &current) == Ordering::Less,
            };
            if replace {
                best = Some(candidate);
            }
        });
        if let Some((_, pre_id, post_id)) = best {
            genome.brain.edges.push(SynapseGene {
                pre_neuron_id: pre_id,
                post_neuron_id: post_id,
                weight: sample_synapse_weight(INITIAL_SYNAPSE_EXCITATORY_PROBABILITY, rng),
            });
        }
        return;
    }

    let mut weighted_candidates = Vec::new();
    for_each_candidate(genome, rng, |candidate| weighted_candidates.push(candidate));

    // Select the `add_count` smallest candidates, then sort only that slice so
    // the selected set and insertion order match a full sort exactly (the
    // comparator is a total order over unique (pre, post) pairs).
    let selected: &mut [(f32, NeuronId, NeuronId)] = if add_count < weighted_candidates.len() {
        let (top, _, _) = weighted_candidates.select_nth_unstable_by(add_count, candidate_cmp);
        top
    } else {
        &mut weighted_candidates
    };
    selected.sort_unstable_by(candidate_cmp);

    for &mut (_, pre_id, post_id) in selected {
        genome.brain.edges.push(SynapseGene {
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            weight: sample_synapse_weight(INITIAL_SYNAPSE_EXCITATORY_PROBABILITY, rng),
        });
    }
}

fn candidate_cmp(a: &(f32, NeuronId, NeuronId), b: &(f32, NeuronId, NeuronId)) -> Ordering {
    a.0.total_cmp(&b.0)
        .then_with(|| a.1.cmp(&b.1))
        .then_with(|| a.2.cmp(&b.2))
}

/// Enumerates every valid, not-yet-connected (pre, post) pair, drawing one
/// weighted-without-replacement priority per candidate (same RNG sequence as
/// the historical full-sort implementation).
fn for_each_candidate<R: Rng + ?Sized>(
    genome: &OrganismGenome,
    rng: &mut R,
    mut visit: impl FnMut((f32, NeuronId, NeuronId)),
) {
    let num_neurons = genome.topology.num_neurons;
    let all_presynaptic = (0..SENSORY_COUNT)
        .map(NeuronId)
        .chain((0..num_neurons).map(inter_neuron_id));

    let sigma = genome.topology.spatial_prior_sigma.max(0.01);
    let sigma_sq = sigma * sigma;

    // Candidates are enumerated in strictly ascending (pre, post) order and the
    // edge list is sorted by the same key, so existing-pair membership is a
    // single forward merge cursor instead of a binary search per pair.
    let edges = &genome.brain.edges;
    let mut cursor = 0usize;

    for pre_id in all_presynaptic {
        let pre_location = location_for_neuron(pre_id, genome);
        for post_id in post_ids(num_neurons) {
            // By construction every enumerated pair is valid: pre IDs are
            // exactly the sensory + enabled-inter ranges, post IDs the
            // enabled-inter + action ranges, and inter self-edges are
            // permitted (see `is_valid_synapse_pair`), so no release-mode
            // filtering is needed in this O((S+N)·(N+A)) loop.
            debug_assert!(is_valid_synapse_pair(pre_id, post_id, num_neurons));
            while cursor < edges.len()
                && (edges[cursor].pre_neuron_id, edges[cursor].post_neuron_id) < (pre_id, post_id)
            {
                cursor += 1;
            }
            if cursor < edges.len()
                && (edges[cursor].pre_neuron_id, edges[cursor].post_neuron_id) == (pre_id, post_id)
            {
                cursor += 1;
                continue;
            }
            let probability = connection_probability(genome, pre_location, post_id, sigma_sq);
            let priority = weighted_without_replacement_priority(probability, rng);
            visit((priority, pre_id, post_id));
        }
    }
}

pub(super) fn sample_lognormal_weight<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    sample_synapse_weight(0.5, rng)
}

fn post_ids(num_neurons: u32) -> impl Iterator<Item = NeuronId> {
    let inter = (0..num_neurons).map(inter_neuron_id);
    let actions = (0..ACTION_COUNT).map(action_neuron_id);
    inter.chain(actions)
}

fn connection_probability(
    genome: &OrganismGenome,
    pre_location: BrainLocation,
    post: NeuronId,
    sigma_sq: f32,
) -> f32 {
    let post_location = location_for_neuron(post, genome);
    let distance_sq = distance_sq_between_locations(pre_location, post_location);
    let local_bias = (-0.5 * distance_sq / sigma_sq).exp();
    (SPATIAL_PRIOR_LONG_RANGE_FLOOR + (1.0 - SPATIAL_PRIOR_LONG_RANGE_FLOOR) * local_bias)
        .clamp(0.0, 1.0)
}

fn weighted_without_replacement_priority<R: Rng + ?Sized>(weight: f32, rng: &mut R) -> f32 {
    let clamped_weight = weight.max(f32::MIN_POSITIVE);
    let u = rng.random::<f32>().max(f32::MIN_POSITIVE);
    -u.ln() / clamped_weight
}

fn sample_synapse_weight<R: Rng + ?Sized>(excitatory_probability: f32, rng: &mut R) -> f32 {
    let z = standard_normal(rng);
    let magnitude = (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    if rng.random::<f32>() < excitatory_probability {
        magnitude
    } else {
        -magnitude
    }
}
