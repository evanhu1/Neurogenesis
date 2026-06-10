use super::spatial_prior::add_synapse_genes_with_spatial_prior;
use super::*;

pub(crate) fn align_genome_vectors<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    genome.topology.num_neurons = genome.topology.num_neurons.min(MAX_INTER_NEURONS);
    genome.topology.num_synapses = genome
        .topology
        .num_synapses
        .min(max_possible_synapses(genome.topology.num_neurons));
    genome.topology.spatial_prior_sigma = genome.topology.spatial_prior_sigma.max(0.01);

    let target_inter_len = genome.topology.num_neurons as usize;
    align_vec_to(&mut genome.brain.inter_biases, target_inter_len, || {
        sample_initial_bias(rng)
    });
    align_vec_to(
        &mut genome.brain.inter_log_time_constants,
        target_inter_len,
        || sample_initial_log_time_constant(rng),
    );
    align_vec_to(&mut genome.brain.inter_locations, target_inter_len, || {
        sample_uniform_location(rng)
    });
    align_vec_to(
        &mut genome.brain.sensory_locations,
        SENSORY_COUNT as usize,
        || sample_uniform_location(rng),
    );
    align_vec_to(&mut genome.brain.action_locations, ACTION_COUNT, || {
        sample_uniform_location(rng)
    });
    align_vec_to(&mut genome.brain.action_biases, ACTION_COUNT, || {
        sample_initial_bias(rng)
    });
    align_reward_weights(&mut genome.reward_weights);

    // Padding fixes lengths but leaves existing entries untouched; a NaN
    // bias/log-tau/location from malformed intake would flow through
    // `express_genome` and poison activations exactly like a NaN synapse
    // weight (see `sanitize_synapse_genes`), and `perturb_clamped` propagates
    // NaN, so it would persist heritably. Resample non-finite entries;
    // well-formed genomes draw no RNG, preserving bit-for-bit determinism.
    for bias in genome
        .brain
        .inter_biases
        .iter_mut()
        .chain(genome.brain.action_biases.iter_mut())
    {
        if !bias.is_finite() {
            *bias = sample_initial_bias(rng);
        }
    }
    for log_tau in genome.brain.inter_log_time_constants.iter_mut() {
        if !log_tau.is_finite() {
            *log_tau = sample_initial_log_time_constant(rng);
        }
    }
    for location in genome
        .brain
        .sensory_locations
        .iter_mut()
        .chain(genome.brain.inter_locations.iter_mut())
        .chain(genome.brain.action_locations.iter_mut())
    {
        if !(location.x.is_finite() && location.y.is_finite()) {
            *location = sample_uniform_location(rng);
        }
    }

    sanitize_synapse_genes(genome);

    // `sanitize_synapse_genes` may have dropped/deduped malformed edges;
    // reconcile the count so the first `mutate_genome` on a malformed
    // champion doesn't treat the stale target as a request to grow brand-new
    // random synapses. No-op for well-formed genomes.
    genome.topology.num_synapses = genome.brain.edges.len() as u32;
}

/// Debug-only check that a genome entering `mutate_genome` is already
/// well-formed. Every caller passes genomes produced by
/// `generate_seed_genome`, a previous `mutate_genome` pass, or champion-pool
/// intake (`align_genome_vectors` in `reset_with_champion_pool`), all of
/// which end in a full sanitize, so a release-mode entry pass would be pure
/// re-verification; the single authoritative sanitize runs at `mutate_genome`
/// exit via `reconcile_synapse_count`.
pub(super) fn debug_assert_genome_well_formed(genome: &OrganismGenome) {
    debug_assert!(genome.topology.num_neurons <= MAX_INTER_NEURONS);
    debug_assert!(
        genome.topology.num_synapses <= max_possible_synapses(genome.topology.num_neurons)
    );
    debug_assert!(genome.topology.spatial_prior_sigma >= 0.01);
    let target_inter_len = genome.topology.num_neurons as usize;
    debug_assert_eq!(genome.brain.inter_biases.len(), target_inter_len);
    debug_assert_eq!(
        genome.brain.inter_log_time_constants.len(),
        target_inter_len
    );
    debug_assert_eq!(genome.brain.inter_locations.len(), target_inter_len);
    debug_assert_eq!(genome.brain.sensory_locations.len(), SENSORY_COUNT as usize);
    debug_assert_eq!(genome.brain.action_locations.len(), ACTION_COUNT);
    debug_assert_eq!(genome.brain.action_biases.len(), ACTION_COUNT);
    debug_assert_eq!(genome.reward_weights.len(), crate::REWARD_WEIGHT_COUNT);
    debug_assert_eq!(
        genome.brain.edges.len() as u32,
        genome.topology.num_synapses
    );
    debug_assert_synapse_genes_well_formed(genome);
}

/// Debug-only verification that the synapse genes uphold the invariants every
/// mutation operator maintains: sorted-unique (pre, post) keys, valid endpoint
/// pairs, and constrained finite weights.
fn debug_assert_synapse_genes_well_formed(genome: &OrganismGenome) {
    debug_assert!(genome.brain.edges.windows(2).all(|w| {
        (w[0].pre_neuron_id, w[0].post_neuron_id) < (w[1].pre_neuron_id, w[1].post_neuron_id)
    }));
    debug_assert!(genome.brain.edges.iter().all(|edge| {
        is_valid_synapse_pair(
            edge.pre_neuron_id,
            edge.post_neuron_id,
            genome.topology.num_neurons,
        ) && edge.weight == constrain_weight(edge.weight)
    }));
}

fn align_reward_weights(weights: &mut Vec<f32>) {
    use crate::{
        DEFAULT_REWARD_WEIGHTS, REWARD_WEIGHT_COUNT, REWARD_WEIGHT_MAX, REWARD_WEIGHT_MIN,
    };
    while weights.len() < REWARD_WEIGHT_COUNT {
        weights.push(DEFAULT_REWARD_WEIGHTS[weights.len()]);
    }
    weights.truncate(REWARD_WEIGHT_COUNT);
    for (w, default) in weights.iter_mut().zip(DEFAULT_REWARD_WEIGHTS) {
        // `f32::clamp` propagates NaN, so a non-finite weight from malformed
        // intake would scale dopamine with NaN forever; reset it to the
        // default for its channel instead.
        if w.is_finite() {
            *w = w.clamp(REWARD_WEIGHT_MIN, REWARD_WEIGHT_MAX);
        } else {
            *w = default;
        }
    }
}

pub(super) fn reconcile_synapse_count<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    // Every mutation operator maintains sorted-unique (pre, post) keys
    // (sorted-position inserts in topology.rs, the strictly monotone ID remap
    // in `mutate_remove_neuron`), pair validity (candidates are enumerated
    // from the valid ranges only), and constrained finite weights
    // (`constrain_weight` / clamped weight sampling), and disk intake runs
    // the full `sanitize_synapse_genes` in `align_genome_vectors` — so an
    // unconditional exit sanitize is pure re-verification. Keep it
    // debug-only.
    debug_assert_synapse_genes_well_formed(genome);

    let target = genome.topology.num_synapses as usize;
    if genome.brain.edges.len() < target {
        add_synapse_genes_with_spatial_prior(genome, target - genome.brain.edges.len(), rng);
        sort_synapse_genes(&mut genome.brain.edges);
    }

    genome.topology.num_synapses = genome.brain.edges.len() as u32;
}

pub(super) fn sanitize_synapse_genes(genome: &mut OrganismGenome) {
    let num_neurons = genome.topology.num_neurons;
    // Non-finite weights must be dropped here: `constrain_weight` propagates
    // NaN (NaN != 0.0, NaN.signum() is NaN, clamp keeps NaN), so a NaN weight
    // from malformed intake (e.g. champion_pool.json) would otherwise be
    // copied verbatim into the runtime brain and poison all activations.
    genome.brain.edges.retain(|edge| {
        edge.weight.is_finite()
            && is_valid_synapse_pair(edge.pre_neuron_id, edge.post_neuron_id, num_neurons)
    });

    for edge in &mut genome.brain.edges {
        edge.weight = constrain_weight(edge.weight);
    }

    sort_synapse_genes(&mut genome.brain.edges);
    genome.brain.edges.dedup_by(|a, b| {
        a.pre_neuron_id == b.pre_neuron_id && a.post_neuron_id == b.post_neuron_id
    });
}

pub(super) fn is_valid_synapse_pair(pre: NeuronId, post: NeuronId, num_neurons: u32) -> bool {
    if !is_valid_pre_id(pre, num_neurons) || !is_valid_post_id(post, num_neurons) {
        return false;
    }

    // Inter-neuron self-connections are permitted: they act as gated memory
    // cells because inter→inter edges read previous-tick activations in
    // `evaluate_brain`, so a self-edge lets a neuron retain/gate its own
    // state across ticks (useful for reversal learning / adaptation).
    true
}

fn is_valid_pre_id(id: NeuronId, num_neurons: u32) -> bool {
    is_sensory_id(id) || is_inter_id(id, num_neurons)
}

fn is_valid_post_id(id: NeuronId, num_neurons: u32) -> bool {
    is_inter_id(id, num_neurons) || is_action_id(id)
}

pub(super) fn sort_synapse_genes(edges: &mut [SynapseGene]) {
    edges
        .sort_unstable_by(|a, b| synapse_key_cmp(a, b).then_with(|| a.weight.total_cmp(&b.weight)));
}

fn synapse_key_cmp(a: &SynapseGene, b: &SynapseGene) -> Ordering {
    a.pre_neuron_id
        .cmp(&b.pre_neuron_id)
        .then_with(|| a.post_neuron_id.cmp(&b.post_neuron_id))
}
