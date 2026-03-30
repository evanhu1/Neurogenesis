use super::spatial_prior::add_synapse_genes_with_spatial_prior;
use super::*;

pub(super) fn align_genome_vectors<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    genome.num_synapses = genome
        .num_synapses
        .min(max_possible_synapses(genome.num_neurons));
    genome.spatial_prior_sigma = genome.spatial_prior_sigma.max(0.01);

    let target_inter_len = genome.num_neurons as usize;
    align_vec_to(&mut genome.inter_biases, target_inter_len, || {
        sample_initial_bias(rng)
    });
    align_vec_to(
        &mut genome.inter_log_time_constants,
        target_inter_len,
        || sample_initial_log_time_constant(rng),
    );
    align_vec_to(&mut genome.inter_locations, target_inter_len, || {
        sample_uniform_location(rng)
    });
    align_vec_to(
        &mut genome.sensory_locations,
        SENSORY_COUNT as usize,
        || sample_uniform_location(rng),
    );
    align_vec_to(&mut genome.action_locations, ACTION_COUNT, || {
        sample_uniform_location(rng)
    });

    sanitize_synapse_genes(genome);
}

pub(super) fn sync_synapse_genes_to_target<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    sanitize_synapse_genes(genome);

    let target = genome.num_synapses as usize;
    if genome.edges.len() < target {
        add_synapse_genes_with_spatial_prior(genome, target - genome.edges.len(), rng);
    }

    sort_synapse_genes(&mut genome.edges);
    genome.num_synapses = genome.edges.len() as u32;
}

pub(super) fn sanitize_synapse_genes(genome: &mut OrganismGenome) {
    let num_neurons = genome.num_neurons;
    genome
        .edges
        .retain(|edge| is_valid_synapse_pair(edge.pre_neuron_id, edge.post_neuron_id, num_neurons));

    for edge in &mut genome.edges {
        edge.weight = constrain_weight(edge.weight);
        edge.eligibility = 0.0;
        edge.pending_coactivation = 0.0;
    }

    sort_synapse_genes(&mut genome.edges);
    genome.edges.dedup_by(|a, b| {
        a.pre_neuron_id == b.pre_neuron_id && a.post_neuron_id == b.post_neuron_id
    });
}

pub(super) fn is_valid_synapse_pair(pre: NeuronId, post: NeuronId, num_neurons: u32) -> bool {
    if !is_valid_pre_id(pre, num_neurons) || !is_valid_post_id(post, num_neurons) {
        return false;
    }

    if is_inter_id(pre, num_neurons) && is_inter_id(post, num_neurons) && pre == post {
        return false;
    }

    true
}

fn is_valid_pre_id(id: NeuronId, num_neurons: u32) -> bool {
    is_sensory_id(id) || is_inter_id(id, num_neurons)
}

fn is_valid_post_id(id: NeuronId, num_neurons: u32) -> bool {
    is_inter_id(id, num_neurons) || is_action_id(id)
}

pub(super) fn sort_synapse_genes(edges: &mut [SynapseEdge]) {
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
