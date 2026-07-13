use super::*;

/// Canonicalize an externally supplied genome before it can be expressed or
/// mutated. Stable identities make topology lengths derived facts, so this
/// pass never invents connections to satisfy a stale count gene.
pub(crate) fn align_genome_vectors<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    for node in &mut genome.brain.hidden_nodes {
        if !node.bias.is_finite() {
            node.bias = sample_initial_bias(rng);
        }
        if !node.log_time_constant.is_finite() {
            node.log_time_constant = sample_initial_log_time_constant(rng);
        }
        node.bias = node.bias.clamp(-BIAS_MAX, BIAS_MAX);
        node.log_time_constant = node
            .log_time_constant
            .clamp(INTER_LOG_TIME_CONSTANT_MIN, INTER_LOG_TIME_CONSTANT_MAX);
    }

    // Total ordering across malformed duplicates makes the retained allele
    // deterministic even when input vectors arrive in arbitrary order.
    genome.brain.hidden_nodes.sort_unstable_by(|a, b| {
        a.id.cmp(&b.id)
            .then_with(|| a.bias.total_cmp(&b.bias))
            .then_with(|| a.log_time_constant.total_cmp(&b.log_time_constant))
    });
    genome
        .brain
        .hidden_nodes
        .retain(|node| is_hidden_gene_node_id(node.id));
    genome.brain.hidden_nodes.dedup_by_key(|node| node.id);
    genome
        .brain
        .hidden_nodes
        .truncate(MAX_INTER_NEURONS as usize);

    align_vec_to(&mut genome.brain.action_biases, ACTION_COUNT, || {
        sample_initial_bias(rng)
    });
    for bias in &mut genome.brain.action_biases {
        if !bias.is_finite() {
            *bias = sample_initial_bias(rng);
        }
        *bias = bias.clamp(-BIAS_MAX, BIAS_MAX);
    }

    sanitize_synapse_genes(genome);
}

/// Debug-only check for genomes produced internally by seed generation and
/// mutation. Every vector must already be canonical before crossover can use
/// merge walks over stable identities.
pub(super) fn debug_assert_genome_well_formed(genome: &OrganismGenome) {
    debug_assert!(genome.brain.hidden_nodes.len() <= MAX_INTER_NEURONS as usize);
    debug_assert_eq!(genome.brain.action_biases.len(), ACTION_COUNT);
    debug_assert!(genome
        .brain
        .hidden_nodes
        .windows(2)
        .all(|w| w[0].id < w[1].id));
    debug_assert!(genome.brain.hidden_nodes.iter().all(|node| {
        is_hidden_gene_node_id(node.id)
            && node.bias.is_finite()
            && node.bias == node.bias.clamp(-BIAS_MAX, BIAS_MAX)
            && node.log_time_constant.is_finite()
            && node.log_time_constant
                == node
                    .log_time_constant
                    .clamp(INTER_LOG_TIME_CONSTANT_MIN, INTER_LOG_TIME_CONSTANT_MAX)
    }));
    debug_assert_synapse_genes_well_formed(genome);
}

fn debug_assert_synapse_genes_well_formed(genome: &OrganismGenome) {
    debug_assert!(genome
        .brain
        .edges
        .windows(2)
        .all(|w| w[0].innovation < w[1].innovation));
    debug_assert!(genome.brain.edges.iter().all(|edge| {
        is_valid_synapse_pair(genome, edge.pre_node_id, edge.post_node_id)
            && edge.weight.is_finite()
            && edge.weight == constrain_weight(edge.weight)
    }));
    debug_assert!(genome.brain.edges.iter().enumerate().all(|(index, edge)| {
        !genome.brain.edges[..index].iter().any(|previous| {
            (previous.pre_node_id, previous.post_node_id) == (edge.pre_node_id, edge.post_node_id)
        })
    }));
}

pub(super) fn sanitize_synapse_genes(genome: &mut OrganismGenome) {
    let hidden_nodes = &genome.brain.hidden_nodes;
    genome.brain.edges.retain_mut(|edge| {
        if !edge.weight.is_finite()
            || !is_valid_synapse_pair_for_nodes(hidden_nodes, edge.pre_node_id, edge.post_node_id)
        {
            return false;
        }
        edge.weight = constrain_weight(edge.weight);
        true
    });

    // Historical markings belong to the evolutionary run and are not
    // reconstructible from endpoints. Preserve valid supplied innovations,
    // while choosing the oldest deterministic representative if malformed
    // input assigns multiple markings to the same structural connection.
    deduplicate_endpoint_pairs(&mut genome.brain.edges);
    sort_synapse_genes(&mut genome.brain.edges);
    reject_colliding_innovation_groups(&mut genome.brain.edges);
}

fn deduplicate_endpoint_pairs(edges: &mut Vec<SynapseGene>) {
    edges.sort_unstable_by(|a, b| {
        a.pre_node_id
            .cmp(&b.pre_node_id)
            .then_with(|| a.post_node_id.cmp(&b.post_node_id))
            .then_with(|| a.innovation.cmp(&b.innovation))
            .then_with(|| b.enabled.cmp(&a.enabled))
            .then_with(|| a.weight.total_cmp(&b.weight))
    });
    edges.dedup_by_key(|edge| (edge.pre_node_id, edge.post_node_id));
}

/// Keep one deterministic representative for exact duplicate markings, but
/// drop the entire innovation group if the same marking names different
/// endpoint pairs. Retaining either side of a true collision would alias two
/// unrelated historical loci in every later crossover merge.
pub(super) fn reject_colliding_innovation_groups(edges: &mut Vec<SynapseGene>) {
    let mut read = 0;
    let mut write = 0;
    while read < edges.len() {
        let innovation = edges[read].innovation;
        let first_endpoints = (edges[read].pre_node_id, edges[read].post_node_id);
        let mut end = read + 1;
        let mut collision = false;
        while end < edges.len() && edges[end].innovation == innovation {
            collision |= (edges[end].pre_node_id, edges[end].post_node_id) != first_endpoints;
            end += 1;
        }

        if !collision {
            // sort_synapse_genes orders exact duplicates enabled-first and then
            // by total weight, so the first representative is deterministic.
            edges[write] = edges[read];
            write += 1;
        }
        read = end;
    }
    edges.truncate(write);
}

pub(super) fn is_valid_synapse_pair(
    genome: &OrganismGenome,
    pre: GeneNodeId,
    post: GeneNodeId,
) -> bool {
    is_valid_synapse_pair_for_nodes(&genome.brain.hidden_nodes, pre, post)
}

pub(super) fn is_valid_synapse_pair_for_nodes(
    hidden_nodes: &[HiddenNodeGene],
    pre: GeneNodeId,
    post: GeneNodeId,
) -> bool {
    is_valid_pre_id(hidden_nodes, pre) && is_valid_post_id(hidden_nodes, post)
}

fn is_valid_pre_id(hidden_nodes: &[HiddenNodeGene], id: GeneNodeId) -> bool {
    sensory_gene_node_index(id).is_some_and(|idx| idx < SENSORY_COUNT)
        || hidden_nodes
            .binary_search_by_key(&id, |node| node.id)
            .is_ok()
}

fn is_valid_post_id(hidden_nodes: &[HiddenNodeGene], id: GeneNodeId) -> bool {
    hidden_nodes
        .binary_search_by_key(&id, |node| node.id)
        .is_ok()
        || action_gene_node_index(id).is_some_and(|idx| idx < ACTION_COUNT)
}

pub(super) fn sort_synapse_genes(edges: &mut [SynapseGene]) {
    edges.sort_unstable_by(|a, b| {
        a.innovation
            .cmp(&b.innovation)
            .then_with(|| a.pre_node_id.cmp(&b.pre_node_id))
            .then_with(|| a.post_node_id.cmp(&b.post_node_id))
            // Prefer an enabled representative when malformed input contains
            // duplicate genes for one innovation.
            .then_with(|| b.enabled.cmp(&a.enabled))
            .then_with(|| a.weight.total_cmp(&b.weight))
    });
}
