use super::*;

pub(crate) fn mutate_add_synapse<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let max_synapses = max_possible_synapses(genome.topology.num_neurons) as usize;
    if genome.brain.edges.len() >= max_synapses {
        genome.topology.num_synapses = genome.brain.edges.len() as u32;
        return;
    }

    let before_len = genome.brain.edges.len();
    super::synapse_creation::add_synapse_genes(genome, 1, rng);
    if genome.brain.edges.len() > before_len {
        genome.topology.num_synapses = genome.topology.num_synapses.saturating_add(1);
        // The new edge was appended at the end; move it to its sorted
        // (pre, post) position instead of re-sorting the whole list.
        let edge = genome.brain.edges.pop().expect("edge was just appended");
        let pos = genome.brain.edges.partition_point(|e| {
            (e.pre_neuron_id, e.post_neuron_id) < (edge.pre_neuron_id, edge.post_neuron_id)
        });
        genome.brain.edges.insert(pos, edge);
    }
}

pub(crate) fn mutate_remove_synapse<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    if genome.brain.edges.is_empty() {
        genome.topology.num_synapses = 0;
        return;
    }

    let idx = rng.random_range(0..genome.brain.edges.len());
    genome.brain.edges.remove(idx);
    genome.topology.num_synapses = genome.topology.num_synapses.saturating_sub(1);
}

pub(crate) fn mutate_remove_neuron<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    if genome.topology.num_neurons == 0 {
        return;
    }

    let previous_num_neurons = genome.topology.num_neurons;
    let removed_inter_idx = rng.random_range(0..genome.topology.num_neurons);
    let removed_neuron_id = inter_neuron_id(removed_inter_idx);

    genome.topology.num_neurons = genome.topology.num_neurons.saturating_sub(1);

    let removed_idx = removed_inter_idx as usize;
    genome.brain.inter_biases.remove(removed_idx);
    genome.brain.inter_log_time_constants.remove(removed_idx);

    genome.brain.edges.retain_mut(|edge| {
        if edge.pre_neuron_id == removed_neuron_id || edge.post_neuron_id == removed_neuron_id {
            return false;
        }

        if edge.pre_neuron_id.0 > removed_neuron_id.0
            && is_inter_id(edge.pre_neuron_id, previous_num_neurons)
        {
            edge.pre_neuron_id.0 = edge.pre_neuron_id.0.saturating_sub(1);
        }
        if edge.post_neuron_id.0 > removed_neuron_id.0
            && is_inter_id(edge.post_neuron_id, previous_num_neurons)
        {
            edge.post_neuron_id.0 = edge.post_neuron_id.0.saturating_sub(1);
        }

        true
    });

    // The retain_mut above removes edges touching the dead neuron and applies
    // a strictly monotone ID remap to the survivors, so the edge list stays
    // sorted by (pre, post) with unique keys and every ID remains valid.
    debug_assert!(genome.brain.edges.windows(2).all(|w| {
        (w[0].pre_neuron_id, w[0].post_neuron_id) < (w[1].pre_neuron_id, w[1].post_neuron_id)
    }));
    genome.topology.num_synapses = genome.brain.edges.len() as u32;
}

pub(crate) fn mutate_add_neuron_split_edge<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    if genome.brain.edges.is_empty() || genome.topology.num_neurons >= MAX_INTER_NEURONS {
        return;
    }

    debug_assert_eq!(
        genome.brain.inter_biases.len(),
        genome.topology.num_neurons as usize
    );
    debug_assert_eq!(
        genome.brain.inter_log_time_constants.len(),
        genome.topology.num_neurons as usize
    );

    let selected_idx = select_weighted_edge_index(&genome.brain.edges, rng);
    // Ordered remove: `swap_remove` would break the sorted-by-(pre, post)
    // invariant that sibling operators binary-search/debug_assert against.
    let selected_edge = genome.brain.edges.remove(selected_idx);
    let new_inter_id = inter_neuron_id(genome.topology.num_neurons);

    let (new_bias, new_log_tau) = derive_split_neuron_params(&selected_edge, genome, rng);

    genome.topology.num_neurons = genome.topology.num_neurons.saturating_add(1);
    genome.brain.inter_biases.push(new_bias);
    genome.brain.inter_log_time_constants.push(new_log_tau);

    insert_split_edges(genome, &selected_edge, new_inter_id);

    // The caller (`mutate_genome`) unconditionally reconciles the synapse
    // count after all topology mutations, so no inner reconcile is needed.
    genome.topology.num_synapses = genome.topology.num_synapses.saturating_add(1);
}

fn derive_split_neuron_params<R: Rng + ?Sized>(
    edge: &SynapseGene,
    genome: &OrganismGenome,
    rng: &mut R,
) -> (f32, f32) {
    let pre_tau = inter_log_time_constant_for_neuron(edge.pre_neuron_id, genome);
    let post_tau = inter_log_time_constant_for_neuron(edge.post_neuron_id, genome);
    let base_tau = match (pre_tau, post_tau) {
        (Some(pre), Some(post)) => 0.5 * (pre + post),
        (Some(v), None) | (None, Some(v)) => v,
        (None, None) => DEFAULT_INTER_LOG_TIME_CONSTANT,
    };
    let new_log_tau = perturb_clamped(
        base_tau,
        INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV * NEW_NEURON_PERTURBATION_SCALE,
        INTER_LOG_TIME_CONSTANT_MIN,
        INTER_LOG_TIME_CONSTANT_MAX,
        rng,
    );
    let new_bias = perturb_clamped(
        0.0,
        BIAS_PERTURBATION_STDDEV * NEW_NEURON_PERTURBATION_SCALE,
        -BIAS_MAX,
        BIAS_MAX,
        rng,
    );

    (new_bias, new_log_tau)
}

fn insert_split_edges(
    genome: &mut OrganismGenome,
    original_edge: &SynapseGene,
    new_inter_id: NeuronId,
) {
    // `new_inter_id` is brand new, so neither replacement edge can collide
    // with an existing (pre, post) key; sorted-position inserts keep the
    // edge list's sorted-unique invariant intact mid-mutation-pass.
    insert_edge_sorted(
        &mut genome.brain.edges,
        SynapseGene {
            pre_neuron_id: original_edge.pre_neuron_id,
            post_neuron_id: new_inter_id,
            weight: 1.0,
        },
    );
    insert_edge_sorted(
        &mut genome.brain.edges,
        SynapseGene {
            pre_neuron_id: new_inter_id,
            post_neuron_id: original_edge.post_neuron_id,
            weight: constrain_weight(original_edge.weight),
        },
    );
}

fn insert_edge_sorted(edges: &mut Vec<SynapseGene>, edge: SynapseGene) {
    let pos = edges.partition_point(|e| {
        (e.pre_neuron_id, e.post_neuron_id) < (edge.pre_neuron_id, edge.post_neuron_id)
    });
    edges.insert(pos, edge);
}

fn select_weighted_edge_index<R: Rng + ?Sized>(edges: &[SynapseGene], rng: &mut R) -> usize {
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

pub(super) fn inter_log_time_constant_for_neuron(
    neuron_id: NeuronId,
    genome: &OrganismGenome,
) -> Option<f32> {
    if !is_inter_id(neuron_id, genome.topology.num_neurons) {
        return None;
    }
    let inter_idx = inter_index(neuron_id, genome.topology.num_neurons as usize)?;
    genome
        .brain
        .inter_log_time_constants
        .get(inter_idx)
        .copied()
}
