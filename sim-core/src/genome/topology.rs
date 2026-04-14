use super::sanitization::{reconcile_synapse_count, sanitize_synapse_genes, sort_synapse_genes};
use super::*;

pub(crate) fn mutate_add_synapse<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let max_synapses = max_possible_synapses(genome.topology.num_neurons) as usize;
    if genome.brain.edges.len() >= max_synapses {
        genome.topology.num_synapses = genome.brain.edges.len() as u32;
        return;
    }

    let before_len = genome.brain.edges.len();
    super::spatial_prior::add_synapse_genes_with_spatial_prior(genome, 1, rng);
    if genome.brain.edges.len() > before_len {
        genome.topology.num_synapses = genome.topology.num_synapses.saturating_add(1);
        sort_synapse_genes(&mut genome.brain.edges);
    }
}

pub(crate) fn mutate_remove_synapse<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    if genome.brain.edges.is_empty() {
        genome.topology.num_synapses = 0;
        return;
    }

    let idx = rng.random_range(0..genome.brain.edges.len());
    genome.brain.edges.swap_remove(idx);
    genome.topology.num_synapses = genome.topology.num_synapses.saturating_sub(1);
    sort_synapse_genes(&mut genome.brain.edges);
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
    genome.brain.inter_locations.remove(removed_idx);

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

    sanitize_synapse_genes(genome);
    genome.topology.num_synapses = genome.brain.edges.len() as u32;
}

pub(crate) fn mutate_add_neuron_split_edge<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    if genome.brain.edges.is_empty()
        || genome.topology.num_neurons >= u32::MAX.saturating_sub(INTER_ID_BASE)
    {
        return;
    }

    ensure_neuron_vectors_sized(genome);

    let selected_idx = select_weighted_edge_index(&genome.brain.edges, rng);
    let selected_edge = genome.brain.edges.swap_remove(selected_idx);
    let new_inter_id = inter_neuron_id(genome.topology.num_neurons);

    let (new_bias, new_log_tau, new_location) =
        derive_split_neuron_params(&selected_edge, genome, rng);

    genome.topology.num_neurons = genome.topology.num_neurons.saturating_add(1);
    genome.brain.inter_biases.push(new_bias);
    genome.brain.inter_log_time_constants.push(new_log_tau);
    genome.brain.inter_locations.push(new_location);

    insert_split_edges(genome, &selected_edge, new_inter_id);

    genome.topology.num_synapses = genome.topology.num_synapses.saturating_add(1);
    reconcile_synapse_count(genome, rng);
}

fn ensure_neuron_vectors_sized(genome: &mut OrganismGenome) {
    let target_inter_len = genome.topology.num_neurons as usize;
    genome.brain.inter_biases.resize(target_inter_len, 0.0);
    genome
        .brain
        .inter_log_time_constants
        .resize(target_inter_len, DEFAULT_INTER_LOG_TIME_CONSTANT);
    genome
        .brain
        .inter_locations
        .resize(target_inter_len, DEFAULT_BRAIN_LOCATION);
    genome
        .brain
        .sensory_locations
        .resize(SENSORY_COUNT as usize, DEFAULT_BRAIN_LOCATION);
    genome
        .brain
        .action_locations
        .resize(ACTION_COUNT, DEFAULT_BRAIN_LOCATION);
}

fn derive_split_neuron_params<R: Rng + ?Sized>(
    edge: &SynapseEdge,
    genome: &OrganismGenome,
    rng: &mut R,
) -> (f32, f32, BrainLocation) {
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

    let pre_loc = location_for_neuron(edge.pre_neuron_id, genome);
    let post_loc = location_for_neuron(edge.post_neuron_id, genome);
    let midpoint = BrainLocation {
        x: 0.5 * (pre_loc.x + post_loc.x),
        y: 0.5 * (pre_loc.y + post_loc.y),
    };
    let new_location = BrainLocation {
        x: perturb_clamped(
            midpoint.x,
            LOCATION_PERTURBATION_STDDEV * NEW_NEURON_PERTURBATION_SCALE,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        ),
        y: perturb_clamped(
            midpoint.y,
            LOCATION_PERTURBATION_STDDEV * NEW_NEURON_PERTURBATION_SCALE,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        ),
    };

    (new_bias, new_log_tau, new_location)
}

fn insert_split_edges(
    genome: &mut OrganismGenome,
    original_edge: &SynapseEdge,
    new_inter_id: NeuronId,
) {
    genome.brain.edges.push(SynapseEdge {
        pre_neuron_id: original_edge.pre_neuron_id,
        post_neuron_id: new_inter_id,
        weight: 1.0,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
    genome.brain.edges.push(SynapseEdge {
        pre_neuron_id: new_inter_id,
        post_neuron_id: original_edge.post_neuron_id,
        weight: constrain_weight(original_edge.weight),
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
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

pub(super) fn inter_log_time_constant_for_neuron(
    neuron_id: NeuronId,
    genome: &OrganismGenome,
) -> Option<f32> {
    if !is_inter_id(neuron_id, genome.topology.num_neurons) {
        return None;
    }
    let inter_idx = inter_index(neuron_id, genome.topology.num_neurons as usize)?;
    genome.brain.inter_log_time_constants.get(inter_idx).copied()
}

pub(super) fn location_for_neuron(neuron_id: NeuronId, genome: &OrganismGenome) -> BrainLocation {
    if is_sensory_id(neuron_id) {
        return genome
            .brain
            .sensory_locations
            .get(neuron_id.0 as usize)
            .copied()
            .unwrap_or(DEFAULT_BRAIN_LOCATION);
    }
    if is_inter_id(neuron_id, genome.topology.num_neurons) {
        return inter_index(neuron_id, genome.topology.num_neurons as usize)
            .and_then(|idx| genome.brain.inter_locations.get(idx).copied())
            .unwrap_or(DEFAULT_BRAIN_LOCATION);
    }
    if is_action_id(neuron_id) {
        return action_array_index(neuron_id)
            .and_then(|idx| genome.brain.action_locations.get(idx).copied())
            .unwrap_or(DEFAULT_BRAIN_LOCATION);
    }
    DEFAULT_BRAIN_LOCATION
}
