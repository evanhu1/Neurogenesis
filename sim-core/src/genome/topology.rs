use super::sanitization::{
    sanitize_synapse_genes, sort_synapse_genes, sync_synapse_genes_to_target,
};
use super::*;

pub(crate) fn mutate_add_synapse<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let max_synapses = max_possible_synapses(genome.num_neurons) as usize;
    if genome.edges.len() >= max_synapses {
        genome.num_synapses = genome.edges.len() as u32;
        return;
    }

    let before_len = genome.edges.len();
    super::spatial_prior::add_synapse_genes_with_spatial_prior(genome, 1, rng);
    if genome.edges.len() > before_len {
        genome.num_synapses = genome.num_synapses.saturating_add(1);
        sort_synapse_genes(&mut genome.edges);
    }
}

pub(crate) fn mutate_remove_synapse<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    if genome.edges.is_empty() {
        genome.num_synapses = 0;
        return;
    }

    let idx = rng.random_range(0..genome.edges.len());
    genome.edges.swap_remove(idx);
    genome.num_synapses = genome.num_synapses.saturating_sub(1);
    sort_synapse_genes(&mut genome.edges);
}

pub(crate) fn mutate_remove_neuron<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    if genome.num_neurons == 0 {
        return;
    }

    let previous_num_neurons = genome.num_neurons;
    let removed_inter_idx = rng.random_range(0..genome.num_neurons);
    let removed_neuron_id = inter_neuron_id(removed_inter_idx);

    genome.num_neurons = genome.num_neurons.saturating_sub(1);

    let removed_idx = removed_inter_idx as usize;
    genome.inter_biases.remove(removed_idx);
    genome.inter_log_time_constants.remove(removed_idx);
    genome.inter_locations.remove(removed_idx);

    genome.edges.retain_mut(|edge| {
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
    genome.num_synapses = genome.edges.len() as u32;
}

pub(crate) fn mutate_add_neuron_split_edge<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    if genome.edges.is_empty() || genome.num_neurons >= u32::MAX.saturating_sub(INTER_ID_BASE) {
        return;
    }

    let target_inter_len = genome.num_neurons as usize;
    genome.inter_biases.resize(target_inter_len, 0.0);
    genome
        .inter_log_time_constants
        .resize(target_inter_len, DEFAULT_INTER_LOG_TIME_CONSTANT);
    genome
        .inter_locations
        .resize(target_inter_len, BrainLocation { x: 5.0, y: 5.0 });
    genome
        .sensory_locations
        .resize(SENSORY_COUNT as usize, BrainLocation { x: 5.0, y: 5.0 });
    genome
        .action_locations
        .resize(ACTION_COUNT, BrainLocation { x: 5.0, y: 5.0 });

    let selected_idx = select_weighted_edge_index(&genome.edges, rng);
    let selected_edge = genome.edges.swap_remove(selected_idx);

    let new_inter_id = inter_neuron_id(genome.num_neurons);

    let pre_tau = inter_log_time_constant_for_neuron(selected_edge.pre_neuron_id, genome);
    let post_tau = inter_log_time_constant_for_neuron(selected_edge.post_neuron_id, genome);
    let base_tau = match (pre_tau, post_tau) {
        (Some(pre), Some(post)) => 0.5 * (pre + post),
        (Some(pre), None) => pre,
        (None, Some(post)) => post,
        (None, None) => DEFAULT_INTER_LOG_TIME_CONSTANT,
    };
    let new_log_tau = perturb_clamped(
        base_tau,
        INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV * 0.5,
        INTER_LOG_TIME_CONSTANT_MIN,
        INTER_LOG_TIME_CONSTANT_MAX,
        rng,
    );
    let new_bias = perturb_clamped(
        0.0,
        BIAS_PERTURBATION_STDDEV * 0.5,
        -BIAS_MAX,
        BIAS_MAX,
        rng,
    );

    let pre_location = location_for_neuron(selected_edge.pre_neuron_id, genome);
    let post_location = location_for_neuron(selected_edge.post_neuron_id, genome);
    let midpoint = BrainLocation {
        x: 0.5 * (pre_location.x + post_location.x),
        y: 0.5 * (pre_location.y + post_location.y),
    };
    let new_location = BrainLocation {
        x: perturb_clamped(
            midpoint.x,
            LOCATION_PERTURBATION_STDDEV * 0.5,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        ),
        y: perturb_clamped(
            midpoint.y,
            LOCATION_PERTURBATION_STDDEV * 0.5,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        ),
    };

    genome.num_neurons = genome.num_neurons.saturating_add(1);
    genome.inter_biases.push(new_bias);
    genome.inter_log_time_constants.push(new_log_tau);
    genome.inter_locations.push(new_location);

    genome.edges.push(SynapseEdge {
        pre_neuron_id: selected_edge.pre_neuron_id,
        post_neuron_id: new_inter_id,
        weight: 1.0,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
    genome.edges.push(SynapseEdge {
        pre_neuron_id: new_inter_id,
        post_neuron_id: selected_edge.post_neuron_id,
        weight: constrain_weight(selected_edge.weight),
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });

    genome.num_synapses = genome.num_synapses.saturating_add(1);
    sync_synapse_genes_to_target(genome, rng);
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
    if !is_inter_id(neuron_id, genome.num_neurons) {
        return None;
    }
    let inter_idx = inter_index(neuron_id, genome.num_neurons as usize)?;
    genome.inter_log_time_constants.get(inter_idx).copied()
}

pub(super) fn location_for_neuron(neuron_id: NeuronId, genome: &OrganismGenome) -> BrainLocation {
    if is_sensory_id(neuron_id) {
        return genome
            .sensory_locations
            .get(neuron_id.0 as usize)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
    }
    if is_inter_id(neuron_id, genome.num_neurons) {
        let Some(inter_idx) = inter_index(neuron_id, genome.num_neurons as usize) else {
            return BrainLocation { x: 5.0, y: 5.0 };
        };
        return genome
            .inter_locations
            .get(inter_idx)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
    }
    if is_action_id(neuron_id) {
        let Some(action_idx) = action_array_index(neuron_id) else {
            return BrainLocation { x: 5.0, y: 5.0 };
        };
        return genome
            .action_locations
            .get(action_idx)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
    }

    BrainLocation { x: 5.0, y: 5.0 }
}
