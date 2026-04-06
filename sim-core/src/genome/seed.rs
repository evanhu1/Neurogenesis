use super::sanitization::sync_synapse_genes_to_target;
use super::*;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let num_neurons = config.num_neurons;
    let max_synapses = max_possible_synapses(num_neurons);
    let inter_biases = (0..num_neurons).map(|_| sample_initial_bias(rng)).collect();
    let inter_log_time_constants = (0..num_neurons)
        .map(|_| sample_initial_log_time_constant(rng))
        .collect();
    let inter_locations = (0..num_neurons)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let sensory_locations = (0..SENSORY_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_locations = (0..ACTION_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();

    let mut genome = OrganismGenome {
        num_neurons,
        num_synapses: config.num_synapses.min(max_synapses),
        spatial_prior_sigma: config.spatial_prior_sigma.max(0.01),
        vision_distance: config.vision_distance,
        max_health: config.max_health,
        age_of_maturity: config.age_of_maturity,
        gestation_ticks: config.gestation_ticks,
        max_organism_age: config.max_organism_age,
        plasticity_start_age: config.plasticity_start_age,
        hebb_eta_gain: config.hebb_eta_gain,
        juvenile_eta_scale: config.juvenile_eta_scale,
        eligibility_retention: config.eligibility_retention,
        max_weight_delta_per_tick: config.max_weight_delta_per_tick,
        synapse_prune_threshold: config.synapse_prune_threshold,
        mutation_rate_age_of_maturity: config.mutation_rate_age_of_maturity,
        mutation_rate_gestation_ticks: config.mutation_rate_gestation_ticks,
        mutation_rate_max_organism_age: config.mutation_rate_max_organism_age,
        mutation_rate_vision_distance: config.mutation_rate_vision_distance,
        mutation_rate_max_health: config.mutation_rate_max_health,
        mutation_rate_inter_bias: config.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: config.mutation_rate_inter_update_rate,
        mutation_rate_eligibility_retention: config.mutation_rate_eligibility_retention,
        mutation_rate_synapse_prune_threshold: config.mutation_rate_synapse_prune_threshold,
        mutation_rate_neuron_location: config.mutation_rate_neuron_location,
        mutation_rate_synapse_weight_perturbation: config.mutation_rate_synapse_weight_perturbation,
        mutation_rate_add_synapse: config.mutation_rate_add_synapse,
        mutation_rate_remove_synapse: config.mutation_rate_remove_synapse,
        mutation_rate_remove_neuron: config.mutation_rate_remove_neuron,
        mutation_rate_add_neuron_split_edge: config.mutation_rate_add_neuron_split_edge,
        inter_biases,
        inter_log_time_constants,
        sensory_locations,
        inter_locations,
        action_locations,
        edges: Vec::new(),
    };
    sync_synapse_genes_to_target(&mut genome, rng);
    genome
}
