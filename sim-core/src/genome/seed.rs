use super::sanitization::reconcile_synapse_count;
use super::*;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let num_neurons = config.num_neurons.min(MAX_INTER_NEURONS);
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
    let action_biases = (0..ACTION_COUNT)
        .map(|_| sample_initial_bias(rng))
        .collect();

    let mut genome = OrganismGenome {
        topology: TopologyGenes {
            num_neurons,
            num_synapses: config.num_synapses.min(max_synapses),
            spatial_prior_sigma: config.spatial_prior_sigma.max(0.01),
            vision_distance: config.vision_distance,
        },
        lifecycle: LifecycleGenes {
            body_color: sample_body_color(rng),
            age_of_maturity: config.age_of_maturity,
            gestation_ticks: config.gestation_ticks,
            max_organism_age: config.max_organism_age,
        },
        plasticity: PlasticityGenes {
            hebb_eta_gain: config.hebb_eta_gain,
            juvenile_eta_scale: config.juvenile_eta_scale,
            eligibility_retention: config.eligibility_retention,
            max_weight_delta_per_tick: config.max_weight_delta_per_tick,
            synapse_prune_threshold: config.synapse_prune_threshold,
        },
        mutation_rates: MutationRateGenes {
            age_of_maturity: config.mutation_rate_age_of_maturity,
            gestation_ticks: config.mutation_rate_gestation_ticks,
            max_organism_age: config.mutation_rate_max_organism_age,
            vision_distance: config.mutation_rate_vision_distance,
            hebb_eta_gain: config.mutation_rate_hebb_eta_gain,
            juvenile_eta_scale: config.mutation_rate_juvenile_eta_scale,
            inter_bias: config.mutation_rate_inter_bias,
            inter_update_rate: config.mutation_rate_inter_update_rate,
            eligibility_retention: config.mutation_rate_eligibility_retention,
            synapse_prune_threshold: config.mutation_rate_synapse_prune_threshold,
            neuron_location: config.mutation_rate_neuron_location,
            synapse_weight_perturbation: config.mutation_rate_synapse_weight_perturbation,
            add_synapse: config.mutation_rate_add_synapse,
            remove_synapse: config.mutation_rate_remove_synapse,
            remove_neuron: config.mutation_rate_remove_neuron,
            add_neuron_split_edge: config.mutation_rate_add_neuron_split_edge,
            spatial_prior_sigma: config.mutation_rate_spatial_prior_sigma,
            max_weight_delta_per_tick: config.mutation_rate_max_weight_delta_per_tick,
        },
        brain: BrainTopology {
            inter_biases,
            inter_log_time_constants,
            sensory_locations,
            inter_locations,
            action_locations,
            action_biases,
            edges: Vec::new(),
        },
    };
    reconcile_synapse_count(&mut genome, rng);
    genome
}
