use super::*;

pub fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    predation_enabled: bool,
    rng: &mut R,
) -> OrganismGenome {
    let num_neurons = config.num_neurons.min(MAX_INTER_NEURONS) as usize;
    let max_synapses = max_possible_synapses(num_neurons, predation_enabled);
    let mut hidden_nodes = (0..num_neurons)
        .map(|index| HiddenNodeGene {
            id: seed_hidden_gene_node_id(index as u32),
            bias: sample_initial_bias(rng),
            log_time_constant: sample_initial_log_time_constant(rng),
        })
        .collect::<Vec<_>>();
    hidden_nodes.sort_unstable_by_key(|node| node.id);
    let action_biases = (0..ACTION_COUNT)
        .map(|_| sample_initial_bias(rng))
        .collect();

    let mut genome = OrganismGenome {
        lifecycle: LifecycleGenes {
            plasticity_maturity_ticks: config.plasticity_maturity_ticks,
        },
        plasticity: PlasticityGenes {
            hebb_eta_gain: config.hebb_eta_gain,
            juvenile_eta_scale: config.juvenile_eta_scale,
            eligibility_retention: config.eligibility_retention,
            max_weight_delta_per_tick: config.max_weight_delta_per_tick,
            synapse_prune_threshold: config.synapse_prune_threshold,
        },
        brain: BrainTopology {
            hidden_nodes,
            action_biases,
            edges: Vec::new(),
        },
    };
    super::synapse_creation::add_synapse_genes(
        &mut genome,
        (config.num_synapses as usize).min(max_synapses),
        predation_enabled,
        rng,
    );
    super::restrict_predation_genes(&mut genome, predation_enabled);
    super::sanitization::sort_synapse_genes(&mut genome.brain.edges);
    debug_assert_genome_well_formed(&genome);
    genome
}
