use super::*;

pub fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let num_neurons = config.num_neurons.min(MAX_INTER_NEURONS) as usize;
    let max_synapses = max_possible_synapses(num_neurons);
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
            initial_learning_rate: config.initial_learning_rate,
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
    if config.num_synapses > 0 {
        if let Some(hidden) = genome.brain.hidden_nodes.first() {
            genome.brain.edges.push(SynapseGene {
                innovation: connection_innovation_id(
                    hidden.id,
                    hidden.id,
                    SynapseTiming::PreviousTick,
                ),
                pre_node_id: hidden.id,
                post_node_id: hidden.id,
                timing: SynapseTiming::PreviousTick,
                weight: super::synapse_creation::sample_synapse_weight(
                    INITIAL_SYNAPSE_EXCITATORY_PROBABILITY,
                    rng,
                ),
                plasticity_coefficient: 1.0,
                enabled: true,
            });
        }
    }
    let seeded_recurrent = genome.brain.edges.len();
    super::synapse_creation::add_synapse_genes(
        &mut genome,
        (config.num_synapses as usize)
            .min(max_synapses)
            .saturating_sub(seeded_recurrent),
        rng,
    );
    super::sanitization::sort_synapse_genes(&mut genome.brain.edges);
    debug_assert_genome_well_formed(&genome);
    genome
}
