use super::*;

pub(crate) fn express_genome(genome: &OrganismGenome) -> BrainState {
    // `align_genome_vectors` forces every genome vector to its exact target
    // length on all paths into expression (seed generation, mutation,
    // external-genome intake) — fail fast instead of silently expressing
    // zero-bias / default-alpha neurons.
    let num_inter = genome.topology.num_neurons as usize;
    debug_assert_eq!(genome.brain.inter_biases.len(), num_inter);
    debug_assert_eq!(genome.brain.inter_log_time_constants.len(), num_inter);

    let sensory = sensory_receptors_in_order()
        .map(|(sensory_id, receptor)| make_sensory_neuron(sensory_id, receptor))
        .collect::<Vec<_>>();

    let mut inter = Vec::with_capacity(num_inter);
    for i in 0..genome.topology.num_neurons {
        let idx = i as usize;
        let bias = genome.brain.inter_biases[idx];
        let log_time_constant = genome.brain.inter_log_time_constants[idx];
        let alpha = inter_alpha_from_log_time_constant(log_time_constant);
        inter.push(InterNeuronState {
            neuron: make_neuron(inter_neuron_id(i), NeuronType::Inter, bias),
            state: 0.0,
            alpha,
            synapses: Vec::new(),
            action_synapse_start: 0,
        });
    }

    let mut action = Vec::with_capacity(ACTION_COUNT);
    for (idx, action_type) in ActionType::ALL.iter().copied().enumerate() {
        action.push(make_action_neuron(action_neuron_id(idx).0, action_type));
    }

    let num_sensory = sensory.len();
    let num_inter = inter.len();
    let mut brain = BrainState {
        sensory,
        inter,
        action,
        synapse_count: 0,
        sensory_mean_activation: vec![0.0; num_sensory],
        inter_mean_activation: vec![0.0; num_inter],
        action_mean_activation: vec![0.0; ACTION_COUNT],
        means_initialized: false,
    };
    wire_birth_synapses_from_genome(genome, &mut brain.sensory, &mut brain.inter);
    refresh_action_synapse_starts_and_count(&mut brain);
    brain
}

pub(crate) fn make_sensory_neuron(id: u32, receptor: SensoryReceptor) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, 0.0),
        receptor,
        synapses: Vec::new(),
        action_synapse_start: 0,
    }
}

pub(crate) fn make_action_neuron(id: u32, action_type: ActionType) -> ActionNeuronState {
    ActionNeuronState {
        neuron_id: NeuronId(id),
        logit: 0.0,
        action_type,
    }
}

fn sensory_receptors_in_order() -> impl Iterator<Item = (u32, SensoryReceptor)> {
    SensoryReceptor::ordered()
        .enumerate()
        .map(|(idx, receptor)| (idx as u32, receptor))
}

fn wire_birth_synapses_from_genome(
    genome: &OrganismGenome,
    sensory: &mut [SensoryNeuronState],
    inter: &mut [InterNeuronState],
) {
    let max_inter_id = INTER_ID_BASE + inter.len() as u32;
    let max_action_id = ACTION_ID_BASE + ACTION_COUNT_U32;

    for edge in &genome.brain.edges {
        let post_is_inter = (INTER_ID_BASE..max_inter_id).contains(&edge.post_neuron_id.0);
        let post_is_action = (ACTION_ID_BASE..max_action_id).contains(&edge.post_neuron_id.0);
        if !(post_is_inter || post_is_action) {
            continue;
        }

        if edge.pre_neuron_id.0 < SENSORY_COUNT {
            let Some(pre) = sensory.get_mut(edge.pre_neuron_id.0 as usize) else {
                continue;
            };
            pre.synapses.push(runtime_edge_from_gene(edge));
            continue;
        }

        if !(INTER_ID_BASE..max_inter_id).contains(&edge.pre_neuron_id.0) {
            continue;
        }
        // Inter-neuron self-edges are preserved: `evaluate_brain` reads the
        // presynaptic inter's previous-tick activation before updating state,
        // so a self-edge acts as a gated memory retention term.
        let pre_idx = (edge.pre_neuron_id.0 - INTER_ID_BASE) as usize;
        let Some(pre) = inter.get_mut(pre_idx) else {
            continue;
        };
        pre.synapses.push(runtime_edge_from_gene(edge));
    }

    // Genome edges are sorted by (pre, post) with unique keys — sanitization
    // runs on every path into expression (seed generation, mutation,
    // external-genome intake) — and edges are pushed per pre-neuron in genome
    // iteration order, so every per-pre list is already sorted by post ID.
    // Partition-based routing and plasticity assume this invariant.
    if cfg!(debug_assertions) {
        for sensory_neuron in sensory.iter() {
            crate::topology::debug_assert_sorted_by_post_neuron_id(&sensory_neuron.synapses);
        }
        for inter_neuron in inter.iter() {
            crate::topology::debug_assert_sorted_by_post_neuron_id(&inter_neuron.synapses);
        }
    }
}

fn runtime_edge_from_gene(gene: &SynapseGene) -> SynapseEdge {
    // Gene weights are already constrained: `sanitize_synapse_genes` clamps
    // every retained edge and spatial-prior additions sample within
    // [SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX].
    debug_assert_eq!(gene.weight, constrain_weight(gene.weight));
    SynapseEdge {
        pre_neuron_id: gene.pre_neuron_id,
        post_neuron_id: gene.post_neuron_id,
        weight: gene.weight,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }
}

fn make_neuron(id: NeuronId, neuron_type: NeuronType, bias: f32) -> NeuronState {
    NeuronState {
        neuron_id: id,
        neuron_type,
        bias,
        activation: 0.0,
    }
}
