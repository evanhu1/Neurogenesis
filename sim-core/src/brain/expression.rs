use super::*;

pub(crate) fn express_genome(genome: &OrganismGenome, predation_enabled: bool) -> BrainState {
    // Heritable node IDs are stable structural hashes. The runtime remains a
    // dense array machine: canonical node order is remapped to compact
    // `NeuronId`s once at birth.
    let num_inter = genome.brain.hidden_nodes.len();

    let sensory = sensory_receptors_in_order(predation_enabled)
        .map(|(sensory_id, receptor)| make_sensory_neuron(sensory_id, receptor))
        .collect::<Vec<_>>();

    let mut inter = Vec::with_capacity(num_inter);
    for (index, gene) in genome.brain.hidden_nodes.iter().enumerate() {
        let alpha = inter_alpha_from_log_time_constant(gene.log_time_constant);
        inter.push(InterNeuronState {
            neuron: make_neuron(inter_neuron_id(index as u32), NeuronType::Inter, gene.bias),
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

fn sensory_receptors_in_order(
    predation_enabled: bool,
) -> impl Iterator<Item = (u32, SensoryReceptor)> {
    SensoryReceptor::active(predation_enabled).filter_map(|receptor| {
        receptor
            .neuron_id()
            .map(|neuron_id| (neuron_id.0, receptor))
    })
}

fn wire_birth_synapses_from_genome(
    genome: &OrganismGenome,
    sensory: &mut [SensoryNeuronState],
    inter: &mut [InterNeuronState],
) {
    for edge in &genome.brain.edges {
        if !edge.enabled {
            continue;
        }
        let Some(pre_runtime_id) = runtime_neuron_id(genome, edge.pre_node_id) else {
            continue;
        };
        let Some(post_runtime_id) = runtime_neuron_id(genome, edge.post_node_id) else {
            continue;
        };

        if pre_runtime_id.0 < SENSORY_COUNT {
            let Some(pre) = sensory.get_mut(pre_runtime_id.0 as usize) else {
                continue;
            };
            pre.synapses.push(runtime_edge_from_gene(
                edge,
                pre_runtime_id,
                post_runtime_id,
            ));
            continue;
        }

        let Some(pre_index) = crate::topology::inter_index(pre_runtime_id, inter.len()) else {
            continue;
        };
        // Inter-neuron self-edges are preserved: `evaluate_brain` reads the
        // presynaptic inter's previous-tick activation before updating state,
        // so a self-edge acts as a gated memory retention term.
        let Some(pre) = inter.get_mut(pre_index) else {
            continue;
        };
        pre.synapses.push(runtime_edge_from_gene(
            edge,
            pre_runtime_id,
            post_runtime_id,
        ));
    }

    // Innovation order is unrelated to dense runtime IDs. Restore the runtime
    // inter-target/action-target partition required by routing. Post IDs remain
    // ordered inside each group, but hidden IDs above the stable action-ID
    // island are intentionally numerically greater than action IDs.
    for neuron in sensory.iter_mut() {
        crate::topology::sort_runtime_synapses(&mut neuron.synapses);
    }
    for neuron in inter.iter_mut() {
        crate::topology::sort_runtime_synapses(&mut neuron.synapses);
    }
    if cfg!(debug_assertions) {
        for sensory_neuron in sensory.iter() {
            crate::topology::debug_assert_runtime_synapse_order(&sensory_neuron.synapses);
        }
        for inter_neuron in inter.iter() {
            crate::topology::debug_assert_runtime_synapse_order(&inter_neuron.synapses);
        }
    }
}

fn runtime_neuron_id(genome: &OrganismGenome, gene_id: GeneNodeId) -> Option<NeuronId> {
    if let Some(index) = sensory_gene_node_index(gene_id) {
        return (index < SENSORY_COUNT).then_some(NeuronId(index));
    }
    if let Some(index) = action_gene_node_index(gene_id) {
        return (index < ACTION_COUNT).then(|| action_neuron_id(index));
    }
    let index = genome
        .brain
        .hidden_nodes
        .binary_search_by_key(&gene_id, |node| node.id)
        .ok()?;
    Some(inter_neuron_id(index as u32))
}

fn runtime_edge_from_gene(
    gene: &SynapseGene,
    pre_neuron_id: NeuronId,
    post_neuron_id: NeuronId,
) -> SynapseEdge {
    // Gene weights are already constrained: `sanitize_synapse_genes` clamps
    // every retained edge and spatial-prior additions sample within
    // [SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX].
    debug_assert_eq!(gene.weight, constrain_weight(gene.weight));
    SynapseEdge {
        pre_neuron_id,
        post_neuron_id,
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
