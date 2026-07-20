use super::*;

pub fn express_genome(genome: &OrganismGenome) -> BrainState {
    // Heritable node IDs are stable structural hashes. The runtime remains a
    // dense array machine: canonical gene order is remapped once at birth into
    // deterministic topological order for the instantaneous current-tick DAG.
    // Temporal recurrent edges are compiled separately below.
    let num_inter = genome.brain.hidden_nodes.len();
    let topological_order = hidden_topological_order(genome);
    let mut runtime_index_by_gene_index = vec![0usize; num_inter];
    for (runtime_index, &gene_index) in topological_order.iter().enumerate() {
        runtime_index_by_gene_index[gene_index] = runtime_index;
    }

    let sensory = sensory_receptors_in_order()
        .map(|(sensory_id, receptor)| make_sensory_neuron(sensory_id, receptor))
        .collect::<Vec<_>>();

    let mut inter = Vec::with_capacity(num_inter);
    for (runtime_index, &gene_index) in topological_order.iter().enumerate() {
        let gene = &genome.brain.hidden_nodes[gene_index];
        let alpha = inter_alpha_from_log_time_constant(gene.log_time_constant);
        inter.push(InterNeuronState {
            neuron: make_neuron(
                inter_neuron_id(runtime_index as u32),
                NeuronType::Inter,
                gene.bias,
            ),
            state: 0.0,
            alpha,
            neuromodulatory_receptor: gene.neuromodulatory_receptor,
            synapses: Vec::new(),
            output_synapse_start: 0,
        });
    }

    let mut action = Vec::with_capacity(ACTION_COUNT);
    for symbol in Symbol::ALL {
        action.push(make_action_neuron(symbol.action_neuron_id().0, symbol));
    }

    let num_sensory = sensory.len();
    let num_inter = inter.len();
    let mut brain = BrainState {
        sensory,
        inter,
        action,
        recurrent_synapses: Vec::new(),
        action_feedback_synapses: Vec::new(),
        previous_inter_activations: vec![0.0; num_inter],
        previous_action_activations: [0.0; ACTION_COUNT],
        previous_prediction_error: 0.0,
        value_bias: genome.brain.value_bias,
        inherited_value_bias: genome.brain.value_bias,
        value_bias_eligibility: 0.0,
        synapse_count: 0,
        sensory_mean_activation: vec![0.0; num_sensory],
        inter_mean_activation: vec![0.0; num_inter],
        action_mean_activation: vec![0.0; ACTION_COUNT],
        means_initialized: false,
    };
    wire_birth_synapses_from_genome(genome, &runtime_index_by_gene_index, &mut brain);
    refresh_output_synapse_starts_and_count(&mut brain);
    brain
}

pub fn make_sensory_neuron(id: u32, receptor: SensoryReceptor) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, 0.0),
        receptor,
        synapses: Vec::new(),
        output_synapse_start: 0,
    }
}

pub fn make_action_neuron(id: u32, symbol: Symbol) -> ActionNeuronState {
    ActionNeuronState {
        neuron_id: NeuronId(id),
        logit: 0.0,
        symbol,
    }
}

fn sensory_receptors_in_order() -> impl Iterator<Item = (u32, SensoryReceptor)> {
    SensoryReceptor::ordered().filter_map(|receptor| {
        receptor
            .neuron_id()
            .map(|neuron_id| (neuron_id.0, receptor))
    })
}

fn wire_birth_synapses_from_genome(
    genome: &OrganismGenome,
    runtime_index_by_gene_index: &[usize],
    brain: &mut BrainState,
) {
    for edge in &genome.brain.edges {
        if !edge.enabled {
            continue;
        }
        let Some(pre_runtime_id) =
            runtime_neuron_id(genome, runtime_index_by_gene_index, edge.pre_node_id)
        else {
            continue;
        };
        let Some(post_runtime_id) =
            runtime_neuron_id(genome, runtime_index_by_gene_index, edge.post_node_id)
        else {
            continue;
        };

        if edge.timing == SynapseTiming::PreviousTick {
            let mut runtime_edge = runtime_edge_from_gene(edge, pre_runtime_id, post_runtime_id);
            runtime_edge.post_inter_index =
                crate::topology::inter_index(post_runtime_id, brain.inter.len())
                    .and_then(|index| u32::try_from(index).ok());
            runtime_edge.pre_inter_index =
                crate::topology::inter_index(pre_runtime_id, brain.inter.len())
                    .and_then(|index| u32::try_from(index).ok());
            runtime_edge.pre_action_index = crate::topology::action_array_index(pre_runtime_id)
                .and_then(|index| u32::try_from(index).ok());
            if runtime_edge.pre_inter_index.is_some() && runtime_edge.post_inter_index.is_some() {
                brain.recurrent_synapses.push(runtime_edge);
            } else if runtime_edge.pre_action_index.is_some()
                && runtime_edge.post_inter_index.is_some()
            {
                brain.action_feedback_synapses.push(runtime_edge);
            }
            continue;
        }

        if pre_runtime_id.0 < SENSORY_COUNT {
            let Some(pre) = brain.sensory.get_mut(pre_runtime_id.0 as usize) else {
                continue;
            };
            pre.synapses.push(runtime_edge_from_gene(
                edge,
                pre_runtime_id,
                post_runtime_id,
            ));
            continue;
        }

        let Some(pre_index) = crate::topology::inter_index(pre_runtime_id, brain.inter.len())
        else {
            continue;
        };
        let Some(pre) = brain.inter.get_mut(pre_index) else {
            continue;
        };
        pre.synapses.push(runtime_edge_from_gene(
            edge,
            pre_runtime_id,
            post_runtime_id,
        ));
    }

    // Innovation order is unrelated to dense runtime IDs. Restore the runtime
    // inter-target/output-target partition required by routing. Post IDs remain
    // ordered inside each group, while hidden IDs skip both stable output-ID
    // islands.
    for neuron in brain.sensory.iter_mut() {
        crate::topology::sort_runtime_synapses(&mut neuron.synapses);
    }
    for neuron in brain.inter.iter_mut() {
        crate::topology::sort_runtime_synapses(&mut neuron.synapses);
    }
    if cfg!(debug_assertions) {
        for sensory_neuron in brain.sensory.iter() {
            crate::topology::debug_assert_runtime_synapse_order(&sensory_neuron.synapses);
        }
        for inter_neuron in brain.inter.iter() {
            crate::topology::debug_assert_runtime_synapse_order(&inter_neuron.synapses);
        }
        for (pre_index, inter_neuron) in brain.inter.iter().enumerate() {
            debug_assert!(inter_neuron.synapses[..inter_neuron.output_synapse_start]
                .iter()
                .all(
                    |edge| crate::topology::inter_index(edge.post_neuron_id, brain.inter.len())
                        .is_some_and(|post_index| post_index > pre_index)
                ));
        }
        debug_assert!(brain.recurrent_synapses.iter().all(|edge| {
            edge.timing == SynapseTiming::PreviousTick
                && edge.pre_inter_index.is_some()
                && edge.pre_action_index.is_none()
                && edge.post_inter_index.is_some()
                && crate::topology::inter_index(edge.pre_neuron_id, brain.inter.len()).is_some()
                && crate::topology::inter_index(edge.post_neuron_id, brain.inter.len()).is_some()
        }));
        debug_assert!(brain.action_feedback_synapses.iter().all(|edge| {
            edge.timing == SynapseTiming::PreviousTick
                && edge.pre_inter_index.is_none()
                && edge.pre_action_index.is_some()
                && edge.post_inter_index.is_some()
                && crate::topology::action_array_index(edge.pre_neuron_id).is_some()
                && crate::topology::inter_index(edge.post_neuron_id, brain.inter.len()).is_some()
        }));
    }
}

fn runtime_neuron_id(
    genome: &OrganismGenome,
    runtime_index_by_gene_index: &[usize],
    gene_id: GeneNodeId,
) -> Option<NeuronId> {
    if let Some(index) = sensory_gene_node_index(gene_id) {
        return (index < SENSORY_COUNT).then_some(NeuronId(index));
    }
    if types::is_value_gene_node_id(gene_id) {
        return Some(crate::topology::value_neuron_id());
    }
    if let Some(index) = action_gene_node_index(gene_id) {
        return (index < ACTION_COUNT).then(|| action_neuron_id(index));
    }
    let index = genome
        .brain
        .hidden_nodes
        .binary_search_by_key(&gene_id, |node| node.id)
        .ok()?;
    Some(inter_neuron_id(
        runtime_index_by_gene_index.get(index).copied()? as u32,
    ))
}

fn hidden_topological_order(genome: &OrganismGenome) -> Vec<usize> {
    let count = genome.brain.hidden_nodes.len();
    let mut indegree = vec![0usize; count];
    let mut outgoing = vec![Vec::<usize>::new(); count];
    for edge in genome
        .brain
        .edges
        .iter()
        .filter(|edge| edge.enabled && edge.timing == SynapseTiming::CurrentTick)
    {
        let Ok(pre) = genome
            .brain
            .hidden_nodes
            .binary_search_by_key(&edge.pre_node_id, |node| node.id)
        else {
            continue;
        };
        let Ok(post) = genome
            .brain
            .hidden_nodes
            .binary_search_by_key(&edge.post_node_id, |node| node.id)
        else {
            continue;
        };
        outgoing[pre].push(post);
        indegree[post] += 1;
    }
    for targets in &mut outgoing {
        targets.sort_unstable();
    }

    let mut ready = (0..count)
        .filter(|&index| indegree[index] == 0)
        .collect::<std::collections::BTreeSet<_>>();
    let mut order = Vec::with_capacity(count);
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for &post in &outgoing[index] {
            indegree[post] -= 1;
            if indegree[post] == 0 {
                ready.insert(post);
            }
        }
    }
    assert_eq!(
        order.len(),
        count,
        "expressed current-tick hidden graph must be acyclic"
    );
    order
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
        timing: gene.timing,
        pre_inter_index: None,
        pre_action_index: None,
        post_inter_index: None,
        inherited_weight: gene.weight,
        weight: gene.weight,
        plasticity_coefficient: gene.plasticity_coefficient,
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
