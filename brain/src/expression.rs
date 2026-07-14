use super::*;

pub fn express_genome(genome: &OrganismGenome, predation_enabled: bool) -> BrainState {
    // Heritable node IDs are stable structural hashes. The runtime remains a
    // dense array machine: canonical gene order is remapped once at birth into
    // deterministic topological order for a same-tick feed-forward pass.
    let num_inter = genome.brain.hidden_nodes.len();
    let topological_order = hidden_topological_order(genome);
    let mut runtime_index_by_gene_index = vec![0usize; num_inter];
    for (runtime_index, &gene_index) in topological_order.iter().enumerate() {
        runtime_index_by_gene_index[gene_index] = runtime_index;
    }

    let sensory = sensory_receptors_in_order(predation_enabled)
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
    wire_birth_synapses_from_genome(
        genome,
        &runtime_index_by_gene_index,
        &mut brain.sensory,
        &mut brain.inter,
    );
    refresh_action_synapse_starts_and_count(&mut brain);
    brain
}

pub fn make_sensory_neuron(id: u32, receptor: SensoryReceptor) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, 0.0),
        receptor,
        synapses: Vec::new(),
        action_synapse_start: 0,
    }
}

pub fn make_action_neuron(id: u32, action_type: ActionType) -> ActionNeuronState {
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
    runtime_index_by_gene_index: &[usize],
    sensory: &mut [SensoryNeuronState],
    inter: &mut [InterNeuronState],
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
        for (pre_index, inter_neuron) in inter.iter().enumerate() {
            debug_assert!(inter_neuron.synapses[..inter_neuron.action_synapse_start]
                .iter()
                .all(
                    |edge| crate::topology::inter_index(edge.post_neuron_id, inter.len())
                        .is_some_and(|post_index| post_index > pre_index)
                ));
        }
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
    for edge in genome.brain.edges.iter().filter(|edge| edge.enabled) {
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
    assert_eq!(order.len(), count, "expressed hidden graph must be acyclic");
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
