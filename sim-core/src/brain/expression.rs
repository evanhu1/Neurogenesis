use super::*;

const AUX_SENSORY_RECEPTORS: [(u32, SensoryReceptor); 3] = [
    (CONTACT_SENSORY_ID, SensoryReceptor::ContactAhead),
    (DAMAGE_SENSORY_ID, SensoryReceptor::Damage),
    (ENERGY_SENSORY_ID, SensoryReceptor::Energy),
];

pub(crate) fn express_genome(genome: &OrganismGenome) -> BrainState {
    let sensory_spawn = sensory_spawn_location();
    let action_spawn = action_spawn_location();

    let sensory = sensory_receptors_in_order()
        .map(|(sensory_id, receptor)| make_sensory_neuron(sensory_id, receptor, sensory_spawn))
        .collect::<Vec<_>>();

    let mut inter = Vec::with_capacity(genome.num_neurons as usize);
    for i in 0..genome.num_neurons {
        let idx = i as usize;
        let bias = genome.inter_biases.get(idx).copied().unwrap_or(0.0);
        let log_time_constant = genome
            .inter_log_time_constants
            .get(idx)
            .copied()
            .unwrap_or(DEFAULT_INTER_LOG_TIME_CONSTANT);
        let alpha = inter_alpha_from_log_time_constant(log_time_constant);
        inter.push(InterNeuronState {
            neuron: make_neuron(
                inter_neuron_id(i),
                NeuronType::Inter,
                bias,
                location_or_default(&genome.inter_locations, idx),
            ),
            state: 0.0,
            alpha,
            synapses: Vec::new(),
        });
    }

    let mut action = Vec::with_capacity(ACTION_COUNT);
    for (idx, action_type) in ActionType::ALL.iter().copied().enumerate() {
        action.push(make_action_neuron(
            action_neuron_id(idx).0,
            action_type,
            action_spawn,
        ));
    }

    let mut brain = BrainState {
        sensory,
        inter,
        action,
        synapse_count: 0,
    };
    wire_birth_synapses_from_genome(genome, &mut brain.sensory, &mut brain.inter);
    refresh_parent_ids_and_synapse_count(&mut brain);
    brain
}

pub(crate) fn make_sensory_neuron(
    id: u32,
    receptor: SensoryReceptor,
    location: BrainLocation,
) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, DEFAULT_BIAS, location),
        receptor,
        synapses: Vec::new(),
    }
}

pub(crate) fn make_action_neuron(
    id: u32,
    action_type: ActionType,
    location: BrainLocation,
) -> ActionNeuronState {
    ActionNeuronState {
        neuron_id: NeuronId(id),
        x: location.x,
        y: location.y,
        logit: 0.0,
        parent_ids: Vec::new(),
        action_type,
    }
}

fn sensory_receptors_in_order() -> impl Iterator<Item = (u32, SensoryReceptor)> {
    let look_receptors = SensoryReceptor::LOOK_RAY_OFFSETS
        .into_iter()
        .flat_map(|ray_offset| {
            LOOK_TARGETS
                .into_iter()
                .map(move |look_target| SensoryReceptor::LookRay {
                    ray_offset,
                    look_target,
                })
        })
        .enumerate()
        .map(|(idx, receptor)| (idx as u32, receptor));
    look_receptors.chain(AUX_SENSORY_RECEPTORS)
}

fn location_or_default(locations: &[BrainLocation], index: usize) -> BrainLocation {
    locations.get(index).copied().unwrap_or(BrainLocation {
        x: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
        y: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
    })
}

fn sensory_spawn_location() -> BrainLocation {
    BrainLocation {
        x: BRAIN_SPACE_MIN,
        y: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
    }
}

fn action_spawn_location() -> BrainLocation {
    BrainLocation {
        x: BRAIN_SPACE_MAX,
        y: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
    }
}

fn wire_birth_synapses_from_genome(
    genome: &OrganismGenome,
    sensory: &mut [SensoryNeuronState],
    inter: &mut [InterNeuronState],
) {
    let max_inter_id = INTER_ID_BASE + inter.len() as u32;
    let max_action_id = ACTION_ID_BASE + ACTION_COUNT_U32;

    for edge in &genome.edges {
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
        if edge.pre_neuron_id == edge.post_neuron_id {
            continue;
        }
        let pre_idx = (edge.pre_neuron_id.0 - INTER_ID_BASE) as usize;
        let Some(pre) = inter.get_mut(pre_idx) else {
            continue;
        };
        pre.synapses.push(runtime_edge_from_gene(edge));
    }

    sort_outgoing_synapses(
        sensory
            .iter_mut()
            .map(|sensory_neuron| &mut sensory_neuron.synapses),
    );
    sort_outgoing_synapses(
        inter
            .iter_mut()
            .map(|inter_neuron| &mut inter_neuron.synapses),
    );
}

fn runtime_edge_from_gene(edge: &SynapseEdge) -> SynapseEdge {
    SynapseEdge {
        pre_neuron_id: edge.pre_neuron_id,
        post_neuron_id: edge.post_neuron_id,
        weight: constrain_weight(edge.weight),
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }
}

fn sort_outgoing_synapses<'a>(synapse_groups: impl Iterator<Item = &'a mut Vec<SynapseEdge>>) {
    for synapses in synapse_groups {
        // Keep outgoing edges sorted by post ID; partition-based routing and plasticity assume it.
        synapses.sort_by(|a, b| {
            a.post_neuron_id
                .cmp(&b.post_neuron_id)
                .then_with(|| a.weight.total_cmp(&b.weight))
        });
    }
}

fn make_neuron(
    id: NeuronId,
    neuron_type: NeuronType,
    bias: f32,
    location: BrainLocation,
) -> NeuronState {
    NeuronState {
        neuron_id: id,
        neuron_type,
        bias,
        x: location.x,
        y: location.y,
        activation: 0.0,
        parent_ids: Vec::new(),
    }
}
