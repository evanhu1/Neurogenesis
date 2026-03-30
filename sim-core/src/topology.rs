use crate::genome::{SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN};
use sim_types::{ActionType, BrainState, NeuronId, SensoryReceptor, SynapseEdge};

pub(crate) const CONTACT_SENSORY_ID: u32 = SensoryReceptor::LOOK_NEURON_COUNT;
pub(crate) const DAMAGE_SENSORY_ID: u32 = CONTACT_SENSORY_ID + 1;
pub(crate) const ENERGY_SENSORY_ID: u32 = DAMAGE_SENSORY_ID + 1;
pub(crate) const SENSORY_COUNT: u32 = ENERGY_SENSORY_ID + 1;
pub(crate) const ACTION_COUNT: usize = ActionType::ALL.len();
pub(crate) const ACTION_COUNT_U32: u32 = ACTION_COUNT as u32;
pub(crate) const INTER_ID_BASE: u32 = 1000;
pub(crate) const ACTION_ID_BASE: u32 = 2000;

pub(crate) fn action_index(action: ActionType) -> usize {
    ActionType::ALL
        .iter()
        .position(|candidate| *candidate == action)
        .expect("idle has no action neuron")
}

pub(crate) fn inter_neuron_id(index: u32) -> NeuronId {
    NeuronId(INTER_ID_BASE + index)
}

pub(crate) fn action_neuron_id(index: usize) -> NeuronId {
    NeuronId(ACTION_ID_BASE + index as u32)
}

pub(crate) fn is_sensory_id(id: NeuronId) -> bool {
    id.0 < SENSORY_COUNT
}

pub(crate) fn is_inter_id(id: NeuronId, num_neurons: u32) -> bool {
    (INTER_ID_BASE..INTER_ID_BASE + num_neurons).contains(&id.0)
}

pub(crate) fn is_action_id(id: NeuronId) -> bool {
    (ACTION_ID_BASE..ACTION_ID_BASE + ACTION_COUNT_U32).contains(&id.0)
}

pub(crate) fn inter_index(id: NeuronId, num_neurons: usize) -> Option<usize> {
    let idx = id.0.checked_sub(INTER_ID_BASE)? as usize;
    (idx < num_neurons).then_some(idx)
}

pub(crate) fn action_array_index(id: NeuronId) -> Option<usize> {
    let idx = id.0.checked_sub(ACTION_ID_BASE)? as usize;
    (idx < ACTION_COUNT).then_some(idx)
}

pub(crate) fn constrain_weight(weight: f32) -> f32 {
    if weight == 0.0 {
        return SYNAPSE_STRENGTH_MIN;
    }
    weight.signum()
        * weight
            .abs()
            .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
}

pub(crate) fn refresh_parent_ids_and_synapse_count(brain: &mut BrainState) {
    let inter_len = brain.inter.len();
    let action_len = brain.action.len();
    let mut inter_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); inter_len];
    let mut action_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); action_len];

    for sensory in &brain.sensory {
        collect_parent_ids(
            sensory.neuron.neuron_id,
            &sensory.synapses,
            &mut inter_parent_ids,
            &mut action_parent_ids,
        );
    }

    for inter in &brain.inter {
        collect_parent_ids(
            inter.neuron.neuron_id,
            &inter.synapses,
            &mut inter_parent_ids,
            &mut action_parent_ids,
        );
    }

    for (idx, inter) in brain.inter.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut inter_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        inter.neuron.parent_ids = parents;
    }
    for (idx, action) in brain.action.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut action_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        action.parent_ids = parents;
    }

    let synapse_count = brain
        .sensory
        .iter()
        .map(|n| n.synapses.len())
        .sum::<usize>()
        + brain.inter.iter().map(|n| n.synapses.len()).sum::<usize>();
    brain.synapse_count = synapse_count as u32;
}

pub(crate) fn split_inter_and_action_edges(
    edges: &[SynapseEdge],
) -> (&[SynapseEdge], &[SynapseEdge]) {
    debug_assert_sorted_by_post_neuron_id(edges);
    let split_idx = edges.partition_point(|edge| edge.post_neuron_id.0 < ACTION_ID_BASE);
    edges.split_at(split_idx)
}

pub(crate) fn split_inter_and_action_edges_mut(
    edges: &mut [SynapseEdge],
) -> (&mut [SynapseEdge], &mut [SynapseEdge]) {
    debug_assert_sorted_by_post_neuron_id(edges);
    let split_idx = edges.partition_point(|edge| edge.post_neuron_id.0 < ACTION_ID_BASE);
    edges.split_at_mut(split_idx)
}

fn collect_parent_ids(
    pre_id: NeuronId,
    synapses: &[SynapseEdge],
    inter_parent_ids: &mut [Vec<NeuronId>],
    action_parent_ids: &mut [Vec<NeuronId>],
) {
    for synapse in synapses {
        if let Some(inter_idx) = inter_index(synapse.post_neuron_id, inter_parent_ids.len()) {
            inter_parent_ids[inter_idx].push(pre_id);
            continue;
        }
        if let Some(action_idx) = action_array_index(synapse.post_neuron_id) {
            action_parent_ids[action_idx].push(pre_id);
        }
    }
}

fn debug_assert_sorted_by_post_neuron_id(edges: &[SynapseEdge]) {
    debug_assert!(
        edges
            .windows(2)
            .all(|pair| pair[0].post_neuron_id <= pair[1].post_neuron_id),
        "outgoing synapses must stay sorted by post_neuron_id"
    );
}
