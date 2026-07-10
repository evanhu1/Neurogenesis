use crate::genome::{SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN};
use sim_types::{ActionType, BrainState, NeuronId, SensoryReceptor, SynapseEdge};

pub(crate) const SENSORY_COUNT: u32 = SensoryReceptor::TOTAL_NEURON_COUNT;
pub(crate) const ACTION_COUNT: usize = ActionType::ALL.len();
// Single source of truth for the wire-visible neuron-ID layout lives in
// sim-types; these are local aliases, not independent definitions.
pub(crate) const INTER_ID_BASE: u32 = sim_types::INTER_NEURON_ID_BASE;
pub(crate) const ACTION_ID_BASE: u32 = sim_types::ACTION_NEURON_ID_BASE;

pub(crate) use sim_types::inter_neuron_id;

pub(crate) fn action_index(action: ActionType) -> usize {
    ActionType::ALL
        .iter()
        .position(|candidate| *candidate == action)
        .expect("idle has no action neuron")
}

pub(crate) fn action_neuron_id(index: usize) -> NeuronId {
    NeuronId(ACTION_ID_BASE + index as u32)
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

/// Recomputes every sensory and inter neuron's `action_synapse_start` and
/// refreshes `synapse_count`. Called at birth and from the runtime prune
/// path; allocation-free.
pub(crate) fn refresh_action_synapse_starts_and_count(brain: &mut BrainState) {
    let mut total_synapses = 0usize;
    for sensory_neuron in brain.sensory.iter_mut() {
        debug_assert_sorted_by_post_neuron_id(&sensory_neuron.synapses);
        sensory_neuron.action_synapse_start = sensory_neuron
            .synapses
            .partition_point(|edge| edge.post_neuron_id.0 < ACTION_ID_BASE);
        total_synapses += sensory_neuron.synapses.len();
    }

    for inter_neuron in brain.inter.iter_mut() {
        debug_assert_sorted_by_post_neuron_id(&inter_neuron.synapses);
        inter_neuron.action_synapse_start = inter_neuron
            .synapses
            .partition_point(|edge| edge.post_neuron_id.0 < ACTION_ID_BASE);
        total_synapses += inter_neuron.synapses.len();
    }

    brain.synapse_count = total_synapses as u32;
}

pub(crate) fn debug_assert_sorted_by_post_neuron_id(edges: &[SynapseEdge]) {
    debug_assert!(
        edges
            .windows(2)
            .all(|pair| pair[0].post_neuron_id <= pair[1].post_neuron_id),
        "outgoing synapses must stay sorted by post_neuron_id"
    );
}
