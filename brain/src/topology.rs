use crate::genome::{SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN};
use types::{BrainState, NeuronId, SensoryReceptor, Symbol, SynapseEdge};

pub const SENSORY_COUNT: u32 = SensoryReceptor::TOTAL_NEURON_COUNT;
pub const ACTION_COUNT: usize = Symbol::COUNT;
// Single source of truth for the wire-visible neuron-ID layout lives in
// types; these are local aliases, not independent definitions.
pub const INTER_ID_BASE: u32 = types::INTER_NEURON_ID_BASE;
pub const ACTION_ID_BASE: u32 = types::ACTION_NEURON_ID_BASE;

pub use types::inter_neuron_id;

pub const fn action_index(symbol: Symbol) -> usize {
    symbol.index()
}

pub fn action_neuron_id(index: usize) -> NeuronId {
    NeuronId(ACTION_ID_BASE + index as u32)
}

pub const fn value_neuron_id() -> NeuronId {
    NeuronId(u32::MAX)
}

pub const fn is_value_neuron_id(id: NeuronId) -> bool {
    id.0 == value_neuron_id().0
}

pub fn inter_index(id: NeuronId, num_neurons: usize) -> Option<usize> {
    let num_neurons = u32::try_from(num_neurons).ok()?;
    types::inter_neuron_index(id, num_neurons).map(|index| index as usize)
}

pub fn action_array_index(id: NeuronId) -> Option<usize> {
    let idx = id.0.checked_sub(ACTION_ID_BASE)? as usize;
    (idx < ACTION_COUNT).then_some(idx)
}

pub fn constrain_weight(weight: f32) -> f32 {
    if weight == 0.0 {
        return SYNAPSE_STRENGTH_MIN;
    }
    weight.signum()
        * weight
            .abs()
            .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
}

/// Recomputes every sensory and inter neuron's `output_synapse_start` and
/// refreshes `synapse_count`. Called at birth and from the runtime prune
/// path; allocation-free.
pub fn refresh_output_synapse_starts_and_count(brain: &mut BrainState) {
    let mut total_synapses = brain
        .recurrent_synapses
        .len()
        .saturating_add(brain.action_feedback_synapses.len());
    for sensory_neuron in brain.sensory.iter_mut() {
        debug_assert_runtime_synapse_order(&sensory_neuron.synapses);
        sensory_neuron.output_synapse_start = sensory_neuron
            .synapses
            .partition_point(|edge| !is_output_neuron(edge.post_neuron_id));
        total_synapses += sensory_neuron.synapses.len();
    }

    for inter_neuron in brain.inter.iter_mut() {
        debug_assert_runtime_synapse_order(&inter_neuron.synapses);
        inter_neuron.output_synapse_start = inter_neuron
            .synapses
            .partition_point(|edge| !is_output_neuron(edge.post_neuron_id));
        total_synapses += inter_neuron.synapses.len();
    }

    brain.synapse_count = total_synapses as u32;
}

pub fn debug_assert_runtime_synapse_order(edges: &[SynapseEdge]) {
    debug_assert!(
        edges.windows(2).all(|pair| {
            runtime_post_sort_key(pair[0].post_neuron_id)
                <= runtime_post_sort_key(pair[1].post_neuron_id)
        }),
        "outgoing synapses must keep inter targets before output targets"
    );
}

pub fn sort_runtime_synapses(edges: &mut [SynapseEdge]) {
    edges.sort_unstable_by_key(|edge| runtime_post_sort_key(edge.post_neuron_id));
}

fn runtime_post_sort_key(id: NeuronId) -> (bool, u32) {
    (is_output_neuron(id), id.0)
}

fn is_output_neuron(id: NeuronId) -> bool {
    action_array_index(id).is_some() || is_value_neuron_id(id)
}
