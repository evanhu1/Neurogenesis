mod evaluation;
mod expression;
mod sensing;

use crate::genome::{
    inter_alpha_from_log_time_constant, BRAIN_SPACE_MAX, BRAIN_SPACE_MIN,
    DEFAULT_INTER_LOG_TIME_CONSTANT,
};
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use crate::topology::{
    action_neuron_id, constrain_weight, inter_index, inter_neuron_id,
    refresh_parent_ids_and_synapse_count, split_inter_and_action_edges,
};
use sim_types::{
    ActionNeuronState, ActionType, BrainLocation, BrainState, EntityType, InterNeuronState,
    NeuronId, NeuronState, NeuronType, Occupant, OrganismGenome, OrganismId, OrganismState,
    SensoryNeuronState, SensoryReceptor, SynapseEdge,
};
#[cfg(feature = "profiling")]
use std::time::Instant;

pub(crate) use crate::topology::{
    action_index, ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT,
};
#[cfg_attr(not(test), allow(unused_imports))]
pub use evaluation::derive_active_action_neuron_id;
pub(crate) use evaluation::{evaluate_brain, BrainEvalContext};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use expression::{express_genome, make_action_neuron, make_sensory_neuron};
#[cfg_attr(not(feature = "instrumentation"), allow(unused_imports))]
pub(crate) use sensing::scan_rays;

const DEFAULT_BIAS: f32 = 0.0;
const MIN_ENERGY_SENSOR_SCALE: f32 = 1.0;
const ENERGY_SENSOR_CURVE_EXPONENT: f32 = 2.0;
const MIN_ACTION_TEMPERATURE: f32 = 1.0e-6;
pub(crate) const EXPLICIT_IDLE_LOGIT_BIAS: f32 = -0.01;
pub(crate) const LOOK_RAY_COUNT: usize = SensoryReceptor::LOOK_RAY_OFFSETS.len();

#[derive(Default)]
pub(crate) struct BrainEvaluation {
    pub(crate) selected_action: ActionType,
    pub(crate) action_logits: [f32; ACTION_COUNT],
    pub(crate) synapse_ops: u64,
    #[cfg(feature = "instrumentation")]
    pub(crate) food_ahead: bool,
    #[cfg(feature = "instrumentation")]
    pub(crate) food_left: bool,
    #[cfg(feature = "instrumentation")]
    pub(crate) food_right: bool,
    #[cfg(feature = "instrumentation")]
    pub(crate) food_behind: bool,
}

pub(crate) struct BrainScratch {
    pub(crate) inter_inputs: Vec<f32>,
    pub(crate) prev_inter: Vec<f32>,
    pub(crate) prev_inter_states: Vec<f32>,
    pub(crate) inter_activations: Vec<f32>,
    pub(crate) centered_action_post_signals: [f32; ACTION_COUNT],
    pub(crate) selected_action_index: Option<usize>,
    pub(crate) selected_action_confidence: f32,
}

impl BrainScratch {
    pub(crate) fn new() -> Self {
        Self {
            inter_inputs: Vec::new(),
            prev_inter: Vec::new(),
            prev_inter_states: Vec::new(),
            inter_activations: Vec::new(),
            centered_action_post_signals: [0.0; ACTION_COUNT],
            selected_action_index: None,
            selected_action_confidence: 0.0,
        }
    }

    fn prepare_inter_buffers(&mut self, brain: &BrainState) {
        self.inter_inputs.clear();
        self.inter_inputs
            .extend(brain.inter.iter().map(|inter| inter.neuron.bias));
        self.prev_inter.clear();
        self.prev_inter
            .extend(brain.inter.iter().map(|inter| inter.neuron.activation));
        self.prev_inter_states.clear();
        self.prev_inter_states
            .extend(brain.inter.iter().map(|inter| inter.state));
    }

    fn update_action_post_signals(&mut self, action_inputs: &[f32; ACTION_COUNT]) {
        let centered_action_mean = action_inputs.iter().sum::<f32>() / ACTION_COUNT as f32;
        for (idx, centered_signal) in self.centered_action_post_signals.iter_mut().enumerate() {
            *centered_signal = action_inputs[idx] - centered_action_mean;
        }
    }
}
