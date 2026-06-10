mod evaluation;
mod expression;
mod sensing;

use crate::genome::inter_alpha_from_log_time_constant;
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use crate::topology::{
    action_neuron_id, constrain_weight, inter_index, inter_neuron_id,
    refresh_action_synapse_starts_and_count,
};
use sim_types::{
    ActionNeuronState, ActionType, BrainLocation, BrainState, InterNeuronState, NeuronId,
    NeuronState, NeuronType, Occupant, OrganismGenome, OrganismId, OrganismState,
    SensoryNeuronState, SensoryReceptor, SynapseEdge, SynapseGene, VisionChannel, VisualProperties,
};
#[cfg(feature = "profiling")]
use std::time::Instant;

pub(crate) use crate::topology::{
    action_index, ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT,
};
pub(crate) use evaluation::{evaluate_brain, BrainEvalContext};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use expression::{express_genome, make_action_neuron, make_sensory_neuron};
#[cfg_attr(not(feature = "instrumentation"), allow(unused_imports))]
pub(crate) use sensing::scan_rays;

const MIN_ACTION_TEMPERATURE: f32 = 1.0e-6;
pub(crate) const EXPLICIT_IDLE_LOGIT_BIAS: f32 = -0.01;
pub(crate) const VISION_RAY_COUNT: usize = SensoryReceptor::VISION_RAY_OFFSETS.len();

/// Fast tanh approximation using a Padé rational polynomial.
/// Accurate to ~1e-5 for |x| ≲ 4.2; worst-case absolute error grows to ~1e-4
/// near the cutoff (≈9.6e-5 at |x| = 4.97). The 4.97 cutoff is where the
/// rational reaches ≈1.0, chosen so the clamp to exact ±1.0 for larger inputs
/// is continuous (jump ≈6e-7), not for the 1e-5 bound.
#[inline(always)]
pub(crate) fn fast_tanh(x: f32) -> f32 {
    if x >= 4.97 {
        return 1.0;
    }
    if x <= -4.97 {
        return -1.0;
    }
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    num / den
}

#[derive(Default)]
pub(crate) struct BrainEvaluation {
    pub(crate) selected_action: ActionType,
    pub(crate) action_logits: [f32; ACTION_COUNT],
    pub(crate) synapse_ops: u64,
    #[cfg(feature = "instrumentation")]
    pub(crate) food_visible: [bool; VISION_RAY_COUNT],
}

pub(crate) struct BrainScratch {
    pub(crate) inter_inputs: Vec<f32>,
    pub(crate) prev_inter: Vec<f32>,
    pub(crate) inter_activations: Vec<f32>,
    pub(crate) action_probabilities: [f32; ACTION_COUNT],
    pub(crate) selected_action_index: Option<usize>,
}

impl BrainScratch {
    pub(crate) fn new() -> Self {
        Self {
            inter_inputs: Vec::with_capacity(32),
            prev_inter: Vec::with_capacity(32),
            inter_activations: Vec::with_capacity(32),
            action_probabilities: [0.0; ACTION_COUNT],
            selected_action_index: None,
        }
    }

    fn prepare_inter_buffers(&mut self, brain: &BrainState) {
        self.inter_inputs.clear();
        self.inter_inputs
            .extend(brain.inter.iter().map(|inter| inter.neuron.bias));
        self.prev_inter.clear();
        self.prev_inter
            .extend(brain.inter.iter().map(|inter| inter.neuron.activation));
    }
}
