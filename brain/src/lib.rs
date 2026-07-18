mod evaluation;
mod expression;
pub mod genome;
mod learning;
#[cfg(feature = "profiling")]
mod profiling;
mod topology;

use crate::genome::inter_alpha_from_log_time_constant;
#[cfg(feature = "profiling")]
use crate::profiling::BrainStage;
use crate::topology::{
    action_neuron_id, constrain_weight, inter_index, inter_neuron_id,
    refresh_output_synapse_starts_and_count,
};
#[cfg(feature = "profiling")]
use std::time::Instant;
use types::{
    action_gene_node_index, sensory_gene_node_index, ActionNeuronState, BrainState, GeneNodeId,
    InterNeuronState, NeuronId, NeuronState, NeuronType, OrganismGenome, SensoryNeuronState,
    SensoryReceptor, Symbol, SynapseEdge, SynapseGene, SynapseTiming,
};

pub use crate::topology::{action_index, ACTION_COUNT, ACTION_ID_BASE, SENSORY_COUNT};
pub use evaluation::{evaluate_brain, evaluate_brain_state, BrainEvalContext};
#[cfg_attr(not(test), allow(unused_imports))]
pub use expression::{express_genome, make_action_neuron, make_sensory_neuron};
pub use learning::{
    apply_immediate_action_reward, reset_dynamics_preserving_weights, ImmediateLearningReport,
};
pub use plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};

mod plasticity;

const MIN_ACTION_TEMPERATURE: f32 = 1.0e-6;

/// Fast tanh approximation using a Padé rational polynomial.
/// Accurate to ~1e-5 for |x| ≲ 4.2; worst-case absolute error grows to ~1e-4
/// near the cutoff (≈9.6e-5 at |x| = 4.97). The 4.97 cutoff is where the
/// rational reaches ≈1.0, chosen so the clamp to exact ±1.0 for larger inputs
/// is continuous (jump ≈6e-7), not for the 1e-5 bound.
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
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
pub struct BrainEvaluation {
    pub selected_symbol: Symbol,
    pub action_logits: [f32; ACTION_COUNT],
    pub synapse_ops: u64,
}

pub struct BrainScratch {
    pub inter_inputs: Vec<f32>,
    pub prev_inter: Vec<f32>,
    pub inter_activations: Vec<f32>,
    pub action_probabilities: [f32; ACTION_COUNT],
}

impl BrainScratch {
    pub fn new() -> Self {
        Self {
            inter_inputs: Vec::with_capacity(32),
            prev_inter: Vec::with_capacity(32),
            inter_activations: Vec::with_capacity(32),
            action_probabilities: [0.0; ACTION_COUNT],
        }
    }

    fn prepare_inter_buffers(&mut self, brain: &BrainState) {
        self.inter_inputs.clear();
        self.inter_inputs
            .extend(brain.inter.iter().map(|inter| inter.neuron.bias));
        self.prev_inter.clear();
        if !brain.recurrent_synapses.is_empty() {
            self.prev_inter
                .extend_from_slice(&brain.previous_inter_activations);
        }
    }
}

impl Default for BrainScratch {
    fn default() -> Self {
        Self::new()
    }
}
