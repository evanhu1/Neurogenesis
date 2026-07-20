//! Brain- and optimizer-independent symbolic task environments.
//!
//! A task owns only its observable world, legal actions, reward/resource
//! consequences, and episode boundaries. Agent execution, learning, and
//! reproduction live in downstream crates.

pub mod continual_learning;
pub mod memory;
pub mod next_token_prediction;
pub mod reaction;
pub mod renewable_resource;

use anyhow::Result;
use serde::Serialize;
use types::Symbol;

#[derive(Debug, Clone, Copy, Default)]
pub struct Observation {
    /// Optional one-hot symbolic stimulus. `None` means a zero-input step.
    pub symbol: Option<Symbol>,
}

#[derive(Debug, Clone, Copy)]
pub struct Transition {
    pub reward: f32,
    /// Correct action for evaluator-only probability diagnostics. This is not
    /// included in the next observation.
    pub expected_action: Option<Symbol>,
    /// Count of atomic successes produced by this environment transition.
    /// Consumers decide how, or whether, these events affect optimization.
    pub success_events: u32,
    /// Task-relative correctness for instrumentation only.
    pub correct: bool,
    /// A semantic trial outcome, present only at a trial boundary. Consumers
    /// own any agent-state policy applied there.
    pub trial_outcome: Option<bool>,
    pub done: bool,
}

/// An environment contract with no knowledge of genomes, neurons, plasticity,
/// selection, mutation, or reproduction.
pub trait SymbolicTask: Sync {
    type Config: Clone + Serialize;
    type State: Send;

    fn name(&self) -> &'static str;
    fn config(&self) -> Self::Config;
    fn validate(&self) -> Result<()>;
    fn observes_symbols(&self) -> bool {
        false
    }
    fn action_enabled(&self, action: Symbol) -> bool;
    fn max_steps_per_instance(&self) -> usize;
    fn start(&self, panel_seed: u64, instance: usize) -> Self::State;
    fn observe(&self, state: &Self::State) -> Observation;
    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition;

    /// Number of frozen, greedy policy steps used after the learning lifetime.
    /// Zero means the task's ordinary transitions are also its scored behavior.
    fn probe_steps_per_instance(&self) -> usize {
        0
    }

    /// Prepare only the environment for a post-learning probe. The consumer
    /// owns agent-state reset, action policy, and plasticity policy.
    fn begin_probe(&self, _state: &mut Self::State) {}

    fn probe_observe(&self, state: &Self::State) -> Observation {
        self.observe(state)
    }

    fn probe_step(&self, _state: &mut Self::State, _action: Symbol) -> Transition {
        unreachable!("task declares no probe")
    }
}

pub(crate) fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
