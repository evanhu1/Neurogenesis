use crate::{mix64, Observation, SymbolicTask, Transition};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use types::Symbol;

const ACTIONS: [Symbol; 8] = [
    Symbol::A,
    Symbol::B,
    Symbol::C,
    Symbol::D,
    Symbol::E,
    Symbol::F,
    Symbol::G,
    Symbol::H,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub sequence_length: usize,
    pub attempts: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            sequence_length: 4,
            attempts: 32,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MemoryTask {
    pub config: MemoryConfig,
}

pub struct MemoryState {
    target: Vec<Symbol>,
    position: usize,
    attempt: usize,
    prefix_intact: bool,
    probe_position: usize,
    probe_exact: bool,
}

impl SymbolicTask for MemoryTask {
    type Config = MemoryConfig;
    type State = MemoryState;

    fn name(&self) -> &'static str {
        "basic_memory"
    }
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
    fn validate(&self) -> Result<()> {
        if self.config.sequence_length == 0 || self.config.attempts == 0 {
            bail!("memory length and attempts must be positive");
        }
        Ok(())
    }
    fn action_enabled(&self, action: Symbol) -> bool {
        action != Symbol::End
    }
    fn max_steps_per_instance(&self) -> usize {
        self.config.sequence_length * self.config.attempts
    }
    fn start(&self, panel_seed: u64, instance: usize) -> Self::State {
        let seed = mix64(panel_seed ^ instance as u64);
        let target = (0..self.config.sequence_length)
            .map(|position| ACTIONS[(mix64(seed ^ position as u64) as usize) % ACTIONS.len()])
            .collect();
        MemoryState {
            target,
            position: 0,
            attempt: 0,
            prefix_intact: true,
            probe_position: 0,
            probe_exact: true,
        }
    }
    fn observe(&self, _state: &Self::State) -> Observation {
        Observation::default()
    }
    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.target[state.position];
        let correct = action == expected;
        state.prefix_intact &= correct;
        state.position += 1;
        let attempt_done = state.position == state.target.len();
        let trial_outcome = attempt_done.then_some(state.prefix_intact);
        if attempt_done {
            state.attempt += 1;
            state.position = 0;
            state.prefix_intact = true;
        }
        Transition {
            reward: if correct { 1.0 } else { -1.0 / 7.0 },
            expected_action: Some(expected),
            success_events: 0,
            correct,
            trial_outcome,
            done: state.attempt >= self.config.attempts,
        }
    }

    fn probe_steps_per_instance(&self) -> usize {
        self.config.sequence_length
    }

    fn begin_probe(&self, state: &mut Self::State) {
        state.probe_position = 0;
        state.probe_exact = true;
    }

    fn probe_observe(&self, _state: &Self::State) -> Observation {
        Observation::default()
    }

    fn probe_step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.target[state.probe_position];
        let correct = action == expected;
        state.probe_exact &= correct;
        state.probe_position += 1;
        let done = state.probe_position == state.target.len();
        Transition {
            reward: 0.0,
            expected_action: Some(expected),
            // Every correct final-probe position is an independent atomic
            // success. Positions are symmetric; no prefix is privileged.
            success_events: u32::from(correct),
            correct,
            trial_outcome: done.then_some(state.probe_exact),
            done,
        }
    }
}
