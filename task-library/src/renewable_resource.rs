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
pub struct RenewableResourceConfig {
    pub ticks_per_instance: usize,
    pub stock: u32,
}

impl Default for RenewableResourceConfig {
    fn default() -> Self {
        Self {
            ticks_per_instance: 512,
            stock: 64,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RenewableResourceTask {
    pub config: RenewableResourceConfig,
}

pub struct RenewableResourceState {
    seed: u64,
    target: Symbol,
    remaining: u32,
    tick: usize,
    renewal: u64,
}

impl SymbolicTask for RenewableResourceTask {
    type Config = RenewableResourceConfig;
    type State = RenewableResourceState;

    fn name(&self) -> &'static str {
        "renewable_symbolic_resource"
    }
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
    fn validate(&self) -> Result<()> {
        if self.config.ticks_per_instance == 0 || self.config.stock == 0 {
            bail!("renewable-resource horizon and stock must be positive");
        }
        Ok(())
    }
    fn action_enabled(&self, action: Symbol) -> bool {
        action != Symbol::End
    }
    fn max_steps_per_instance(&self) -> usize {
        self.config.ticks_per_instance
    }
    fn start(&self, panel_seed: u64, instance: usize) -> Self::State {
        let seed = mix64(panel_seed ^ instance as u64);
        RenewableResourceState {
            seed,
            target: ACTIONS[instance % ACTIONS.len()],
            remaining: self.config.stock,
            tick: 0,
            renewal: 0,
        }
    }
    fn observe(&self, _state: &Self::State) -> Observation {
        Observation::default()
    }
    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.target;
        let correct = action == expected;
        if correct {
            state.remaining -= 1;
        }
        if state.remaining == 0 {
            state.renewal += 1;
            let mut next = ACTIONS[(mix64(state.seed ^ state.renewal) as usize) % ACTIONS.len()];
            if next == state.target {
                next = ACTIONS[(next.index() + 1) % ACTIONS.len()];
            }
            state.target = next;
            state.remaining = self.config.stock;
        }
        state.tick += 1;
        Transition {
            reward: if correct { 1.0 } else { -1.0 / 7.0 },
            expected_action: Some(expected),
            success_events: u32::from(correct),
            correct,
            trial_outcome: None,
            done: state.tick >= self.config.ticks_per_instance,
        }
    }
}
