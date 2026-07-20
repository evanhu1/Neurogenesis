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
pub struct ContinualLearningConfig {
    pub lifetime_ticks: usize,
    pub minimum_regime_ticks: usize,
    pub maximum_regime_ticks: usize,
}

impl Default for ContinualLearningConfig {
    fn default() -> Self {
        Self {
            lifetime_ticks: 512,
            minimum_regime_ticks: 32,
            maximum_regime_ticks: 96,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ContinualLearningTask {
    pub config: ContinualLearningConfig,
}

pub struct ContinualLearningState {
    seed: u64,
    target: Symbol,
    ticks_remaining_in_regime: usize,
    tick: usize,
    reversal: u64,
}

impl ContinualLearningTask {
    fn regime_ticks(&self, seed: u64, reversal: u64) -> usize {
        let span = self.config.maximum_regime_ticks - self.config.minimum_regime_ticks + 1;
        self.config.minimum_regime_ticks + (mix64(seed ^ reversal) as usize % span)
    }
}

impl SymbolicTask for ContinualLearningTask {
    type Config = ContinualLearningConfig;
    type State = ContinualLearningState;

    fn name(&self) -> &'static str {
        "basic_continual_learning"
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }

    fn validate(&self) -> Result<()> {
        if self.config.lifetime_ticks == 0
            || self.config.minimum_regime_ticks == 0
            || self.config.maximum_regime_ticks < self.config.minimum_regime_ticks
        {
            bail!("continual-learning lifetime and regime bounds must be positive and ordered");
        }
        Ok(())
    }

    fn action_enabled(&self, action: Symbol) -> bool {
        action != Symbol::End
    }

    fn max_steps_per_instance(&self) -> usize {
        self.config.lifetime_ticks
    }

    fn start(&self, panel_seed: u64, instance: usize) -> Self::State {
        let seed = mix64(panel_seed ^ instance as u64);
        ContinualLearningState {
            seed,
            target: ACTIONS[instance % ACTIONS.len()],
            ticks_remaining_in_regime: self.regime_ticks(seed, 0),
            tick: 0,
            reversal: 0,
        }
    }

    fn observe(&self, _state: &Self::State) -> Observation {
        Observation::default()
    }

    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.target;
        let correct = action == expected;
        state.tick += 1;
        state.ticks_remaining_in_regime -= 1;
        if state.ticks_remaining_in_regime == 0 && state.tick < self.config.lifetime_ticks {
            state.reversal += 1;
            let offset = 1 + (mix64(state.seed ^ state.reversal) as usize % (ACTIONS.len() - 1));
            state.target = ACTIONS[(state.target.index() + offset) % ACTIONS.len()];
            state.ticks_remaining_in_regime = self.regime_ticks(state.seed, state.reversal);
        }
        Transition {
            reward: if correct { 1.0 } else { -1.0 / 7.0 },
            expected_action: Some(expected),
            success_events: u32::from(correct),
            correct,
            trial_outcome: None,
            done: state.tick >= self.config.lifetime_ticks,
        }
    }
}
