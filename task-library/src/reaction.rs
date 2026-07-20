use crate::{mix64, Observation, SymbolicTask, Transition};
use anyhow::{bail, Result};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use types::Symbol;

const ALPHABET: [Symbol; 4] = [Symbol::A, Symbol::B, Symbol::C, Symbol::D];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionConfig {
    pub symbols_per_instance: usize,
}

impl Default for ReactionConfig {
    fn default() -> Self {
        Self {
            symbols_per_instance: 17,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReactionTask {
    pub config: ReactionConfig,
}

pub struct ReactionState {
    stream: Vec<Symbol>,
    position: usize,
}

impl SymbolicTask for ReactionTask {
    type Config = ReactionConfig;
    type State = ReactionState;

    fn name(&self) -> &'static str {
        "basic_reaction"
    }
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
    fn validate(&self) -> Result<()> {
        if self.config.symbols_per_instance < ALPHABET.len() + 1 {
            bail!("reaction instances must include four body symbols plus end");
        }
        Ok(())
    }
    fn observes_symbols(&self) -> bool {
        true
    }
    fn action_enabled(&self, action: Symbol) -> bool {
        matches!(
            action,
            Symbol::A | Symbol::B | Symbol::C | Symbol::D | Symbol::End
        )
    }
    fn max_steps_per_instance(&self) -> usize {
        self.config.symbols_per_instance
    }
    fn start(&self, panel_seed: u64, instance: usize) -> Self::State {
        let mut rng = ChaCha8Rng::seed_from_u64(mix64(panel_seed ^ instance as u64));
        let body_len = self.config.symbols_per_instance - 1;
        let mut stream = ALPHABET.to_vec();
        while stream.len() < body_len {
            stream.push(ALPHABET[rng.random_range(0..ALPHABET.len())]);
        }
        stream.shuffle(&mut rng);
        stream.push(Symbol::End);
        ReactionState {
            stream,
            position: 0,
        }
    }
    fn observe(&self, state: &Self::State) -> Observation {
        Observation {
            symbol: state.stream.get(state.position).copied(),
        }
    }
    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let correct = state
            .stream
            .get(state.position)
            .is_some_and(|target| *target == action);
        state.position += 1;
        Transition {
            reward: if correct { 1.0 } else { -1.0 },
            expected_action: state.stream.get(state.position - 1).copied(),
            success_events: u32::from(correct),
            correct,
            trial_outcome: None,
            done: state.position >= state.stream.len(),
        }
    }
}
