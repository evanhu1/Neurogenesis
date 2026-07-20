use crate::{Observation, SymbolicTask, Transition};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use types::Symbol;

pub const DEFAULT_SNIPPET: &str = "the quick brown fox jumps over the lazy dog";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextTokenPredictionConfig {
    pub snippet: String,
    pub learning_passes: usize,
}

impl Default for NextTokenPredictionConfig {
    fn default() -> Self {
        Self {
            snippet: DEFAULT_SNIPPET.to_owned(),
            learning_passes: 4,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NextTokenPredictionTask {
    pub config: NextTokenPredictionConfig,
}

pub struct NextTokenPredictionState {
    targets: Vec<Symbol>,
    position: usize,
    learning_pass: usize,
    pass_exact: bool,
    probe_position: usize,
    probe_exact: bool,
}

impl NextTokenPredictionTask {
    fn targets(&self) -> Result<Vec<Symbol>> {
        let mut targets = self
            .config
            .snippet
            .chars()
            .map(|character| {
                Symbol::from_ascii_char(character).ok_or_else(|| {
                    anyhow::anyhow!(
                        "next-token snippet accepts only lowercase ASCII letters and spaces"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        targets.push(Symbol::End);
        Ok(targets)
    }
}

impl SymbolicTask for NextTokenPredictionTask {
    type Config = NextTokenPredictionConfig;
    type State = NextTokenPredictionState;

    fn name(&self) -> &'static str {
        "basic_next_token_prediction"
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }

    fn validate(&self) -> Result<()> {
        let targets = self.targets()?;
        if targets.len() < 3 {
            bail!("next-token snippet must contain at least two characters");
        }
        if self.config.learning_passes == 0 {
            bail!("next-token learning passes must be positive");
        }
        let mut alphabet = [false; Symbol::COUNT];
        for target in targets {
            alphabet[target.index()] = true;
        }
        if !(Symbol::A.index()..=Symbol::Z.index()).all(|index| alphabet[index]) {
            bail!("next-token snippet must contain every letter a through z");
        }
        if !alphabet[Symbol::Space.index()] {
            bail!("next-token snippet must contain at least one space");
        }
        Ok(())
    }

    fn observes_symbols(&self) -> bool {
        true
    }

    fn action_enabled(&self, _action: Symbol) -> bool {
        true
    }

    fn max_steps_per_instance(&self) -> usize {
        (self.config.snippet.chars().count() + 1) * self.config.learning_passes
    }

    fn start(&self, _panel_seed: u64, _instance: usize) -> Self::State {
        NextTokenPredictionState {
            targets: self.targets().expect("validated next-token snippet"),
            position: 0,
            learning_pass: 0,
            pass_exact: true,
            probe_position: 0,
            probe_exact: true,
        }
    }

    fn observe(&self, state: &Self::State) -> Observation {
        Observation {
            symbol: Some(if state.position == 0 {
                Symbol::End
            } else {
                state.targets[state.position - 1]
            }),
        }
    }

    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.targets[state.position];
        let correct = action == expected;
        state.pass_exact &= correct;
        state.position += 1;
        let pass_done = state.position == state.targets.len();
        let trial_outcome = pass_done.then_some(state.pass_exact);
        if pass_done {
            state.learning_pass += 1;
            state.position = 0;
            state.pass_exact = true;
        }
        Transition {
            reward: if correct { 1.0 } else { -1.0 / 27.0 },
            expected_action: Some(expected),
            success_events: 0,
            correct,
            trial_outcome,
            done: state.learning_pass == self.config.learning_passes,
        }
    }

    fn probe_steps_per_instance(&self) -> usize {
        self.config.snippet.chars().count() + 1
    }

    fn begin_probe(&self, state: &mut Self::State) {
        state.probe_position = 0;
        state.probe_exact = true;
    }

    fn probe_observe(&self, state: &Self::State) -> Observation {
        Observation {
            symbol: Some(if state.probe_position == 0 {
                Symbol::End
            } else {
                state.targets[state.probe_position - 1]
            }),
        }
    }

    fn probe_step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.targets[state.probe_position];
        let correct = action == expected;
        state.probe_exact &= correct;
        state.probe_position += 1;
        let done = state.probe_position == state.targets.len();
        Transition {
            reward: 0.0,
            expected_action: Some(expected),
            success_events: u32::from(correct),
            correct,
            trial_outcome: done.then_some(state.probe_exact),
            done,
        }
    }
}
