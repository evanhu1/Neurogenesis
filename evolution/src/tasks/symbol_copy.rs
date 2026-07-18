use crate::task::EvaluationTask;
use anyhow::{bail, Result};
use brain::{evaluate_brain_state, express_genome, BrainEvalContext, BrainScratch};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use types::{OrganismGenome, SensoryReceptor, Symbol};

pub const OBJECTIVE_NAME: &str = "correct_symbols";
pub const TASK_NAME: &str = "symbol_copy_training";
pub const DEFAULT_TRAINING_STREAM_COUNT: usize = 32;
pub const DEFAULT_HOLDOUT_STREAM_COUNT: usize = 32;
pub const DEFAULT_STREAM_BODY_LENGTH: usize = 16;
pub const TRAINING_CORPUS_SEED: u64 = 0x5452_4149_4e5f_434f;
pub const HOLDOUT_CORPUS_SEED: u64 = 0x484f_4c44_4f55_545f;

const BODY_ALPHABET: [Symbol; 8] = [
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
pub struct SymbolCopyTaskConfig {
    pub training_streams: Vec<Vec<Symbol>>,
    pub holdout_streams: Vec<Vec<Symbol>>,
    pub leaky_neurons_enabled: bool,
}

impl Default for SymbolCopyTaskConfig {
    fn default() -> Self {
        Self {
            training_streams: generate_streams(
                TRAINING_CORPUS_SEED,
                DEFAULT_TRAINING_STREAM_COUNT,
                DEFAULT_STREAM_BODY_LENGTH,
            ),
            holdout_streams: generate_streams(
                HOLDOUT_CORPUS_SEED,
                DEFAULT_HOLDOUT_STREAM_COUNT,
                DEFAULT_STREAM_BODY_LENGTH,
            ),
            leaky_neurons_enabled: false,
        }
    }
}

impl SymbolCopyTaskConfig {
    pub fn validate(&self) -> Result<()> {
        validate_streams("training", &self.training_streams)?;
        validate_streams("holdout", &self.holdout_streams)
    }

    pub fn training_symbols_per_genome(&self) -> u64 {
        symbol_count(&self.training_streams)
    }

    pub fn holdout_symbols_per_winner(&self) -> u64 {
        symbol_count(&self.holdout_streams)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolCopyTask {
    pub config: SymbolCopyTaskConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvaluation {
    pub input: Vec<Symbol>,
    pub output: Vec<Symbol>,
    pub correct: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolCopyEvaluation {
    pub correct: u64,
    pub total: u64,
    pub accuracy: f64,
    pub streams: Vec<StreamEvaluation>,
}

pub type SymbolCopyRunResult = crate::RunResult<SymbolCopyTaskConfig, SymbolCopyEvaluation>;

impl EvaluationTask for SymbolCopyTask {
    type Config = SymbolCopyTaskConfig;
    type Evaluation = SymbolCopyEvaluation;

    fn name(&self) -> &'static str {
        TASK_NAME
    }

    fn objective(&self) -> &'static str {
        OBJECTIVE_NAME
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }

    fn validate(&self) -> Result<()> {
        self.config.validate()
    }

    fn evaluate(&self, genome: &OrganismGenome) -> Result<Self::Evaluation> {
        Ok(evaluate_streams(
            genome,
            &self.config.training_streams,
            self.config.leaky_neurons_enabled,
        ))
    }

    fn fitness(&self, evaluation: &Self::Evaluation) -> f64 {
        evaluation.correct as f64
    }

    fn normalized_fitness(&self, evaluation: &Self::Evaluation) -> Option<f64> {
        Some(evaluation.accuracy)
    }

    fn validation_evaluation(&self, genome: &OrganismGenome) -> Result<Option<Self::Evaluation>> {
        Ok(Some(evaluate_streams(
            genome,
            &self.config.holdout_streams,
            self.config.leaky_neurons_enabled,
        )))
    }
}

/// Build a reproducible corpus in which every stream contains every body
/// symbol in shuffled order and terminates with `end`.
pub fn generate_streams(seed: u64, stream_count: usize, body_length: usize) -> Vec<Vec<Symbol>> {
    assert!(body_length >= BODY_ALPHABET.len());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..stream_count)
        .map(|_| {
            let mut stream = Vec::with_capacity(body_length + 1);
            stream.extend(BODY_ALPHABET);
            while stream.len() < body_length {
                stream.push(BODY_ALPHABET[rng.random_range(0..BODY_ALPHABET.len())]);
            }
            stream.shuffle(&mut rng);
            stream.push(Symbol::End);
            stream
        })
        .collect()
}

fn validate_streams(name: &str, streams: &[Vec<Symbol>]) -> Result<()> {
    if streams.is_empty() {
        bail!("at least one {name} symbol stream is required");
    }
    for (index, stream) in streams.iter().enumerate() {
        if stream.is_empty() || stream.last() != Some(&Symbol::End) {
            bail!("{name} stream {index} must be nonempty and end with `end`");
        }
        if stream[..stream.len() - 1].contains(&Symbol::End) {
            bail!("{name} stream {index} contains `end` before its final position");
        }
    }
    Ok(())
}

fn symbol_count(streams: &[Vec<Symbol>]) -> u64 {
    streams.iter().map(|stream| stream.len() as u64).sum()
}

fn evaluate_streams(
    genome: &OrganismGenome,
    inputs: &[Vec<Symbol>],
    leaky_neurons_enabled: bool,
) -> SymbolCopyEvaluation {
    let mut streams = Vec::with_capacity(inputs.len());
    let mut correct = 0_u64;
    let mut total = 0_u64;
    for input in inputs {
        let mut brain = express_genome(genome);
        let mut scratch = BrainScratch::new();
        let mut output = Vec::with_capacity(input.len());
        let mut stream_correct = 0_u64;
        for &symbol in input {
            for sensory in &mut brain.sensory {
                sensory.neuron.activation = match sensory.receptor {
                    SensoryReceptor::Symbol { symbol: receptor } => f32::from(receptor == symbol),
                };
            }
            let evaluation = evaluate_brain_state(
                &mut brain,
                genome,
                BrainEvalContext {
                    leaky_neurons_enabled,
                    action_temperature: 1.0,
                    action_sample: None,
                },
                &mut scratch,
            );
            let emitted = argmax_symbol(evaluation.action_logits);
            output.push(emitted);
            stream_correct += u64::from(emitted == symbol);
        }
        correct += stream_correct;
        total += input.len() as u64;
        streams.push(StreamEvaluation {
            input: input.clone(),
            output,
            correct: stream_correct,
        });
    }
    SymbolCopyEvaluation {
        correct,
        total,
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        },
        streams,
    }
}

fn argmax_symbol(logits: [f32; Symbol::COUNT]) -> Symbol {
    Symbol::ALL
        .into_iter()
        .max_by(|left, right| {
            logits[left.index()]
                .total_cmp(&logits[right.index()])
                .then_with(|| right.index().cmp(&left.index()))
        })
        .expect("the symbol alphabet is nonempty")
}
