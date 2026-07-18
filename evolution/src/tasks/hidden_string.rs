use crate::EvaluationTask;
use anyhow::{bail, Result};
use brain::{
    apply_immediate_action_reward, evaluate_brain_state, express_genome,
    reset_dynamics_preserving_weights, BrainEvalContext, BrainScratch,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use types::{
    action_gene_node_id, connection_innovation_id, seed_hidden_gene_node_id, HiddenNodeGene,
    OrganismGenome, SensoryReceptor, Symbol, SynapseGene, SynapseTiming,
};

pub const TASK_NAME: &str = "hidden_string_adaptation_v4";
pub const OBJECTIVE_NAME: &str = "post_learning_hidden_string_greedy_prefix_score";
pub const TARGET_LEN: usize = 4;
pub const DEFAULT_TARGET_PANEL_SEED: u64 = 0x5041_4e45_4c5f_5632;

const SYMBOLS_PER_ORBIT: usize = 8;
const DISTINCT_COUNT_WEIGHTS: [usize; 3] = [3, 16, 13];
const ACTIVE_ACTIONS: [Symbol; 8] = [
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
#[serde(deny_unknown_fields)]
pub struct HiddenStringTaskConfig {
    pub attempts: u32,
    pub training_probe_after_attempts: Vec<u32>,
    pub report_probe_after_attempts: Vec<u32>,
    pub exploration_temperature: f32,
    pub reward_correct: f32,
    pub reward_incorrect: f32,
    pub target_panel_seed: u64,
    pub training_target_count: usize,
    pub development_target_count: usize,
    pub sealed_target_count: usize,
    pub development_interval: u32,
    pub training_rollout_seeds: Vec<u64>,
    pub development_rollout_seeds: Vec<u64>,
    pub sealed_rollout_seeds: Vec<u64>,
}

impl Default for HiddenStringTaskConfig {
    fn default() -> Self {
        Self {
            attempts: 32,
            training_probe_after_attempts: vec![32],
            report_probe_after_attempts: vec![0, 8, 16, 32],
            exploration_temperature: 1.0,
            reward_correct: 1.0,
            reward_incorrect: -1.0 / 3.0,
            target_panel_seed: DEFAULT_TARGET_PANEL_SEED,
            training_target_count: 1024,
            development_target_count: 256,
            sealed_target_count: 1024,
            development_interval: 25,
            training_rollout_seeds: vec![0x5452_4149_4e5f_3031, 0x5452_4149_4e5f_3032],
            development_rollout_seeds: vec![0x4445_565f_3030_3031],
            sealed_rollout_seeds: vec![0x5345_414c_4544_3031, 0x5345_414c_4544_3032],
        }
    }
}

impl HiddenStringTaskConfig {
    pub fn validate(&self) -> Result<()> {
        if self.attempts == 0 {
            bail!("attempts must be positive");
        }
        validate_probe_schedule(
            "training",
            &self.training_probe_after_attempts,
            self.attempts,
            false,
        )?;
        validate_probe_schedule(
            "report",
            &self.report_probe_after_attempts,
            self.attempts,
            true,
        )?;
        if !self.exploration_temperature.is_finite() || self.exploration_temperature <= 0.0 {
            bail!("exploration_temperature must be finite and positive");
        }
        if !self.reward_correct.is_finite()
            || !self.reward_incorrect.is_finite()
            || self.reward_correct <= 0.0
            || self.reward_incorrect >= 0.0
        {
            bail!("reward_correct must be positive and reward_incorrect negative");
        }
        for (name, count) in [
            ("training", self.training_target_count),
            ("development", self.development_target_count),
            ("sealed", self.sealed_target_count),
        ] {
            if count == 0 || !count.is_multiple_of(SYMBOLS_PER_ORBIT) {
                bail!("{name}_target_count must be a positive multiple of 8");
            }
        }
        if self.development_interval == 0 {
            bail!("development_interval must be positive");
        }
        for (name, seeds) in [
            ("training", &self.training_rollout_seeds),
            ("development", &self.development_rollout_seeds),
            ("sealed", &self.sealed_rollout_seeds),
        ] {
            if seeds.is_empty() {
                bail!("{name}_rollout_seeds must be nonempty");
            }
        }
        build_target_panels(self).map(|_| ())
    }
}

fn validate_probe_schedule(
    name: &str,
    schedule: &[u32],
    attempts: u32,
    require_zero: bool,
) -> Result<()> {
    if schedule.is_empty()
        || schedule.last() != Some(&attempts)
        || !schedule.windows(2).all(|pair| pair[0] < pair[1])
    {
        bail!("{name} probe schedule must increase strictly and end at attempts");
    }
    if require_zero && schedule.first() != Some(&0) {
        bail!("{name} probe schedule must begin at zero");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetPanelSummary {
    pub target_count: usize,
    pub rollout_count: usize,
    pub distinct_symbol_counts: [usize; TARGET_LEN + 1],
    pub position_symbol_counts: [[usize; ACTIVE_ACTIONS.len()]; TARGET_LEN],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeMetrics {
    pub after_attempts: u32,
    pub accuracy: f64,
    pub exact_string_rate: f64,
    /// Mean longest-correct-prefix length divided by the four-symbol target
    /// length, using greedy argmax outputs.
    pub greedy_prefix_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    pub target_count: usize,
    pub rollout_count: u32,
    pub probes: Vec<ProbeMetrics>,
    pub pre_learning_accuracy: Option<f64>,
    pub final_accuracy: f64,
    pub final_exact_string_rate: f64,
    pub final_greedy_prefix_score: f64,
    pub brain_synapse_operations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationControls {
    pub shuffled_reward: AdaptationMetrics,
    pub reset_weights_each_attempt: AdaptationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStringEvaluation {
    pub cohort: String,
    pub fitness: f64,
    pub learning_rate: f32,
    pub primary: AdaptationMetrics,
    pub controls: Option<AdaptationControls>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HiddenStringCondition {
    Primary,
    PlasticityOff,
    ShuffledReward,
    ResetWeightsEachAttempt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStringConditionEvaluation {
    pub panel: String,
    pub condition: HiddenStringCondition,
    pub metrics: AdaptationMetrics,
}

pub type HiddenStringRunResult = crate::RunResult<HiddenStringTaskConfig, HiddenStringEvaluation>;

#[derive(Clone)]
pub struct HiddenStringTask {
    pub config: HiddenStringTaskConfig,
    panels: Arc<TargetPanels>,
}

impl Default for HiddenStringTask {
    fn default() -> Self {
        Self::new(HiddenStringTaskConfig::default())
            .expect("default hidden-string task contract must be valid")
    }
}

impl HiddenStringTask {
    pub fn new(config: HiddenStringTaskConfig) -> Result<Self> {
        config.validate()?;
        let panels = Arc::new(build_target_panels(&config)?);
        Ok(Self { config, panels })
    }

    pub fn target_panel_summaries(
        &self,
    ) -> (TargetPanelSummary, TargetPanelSummary, TargetPanelSummary) {
        (
            summarize_panel(
                &self.panels.training,
                self.config.training_rollout_seeds.len(),
            ),
            summarize_panel(
                &self.panels.development,
                self.config.development_rollout_seeds.len(),
            ),
            summarize_panel(&self.panels.sealed, self.config.sealed_rollout_seeds.len()),
        )
    }

    pub fn evaluate_frozen_conditions(
        &self,
        genome: &OrganismGenome,
        panel: &str,
        conditions: &[HiddenStringCondition],
    ) -> Result<Vec<HiddenStringConditionEvaluation>> {
        let (targets, rollout_seeds) = match panel {
            "training" => (
                self.panels.training.as_slice(),
                self.config.training_rollout_seeds.as_slice(),
            ),
            "development" => (
                self.panels.development.as_slice(),
                self.config.development_rollout_seeds.as_slice(),
            ),
            "sealed" => (
                self.panels.sealed.as_slice(),
                self.config.sealed_rollout_seeds.as_slice(),
            ),
            other => bail!("unknown hidden-string target panel `{other}`"),
        };
        Ok(conditions
            .iter()
            .map(|&condition| HiddenStringConditionEvaluation {
                panel: panel.to_owned(),
                condition,
                metrics: self.run_condition(
                    genome,
                    targets,
                    rollout_seeds,
                    &self.config.report_probe_after_attempts,
                    condition.into(),
                ),
            })
            .collect())
    }
}

impl EvaluationTask for HiddenStringTask {
    type Config = HiddenStringTaskConfig;
    type Evaluation = HiddenStringEvaluation;

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
        Ok(self.evaluate_cohort(
            genome,
            "training",
            &self.panels.training,
            &self.config.training_rollout_seeds,
            &self.config.training_probe_after_attempts,
            false,
        ))
    }

    fn fitness(&self, evaluation: &Self::Evaluation) -> f64 {
        evaluation.fitness
    }

    fn normalized_fitness(&self, evaluation: &Self::Evaluation) -> Option<f64> {
        Some(evaluation.primary.final_exact_string_rate)
    }

    fn work_report(&self, evaluation: &Self::Evaluation) -> crate::TaskWorkReport {
        let controls = evaluation.controls.as_ref().map_or(0, |controls| {
            controls
                .shuffled_reward
                .brain_synapse_operations
                .saturating_add(controls.reset_weights_each_attempt.brain_synapse_operations)
        });
        crate::TaskWorkReport {
            brain_synapse_operations: evaluation
                .primary
                .brain_synapse_operations
                .saturating_add(controls),
        }
    }

    fn sensor_enabled(&self, _sensor: SensoryReceptor) -> bool {
        false
    }

    fn action_enabled(&self, symbol: Symbol) -> bool {
        symbol != Symbol::End
    }

    fn prepare_founder_genome(&self, genome: &mut OrganismGenome) -> Result<()> {
        let hidden = seed_hidden_gene_node_id(0);
        genome.brain.hidden_nodes = vec![HiddenNodeGene {
            id: hidden,
            bias: 0.0,
            log_time_constant: 0.0,
        }];
        genome.brain.edges = ACTIVE_ACTIONS
            .into_iter()
            .map(|symbol| SynapseGene {
                innovation: connection_innovation_id(
                    hidden,
                    action_gene_node_id(symbol.index()),
                    SynapseTiming::CurrentTick,
                ),
                pre_node_id: hidden,
                post_node_id: action_gene_node_id(symbol.index()),
                timing: SynapseTiming::CurrentTick,
                weight: 1.0,
                enabled: true,
            })
            .chain(std::iter::once(SynapseGene {
                innovation: connection_innovation_id(hidden, hidden, SynapseTiming::PreviousTick),
                pre_node_id: hidden,
                post_node_id: hidden,
                timing: SynapseTiming::PreviousTick,
                weight: 1.0,
                enabled: true,
            }))
            .collect();
        Ok(())
    }

    fn validation_due(&self, generation: u32, total_generations: u32) -> bool {
        generation + 1 == total_generations
            || (generation + 1).is_multiple_of(self.config.development_interval)
    }

    fn validation_evaluation(&self, genome: &OrganismGenome) -> Result<Option<Self::Evaluation>> {
        Ok(Some(self.evaluate_cohort(
            genome,
            "development",
            &self.panels.development,
            &self.config.development_rollout_seeds,
            &self.config.report_probe_after_attempts,
            false,
        )))
    }

    fn final_evaluation(&self, genome: &OrganismGenome) -> Result<Option<Self::Evaluation>> {
        Ok(Some(self.evaluate_cohort(
            genome,
            "sealed",
            &self.panels.sealed,
            &self.config.sealed_rollout_seeds,
            &self.config.report_probe_after_attempts,
            true,
        )))
    }
}

#[derive(Clone, Copy)]
enum LearningMode {
    Primary,
    PlasticityOff,
    ShuffledReward,
    ResetWeights,
}

impl From<HiddenStringCondition> for LearningMode {
    fn from(condition: HiddenStringCondition) -> Self {
        match condition {
            HiddenStringCondition::Primary => Self::Primary,
            HiddenStringCondition::PlasticityOff => Self::PlasticityOff,
            HiddenStringCondition::ShuffledReward => Self::ShuffledReward,
            HiddenStringCondition::ResetWeightsEachAttempt => Self::ResetWeights,
        }
    }
}

impl HiddenStringTask {
    fn evaluate_cohort(
        &self,
        genome: &OrganismGenome,
        cohort_name: &str,
        targets: &[[Symbol; TARGET_LEN]],
        rollout_seeds: &[u64],
        probe_schedule: &[u32],
        include_controls: bool,
    ) -> HiddenStringEvaluation {
        let primary = self.run_condition(
            genome,
            targets,
            rollout_seeds,
            probe_schedule,
            LearningMode::Primary,
        );
        let fitness = primary.final_greedy_prefix_score;
        let controls = include_controls.then(|| {
            let final_probe = [self.config.attempts];
            AdaptationControls {
                shuffled_reward: self.run_condition(
                    genome,
                    targets,
                    rollout_seeds,
                    &final_probe,
                    LearningMode::ShuffledReward,
                ),
                reset_weights_each_attempt: self.run_condition(
                    genome,
                    targets,
                    rollout_seeds,
                    &final_probe,
                    LearningMode::ResetWeights,
                ),
            }
        });
        HiddenStringEvaluation {
            cohort: cohort_name.to_owned(),
            fitness,
            learning_rate: genome.plasticity.hebb_eta_gain,
            primary,
            controls,
        }
    }

    fn run_condition(
        &self,
        genome: &OrganismGenome,
        targets: &[[Symbol; TARGET_LEN]],
        rollout_seeds: &[u64],
        probe_schedule: &[u32],
        mode: LearningMode,
    ) -> AdaptationMetrics {
        let mut correct = vec![0_u64; probe_schedule.len()];
        let mut exact = vec![0_u64; probe_schedule.len()];
        let mut prefix_correct = vec![0_u64; probe_schedule.len()];
        let mut brain_synapse_operations = 0_u64;
        let cases = targets.len() as u64 * rollout_seeds.len() as u64;
        let inherited_brain = express_genome(genome);
        let mut brain = inherited_brain.clone();
        let mut scratch = BrainScratch::new();

        for target in targets {
            for &rollout_seed in rollout_seeds {
                brain.clone_from(&inherited_brain);
                let mut probe_index = 0_usize;
                if probe_schedule.first() == Some(&0) {
                    brain_synapse_operations =
                        brain_synapse_operations.saturating_add(self.record_probe(
                            &mut brain,
                            genome,
                            target,
                            &mut scratch,
                            &mut correct[probe_index],
                            &mut exact[probe_index],
                            &mut prefix_correct[probe_index],
                        ));
                    probe_index += 1;
                }

                for attempt in 1..=self.config.attempts {
                    if matches!(mode, LearningMode::ResetWeights) {
                        brain.clone_from(&inherited_brain);
                    } else {
                        reset_dynamics_preserving_weights(&mut brain);
                    }
                    for position in 0..TARGET_LEN {
                        zero_sensors(&mut brain);
                        let evaluation = evaluate_brain_state(
                            &mut brain,
                            genome,
                            BrainEvalContext {
                                leaky_neurons_enabled: false,
                                action_temperature: 1.0,
                                action_sample: None,
                            },
                            &mut scratch,
                        );
                        brain_synapse_operations =
                            brain_synapse_operations.saturating_add(evaluation.synapse_ops);
                        let probabilities = body_action_probabilities(
                            evaluation.action_logits,
                            self.config.exploration_temperature,
                        );
                        let sample = deterministic_sample(
                            rollout_seed,
                            encode_target(*target),
                            attempt,
                            position as u32,
                        );
                        let selected = sample_body_action(probabilities, sample);
                        let rewarded_target = if matches!(mode, LearningMode::ShuffledReward) {
                            permuted_symbol(target[position])
                        } else {
                            target[position]
                        };
                        let reward = if selected == rewarded_target {
                            self.config.reward_correct
                        } else {
                            self.config.reward_incorrect
                        };
                        if !matches!(mode, LearningMode::PlasticityOff) {
                            apply_immediate_action_reward(
                                &mut brain,
                                selected,
                                probabilities,
                                reward,
                                genome.plasticity.hebb_eta_gain,
                                genome.plasticity.max_weight_delta_per_tick,
                            );
                        }
                    }

                    if probe_schedule.get(probe_index) == Some(&attempt) {
                        brain_synapse_operations =
                            brain_synapse_operations.saturating_add(self.record_probe(
                                &mut brain,
                                genome,
                                target,
                                &mut scratch,
                                &mut correct[probe_index],
                                &mut exact[probe_index],
                                &mut prefix_correct[probe_index],
                            ));
                        probe_index += 1;
                    }
                }
                debug_assert_eq!(probe_index, probe_schedule.len());
            }
        }

        let probes = probe_schedule
            .iter()
            .enumerate()
            .map(|(index, &after_attempts)| ProbeMetrics {
                after_attempts,
                accuracy: correct[index] as f64 / (cases * TARGET_LEN as u64) as f64,
                exact_string_rate: exact[index] as f64 / cases as f64,
                greedy_prefix_score: prefix_correct[index] as f64
                    / (cases * TARGET_LEN as u64) as f64,
            })
            .collect::<Vec<_>>();
        let pre_learning_accuracy = probes
            .iter()
            .find(|probe| probe.after_attempts == 0)
            .map(|probe| probe.accuracy);
        let final_probe = probes.last().expect("validated probe schedule is nonempty");
        AdaptationMetrics {
            target_count: targets.len(),
            rollout_count: rollout_seeds.len() as u32,
            pre_learning_accuracy,
            final_accuracy: final_probe.accuracy,
            final_exact_string_rate: final_probe.exact_string_rate,
            final_greedy_prefix_score: final_probe.greedy_prefix_score,
            brain_synapse_operations,
            probes,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_probe(
        &self,
        brain: &mut types::BrainState,
        genome: &OrganismGenome,
        target: &[Symbol; TARGET_LEN],
        scratch: &mut BrainScratch,
        correct: &mut u64,
        exact: &mut u64,
        prefix_correct: &mut u64,
    ) -> u64 {
        reset_dynamics_preserving_weights(brain);
        let mut case_correct = 0_u64;
        let mut case_prefix_correct = 0_u64;
        let mut prefix_open = true;
        let mut brain_synapse_operations = 0_u64;
        for &expected in target {
            zero_sensors(brain);
            let evaluation = evaluate_brain_state(
                brain,
                genome,
                BrainEvalContext {
                    leaky_neurons_enabled: false,
                    action_temperature: 1.0,
                    action_sample: None,
                },
                scratch,
            );
            brain_synapse_operations =
                brain_synapse_operations.saturating_add(evaluation.synapse_ops);
            let selected = argmax_body_action(evaluation.action_logits);
            let matches = selected == expected;
            case_correct += u64::from(matches);
            if prefix_open {
                if matches {
                    case_prefix_correct += 1;
                } else {
                    prefix_open = false;
                }
            }
        }
        *correct += case_correct;
        *exact += u64::from(case_correct == TARGET_LEN as u64);
        *prefix_correct += case_prefix_correct;
        brain_synapse_operations
    }
}

#[derive(Debug)]
struct TargetPanels {
    training: Vec<[Symbol; TARGET_LEN]>,
    development: Vec<[Symbol; TARGET_LEN]>,
    sealed: Vec<[Symbol; TARGET_LEN]>,
}

fn build_target_panels(config: &HiddenStringTaskConfig) -> Result<TargetPanels> {
    let mut orbit_buckets = [Vec::new(), Vec::new(), Vec::new()];
    for b in 0..ACTIVE_ACTIONS.len() as u8 {
        for c in 0..ACTIVE_ACTIONS.len() as u8 {
            for d in 0..ACTIVE_ACTIONS.len() as u8 {
                let orbit = [0, b, c, d];
                let distinct = distinct_symbol_count(orbit);
                if distinct >= 2 {
                    orbit_buckets[distinct - 2].push(orbit);
                }
            }
        }
    }
    for (bucket_index, bucket) in orbit_buckets.iter_mut().enumerate() {
        let domain = (bucket_index as u64 + 2).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        bucket.sort_unstable_by_key(|orbit| {
            (
                mix64(config.target_panel_seed ^ domain ^ encode_orbit(*orbit)),
                encode_orbit(*orbit),
            )
        });
    }

    let counts = [
        config.training_target_count,
        config.development_target_count,
        config.sealed_target_count,
    ];
    let mut offsets = [0_usize; 3];
    let mut panels = Vec::with_capacity(3);
    for (panel_index, target_count) in counts.into_iter().enumerate() {
        let allocation = orbit_allocation(target_count)?;
        let mut selected = Vec::with_capacity(target_count / SYMBOLS_PER_ORBIT);
        for bucket_index in 0..3 {
            let start = offsets[bucket_index];
            let end = start + allocation[bucket_index];
            let bucket = &orbit_buckets[bucket_index];
            if end > bucket.len() {
                bail!(
                    "target panels exhaust the available {}-distinct-symbol orbits",
                    bucket_index + 2
                );
            }
            selected.extend_from_slice(&bucket[start..end]);
            offsets[bucket_index] = end;
        }
        let mut targets = expand_orbits(&selected);
        let domain = (panel_index as u64 + 1).wrapping_mul(0xd6e8_feb8_6659_fd93);
        targets.sort_unstable_by_key(|target| {
            (
                mix64(config.target_panel_seed ^ domain ^ encode_target(*target)),
                encode_target(*target),
            )
        });
        panels.push(targets);
    }
    let mut panels = panels.into_iter();
    Ok(TargetPanels {
        training: panels.next().expect("training panel"),
        development: panels.next().expect("development panel"),
        sealed: panels.next().expect("sealed panel"),
    })
}

fn orbit_allocation(target_count: usize) -> Result<[usize; 3]> {
    if !target_count.is_multiple_of(SYMBOLS_PER_ORBIT) {
        bail!("target count must be a multiple of 8");
    }
    let orbit_count = target_count / SYMBOLS_PER_ORBIT;
    let weight_sum = DISTINCT_COUNT_WEIGHTS.iter().sum::<usize>();
    let mut allocation = [0_usize; 3];
    let mut remainders = [0_usize; 3];
    for index in 0..3 {
        let scaled = orbit_count * DISTINCT_COUNT_WEIGHTS[index];
        allocation[index] = scaled / weight_sum;
        remainders[index] = scaled % weight_sum;
    }
    while allocation.iter().sum::<usize>() < orbit_count {
        let index = (0..3)
            .max_by_key(|&candidate| (remainders[candidate], candidate))
            .expect("three allocation buckets");
        allocation[index] += 1;
        remainders[index] = 0;
    }
    Ok(allocation)
}

fn expand_orbits(orbits: &[[u8; TARGET_LEN]]) -> Vec<[Symbol; TARGET_LEN]> {
    let mut targets = Vec::with_capacity(orbits.len() * SYMBOLS_PER_ORBIT);
    for orbit in orbits {
        for offset in 0..ACTIVE_ACTIONS.len() as u8 {
            targets.push(std::array::from_fn(|position| {
                ACTIVE_ACTIONS[((orbit[position] + offset) % ACTIVE_ACTIONS.len() as u8) as usize]
            }));
        }
    }
    targets
}

fn summarize_panel(targets: &[[Symbol; TARGET_LEN]], rollout_count: usize) -> TargetPanelSummary {
    let mut distinct_symbol_counts = [0_usize; TARGET_LEN + 1];
    let mut position_symbol_counts = [[0_usize; ACTIVE_ACTIONS.len()]; TARGET_LEN];
    for target in targets {
        let encoded = target.map(|symbol| symbol.index() as u8);
        distinct_symbol_counts[distinct_symbol_count(encoded)] += 1;
        for (position, symbol) in target.iter().enumerate() {
            position_symbol_counts[position][symbol.index()] += 1;
        }
    }
    TargetPanelSummary {
        target_count: targets.len(),
        rollout_count,
        distinct_symbol_counts,
        position_symbol_counts,
    }
}

fn distinct_symbol_count(symbols: [u8; TARGET_LEN]) -> usize {
    let mut seen = [false; ACTIVE_ACTIONS.len()];
    for symbol in symbols {
        seen[symbol as usize] = true;
    }
    seen.into_iter().filter(|present| *present).count()
}

fn encode_orbit(orbit: [u8; TARGET_LEN]) -> u64 {
    orbit
        .into_iter()
        .fold(0_u64, |encoded, symbol| (encoded << 3) | u64::from(symbol))
}

fn encode_target(target: [Symbol; TARGET_LEN]) -> u64 {
    target.into_iter().fold(0_u64, |encoded, symbol| {
        (encoded << 3) | symbol.index() as u64
    })
}

fn permuted_symbol(symbol: Symbol) -> Symbol {
    let index = ACTIVE_ACTIONS
        .iter()
        .position(|candidate| *candidate == symbol)
        .expect("hidden-string targets use the active alphabet");
    ACTIVE_ACTIONS[(index + 1) % ACTIVE_ACTIONS.len()]
}

fn zero_sensors(brain: &mut types::BrainState) {
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = 0.0;
    }
}

fn body_action_probabilities(
    logits: [f32; Symbol::COUNT],
    temperature: f32,
) -> [f32; Symbol::COUNT] {
    let mut probabilities = [0.0_f32; Symbol::COUNT];
    let inv_temperature = temperature.recip();
    let max_logit = ACTIVE_ACTIONS
        .iter()
        .map(|symbol| logits[symbol.index()] * inv_temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;
    for symbol in ACTIVE_ACTIONS {
        let value = (logits[symbol.index()] * inv_temperature - max_logit).exp();
        probabilities[symbol.index()] = value;
        total += value;
    }
    for symbol in ACTIVE_ACTIONS {
        probabilities[symbol.index()] /= total;
    }
    probabilities
}

fn sample_body_action(probabilities: [f32; Symbol::COUNT], sample: f32) -> Symbol {
    let mut cumulative = 0.0_f32;
    for symbol in ACTIVE_ACTIONS {
        cumulative += probabilities[symbol.index()];
        if sample < cumulative {
            return symbol;
        }
    }
    *ACTIVE_ACTIONS
        .last()
        .expect("body action alphabet is nonempty")
}

fn argmax_body_action(logits: [f32; Symbol::COUNT]) -> Symbol {
    ACTIVE_ACTIONS
        .into_iter()
        .max_by(|left, right| {
            logits[left.index()]
                .total_cmp(&logits[right.index()])
                .then_with(|| right.index().cmp(&left.index()))
        })
        .expect("body action alphabet is nonempty")
}

fn deterministic_sample(seed: u64, target: u64, attempt: u32, position: u32) -> f32 {
    let bits = mix64(
        seed ^ target.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (u64::from(attempt) << 32)
            ^ u64::from(position),
    );
    ((bits >> 40) as f32) / ((1_u32 << 24) as f32)
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
