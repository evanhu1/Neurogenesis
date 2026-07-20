use crate::{
    GenomeTask, ResourceEcologyTask, ResourceLifetimeContext, ResourceLifetimeOutcome,
    TaskWorkReport,
};
use anyhow::{bail, Result};
use brain::{
    apply_immediate_action_reward, apply_target_prediction_error, apply_temporal_action_reward,
    evaluate_brain_state, express_genome, reset_dynamics_preserving_weights, BrainEvalContext,
    BrainScratch, ImmediateLearningNormalization, ImmediateLearningRequest,
    TargetPredictionLearningRequest, TemporalLearningRequest,
};
use serde::{Deserialize, Serialize};
use task_library::SymbolicTask;
use types::{BrainState, OrganismGenome, SensoryReceptor, Symbol};

const TRAINING_DOMAIN: u64 = 0x5453_4b45_434f_5452;
const DEVELOPMENT_DOMAIN: u64 = 0x4445_5645_4c4f_504d;
const SEALED_DOMAIN: u64 = 0x5345_414c_4544_5f54;
const ACTION_DOMAIN: u64 = 0x4143_5449_4f4e_4452;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningRule {
    Disabled,
    ImmediatePolicy,
    TargetPredictionError,
    #[default]
    TemporalPredictionError,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionSelection {
    Greedy,
    #[default]
    Sampled,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningNormalization {
    #[default]
    None,
    Nlms,
}

impl From<LearningNormalization> for ImmediateLearningNormalization {
    fn from(value: LearningNormalization) -> Self {
        match value {
            LearningNormalization::None => Self::None,
            LearningNormalization::Nlms => Self::NormalizedLeastMeanSquares,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvaluationConfig {
    pub training_instances: usize,
    pub development_instances: usize,
    pub sealed_instances: usize,
    pub training_rollouts: usize,
    pub development_rollouts: usize,
    pub sealed_rollouts: usize,
    pub learning_rule: LearningRule,
    pub action_selection: ActionSelection,
    pub exploration_temperature: f32,
    pub learning_normalization: LearningNormalization,
    pub reset_dynamics_at_trial_boundary: bool,
    pub audit_interval: u32,
}

impl Default for AgentEvaluationConfig {
    fn default() -> Self {
        Self {
            training_instances: 64,
            development_instances: 64,
            sealed_instances: 64,
            training_rollouts: 1,
            development_rollouts: 1,
            sealed_rollouts: 1,
            learning_rule: LearningRule::TemporalPredictionError,
            action_selection: ActionSelection::Sampled,
            exploration_temperature: 1.0,
            learning_normalization: LearningNormalization::None,
            reset_dynamics_at_trial_boundary: true,
            audit_interval: 25,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskEcologyConfig<C> {
    pub task: C,
    pub agent: AgentEvaluationConfig,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SymbolicEcologyMetrics {
    pub instances: usize,
    pub ticks: u64,
    pub correct: u64,
    pub accuracy: f64,
    pub learning_ticks: u64,
    pub learning_correct: u64,
    pub learning_accuracy: f64,
    pub probe_ticks: u64,
    pub probe_correct: u64,
    pub probe_accuracy: f64,
    pub mean_probe_target_probability: f64,
    pub mean_probe_sequence_probability: f64,
    pub completed_trials: u64,
    pub successful_trials: u64,
    pub trial_success_rate: f64,
    pub resource_units: u64,
    pub resource_throughput_per_1000_ticks: f64,
    pub mean_reward: f64,
    pub mean_absolute_prediction_error: f64,
    pub mean_reward_prediction: f64,
    pub mean_absolute_applied_delta: f64,
    pub clipped_update_count: u64,
    pub edge_update_count: u64,
    pub brain_synapse_operations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicEcologyAudit {
    pub cohort: String,
    pub primary: SymbolicEcologyMetrics,
    pub efference_copy_off: SymbolicEcologyMetrics,
    pub prediction_error_feedback_off: SymbolicEcologyMetrics,
}

#[derive(Clone)]
pub struct TaskEcology<T> {
    pub task: T,
    pub agent: AgentEvaluationConfig,
}
impl<T> TaskEcology<T> {
    pub fn new(task: T, agent: AgentEvaluationConfig) -> Self {
        Self { task, agent }
    }
}

struct Instance<S> {
    task: S,
    brain: BrainState,
    sample_seed: u64,
    sample_tick: u64,
}
pub struct TaskEvaluationState<S> {
    instances: Vec<Instance<S>>,
}
#[derive(Clone, Copy)]
enum Condition {
    Primary,
    EfferenceCopyOff,
    PredictionErrorFeedbackOff,
}

impl<T: SymbolicTask> GenomeTask for TaskEcology<T> {
    fn sensor_enabled(&self, _sensor: SensoryReceptor) -> bool {
        self.task.observes_symbols()
    }
    fn action_enabled(&self, symbol: Symbol) -> bool {
        self.task.action_enabled(symbol)
    }
    fn action_feedback_enabled(&self) -> bool {
        true
    }
    fn temporal_credit_enabled(&self) -> bool {
        !matches!(self.agent.learning_rule, LearningRule::Disabled)
    }
    fn value_prediction_enabled(&self) -> bool {
        matches!(
            self.agent.learning_rule,
            LearningRule::TemporalPredictionError
        )
    }
    fn lifetime_learning_enabled(&self) -> bool {
        !matches!(self.agent.learning_rule, LearningRule::Disabled)
    }
}

impl<T: SymbolicTask + Clone> ResourceEcologyTask for TaskEcology<T> {
    type Config = TaskEcologyConfig<T::Config>;
    type LifetimeState = TaskEvaluationState<T::State>;
    type LifetimeEvaluation = SymbolicEcologyMetrics;
    type AuditEvaluation = SymbolicEcologyAudit;

    fn name(&self) -> &'static str {
        self.task.name()
    }
    fn objective(&self) -> &'static str {
        "finite_task_resource_capture"
    }
    fn config(&self) -> Self::Config {
        TaskEcologyConfig {
            task: self.task.config(),
            agent: self.agent.clone(),
        }
    }
    fn lifetime_ticks(&self) -> usize {
        self.task.max_steps_per_instance() + self.task.probe_steps_per_instance()
    }
    fn evaluation_lifetimes(&self) -> usize {
        self.agent.training_instances * self.agent.training_rollouts
    }
    fn validate(&self) -> Result<()> {
        self.task.validate()?;
        if !self.agent.exploration_temperature.is_finite()
            || self.agent.exploration_temperature <= 0.0
        {
            bail!("exploration temperature must be finite and positive");
        }
        if self.agent.audit_interval == 0 {
            bail!("audit interval must be positive");
        }
        if self.agent.training_instances == 0
            || self.agent.development_instances == 0
            || self.agent.sealed_instances == 0
            || self.agent.training_rollouts == 0
            || self.agent.development_rollouts == 0
            || self.agent.sealed_rollouts == 0
        {
            bail!("panel instance and rollout counts must be positive");
        }
        Ok(())
    }
    fn initialize_lifetime(
        &self,
        genome: &OrganismGenome,
        _individual_id: u64,
        run_seed: u64,
        _generation: u32,
    ) -> Result<Self::LifetimeState> {
        // The benchmark panel is fixed across generations. Population members
        // still share all stochastic draws within a generation, but evolution
        // cannot be ranked on a moving target distribution.
        let panel_seed = mix64(run_seed ^ TRAINING_DOMAIN);
        Ok(TaskEvaluationState {
            instances: (0..self.agent.training_instances)
                .flat_map(|index| {
                    (0..self.agent.training_rollouts).map(move |rollout| (index, rollout))
                })
                .map(|(index, rollout)| Instance {
                    task: self.task.start(panel_seed, index),
                    brain: express_genome(genome),
                    sample_seed: mix64(
                        panel_seed
                            ^ ACTION_DOMAIN
                            ^ (index as u64).rotate_left(17)
                            ^ rollout as u64,
                    ),
                    sample_tick: 0,
                })
                .collect(),
        })
    }
    fn evaluate_lifetime(
        &self,
        genome: &OrganismGenome,
        state: &mut Self::LifetimeState,
        _context: ResourceLifetimeContext,
    ) -> Result<ResourceLifetimeOutcome<Self::LifetimeEvaluation>> {
        let metrics = self.evaluate_instances(genome, &mut state.instances, Condition::Primary);
        Ok(ResourceLifetimeOutcome {
            reproductive_tickets: metrics.resource_units,
            work: TaskWorkReport {
                brain_synapse_operations: metrics.brain_synapse_operations,
            },
            evaluation: metrics,
        })
    }
    fn audit(
        &self,
        genome: &OrganismGenome,
        cohort: &str,
        audit_seed: u64,
    ) -> Result<Self::AuditEvaluation> {
        let (domain, instance_count, rollout_count) = if cohort == "sealed" {
            (
                SEALED_DOMAIN,
                self.agent.sealed_instances,
                self.agent.sealed_rollouts,
            )
        } else {
            (
                DEVELOPMENT_DOMAIN,
                self.agent.development_instances,
                self.agent.development_rollouts,
            )
        };
        let evaluate = |condition| {
            let panel_seed = mix64(audit_seed ^ domain);
            let mut instances = (0..instance_count)
                .flat_map(|index| (0..rollout_count).map(move |rollout| (index, rollout)))
                .map(|(index, rollout)| Instance {
                    task: self.task.start(panel_seed, index),
                    brain: express_genome(genome),
                    sample_seed: mix64(
                        panel_seed
                            ^ ACTION_DOMAIN
                            ^ (index as u64).rotate_left(17)
                            ^ rollout as u64,
                    ),
                    sample_tick: 0,
                })
                .collect::<Vec<_>>();
            self.evaluate_instances(genome, &mut instances, condition)
        };
        Ok(SymbolicEcologyAudit {
            cohort: cohort.to_owned(),
            primary: evaluate(Condition::Primary),
            efference_copy_off: evaluate(Condition::EfferenceCopyOff),
            prediction_error_feedback_off: evaluate(Condition::PredictionErrorFeedbackOff),
        })
    }
    fn audit_score(&self, audit: &Self::AuditEvaluation) -> f64 {
        audit.primary.accuracy
    }
    fn audit_due(&self, generation: u32, total_generations: u32) -> bool {
        generation + 1 == total_generations
            || (generation + 1).is_multiple_of(self.agent.audit_interval)
    }
}

impl<T: SymbolicTask> TaskEcology<T> {
    fn evaluate_instances(
        &self,
        genome: &OrganismGenome,
        instances: &mut [Instance<T::State>],
        condition: Condition,
    ) -> SymbolicEcologyMetrics {
        let mut metrics = SymbolicEcologyMetrics {
            instances: instances.len(),
            ..Default::default()
        };
        let (mut rewards, mut errors, mut predictions, mut deltas) = (0.0, 0.0, 0.0, 0.0);
        let mut probe_sequence_probability_sum = 0.0;
        for instance in instances {
            let mut scratch = BrainScratch::new();
            let mut immediate_update_scratch = Vec::new();
            for _ in 0..self.task.max_steps_per_instance() {
                apply_observation(&mut instance.brain, self.task.observe(&instance.task));
                let brain_eval = evaluate_brain_state(
                    &mut instance.brain,
                    genome,
                    BrainEvalContext {
                        leaky_neurons_enabled: false,
                        action_temperature: 1.0,
                        action_sample: None,
                    },
                    &mut scratch,
                );
                metrics.brain_synapse_operations += brain_eval.synapse_ops;
                let probabilities = action_probabilities(
                    &self.task,
                    brain_eval.action_logits,
                    self.agent.exploration_temperature * genome.plasticity.action_temperature_scale,
                );
                let selected = match self.agent.action_selection {
                    ActionSelection::Greedy => argmax_action(&self.task, brain_eval.action_logits),
                    ActionSelection::Sampled => sample_action(
                        &self.task,
                        probabilities,
                        deterministic_sample(instance.sample_seed, instance.sample_tick),
                    ),
                };
                let transition = self.task.step(&mut instance.task, selected);
                let (edge_updates, clipped_updates, applied_delta, prediction_error, prediction) =
                    match self.agent.learning_rule {
                        LearningRule::Disabled => {
                            instance.brain.previous_action_activations.fill(0.0);
                            instance.brain.previous_action_activations[selected.index()] = 1.0;
                            instance.brain.previous_prediction_error = 0.0;
                            (0, 0, 0.0, 0.0, 0.0)
                        }
                        LearningRule::ImmediatePolicy => {
                            let report = apply_immediate_action_reward(
                                &mut instance.brain,
                                ImmediateLearningRequest {
                                    selected,
                                    action_probabilities: probabilities,
                                    reward: transition.reward,
                                    learning_rate: genome.plasticity.initial_learning_rate,
                                    fast_weight_retention: genome.plasticity.fast_weight_retention,
                                    max_weight_delta: genome.plasticity.max_weight_delta_per_tick,
                                    normalization: self.agent.learning_normalization.into(),
                                },
                                &mut immediate_update_scratch,
                            );
                            instance.brain.previous_action_activations.fill(0.0);
                            instance.brain.previous_action_activations[selected.index()] = 1.0;
                            instance.brain.previous_prediction_error = transition.reward;
                            (
                                report.edge_update_count,
                                report.clipped_update_count,
                                report.applied_absolute_delta,
                                transition.reward,
                                0.0,
                            )
                        }
                        LearningRule::TargetPredictionError => {
                            let target = transition.expected_action.expect(
                                "target-prediction learning requires an expected task action",
                            );
                            let report = apply_target_prediction_error(
                                &mut instance.brain,
                                TargetPredictionLearningRequest {
                                    target,
                                    action_probabilities: probabilities,
                                    learning_rate: genome.plasticity.initial_learning_rate,
                                    fast_weight_retention: genome.plasticity.fast_weight_retention,
                                    max_weight_delta: genome.plasticity.max_weight_delta_per_tick,
                                    normalization: self.agent.learning_normalization.into(),
                                },
                            );
                            let target_error = 1.0 - probabilities[target.index()];
                            instance.brain.previous_action_activations.fill(0.0);
                            instance.brain.previous_action_activations[selected.index()] = 1.0;
                            instance.brain.previous_prediction_error = target_error;
                            (
                                report.edge_update_count,
                                report.clipped_update_count,
                                report.applied_absolute_delta,
                                target_error,
                                probabilities[target.index()],
                            )
                        }
                        LearningRule::TemporalPredictionError => {
                            let report = apply_temporal_action_reward(
                                &mut instance.brain,
                                TemporalLearningRequest {
                                    selected,
                                    reward: transition.reward,
                                    value_prediction: brain_eval.value_prediction,
                                    learning_rate: genome.plasticity.initial_learning_rate,
                                    eligibility_retention: genome.plasticity.eligibility_retention,
                                    fast_weight_retention: genome.plasticity.fast_weight_retention,
                                    max_weight_delta: genome.plasticity.max_weight_delta_per_tick,
                                    normalization: self.agent.learning_normalization.into(),
                                    plasticity_enabled: true,
                                },
                            );
                            (
                                report.edge_update_count,
                                report.clipped_update_count,
                                report.applied_absolute_delta,
                                report.prediction_error,
                                report.reward_prediction,
                            )
                        }
                    };
                if matches!(condition, Condition::EfferenceCopyOff) {
                    instance.brain.previous_action_activations.fill(0.0);
                }
                if matches!(condition, Condition::PredictionErrorFeedbackOff) {
                    instance.brain.previous_prediction_error = 0.0;
                }
                metrics.ticks += 1;
                metrics.learning_ticks += 1;
                metrics.learning_correct += u64::from(transition.correct);
                if self.task.probe_steps_per_instance() == 0 {
                    metrics.correct += u64::from(transition.correct);
                    metrics.resource_units += u64::from(transition.success_events);
                    if let Some(successful) = transition.trial_outcome {
                        metrics.completed_trials += 1;
                        metrics.successful_trials += u64::from(successful);
                    }
                }
                metrics.edge_update_count += edge_updates;
                metrics.clipped_update_count += clipped_updates;
                rewards += f64::from(transition.reward);
                errors += f64::from(prediction_error.abs());
                predictions += f64::from(prediction);
                deltas += applied_delta;
                instance.sample_tick += 1;
                if transition.trial_outcome.is_some() && self.agent.reset_dynamics_at_trial_boundary
                {
                    reset_dynamics_preserving_weights(&mut instance.brain);
                }
                if transition.done {
                    break;
                }
            }

            let probe_steps = self.task.probe_steps_per_instance();
            if probe_steps > 0 {
                reset_dynamics_preserving_weights(&mut instance.brain);
                self.task.begin_probe(&mut instance.task);
                let mut sequence_probability = 1.0;
                for _ in 0..probe_steps {
                    apply_observation(&mut instance.brain, self.task.probe_observe(&instance.task));
                    let brain_eval = evaluate_brain_state(
                        &mut instance.brain,
                        genome,
                        BrainEvalContext {
                            leaky_neurons_enabled: false,
                            action_temperature: 1.0,
                            action_sample: None,
                        },
                        &mut scratch,
                    );
                    metrics.brain_synapse_operations += brain_eval.synapse_ops;
                    let probabilities = action_probabilities(
                        &self.task,
                        brain_eval.action_logits,
                        self.agent.exploration_temperature
                            * genome.plasticity.action_temperature_scale,
                    );
                    let selected = argmax_action(&self.task, brain_eval.action_logits);
                    let transition = self.task.probe_step(&mut instance.task, selected);
                    if let Some(expected) = transition.expected_action {
                        sequence_probability *= f64::from(probabilities[expected.index()]);
                        metrics.mean_probe_target_probability +=
                            f64::from(probabilities[expected.index()]);
                    }
                    metrics.ticks += 1;
                    metrics.probe_ticks += 1;
                    metrics.probe_correct += u64::from(transition.correct);
                    metrics.correct += u64::from(transition.correct);
                    metrics.resource_units += u64::from(transition.success_events);
                    if let Some(successful) = transition.trial_outcome {
                        metrics.completed_trials += 1;
                        metrics.successful_trials += u64::from(successful);
                    }
                    if transition.done {
                        break;
                    }
                }
                probe_sequence_probability_sum += sequence_probability;
            }
        }
        if metrics.learning_ticks > 0 {
            metrics.learning_accuracy =
                metrics.learning_correct as f64 / metrics.learning_ticks as f64;
            metrics.mean_reward = rewards / metrics.learning_ticks as f64;
            metrics.mean_absolute_prediction_error = errors / metrics.learning_ticks as f64;
            metrics.mean_reward_prediction = predictions / metrics.learning_ticks as f64;
        }
        if metrics.probe_ticks > 0 {
            metrics.probe_accuracy = metrics.probe_correct as f64 / metrics.probe_ticks as f64;
            metrics.mean_probe_target_probability /= metrics.probe_ticks as f64;
            metrics.mean_probe_sequence_probability =
                probe_sequence_probability_sum / metrics.instances as f64;
            metrics.accuracy = metrics.probe_accuracy;
        } else if metrics.learning_ticks > 0 {
            metrics.accuracy = metrics.learning_accuracy;
        }
        if metrics.ticks > 0 {
            metrics.resource_throughput_per_1000_ticks =
                metrics.resource_units as f64 * 1000.0 / metrics.ticks as f64;
        }
        if metrics.completed_trials > 0 {
            metrics.trial_success_rate =
                metrics.successful_trials as f64 / metrics.completed_trials as f64;
        }
        if metrics.edge_update_count > 0 {
            metrics.mean_absolute_applied_delta = deltas / metrics.edge_update_count as f64;
        }
        metrics
    }
}

fn apply_observation(brain: &mut BrainState, observation: task_library::Observation) {
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = match (sensory.receptor, observation.symbol) {
            (SensoryReceptor::Symbol { symbol: receptor }, Some(symbol)) => {
                f32::from(receptor == symbol)
            }
            _ => 0.0,
        };
    }
}

fn argmax_action<T: SymbolicTask>(task: &T, logits: [f32; Symbol::COUNT]) -> Symbol {
    Symbol::ALL
        .into_iter()
        .filter(|action| task.action_enabled(*action))
        .max_by(|left, right| {
            logits[left.index()]
                .total_cmp(&logits[right.index()])
                .then_with(|| right.index().cmp(&left.index()))
        })
        .expect("validated task exposes at least one action")
}

fn action_probabilities<T: SymbolicTask>(
    task: &T,
    logits: [f32; Symbol::COUNT],
    temperature: f32,
) -> [f32; Symbol::COUNT] {
    let mut probabilities = [0.0; Symbol::COUNT];
    let max = Symbol::ALL
        .into_iter()
        .filter(|a| task.action_enabled(*a))
        .map(|a| logits[a.index()] / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for action in Symbol::ALL.into_iter().filter(|a| task.action_enabled(*a)) {
        probabilities[action.index()] = (logits[action.index()] / temperature - max).exp();
        total += probabilities[action.index()];
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn sample_action<T: SymbolicTask>(
    task: &T,
    probabilities: [f32; Symbol::COUNT],
    sample: f32,
) -> Symbol {
    let mut cumulative = 0.0;
    let mut last = Symbol::A;
    for action in Symbol::ALL.into_iter().filter(|a| task.action_enabled(*a)) {
        last = action;
        cumulative += probabilities[action.index()];
        if sample < cumulative {
            return action;
        }
    }
    last
}

fn deterministic_sample(seed: u64, tick: u64) -> f32 {
    let bits = mix64(seed ^ tick);
    ((bits >> 40) as f32 + 0.5) / ((1_u32 << 24) as f32)
}
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
