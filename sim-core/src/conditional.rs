//! Delayed-conditional task-program pilot for progressive neuroevolution.
//!
//! The organism only observes the ordinary food rays.  A task presents a
//! sequence of single-food left/right cues, removes every cue, enforces an
//! empty delay while restoring the same pose, and then presents identical
//! two-food choices.  Correct turns must replay the earlier cue sequence.  The
//! evaluator never writes a task id, context id, target bit, or oracle value
//! into organism-visible state; only the ordinary food geometry differs.
//!
//! This module is intentionally an adversarial vertical pilot, not a claim of
//! open-endedness.  Task rank has no experiment-internal depth cap (the CLI's
//! stage/search budgets bound a run), while controller growth remains subject
//! to the physical `u32` id space and machine memory.

use crate::brain::EXPLICIT_IDLE_LOGIT_BIAS;
use crate::grid::{hex_neighbor, rotate_by_steps};
use crate::progressive::{
    enforce_retention, exact_fingerprint, verify_extension_effect, AllHistoryRetention,
    ExtensionEffectEvidence, ProtectedResidual, RetentionRequirementHeader, TaskCheckpoint,
    TaskReplay,
};
use crate::turn::deterministic_action_sample;
use crate::{SimError, Simulation};
use serde::{Deserialize, Serialize};
use sim_types::{
    action_gene_node_id, connection_innovation_id, food_visual, seed_hidden_gene_node_id,
    sensory_gene_node_id, ActionType, BrainState, EnergyLedgerRow, FacingDirection, FoodId,
    FoodKind, FoodState, GeneNodeId, HiddenNodeGene, InnovationId, Occupant, OrganismGenome,
    SensoryReceptor, SynapseGene, WorldConfig,
};
use std::collections::BTreeSet;
use thiserror::Error;

const PANEL_SIZE: usize = 16;
const MIN_ADMISSION_PASSES: u32 = 14;
const MAX_CONTROL_PASSES: u32 = 2;
const MIN_CAUSAL_DROP: u32 = 8;
const TASK_WORLD_WIDTH: u32 = 15;
const DEFAULT_ESCROW_ENERGY: f32 = 64.0;
const MIN_PRECEDING_ECOLOGY_PLANT_CAPTURES: u64 = 1;
const MIN_NORMAL_RESPONSE_LOGIT_MARGIN: f32 = 0.1;
const EVALUATOR_CONTRACT_VERSION: &str = "conditional-program-evaluator-v3";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalProgramConfig {
    pub outer_seeds: Vec<u64>,
    /// Runtime budget only.  It is not encoded into the task grammar.
    pub stage_budget: u32,
    /// Number of deterministic controller proposals inspected per stage.
    pub search_budget: u32,
    /// Four bits are the minimum needed for sixteen unique replay contexts.
    pub starting_rank: u32,
    pub empty_delay_ticks: u32,
    pub escrow_energy: f32,
    pub ecology_horizon: u32,
}

impl Default for ConditionalProgramConfig {
    fn default() -> Self {
        Self {
            outer_seeds: vec![7, 42, 123],
            stage_budget: 3,
            search_budget: 16,
            starting_rank: 4,
            empty_delay_ticks: 2,
            escrow_energy: DEFAULT_ESCROW_ENERGY,
            ecology_horizon: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalProgramExperiment {
    pub schema: String,
    pub config: ConditionalProgramConfig,
    pub effective_evaluator_configs: EffectiveEvaluatorConfigs,
    pub contract: ExperimentContract,
    pub scope_limitations: Vec<String>,
    pub outer_runs: Vec<OuterRun>,
    pub summary: ExperimentSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EffectiveEvaluatorConfigs {
    pub task_world: WorldConfig,
    pub task_world_fingerprint: String,
    pub random_action_task_world: WorldConfig,
    pub random_action_task_world_fingerprint: String,
    pub ecology_world: WorldConfig,
    pub ecology_world_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentContract {
    pub evaluator_contract_version: String,
    pub organism_observations: Vec<String>,
    pub forbidden_observations: Vec<String>,
    pub pose_reset: String,
    pub energy: String,
    pub search_and_admission: String,
    pub ecology_gate: String,
    pub cue_symbol_equivariance_replication: String,
    pub action_margin_gate: String,
    pub rank_limit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentSummary {
    pub outer_seed_count: usize,
    pub qualifying_discoveries: u32,
    pub maximum_accepted_rank: Option<u32>,
    pub seeds_with_at_least_one_discovery: u32,
    pub terminal_failure_modes: Vec<String>,
    pub open_endedness_demonstrated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OuterRun {
    pub outer_seed: u64,
    pub initial_controller_fingerprint: String,
    pub initial_controller_genome: OrganismGenome,
    pub stages: Vec<StageEvidence>,
    pub accepted_tasks: Vec<AcceptedTaskRecord>,
    pub accepted_solvers: Vec<AcceptedSolverRecord>,
    pub final_crossplay: Vec<MatrixCell>,
    pub final_controller_fingerprint: String,
    pub stopped_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageEvidence {
    pub archive_index: u64,
    pub evaluator_contract_version: String,
    pub task_world_config_fingerprint: String,
    pub random_action_task_world_config_fingerprint: String,
    pub ecology_world_config_fingerprint: String,
    pub ecology_horizon: u32,
    pub task: ConditionalTaskProgram,
    pub task_program_fingerprint: String,
    pub search_contexts_fingerprint: String,
    pub admission_contexts_fingerprint: String,
    /// Composite identity binding the task program and both context panels.
    pub task_fingerprint: String,
    pub semantic_rank: SemanticRankCertificate,
    pub task_frozen_before_solver_search: bool,
    pub search_contexts: Vec<ConditionalContext>,
    pub admission_contexts: Vec<ConditionalContext>,
    pub context_world_seeds_disjoint: bool,
    pub context_evidence_ids_disjoint: bool,
    pub archived_solver_new_task_search: Vec<SolverPanelScore>,
    pub search_attempts: Vec<SearchAttempt>,
    pub admission: Option<AdmissionEvidence>,
    pub accepted: bool,
    pub failure_modes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticRankCertificate {
    /// Property of the complete delayed-copy task grammar, not of the sampled
    /// admission panel.
    pub formal_task_language_history_count: String,
    /// Lower bound for an exact solver over the complete task grammar.
    pub formal_exact_solver_memory_lower_bound_bits: u32,
    pub search_evaluated_unique_histories: u32,
    pub admission_evaluated_unique_histories: u32,
    pub semantic_history_overlap_count: u32,
    pub search_only_semantic_history_count: u32,
    pub admission_only_semantic_history_count: u32,
    pub each_panel_exhausts_formal_language: bool,
    pub admission_claimed_as_unseen_semantic_history_holdout: bool,
    pub admission_panel_role: String,
    /// Fail-closed empirical lower bound supported by the finite panel only.
    pub empirical_distinguishable_history_lower_bound_bits: u32,
    pub formal_rank_exceeds_empirical_panel: bool,
    pub required_response_vector_length: u32,
    pub retention_horizon_ticks: u32,
    pub blank_delay_excluded_from_semantic_rank: bool,
    pub generator_was_predeclared: bool,
    pub behavioral_novelty_claimed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConditionalTaskProgram {
    /// Archive bookkeeping only; never enters Simulation or sensory state.
    pub archive_index: u64,
    pub rank: u32,
    pub empty_delay_ticks: u32,
    pub context_count: u32,
    pub generator: String,
    pub fixed_episode_escrow_bits: u32,
    pub response_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConditionalContext {
    pub context_index: u32,
    /// Domain/task/outer-seed-scoped evidence key. Paired contexts
    /// intentionally share the Simulation seed, so generic progressive
    /// `trial_seeds` vectors carry this separately validated unique key.
    pub evidence_context_id: u64,
    pub complement_pair_index: u32,
    pub complement_member: bool,
    pub world_seed: u64,
    pub cue_bits: Vec<bool>,
    pub anchor_q: i32,
    pub anchor_r: i32,
    pub facing: FacingDirection,
    pub cue_distance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialMode {
    Normal,
    MatchedCueSymbolEquivarianceReplication,
    NuisancePerturbed,
    SemanticPermutation,
    FixedCueReplay,
    CueErased,
    FullBrainReset,
    DonorBrainSwapFollowDonor,
    DonorBrainSwapFollowHost,
    RandomActions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchAttempt {
    pub proposal_index: u32,
    pub controller_fingerprint: String,
    pub controller_genome: OrganismGenome,
    pub module: DelayLineModule,
    pub new_task_search_passes: u32,
    pub historical_search_passes: Vec<SolverPanelScore>,
    pub reached_sealed_admission: bool,
    pub rejection: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DelayLineModule {
    pub rank: u32,
    pub hidden_nodes: Vec<GeneNodeId>,
    pub input_weight: f32,
    pub chain_weight: f32,
    pub output_weight: f32,
    /// Complete encoded residual edge inventory.
    pub causal_edges: Vec<CausalEdge>,
    /// Mechanism-level lesions.  Paired inclusion-exclusion terms are a
    /// coupled causal unit; auditing each floating-point term in isolation
    /// would mislabel deliberate algebraic redundancy as non-causality.
    pub causal_slices: Vec<CausalSlice>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CausalEdge {
    pub label: String,
    pub innovation: InnovationId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CausalSlice {
    pub label: String,
    pub innovations: Vec<InnovationId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdmissionEvidence {
    pub candidate_controller_fingerprint: String,
    pub candidate_controller_genome: OrganismGenome,
    pub knockout_controller_fingerprint: String,
    pub knockout_controller_genome: OrganismGenome,
    pub knockout_is_exact_preceding_controller: bool,
    pub normal: PanelEvidence,
    pub matched_cue_symbol_equivariance_replication: PanelEvidence,
    pub nuisance_perturbed: PanelEvidence,
    pub semantic_permutation: PanelEvidence,
    pub fixed_cue_replay: PanelEvidence,
    pub cue_erased: PanelEvidence,
    pub full_brain_reset: PanelEvidence,
    pub donor_brain_swap_follow_donor: PanelEvidence,
    pub donor_brain_swap_follow_host: PanelEvidence,
    pub random_actions: PanelEvidence,
    pub exact_knockout: PanelEvidence,
    pub action_margin_audit: ActionMarginAudit,
    pub nuisance_outcome_differences: u32,
    pub fully_correct_complement_pairs: u32,
    pub archived_solver_admission: Vec<SolverPanelScore>,
    pub archived_solver_max_passes: u32,
    pub retention: RetentionAudit,
    pub extension_effect: ExtensionEffectEvidence,
    pub causal_necessities: Vec<CausalNecessity>,
    pub ecology_noninferiority: EcologyNoninferiority,
    pub task_solver_context_matrix: Vec<MatrixCell>,
    pub gate: AdmissionGate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PanelEvidence {
    pub mode: TrialMode,
    pub passes: u32,
    pub trials: Vec<TrialEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrialEvidence {
    pub context: ConditionalContext,
    pub passed: bool,
    pub correct_responses: u32,
    pub total_responses: u32,
    pub reward_released: bool,
    pub final_escrow_energy_bits: u32,
    pub behavior_fingerprint: String,
    pub trace: BehaviorTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BehaviorTrace {
    pub mode: TrialMode,
    pub reset_contract: String,
    pub cue_bits_presented: Vec<bool>,
    pub expected_response_bits: Vec<bool>,
    pub donor_cue_bits: Option<Vec<bool>>,
    pub donor_state_evidence: Option<DonorStateEvidence>,
    pub ticks: Vec<ConditionalTickTrace>,
    pub interventions: Vec<TaskEnergyIntervention>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DonorStateEvidence {
    pub world_seed: u64,
    pub cue_bits: Vec<bool>,
    pub anchor_q: i32,
    pub anchor_r: i32,
    pub facing: FacingDirection,
    pub cue_distance: u32,
    pub prepared_at_turn: u64,
    pub brain_state_fingerprint: String,
    pub ticks: Vec<ConditionalTickTrace>,
    pub interventions: Vec<TaskEnergyIntervention>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalTickTrace {
    pub phase: String,
    pub phase_index: u32,
    pub turn: u64,
    pub selected_action: ActionType,
    pub expected_action: Option<ActionType>,
    pub expected_facing_after_commit: Option<FacingDirection>,
    pub action_correct: Option<bool>,
    pub q_after_tick: i32,
    pub r_after_tick: i32,
    pub facing_after_tick: FacingDirection,
    pub organism_energy_bits: u32,
    pub sensory: Vec<SensoryActivation>,
    pub food_ray_activation_bits: Vec<u32>,
    pub hidden: Vec<HiddenActivation>,
    pub action_logits: Vec<ActionLogitTrace>,
    pub action_temperature: f32,
    pub action_temperature_bits: u32,
    pub deterministic_action_sample_tick: u64,
    pub deterministic_action_sample: f32,
    pub deterministic_action_sample_bits: u32,
    pub force_random_actions: bool,
    pub core_energy_ledger: EnergyLedgerRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensoryActivation {
    pub runtime_id: u32,
    pub receptor: SensoryReceptor,
    pub activation: f32,
    pub activation_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionLogitTrace {
    pub runtime_id: u32,
    pub action_type: ActionType,
    pub logit: f32,
    pub logit_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionMarginAudit {
    pub expected_normal_response_ticks: u32,
    pub observed_normal_response_ticks: u32,
    pub every_tick_has_complete_action_logits: bool,
    pub every_selected_action_is_unique_argmax: bool,
    pub minimum_required_raw_logit_margin: f32,
    pub minimum_observed_raw_logit_margin: Option<f32>,
    pub maximum_observed_raw_logit_margin: Option<f32>,
    pub minimum_deterministic_action_sample: Option<f32>,
    pub maximum_deterministic_action_sample: Option<f32>,
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HiddenActivation {
    pub runtime_id: u32,
    pub state_bits: u32,
    pub activation_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskEnergyIntervention {
    pub label: String,
    pub organism_before: f64,
    pub food_before: f64,
    pub locked_escrow_before: f64,
    pub organism_after: f64,
    pub food_after: f64,
    pub locked_escrow_after: f64,
    pub released_energy: f64,
    pub standing_task_food_energy: f64,
    pub captured_by_organism: f64,
    pub release_transfer_residual: f64,
    pub total_residual: f64,
    pub residual_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SolverPanelScore {
    pub solver_label: String,
    pub controller_fingerprint: String,
    pub task_archive_index: u64,
    pub panel: String,
    pub passes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatrixCell {
    pub solver_label: String,
    pub controller_fingerprint: String,
    pub task_archive_index: u64,
    pub task_fingerprint: String,
    pub panel: String,
    pub context_index: u32,
    pub evidence_context_id: u64,
    pub complement_pair_index: u32,
    pub complement_member: bool,
    pub world_seed: u64,
    pub passed: bool,
    pub behavior_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalNecessity {
    pub label: String,
    pub removed_innovations: Vec<InnovationId>,
    pub ablated_passes: u32,
    pub pass_drop: u32,
    pub necessary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetentionAudit {
    pub checkpoints: Vec<TaskCheckpoint>,
    pub evidence: AllHistoryRetention,
    pub accepted: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EcologyNoninferiority {
    pub horizon: u32,
    pub pairs: Vec<EcologyPairAudit>,
    pub minimum_preceding_plant_captures: u64,
    pub preceding_competence_floor_met: bool,
    pub every_seed_pair_noninferior: bool,
    pub preceding_survivor_ticks: u64,
    pub candidate_survivor_ticks: u64,
    pub preceding_plant_consumptions: u64,
    pub candidate_plant_consumptions: u64,
    pub aggregate_survivor_ticks_noninferior: bool,
    pub aggregate_plant_consumption_noninferior: bool,
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EcologyPairAudit {
    pub seed: u64,
    pub preceding: EcologyTrial,
    pub candidate: EcologyTrial,
    pub survivor_ticks_noninferior: bool,
    pub plant_consumptions_noninferior: bool,
    pub final_organism_energy_tolerance: f64,
    pub final_organism_energy_noninferior: bool,
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EcologyTrial {
    pub seed: u64,
    pub survivor_ticks: u64,
    pub final_population: u32,
    pub plant_consumptions: u64,
    pub final_organism_energy: f64,
    pub maximum_core_energy_residual: f64,
    pub maximum_core_energy_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdmissionGate {
    pub candidate_at_least_14_of_16: bool,
    pub every_archived_solver_at_most_2_of_16: bool,
    pub all_history_retained: bool,
    pub matched_cue_symbol_equivariance_replication_at_least_14_of_16: bool,
    pub random_at_most_2_of_16: bool,
    pub replay_at_most_2_of_16: bool,
    pub cue_erasure_at_most_2_of_16: bool,
    pub full_brain_reset_at_most_2_of_16: bool,
    pub donor_swap_follows_donor_at_least_14_of_16: bool,
    pub donor_swap_follows_host_at_most_2_of_16: bool,
    pub exact_knockout_at_most_2_of_16: bool,
    pub semantic_permutation_at_most_2_of_16: bool,
    pub nuisance_changes_at_most_2: bool,
    pub at_least_six_of_eight_complement_pairs: bool,
    pub every_claimed_causal_slice_necessary: bool,
    pub normal_response_actions_have_decisive_unique_argmax: bool,
    pub fixed_ecology_noninferior: bool,
    pub energy_accounting_closed: bool,
    pub qualified: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcceptedTaskRecord {
    pub task: ConditionalTaskProgram,
    pub evaluator_contract_version: String,
    pub task_program_fingerprint: String,
    pub search_contexts_fingerprint: String,
    pub admission_contexts_fingerprint: String,
    /// Composite identity also stored in the progressive checkpoint.
    pub task_fingerprint: String,
    pub contexts: Vec<ConditionalContext>,
    pub checkpoint: TaskCheckpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcceptedSolverRecord {
    pub solver_label: String,
    pub controller_fingerprint: String,
    pub controller_genome: OrganismGenome,
    pub accepted_at_archive_index: Option<u64>,
}

#[derive(Debug, Error)]
pub enum ConditionalProgramError {
    #[error("invalid conditional-program config: {0}")]
    InvalidConfig(String),
    #[error(transparent)]
    Simulation(#[from] SimError),
    #[error("conditional-program integrity failure: {0}")]
    Integrity(String),
}

#[derive(Debug, Clone)]
struct ArchivedTaskRuntime {
    record: AcceptedTaskRecord,
    accepted_genome: OrganismGenome,
}

#[derive(Debug, Clone)]
struct ArchivedSolverRuntime {
    label: String,
    genome: OrganismGenome,
    fingerprint: String,
    accepted_at: Option<u64>,
}

#[derive(Serialize)]
struct FrozenTaskIdentity<'a> {
    evaluator_contract_version: &'a str,
    task_world_config_fingerprint: &'a str,
    random_action_task_world_config_fingerprint: &'a str,
    ecology_world_config_fingerprint: &'a str,
    ecology_horizon: u32,
    task_program_fingerprint: &'a str,
    search_contexts_fingerprint: &'a str,
    admission_contexts_fingerprint: &'a str,
}

#[derive(Debug, Clone)]
struct PreparedDonorState {
    brain: BrainState,
    evidence: DonorStateEvidence,
}

pub fn run_conditional_program_experiment(
    config: ConditionalProgramConfig,
) -> Result<ConditionalProgramExperiment, ConditionalProgramError> {
    validate_experiment_config(&config)?;
    let effective_evaluator_configs = effective_evaluator_configs()?;
    let mut outer_runs = Vec::with_capacity(config.outer_seeds.len());
    for &outer_seed in &config.outer_seeds {
        outer_runs.push(run_outer_seed(
            &config,
            outer_seed,
            &effective_evaluator_configs,
        )?);
    }

    let qualifying_discoveries = outer_runs
        .iter()
        .flat_map(|run| &run.stages)
        .filter(|stage| stage.accepted)
        .count() as u32;
    let maximum_accepted_rank = outer_runs
        .iter()
        .flat_map(|run| &run.stages)
        .filter(|stage| stage.accepted)
        .map(|stage| stage.task.rank)
        .max();
    let seeds_with_at_least_one_discovery = outer_runs
        .iter()
        .filter(|run| run.stages.iter().any(|stage| stage.accepted))
        .count() as u32;
    let terminal_failure_modes = outer_runs
        .iter()
        .map(|run| format!("outer_seed_{}: {}", run.outer_seed, run.stopped_reason))
        .collect();

    let outer_seed_count = config.outer_seeds.len();
    Ok(ConditionalProgramExperiment {
        schema: "conditional-program-pilot-v3".to_owned(),
        config,
        effective_evaluator_configs,
        contract: ExperimentContract {
            evaluator_contract_version: EVALUATOR_CONTRACT_VERSION.to_owned(),
            organism_observations: vec![
                "three ordinary relative food rays".to_owned(),
                "ordinary contact-ahead".to_owned(),
                "ordinary energy sensor".to_owned(),
            ],
            forbidden_observations: vec![
                "task or archive id".to_owned(),
                "task fingerprint".to_owned(),
                "context index".to_owned(),
                "target response bit".to_owned(),
                "private task RNG".to_owned(),
            ],
            pose_reset: "before every cue, delay, and response tick: restore q/r/facing, health, damage, last action, task-related consumption counters, and energy_at_last_sensing; preserve the complete BrainState except in explicit reset/swap arms".to_owned(),
            energy: "one fixed latent escrow per episode; task interventions and every canonical Simulation tick fail closed independently".to_owned(),
            search_and_admission: "the evaluator contract version, task program, and both context panels are fingerprinted into one frozen-task identity before deterministic solver proposals; search and sealed admission use disjoint world seeds and evidence ids but may contain the same semantic histories; admission is a sealed world/RNG/pose replication panel, not an unseen-semantic-history holdout (rank 4 enumerates all 16 histories in both panels); archived-solver prefiltering uses search only; the first proposal selected entirely by search plus all-history replay is audited exactly once, and the stage stops regardless of the sealed verdict".to_owned(),
            ecology_gate: format!("four fixed paired ecology seeds; candidate survivor ticks, plant captures, and final organism energy must each be noninferior on every seed pair; energy uses a 32*f32::EPSILON*max(abs(preceding),abs(candidate),1) tolerance; the preceding controller must capture at least {MIN_PRECEDING_ECOLOGY_PLANT_CAPTURES} plant across the four-seed panel"),
            cue_symbol_equivariance_replication: "complementing both visible cue symbols and their matched response semantics is a redundant matched-equivariance replication of the normal panel, retained as a consistency check and never counted as independent evidence".to_owned(),
            action_margin_gate: format!("on every normal response tick, the selected action must be the unique raw-logit argmax over the full action vector and exceed the next-highest action logit (or explicit idle bias) by at least {MIN_NORMAL_RESPONSE_LOGIT_MARGIN}; this prevents deterministic seed/turn samples from rescuing marginal decisions"),
            rank_limit: "no encoded experimental maximum; stage/search budgets limit a CLI run, while u32 node ids and machine memory are physical ceilings".to_owned(),
        },
        scope_limitations: vec![
            "the delayed-copy task generator, cue alphabet, and response rule are predeclared; rank growth is capacity evidence, not open-ended behavioral novelty".to_owned(),
            "the pilot requires both the selected turn action and its post-commit facing change, but does not score dedicated task-food consumption events".to_owned(),
            "task runtime and escrow evidence are owned by the sim-core evaluator trace rather than serialized into Simulation and the canonical tick ledger".to_owned(),
            "no exact strong/stutter Mealy-machine quotient or alpha-canonical task hash is implemented in this vertical slice".to_owned(),
            "the cue-symbol relabel arm is a redundant matched-equivariance replication, not an independent capability observation".to_owned(),
            "a finite stage-budgeted run cannot establish an unbounded discovery tail".to_owned(),
        ],
        outer_runs,
        summary: ExperimentSummary {
            outer_seed_count,
            qualifying_discoveries,
            maximum_accepted_rank,
            seeds_with_at_least_one_discovery,
            terminal_failure_modes,
            // A finite pilot cannot establish an unbounded tail.  This field is
            // deliberately fail-closed even if every runtime-budgeted stage passes.
            open_endedness_demonstrated: false,
        },
    })
}

fn validate_experiment_config(
    config: &ConditionalProgramConfig,
) -> Result<(), ConditionalProgramError> {
    if config.outer_seeds.is_empty() {
        return Err(ConditionalProgramError::InvalidConfig(
            "outer_seeds must not be empty".to_owned(),
        ));
    }
    if config.outer_seeds.iter().collect::<BTreeSet<_>>().len() != config.outer_seeds.len() {
        return Err(ConditionalProgramError::InvalidConfig(
            "outer_seeds must not contain duplicates".to_owned(),
        ));
    }
    if config.stage_budget == 0 || config.search_budget == 0 {
        return Err(ConditionalProgramError::InvalidConfig(
            "stage_budget and search_budget must be at least one".to_owned(),
        ));
    }
    if config.starting_rank < 4 {
        return Err(ConditionalProgramError::InvalidConfig(
            "starting_rank must be >= 4 so sixteen contexts can carry unique sequences".to_owned(),
        ));
    }
    if config.empty_delay_ticks == 0 {
        return Err(ConditionalProgramError::InvalidConfig(
            "empty_delay_ticks must be at least one".to_owned(),
        ));
    }
    if !config.escrow_energy.is_finite() || config.escrow_energy < 1.0 {
        return Err(ConditionalProgramError::InvalidConfig(
            "escrow_energy must be finite and >= 1.0 so a positive transfer cannot round to zero at organism scale".to_owned(),
        ));
    }
    if config.ecology_horizon == 0 {
        return Err(ConditionalProgramError::InvalidConfig(
            "ecology_horizon must be at least one".to_owned(),
        ));
    }
    config
        .starting_rank
        .checked_add(config.stage_budget.saturating_sub(1))
        .and_then(|rank| rank.checked_add(config.empty_delay_ticks))
        .and_then(|nodes| nodes.checked_add(1))
        .ok_or_else(|| {
            ConditionalProgramError::InvalidConfig(
                "requested rank/stage/delay overflows u32".to_owned(),
            )
        })?;
    Ok(())
}

fn run_outer_seed(
    config: &ConditionalProgramConfig,
    outer_seed: u64,
    effective_configs: &EffectiveEvaluatorConfigs,
) -> Result<OuterRun, ConditionalProgramError> {
    let base = OrganismGenome::test_fixture();
    let base_fingerprint = fingerprint(&base)?;
    let mut current = base.clone();
    let mut archived_tasks: Vec<ArchivedTaskRuntime> = Vec::new();
    let mut archived_solvers = vec![ArchivedSolverRuntime {
        label: "solver_0_initial".to_owned(),
        genome: base.clone(),
        fingerprint: base_fingerprint.clone(),
        accepted_at: None,
    }];
    let mut stages = Vec::new();
    let mut stopped_reason = format!(
        "runtime stage budget {} exhausted; finite evidence is not an open-ended tail",
        config.stage_budget
    );

    for stage_offset in 0..config.stage_budget {
        let archive_index = stage_offset as u64;
        let rank = config
            .starting_rank
            .checked_add(stage_offset)
            .ok_or_else(|| ConditionalProgramError::Integrity("rank overflow".to_owned()))?;
        let task = ConditionalTaskProgram {
            archive_index,
            rank,
            empty_delay_ticks: config.empty_delay_ticks,
            context_count: PANEL_SIZE as u32,
            generator: "unique-low-four-bits-plus-domain-separated-prefix-v1".to_owned(),
            fixed_episode_escrow_bits: config.escrow_energy.to_bits(),
            response_rule: "after empty delay, turn left for a left cue and right for a right cue, in original cue order".to_owned(),
        };
        // The task program and both panels are frozen and fingerprinted before
        // any proposal is constructed. None contains a solver-dependent field.
        let task_program_fingerprint = fingerprint(&task)?;
        let search_contexts = make_contexts(&task, outer_seed, PanelDomain::Search, false);
        let admission_contexts = make_contexts(&task, outer_seed, PanelDomain::Admission, false);
        validate_context_panel(&search_contexts, "search")?;
        validate_context_panel(&admission_contexts, "admission")?;
        let search_contexts_fingerprint = fingerprint(&search_contexts)?;
        let admission_contexts_fingerprint = fingerprint(&admission_contexts)?;
        let task_fingerprint = fingerprint(&FrozenTaskIdentity {
            evaluator_contract_version: EVALUATOR_CONTRACT_VERSION,
            task_world_config_fingerprint: &effective_configs.task_world_fingerprint,
            random_action_task_world_config_fingerprint: &effective_configs
                .random_action_task_world_fingerprint,
            ecology_world_config_fingerprint: &effective_configs.ecology_world_fingerprint,
            ecology_horizon: config.ecology_horizon,
            task_program_fingerprint: &task_program_fingerprint,
            search_contexts_fingerprint: &search_contexts_fingerprint,
            admission_contexts_fingerprint: &admission_contexts_fingerprint,
        })?;
        let search_seeds = search_contexts
            .iter()
            .map(|context| context.world_seed)
            .collect::<BTreeSet<_>>();
        let context_world_seeds_disjoint = admission_contexts
            .iter()
            .all(|context| !search_seeds.contains(&context.world_seed));
        if !context_world_seeds_disjoint {
            return Err(ConditionalProgramError::Integrity(
                "search and admission world seeds overlap".to_owned(),
            ));
        }
        let search_evidence_ids = search_contexts
            .iter()
            .map(|context| context.evidence_context_id)
            .collect::<BTreeSet<_>>();
        let context_evidence_ids_disjoint = admission_contexts
            .iter()
            .all(|context| !search_evidence_ids.contains(&context.evidence_context_id));
        if !context_evidence_ids_disjoint {
            return Err(ConditionalProgramError::Integrity(
                "search and admission evidence context ids overlap".to_owned(),
            ));
        }

        let mut old_scores = Vec::new();
        for solver in &archived_solvers {
            let panel = evaluate_panel(
                &task,
                &solver.genome,
                &search_contexts,
                TrialMode::Normal,
                config.escrow_energy,
                effective_configs,
            )?;
            old_scores.push(score_panel(
                solver,
                archive_index,
                "new_task_search",
                &panel,
            ));
        }
        let archived_solver_max = old_scores
            .iter()
            .map(|score| score.passes)
            .max()
            .unwrap_or(0);

        let protector = ProtectedResidual::seal(&current)
            .map_err(|error| ConditionalProgramError::Integrity(error.to_string()))?;
        let mut search_attempts = Vec::new();
        let mut accepted_admission = None;
        let mut accepted_genome = None;

        if archived_solver_max <= MAX_CONTROL_PASSES {
            for proposal_index in 0..config.search_budget {
                let (candidate, module) =
                    make_delay_line_candidate(&protector, &task, proposal_index)?;
                let candidate_fingerprint = fingerprint(&candidate)?;
                let search_panel = evaluate_panel(
                    &task,
                    &candidate,
                    &search_contexts,
                    TrialMode::Normal,
                    config.escrow_energy,
                    effective_configs,
                )?;
                let mut historical_search_passes = Vec::new();
                let mut history_ok = true;
                if search_panel.passes == PANEL_SIZE as u32 {
                    for archived in &archived_tasks {
                        let panel = evaluate_panel(
                            &archived.record.task,
                            &candidate,
                            &archived.record.contexts,
                            TrialMode::Normal,
                            f32::from_bits(archived.record.task.fixed_episode_escrow_bits),
                            effective_configs,
                        )?;
                        history_ok &= panel.passes >= MIN_ADMISSION_PASSES;
                        historical_search_passes.push(SolverPanelScore {
                            solver_label: format!("candidate_{proposal_index}"),
                            controller_fingerprint: candidate_fingerprint.clone(),
                            task_archive_index: archived.record.task.archive_index,
                            panel: "historical_search_replay".to_owned(),
                            passes: panel.passes,
                        });
                    }
                } else {
                    history_ok = false;
                }

                let mut attempt = SearchAttempt {
                    proposal_index,
                    controller_fingerprint: candidate_fingerprint,
                    controller_genome: candidate.clone(),
                    module: module.clone(),
                    new_task_search_passes: search_panel.passes,
                    historical_search_passes,
                    reached_sealed_admission: false,
                    rejection: None,
                };
                if search_panel.passes != PANEL_SIZE as u32 {
                    attempt.rejection = Some(format!(
                        "search panel {}/16; deterministic proposal did not solve the frozen task",
                        search_panel.passes
                    ));
                    search_attempts.push(attempt);
                    continue;
                }
                if !history_ok {
                    attempt.rejection = Some(
                        "candidate solved the new search panel but failed an all-history search replay"
                            .to_owned(),
                    );
                    search_attempts.push(attempt);
                    continue;
                }

                attempt.reached_sealed_admission = true;
                let admission = audit_candidate(
                    config,
                    outer_seed,
                    &task,
                    &task_fingerprint,
                    &protector,
                    &current,
                    &candidate,
                    &module,
                    &admission_contexts,
                    &archived_tasks,
                    &archived_solvers,
                    effective_configs,
                )?;
                let qualified = admission.gate.qualified;
                if !qualified {
                    attempt.rejection = Some(admission.gate.failures.join("; "));
                }
                search_attempts.push(attempt);
                if qualified {
                    accepted_genome = Some(candidate);
                }
                // A sealed panel is a one-shot verdict, never feedback for
                // choosing a later proposal on the same holdout contexts.
                accepted_admission = Some(admission);
                break;
            }
        }

        let mut failure_modes = Vec::new();
        if archived_solver_max > MAX_CONTROL_PASSES {
            failure_modes.push(format!(
                "bounded novelty: archived solver reaches {archived_solver_max}/16 on proposed task (maximum 2)"
            ));
        }
        if search_attempts.is_empty() && archived_solver_max <= MAX_CONTROL_PASSES {
            failure_modes.push("no deterministic solver proposal was evaluated".to_owned());
        }
        let accepted = accepted_genome.is_some();
        if !accepted {
            if let Some(admission) = accepted_admission.as_ref() {
                failure_modes.extend(admission.gate.failures.clone());
            } else if archived_solver_max <= MAX_CONTROL_PASSES {
                let best_history = search_attempts
                    .iter()
                    .filter(|attempt| attempt.new_task_search_passes == PANEL_SIZE as u32)
                    .filter_map(|attempt| {
                        attempt
                            .historical_search_passes
                            .iter()
                            .map(|score| score.passes)
                            .min()
                    })
                    .max();
                if let Some(best_history) = best_history {
                    failure_modes.push(format!(
                        "all-history retention failure: the new task was solved 16/16, but the best proposal's worst historical replay was {best_history}/16 (minimum {MIN_ADMISSION_PASSES}/16)"
                    ));
                } else {
                    let best = search_attempts
                        .iter()
                        .map(|attempt| attempt.new_task_search_passes)
                        .max()
                        .unwrap_or(0);
                    failure_modes.push(format!(
                        "solver search plateau: best frozen-task search score {best}/16"
                    ));
                }
            }
        }

        let search_histories = search_contexts
            .iter()
            .map(|context| context.cue_bits.clone())
            .collect::<BTreeSet<_>>();
        let admission_histories = admission_contexts
            .iter()
            .map(|context| context.cue_bits.clone())
            .collect::<BTreeSet<_>>();
        let semantic_history_overlap_count =
            search_histories.intersection(&admission_histories).count() as u32;
        let search_unique_histories = search_histories.len() as u32;
        let admission_unique_histories = admission_histories.len() as u32;
        let empirical_history_bits = admission_unique_histories.ilog2();
        let formal_history_count = 1_u128.checked_shl(task.rank).unwrap_or(u128::MAX);
        let stage = StageEvidence {
            archive_index,
            evaluator_contract_version: EVALUATOR_CONTRACT_VERSION.to_owned(),
            task_world_config_fingerprint: effective_configs.task_world_fingerprint.clone(),
            random_action_task_world_config_fingerprint: effective_configs
                .random_action_task_world_fingerprint
                .clone(),
            ecology_world_config_fingerprint: effective_configs.ecology_world_fingerprint.clone(),
            ecology_horizon: config.ecology_horizon,
            task: task.clone(),
            task_program_fingerprint: task_program_fingerprint.clone(),
            search_contexts_fingerprint: search_contexts_fingerprint.clone(),
            admission_contexts_fingerprint: admission_contexts_fingerprint.clone(),
            task_fingerprint: task_fingerprint.clone(),
            semantic_rank: SemanticRankCertificate {
                formal_task_language_history_count: format!("2^{}", task.rank),
                formal_exact_solver_memory_lower_bound_bits: task.rank,
                search_evaluated_unique_histories: search_unique_histories,
                admission_evaluated_unique_histories: admission_unique_histories,
                semantic_history_overlap_count,
                search_only_semantic_history_count: search_unique_histories
                    - semantic_history_overlap_count,
                admission_only_semantic_history_count: admission_unique_histories
                    - semantic_history_overlap_count,
                each_panel_exhausts_formal_language: formal_history_count
                    == u128::from(search_unique_histories)
                    && formal_history_count == u128::from(admission_unique_histories),
                admission_claimed_as_unseen_semantic_history_holdout: false,
                admission_panel_role: "sealed world/RNG/pose replication panel; not claimed as an unseen-semantic-history holdout".to_owned(),
                empirical_distinguishable_history_lower_bound_bits: empirical_history_bits,
                formal_rank_exceeds_empirical_panel: task.rank > empirical_history_bits,
                required_response_vector_length: task.rank,
                retention_horizon_ticks: task.empty_delay_ticks,
                blank_delay_excluded_from_semantic_rank: true,
                generator_was_predeclared: true,
                behavioral_novelty_claimed: false,
            },
            task_frozen_before_solver_search: true,
            search_contexts,
            admission_contexts: admission_contexts.clone(),
            context_world_seeds_disjoint,
            context_evidence_ids_disjoint,
            archived_solver_new_task_search: old_scores,
            search_attempts,
            admission: accepted_admission,
            accepted,
            failure_modes: failure_modes.clone(),
        };

        if let Some(candidate) = accepted_genome {
            let admission = stage
                .admission
                .as_ref()
                .expect("accepted candidate has admission evidence");
            let checkpoint =
                checkpoint_from_panel(&task, &task_fingerprint, &candidate, &admission.normal)?;
            current = candidate.clone();
            archived_tasks.push(ArchivedTaskRuntime {
                record: AcceptedTaskRecord {
                    task: task.clone(),
                    evaluator_contract_version: EVALUATOR_CONTRACT_VERSION.to_owned(),
                    task_program_fingerprint: task_program_fingerprint.clone(),
                    search_contexts_fingerprint: search_contexts_fingerprint.clone(),
                    admission_contexts_fingerprint: admission_contexts_fingerprint.clone(),
                    task_fingerprint: task_fingerprint.clone(),
                    contexts: admission_contexts,
                    checkpoint,
                },
                accepted_genome: candidate.clone(),
            });
            archived_solvers.push(ArchivedSolverRuntime {
                label: format!("solver_{}_rank_{}", archive_index + 1, rank),
                fingerprint: fingerprint(&candidate)?,
                genome: candidate,
                accepted_at: Some(archive_index),
            });
            stages.push(stage);
        } else {
            stopped_reason = if failure_modes.is_empty() {
                "candidate rejected without a classified failure (integrity bug)".to_owned()
            } else {
                failure_modes.join("; ")
            };
            stages.push(stage);
            break;
        }
    }

    let final_crossplay = build_crossplay(
        &archived_solvers,
        &archived_tasks,
        config.escrow_energy,
        effective_configs,
    )?;
    Ok(OuterRun {
        outer_seed,
        initial_controller_fingerprint: base_fingerprint,
        initial_controller_genome: base,
        stages,
        accepted_tasks: archived_tasks
            .iter()
            .map(|task| task.record.clone())
            .collect(),
        accepted_solvers: archived_solvers
            .iter()
            .map(|solver| AcceptedSolverRecord {
                solver_label: solver.label.clone(),
                controller_fingerprint: solver.fingerprint.clone(),
                controller_genome: solver.genome.clone(),
                accepted_at_archive_index: solver.accepted_at,
            })
            .collect(),
        final_crossplay,
        final_controller_fingerprint: fingerprint(&current)?,
        stopped_reason,
    })
}

#[allow(clippy::too_many_arguments)]
fn audit_candidate(
    config: &ConditionalProgramConfig,
    outer_seed: u64,
    task: &ConditionalTaskProgram,
    task_fingerprint: &str,
    protector: &ProtectedResidual,
    preceding: &OrganismGenome,
    candidate: &OrganismGenome,
    module: &DelayLineModule,
    contexts: &[ConditionalContext],
    archived_tasks: &[ArchivedTaskRuntime],
    archived_solvers: &[ArchivedSolverRuntime],
    effective_configs: &EffectiveEvaluatorConfigs,
) -> Result<AdmissionEvidence, ConditionalProgramError> {
    let normal = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::Normal,
        config.escrow_energy,
        effective_configs,
    )?;
    let matched_cue_symbol_equivariance_replication = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::MatchedCueSymbolEquivarianceReplication,
        config.escrow_energy,
        effective_configs,
    )?;
    let nuisance_perturbed = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::NuisancePerturbed,
        config.escrow_energy,
        effective_configs,
    )?;
    let semantic_permutation = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::SemanticPermutation,
        config.escrow_energy,
        effective_configs,
    )?;
    let fixed_cue_replay = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::FixedCueReplay,
        config.escrow_energy,
        effective_configs,
    )?;
    let cue_erased = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::CueErased,
        config.escrow_energy,
        effective_configs,
    )?;
    let full_brain_reset = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::FullBrainReset,
        config.escrow_energy,
        effective_configs,
    )?;
    let donor_brain_swap_follow_donor = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::DonorBrainSwapFollowDonor,
        config.escrow_energy,
        effective_configs,
    )?;
    let donor_brain_swap_follow_host = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::DonorBrainSwapFollowHost,
        config.escrow_energy,
        effective_configs,
    )?;
    let random_actions = evaluate_panel(
        task,
        candidate,
        contexts,
        TrialMode::RandomActions,
        config.escrow_energy,
        effective_configs,
    )?;
    let knockout = protector.knockout_extension();
    let exact_knockout = evaluate_panel(
        task,
        &knockout,
        contexts,
        TrialMode::Normal,
        config.escrow_energy,
        effective_configs,
    )?;
    let candidate_fingerprint = fingerprint(candidate)?;
    let knockout_fingerprint = fingerprint(&knockout)?;
    let preceding_fingerprint = fingerprint(preceding)?;

    let nuisance_outcome_differences = normal
        .trials
        .iter()
        .zip(&nuisance_perturbed.trials)
        .filter(|(left, right)| left.passed != right.passed)
        .count() as u32;
    let fully_correct_complement_pairs = count_fully_correct_complement_pairs(&normal)?;
    let action_margin_audit = audit_normal_response_action_margins(task, &normal);

    let checkpoints = archived_tasks
        .iter()
        .map(|task| task.record.checkpoint.clone())
        .collect::<Vec<_>>();
    let mut ancestor_replays = Vec::new();
    let mut candidate_replays = Vec::new();
    for archived in archived_tasks {
        let ancestor_panel = evaluate_panel(
            &archived.record.task,
            &archived.accepted_genome,
            &archived.record.contexts,
            TrialMode::Normal,
            f32::from_bits(archived.record.task.fixed_episode_escrow_bits),
            effective_configs,
        )?;
        ancestor_replays.push(replay_from_panel(
            &archived.record.task,
            &archived.record.task_fingerprint,
            &archived.accepted_genome,
            &ancestor_panel,
        )?);
        let candidate_panel = evaluate_panel(
            &archived.record.task,
            candidate,
            &archived.record.contexts,
            TrialMode::Normal,
            f32::from_bits(archived.record.task.fixed_episode_escrow_bits),
            effective_configs,
        )?;
        candidate_replays.push(replay_from_panel(
            &archived.record.task,
            &archived.record.task_fingerprint,
            candidate,
            &candidate_panel,
        )?);
    }
    let retention_evidence = AllHistoryRetention {
        candidate_controller_fingerprint: candidate_fingerprint.clone(),
        ancestor_replays,
        candidate_replays,
    };
    let retention_result =
        enforce_retention(&checkpoints, &retention_evidence).map_err(|error| error.to_string());
    let retention = RetentionAudit {
        checkpoints,
        evidence: retention_evidence,
        accepted: retention_result.is_ok(),
        error: retention_result.err(),
    };

    let extension_effect = ExtensionEffectEvidence {
        task_id: task.archive_index,
        task_fingerprint: task_fingerprint.to_owned(),
        trial_seeds: contexts
            .iter()
            .map(|context| context.evidence_context_id)
            .collect(),
        enabled_controller_fingerprint: candidate_fingerprint.clone(),
        knockout_controller_fingerprint: knockout_fingerprint.clone(),
        enabled_passes: normal.passes,
        knockout_passes: exact_knockout.passes,
        minimum_enabled_passes: MIN_ADMISSION_PASSES,
        maximum_knockout_passes: MAX_CONTROL_PASSES,
        enabled_behavior_fingerprints: normal
            .trials
            .iter()
            .map(|trial| trial.behavior_fingerprint.clone())
            .collect(),
        knockout_behavior_fingerprints: exact_knockout
            .trials
            .iter()
            .map(|trial| trial.behavior_fingerprint.clone())
            .collect(),
    };
    let extension_verified = verify_extension_effect(&preceding_fingerprint, &extension_effect)
        .map_err(|error| error.to_string());

    let mut causal_necessities = Vec::new();
    for slice in &module.causal_slices {
        let mut ablated = candidate.clone();
        for innovation in &slice.innovations {
            let gene = ablated
                .brain
                .edges
                .iter_mut()
                .find(|gene| gene.innovation == *innovation)
                .ok_or_else(|| {
                    ConditionalProgramError::Integrity(format!(
                        "causal slice {} references missing innovation {:?}",
                        slice.label, innovation
                    ))
                })?;
            gene.enabled = false;
        }
        let panel = evaluate_panel(
            task,
            &ablated,
            contexts,
            TrialMode::Normal,
            config.escrow_energy,
            effective_configs,
        )?;
        let pass_drop = normal.passes.saturating_sub(panel.passes);
        causal_necessities.push(CausalNecessity {
            label: slice.label.clone(),
            removed_innovations: slice.innovations.clone(),
            ablated_passes: panel.passes,
            pass_drop,
            necessary: pass_drop >= MIN_CAUSAL_DROP,
        });
    }

    let ecology_noninferiority = ecology_noninferiority(
        config.ecology_horizon,
        outer_seed,
        preceding,
        candidate,
        &effective_configs.ecology_world,
    )?;

    // This is the first and only archived-solver evaluation on the sealed
    // admission panel. Its result cannot influence proposal selection.
    let archived_admission_trials = archived_solvers
        .iter()
        .map(|solver| {
            evaluate_panel(
                task,
                &solver.genome,
                contexts,
                TrialMode::Normal,
                config.escrow_energy,
                effective_configs,
            )
            .map(|panel| (solver.clone(), panel))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let archived_solver_admission = archived_admission_trials
        .iter()
        .map(|(solver, panel)| score_panel(solver, task.archive_index, "new_task_admission", panel))
        .collect::<Vec<_>>();
    let archived_solver_max_passes = archived_solver_admission
        .iter()
        .map(|score| score.passes)
        .max()
        .unwrap_or(0);
    let mut matrix = Vec::new();
    for (solver, panel) in &archived_admission_trials {
        append_matrix(
            &mut matrix,
            solver,
            task,
            task_fingerprint,
            "archived_solver_new_task_admission",
            panel,
        );
    }
    let candidate_runtime = ArchivedSolverRuntime {
        label: "candidate".to_owned(),
        genome: candidate.clone(),
        fingerprint: candidate_fingerprint.clone(),
        accepted_at: Some(task.archive_index),
    };
    for (label, panel) in [
        ("candidate_normal", &normal),
        (
            "candidate_matched_cue_symbol_equivariance_replication",
            &matched_cue_symbol_equivariance_replication,
        ),
        ("candidate_nuisance", &nuisance_perturbed),
        ("candidate_semantic", &semantic_permutation),
        ("candidate_replay", &fixed_cue_replay),
        ("candidate_cue_erased", &cue_erased),
        ("candidate_full_brain_reset", &full_brain_reset),
        (
            "candidate_donor_brain_swap_follow_donor",
            &donor_brain_swap_follow_donor,
        ),
        (
            "candidate_donor_brain_swap_follow_host",
            &donor_brain_swap_follow_host,
        ),
        ("candidate_random", &random_actions),
    ] {
        append_matrix(
            &mut matrix,
            &candidate_runtime,
            task,
            task_fingerprint,
            label,
            panel,
        );
    }
    let knockout_runtime = ArchivedSolverRuntime {
        label: "exact_preceding_controller_knockout".to_owned(),
        genome: knockout.clone(),
        fingerprint: knockout_fingerprint.clone(),
        accepted_at: None,
    };
    append_matrix(
        &mut matrix,
        &knockout_runtime,
        task,
        task_fingerprint,
        "exact_knockout",
        &exact_knockout,
    );

    let every_claimed_causal_slice_necessary =
        causal_necessities.iter().all(|slice| slice.necessary);
    let energy_accounting_closed = [
        &normal,
        &matched_cue_symbol_equivariance_replication,
        &nuisance_perturbed,
        &semantic_permutation,
        &fixed_cue_replay,
        &cue_erased,
        &full_brain_reset,
        &donor_brain_swap_follow_donor,
        &donor_brain_swap_follow_host,
        &random_actions,
        &exact_knockout,
    ]
    .into_iter()
    .all(panel_energy_closed)
        && ecology_energy_closed(&ecology_noninferiority);

    let mut gate = AdmissionGate {
        candidate_at_least_14_of_16: normal.passes >= MIN_ADMISSION_PASSES,
        every_archived_solver_at_most_2_of_16: archived_solver_max_passes <= MAX_CONTROL_PASSES,
        all_history_retained: retention.accepted,
        matched_cue_symbol_equivariance_replication_at_least_14_of_16:
            matched_cue_symbol_equivariance_replication.passes >= MIN_ADMISSION_PASSES,
        random_at_most_2_of_16: random_actions.passes <= MAX_CONTROL_PASSES,
        replay_at_most_2_of_16: fixed_cue_replay.passes <= MAX_CONTROL_PASSES,
        cue_erasure_at_most_2_of_16: cue_erased.passes <= MAX_CONTROL_PASSES,
        full_brain_reset_at_most_2_of_16: full_brain_reset.passes <= MAX_CONTROL_PASSES,
        donor_swap_follows_donor_at_least_14_of_16: donor_brain_swap_follow_donor.passes
            >= MIN_ADMISSION_PASSES,
        donor_swap_follows_host_at_most_2_of_16: donor_brain_swap_follow_host.passes
            <= MAX_CONTROL_PASSES,
        exact_knockout_at_most_2_of_16: exact_knockout.passes <= MAX_CONTROL_PASSES,
        semantic_permutation_at_most_2_of_16: semantic_permutation.passes <= MAX_CONTROL_PASSES,
        nuisance_changes_at_most_2: nuisance_outcome_differences <= 2,
        at_least_six_of_eight_complement_pairs: fully_correct_complement_pairs >= 6,
        every_claimed_causal_slice_necessary,
        normal_response_actions_have_decisive_unique_argmax: action_margin_audit.accepted,
        fixed_ecology_noninferior: ecology_noninferiority.accepted,
        energy_accounting_closed,
        qualified: false,
        failures: Vec::new(),
    };
    collect_gate_failures(&mut gate);
    if let Err(error) = extension_verified {
        gate.failures
            .push(format!("protected-residual capability audit: {error}"));
    }
    gate.qualified = gate.failures.is_empty();

    Ok(AdmissionEvidence {
        candidate_controller_fingerprint: candidate_fingerprint,
        candidate_controller_genome: candidate.clone(),
        knockout_controller_fingerprint: knockout_fingerprint,
        knockout_controller_genome: knockout,
        knockout_is_exact_preceding_controller: protector.knockout_extension() == *preceding,
        normal,
        matched_cue_symbol_equivariance_replication,
        nuisance_perturbed,
        semantic_permutation,
        fixed_cue_replay,
        cue_erased,
        full_brain_reset,
        donor_brain_swap_follow_donor,
        donor_brain_swap_follow_host,
        random_actions,
        exact_knockout,
        action_margin_audit,
        nuisance_outcome_differences,
        fully_correct_complement_pairs,
        archived_solver_admission,
        archived_solver_max_passes,
        retention,
        extension_effect,
        causal_necessities,
        ecology_noninferiority,
        task_solver_context_matrix: matrix,
        gate,
    })
}

fn collect_gate_failures(gate: &mut AdmissionGate) {
    let checks = [
        (
            gate.candidate_at_least_14_of_16,
            "candidate admission below 14/16",
        ),
        (
            gate.every_archived_solver_at_most_2_of_16,
            "an archived solver exceeds 2/16 on the new task",
        ),
        (gate.all_history_retained, "all-history retention failed"),
        (
            gate.matched_cue_symbol_equivariance_replication_at_least_14_of_16,
            "redundant matched cue-symbol equivariance replication below 14/16",
        ),
        (gate.random_at_most_2_of_16, "random control exceeds 2/16"),
        (
            gate.replay_at_most_2_of_16,
            "fixed replay control exceeds 2/16",
        ),
        (
            gate.cue_erasure_at_most_2_of_16,
            "cue erasure control exceeds 2/16",
        ),
        (
            gate.full_brain_reset_at_most_2_of_16,
            "full post-cue BrainState reset exceeds 2/16",
        ),
        (
            gate.donor_swap_follows_donor_at_least_14_of_16,
            "donor BrainState swap follows donor cue below 14/16",
        ),
        (
            gate.donor_swap_follows_host_at_most_2_of_16,
            "donor BrainState swap still follows host truth above 2/16",
        ),
        (
            gate.exact_knockout_at_most_2_of_16,
            "exact preceding-controller knockout exceeds 2/16",
        ),
        (
            gate.semantic_permutation_at_most_2_of_16,
            "causal semantic permutation exceeds 2/16",
        ),
        (
            gate.nuisance_changes_at_most_2,
            "nuisance perturbation changes more than two outcomes",
        ),
        (
            gate.at_least_six_of_eight_complement_pairs,
            "fewer than six of eight paired action-stream complement contexts are jointly correct",
        ),
        (
            gate.every_claimed_causal_slice_necessary,
            "one or more claimed causal slices fail the >=8/16 necessity drop",
        ),
        (
            gate.normal_response_actions_have_decisive_unique_argmax,
            "normal response actions are not all unique raw-logit argmaxes with >=0.1 margin",
        ),
        (
            gate.fixed_ecology_noninferior,
            "fixed-ecology per-seed survival/plant/final-energy noninferiority or non-vacuous preceding competence floor failed",
        ),
        (
            gate.energy_accounting_closed,
            "task or core energy accounting did not close",
        ),
    ];
    gate.failures.extend(
        checks
            .into_iter()
            .filter_map(|(passed, label)| (!passed).then_some(label.to_owned())),
    );
}

fn make_delay_line_candidate(
    protector: &ProtectedResidual,
    task: &ConditionalTaskProgram,
    proposal_index: u32,
) -> Result<(OrganismGenome, DelayLineModule), ConditionalProgramError> {
    let mut candidate = protector.seed_extension();
    let start = u32::try_from(candidate.brain.hidden_nodes.len()).map_err(|_| {
        ConditionalProgramError::Integrity("hidden-node count exceeds u32".to_owned())
    })?;
    let delay_len = task
        .rank
        .checked_add(task.empty_delay_ticks)
        .ok_or_else(|| ConditionalProgramError::Integrity("module size overflow".to_owned()))?;
    let delay_nodes = (0..delay_len)
        .map(|offset| {
            start
                .checked_add(offset)
                .map(seed_hidden_gene_node_id)
                .ok_or_else(|| {
                    ConditionalProgramError::Integrity("hidden-node id overflow".to_owned())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for &id in &delay_nodes {
        candidate.brain.hidden_nodes.push(HiddenNodeGene {
            id,
            bias: 0.0,
            log_time_constant: -1.203_972_8,
        });
    }

    let (input_weight, chain_weight, output_weight) = proposal_weights(proposal_index);
    let mut causal_edges = Vec::new();
    let first = delay_nodes[0];
    push_causal_edge(
        &mut candidate,
        &mut causal_edges,
        "cue_left_to_delay_line",
        sensory_gene_node_id(0),
        first,
        input_weight,
    );
    push_causal_edge(
        &mut candidate,
        &mut causal_edges,
        "cue_right_to_delay_line",
        sensory_gene_node_id(2),
        first,
        -input_weight,
    );
    for (index, pair) in delay_nodes.windows(2).enumerate() {
        push_causal_edge(
            &mut candidate,
            &mut causal_edges,
            &format!("delay_transition_{index}_to_{}", index + 1),
            pair[0],
            pair[1],
            chain_weight,
        );
    }
    let memory_source = *delay_nodes.last().expect("module has at least one node");

    // Inclusion-exclusion is an exact paired interaction gate.  For one or
    // zero side rays the four terms cancel algebraically, so the residual does
    // not rewrite ordinary one-food ecology.  Only the task's simultaneous
    // left+right choice scene opens the gate:
    //   tanh(m+L+R-b) - tanh(m+L-b) - tanh(m+R-b) + tanh(m-b).
    // Mirrored copies use opposite memory signs for the two turn actions.
    const CHOICE_WEIGHT: f32 = 1.0;
    let mut hidden_nodes = delay_nodes;
    for (action_label, action_index, memory_sign) in [
        ("turn_left", 0_usize, -1.0_f32),
        ("turn_right", 1_usize, 1.0_f32),
    ] {
        for (term_index, (term_label, has_left, has_right, coefficient)) in [
            ("none", false, false, 1.0_f32),
            ("left", true, false, -1.0_f32),
            ("right", false, true, -1.0_f32),
            ("left_right", true, true, 1.0_f32),
        ]
        .into_iter()
        .enumerate()
        {
            let offset = u32::try_from(hidden_nodes.len()).map_err(|_| {
                ConditionalProgramError::Integrity("hidden-node count exceeds u32".to_owned())
            })?;
            let gate_id = start
                .checked_add(offset)
                .map(seed_hidden_gene_node_id)
                .ok_or_else(|| {
                    ConditionalProgramError::Integrity("gate-node id overflow".to_owned())
                })?;
            candidate.brain.hidden_nodes.push(HiddenNodeGene {
                id: gate_id,
                bias: -CHOICE_WEIGHT,
                log_time_constant: -1.203_972_8,
            });
            hidden_nodes.push(gate_id);
            push_causal_edge(
                &mut candidate,
                &mut causal_edges,
                &format!("memory_to_{action_label}_gate_{term_label}"),
                memory_source,
                gate_id,
                memory_sign * chain_weight,
            );
            if has_left {
                push_causal_edge(
                    &mut candidate,
                    &mut causal_edges,
                    &format!("choice_left_to_{action_label}_gate_{term_label}"),
                    sensory_gene_node_id(0),
                    gate_id,
                    CHOICE_WEIGHT,
                );
            }
            if has_right {
                push_causal_edge(
                    &mut candidate,
                    &mut causal_edges,
                    &format!("choice_right_to_{action_label}_gate_{term_label}"),
                    sensory_gene_node_id(2),
                    gate_id,
                    CHOICE_WEIGHT,
                );
            }
            push_causal_edge(
                &mut candidate,
                &mut causal_edges,
                &format!("{action_label}_gate_{term_index}_to_action"),
                gate_id,
                action_gene_node_id(action_index),
                coefficient * output_weight,
            );
        }
    }
    let mut causal_slices = Vec::new();
    for edge in causal_edges.iter().filter(|edge| {
        edge.label == "cue_left_to_delay_line"
            || edge.label == "cue_right_to_delay_line"
            || edge.label.starts_with("delay_transition_")
    }) {
        causal_slices.push(CausalSlice {
            label: edge.label.clone(),
            innovations: vec![edge.innovation],
        });
    }
    for (label, predicate) in [
        ("all_memory_to_paired_gates", "memory_to_"),
        ("all_choice_scene_inputs_to_paired_gates", "choice_"),
        ("turn_left_paired_gate_outputs", "turn_left_gate_"),
        ("turn_right_paired_gate_outputs", "turn_right_gate_"),
    ] {
        let innovations = causal_edges
            .iter()
            .filter(|edge| {
                edge.label.starts_with(predicate)
                    && if predicate.ends_with("gate_") {
                        edge.label.ends_with("_to_action")
                    } else {
                        true
                    }
            })
            .map(|edge| edge.innovation)
            .collect::<Vec<_>>();
        if innovations.is_empty() {
            return Err(ConditionalProgramError::Integrity(format!(
                "causal slice `{label}` is empty"
            )));
        }
        causal_slices.push(CausalSlice {
            label: label.to_owned(),
            innovations,
        });
    }

    protector.project(&mut candidate);
    protector
        .verify(&candidate)
        .map_err(|error| ConditionalProgramError::Integrity(error.to_string()))?;
    Ok((
        candidate,
        DelayLineModule {
            rank: task.rank,
            hidden_nodes,
            input_weight,
            chain_weight,
            output_weight,
            causal_edges,
            causal_slices,
        },
    ))
}

fn proposal_weights(index: u32) -> (f32, f32, f32) {
    if index == 0 {
        return (1.5, 1.5, 1.5);
    }
    let sample = |domain: u64| {
        let value = mix64(u64::from(index) ^ domain);
        0.8 + (value % 701) as f32 / 1_000.0
    };
    (
        sample(0x494e_5055_545f_5754),
        sample(0x4348_4149_4e5f_5754),
        sample(0x4f55_5450_5554_5754),
    )
}

fn push_causal_edge(
    genome: &mut OrganismGenome,
    causal: &mut Vec<CausalEdge>,
    label: &str,
    pre: GeneNodeId,
    post: GeneNodeId,
    weight: f32,
) {
    let innovation = connection_innovation_id(pre, post);
    genome.brain.edges.push(SynapseGene {
        innovation,
        pre_node_id: pre,
        post_node_id: post,
        weight,
        enabled: true,
    });
    causal.push(CausalEdge {
        label: label.to_owned(),
        innovation,
    });
}

#[derive(Debug, Clone, Copy)]
enum PanelDomain {
    Search,
    Admission,
}

fn make_contexts(
    task: &ConditionalTaskProgram,
    outer_seed: u64,
    domain: PanelDomain,
    nuisance: bool,
) -> Vec<ConditionalContext> {
    let domain_tag = match domain {
        PanelDomain::Search => 0x5345_4152_4348_0001,
        PanelDomain::Admission => 0x4144_4d49_5353_0002,
    };
    (0..(PANEL_SIZE as u32 / 2))
        .flat_map(|pair_index| {
            let world_seed = mix64(
                outer_seed
                    ^ domain_tag
                    ^ task.archive_index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ u64::from(pair_index),
            );
            let sequence_value = mix64(
                outer_seed
                    ^ domain_tag.rotate_left(13)
                    ^ task.archive_index.wrapping_mul(0xbf58_476d_1ce4_e5b9)
                    ^ u64::from(pair_index),
            );
            // Base histories have a zero in the fourth-lowest bit and the
            // three-bit pair index below it. Their complements therefore have
            // a one there, yielding all sixteen rank-4 histories exactly once.
            let base_bits = context_bits(task.rank, pair_index, sequence_value);
            let facing_index = if nuisance {
                ((world_seed >> 9) as usize + 2) % FacingDirection::ALL.len()
            } else {
                (world_seed >> 9) as usize % FacingDirection::ALL.len()
            };
            let width = TASK_WORLD_WIDTH as i32;
            let anchor_q = if nuisance {
                ((world_seed >> 17) as i32 + 3).rem_euclid(width)
            } else {
                ((world_seed >> 17) as i32).rem_euclid(width)
            };
            let anchor_r = if nuisance {
                ((world_seed >> 33) as i32 + 5).rem_euclid(width)
            } else {
                ((world_seed >> 33) as i32).rem_euclid(width)
            };
            [false, true].into_iter().map(move |complement_member| {
                let context_index = pair_index * 2 + u32::from(complement_member);
                let cue_bits = if complement_member {
                    base_bits.iter().map(|bit| !bit).collect()
                } else {
                    base_bits.clone()
                };
                ConditionalContext {
                    context_index,
                    evidence_context_id: mix64(
                        outer_seed
                            ^ domain_tag.rotate_left(29)
                            ^ task.archive_index.wrapping_mul(0xd6e8_feb8_6659_fd93)
                            ^ u64::from(task.rank).rotate_left(17)
                            ^ u64::from(task.empty_delay_ticks).rotate_left(41)
                            ^ u64::from(task.fixed_episode_escrow_bits).rotate_left(7)
                            ^ u64::from(context_index),
                    ),
                    complement_pair_index: pair_index,
                    complement_member,
                    world_seed,
                    cue_bits,
                    anchor_q,
                    anchor_r,
                    facing: FacingDirection::ALL[facing_index],
                    cue_distance: if nuisance { 2 } else { 1 },
                }
            })
        })
        .collect()
}

fn validate_context_panel(
    contexts: &[ConditionalContext],
    label: &str,
) -> Result<(), ConditionalProgramError> {
    if contexts.len() != PANEL_SIZE {
        return Err(ConditionalProgramError::Integrity(format!(
            "{label} context panel has {} rows instead of {PANEL_SIZE}",
            contexts.len()
        )));
    }
    let context_indices = contexts
        .iter()
        .map(|context| context.context_index)
        .collect::<BTreeSet<_>>();
    let evidence_ids = contexts
        .iter()
        .map(|context| context.evidence_context_id)
        .collect::<BTreeSet<_>>();
    let histories = contexts
        .iter()
        .map(|context| &context.cue_bits)
        .collect::<BTreeSet<_>>();
    if context_indices.len() != PANEL_SIZE
        || evidence_ids.len() != PANEL_SIZE
        || histories.len() != PANEL_SIZE
    {
        return Err(ConditionalProgramError::Integrity(format!(
            "{label} context panel does not contain {PANEL_SIZE} unique indices, evidence ids, and histories"
        )));
    }
    Ok(())
}

fn context_bits(rank: u32, context_index: u32, prefix: u64) -> Vec<bool> {
    let rank = rank as usize;
    (0..rank)
        .map(|position| {
            let from_low_unique_suffix = position + 4 >= rank;
            if from_low_unique_suffix {
                let shift = rank - position - 1;
                // Only the lowest three bits carry pair identity; the fourth
                // is zero for base members and becomes one under complement.
                if shift == 3 {
                    false
                } else {
                    ((context_index >> shift) & 1) != 0
                }
            } else {
                let shift = (rank - position - 1) % 64;
                ((prefix >> shift) & 1) != 0
            }
        })
        .collect()
}

fn evaluate_panel(
    task: &ConditionalTaskProgram,
    genome: &OrganismGenome,
    contexts: &[ConditionalContext],
    mode: TrialMode,
    escrow_energy: f32,
    effective_configs: &EffectiveEvaluatorConfigs,
) -> Result<PanelEvidence, ConditionalProgramError> {
    let mut trials = Vec::with_capacity(contexts.len());
    for context in contexts {
        trials.push(evaluate_trial(
            task,
            genome,
            context,
            mode.clone(),
            escrow_energy,
            effective_configs,
        )?);
    }
    let passes = trials.iter().filter(|trial| trial.passed).count() as u32;
    Ok(PanelEvidence {
        mode,
        passes,
        trials,
    })
}

fn evaluate_trial(
    task: &ConditionalTaskProgram,
    genome: &OrganismGenome,
    context: &ConditionalContext,
    mode: TrialMode,
    escrow_energy: f32,
    effective_configs: &EffectiveEvaluatorConfigs,
) -> Result<TrialEvidence, ConditionalProgramError> {
    let world = if matches!(mode, TrialMode::RandomActions) {
        &effective_configs.random_action_task_world
    } else {
        &effective_configs.task_world
    };
    let mut sim = Simulation::new_with_champion_pool(
        world.clone(),
        context.world_seed,
        vec![genome.clone()],
    )?;
    debug_assert_eq!(
        sim.config().force_random_actions,
        world.force_random_actions
    );
    let fresh_brain = sim
        .organisms()
        .first()
        .ok_or_else(|| {
            ConditionalProgramError::Integrity("conditional arena has no founder".to_owned())
        })?
        .brain
        .clone();

    let mut effective_context = context.clone();
    if matches!(mode, TrialMode::NuisancePerturbed) {
        effective_context = nuisance_context(context);
    }
    let actual_bits = context.cue_bits.clone();
    let donor_bits = matches!(
        mode,
        TrialMode::DonorBrainSwapFollowDonor | TrialMode::DonorBrainSwapFollowHost
    )
    .then(|| actual_bits.iter().map(|bit| !bit).collect::<Vec<_>>());
    let donor_state = donor_bits
        .as_ref()
        .map(|bits| {
            advance_donor_to_response_brain(
                task,
                genome,
                &effective_context,
                bits,
                &effective_configs.task_world,
            )
        })
        .transpose()?;
    let replay_bits = vec![false; actual_bits.len()];
    let presented_bits = match mode {
        TrialMode::MatchedCueSymbolEquivarianceReplication => {
            actual_bits.iter().map(|bit| !bit).collect()
        }
        TrialMode::FixedCueReplay => replay_bits,
        TrialMode::CueErased => actual_bits.clone(),
        _ => actual_bits.clone(),
    };
    let expected_bits = match mode {
        TrialMode::MatchedCueSymbolEquivarianceReplication | TrialMode::SemanticPermutation => {
            actual_bits.iter().map(|bit| !bit).collect()
        }
        TrialMode::DonorBrainSwapFollowDonor => {
            donor_bits.clone().expect("donor-follow arm has donor bits")
        }
        _ => actual_bits.clone(),
    };

    let mut escrow = f64::from(escrow_energy);
    let mut trace = BehaviorTrace {
        mode: mode.clone(),
        reset_contract: "before every tick restore q/r/facing, health, damage, last action, task consumption counters, and energy_at_last_sensing; preserve complete BrainState except in named reset/swap arms".to_owned(),
        cue_bits_presented: presented_bits.clone(),
        expected_response_bits: expected_bits.clone(),
        donor_cue_bits: donor_bits,
        donor_state_evidence: donor_state.as_ref().map(|donor| donor.evidence.clone()),
        ticks: Vec::new(),
        interventions: Vec::new(),
    };

    for (index, bit) in presented_bits.iter().copied().enumerate() {
        let rays: Vec<i8> = if matches!(mode, TrialMode::CueErased) {
            Vec::new()
        } else if bit {
            vec![1]
        } else {
            vec![-1]
        };
        let intervention = prepare_phase(
            &mut sim,
            &effective_context,
            &rays,
            &mut escrow,
            0.0,
            format!("cue_{index}"),
        )?;
        trace.interventions.push(intervention);
        sim.tick();
        trace
            .ticks
            .push(capture_tick(&sim, "cue", index as u32, None, None, None)?);
    }

    if matches!(mode, TrialMode::FullBrainReset) {
        trace.interventions.push(replace_full_brain_state(
            &mut sim,
            &fresh_brain,
            escrow,
            "post_cue_full_brain_reset",
        )?);
    }

    for index in 0..task.empty_delay_ticks {
        let intervention = prepare_phase(
            &mut sim,
            &effective_context,
            &[],
            &mut escrow,
            0.0,
            format!("empty_delay_{index}"),
        )?;
        trace.interventions.push(intervention);
        sim.tick();
        trace
            .ticks
            .push(capture_tick(&sim, "empty_delay", index, None, None, None)?);
    }

    if let Some(donor) = donor_state.as_ref() {
        trace.interventions.push(replace_full_brain_state(
            &mut sim,
            &donor.brain,
            escrow,
            "pre_response_full_brain_swap_from_complement_cue",
        )?);
    }

    let mut correct_responses = 0_u32;
    for (index, expected_bit) in expected_bits.iter().copied().enumerate() {
        let intervention = prepare_phase(
            &mut sim,
            &effective_context,
            &[-1, 1],
            &mut escrow,
            0.0,
            format!("identical_choice_{index}"),
        )?;
        trace.interventions.push(intervention);
        sim.tick();
        let expected_action = if expected_bit {
            ActionType::TurnRight
        } else {
            ActionType::TurnLeft
        };
        let selected = sim
            .organisms()
            .first()
            .ok_or_else(|| {
                ConditionalProgramError::Integrity(
                    "task organism died despite zero task metabolism/cost".to_owned(),
                )
            })?
            .last_action_taken;
        let expected_facing =
            rotate_by_steps(effective_context.facing, if expected_bit { 1 } else { -1 });
        let committed_facing = sim
            .organisms()
            .first()
            .expect("organism existence checked above")
            .facing;
        let correct = selected == expected_action && committed_facing == expected_facing;
        correct_responses += u32::from(correct);
        trace.ticks.push(capture_tick(
            &sim,
            "identical_choice",
            index as u32,
            Some(expected_action),
            Some(expected_facing),
            Some(correct),
        )?);
    }

    let passed_actions = correct_responses == task.rank;
    let reward = if passed_actions { escrow } else { 0.0 };
    let reward_intervention = prepare_phase(
        &mut sim,
        &effective_context,
        &[],
        &mut escrow,
        reward,
        "fixed_escrow_release".to_owned(),
    )?;
    trace.interventions.push(reward_intervention);
    let behavior_fingerprint = fingerprint(&trace)?;
    Ok(TrialEvidence {
        context: effective_context,
        passed: passed_actions && reward > 0.0 && escrow == 0.0,
        correct_responses,
        total_responses: task.rank,
        reward_released: reward > 0.0,
        final_escrow_energy_bits: (escrow as f32).to_bits(),
        behavior_fingerprint,
        trace,
    })
}

fn advance_donor_to_response_brain(
    task: &ConditionalTaskProgram,
    genome: &OrganismGenome,
    context: &ConditionalContext,
    donor_bits: &[bool],
    task_world_config: &WorldConfig,
) -> Result<PreparedDonorState, ConditionalProgramError> {
    let mut sim = Simulation::new_with_champion_pool(
        task_world_config.clone(),
        context.world_seed,
        vec![genome.clone()],
    )?;
    let mut escrow = f64::from(f32::from_bits(task.fixed_episode_escrow_bits));
    let mut ticks = Vec::with_capacity(donor_bits.len() + task.empty_delay_ticks as usize);
    let mut interventions = Vec::with_capacity(ticks.capacity());
    for (index, bit) in donor_bits.iter().copied().enumerate() {
        let ray = if bit { 1 } else { -1 };
        interventions.push(prepare_phase(
            &mut sim,
            context,
            &[ray],
            &mut escrow,
            0.0,
            format!("donor_cue_{index}"),
        )?);
        sim.tick();
        ticks.push(capture_tick(
            &sim,
            "donor_cue",
            index as u32,
            None,
            None,
            None,
        )?);
    }
    for index in 0..task.empty_delay_ticks {
        interventions.push(prepare_phase(
            &mut sim,
            context,
            &[],
            &mut escrow,
            0.0,
            format!("donor_empty_delay_{index}"),
        )?);
        sim.tick();
        ticks.push(capture_tick(
            &sim,
            "donor_empty_delay",
            index,
            None,
            None,
            None,
        )?);
    }
    let brain = sim
        .organisms()
        .first()
        .map(|organism| organism.brain.clone())
        .ok_or_else(|| {
            ConditionalProgramError::Integrity(
                "donor task organism died despite zero metabolism/cost".to_owned(),
            )
        })?;
    let brain_state_fingerprint = fingerprint(&brain)?;
    Ok(PreparedDonorState {
        brain,
        evidence: DonorStateEvidence {
            world_seed: context.world_seed,
            cue_bits: donor_bits.to_vec(),
            anchor_q: context.anchor_q,
            anchor_r: context.anchor_r,
            facing: context.facing,
            cue_distance: context.cue_distance,
            prepared_at_turn: sim.turn(),
            brain_state_fingerprint,
            ticks,
            interventions,
        },
    })
}

fn replace_full_brain_state(
    sim: &mut Simulation,
    brain: &BrainState,
    locked_escrow: f64,
    label: &str,
) -> Result<TaskEnergyIntervention, ConditionalProgramError> {
    let organism_before = organism_energy(sim)?;
    let food_before = food_energy(sim)?;
    let organism = sim.organisms.first_mut().ok_or_else(|| {
        ConditionalProgramError::Integrity("brain intervention has no organism".to_owned())
    })?;
    organism.brain = brain.clone();
    organism.energy_at_last_sensing = organism.energy;
    organism.last_action_taken = ActionType::Idle;
    organism.damage_taken_last_turn = 0.0;
    let organism_after = organism_energy(sim)?;
    let food_after = food_energy(sim)?;
    let total_residual = organism_after + food_after + locked_escrow
        - (organism_before + food_before + locked_escrow);
    let scale = organism_before.abs()
        + organism_after.abs()
        + food_before.abs()
        + food_after.abs()
        + 2.0 * locked_escrow.abs();
    let residual_tolerance = 32.0 * f64::from(f32::EPSILON) * scale.max(1.0);
    if !total_residual.is_finite() || total_residual.abs() > residual_tolerance {
        return Err(ConditionalProgramError::Integrity(format!(
            "brain intervention `{label}` changed task energy: residual {total_residual}, tolerance {residual_tolerance}"
        )));
    }
    Ok(TaskEnergyIntervention {
        label: label.to_owned(),
        organism_before,
        food_before,
        locked_escrow_before: locked_escrow,
        organism_after,
        food_after,
        locked_escrow_after: locked_escrow,
        released_energy: 0.0,
        standing_task_food_energy: food_after,
        captured_by_organism: 0.0,
        release_transfer_residual: 0.0,
        total_residual,
        residual_tolerance,
    })
}

fn task_world(force_random_actions: bool) -> WorldConfig {
    let mut world = WorldConfig::test_fixture();
    world.world_width = TASK_WORLD_WIDTH;
    world.num_organisms = 1;
    world.food_tile_fraction = 0.0;
    world.terrain_threshold = 1.0;
    world.passive_metabolism_cost_per_unit = 0.0;
    world.body_mass_metabolic_cost_coeff = 0.0;
    world.move_action_energy_cost = 0.0;
    world.action_temperature = 0.01;
    world.intent_parallel_threads = 1;
    world.runtime_plasticity_enabled = false;
    world.leaky_neurons_enabled = false;
    world.predation_enabled = false;
    world.force_random_actions = force_random_actions;
    world
}

fn effective_evaluator_configs() -> Result<EffectiveEvaluatorConfigs, ConditionalProgramError> {
    let task_world_config = task_world(false);
    let random_action_task_world = task_world(true);
    let ecology_world = ecology_world()?;
    Ok(EffectiveEvaluatorConfigs {
        task_world_fingerprint: fingerprint(&task_world_config)?,
        random_action_task_world_fingerprint: fingerprint(&random_action_task_world)?,
        ecology_world_fingerprint: fingerprint(&ecology_world)?,
        task_world: task_world_config,
        random_action_task_world,
        ecology_world,
    })
}

fn ecology_world() -> Result<WorldConfig, ConditionalProgramError> {
    let mut world = sim_config::load_default_world_config()
        .map_err(|error| ConditionalProgramError::Integrity(error.to_string()))?;
    world.world_width = 25;
    world.num_organisms = 1;
    world.intent_parallel_threads = 1;
    world.runtime_plasticity_enabled = false;
    world.leaky_neurons_enabled = false;
    world.predation_enabled = false;
    world.force_random_actions = false;
    Ok(world)
}

fn nuisance_context(context: &ConditionalContext) -> ConditionalContext {
    let mut changed = context.clone();
    changed.anchor_q = (changed.anchor_q + 3).rem_euclid(TASK_WORLD_WIDTH as i32);
    changed.anchor_r = (changed.anchor_r + 5).rem_euclid(TASK_WORLD_WIDTH as i32);
    changed.facing = rotate_by_steps(changed.facing, 2);
    changed.cue_distance = 2;
    changed
}

fn prepare_phase(
    sim: &mut Simulation,
    context: &ConditionalContext,
    ray_offsets: &[i8],
    escrow: &mut f64,
    escrow_to_organism: f64,
    label: String,
) -> Result<TaskEnergyIntervention, ConditionalProgramError> {
    let organism_before = organism_energy(sim)?;
    let food_before = food_energy(sim)?;
    let escrow_before = *escrow;

    // Remove prior zero-energy task markers and every regrowth hook.  The task
    // world starts with no fertile cells, so a positive-energy removal would
    // be an evaluator integrity failure rather than an allowed sink.
    if food_before != 0.0 {
        return Err(ConditionalProgramError::Integrity(format!(
            "task phase `{label}` tried to clear nonzero food energy {food_before}"
        )));
    }
    for slot in &mut sim.occupancy {
        if matches!(slot, Some(Occupant::Food(_)) | Some(Occupant::Organism(_))) {
            *slot = None;
        }
    }
    sim.foods.clear();
    sim.food_tiles.fill(false);
    sim.food_regrowth_schedule.clear();
    sim.food_regrowth_due_turn.fill(u64::MAX);

    let anchor_idx = sim.cell_index(context.anchor_q, context.anchor_r);
    if sim.terrain_map[anchor_idx] {
        return Err(ConditionalProgramError::Integrity(
            "task anchor intersects blocked terrain".to_owned(),
        ));
    }
    let organism_id = {
        let organism = sim.organisms.first_mut().ok_or_else(|| {
            ConditionalProgramError::Integrity("conditional arena has no organism".to_owned())
        })?;
        organism.q = context.anchor_q;
        organism.r = context.anchor_r;
        organism.facing = context.facing;
        organism.health = organism.max_health;
        organism.damage_taken_last_turn = 0.0;
        organism.last_action_taken = ActionType::Idle;
        organism.consumptions_count = 0;
        organism.plant_consumptions_count = 0;
        organism.prey_consumptions_count = 0;
        organism.energy_at_last_sensing = organism.energy;
        if escrow_to_organism > 0.0 {
            if escrow_to_organism > *escrow {
                return Err(ConditionalProgramError::Integrity(
                    "escrow release exceeds remaining fixed episode escrow".to_owned(),
                ));
            }
            organism.energy += escrow_to_organism as f32;
            *escrow -= escrow_to_organism;
        }
        organism.id
    };
    sim.occupancy[anchor_idx] = Some(Occupant::Organism(organism_id));

    for &offset in ray_offsets {
        let direction = rotate_by_steps(context.facing, offset);
        let mut position = (context.anchor_q, context.anchor_r);
        for _ in 0..context.cue_distance {
            position = hex_neighbor(position, direction, TASK_WORLD_WIDTH as i32);
        }
        let idx = sim.cell_index(position.0, position.1);
        if sim.occupancy[idx].is_some() || sim.terrain_map[idx] {
            return Err(ConditionalProgramError::Integrity(format!(
                "task food placement collision at ({}, {})",
                position.0, position.1
            )));
        }
        let id = FoodId(sim.next_food_id);
        sim.next_food_id = sim.next_food_id.checked_add(1).ok_or_else(|| {
            ConditionalProgramError::Integrity("task food id overflow".to_owned())
        })?;
        let food = FoodState {
            id,
            q: position.0,
            r: position.1,
            energy: 0.0,
            kind: FoodKind::Plant,
            visual: food_visual(FoodKind::Plant),
        };
        sim.occupancy[idx] = Some(Occupant::Food(id));
        sim.foods.push(food);
    }
    sim.debug_assert_consistent_state();

    let organism_after = organism_energy(sim)?;
    let food_after = food_energy(sim)?;
    let escrow_after = *escrow;
    let captured_by_organism = organism_after - organism_before;
    let release_transfer_residual = captured_by_organism - escrow_to_organism;
    let total_residual = organism_after + food_after + escrow_after
        - (organism_before + food_before + escrow_before);
    let scale = organism_before.abs()
        + food_before.abs()
        + escrow_before.abs()
        + organism_after.abs()
        + food_after.abs()
        + escrow_after.abs()
        + escrow_to_organism.abs();
    let residual_tolerance = 32.0 * f64::from(f32::EPSILON) * scale.max(1.0);
    if escrow_to_organism > 0.0
        && (!escrow_to_organism.is_finite()
            || captured_by_organism <= 0.0
            || escrow_after != 0.0
            || release_transfer_residual.abs() > residual_tolerance)
    {
        return Err(ConditionalProgramError::Integrity(format!(
            "task reward `{label}` was not a positive all-or-nothing captured transfer: released {escrow_to_organism}, captured {captured_by_organism}, locked after {escrow_after}, transfer residual {release_transfer_residual}, tolerance {residual_tolerance}"
        )));
    }
    if !total_residual.is_finite() || total_residual.abs() > residual_tolerance {
        return Err(ConditionalProgramError::Integrity(format!(
            "task energy intervention `{label}` does not close: residual {total_residual}, tolerance {residual_tolerance}"
        )));
    }
    Ok(TaskEnergyIntervention {
        label,
        organism_before,
        food_before,
        locked_escrow_before: escrow_before,
        organism_after,
        food_after,
        locked_escrow_after: escrow_after,
        released_energy: escrow_to_organism,
        standing_task_food_energy: food_after,
        captured_by_organism,
        release_transfer_residual,
        total_residual,
        residual_tolerance,
    })
}

fn capture_tick(
    sim: &Simulation,
    phase: &str,
    phase_index: u32,
    expected_action: Option<ActionType>,
    expected_facing_after_commit: Option<FacingDirection>,
    action_correct: Option<bool>,
) -> Result<ConditionalTickTrace, ConditionalProgramError> {
    let organism = sim.organisms().first().ok_or_else(|| {
        ConditionalProgramError::Integrity("conditional task organism died".to_owned())
    })?;
    let deterministic_action_sample_tick = sim.turn().saturating_sub(1);
    let action_sample =
        deterministic_action_sample(sim.seed, deterministic_action_sample_tick, organism.id);
    Ok(ConditionalTickTrace {
        phase: phase.to_owned(),
        phase_index,
        turn: sim.turn(),
        selected_action: organism.last_action_taken,
        expected_action,
        expected_facing_after_commit,
        action_correct,
        q_after_tick: organism.q,
        r_after_tick: organism.r,
        facing_after_tick: organism.facing,
        organism_energy_bits: organism.energy.to_bits(),
        sensory: organism
            .brain
            .sensory
            .iter()
            .map(|sensor| SensoryActivation {
                runtime_id: sensor.neuron.neuron_id.0,
                receptor: sensor.receptor,
                activation: sensor.neuron.activation,
                activation_bits: sensor.neuron.activation.to_bits(),
            })
            .collect(),
        food_ray_activation_bits: organism
            .brain
            .sensory
            .iter()
            .take(3)
            .map(|sensor| sensor.neuron.activation.to_bits())
            .collect(),
        hidden: organism
            .brain
            .inter
            .iter()
            .map(|node| HiddenActivation {
                runtime_id: node.neuron.neuron_id.0,
                state_bits: node.state.to_bits(),
                activation_bits: node.neuron.activation.to_bits(),
            })
            .collect(),
        action_logits: organism
            .brain
            .action
            .iter()
            .map(|action| ActionLogitTrace {
                runtime_id: action.neuron_id.0,
                action_type: action.action_type,
                logit: action.logit,
                logit_bits: action.logit.to_bits(),
            })
            .collect(),
        action_temperature: sim.config().action_temperature,
        action_temperature_bits: sim.config().action_temperature.to_bits(),
        deterministic_action_sample_tick,
        deterministic_action_sample: action_sample,
        deterministic_action_sample_bits: action_sample.to_bits(),
        force_random_actions: sim.config().force_random_actions,
        core_energy_ledger: sim.metrics().energy_ledger_last_turn,
    })
}

fn organism_energy(sim: &Simulation) -> Result<f64, ConditionalProgramError> {
    finite_sum(
        "organism energy",
        sim.organisms.iter().map(|organism| organism.energy),
    )
}

fn food_energy(sim: &Simulation) -> Result<f64, ConditionalProgramError> {
    finite_sum("food energy", sim.foods.iter().map(|food| food.energy))
}

fn finite_sum(
    label: &str,
    values: impl Iterator<Item = f32>,
) -> Result<f64, ConditionalProgramError> {
    let mut total = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return Err(ConditionalProgramError::Integrity(format!(
                "nonfinite {label}: {value}"
            )));
        }
        total += f64::from(value);
    }
    if !total.is_finite() {
        return Err(ConditionalProgramError::Integrity(format!(
            "nonfinite {label} total"
        )));
    }
    Ok(total)
}

fn score_panel(
    solver: &ArchivedSolverRuntime,
    task_archive_index: u64,
    panel: &str,
    evidence: &PanelEvidence,
) -> SolverPanelScore {
    SolverPanelScore {
        solver_label: solver.label.clone(),
        controller_fingerprint: solver.fingerprint.clone(),
        task_archive_index,
        panel: panel.to_owned(),
        passes: evidence.passes,
    }
}

fn checkpoint_from_panel(
    task: &ConditionalTaskProgram,
    task_fingerprint: &str,
    genome: &OrganismGenome,
    panel: &PanelEvidence,
) -> Result<TaskCheckpoint, ConditionalProgramError> {
    Ok(TaskCheckpoint {
        requirement: RetentionRequirementHeader {
            task_id: task.archive_index,
            minimum_passes: MIN_ADMISSION_PASSES,
        },
        task_fingerprint: task_fingerprint.to_owned(),
        accepted_controller_fingerprint: fingerprint(genome)?,
        trial_seeds: panel
            .trials
            .iter()
            .map(|trial| trial.context.evidence_context_id)
            .collect(),
        accepted_passes: panel.passes,
        accepted_behavior_fingerprints: panel
            .trials
            .iter()
            .map(|trial| trial.behavior_fingerprint.clone())
            .collect(),
    })
}

fn replay_from_panel(
    task: &ConditionalTaskProgram,
    task_fingerprint: &str,
    genome: &OrganismGenome,
    panel: &PanelEvidence,
) -> Result<TaskReplay, ConditionalProgramError> {
    Ok(TaskReplay {
        task_id: task.archive_index,
        task_fingerprint: task_fingerprint.to_owned(),
        controller_fingerprint: fingerprint(genome)?,
        trial_seeds: panel
            .trials
            .iter()
            .map(|trial| trial.context.evidence_context_id)
            .collect(),
        passes: panel.passes,
        behavior_fingerprints: panel
            .trials
            .iter()
            .map(|trial| trial.behavior_fingerprint.clone())
            .collect(),
    })
}

fn append_matrix(
    matrix: &mut Vec<MatrixCell>,
    solver: &ArchivedSolverRuntime,
    task: &ConditionalTaskProgram,
    task_fingerprint: &str,
    panel_label: &str,
    panel: &PanelEvidence,
) {
    matrix.extend(panel.trials.iter().map(|trial| MatrixCell {
        solver_label: solver.label.clone(),
        controller_fingerprint: solver.fingerprint.clone(),
        task_archive_index: task.archive_index,
        task_fingerprint: task_fingerprint.to_owned(),
        panel: panel_label.to_owned(),
        context_index: trial.context.context_index,
        evidence_context_id: trial.context.evidence_context_id,
        complement_pair_index: trial.context.complement_pair_index,
        complement_member: trial.context.complement_member,
        world_seed: trial.context.world_seed,
        passed: trial.passed,
        behavior_fingerprint: trial.behavior_fingerprint.clone(),
    }));
}

fn build_crossplay(
    solvers: &[ArchivedSolverRuntime],
    tasks: &[ArchivedTaskRuntime],
    default_escrow: f32,
    effective_configs: &EffectiveEvaluatorConfigs,
) -> Result<Vec<MatrixCell>, ConditionalProgramError> {
    let mut matrix = Vec::new();
    for task in tasks {
        for solver in solvers {
            let escrow = f32::from_bits(task.record.task.fixed_episode_escrow_bits);
            let panel = evaluate_panel(
                &task.record.task,
                &solver.genome,
                &task.record.contexts,
                TrialMode::Normal,
                if escrow.is_finite() && escrow > 0.0 {
                    escrow
                } else {
                    default_escrow
                },
                effective_configs,
            )?;
            append_matrix(
                &mut matrix,
                solver,
                &task.record.task,
                &task.record.task_fingerprint,
                "final_all_history_crossplay",
                &panel,
            );
        }
    }
    Ok(matrix)
}

fn count_fully_correct_complement_pairs(
    panel: &PanelEvidence,
) -> Result<u32, ConditionalProgramError> {
    let mut count = 0_u32;
    for pair_index in 0..(PANEL_SIZE as u32 / 2) {
        let members = panel
            .trials
            .iter()
            .filter(|trial| trial.context.complement_pair_index == pair_index)
            .collect::<Vec<_>>();
        if members.len() != 2 {
            return Err(ConditionalProgramError::Integrity(format!(
                "complement pair {pair_index} has {} members instead of two",
                members.len()
            )));
        }
        let (base, complement) = if !members[0].context.complement_member
            && members[1].context.complement_member
        {
            (members[0], members[1])
        } else if !members[1].context.complement_member && members[0].context.complement_member {
            (members[1], members[0])
        } else {
            return Err(ConditionalProgramError::Integrity(format!(
                "complement pair {pair_index} does not contain one base and one complement member"
            )));
        };
        let paired_external_contract = base.context.world_seed == complement.context.world_seed
            && base.context.anchor_q == complement.context.anchor_q
            && base.context.anchor_r == complement.context.anchor_r
            && base.context.facing == complement.context.facing
            && base.context.cue_distance == complement.context.cue_distance
            && base.context.evidence_context_id != complement.context.evidence_context_id
            && base.context.cue_bits.len() == complement.context.cue_bits.len()
            && base
                .context
                .cue_bits
                .iter()
                .zip(&complement.context.cue_bits)
                .all(|(left, right)| left != right);
        if !paired_external_contract {
            return Err(ConditionalProgramError::Integrity(format!(
                "complement pair {pair_index} violates the paired world/action-stream contract"
            )));
        }
        let base_actions = response_actions(&base.trace);
        let complement_actions = response_actions(&complement.trace);
        let actions_are_complements = base_actions.len() == complement_actions.len()
            && base_actions
                .iter()
                .zip(&complement_actions)
                .all(|(left, right)| {
                    matches!(
                        (left, right),
                        (ActionType::TurnLeft, ActionType::TurnRight)
                            | (ActionType::TurnRight, ActionType::TurnLeft)
                    )
                });
        if base.passed && complement.passed && actions_are_complements {
            count += 1;
        }
    }
    Ok(count)
}

fn audit_normal_response_action_margins(
    task: &ConditionalTaskProgram,
    normal: &PanelEvidence,
) -> ActionMarginAudit {
    let expected_normal_response_ticks = normal.trials.len() as u32 * task.rank;
    let mut observed_normal_response_ticks = 0_u32;
    let mut every_tick_has_complete_action_logits = true;
    let mut every_selected_action_is_unique_argmax = true;
    let mut minimum_observed_raw_logit_margin: Option<f32> = None;
    let mut maximum_observed_raw_logit_margin: Option<f32> = None;
    let mut minimum_deterministic_action_sample: Option<f32> = None;
    let mut maximum_deterministic_action_sample: Option<f32> = None;

    for tick in normal
        .trials
        .iter()
        .flat_map(|trial| &trial.trace.ticks)
        .filter(|tick| tick.phase == "identical_choice")
    {
        observed_normal_response_ticks += 1;
        let complete = tick.action_logits.len() == ActionType::ALL.len()
            && tick
                .action_logits
                .iter()
                .all(|logit| logit.logit.is_finite())
            && ActionType::ALL.iter().all(|action| {
                tick.action_logits
                    .iter()
                    .filter(|logit| logit.action_type == *action)
                    .count()
                    == 1
            });
        every_tick_has_complete_action_logits &= complete;

        let Some(selected) = tick
            .action_logits
            .iter()
            .find(|logit| logit.action_type == tick.selected_action)
        else {
            every_selected_action_is_unique_argmax = false;
            continue;
        };
        let mut next_highest = EXPLICIT_IDLE_LOGIT_BIAS;
        for competitor in tick
            .action_logits
            .iter()
            .filter(|logit| logit.action_type != tick.selected_action)
        {
            next_highest = next_highest.max(competitor.logit);
        }
        let margin = selected.logit - next_highest;
        if !margin.is_finite() {
            every_selected_action_is_unique_argmax = false;
        } else {
            every_selected_action_is_unique_argmax &= margin > 0.0;
            minimum_observed_raw_logit_margin = Some(
                minimum_observed_raw_logit_margin.map_or(margin, |current| current.min(margin)),
            );
            maximum_observed_raw_logit_margin = Some(
                maximum_observed_raw_logit_margin.map_or(margin, |current| current.max(margin)),
            );
        }
        let sample = tick.deterministic_action_sample;
        minimum_deterministic_action_sample =
            Some(minimum_deterministic_action_sample.map_or(sample, |current| current.min(sample)));
        maximum_deterministic_action_sample =
            Some(maximum_deterministic_action_sample.map_or(sample, |current| current.max(sample)));
    }

    let accepted = observed_normal_response_ticks == expected_normal_response_ticks
        && every_tick_has_complete_action_logits
        && every_selected_action_is_unique_argmax
        && minimum_observed_raw_logit_margin
            .is_some_and(|margin| margin >= MIN_NORMAL_RESPONSE_LOGIT_MARGIN);
    ActionMarginAudit {
        expected_normal_response_ticks,
        observed_normal_response_ticks,
        every_tick_has_complete_action_logits,
        every_selected_action_is_unique_argmax,
        minimum_required_raw_logit_margin: MIN_NORMAL_RESPONSE_LOGIT_MARGIN,
        minimum_observed_raw_logit_margin,
        maximum_observed_raw_logit_margin,
        minimum_deterministic_action_sample,
        maximum_deterministic_action_sample,
        accepted,
    }
}

fn response_actions(trace: &BehaviorTrace) -> Vec<ActionType> {
    trace
        .ticks
        .iter()
        .filter(|tick| tick.phase == "identical_choice")
        .map(|tick| tick.selected_action)
        .collect()
}

fn panel_energy_closed(panel: &PanelEvidence) -> bool {
    panel.trials.iter().all(|trial| {
        trial
            .trace
            .interventions
            .iter()
            .all(task_intervention_energy_closed)
            && trial.trace.ticks.iter().all(tick_energy_closed)
            && trial
                .trace
                .donor_state_evidence
                .as_ref()
                .is_none_or(|donor| {
                    donor
                        .interventions
                        .iter()
                        .all(task_intervention_energy_closed)
                        && donor.ticks.iter().all(tick_energy_closed)
                })
    })
}

fn task_intervention_energy_closed(row: &TaskEnergyIntervention) -> bool {
    row.total_residual.is_finite()
        && row.total_residual.abs() <= row.residual_tolerance
        && row.release_transfer_residual.abs() <= row.residual_tolerance
        && (row.released_energy == 0.0
            || (row.released_energy > 0.0
                && row.captured_by_organism > 0.0
                && row.locked_escrow_after == 0.0))
}

fn tick_energy_closed(tick: &ConditionalTickTrace) -> bool {
    let row = tick.core_energy_ledger;
    row.organism_residual.abs() <= row.residual_tolerance
        && row.food_residual.abs() <= row.residual_tolerance
        && row.transfer_residual.abs() <= row.residual_tolerance
        && row.total_residual.abs() <= row.residual_tolerance
}

fn ecology_noninferiority(
    horizon: u32,
    outer_seed: u64,
    preceding: &OrganismGenome,
    candidate: &OrganismGenome,
    ecology_world_config: &WorldConfig,
) -> Result<EcologyNoninferiority, ConditionalProgramError> {
    let seeds = (0..4)
        .map(|index| mix64(outer_seed ^ 0x4543_4f4c_4f47_5900 ^ index))
        .collect::<Vec<_>>();
    let preceding_controller = seeds
        .iter()
        .map(|&seed| run_ecology_trial(preceding, seed, horizon, ecology_world_config))
        .collect::<Result<Vec<_>, _>>()?;
    let candidate_controller = seeds
        .iter()
        .map(|&seed| run_ecology_trial(candidate, seed, horizon, ecology_world_config))
        .collect::<Result<Vec<_>, _>>()?;
    let pairs = preceding_controller
        .into_iter()
        .zip(candidate_controller)
        .map(|(preceding, candidate)| {
            if preceding.seed != candidate.seed {
                return Err(ConditionalProgramError::Integrity(
                    "paired ecology trials have mismatched seeds".to_owned(),
                ));
            }
            let survivor_ticks_noninferior = candidate.survivor_ticks >= preceding.survivor_ticks;
            let plant_consumptions_noninferior =
                candidate.plant_consumptions >= preceding.plant_consumptions;
            let energy_scale = preceding
                .final_organism_energy
                .abs()
                .max(candidate.final_organism_energy.abs())
                .max(1.0);
            let final_organism_energy_tolerance = 32.0 * f64::from(f32::EPSILON) * energy_scale;
            let final_organism_energy_noninferior = candidate.final_organism_energy
                + final_organism_energy_tolerance
                >= preceding.final_organism_energy;
            Ok(EcologyPairAudit {
                seed: preceding.seed,
                preceding,
                candidate,
                survivor_ticks_noninferior,
                plant_consumptions_noninferior,
                final_organism_energy_tolerance,
                final_organism_energy_noninferior,
                accepted: survivor_ticks_noninferior
                    && plant_consumptions_noninferior
                    && final_organism_energy_noninferior,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let preceding_survivor_ticks = pairs.iter().map(|pair| pair.preceding.survivor_ticks).sum();
    let candidate_survivor_ticks = pairs.iter().map(|pair| pair.candidate.survivor_ticks).sum();
    let preceding_plant_consumptions = pairs
        .iter()
        .map(|pair| pair.preceding.plant_consumptions)
        .sum();
    let candidate_plant_consumptions = pairs
        .iter()
        .map(|pair| pair.candidate.plant_consumptions)
        .sum();
    let aggregate_survivor_ticks_noninferior = candidate_survivor_ticks >= preceding_survivor_ticks;
    let aggregate_plant_consumption_noninferior =
        candidate_plant_consumptions >= preceding_plant_consumptions;
    let preceding_competence_floor_met =
        preceding_plant_consumptions >= MIN_PRECEDING_ECOLOGY_PLANT_CAPTURES;
    let every_seed_pair_noninferior = pairs.iter().all(|pair| pair.accepted);
    Ok(EcologyNoninferiority {
        horizon,
        pairs,
        minimum_preceding_plant_captures: MIN_PRECEDING_ECOLOGY_PLANT_CAPTURES,
        preceding_competence_floor_met,
        every_seed_pair_noninferior,
        preceding_survivor_ticks,
        candidate_survivor_ticks,
        preceding_plant_consumptions,
        candidate_plant_consumptions,
        aggregate_survivor_ticks_noninferior,
        aggregate_plant_consumption_noninferior,
        accepted: preceding_competence_floor_met && every_seed_pair_noninferior,
    })
}

fn run_ecology_trial(
    genome: &OrganismGenome,
    seed: u64,
    horizon: u32,
    ecology_world_config: &WorldConfig,
) -> Result<EcologyTrial, ConditionalProgramError> {
    let mut sim = Simulation::new_with_champion_pool(
        ecology_world_config.clone(),
        seed,
        vec![genome.clone()],
    )?;
    let mut survivor_ticks = 0_u64;
    let mut max_residual = 0.0_f64;
    let mut max_tolerance = 0.0_f64;
    for _ in 0..horizon {
        sim.tick();
        survivor_ticks += sim.organisms().len() as u64;
        let row = sim.metrics().energy_ledger_last_turn;
        max_residual = max_residual
            .max(row.organism_residual.abs())
            .max(row.food_residual.abs())
            .max(row.transfer_residual.abs())
            .max(row.total_residual.abs());
        max_tolerance = max_tolerance.max(row.residual_tolerance);
    }
    Ok(EcologyTrial {
        seed,
        survivor_ticks,
        final_population: sim.organisms().len() as u32,
        plant_consumptions: sim.metrics().total_plant_consumptions,
        final_organism_energy: sim
            .organisms()
            .iter()
            .map(|organism| f64::from(organism.energy))
            .sum(),
        maximum_core_energy_residual: max_residual,
        maximum_core_energy_tolerance: max_tolerance,
    })
}

fn ecology_energy_closed(evidence: &EcologyNoninferiority) -> bool {
    evidence
        .pairs
        .iter()
        .flat_map(|pair| [&pair.preceding, &pair.candidate])
        .all(|trial| trial.maximum_core_energy_residual <= trial.maximum_core_energy_tolerance)
}

fn fingerprint<T: Serialize>(value: &T) -> Result<String, ConditionalProgramError> {
    exact_fingerprint(value).map_err(|error| ConditionalProgramError::Integrity(error.to_string()))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
