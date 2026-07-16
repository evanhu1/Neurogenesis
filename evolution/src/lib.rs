//! Canonical, generational NEAT: the sole evolutionary system for NeuroGenesis.
//!
//! This module owns species and innovation history. The simulation
//! (`world_sim::Simulation`) is a deterministic fitness evaluator; NEAT is the outer
//! loop. Candidates are evaluated in pairwise or shared multi-genome arenas
//! under fixed world seeds. In-world reproduction does not
//! exist; all genetic variation is owned by the outer loop.

use anyhow::{anyhow, bail, Result};
use brain::genome::MAX_INTER_NEURONS;
use config::WorldConfig;
#[cfg(feature = "instrumentation")]
use metrics::{derive_interval_metrics, Ledger};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use types::{
    action_gene_node_id, is_hidden_gene_node_id, sensory_gene_node_id, split_hidden_gene_node_id,
    ActionType, GeneNodeId, HiddenNodeGene, InnovationId, OrganismGenome, OrganismId,
    SensoryReceptor, SynapseGene, SynapseTiming,
};
use world_sim::{AttackOutcome, Simulation};

const WEIGHT_MIN_ABS: f32 = 0.001;
const WEIGHT_MAX_ABS: f32 = 1.5;
const BIAS_MAX_ABS: f32 = 1.0;
const BREED_SELECTION_DOMAIN: u64 = 0x4252_4545_445f_5345;
const CROSSOVER_DOMAIN: u64 = 0x4352_4f53_534f_5645;
const MUTATION_DOMAIN: u64 = 0x4d55_5441_5449_4f4e;
const OPPONENT_DOMAIN: u64 = 0x4f50_504f_4e45_4e54;
const ARENA_SLOT_DOMAIN: u64 = 0x4152_454e_415f_534c;
pub const OBJECTIVE_NAME: &str = "contextual_lower_tail_survival_fraction";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationTopology {
    Pairwise,
    SharedPopulation,
}

impl EvaluationTopology {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "pairwise" => Ok(Self::Pairwise),
            "shared" | "shared_population" | "shared-population" => Ok(Self::SharedPopulation),
            other => {
                bail!("unknown evaluator topology `{other}`; valid: pairwise shared_population")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatConfig {
    pub population_size: usize,
    pub generations: u32,
    /// Persist the complete evaluated population every N generations. Compact
    /// metrics and the same-generation winner are still retained every generation.
    pub population_checkpoint_interval: u32,
    pub episode_horizons: Vec<u64>,
    /// Positive per-window weights applied across each episode from early to
    /// late ticks. `[1]` is uniform survival; `[1,2,4,8]` values later survival
    /// more while preserving a positive gradient for every alive tick.
    pub survival_window_weights: Vec<f64>,
    /// How the contemporary population is arranged into evaluator worlds.
    pub evaluation_topology: EvaluationTopology,
    /// Total contemporary opponents represented per candidate. Pairwise mode
    /// samples this many distinct opponents; shared-population mode requires
    /// exactly `population_size - 1`.
    pub eval_opponents: usize,
    /// Diagnostic oracle: in evaluator worlds, attacks only affect organisms
    /// from other founder-pool entries. This removes friendly fire to test
    /// whether identity ambiguity is the binding predation constraint.
    pub cross_pool_predation_only: bool,
    pub world_seeds: Vec<u64>,
    /// `0` freezes training layouts. Otherwise a fresh deterministic seed
    /// suite is derived every N generations.
    pub training_seed_rotation_period: u32,
    /// Select on the mean of the worst-performing fraction of scenario/seed
    /// cases. `1.0` is the ordinary mean; `0.25` is lower-quartile CVaR.
    pub objective_cvar_fraction: f64,
    pub evaluator_workers: usize,
    pub compatibility_threshold: f64,
    pub target_species: usize,
    pub compatibility_threshold_adjustment: f64,
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub weight_coefficient: f64,
    pub survival_fraction: f64,
    pub crossover_probability: f64,
    pub interspecies_mate_probability: f64,
    pub mutate_weight_probability: f64,
    pub per_connection_weight_mutation_probability: f64,
    pub replace_weight_probability: f64,
    pub weight_perturb_stddev: f32,
    pub mutate_bias_probability: f64,
    pub bias_perturb_stddev: f32,
    pub mutate_time_constant_probability: f64,
    pub time_constant_perturb_stddev: f32,
    pub add_connection_probability: f64,
    pub add_node_probability: f64,
    pub disabled_inheritance_probability: f64,
    pub young_species_grace_generations: u32,
    pub min_young_species_offspring: usize,
    pub elitism_min_species_size: usize,
}

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            generations: 20,
            population_checkpoint_interval: 10,
            episode_horizons: vec![500],
            survival_window_weights: vec![1.0],
            evaluation_topology: EvaluationTopology::SharedPopulation,
            eval_opponents: 63,
            cross_pool_predation_only: false,
            world_seeds: (11..75).collect(),
            training_seed_rotation_period: 0,
            objective_cvar_fraction: 1.0,
            evaluator_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            compatibility_threshold: 1.0,
            target_species: 4,
            compatibility_threshold_adjustment: 0.05,
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.2,
            survival_fraction: 0.2,
            crossover_probability: 0.75,
            interspecies_mate_probability: 0.001,
            mutate_weight_probability: 0.8,
            per_connection_weight_mutation_probability: 0.8,
            replace_weight_probability: 0.1,
            weight_perturb_stddev: 0.15,
            mutate_bias_probability: 0.2,
            bias_perturb_stddev: 0.1,
            mutate_time_constant_probability: 0.2,
            time_constant_perturb_stddev: 0.1,
            add_connection_probability: 0.05,
            add_node_probability: 0.03,
            disabled_inheritance_probability: 0.75,
            young_species_grace_generations: 5,
            min_young_species_offspring: 2,
            elitism_min_species_size: 5,
        }
    }
}

impl NeatConfig {
    pub fn validate(&self) -> Result<()> {
        if self.population_size < 2 {
            bail!("NEAT population_size must be >= 2");
        }
        if self.generations == 0
            || self.episode_horizons.is_empty()
            || self.episode_horizons.contains(&0)
        {
            bail!("NEAT generations and episode horizons must be nonzero and nonempty");
        }
        if self.population_checkpoint_interval == 0 {
            bail!("NEAT population_checkpoint_interval must be nonzero");
        }
        if self
            .episode_horizons
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            bail!("NEAT episode horizons must be strictly increasing and unique");
        }
        if self.survival_window_weights.is_empty()
            || self
                .survival_window_weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
        {
            bail!("NEAT survival window weights must be finite, positive, and nonempty");
        }
        match self.evaluation_topology {
            EvaluationTopology::Pairwise => {
                if self.eval_opponents == 0 || self.eval_opponents >= self.population_size {
                    bail!("pairwise eval_opponents must be in 1..population_size");
                }
                if !self.population_size.is_multiple_of(2) {
                    bail!("population_size must be even for balanced pairwise evaluation");
                }
                if !self
                    .population_size
                    .saturating_mul(self.eval_opponents)
                    .is_multiple_of(2)
                {
                    bail!("pairwise schedule cannot balance the requested opponent count");
                }
                if !self.world_seeds.len().is_multiple_of(4) {
                    bail!(
                        "balanced pairwise evaluation needs a multiple of four training world seeds"
                    );
                }
            }
            EvaluationTopology::SharedPopulation => {
                let expected = self.population_size.saturating_sub(1);
                if self.eval_opponents != expected {
                    bail!(
                        "shared-population evaluation requires eval_opponents = population_size - 1 ({expected})"
                    );
                }
            }
        }
        if self.world_seeds.is_empty() || self.evaluator_workers == 0 {
            bail!("NEAT needs at least one world seed and evaluator worker");
        }
        if !(0.0..=1.0).contains(&self.objective_cvar_fraction)
            || self.objective_cvar_fraction == 0.0
        {
            bail!("objective_cvar_fraction must be in (0,1]");
        }
        let mut seen = std::collections::BTreeSet::new();
        for seed in &self.world_seeds {
            if !seen.insert(*seed) {
                bail!("duplicate training world seed {seed}");
            }
        }
        if self.compatibility_threshold <= 0.0 || self.survival_fraction <= 0.0 {
            bail!("compatibility_threshold and survival_fraction must be positive");
        }
        if self.target_species == 0 || self.min_young_species_offspring == 0 {
            bail!("target_species and min_young_species_offspring must be >= 1");
        }
        for (name, value) in [
            ("survival_fraction", self.survival_fraction),
            ("crossover_probability", self.crossover_probability),
            (
                "interspecies_mate_probability",
                self.interspecies_mate_probability,
            ),
            ("mutate_weight_probability", self.mutate_weight_probability),
            (
                "per_connection_weight_mutation_probability",
                self.per_connection_weight_mutation_probability,
            ),
            (
                "replace_weight_probability",
                self.replace_weight_probability,
            ),
            ("mutate_bias_probability", self.mutate_bias_probability),
            (
                "mutate_time_constant_probability",
                self.mutate_time_constant_probability,
            ),
            (
                "add_connection_probability",
                self.add_connection_probability,
            ),
            ("add_node_probability", self.add_node_probability),
            (
                "disabled_inheritance_probability",
                self.disabled_inheritance_probability,
            ),
        ] {
            if !(0.0..=1.0).contains(&value) {
                bail!("{name} must be in [0,1], got {value}");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Evaluation {
    /// The only selection objective in this baseline.
    pub mean_objective_score: f64,
    /// Ordinary mean across cases, retained beside the robust lower-tail
    /// selection objective as a diagnostic.
    pub mean_case_score: f64,
    /// Population standard deviation and extrema across deterministic
    /// evaluator cases. These quantify seed/context sensitivity; they do not
    /// contribute to selection.
    pub case_score_stddev: f64,
    pub min_case_score: f64,
    pub max_case_score: f64,
    /// Ordinary mean of the absolute candidate survival component.
    pub mean_absolute_survival_fraction: f64,
    /// Ordinary mean candidate-founder alive-ticks across evaluation cases.
    /// Unlike the normalized fraction, this is the direct total-survival-time
    /// quantity accumulated by the bounded simulation evaluator.
    pub mean_candidate_alive_ticks: f64,
    pub mean_late_weighted_survival_fraction: f64,
    /// Ordinary mean of the bounded relative component. Neutral competition is
    /// 1.0 and candidate advantage approaches 2.0.
    pub mean_relative_survival_advantage: f64,
    /// Component means over exactly the cases selected into the objective CVaR.
    pub objective_cvar_absolute_survival_fraction: f64,
    pub objective_cvar_late_weighted_survival_fraction: f64,
    pub objective_cvar_relative_survival_advantage: f64,
    pub objective_cvar_case_count: usize,
    pub zero_combined_alive_tick_cases: usize,
    /// Number of scored opponent-and-seed cases and distinct opponent genome
    /// indices represented across them. In the pairwise evaluator this
    /// is 32 cases over eight unique contemporary opponents. Historical slots
    /// can repeat while the archive is smaller than their requested count.
    pub pair_seed_cases: usize,
    pub unique_opponents: usize,
    /// Mean fraction of candidate founders alive at the episode boundary.
    /// Distinct from the objective, which integrates survival over all ticks.
    pub mean_candidate_end_survival_fraction: f64,
    /// Shared `metrics` definitions, computed from this lineage's actions
    /// in the actual competitive evaluation worlds. These are observational
    /// diagnostics and never contribute to fitness.
    pub mean_action_effectiveness: Option<f64>,
    pub mean_successful_attack_rate: Option<f64>,
    pub mean_learning_slope: Option<f64>,
    /// Candidate-lineage energy-flow diagnostics only; these never contribute
    /// to fitness. Gross acquired energy counts attack-transfer credits.
    /// Starting energy is excluded and each transfer is counted exactly once.
    pub mean_gross_energy_acquired: f64,
    pub mean_attack_energy_received: f64,
    pub mean_attack_energy_lost: f64,
    pub mean_attack_attempt_energy_cost: f64,
    pub mean_net_attack_energy_balance: f64,
    pub mean_attack_no_organism_targets: f64,
    pub mean_attack_target_evaded: f64,
    pub mean_attack_same_pool_blocked: f64,
    pub mean_attack_insufficient_energy: f64,
    pub mean_attack_eligible_attempts: f64,
    pub mean_attack_hits: f64,
    pub mean_attack_nonlethal_hits: f64,
    pub mean_attack_kills: f64,
    pub mean_attack_same_pair_followups: f64,
    pub mean_distinct_attack_victims: f64,
    /// Ratio of successful focal hits after the first hit by the same attacker
    /// against the same victim to all successful focal hits, pooled across
    /// cases. `None` means no successful focal hits occurred.
    pub attack_repeat_hit_fraction: Option<f64>,
    pub mean_spatial_coverage: f64,
    /// Organism-tick action distribution in ActionType declaration order,
    /// including Idle. This is an observational behavior descriptor only.
    pub mean_action_fractions: [f64; 5],
    /// Mean number of explicit commands emitted per organism tick. This equals
    /// the sum of the five non-idle action fractions and can exceed one only
    /// under compositional control.
    pub mean_commands_per_tick: f64,
    pub mean_multi_command_tick_fraction: f64,
    pub mean_world_final_population: f64,
}

#[derive(Debug, Clone, Default)]
struct LineageCaseDiagnostics {
    attack_no_organism_targets: u64,
    attack_target_evaded: u64,
    attack_same_pool_blocked: u64,
    attack_insufficient_energy: u64,
    attack_eligible_attempts: u64,
    attack_hits: u64,
    attack_nonlethal_hits: u64,
    attack_kills: u64,
    attack_same_pair_followups: u64,
    attack_followup_latency_ticks_sum: u64,
    attack_energy_received: f64,
    attack_energy_lost: f64,
    attack_attempt_energy_cost: f64,
    attack_victim_energy_before_sum: f64,
    attack_victim_energy_after_sum: f64,
    distinct_attack_victims: u64,
    action_effectiveness: Option<f64>,
    successful_attack_rate: Option<f64>,
    learning_slope: Option<f64>,
    spatial_coverage: f64,
    action_fractions: [f64; 5],
    commands_per_tick: f64,
    multi_command_tick_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseEvaluation {
    pub scenario: String,
    pub world_seed: u64,
    pub episode_horizon: u64,
    pub objective_score: f64,
    pub absolute_survival_fraction: f64,
    pub late_weighted_survival_fraction: f64,
    pub relative_survival_advantage: f64,
    pub zero_combined_alive_ticks: bool,
    pub requested_opponents: usize,
    pub opponent_population_indices: Vec<usize>,
    pub founders_by_pool: Vec<u64>,
    pub alive_ticks_by_pool: Vec<u64>,
    pub weighted_alive_ticks_by_pool: Vec<f64>,
    pub end_survivors_by_pool: Vec<u64>,
    pub founder_pool_size: usize,
    pub focal_pool_index: usize,
    pub world_founders: usize,
    pub candidate_founders: u64,
    pub candidate_end_survivors: u64,
    pub action_effectiveness: Option<f64>,
    pub successful_attack_rate: Option<f64>,
    pub learning_slope: Option<f64>,
    /// Attack-transfer credits acquired by the focal lineage.
    /// Starting energy is excluded and each direct transfer is counted once.
    pub gross_energy_acquired: f64,
    pub attack_energy_received: f64,
    pub attack_energy_lost: f64,
    pub attack_attempt_energy_cost: f64,
    pub net_attack_energy_balance: f64,
    pub attack_no_organism_targets: u64,
    pub attack_target_evaded: u64,
    pub attack_same_pool_blocked: u64,
    pub attack_insufficient_energy: u64,
    pub attack_eligible_attempts: u64,
    pub attack_hits: u64,
    pub attack_nonlethal_hits: u64,
    pub attack_kills: u64,
    pub attack_same_pair_followups: u64,
    pub attack_followup_latency_ticks_sum: u64,
    pub distinct_attack_victims: u64,
    /// Successful hits after the first successful hit by the same attacker
    /// against the same victim, divided by all successful focal hits.
    pub attack_repeat_hit_fraction: Option<f64>,
    pub attack_victim_energy_before_sum: f64,
    pub attack_victim_energy_after_sum: f64,
    pub spatial_coverage: f64,
    pub action_fractions: [f64; 5],
    pub commands_per_tick: f64,
    pub multi_command_tick_fraction: f64,
    pub world_final_population: usize,
    /// Observation-only facts for every lineage in the shared world. They are
    /// retained in memory just long enough to construct accurate mirrored
    /// `CaseEvaluation`s and are not part of the result schema.
    #[serde(skip)]
    lineage_diagnostics_by_pool: Vec<LineageCaseDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelEvaluation {
    pub summary: Evaluation,
    pub cases: Vec<CaseEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairEvaluation {
    pub left: PanelEvaluation,
    pub right: PanelEvaluation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub name: String,
    pub world: WorldConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub id: u64,
    pub size: usize,
}

/// Stable identity of an opponent within one evaluated generation.
///
/// Contemporary indices refer to that generation's persisted population
/// checkpoint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OpponentIdentity {
    Contemporary { population_index: usize },
}

/// Scores are grouped by opponent identity before dispersion is computed, so
/// seed/scenario variation is not mistaken for opponent-context sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpponentMeanScore {
    pub opponent: OpponentIdentity,
    pub case_count: usize,
    pub mean_score: f64,
    pub min_case_score: f64,
    pub max_case_score: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpponentScoreProfile {
    /// One inspectable row per distinct opponent, in stable identity order.
    pub opponents: Vec<OpponentMeanScore>,
    /// Population standard deviation across the opponent means above.
    pub mean_score_stddev: Option<f64>,
    pub min_opponent_mean: Option<f64>,
    pub max_opponent_mean: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReproductionKind {
    Initial,
    EliteClone,
    AsexualMutation,
    Crossover,
}

/// Parent population coordinates identify lineage relationships. Complete
/// population records are persisted periodically rather than every generation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParentReference {
    pub generation: u32,
    pub population_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSummary {
    pub generation: u32,
    pub training_seed_epoch: u32,
    pub effective_training_seeds: Vec<u64>,
    pub eval_opponents: usize,
    pub evaluation_cases_per_genome: usize,
    /// Actual pairwise worlds simulated. Each world scores both contemporary
    /// lineages.
    pub evaluation_worlds: usize,
    /// Selection score is meaningful only relative to this contemporary
    /// population and evaluation context.
    pub winner_contextual_score: f64,
    pub winner_case_score_stddev: f64,
    pub mean_case_score_stddev: f64,
    pub max_case_score_stddev: f64,
    pub winner_absolute_survival_fraction: f64,
    pub winner_candidate_alive_ticks: f64,
    pub winner_late_weighted_survival_fraction: f64,
    pub winner_relative_survival_advantage: f64,
    pub mean_absolute_survival_fraction: f64,
    pub mean_candidate_alive_ticks: f64,
    pub mean_late_weighted_survival_fraction: f64,
    pub mean_relative_survival_advantage: f64,
    pub winner_action_effectiveness: Option<f64>,
    pub winner_successful_attack_rate: Option<f64>,
    pub winner_mean_attack_kills: f64,
    pub mean_action_effectiveness: Option<f64>,
    pub mean_successful_attack_rate: Option<f64>,
    /// Exact population distribution of attack-transfer credits (excluding
    /// starting energy), plus summaries.
    pub winner_gross_energy_acquired: f64,
    pub mean_gross_energy_acquired: f64,
    pub median_gross_energy_acquired: f64,
    pub gross_energy_acquired_distribution: Vec<f64>,
    pub winner_attack_energy_received: f64,
    pub mean_attack_energy_received: f64,
    pub winner_attack_energy_lost: f64,
    pub mean_attack_energy_lost: f64,
    pub winner_attack_attempt_energy_cost: f64,
    pub mean_attack_attempt_energy_cost: f64,
    pub winner_net_energy_profit: f64,
    pub mean_net_energy_profit: f64,
    pub median_net_energy_profit: f64,
    pub winner_attack_precision: Option<f64>,
    pub population_attack_precision: Option<f64>,
    pub winner_net_attack_energy_balance: f64,
    pub mean_net_attack_energy_balance: f64,
    pub winner_distinct_attack_victims: f64,
    pub mean_distinct_attack_victims: f64,
    pub winner_attack_target_evaded: f64,
    pub mean_attack_target_evaded: f64,
    pub winner_attack_repeat_hit_fraction: Option<f64>,
    pub mean_attack_repeat_hit_fraction: Option<f64>,
    pub winner_action_fractions: [f64; 5],
    pub mean_action_fractions: [f64; 5],
    pub winner_commands_per_tick: f64,
    pub mean_commands_per_tick: f64,
    pub winner_multi_command_tick_fraction: f64,
    pub mean_multi_command_tick_fraction: f64,
    pub winner_spatial_coverage: f64,
    pub mean_spatial_coverage: f64,
    pub winner_end_survival_fraction: f64,
    pub mean_end_survival_fraction: f64,
    /// Across-member summary of opponent-context sensitivity. Inspect each
    /// member's profile in `population_checkpoint` for opponent identities and
    /// exact opponent means.
    pub mean_opponent_score_stddev: Option<f64>,
    pub max_opponent_score_stddev: Option<f64>,
    /// Same-generation winner persisted solely for explicit frozen crossplay.
    pub crossplay_checkpoint_genome: Option<OrganismGenome>,
    /// Complete evaluated population at the configured checkpoint interval and
    /// final generation. Empty on compact-only generations.
    pub population_checkpoint: Vec<PopulationMemberResult>,
    pub compatibility_threshold: f64,
    pub winner_hidden_nodes: usize,
    pub winner_enabled_connections: usize,
    pub winner_encoded_connections: usize,
    /// Enabled structure on at least one directed sensory-to-action path.
    pub winner_expressed_hidden_nodes: usize,
    pub winner_expressed_connections: usize,
    pub mean_expressed_hidden_nodes: f64,
    pub mean_expressed_connections: f64,
    pub new_connection_innovations: usize,
    pub new_node_innovations: usize,
    pub new_expressed_connection_innovations: usize,
    pub expressed_connection_innovations: usize,
    pub connection_innovations_reaching_ten_percent: usize,
    pub connection_innovations_reaching_majority: usize,
    pub non_elite_offspring: usize,
    pub structural_mutation_attempts: usize,
    pub structural_mutation_successes: usize,
    pub registry_new_structural_mutations: usize,
    pub new_origin_offspring: usize,
    pub new_origin_offspring_rate: Option<f64>,
    pub species: Vec<SpeciesSummary>,
    pub offspring_crossovers: usize,
    pub offspring_clones: usize,
}

/// Reproducible snapshot of one member in an evaluated generation. Every
/// generation retains these records so analyses can reconstruct the full
/// ranking, behavior distributions, and parent graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationMemberResult {
    pub generation: u32,
    pub population_index: usize,
    pub reproduction: ReproductionKind,
    pub parents: Vec<ParentReference>,
    pub contextual_score: f64,
    pub evaluation: Evaluation,
    pub opponent_scores: OpponentScoreProfile,
    pub genome: OrganismGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenOuterLoopContract {
    pub fully_connected_initial_topology: bool,
    pub current_tick_hidden_graph_acyclic: bool,
    pub previous_tick_hidden_recurrence_enabled: bool,
    pub evaluation_topology: EvaluationTopology,
    pub balanced_pairwise_evaluation: bool,
    pub symmetric_founder_slot_rotation: bool,
    pub runtime_plasticity_enabled: bool,
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
    pub force_random_actions: bool,
    pub compositional_actions_enabled: bool,
    pub cross_pool_predation_only: bool,
    pub intent_parallel_threads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub result_schema_version: u32,
    pub algorithm: String,
    pub objective: String,
    pub seed: u64,
    pub neat_config: NeatConfig,
    pub frozen_outer_loop_contract: FrozenOuterLoopContract,
    pub world_width: u32,
    pub founder_cohort_size: u32,
    pub evaluation_scenarios: Vec<ScenarioManifest>,
    pub generations: Vec<GenerationSummary>,
    pub final_population: Vec<PopulationMemberResult>,
    pub connection_innovation_history: Vec<ConnectionInnovationRecord>,
    pub node_innovation_history: Vec<NodeInnovationRecord>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InnovationKind {
    Initial,
    AddConnection,
    SplitIncoming,
    SplitOutgoing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInnovationRecord {
    pub innovation: InnovationId,
    pub pre_node_id: GeneNodeId,
    pub post_node_id: GeneNodeId,
    pub timing: SynapseTiming,
    /// `None` identifies the shared minimal starting topology.
    pub origin_generation: Option<u32>,
    pub kind: InnovationKind,
    pub first_expressed_generation: Option<u32>,
    pub first_ten_percent_generation: Option<u32>,
    pub first_majority_generation: Option<u32>,
    pub last_present_generation: Option<u32>,
    pub max_encoded_frequency: f64,
    pub max_expressed_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInnovationRecord {
    pub node_id: GeneNodeId,
    pub split_from_innovation: InnovationId,
    pub origin_generation: u32,
    pub first_expressed_generation: Option<u32>,
    pub first_ten_percent_generation: Option<u32>,
    pub first_majority_generation: Option<u32>,
    pub last_present_generation: Option<u32>,
    pub max_encoded_frequency: f64,
    pub max_expressed_frequency: f64,
}

#[derive(Clone)]
struct Individual {
    genome: OrganismGenome,
    evaluation: Evaluation,
    opponent_scores: OpponentScoreProfile,
    reproduction: ReproductionKind,
    parents: Vec<ParentReference>,
    fitness: f64,
    selection_score: f64,
}

#[derive(Clone)]
struct SpeciesRecord {
    id: u64,
    representative: OrganismGenome,
    members: Vec<usize>,
    created_generation: u32,
}

struct ActiveStructure {
    hidden_nodes: BTreeSet<GeneNodeId>,
    connections: BTreeSet<InnovationId>,
}

struct ComplexificationSnapshot {
    winner_expressed_hidden_nodes: usize,
    winner_expressed_connections: usize,
    mean_expressed_hidden_nodes: f64,
    mean_expressed_connections: f64,
    new_connection_innovations: usize,
    new_node_innovations: usize,
    new_expressed_connection_innovations: usize,
    expressed_connection_innovations: usize,
    connection_innovations_reaching_ten_percent: usize,
    connection_innovations_reaching_majority: usize,
}

#[derive(Clone, Copy, Default)]
struct BreedingTelemetry {
    non_elite_offspring: usize,
    structural_mutation_attempts: usize,
    structural_mutation_successes: usize,
    registry_new_structural_mutations: usize,
    new_origin_offspring: usize,
}

#[derive(Clone, Copy, Default)]
struct MutationOutcome {
    structural_attempts: usize,
    structural_successes: usize,
    registry_new_structures: usize,
}

#[derive(Clone, Copy)]
struct SplitRecord {
    node: GeneNodeId,
    incoming: InnovationId,
    outgoing: InnovationId,
    incoming_timing: SynapseTiming,
    outgoing_timing: SynapseTiming,
}

/// Run-owned historical markings. A structural event receives one monotonic
/// innovation number and every lineage encountering that event reuses it.
#[derive(Default)]
struct InnovationRegistry {
    next: u64,
    connections: HashMap<(GeneNodeId, GeneNodeId, SynapseTiming), InnovationId>,
    splits: HashMap<InnovationId, SplitRecord>,
    connection_history: BTreeMap<InnovationId, ConnectionInnovationRecord>,
    node_history: BTreeMap<GeneNodeId, NodeInnovationRecord>,
}

impl InnovationRegistry {
    fn connection(
        &mut self,
        pre: GeneNodeId,
        post: GeneNodeId,
        timing: SynapseTiming,
        origin_generation: Option<u32>,
        kind: InnovationKind,
    ) -> InnovationId {
        if let Some(id) = self.connections.get(&(pre, post, timing)) {
            return *id;
        }
        let id = InnovationId(self.next);
        self.next = self.next.saturating_add(1);
        self.connections.insert((pre, post, timing), id);
        self.connection_history.insert(
            id,
            ConnectionInnovationRecord {
                innovation: id,
                pre_node_id: pre,
                post_node_id: post,
                timing,
                origin_generation,
                kind,
                first_expressed_generation: None,
                first_ten_percent_generation: None,
                first_majority_generation: None,
                last_present_generation: None,
                max_encoded_frequency: 0.0,
                max_expressed_frequency: 0.0,
            },
        );
        id
    }

    fn split(
        &mut self,
        original: InnovationId,
        pre: GeneNodeId,
        post: GeneNodeId,
        timing: SynapseTiming,
        origin_generation: u32,
    ) -> SplitRecord {
        if let Some(record) = self.splits.get(&original) {
            return *record;
        }
        let node = split_hidden_gene_node_id(original);
        let incoming_timing = timing;
        let outgoing_timing = SynapseTiming::CurrentTick;
        let record = SplitRecord {
            node,
            incoming: self.connection(
                pre,
                node,
                incoming_timing,
                Some(origin_generation),
                InnovationKind::SplitIncoming,
            ),
            outgoing: self.connection(
                node,
                post,
                outgoing_timing,
                Some(origin_generation),
                InnovationKind::SplitOutgoing,
            ),
            incoming_timing,
            outgoing_timing,
        };
        self.splits.insert(original, record);
        self.node_history.insert(
            node,
            NodeInnovationRecord {
                node_id: node,
                split_from_innovation: original,
                origin_generation,
                first_expressed_generation: None,
                first_ten_percent_generation: None,
                first_majority_generation: None,
                last_present_generation: None,
                max_encoded_frequency: 0.0,
                max_expressed_frequency: 0.0,
            },
        );
        record
    }
}

/// Execute a complete deterministic NEAT run. The callback receives each
/// finished generation and is observational only.
pub fn run_neat(
    mut config: NeatConfig,
    mut world: WorldConfig,
    seed: u64,
    mut on_generation: impl FnMut(&GenerationSummary),
) -> Result<RunResult> {
    config.validate()?;
    world.predation_enabled = true;
    if world.force_random_actions {
        bail!(
            "NEAT requires genome-controlled actions; force_random_actions is reserved for explicit null-control simulations"
        );
    }
    configure_evaluation_world(&mut world);
    if config.evaluation_topology == EvaluationTopology::SharedPopulation
        && world.num_organisms as usize != config.population_size
    {
        bail!(
            "shared-population evaluation requires exactly one founder per genome: founders={}, population={}",
            world.num_organisms,
            config.population_size
        );
    }
    let required_founder_divisor = match config.evaluation_topology {
        EvaluationTopology::Pairwise => 2,
        EvaluationTopology::SharedPopulation => config.population_size,
    };
    if !(world.num_organisms as usize).is_multiple_of(required_founder_divisor) {
        bail!(
            "founder count {} must be divisible by {} for {:?} evaluation",
            world.num_organisms,
            required_founder_divisor,
            config.evaluation_topology
        );
    }
    if !world.leaky_neurons_enabled {
        config.mutate_time_constant_probability = 0.0;
    }
    let evaluation_scenarios = vec![ScenarioManifest {
        name: "energy_stealing".to_owned(),
        world: world.clone(),
    }];
    let template_sim = Simulation::new(world.clone(), seed).map_err(|e| anyhow!("{e}"))?;
    let mut template = template_sim
        .organisms()
        .first()
        .ok_or_else(|| anyhow!("NEAT evaluation world produced no founders"))?
        .genome
        .clone();
    template.brain.hidden_nodes.clear();
    template.brain.edges.clear();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut innovations = InnovationRegistry::default();
    for receptor in SensoryReceptor::active(world.predation_enabled) {
        let pre = sensory_gene_node_id(
            receptor
                .current_index()
                .expect("active sensory receptor has a canonical index") as u32,
        );
        for action in ActionType::active(world.predation_enabled) {
            let action_index = ActionType::ALL
                .iter()
                .position(|candidate| *candidate == action)
                .expect("active action has a canonical index");
            let post = action_gene_node_id(action_index);
            template.brain.edges.push(SynapseGene {
                innovation: innovations.connection(
                    pre,
                    post,
                    SynapseTiming::CurrentTick,
                    None,
                    InnovationKind::Initial,
                ),
                pre_node_id: pre,
                post_node_id: post,
                timing: SynapseTiming::CurrentTick,
                // Every population member receives an independently sampled
                // weight below; this placeholder keeps construction free of
                // population-order effects.
                weight: WEIGHT_MIN_ABS,
                enabled: true,
            });
        }
    }
    canonicalize_initial_markings(&mut template, &mut innovations);

    let mut population = Vec::with_capacity(config.population_size);
    for _ in 0..config.population_size {
        let mut genome = template.clone();
        randomize_parameters(&mut genome, &mut rng);
        brain::genome::restrict_action_genes(&mut genome, world.predation_enabled);
        population.push(Individual {
            genome,
            evaluation: Evaluation::default(),
            opponent_scores: OpponentScoreProfile::default(),
            reproduction: ReproductionKind::Initial,
            parents: Vec::new(),
            fitness: 0.0,
            selection_score: 0.0,
        });
    }

    let frozen_outer_loop_contract = FrozenOuterLoopContract {
        fully_connected_initial_topology: true,
        current_tick_hidden_graph_acyclic: true,
        previous_tick_hidden_recurrence_enabled: true,
        evaluation_topology: config.evaluation_topology,
        balanced_pairwise_evaluation: config.evaluation_topology == EvaluationTopology::Pairwise,
        symmetric_founder_slot_rotation: true,
        runtime_plasticity_enabled: world.runtime_plasticity_enabled,
        leaky_neurons_enabled: world.leaky_neurons_enabled,
        predation_enabled: world.predation_enabled,
        force_random_actions: world.force_random_actions,
        compositional_actions_enabled: world.compositional_actions_enabled,
        cross_pool_predation_only: config.cross_pool_predation_only,
        intent_parallel_threads: world.intent_parallel_threads,
    };
    let world_width = world.world_width;
    let founder_cohort_size = world.num_organisms;

    let mut previous_species: Vec<SpeciesRecord> = Vec::new();
    let mut next_species_id = 0u64;
    let mut compatibility_threshold = config.compatibility_threshold;
    let mut summaries = Vec::with_capacity(config.generations as usize);
    let mut incoming_breeding_telemetry = BreedingTelemetry::default();

    for generation in 0..config.generations {
        let seed_epoch = training_seed_epoch(&config, generation);
        let effective_training_seeds =
            effective_training_seeds(&config.world_seeds, seed, seed_epoch);
        evaluate_population(
            &mut population,
            &evaluation_scenarios,
            &effective_training_seeds,
            &config,
            seed,
            generation,
        )?;
        assign_selection_scores(&mut population);
        let selection_best_index = best_selection_index(&population);
        let mut species = assign_species(
            &population,
            &previous_species,
            &config,
            compatibility_threshold,
            generation,
            &mut next_species_id,
        );
        let winner_index = selection_best_index;
        let crossplay_checkpoint_genome = Some(population[winner_index].genome.clone());
        let cases_per_match = evaluation_scenarios
            .len()
            .saturating_mul(effective_training_seeds.len())
            .saturating_mul(config.episode_horizons.len());
        let (evaluation_cases_per_genome, evaluation_worlds) = match config.evaluation_topology {
            EvaluationTopology::Pairwise => (
                config.eval_opponents.saturating_mul(cases_per_match),
                population
                    .len()
                    .saturating_mul(config.eval_opponents)
                    .saturating_div(2)
                    .saturating_mul(cases_per_match),
            ),
            EvaluationTopology::SharedPopulation => (cases_per_match, cases_per_match),
        };
        let complexification =
            observe_complexification(generation, &population, winner_index, &mut innovations);

        let (next_population, offspring_crossovers, offspring_clones, next_breeding_telemetry) =
            if generation + 1 < config.generations {
                breed_next_generation(
                    &population,
                    &mut species,
                    &config,
                    &mut innovations,
                    seed,
                    generation,
                    selection_best_index,
                    generation.saturating_add(1),
                    world.predation_enabled,
                )?
            } else {
                (Vec::new(), 0, 0, BreedingTelemetry::default())
            };

        let summary = generation_summary(
            generation,
            &population,
            &species,
            winner_index,
            offspring_crossovers,
            offspring_clones,
            complexification,
            compatibility_threshold,
            seed_epoch,
            effective_training_seeds,
            config.eval_opponents,
            evaluation_cases_per_genome,
            evaluation_worlds,
            incoming_breeding_telemetry,
            crossplay_checkpoint_genome,
            generation.is_multiple_of(config.population_checkpoint_interval)
                || generation + 1 == config.generations,
        );
        on_generation(&summary);
        summaries.push(summary);
        if species.len() > config.target_species {
            compatibility_threshold += config.compatibility_threshold_adjustment;
        } else if species.len() < config.target_species {
            compatibility_threshold =
                (compatibility_threshold - config.compatibility_threshold_adjustment).max(0.1);
        }
        previous_species = species;
        if generation + 1 < config.generations {
            population = next_population;
            incoming_breeding_telemetry = next_breeding_telemetry;
        }
    }

    let final_generation = config.generations.saturating_sub(1);
    let final_population = population
        .into_iter()
        .enumerate()
        .map(|(population_index, individual)| PopulationMemberResult {
            generation: final_generation,
            population_index,
            reproduction: individual.reproduction,
            parents: individual.parents,
            contextual_score: individual.fitness,
            evaluation: individual.evaluation,
            opponent_scores: individual.opponent_scores,
            genome: individual.genome,
        })
        .collect();
    Ok(RunResult {
        result_schema_version: 31,
        algorithm: match config.evaluation_topology {
            EvaluationTopology::Pairwise => "competitive_pairwise_NEAT",
            EvaluationTopology::SharedPopulation => "competitive_shared_population_NEAT",
        }
        .to_string(),
        objective: OBJECTIVE_NAME.to_string(),
        seed,
        neat_config: config,
        frozen_outer_loop_contract,
        world_width,
        founder_cohort_size,
        evaluation_scenarios,
        generations: summaries,
        final_population,
        connection_innovation_history: innovations.connection_history.values().cloned().collect(),
        node_innovation_history: innovations.node_history.values().cloned().collect(),
    })
}

fn observe_complexification(
    generation: u32,
    population: &[Individual],
    best_index: usize,
    innovations: &mut InnovationRegistry,
) -> ComplexificationSnapshot {
    let active: Vec<_> = population
        .iter()
        .map(|individual| active_structure(&individual.genome))
        .collect();
    let mean_expressed_hidden_nodes = active
        .iter()
        .map(|structure| structure.hidden_nodes.len() as f64)
        .sum::<f64>()
        / active.len() as f64;
    let mean_expressed_connections = active
        .iter()
        .map(|structure| structure.connections.len() as f64)
        .sum::<f64>()
        / active.len() as f64;

    let mut newly_ten_percent = 0usize;
    let mut newly_majority = 0usize;
    let currently_expressed_connections = active
        .iter()
        .flat_map(|structure| structure.connections.iter().copied())
        .collect::<BTreeSet<_>>();
    let pop_len = population.len() as f64;
    for record in innovations.connection_history.values_mut() {
        let mut encoded_count = 0usize;
        let mut expressed_count = 0usize;
        for (index, individual) in population.iter().enumerate() {
            let encoded = individual
                .genome
                .brain
                .edges
                .binary_search_by_key(&record.innovation, |edge| edge.innovation)
                .is_ok();
            if encoded {
                encoded_count += 1;
            }
            if active[index].connections.contains(&record.innovation) {
                expressed_count += 1;
            }
        }
        let encoded_frequency = encoded_count as f64 / pop_len;
        let expressed_frequency = expressed_count as f64 / pop_len;
        record.max_encoded_frequency = record.max_encoded_frequency.max(encoded_frequency);
        record.max_expressed_frequency = record.max_expressed_frequency.max(expressed_frequency);
        if encoded_count > 0 {
            record.last_present_generation = Some(generation);
        }
        if expressed_count > 0 && record.first_expressed_generation.is_none() {
            record.first_expressed_generation = Some(generation);
        }
        if expressed_frequency >= 0.10 && record.first_ten_percent_generation.is_none() {
            record.first_ten_percent_generation = Some(generation);
            if record.origin_generation.is_some() {
                newly_ten_percent += 1;
            }
        }
        if expressed_frequency >= 0.50 && record.first_majority_generation.is_none() {
            record.first_majority_generation = Some(generation);
            if record.origin_generation.is_some() {
                newly_majority += 1;
            }
        }
    }

    for record in innovations.node_history.values_mut() {
        let encoded_count = population
            .iter()
            .filter(|individual| {
                individual
                    .genome
                    .brain
                    .hidden_nodes
                    .binary_search_by_key(&record.node_id, |node| node.id)
                    .is_ok()
            })
            .count();
        let expressed_count = active
            .iter()
            .filter(|structure| structure.hidden_nodes.contains(&record.node_id))
            .count();
        let encoded_frequency = encoded_count as f64 / pop_len;
        let expressed_frequency = expressed_count as f64 / pop_len;
        record.max_encoded_frequency = record.max_encoded_frequency.max(encoded_frequency);
        record.max_expressed_frequency = record.max_expressed_frequency.max(expressed_frequency);
        if encoded_count > 0 {
            record.last_present_generation = Some(generation);
        }
        if expressed_count > 0 && record.first_expressed_generation.is_none() {
            record.first_expressed_generation = Some(generation);
        }
        if expressed_frequency >= 0.10 && record.first_ten_percent_generation.is_none() {
            record.first_ten_percent_generation = Some(generation);
        }
        if expressed_frequency >= 0.50 && record.first_majority_generation.is_none() {
            record.first_majority_generation = Some(generation);
        }
    }

    ComplexificationSnapshot {
        winner_expressed_hidden_nodes: active[best_index].hidden_nodes.len(),
        winner_expressed_connections: active[best_index].connections.len(),
        mean_expressed_hidden_nodes,
        mean_expressed_connections,
        new_connection_innovations: innovations
            .connection_history
            .values()
            .filter(|record| record.origin_generation == Some(generation))
            .count(),
        new_node_innovations: innovations
            .node_history
            .values()
            .filter(|record| record.origin_generation == generation)
            .count(),
        new_expressed_connection_innovations: innovations
            .connection_history
            .values()
            .filter(|record| record.origin_generation == Some(generation))
            .filter(|record| currently_expressed_connections.contains(&record.innovation))
            .count(),
        expressed_connection_innovations: currently_expressed_connections.len(),
        connection_innovations_reaching_ten_percent: newly_ten_percent,
        connection_innovations_reaching_majority: newly_majority,
    }
}

fn active_structure(genome: &OrganismGenome) -> ActiveStructure {
    let mut reachable = (0..SensoryReceptor::TOTAL_NEURON_COUNT)
        .map(sensory_gene_node_id)
        .collect::<BTreeSet<_>>();
    loop {
        let before = reachable.len();
        for edge in genome.brain.edges.iter().filter(|edge| edge.enabled) {
            if reachable.contains(&edge.pre_node_id) {
                reachable.insert(edge.post_node_id);
            }
        }
        if reachable.len() == before {
            break;
        }
    }

    let mut reaches_action = (0..ActionType::ALL.len())
        .map(action_gene_node_id)
        .collect::<BTreeSet<_>>();
    loop {
        let before = reaches_action.len();
        for edge in genome.brain.edges.iter().filter(|edge| edge.enabled) {
            if reaches_action.contains(&edge.post_node_id) {
                reaches_action.insert(edge.pre_node_id);
            }
        }
        if reaches_action.len() == before {
            break;
        }
    }

    let connections = genome
        .brain
        .edges
        .iter()
        .filter(|edge| {
            edge.enabled
                && reachable.contains(&edge.pre_node_id)
                && reaches_action.contains(&edge.post_node_id)
        })
        .map(|edge| edge.innovation)
        .collect();
    let hidden_nodes = genome
        .brain
        .hidden_nodes
        .iter()
        .filter(|node| reachable.contains(&node.id) && reaches_action.contains(&node.id))
        .map(|node| node.id)
        .collect();
    ActiveStructure {
        hidden_nodes,
        connections,
    }
}

fn configure_evaluation_world(world: &mut WorldConfig) {
    // Within-life plasticity is NOT forced off: the eval is the within-lifetime
    // fitness function, so learning is governed by `runtime_plasticity_enabled`
    // like everywhere else. With it on, evaluation is Baldwinian (fixed genome +
    // within-life learning); with it off, behavior is the genome alone.
    // Parallelize across independent candidate evaluations, not within a tiny
    // colony. This avoids nested pools and is substantially cheaper.
    world.intent_parallel_threads = 1;
}

fn training_seed_epoch(config: &NeatConfig, generation: u32) -> u32 {
    if config.training_seed_rotation_period == 0 {
        0
    } else {
        generation / config.training_seed_rotation_period
    }
}

fn effective_training_seeds(base: &[u64], run_seed: u64, epoch: u32) -> Vec<u64> {
    if epoch == 0 {
        return base.to_vec();
    }
    let rotated = base
        .iter()
        .enumerate()
        .map(|(index, seed)| {
            mix64(
                *seed
                    ^ run_seed.rotate_left(17)
                    ^ u64::from(epoch).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
            )
        })
        .collect::<Vec<_>>();
    let mut replay = Vec::with_capacity(base.len() + rotated.len());
    replay.extend_from_slice(base);
    replay.extend(rotated);
    replay
}

fn canonicalize_initial_markings(genome: &mut OrganismGenome, registry: &mut InnovationRegistry) {
    genome
        .brain
        .edges
        .sort_unstable_by_key(|edge| (edge.pre_node_id, edge.post_node_id, edge.timing));
    for edge in &mut genome.brain.edges {
        edge.innovation = registry.connection(
            edge.pre_node_id,
            edge.post_node_id,
            edge.timing,
            None,
            InnovationKind::Initial,
        );
    }
    genome
        .brain
        .edges
        .sort_unstable_by_key(|edge| edge.innovation);
}

fn evaluate_population(
    population: &mut [Individual],
    scenarios: &[ScenarioManifest],
    training_seeds: &[u64],
    config: &NeatConfig,
    run_seed: u64,
    generation: u32,
) -> Result<()> {
    match config.evaluation_topology {
        EvaluationTopology::SharedPopulation => {
            return evaluate_population_shared(
                population,
                scenarios,
                training_seeds,
                config,
                run_seed,
                generation,
            );
        }
        EvaluationTopology::Pairwise => {}
    }
    let snapshot = population
        .iter()
        .map(|individual| individual.genome.clone())
        .collect::<Vec<_>>();
    let pairings = balanced_pairings(
        population.len(),
        config.eval_opponents,
        run_seed,
        generation,
    );
    let next = AtomicUsize::new(0);
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(pairings.len())
            .collect::<Vec<_>>(),
    );
    if !pairings.is_empty() {
        let workers = config.evaluator_workers.min(pairings.len()).max(1);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let pair_index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(left_index, right_index)) = pairings.get(pair_index) else {
                        break;
                    };
                    let result = evaluate_pairwise_genomes(
                        left_index,
                        &snapshot[left_index],
                        right_index,
                        &snapshot[right_index],
                        pair_index,
                        scenarios,
                        &config.episode_horizons,
                        &config.survival_window_weights,
                        training_seeds,
                        config.objective_cvar_fraction,
                        config.cross_pool_predation_only,
                    );
                    results.lock().expect("evaluation result lock poisoned")[pair_index] =
                        Some(result);
                });
            }
        });
    }

    let mut all_cases = vec![Vec::<CaseEvaluation>::new(); population.len()];
    for result in results
        .into_inner()
        .expect("evaluation result lock poisoned")
    {
        let result = result.ok_or_else(|| anyhow!("missing pairwise NEAT evaluation result"))??;
        all_cases[result.left_index].extend(result.left_cases);
        all_cases[result.right_index].extend(result.right_cases);
    }
    for index in 0..population.len() {
        if all_cases[index].is_empty() {
            bail!("balanced pairwise evaluator produced incomplete cases for genome {index}");
        }
        let evaluation =
            summarize_evaluation_cases(&all_cases[index], config.objective_cvar_fraction);
        population[index].opponent_scores = summarize_opponent_scores(&all_cases[index]);
        population[index].fitness = evaluation.mean_objective_score;
        population[index].evaluation = evaluation;
    }
    Ok(())
}

fn evaluate_population_shared(
    population: &mut [Individual],
    scenarios: &[ScenarioManifest],
    training_seeds: &[u64],
    config: &NeatConfig,
    run_seed: u64,
    generation: u32,
) -> Result<()> {
    let snapshot = population
        .iter()
        .map(|individual| individual.genome.clone())
        .collect::<Vec<_>>();
    if snapshot.is_empty() {
        bail!("shared-population evaluation needs at least one genome");
    }
    let canonical_order = canonical_population_order(&snapshot)?;
    let case_specs = config
        .episode_horizons
        .iter()
        .flat_map(|&episode_ticks| {
            scenarios
                .iter()
                .enumerate()
                .flat_map(move |(scenario_index, _)| {
                    training_seeds
                        .iter()
                        .map(move |&world_seed| (episode_ticks, scenario_index, world_seed))
                })
        })
        .collect::<Vec<_>>();
    let next = AtomicUsize::new(0);
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(case_specs.len())
            .collect::<Vec<_>>(),
    );
    let workers = config.evaluator_workers.min(case_specs.len()).max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let case_index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(episode_ticks, scenario_index, world_seed)) = case_specs.get(case_index)
                else {
                    break;
                };
                let pool_to_population =
                    rotated_arena_order(&canonical_order, run_seed, generation, case_index);
                let pool_genomes = pool_to_population
                    .iter()
                    .map(|&population_index| snapshot[population_index].clone())
                    .collect::<Vec<_>>();
                let (focal, opponents) = pool_genomes
                    .split_first()
                    .expect("a non-empty population produces a non-empty arena pool");
                let result = evaluate_genome_on_seeds_detailed(
                    focal,
                    std::slice::from_ref(&scenarios[scenario_index]),
                    episode_ticks,
                    &config.survival_window_weights,
                    std::slice::from_ref(&world_seed),
                    config.objective_cvar_fraction,
                    config.cross_pool_predation_only,
                    0,
                    opponents,
                )
                .map(|bundle| SharedArenaCase {
                    bundle,
                    pool_to_population,
                });
                results.lock().expect("evaluation result lock poisoned")[case_index] = Some(result);
            });
        }
    });
    let mut cases_by_population = vec![Vec::<CaseEvaluation>::new(); snapshot.len()];
    for result in results
        .into_inner()
        .expect("evaluation result lock poisoned")
    {
        let arena_case =
            result.ok_or_else(|| anyhow!("missing shared-population NEAT evaluation result"))??;
        for source in arena_case.bundle.cases {
            for (pool_index, &population_index) in arena_case.pool_to_population.iter().enumerate()
            {
                let opponent_population_indices = (0..snapshot.len())
                    .filter(|&index| index != population_index)
                    .collect();
                cases_by_population[population_index].push(case_for_focal_pool(
                    &source,
                    pool_index,
                    opponent_population_indices,
                    &config.survival_window_weights,
                ));
            }
        }
    }
    for (population_index, individual) in population.iter_mut().enumerate() {
        let cases = &cases_by_population[population_index];
        if cases.is_empty() {
            bail!("shared-population evaluator produced no cases for genome {population_index}");
        }
        let evaluation = summarize_evaluation_cases(cases, config.objective_cvar_fraction);
        individual.fitness = evaluation.mean_objective_score;
        individual.evaluation = evaluation;
        // Every shared case contains all opponents simultaneously. A case score
        // cannot be attributed to any one opponent, so a per-opponent profile
        // would manufacture false precision.
        individual.opponent_scores = OpponentScoreProfile::default();
    }
    Ok(())
}

struct SharedArenaCase {
    bundle: EvaluationBundle,
    pool_to_population: Vec<usize>,
}

/// Canonicalize the set of contemporary genomes independently of the breeding
/// vector's order. Identical genomes are interchangeable; the original index
/// is used only as a deterministic tie-break so attribution remains total.
fn canonical_population_order(snapshot: &[OrganismGenome]) -> Result<Vec<usize>> {
    let mut keyed = snapshot
        .iter()
        .enumerate()
        .map(|(population_index, genome)| {
            bincode::serialize(genome)
                .map(|bytes| (bytes, population_index))
                .map_err(|error| anyhow!("serializing genome for arena ordering: {error}"))
        })
        .collect::<Result<Vec<_>>>()?;
    keyed.sort_unstable_by(|(left_bytes, left_index), (right_bytes, right_index)| {
        left_bytes
            .cmp(right_bytes)
            .then_with(|| left_index.cmp(right_index))
    });
    Ok(keyed.into_iter().map(|(_, index)| index).collect())
}

/// Use a distinct full-pool cyclic rotation for each case. When the case count
/// equals the population size, every genome occupies every founder-ID slot
/// exactly once. The stride is coprime to the pool length, so rotations do not
/// repeat before the complete cycle even for other contracts.
fn rotated_arena_order(
    canonical_order: &[usize],
    run_seed: u64,
    generation: u32,
    case_index: usize,
) -> Vec<usize> {
    let pool_len = canonical_order.len();
    debug_assert!(pool_len > 0);
    let mut rng = event_rng(run_seed, generation, pool_len, ARENA_SLOT_DOMAIN);
    let base = rng.random_range(0..pool_len);
    let stride = coprime_arena_stride(pool_len);
    let rotation = (base + case_index.wrapping_mul(stride)) % pool_len;
    (0..pool_len)
        .map(|slot| canonical_order[(slot + rotation) % pool_len])
        .collect()
}

fn coprime_arena_stride(pool_len: usize) -> usize {
    if pool_len <= 2 {
        return 1;
    }
    let mut stride = pool_len / 2 + 1;
    while greatest_common_divisor(stride, pool_len) != 1 {
        stride += 1;
    }
    stride % pool_len
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn summarize_opponent_scores(cases: &[CaseEvaluation]) -> OpponentScoreProfile {
    let mut grouped = BTreeMap::<OpponentIdentity, Vec<f64>>::new();
    for case in cases {
        for &population_index in &case.opponent_population_indices {
            grouped
                .entry(OpponentIdentity::Contemporary { population_index })
                .or_default()
                .push(case.objective_score);
        }
    }
    let opponents = grouped
        .into_iter()
        .map(|(opponent, scores)| {
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            OpponentMeanScore {
                opponent,
                case_count: scores.len(),
                mean_score,
                min_case_score: scores.iter().copied().fold(f64::INFINITY, f64::min),
                max_case_score: scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect::<Vec<_>>();
    if opponents.is_empty() {
        return OpponentScoreProfile::default();
    }
    let mean = opponents.iter().map(|row| row.mean_score).sum::<f64>() / opponents.len() as f64;
    let variance = opponents
        .iter()
        .map(|row| (row.mean_score - mean).powi(2))
        .sum::<f64>()
        / opponents.len() as f64;
    OpponentScoreProfile {
        mean_score_stddev: Some(variance.sqrt()),
        min_opponent_mean: Some(
            opponents
                .iter()
                .map(|row| row.mean_score)
                .fold(f64::INFINITY, f64::min),
        ),
        max_opponent_mean: Some(
            opponents
                .iter()
                .map(|row| row.mean_score)
                .fold(f64::NEG_INFINITY, f64::max),
        ),
        opponents,
    }
}

struct PairwiseGenomeEvaluation {
    left_index: usize,
    right_index: usize,
    left_cases: Vec<CaseEvaluation>,
    right_cases: Vec<CaseEvaluation>,
}

fn balanced_pairings(
    population_size: usize,
    opponents_per_genome: usize,
    run_seed: u64,
    generation: u32,
) -> Vec<(usize, usize)> {
    let mut order = (0..population_size).collect::<Vec<_>>();
    let mut rng = event_rng(run_seed, generation, population_size, OPPONENT_DOMAIN);
    order.shuffle(&mut rng);
    let mut pairs = BTreeSet::new();
    for distance in 1..=opponents_per_genome / 2 {
        for position in 0..population_size {
            let left = order[position];
            let right = order[(position + distance) % population_size];
            pairs.insert((left.min(right), left.max(right)));
        }
    }
    if opponents_per_genome % 2 == 1 {
        debug_assert_eq!(population_size % 2, 0);
        for position in 0..population_size / 2 {
            let left = order[position];
            let right = order[position + population_size / 2];
            pairs.insert((left.min(right), left.max(right)));
        }
    }
    debug_assert_eq!(pairs.len() * 2, population_size * opponents_per_genome);
    pairs.into_iter().collect()
}

#[allow(clippy::too_many_arguments)]
fn evaluate_pairwise_genomes(
    left_index: usize,
    left: &OrganismGenome,
    right_index: usize,
    right: &OrganismGenome,
    pair_index: usize,
    scenarios: &[ScenarioManifest],
    episode_horizons: &[u64],
    survival_window_weights: &[f64],
    training_seeds: &[u64],
    objective_cvar_fraction: f64,
    cross_pool_predation_only: bool,
) -> Result<PairwiseGenomeEvaluation> {
    let case_capacity = scenarios
        .len()
        .saturating_mul(training_seeds.len())
        .saturating_mul(episode_horizons.len());
    let mut left_cases = Vec::with_capacity(case_capacity);
    let mut right_cases = Vec::with_capacity(case_capacity);
    let mut case_ordinal = 0usize;
    for &episode_ticks in episode_horizons {
        for scenario in scenarios {
            for &world_seed in training_seeds {
                let swap_slots = case_ordinal % 2 == 1;
                let instrument_left = (case_ordinal / 2 + pair_index).is_multiple_of(2);
                let (focal, opponent, focal_pool_index, focal_population_index, opponent_index) =
                    match (instrument_left, swap_slots) {
                        (true, false) => (left, right, 0, left_index, right_index),
                        (true, true) => (left, right, 1, left_index, right_index),
                        (false, false) => (right, left, 1, right_index, left_index),
                        (false, true) => (right, left, 0, right_index, left_index),
                    };
                let bundle = evaluate_genome_on_seeds_detailed(
                    focal,
                    std::slice::from_ref(scenario),
                    episode_ticks,
                    survival_window_weights,
                    std::slice::from_ref(&world_seed),
                    objective_cvar_fraction,
                    cross_pool_predation_only,
                    focal_pool_index,
                    std::slice::from_ref(opponent),
                )?;
                let mut focal_case = bundle
                    .cases
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("pairwise match produced no case"))?;
                focal_case.opponent_population_indices = vec![opponent_index];
                let other_case = case_for_focal_pool(
                    &focal_case,
                    1usize.saturating_sub(focal_case.focal_pool_index),
                    vec![focal_population_index],
                    survival_window_weights,
                );
                if instrument_left {
                    left_cases.push(focal_case);
                    right_cases.push(other_case);
                } else {
                    right_cases.push(focal_case);
                    left_cases.push(other_case);
                }
                case_ordinal += 1;
            }
        }
    }
    Ok(PairwiseGenomeEvaluation {
        left_index,
        right_index,
        left_cases,
        right_cases,
    })
}

fn case_for_focal_pool(
    source: &CaseEvaluation,
    focal_pool_index: usize,
    opponent_population_indices: Vec<usize>,
    survival_window_weights: &[f64],
) -> CaseEvaluation {
    debug_assert!(focal_pool_index < source.founder_pool_size);
    let mut mirrored = source.clone();
    let candidate_founders = source.founders_by_pool[focal_pool_index].max(1);
    let candidate_alive_ticks = source.alive_ticks_by_pool[focal_pool_index];
    let absolute_survival_fraction =
        candidate_alive_ticks as f64 / (candidate_founders as f64 * source.episode_horizon as f64);
    let total_weight = (0..source.episode_horizon)
        .map(|tick_index| {
            let weight_index = ((tick_index as u128 * survival_window_weights.len() as u128)
                / source.episode_horizon as u128)
                .min(survival_window_weights.len().saturating_sub(1) as u128)
                as usize;
            survival_window_weights[weight_index]
        })
        .sum::<f64>();
    let late_weighted_survival_fraction = source.weighted_alive_ticks_by_pool[focal_pool_index]
        / (candidate_founders as f64 * total_weight);
    let candidate_mean = candidate_alive_ticks as f64 / candidate_founders as f64;
    let opponent_founders = source
        .founders_by_pool
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != focal_pool_index)
        .map(|(_, founders)| *founders)
        .sum::<u64>()
        .max(1);
    let opponent_alive_ticks = source
        .alive_ticks_by_pool
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != focal_pool_index)
        .map(|(_, alive_ticks)| *alive_ticks)
        .sum::<u64>();
    let opponent_mean = opponent_alive_ticks as f64 / opponent_founders as f64;
    let combined = candidate_mean + opponent_mean;
    let relative_survival_advantage = if combined == 0.0 {
        0.0
    } else {
        2.0 * candidate_mean / combined
    };
    mirrored.focal_pool_index = focal_pool_index;
    mirrored.candidate_founders = candidate_founders;
    mirrored.candidate_end_survivors = source.end_survivors_by_pool[focal_pool_index];
    mirrored.absolute_survival_fraction = absolute_survival_fraction;
    mirrored.late_weighted_survival_fraction = late_weighted_survival_fraction;
    mirrored.relative_survival_advantage = relative_survival_advantage;
    mirrored.zero_combined_alive_ticks = combined == 0.0;
    mirrored.objective_score = absolute_survival_fraction;
    mirrored.requested_opponents = source.founder_pool_size.saturating_sub(1);
    mirrored.opponent_population_indices = opponent_population_indices;
    if let Some(diagnostics) = source.lineage_diagnostics_by_pool.get(focal_pool_index) {
        apply_lineage_case_diagnostics(&mut mirrored, diagnostics);
    }
    mirrored
}

fn apply_lineage_case_diagnostics(case: &mut CaseEvaluation, diagnostics: &LineageCaseDiagnostics) {
    case.attack_energy_received = diagnostics.attack_energy_received;
    case.attack_energy_lost = diagnostics.attack_energy_lost;
    case.attack_attempt_energy_cost = diagnostics.attack_attempt_energy_cost;
    case.net_attack_energy_balance = diagnostics.attack_energy_received
        - diagnostics.attack_energy_lost
        - diagnostics.attack_attempt_energy_cost;
    case.gross_energy_acquired = diagnostics.attack_energy_received;
    case.attack_no_organism_targets = diagnostics.attack_no_organism_targets;
    case.attack_target_evaded = diagnostics.attack_target_evaded;
    case.attack_same_pool_blocked = diagnostics.attack_same_pool_blocked;
    case.attack_insufficient_energy = diagnostics.attack_insufficient_energy;
    case.attack_eligible_attempts = diagnostics.attack_eligible_attempts;
    case.attack_hits = diagnostics.attack_hits;
    case.attack_nonlethal_hits = diagnostics.attack_nonlethal_hits;
    case.attack_kills = diagnostics.attack_kills;
    case.attack_same_pair_followups = diagnostics.attack_same_pair_followups;
    case.attack_followup_latency_ticks_sum = diagnostics.attack_followup_latency_ticks_sum;
    case.distinct_attack_victims = diagnostics.distinct_attack_victims;
    case.attack_repeat_hit_fraction = (diagnostics.attack_hits > 0)
        .then(|| diagnostics.attack_same_pair_followups as f64 / diagnostics.attack_hits as f64);
    case.attack_victim_energy_before_sum = diagnostics.attack_victim_energy_before_sum;
    case.attack_victim_energy_after_sum = diagnostics.attack_victim_energy_after_sum;
    case.action_effectiveness = diagnostics.action_effectiveness;
    case.successful_attack_rate = diagnostics.successful_attack_rate;
    case.learning_slope = diagnostics.learning_slope;
    case.spatial_coverage = diagnostics.spatial_coverage;
    case.action_fractions = diagnostics.action_fractions;
    case.commands_per_tick = diagnostics.commands_per_tick;
    case.multi_command_tick_fraction = diagnostics.multi_command_tick_fraction;
}

fn assign_selection_scores(population: &mut [Individual]) {
    for individual in population {
        individual.selection_score = individual.fitness.max(0.0);
    }
}

struct FounderPool {
    genomes: Vec<OrganismGenome>,
    opponent_population_indices: Vec<usize>,
}

struct EvaluationBundle {
    summary: Evaluation,
    cases: Vec<CaseEvaluation>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_genome_on_seeds_detailed(
    genome: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    survival_window_weights: &[f64],
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
    cross_pool_predation_only: bool,
    focal_pool_index: usize,
    opponents: &[OrganismGenome],
) -> Result<EvaluationBundle> {
    if scenarios.is_empty() || world_seeds.is_empty() {
        bail!("a genome evaluation needs at least one scenario and world seed");
    }
    if survival_window_weights.is_empty() {
        bail!("a genome evaluation needs at least one survival window weight");
    }
    let mut cases = Vec::with_capacity(scenarios.len() * world_seeds.len());
    for scenario in scenarios {
        for &world_seed in world_seeds {
            let mut genomes = Vec::with_capacity(opponents.len() + 1);
            genomes.extend_from_slice(opponents);
            genomes.insert(focal_pool_index.min(opponents.len()), genome.clone());
            let mut founder_pool = FounderPool {
                genomes,
                opponent_population_indices: (0..opponents.len()).collect(),
            };
            // A scenario cannot instantiate more genome sources than founders.
            // Cap explicitly (rather than silently leaving requested opponents
            // absent) and serialize the realized pool size in the case record.
            let requested_pool_len = founder_pool.genomes.len();
            founder_pool
                .genomes
                .truncate((scenario.world.num_organisms as usize).max(1));
            founder_pool
                .opponent_population_indices
                .truncate(founder_pool.genomes.len().saturating_sub(1));
            let pool_len = founder_pool.genomes.len();
            if focal_pool_index >= pool_len {
                bail!(
                    "focal pool index {focal_pool_index} is outside realized pool size {pool_len}"
                );
            }
            if pool_len != requested_pool_len {
                bail!(
                    "scenario `{}` can realize only {} of {} requested founder-pool entries",
                    scenario.name,
                    pool_len,
                    requested_pool_len
                );
            }
            let mut sim = Simulation::new_with_founder_genome_pool(
                scenario.world.clone(),
                world_seed,
                founder_pool.genomes,
            )
            .map_err(|e| {
                anyhow!(
                    "candidate evaluation failed in scenario `{}`: {e}",
                    scenario.name
                )
            })?;
            if cross_pool_predation_only {
                sim.set_cross_pool_predation_pool_count(Some(pool_len));
            }
            let world_founders = sim.organisms().len();
            if world_founders == 0 {
                bail!("scenario `{}` spawned no founders", scenario.name);
            }
            // Survival objective: with no in-world reproduction, every organism
            // is a founder attributed to its pool entry by species_id % pool_len
            // (the focal index identifies the candidate). We accumulate alive-ticks per pool entry and
            // score the candidate by the fraction of the episode its founders
            // survived (1.0 = all lived to the end).
            let mut founder_count_by_pool = vec![0u64; pool_len];
            for organism in sim.organisms() {
                founder_count_by_pool[(organism.species_id.0 as usize) % pool_len] += 1;
            }
            if founder_count_by_pool.contains(&0) {
                bail!(
                    "scenario `{}` spawned {} founders for a {}-genome pool; every requested pool entry must be represented",
                    scenario.name,
                    world_founders,
                    pool_len
                );
            }
            if founder_count_by_pool
                .windows(2)
                .any(|pair| pair[0] != pair[1])
            {
                bail!(
                    "scenario `{}` must represent every competitive genome with equal founder counts, got {:?}",
                    scenario.name,
                    founder_count_by_pool
                );
            }
            #[cfg(feature = "instrumentation")]
            let mut behavior_ledgers = {
                let mut ledgers = (0..pool_len).map(|_| Ledger::new()).collect::<Vec<_>>();
                for organism in sim.organisms() {
                    let pool_index = (organism.species_id.0 as usize) % pool_len;
                    ledgers[pool_index].birth(organism.id);
                }
                ledgers
            };
            let mut alive_ticks_by_pool = vec![0u64; pool_len];
            let mut weighted_alive_ticks_by_pool = vec![0.0_f64; pool_len];
            let mut total_survival_tick_weight = 0.0_f64;
            let world_width = sim.config().world_width as usize;
            let mut visited_by_pool =
                vec![vec![false; world_width.saturating_mul(world_width)]; pool_len];
            record_pool_visited_cells(sim.organisms(), pool_len, world_width, &mut visited_by_pool);
            let mut action_counts_by_pool = vec![[0u64; 5]; pool_len];
            let mut action_observations_by_pool = vec![0u64; pool_len];
            let mut command_count_by_pool = vec![0u64; pool_len];
            let mut multi_command_observations_by_pool = vec![0u64; pool_len];
            let mut lineage_diagnostics_by_pool = vec![LineageCaseDiagnostics::default(); pool_len];
            let mut distinct_attack_victims_by_pool = vec![BTreeSet::<OrganismId>::new(); pool_len];
            let mut last_hit_by_pair = HashMap::<(OrganismId, OrganismId), u64>::new();
            #[cfg(feature = "instrumentation")]
            let organism_pool_by_id = sim
                .organisms()
                .iter()
                .map(|organism| (organism.id, (organism.species_id.0 as usize) % pool_len))
                .collect::<HashMap<_, _>>();
            for _ in 0..episode_ticks {
                let delta = sim.tick();
                #[cfg(feature = "instrumentation")]
                {
                    for record in sim.action_records().iter().flatten() {
                        let pool_index = organism_pool_by_id[&record.organism_id];
                        behavior_ledgers[pool_index].record_action(record);
                    }
                }
                for event in sim.attack_events_last_turn() {
                    let attacker_pool = (event.attacker_species_id.0 as usize) % pool_len;
                    let attacker = &mut lineage_diagnostics_by_pool[attacker_pool];
                    attacker.attack_attempt_energy_cost += f64::from(event.attacker_energy_cost);
                    debug_assert!(event.victim_id.is_some() || event.victim_species_id.is_none());
                    match event.outcome {
                        AttackOutcome::InsufficientEnergy => {
                            attacker.attack_insufficient_energy += 1;
                        }
                        AttackOutcome::NoOrganismTarget => {
                            attacker.attack_no_organism_targets += 1;
                        }
                        AttackOutcome::TargetEvaded => {
                            attacker.attack_target_evaded += 1;
                        }
                        AttackOutcome::SamePoolBlocked => {
                            attacker.attack_same_pool_blocked += 1;
                        }
                        AttackOutcome::NonlethalHit | AttackOutcome::Killed => {
                            attacker.attack_eligible_attempts += 1;
                            attacker.attack_hits += 1;
                            attacker.attack_nonlethal_hits +=
                                u64::from(event.outcome == AttackOutcome::NonlethalHit);
                            attacker.attack_kills +=
                                u64::from(event.outcome == AttackOutcome::Killed);
                            attacker.attack_energy_received += f64::from(event.energy_transferred);
                            attacker.attack_victim_energy_before_sum +=
                                f64::from(event.victim_energy_before);
                            attacker.attack_victim_energy_after_sum +=
                                f64::from(event.victim_energy_after);
                            if let Some(victim_id) = event.victim_id {
                                distinct_attack_victims_by_pool[attacker_pool].insert(victim_id);
                                if let Some(previous_turn) = last_hit_by_pair
                                    .insert((event.attacker_id, victim_id), event.turn)
                                {
                                    attacker.attack_same_pair_followups += 1;
                                    attacker.attack_followup_latency_ticks_sum +=
                                        event.turn.saturating_sub(previous_turn);
                                }
                            }
                            let victim_pool = (event
                                .victim_species_id
                                .expect("a successful attack has a victim species")
                                .0 as usize)
                                % pool_len;
                            lineage_diagnostics_by_pool[victim_pool].attack_energy_lost +=
                                f64::from(event.energy_transferred);
                        }
                    }
                }
                let tick_index = delta.turn.saturating_sub(1);
                let weight_index = ((tick_index as u128 * survival_window_weights.len() as u128)
                    / episode_ticks as u128)
                    .min(survival_window_weights.len().saturating_sub(1) as u128)
                    as usize;
                let survival_tick_weight = survival_window_weights[weight_index];
                total_survival_tick_weight += survival_tick_weight;
                record_pool_visited_cells(
                    sim.organisms(),
                    pool_len,
                    world_width,
                    &mut visited_by_pool,
                );
                for organism in sim.organisms() {
                    let pool_index = (organism.species_id.0 as usize) % pool_len;
                    let command_count = organism.last_action_mask.count_ones() as u64;
                    if command_count == 0 {
                        action_counts_by_pool[pool_index][ActionType::Idle.index()] =
                            action_counts_by_pool[pool_index][ActionType::Idle.index()]
                                .checked_add(1)
                                .ok_or_else(|| anyhow!("action counter overflow"))?;
                    } else {
                        for action in ActionType::ALL {
                            if organism.last_action_mask & action.command_bit() != 0 {
                                action_counts_by_pool[pool_index][action.index()] =
                                    action_counts_by_pool[pool_index][action.index()]
                                        .checked_add(1)
                                        .ok_or_else(|| anyhow!("action counter overflow"))?;
                            }
                        }
                    }
                    command_count_by_pool[pool_index] = command_count_by_pool[pool_index]
                        .checked_add(command_count)
                        .ok_or_else(|| anyhow!("command counter overflow"))?;
                    multi_command_observations_by_pool[pool_index] =
                        multi_command_observations_by_pool[pool_index]
                            .checked_add(u64::from(command_count > 1))
                            .ok_or_else(|| anyhow!("multi-command counter overflow"))?;
                    action_observations_by_pool[pool_index] = action_observations_by_pool
                        [pool_index]
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("action-observation counter overflow"))?;
                    alive_ticks_by_pool[pool_index] += 1;
                    weighted_alive_ticks_by_pool[pool_index] += survival_tick_weight;
                }
            }
            if sim.turn() != episode_ticks {
                bail!(
                    "candidate stopped at turn {}; requested {}",
                    sim.turn(),
                    episode_ticks
                );
            }
            // Candidate survival fraction: mean fraction of the episode its
            // founders stayed alive. In the shared arena this is implicitly
            // competitive — surviving means outmaneuvering and out-lasting
            // rivals. Dense from generation 0
            // (a brain that lives longer scores higher even if none survive to
            // the end).
            let candidate_founders = founder_count_by_pool[focal_pool_index].max(1);
            let candidate_alive_ticks = alive_ticks_by_pool[focal_pool_index];
            let absolute_survival_fraction =
                candidate_alive_ticks as f64 / (candidate_founders as f64 * episode_ticks as f64);
            let late_weighted_survival_fraction = weighted_alive_ticks_by_pool[focal_pool_index]
                / (candidate_founders as f64 * total_survival_tick_weight);
            let (relative_survival_advantage, zero_combined_alive_ticks) = if pool_len == 1 {
                (1.0, false)
            } else {
                let opponent_founders = founder_count_by_pool
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != focal_pool_index)
                    .map(|(_, value)| *value)
                    .sum::<u64>();
                let opponent_alive_ticks = alive_ticks_by_pool
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != focal_pool_index)
                    .map(|(_, value)| *value)
                    .sum::<u64>();
                debug_assert!(opponent_founders > 0);
                let candidate_mean_alive_ticks =
                    candidate_alive_ticks as f64 / candidate_founders as f64;
                let opponent_mean_alive_ticks =
                    opponent_alive_ticks as f64 / opponent_founders as f64;
                let combined = candidate_mean_alive_ticks + opponent_mean_alive_ticks;
                if combined == 0.0 {
                    (0.0, true)
                } else {
                    (2.0 * candidate_mean_alive_ticks / combined, false)
                }
            };
            let objective_score = absolute_survival_fraction;
            // Founders still alive at the end, retained per pool so mirrored
            // and multi-lineage evaluations do not infer one pool by subtraction.
            let mut end_survivors_by_pool = vec![0u64; pool_len];
            for organism in sim.organisms() {
                end_survivors_by_pool[(organism.species_id.0 as usize) % pool_len] += 1;
            }
            let candidate_end_survivors = end_survivors_by_pool[focal_pool_index];
            let habitable_cells = sim.habitable_cell_count().max(1) as f64;
            for (pool_index, (diagnostics, victims)) in lineage_diagnostics_by_pool
                .iter_mut()
                .zip(&distinct_attack_victims_by_pool)
                .enumerate()
            {
                diagnostics.distinct_attack_victims = victims.len() as u64;
                #[cfg(feature = "instrumentation")]
                {
                    let row = behavior_ledgers[pool_index].take_behavior_interval(episode_ticks);
                    let metrics = derive_interval_metrics(std::slice::from_ref(&row))
                        .into_iter()
                        .next()
                        .expect("one behavior interval produces one metric row");
                    diagnostics.action_effectiveness = metrics.action_effectiveness;
                    diagnostics.successful_attack_rate = metrics.successful_attack_rate;
                    diagnostics.learning_slope = metrics.learning_slope;
                }
                diagnostics.spatial_coverage = visited_by_pool[pool_index]
                    .iter()
                    .filter(|&&seen| seen)
                    .count() as f64
                    / habitable_cells;
                let observations = action_observations_by_pool[pool_index];
                if observations > 0 {
                    for (fraction, count) in diagnostics
                        .action_fractions
                        .iter_mut()
                        .zip(action_counts_by_pool[pool_index])
                    {
                        *fraction = count as f64 / observations as f64;
                    }
                    diagnostics.commands_per_tick =
                        command_count_by_pool[pool_index] as f64 / observations as f64;
                    diagnostics.multi_command_tick_fraction =
                        multi_command_observations_by_pool[pool_index] as f64 / observations as f64;
                }
            }
            let focal_diagnostics = &lineage_diagnostics_by_pool[focal_pool_index];
            let attack_energy_received = focal_diagnostics.attack_energy_received;
            let attack_energy_lost = focal_diagnostics.attack_energy_lost;
            let attack_attempt_energy_cost = focal_diagnostics.attack_attempt_energy_cost;
            let net_attack_energy_balance =
                attack_energy_received - attack_energy_lost - attack_attempt_energy_cost;
            let gross_energy_acquired = attack_energy_received;
            let attack_repeat_hit_fraction = (focal_diagnostics.attack_hits > 0).then(|| {
                focal_diagnostics.attack_same_pair_followups as f64
                    / focal_diagnostics.attack_hits as f64
            });
            cases.push(CaseEvaluation {
                scenario: scenario.name.clone(),
                world_seed,
                episode_horizon: episode_ticks,
                objective_score,
                absolute_survival_fraction,
                late_weighted_survival_fraction,
                relative_survival_advantage,
                zero_combined_alive_ticks,
                requested_opponents: requested_pool_len.saturating_sub(1),
                opponent_population_indices: founder_pool.opponent_population_indices,
                founders_by_pool: founder_count_by_pool,
                alive_ticks_by_pool,
                weighted_alive_ticks_by_pool,
                end_survivors_by_pool,
                founder_pool_size: pool_len,
                focal_pool_index,
                world_founders,
                candidate_founders,
                candidate_end_survivors,
                action_effectiveness: focal_diagnostics.action_effectiveness,
                successful_attack_rate: focal_diagnostics.successful_attack_rate,
                learning_slope: focal_diagnostics.learning_slope,
                gross_energy_acquired,
                attack_energy_received,
                attack_energy_lost,
                attack_attempt_energy_cost,
                net_attack_energy_balance,
                attack_no_organism_targets: focal_diagnostics.attack_no_organism_targets,
                attack_target_evaded: focal_diagnostics.attack_target_evaded,
                attack_same_pool_blocked: focal_diagnostics.attack_same_pool_blocked,
                attack_insufficient_energy: focal_diagnostics.attack_insufficient_energy,
                attack_eligible_attempts: focal_diagnostics.attack_eligible_attempts,
                attack_hits: focal_diagnostics.attack_hits,
                attack_nonlethal_hits: focal_diagnostics.attack_nonlethal_hits,
                attack_kills: focal_diagnostics.attack_kills,
                attack_same_pair_followups: focal_diagnostics.attack_same_pair_followups,
                attack_followup_latency_ticks_sum: focal_diagnostics
                    .attack_followup_latency_ticks_sum,
                distinct_attack_victims: focal_diagnostics.distinct_attack_victims,
                attack_repeat_hit_fraction,
                attack_victim_energy_before_sum: focal_diagnostics.attack_victim_energy_before_sum,
                attack_victim_energy_after_sum: focal_diagnostics.attack_victim_energy_after_sum,
                spatial_coverage: focal_diagnostics.spatial_coverage,
                action_fractions: focal_diagnostics.action_fractions,
                commands_per_tick: focal_diagnostics.commands_per_tick,
                multi_command_tick_fraction: focal_diagnostics.multi_command_tick_fraction,
                world_final_population: sim.organisms().len(),
                lineage_diagnostics_by_pool,
            });
        }
    }
    let summary = summarize_evaluation_cases(&cases, objective_cvar_fraction);
    Ok(EvaluationBundle { summary, cases })
}

fn summarize_evaluation_cases(
    cases: &[CaseEvaluation],
    objective_cvar_fraction: f64,
) -> Evaluation {
    debug_assert!(!cases.is_empty());
    let n = cases.len() as f64;
    let mut scored_cases = cases
        .iter()
        .enumerate()
        .map(|(index, case)| (case.objective_score, index))
        .collect::<Vec<_>>();
    scored_cases.sort_by(|(left_score, left_index), (right_score, right_index)| {
        left_score
            .total_cmp(right_score)
            .then_with(|| left_index.cmp(right_index))
    });
    let cvar_count = ((scored_cases.len() as f64 * objective_cvar_fraction).ceil() as usize)
        .clamp(1, scored_cases.len());
    let mean_case_score =
        scored_cases.iter().map(|(score, _)| score).sum::<f64>() / scored_cases.len() as f64;
    let case_score_stddev = (scored_cases
        .iter()
        .map(|(score, _)| (score - mean_case_score).powi(2))
        .sum::<f64>()
        / scored_cases.len() as f64)
        .sqrt();
    let mean_objective_score = scored_cases[..cvar_count]
        .iter()
        .map(|(score, _)| score)
        .sum::<f64>()
        / cvar_count as f64;
    let objective_cvar_absolute_survival_fraction = scored_cases[..cvar_count]
        .iter()
        .map(|(_, index)| cases[*index].absolute_survival_fraction)
        .sum::<f64>()
        / cvar_count as f64;
    let objective_cvar_late_weighted_survival_fraction = scored_cases[..cvar_count]
        .iter()
        .map(|(_, index)| cases[*index].late_weighted_survival_fraction)
        .sum::<f64>()
        / cvar_count as f64;
    let objective_cvar_relative_survival_advantage = scored_cases[..cvar_count]
        .iter()
        .map(|(_, index)| cases[*index].relative_survival_advantage)
        .sum::<f64>()
        / cvar_count as f64;
    let mut mean_action_fractions = [0.0; 5];
    for case in cases {
        for (mean, value) in mean_action_fractions.iter_mut().zip(case.action_fractions) {
            *mean += value / n;
        }
    }
    Evaluation {
        mean_objective_score,
        mean_case_score,
        case_score_stddev,
        min_case_score: scored_cases
            .first()
            .expect("evaluation has at least one scored case")
            .0,
        max_case_score: scored_cases
            .last()
            .expect("evaluation has at least one scored case")
            .0,
        mean_absolute_survival_fraction: cases
            .iter()
            .map(|case| case.absolute_survival_fraction)
            .sum::<f64>()
            / n,
        mean_candidate_alive_ticks: cases
            .iter()
            .map(|case| case.alive_ticks_by_pool[case.focal_pool_index] as f64)
            .sum::<f64>()
            / n,
        mean_late_weighted_survival_fraction: cases
            .iter()
            .map(|case| case.late_weighted_survival_fraction)
            .sum::<f64>()
            / n,
        mean_relative_survival_advantage: cases
            .iter()
            .map(|case| case.relative_survival_advantage)
            .sum::<f64>()
            / n,
        objective_cvar_absolute_survival_fraction,
        objective_cvar_late_weighted_survival_fraction,
        objective_cvar_relative_survival_advantage,
        objective_cvar_case_count: cvar_count,
        zero_combined_alive_tick_cases: cases
            .iter()
            .filter(|case| case.zero_combined_alive_ticks)
            .count(),
        pair_seed_cases: cases.len(),
        unique_opponents: cases
            .iter()
            .flat_map(|case| case.opponent_population_indices.iter().copied())
            .collect::<BTreeSet<_>>()
            .len(),
        mean_candidate_end_survival_fraction: cases
            .iter()
            .map(|case| case.candidate_end_survivors as f64 / case.candidate_founders.max(1) as f64)
            .sum::<f64>()
            / n,
        mean_action_effectiveness: mean_optional(
            cases.iter().map(|case| case.action_effectiveness),
        ),
        mean_successful_attack_rate: mean_optional(
            cases.iter().map(|case| case.successful_attack_rate),
        ),
        mean_learning_slope: mean_optional(cases.iter().map(|case| case.learning_slope)),
        mean_gross_energy_acquired: cases
            .iter()
            .map(|case| case.gross_energy_acquired)
            .sum::<f64>()
            / n,
        mean_attack_energy_received: cases
            .iter()
            .map(|case| case.attack_energy_received)
            .sum::<f64>()
            / n,
        mean_attack_energy_lost: cases
            .iter()
            .map(|case| case.attack_energy_lost)
            .sum::<f64>()
            / n,
        mean_attack_attempt_energy_cost: cases
            .iter()
            .map(|case| case.attack_attempt_energy_cost)
            .sum::<f64>()
            / n,
        mean_net_attack_energy_balance: cases
            .iter()
            .map(|case| case.net_attack_energy_balance)
            .sum::<f64>()
            / n,
        mean_attack_no_organism_targets: cases
            .iter()
            .map(|case| case.attack_no_organism_targets as f64)
            .sum::<f64>()
            / n,
        mean_attack_target_evaded: cases
            .iter()
            .map(|case| case.attack_target_evaded as f64)
            .sum::<f64>()
            / n,
        mean_attack_same_pool_blocked: cases
            .iter()
            .map(|case| case.attack_same_pool_blocked as f64)
            .sum::<f64>()
            / n,
        mean_attack_insufficient_energy: cases
            .iter()
            .map(|case| case.attack_insufficient_energy as f64)
            .sum::<f64>()
            / n,
        mean_attack_eligible_attempts: cases
            .iter()
            .map(|case| case.attack_eligible_attempts as f64)
            .sum::<f64>()
            / n,
        mean_attack_hits: cases
            .iter()
            .map(|case| case.attack_hits as f64)
            .sum::<f64>()
            / n,
        mean_attack_nonlethal_hits: cases
            .iter()
            .map(|case| case.attack_nonlethal_hits as f64)
            .sum::<f64>()
            / n,
        mean_attack_kills: cases
            .iter()
            .map(|case| case.attack_kills as f64)
            .sum::<f64>()
            / n,
        mean_attack_same_pair_followups: cases
            .iter()
            .map(|case| case.attack_same_pair_followups as f64)
            .sum::<f64>()
            / n,
        mean_distinct_attack_victims: cases
            .iter()
            .map(|case| case.distinct_attack_victims as f64)
            .sum::<f64>()
            / n,
        attack_repeat_hit_fraction: {
            let hits = cases.iter().map(|case| case.attack_hits).sum::<u64>();
            (hits > 0).then(|| {
                cases
                    .iter()
                    .map(|case| case.attack_same_pair_followups)
                    .sum::<u64>() as f64
                    / hits as f64
            })
        },
        mean_spatial_coverage: cases.iter().map(|case| case.spatial_coverage).sum::<f64>() / n,
        mean_action_fractions,
        mean_commands_per_tick: cases.iter().map(|case| case.commands_per_tick).sum::<f64>() / n,
        mean_multi_command_tick_fraction: cases
            .iter()
            .map(|case| case.multi_command_tick_fraction)
            .sum::<f64>()
            / n,
        mean_world_final_population: cases
            .iter()
            .map(|case| case.world_final_population as f64)
            .sum::<f64>()
            / n,
    }
}

/// Evaluate one frozen focal genome against an explicit, common opponent panel.
/// Every scenario/seed case uses the same ordered opponent genomes, making the
/// survival diagnostics comparable across frozen focal genomes.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_frozen_panel(
    focal: &OrganismGenome,
    opponents: &[OrganismGenome],
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    survival_window_weights: &[f64],
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
    cross_pool_predation_only: bool,
    focal_pool_index: usize,
) -> Result<PanelEvaluation> {
    if opponents.is_empty() {
        bail!("frozen-panel evaluation needs at least one opponent");
    }
    let bundle = evaluate_genome_on_seeds_detailed(
        focal,
        scenarios,
        episode_ticks,
        survival_window_weights,
        world_seeds,
        objective_cvar_fraction,
        cross_pool_predation_only,
        focal_pool_index,
        opponents,
    )?;
    Ok(PanelEvaluation {
        summary: bundle.summary,
        cases: bundle.cases,
    })
}

/// Evaluate two frozen genomes through the exact symmetric two-lineage
/// contract used by balanced pairwise NEAT training. Each world is scored for
/// both genomes, and every four-seed block rotates both genomes through both
/// founder/ID slots and direct behavioral instrumentation slots.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_frozen_pair(
    left: &OrganismGenome,
    right: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_horizons: &[u64],
    survival_window_weights: &[f64],
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
    cross_pool_predation_only: bool,
) -> Result<PairEvaluation> {
    if world_seeds.is_empty() || !world_seeds.len().is_multiple_of(4) {
        bail!("frozen pair evaluation needs a nonempty multiple of four world seeds");
    }
    let evaluated = evaluate_pairwise_genomes(
        0,
        left,
        1,
        right,
        0,
        scenarios,
        episode_horizons,
        survival_window_weights,
        world_seeds,
        objective_cvar_fraction,
        cross_pool_predation_only,
    )?;
    let left_summary = summarize_evaluation_cases(&evaluated.left_cases, objective_cvar_fraction);
    let right_summary = summarize_evaluation_cases(&evaluated.right_cases, objective_cvar_fraction);
    Ok(PairEvaluation {
        left: PanelEvaluation {
            summary: left_summary,
            cases: evaluated.left_cases,
        },
        right: PanelEvaluation {
            summary: right_summary,
            cases: evaluated.right_cases,
        },
    })
}

fn record_pool_visited_cells(
    organisms: &[types::OrganismState],
    pool_len: usize,
    world_width: usize,
    visited_by_pool: &mut [Vec<bool>],
) {
    for organism in organisms {
        let pool_index = (organism.species_id.0 as usize) % pool_len;
        let index = organism.r as usize * world_width + organism.q as usize;
        if let Some(cell) = visited_by_pool[pool_index].get_mut(index) {
            *cell = true;
        }
    }
}

fn mean_optional(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.flatten() {
        sum += value;
        count += 1;
    }
    (count > 0).then(|| sum / count as f64)
}

fn assign_species(
    population: &[Individual],
    previous: &[SpeciesRecord],
    config: &NeatConfig,
    compatibility_threshold: f64,
    generation: u32,
    next_species_id: &mut u64,
) -> Vec<SpeciesRecord> {
    let mut species: Vec<SpeciesRecord> = previous
        .iter()
        .map(|old| SpeciesRecord {
            id: old.id,
            representative: old.representative.clone(),
            members: Vec::new(),
            created_generation: old.created_generation,
        })
        .collect();
    for (index, individual) in population.iter().enumerate() {
        let assigned = species
            .iter()
            .enumerate()
            .filter_map(|(species_index, candidate)| {
                let distance =
                    compatibility_distance(&individual.genome, &candidate.representative, config);
                (distance < compatibility_threshold).then_some((species_index, distance))
            })
            .min_by(
                |(left_index, left_distance), (right_index, right_distance)| {
                    left_distance
                        .total_cmp(right_distance)
                        .then_with(|| species[*left_index].id.cmp(&species[*right_index].id))
                },
            )
            .map(|(species_index, _)| species_index);
        if let Some(species_index) = assigned {
            species[species_index].members.push(index);
        } else {
            species.push(SpeciesRecord {
                id: *next_species_id,
                representative: individual.genome.clone(),
                members: vec![index],
                created_generation: generation,
            });
            *next_species_id = next_species_id.saturating_add(1);
        }
    }
    species.retain(|entry| !entry.members.is_empty());
    for entry in &mut species {
        entry.members.sort_unstable();
        entry.representative = population[entry.members[0]].genome.clone();
    }
    species.sort_by_key(|entry| entry.id);
    species
}

fn compatibility_distance(a: &OrganismGenome, b: &OrganismGenome, config: &NeatConfig) -> f64 {
    let ae = &a.brain.edges;
    let be = &b.brain.edges;
    let max_a = ae.last().map(|edge| edge.innovation.0).unwrap_or(0);
    let max_b = be.last().map(|edge| edge.innovation.0).unwrap_or(0);
    let mut i = 0;
    let mut j = 0;
    let mut matching = 0usize;
    let mut disjoint = 0usize;
    let mut excess = 0usize;
    let mut weight_delta = 0.0f64;
    while i < ae.len() && j < be.len() {
        match ae[i].innovation.cmp(&be[j].innovation) {
            std::cmp::Ordering::Equal => {
                matching += 1;
                weight_delta += f64::from((ae[i].weight - be[j].weight).abs());
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                if ae[i].innovation.0 > max_b {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if be[j].innovation.0 > max_a {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
                j += 1;
            }
        }
    }
    excess += ae.len() - i + be.len() - j;
    let normalizer = if ae.len().max(be.len()) < 20 {
        1.0
    } else {
        ae.len().max(be.len()) as f64
    };
    let mean_weight_delta = if matching == 0 {
        0.0
    } else {
        weight_delta / matching as f64
    };
    config.excess_coefficient * excess as f64 / normalizer
        + config.disjoint_coefficient * disjoint as f64 / normalizer
        + config.weight_coefficient * mean_weight_delta
}

#[allow(clippy::too_many_arguments)]
fn breed_next_generation(
    population: &[Individual],
    species: &mut [SpeciesRecord],
    config: &NeatConfig,
    innovations: &mut InnovationRegistry,
    run_seed: u64,
    parent_generation: u32,
    global_best_index: usize,
    offspring_generation: u32,
    predation_enabled: bool,
) -> Result<(Vec<Individual>, usize, usize, BreedingTelemetry)> {
    let _ = global_best_index;
    let active: Vec<usize> = (0..species.len()).collect();
    let allocations = allocate_offspring(
        population,
        species,
        &active,
        config.population_size,
        config,
        parent_generation,
    );
    let mut next = Vec::with_capacity(config.population_size);
    let mut crossover_count = 0usize;
    let mut clone_count = 0usize;
    let mut telemetry = BreedingTelemetry::default();

    for (&species_index, &allocation) in active.iter().zip(&allocations) {
        if allocation == 0 {
            continue;
        }
        let entry = &species[species_index];
        let mut ranked = entry.members.clone();
        ranked.sort_by(|&a, &b| {
            population[b]
                .selection_score
                .total_cmp(&population[a].selection_score)
                .then_with(|| population[b].fitness.total_cmp(&population[a].fitness))
                .then_with(|| a.cmp(&b))
        });
        let survivor_count = ((ranked.len() as f64 * config.survival_fraction).ceil() as usize)
            .clamp(1, ranked.len());
        let survivors = &ranked[..survivor_count];
        let mut produced = 0usize;
        let young = parent_generation.saturating_sub(entry.created_generation)
            <= config.young_species_grace_generations;
        if ranked[0] == global_best_index
            || young
            || ranked.len() >= config.elitism_min_species_size
        {
            next.push(blank_individual(
                population[ranked[0]].genome.clone(),
                ReproductionKind::EliteClone,
                vec![ParentReference {
                    generation: parent_generation,
                    population_index: ranked[0],
                }],
            ));
            produced += 1;
        }
        while produced < allocation {
            let offspring_slot = next.len();
            let mut selection_rng = event_rng(
                run_seed,
                parent_generation,
                offspring_slot,
                BREED_SELECTION_DOMAIN,
            );
            let parent_a = select_parent(population, survivors, &mut selection_rng);
            let wants_crossover = selection_rng.random_bool(config.crossover_probability);
            let use_interspecies = wants_crossover
                && active.len() > 1
                && selection_rng.random_bool(config.interspecies_mate_probability);
            let parent_b = if use_interspecies {
                let other_species = active
                    .iter()
                    .copied()
                    .filter(|&idx| idx != species_index)
                    .collect::<Vec<_>>();
                let chosen = other_species[selection_rng.random_range(0..other_species.len())];
                Some(select_parent(
                    population,
                    &species[chosen].members,
                    &mut selection_rng,
                ))
            } else if wants_crossover {
                select_parent_excluding(population, survivors, parent_a, &mut selection_rng)
            } else {
                None
            };
            let do_crossover = parent_b.is_some();
            let mut child = if let Some(parent_b) = parent_b {
                let mut crossover_rng = event_rng(
                    run_seed,
                    parent_generation,
                    offspring_slot,
                    CROSSOVER_DOMAIN,
                );
                crossover(
                    &population[parent_a],
                    &population[parent_b],
                    config,
                    &mut crossover_rng,
                )
            } else {
                population[parent_a].genome.clone()
            };
            if do_crossover {
                crossover_count += 1;
            } else {
                clone_count += 1;
            }
            let mut mutation_rng =
                event_rng(run_seed, parent_generation, offspring_slot, MUTATION_DOMAIN);
            let mutation = mutate(
                &mut child,
                innovations,
                config,
                &mut mutation_rng,
                offspring_generation,
                predation_enabled,
            );
            telemetry.non_elite_offspring += 1;
            telemetry.structural_mutation_attempts += mutation.structural_attempts;
            telemetry.structural_mutation_successes += mutation.structural_successes;
            telemetry.registry_new_structural_mutations += mutation.registry_new_structures;
            telemetry.new_origin_offspring += usize::from(mutation.registry_new_structures > 0);
            let mut parents = vec![ParentReference {
                generation: parent_generation,
                population_index: parent_a,
            }];
            if let Some(parent_b) = parent_b {
                parents.push(ParentReference {
                    generation: parent_generation,
                    population_index: parent_b,
                });
            }
            next.push(blank_individual(
                child,
                if do_crossover {
                    ReproductionKind::Crossover
                } else {
                    ReproductionKind::AsexualMutation
                },
                parents,
            ));
            produced += 1;
        }
    }
    while next.len() < config.population_size {
        let offspring_slot = next.len();
        let mut genome = population[global_best_index].genome.clone();
        let mut mutation_rng =
            event_rng(run_seed, parent_generation, offspring_slot, MUTATION_DOMAIN);
        let mutation = mutate(
            &mut genome,
            innovations,
            config,
            &mut mutation_rng,
            offspring_generation,
            predation_enabled,
        );
        telemetry.non_elite_offspring += 1;
        telemetry.structural_mutation_attempts += mutation.structural_attempts;
        telemetry.structural_mutation_successes += mutation.structural_successes;
        telemetry.registry_new_structural_mutations += mutation.registry_new_structures;
        telemetry.new_origin_offspring += usize::from(mutation.registry_new_structures > 0);
        next.push(blank_individual(
            genome,
            ReproductionKind::AsexualMutation,
            vec![ParentReference {
                generation: parent_generation,
                population_index: global_best_index,
            }],
        ));
        clone_count += 1;
    }
    next.truncate(config.population_size);
    Ok((next, crossover_count, clone_count, telemetry))
}

fn allocate_offspring(
    population: &[Individual],
    species: &[SpeciesRecord],
    active: &[usize],
    total: usize,
    config: &NeatConfig,
    generation: u32,
) -> Vec<usize> {
    let scores: Vec<f64> = active
        .iter()
        .map(|&index| {
            let entry = &species[index];
            entry
                .members
                .iter()
                .map(|&member| population[member].selection_score.max(0.0))
                .sum::<f64>()
                / entry.members.len() as f64
        })
        .collect();
    let sum: f64 = scores.iter().sum();
    let minimums: Vec<usize> = active
        .iter()
        .map(|&index| {
            let young = generation.saturating_sub(species[index].created_generation)
                <= config.young_species_grace_generations;
            if young {
                config.min_young_species_offspring
            } else {
                1
            }
        })
        .collect();
    let reserved = minimums.iter().sum::<usize>().min(total);
    let distributable = total.saturating_sub(reserved);
    let quotas: Vec<f64> = if sum > 0.0 {
        scores
            .iter()
            .map(|score| score / sum * distributable as f64)
            .collect()
    } else {
        vec![distributable as f64 / active.len() as f64; active.len()]
    };
    let mut allocations: Vec<usize> = quotas
        .iter()
        .enumerate()
        .map(|(index, quota)| minimums[index] + quota.floor() as usize)
        .collect();
    if allocations.iter().sum::<usize>() > total {
        // More protected species than slots: deterministic one-slot survival
        // in descending score order. The adaptive threshold will merge them in
        // subsequent generations.
        allocations.fill(0);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .total_cmp(&scores[a])
                .then_with(|| species[active[a]].id.cmp(&species[active[b]].id))
        });
        for index in order.into_iter().take(total) {
            allocations[index] = 1;
        }
        return allocations;
    }
    let mut remainder = total.saturating_sub(allocations.iter().sum());
    let mut order: Vec<usize> = (0..quotas.len()).collect();
    order.sort_by(|&a, &b| {
        (quotas[b] - quotas[b].floor())
            .total_cmp(&(quotas[a] - quotas[a].floor()))
            .then_with(|| species[active[a]].id.cmp(&species[active[b]].id))
    });
    for index in order.into_iter().cycle().take(remainder) {
        allocations[index] += 1;
        remainder -= 1;
        if remainder == 0 {
            break;
        }
    }
    allocations
}

fn select_parent(population: &[Individual], candidates: &[usize], rng: &mut ChaCha8Rng) -> usize {
    let total: f64 = candidates
        .iter()
        .map(|&index| population[index].selection_score.max(0.0))
        .sum();
    if total <= 0.0 {
        return candidates[rng.random_range(0..candidates.len())];
    }
    let mut draw = rng.random_range(0.0..total);
    for &index in candidates {
        draw -= population[index].selection_score.max(0.0);
        if draw <= 0.0 {
            return index;
        }
    }
    *candidates.last().expect("species candidates are non-empty")
}

fn select_parent_excluding(
    population: &[Individual],
    candidates: &[usize],
    excluded: usize,
    rng: &mut ChaCha8Rng,
) -> Option<usize> {
    let total: f64 = candidates
        .iter()
        .copied()
        .filter(|&index| index != excluded)
        .map(|index| population[index].selection_score.max(0.0))
        .sum();
    if total <= 0.0 {
        return None;
    }
    let mut draw = rng.random_range(0.0..total);
    let mut last = None;
    for &index in candidates {
        if index == excluded {
            continue;
        }
        last = Some(index);
        draw -= population[index].selection_score.max(0.0);
        if draw <= 0.0 {
            return Some(index);
        }
    }
    last
}

fn crossover(
    a: &Individual,
    b: &Individual,
    config: &NeatConfig,
    rng: &mut ChaCha8Rng,
) -> OrganismGenome {
    let a_fitter = a.selection_score > b.selection_score;
    let b_fitter = b.selection_score > a.selection_score;
    let base = if b_fitter { &b.genome } else { &a.genome };
    let mut child = base.clone();
    child.brain.edges.clear();
    child.brain.hidden_nodes.clear();
    child.brain.action_biases = a
        .genome
        .brain
        .action_biases
        .iter()
        .zip(&b.genome.brain.action_biases)
        .map(|(&av, &bv)| if rng.random_bool(0.5) { av } else { bv })
        .collect();

    let ae = &a.genome.brain.edges;
    let be = &b.genome.brain.edges;
    let mut i = 0;
    let mut j = 0;
    while i < ae.len() || j < be.len() {
        let chosen = match (ae.get(i), be.get(j)) {
            (Some(left), Some(right)) if left.innovation == right.innovation => {
                i += 1;
                j += 1;
                let mut gene = if rng.random_bool(0.5) { *left } else { *right };
                if !left.enabled || !right.enabled {
                    gene.enabled = !rng.random_bool(config.disabled_inheritance_probability);
                }
                Some(gene)
            }
            (Some(left), Some(right)) if left.innovation < right.innovation => {
                i += 1;
                (a_fitter || (!b_fitter && rng.random_bool(0.5))).then_some(*left)
            }
            (Some(_), Some(right)) => {
                j += 1;
                (b_fitter || (!a_fitter && rng.random_bool(0.5))).then_some(*right)
            }
            (Some(left), None) => {
                i += 1;
                (a_fitter || (!b_fitter && rng.random_bool(0.5))).then_some(*left)
            }
            (None, Some(right)) => {
                j += 1;
                (b_fitter || (!a_fitter && rng.random_bool(0.5))).then_some(*right)
            }
            (None, None) => None,
        };
        if let Some(gene) = chosen {
            child.brain.edges.push(gene);
        }
    }

    let a_nodes: BTreeMap<_, _> = a
        .genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| (node.id, *node))
        .collect();
    let b_nodes: BTreeMap<_, _> = b
        .genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| (node.id, *node))
        .collect();
    let mut required = Vec::new();
    for edge in &child.brain.edges {
        if is_hidden_gene_node_id(edge.pre_node_id) {
            required.push(edge.pre_node_id);
        }
        if is_hidden_gene_node_id(edge.post_node_id) {
            required.push(edge.post_node_id);
        }
    }
    required.sort_unstable();
    required.dedup();
    required.retain(|id| a_nodes.contains_key(id) || b_nodes.contains_key(id));
    // Evaluator intake retains the lowest stable hidden IDs at the runtime
    // ceiling. Mirror that rule here so selection and reported genotypes see
    // exactly the phenotype that will be materialized.
    required.truncate(MAX_INTER_NEURONS as usize);
    let retained_hidden: BTreeSet<_> = required.iter().copied().collect();
    child.brain.edges.retain(|edge| {
        (!is_hidden_gene_node_id(edge.pre_node_id) || retained_hidden.contains(&edge.pre_node_id))
            && (!is_hidden_gene_node_id(edge.post_node_id)
                || retained_hidden.contains(&edge.post_node_id))
    });
    for id in required {
        let node = match (a_nodes.get(&id), b_nodes.get(&id)) {
            (Some(left), Some(right)) => {
                if rng.random_bool(0.5) {
                    *left
                } else {
                    *right
                }
            }
            (Some(node), None) | (None, Some(node)) => *node,
            (None, None) => continue,
        };
        child.brain.hidden_nodes.push(node);
    }
    child
        .brain
        .hidden_nodes
        .sort_unstable_by_key(|node| node.id);
    child
        .brain
        .edges
        .sort_unstable_by_key(|edge| edge.innovation);
    brain::genome::enforce_feed_forward_edges(&mut child);
    child
}

fn mutate(
    genome: &mut OrganismGenome,
    innovations: &mut InnovationRegistry,
    config: &NeatConfig,
    rng: &mut ChaCha8Rng,
    offspring_generation: u32,
    predation_enabled: bool,
) -> MutationOutcome {
    let mut outcome = MutationOutcome::default();
    if rng.random_bool(config.mutate_weight_probability) {
        mutate_parameters(genome, config, rng);
    }
    if rng.random_bool(config.add_connection_probability) {
        outcome.structural_attempts += 1;
        let (succeeded, registry_new) = mutate_add_connection(
            genome,
            innovations,
            rng,
            offspring_generation,
            predation_enabled,
        );
        outcome.structural_successes += usize::from(succeeded);
        outcome.registry_new_structures += usize::from(registry_new);
    }
    if rng.random_bool(config.add_node_probability) {
        outcome.structural_attempts += 1;
        let (succeeded, registry_new) =
            mutate_add_node(genome, innovations, rng, offspring_generation);
        outcome.structural_successes += usize::from(succeeded);
        outcome.registry_new_structures += usize::from(registry_new);
    }
    brain::genome::restrict_action_genes(genome, predation_enabled);
    outcome
}

fn mutate_parameters(genome: &mut OrganismGenome, config: &NeatConfig, rng: &mut ChaCha8Rng) {
    for edge in &mut genome.brain.edges {
        if !rng.random_bool(config.per_connection_weight_mutation_probability) {
            continue;
        }
        if rng.random_bool(config.replace_weight_probability) {
            edge.weight = random_weight(rng);
        } else {
            edge.weight =
                constrain_weight(edge.weight + normal(rng) * config.weight_perturb_stddev);
        }
    }
    for bias in &mut genome.brain.action_biases {
        if rng.random_bool(config.mutate_bias_probability) {
            *bias = (*bias + normal(rng) * config.bias_perturb_stddev)
                .clamp(-BIAS_MAX_ABS, BIAS_MAX_ABS);
        }
    }
    for node in &mut genome.brain.hidden_nodes {
        if rng.random_bool(config.mutate_bias_probability) {
            node.bias = (node.bias + normal(rng) * config.bias_perturb_stddev)
                .clamp(-BIAS_MAX_ABS, BIAS_MAX_ABS);
        }
        if rng.random_bool(config.mutate_time_constant_probability) {
            node.log_time_constant = (node.log_time_constant
                + normal(rng) * config.time_constant_perturb_stddev)
                .clamp(-std::f32::consts::LN_10, std::f32::consts::LN_10);
        }
    }
}

fn mutate_add_connection(
    genome: &mut OrganismGenome,
    innovations: &mut InnovationRegistry,
    rng: &mut ChaCha8Rng,
    offspring_generation: u32,
    predation_enabled: bool,
) -> (bool, bool) {
    let pres: Vec<_> = SensoryReceptor::active(predation_enabled)
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| sensory_gene_node_id(id.0))
        .chain(genome.brain.hidden_nodes.iter().map(|node| node.id))
        .collect();
    let posts: Vec<_> = genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| node.id)
        .chain(ActionType::active(predation_enabled).map(|action| {
            let index = ActionType::ALL
                .iter()
                .position(|candidate| *candidate == action)
                .expect("active action belongs to ActionType::ALL");
            action_gene_node_id(index)
        }))
        .collect();
    let mut candidates = Vec::new();
    for &pre in &pres {
        for &post in &posts {
            let timing = SynapseTiming::CurrentTick;
            if brain::genome::connection_would_create_cycle(genome, pre, post, timing) {
                continue;
            }
            if genome.brain.edges.iter().any(|edge| {
                (edge.pre_node_id, edge.post_node_id, edge.timing) == (pre, post, timing)
                    && edge.enabled
            }) {
                continue;
            }
            candidates.push((pre, post, timing));
        }
    }
    for pre in genome.brain.hidden_nodes.iter().map(|node| node.id) {
        for post in genome.brain.hidden_nodes.iter().map(|node| node.id) {
            let timing = SynapseTiming::PreviousTick;
            if genome.brain.edges.iter().any(|edge| {
                (edge.pre_node_id, edge.post_node_id, edge.timing) == (pre, post, timing)
                    && edge.enabled
            }) {
                continue;
            }
            candidates.push((pre, post, timing));
        }
    }
    if candidates.is_empty() {
        return (false, false);
    }
    let (pre, post, timing) = candidates[rng.random_range(0..candidates.len())];
    if let Some(edge) = genome
        .brain
        .edges
        .iter_mut()
        .find(|edge| (edge.pre_node_id, edge.post_node_id, edge.timing) == (pre, post, timing))
    {
        edge.enabled = true;
        return (true, false);
    }
    let before = innovations.connection_history.len();
    let innovation = innovations.connection(
        pre,
        post,
        timing,
        Some(offspring_generation),
        InnovationKind::AddConnection,
    );
    genome.brain.edges.push(SynapseGene {
        innovation,
        pre_node_id: pre,
        post_node_id: post,
        timing,
        weight: random_weight(rng),
        enabled: true,
    });
    genome
        .brain
        .edges
        .sort_unstable_by_key(|edge| edge.innovation);
    (true, innovations.connection_history.len() > before)
}

fn mutate_add_node(
    genome: &mut OrganismGenome,
    innovations: &mut InnovationRegistry,
    rng: &mut ChaCha8Rng,
    offspring_generation: u32,
) -> (bool, bool) {
    if genome.brain.hidden_nodes.len() >= MAX_INTER_NEURONS as usize {
        return (false, false);
    }
    let enabled: Vec<_> = genome
        .brain
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.enabled)
        .map(|(index, _)| index)
        .collect();
    if enabled.is_empty() {
        return (false, false);
    }
    let edge_index = enabled[rng.random_range(0..enabled.len())];
    let old = genome.brain.edges[edge_index];
    let history_before = innovations.connection_history.len() + innovations.node_history.len();
    let split = innovations.split(
        old.innovation,
        old.pre_node_id,
        old.post_node_id,
        old.timing,
        offspring_generation,
    );
    if genome
        .brain
        .hidden_nodes
        .iter()
        .any(|node| node.id == split.node)
    {
        return (false, false);
    }
    genome.brain.edges[edge_index].enabled = false;
    genome.brain.hidden_nodes.push(HiddenNodeGene {
        id: split.node,
        bias: 0.0,
        log_time_constant: -1.203_972_8,
    });
    genome
        .brain
        .hidden_nodes
        .sort_unstable_by_key(|node| node.id);
    genome.brain.edges.push(SynapseGene {
        innovation: split.incoming,
        pre_node_id: old.pre_node_id,
        post_node_id: split.node,
        timing: split.incoming_timing,
        weight: 1.0,
        enabled: true,
    });
    genome.brain.edges.push(SynapseGene {
        innovation: split.outgoing,
        pre_node_id: split.node,
        post_node_id: old.post_node_id,
        timing: split.outgoing_timing,
        weight: old.weight,
        enabled: true,
    });
    genome
        .brain
        .edges
        .sort_unstable_by_key(|edge| edge.innovation);
    let history_after = innovations.connection_history.len() + innovations.node_history.len();
    (true, history_after > history_before)
}

#[allow(clippy::too_many_arguments)]
fn generation_summary(
    generation: u32,
    population: &[Individual],
    species: &[SpeciesRecord],
    winner_index: usize,
    offspring_crossovers: usize,
    offspring_clones: usize,
    complexification: ComplexificationSnapshot,
    compatibility_threshold: f64,
    training_seed_epoch: u32,
    effective_training_seeds: Vec<u64>,
    eval_opponents: usize,
    evaluation_cases_per_genome: usize,
    evaluation_worlds: usize,
    breeding_telemetry: BreedingTelemetry,
    crossplay_checkpoint_genome: Option<OrganismGenome>,
    persist_population_checkpoint: bool,
) -> GenerationSummary {
    let winner = &population[winner_index];
    let mut gross_energy_acquired_distribution = population
        .iter()
        .map(|individual| individual.evaluation.mean_gross_energy_acquired)
        .collect::<Vec<_>>();
    gross_energy_acquired_distribution.sort_by(f64::total_cmp);
    let net_energy_profit = |evaluation: &Evaluation| {
        evaluation.mean_attack_energy_received
            - evaluation.mean_attack_energy_lost
            - evaluation.mean_attack_attempt_energy_cost
    };
    let mut net_energy_profit_distribution = population
        .iter()
        .map(|individual| net_energy_profit(&individual.evaluation))
        .collect::<Vec<_>>();
    net_energy_profit_distribution.sort_by(f64::total_cmp);
    let mean_net_energy_profit =
        net_energy_profit_distribution.iter().sum::<f64>() / population.len() as f64;
    let median_net_energy_profit = if net_energy_profit_distribution.len().is_multiple_of(2) {
        let high = net_energy_profit_distribution.len() / 2;
        (net_energy_profit_distribution[high - 1] + net_energy_profit_distribution[high]) / 2.0
    } else {
        net_energy_profit_distribution[net_energy_profit_distribution.len() / 2]
    };
    let attack_attempts = |evaluation: &Evaluation| {
        evaluation.mean_attack_no_organism_targets
            + evaluation.mean_attack_target_evaded
            + evaluation.mean_attack_same_pool_blocked
            + evaluation.mean_attack_insufficient_energy
            + evaluation.mean_attack_eligible_attempts
    };
    let population_attack_hits = population
        .iter()
        .map(|individual| individual.evaluation.mean_attack_hits)
        .sum::<f64>();
    let population_attack_attempts = population
        .iter()
        .map(|individual| attack_attempts(&individual.evaluation))
        .sum::<f64>();
    let mean_gross_energy_acquired =
        gross_energy_acquired_distribution.iter().sum::<f64>() / population.len() as f64;
    let median_gross_energy_acquired = if gross_energy_acquired_distribution.len().is_multiple_of(2)
    {
        let high = gross_energy_acquired_distribution.len() / 2;
        (gross_energy_acquired_distribution[high - 1] + gross_energy_acquired_distribution[high])
            / 2.0
    } else {
        gross_energy_acquired_distribution[gross_energy_acquired_distribution.len() / 2]
    };
    let mut mean_action_fractions = [0.0; 5];
    for individual in population {
        for (mean, value) in mean_action_fractions
            .iter_mut()
            .zip(individual.evaluation.mean_action_fractions)
        {
            *mean += value / population.len() as f64;
        }
    }
    let opponent_stddevs = population
        .iter()
        .filter_map(|individual| individual.opponent_scores.mean_score_stddev)
        .collect::<Vec<_>>();
    let population_checkpoint = if persist_population_checkpoint {
        population
            .iter()
            .enumerate()
            .map(|(population_index, individual)| PopulationMemberResult {
                generation,
                population_index,
                reproduction: individual.reproduction,
                parents: individual.parents.clone(),
                contextual_score: individual.fitness,
                evaluation: individual.evaluation,
                opponent_scores: individual.opponent_scores.clone(),
                genome: individual.genome.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let species = species
        .iter()
        .map(|entry| SpeciesSummary {
            id: entry.id,
            size: entry.members.len(),
        })
        .collect();
    GenerationSummary {
        generation,
        training_seed_epoch,
        effective_training_seeds,
        eval_opponents,
        evaluation_cases_per_genome,
        evaluation_worlds,
        winner_contextual_score: winner.fitness,
        winner_case_score_stddev: winner.evaluation.case_score_stddev,
        mean_case_score_stddev: population
            .iter()
            .map(|individual| individual.evaluation.case_score_stddev)
            .sum::<f64>()
            / population.len() as f64,
        max_case_score_stddev: population
            .iter()
            .map(|individual| individual.evaluation.case_score_stddev)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        winner_absolute_survival_fraction: winner.evaluation.mean_absolute_survival_fraction,
        winner_candidate_alive_ticks: winner.evaluation.mean_candidate_alive_ticks,
        winner_late_weighted_survival_fraction: winner
            .evaluation
            .mean_late_weighted_survival_fraction,
        winner_relative_survival_advantage: winner.evaluation.mean_relative_survival_advantage,
        mean_absolute_survival_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_absolute_survival_fraction)
            .sum::<f64>()
            / population.len() as f64,
        mean_candidate_alive_ticks: population
            .iter()
            .map(|individual| individual.evaluation.mean_candidate_alive_ticks)
            .sum::<f64>()
            / population.len() as f64,
        mean_late_weighted_survival_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_late_weighted_survival_fraction)
            .sum::<f64>()
            / population.len() as f64,
        mean_relative_survival_advantage: population
            .iter()
            .map(|individual| individual.evaluation.mean_relative_survival_advantage)
            .sum::<f64>()
            / population.len() as f64,
        winner_action_effectiveness: winner.evaluation.mean_action_effectiveness,
        winner_successful_attack_rate: winner.evaluation.mean_successful_attack_rate,
        winner_mean_attack_kills: winner.evaluation.mean_attack_kills,
        mean_action_effectiveness: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.mean_action_effectiveness),
        ),
        mean_successful_attack_rate: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.mean_successful_attack_rate),
        ),
        winner_gross_energy_acquired: winner.evaluation.mean_gross_energy_acquired,
        mean_gross_energy_acquired,
        median_gross_energy_acquired,
        gross_energy_acquired_distribution,
        winner_attack_energy_received: winner.evaluation.mean_attack_energy_received,
        mean_attack_energy_received: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_energy_received)
            .sum::<f64>()
            / population.len() as f64,
        winner_attack_energy_lost: winner.evaluation.mean_attack_energy_lost,
        mean_attack_energy_lost: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_energy_lost)
            .sum::<f64>()
            / population.len() as f64,
        winner_attack_attempt_energy_cost: winner.evaluation.mean_attack_attempt_energy_cost,
        mean_attack_attempt_energy_cost: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_attempt_energy_cost)
            .sum::<f64>()
            / population.len() as f64,
        winner_net_energy_profit: net_energy_profit(&winner.evaluation),
        mean_net_energy_profit,
        median_net_energy_profit,
        winner_attack_precision: {
            let attempts = attack_attempts(&winner.evaluation);
            (attempts > 0.0).then(|| winner.evaluation.mean_attack_hits / attempts)
        },
        population_attack_precision: (population_attack_attempts > 0.0)
            .then(|| population_attack_hits / population_attack_attempts),
        winner_net_attack_energy_balance: winner.evaluation.mean_net_attack_energy_balance,
        mean_net_attack_energy_balance: population
            .iter()
            .map(|individual| individual.evaluation.mean_net_attack_energy_balance)
            .sum::<f64>()
            / population.len() as f64,
        winner_distinct_attack_victims: winner.evaluation.mean_distinct_attack_victims,
        mean_distinct_attack_victims: population
            .iter()
            .map(|individual| individual.evaluation.mean_distinct_attack_victims)
            .sum::<f64>()
            / population.len() as f64,
        winner_attack_target_evaded: winner.evaluation.mean_attack_target_evaded,
        mean_attack_target_evaded: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_target_evaded)
            .sum::<f64>()
            / population.len() as f64,
        winner_attack_repeat_hit_fraction: winner.evaluation.attack_repeat_hit_fraction,
        mean_attack_repeat_hit_fraction: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.attack_repeat_hit_fraction),
        ),
        winner_action_fractions: winner.evaluation.mean_action_fractions,
        mean_action_fractions,
        winner_commands_per_tick: winner.evaluation.mean_commands_per_tick,
        mean_commands_per_tick: population
            .iter()
            .map(|individual| individual.evaluation.mean_commands_per_tick)
            .sum::<f64>()
            / population.len() as f64,
        winner_multi_command_tick_fraction: winner.evaluation.mean_multi_command_tick_fraction,
        mean_multi_command_tick_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_multi_command_tick_fraction)
            .sum::<f64>()
            / population.len() as f64,
        winner_spatial_coverage: winner.evaluation.mean_spatial_coverage,
        mean_spatial_coverage: population
            .iter()
            .map(|individual| individual.evaluation.mean_spatial_coverage)
            .sum::<f64>()
            / population.len() as f64,
        winner_end_survival_fraction: winner.evaluation.mean_candidate_end_survival_fraction,
        mean_end_survival_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_candidate_end_survival_fraction)
            .sum::<f64>()
            / population.len() as f64,
        mean_opponent_score_stddev: (!opponent_stddevs.is_empty())
            .then(|| opponent_stddevs.iter().sum::<f64>() / opponent_stddevs.len() as f64),
        max_opponent_score_stddev: opponent_stddevs.into_iter().max_by(f64::total_cmp),
        crossplay_checkpoint_genome,
        population_checkpoint,
        compatibility_threshold,
        winner_hidden_nodes: winner.genome.hidden_node_count(),
        winner_enabled_connections: winner.genome.enabled_connection_count(),
        winner_encoded_connections: winner.genome.encoded_connection_count(),
        winner_expressed_hidden_nodes: complexification.winner_expressed_hidden_nodes,
        winner_expressed_connections: complexification.winner_expressed_connections,
        mean_expressed_hidden_nodes: complexification.mean_expressed_hidden_nodes,
        mean_expressed_connections: complexification.mean_expressed_connections,
        new_connection_innovations: complexification.new_connection_innovations,
        new_node_innovations: complexification.new_node_innovations,
        new_expressed_connection_innovations: complexification.new_expressed_connection_innovations,
        expressed_connection_innovations: complexification.expressed_connection_innovations,
        connection_innovations_reaching_ten_percent: complexification
            .connection_innovations_reaching_ten_percent,
        connection_innovations_reaching_majority: complexification
            .connection_innovations_reaching_majority,
        non_elite_offspring: breeding_telemetry.non_elite_offspring,
        structural_mutation_attempts: breeding_telemetry.structural_mutation_attempts,
        structural_mutation_successes: breeding_telemetry.structural_mutation_successes,
        registry_new_structural_mutations: breeding_telemetry.registry_new_structural_mutations,
        new_origin_offspring: breeding_telemetry.new_origin_offspring,
        new_origin_offspring_rate: (breeding_telemetry.non_elite_offspring > 0).then(|| {
            breeding_telemetry.new_origin_offspring as f64
                / breeding_telemetry.non_elite_offspring as f64
        }),
        species,
        offspring_crossovers,
        offspring_clones,
    }
}

fn best_selection_index(population: &[Individual]) -> usize {
    population
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.selection_score
                .total_cmp(&right.selection_score)
                .then_with(|| left.fitness.total_cmp(&right.fitness))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("NEAT population is non-empty")
}

fn blank_individual(
    genome: OrganismGenome,
    reproduction: ReproductionKind,
    parents: Vec<ParentReference>,
) -> Individual {
    Individual {
        genome,
        evaluation: Evaluation::default(),
        opponent_scores: OpponentScoreProfile::default(),
        reproduction,
        parents,
        fitness: 0.0,
        selection_score: 0.0,
    }
}

fn randomize_parameters(genome: &mut OrganismGenome, rng: &mut ChaCha8Rng) {
    for edge in &mut genome.brain.edges {
        edge.weight = random_weight(rng);
    }
    for bias in &mut genome.brain.action_biases {
        *bias = rng.random_range(-0.25..0.25);
    }
}

fn random_weight(rng: &mut ChaCha8Rng) -> f32 {
    constrain_weight(rng.random_range(-WEIGHT_MAX_ABS..WEIGHT_MAX_ABS))
}

fn constrain_weight(weight: f32) -> f32 {
    if weight.abs() < WEIGHT_MIN_ABS {
        return if weight.is_sign_negative() {
            -WEIGHT_MIN_ABS
        } else {
            WEIGHT_MIN_ABS
        };
    }
    weight.clamp(-WEIGHT_MAX_ABS, WEIGHT_MAX_ABS)
}

fn normal(rng: &mut ChaCha8Rng) -> f32 {
    StandardNormal.sample(rng)
}

fn event_rng(run_seed: u64, generation: u32, slot: usize, domain: u64) -> ChaCha8Rng {
    let generation = u64::from(generation).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let slot = (slot as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    ChaCha8Rng::seed_from_u64(mix64(run_seed ^ domain ^ generation ^ slot))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena_genomes(count: usize) -> Vec<OrganismGenome> {
        let mut world = WorldConfig::test_fixture();
        world.num_organisms = count as u32;
        world.predation_enabled = true;
        let template = Simulation::new(world, 1)
            .expect("test world")
            .organisms()
            .first()
            .expect("founder")
            .genome
            .clone();
        (0..count)
            .map(|index| {
                let mut genome = template.clone();
                genome.brain.action_biases[0] = index as f32 * 0.01;
                genome
            })
            .collect()
    }

    fn ordered_genome_bytes(genomes: &[OrganismGenome]) -> Vec<Vec<u8>> {
        canonical_population_order(genomes)
            .expect("canonical order")
            .into_iter()
            .map(|index| bincode::serialize(&genomes[index]).expect("serialize genome"))
            .collect()
    }

    #[test]
    fn arena_canonical_order_is_population_permutation_invariant() {
        let genomes = arena_genomes(8);
        let permuted = [5, 1, 7, 0, 3, 6, 2, 4]
            .into_iter()
            .map(|index| genomes[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_genome_bytes(&genomes),
            ordered_genome_bytes(&permuted)
        );
    }

    #[test]
    fn one_case_per_genome_balances_every_founder_slot() {
        let population_size = 64;
        let canonical = (0..population_size).collect::<Vec<_>>();
        let mut slots_by_genome = vec![BTreeSet::new(); population_size];
        for case_index in 0..population_size {
            for (slot, genome) in rotated_arena_order(&canonical, 7, 13, case_index)
                .into_iter()
                .enumerate()
            {
                slots_by_genome[genome].insert(slot);
            }
        }
        assert!(slots_by_genome
            .iter()
            .all(|slots| slots.len() == population_size));
    }

    #[test]
    fn shared_arena_scores_are_population_permutation_invariant() {
        let genomes = arena_genomes(4);
        let mut left = genomes
            .iter()
            .cloned()
            .map(|genome| blank_individual(genome, ReproductionKind::Initial, Vec::new()))
            .collect::<Vec<_>>();
        let permutation = [2, 0, 3, 1];
        let mut right = permutation
            .into_iter()
            .map(|index| {
                blank_individual(
                    genomes[index].clone(),
                    ReproductionKind::Initial,
                    Vec::new(),
                )
            })
            .collect::<Vec<_>>();
        let mut world = WorldConfig::test_fixture();
        world.world_width = 12;
        world.num_organisms = 4;
        world.predation_enabled = true;
        world.runtime_plasticity_enabled = false;
        let scenarios = vec![ScenarioManifest {
            name: "energy_stealing".to_owned(),
            world,
        }];
        let config = NeatConfig {
            population_size: 4,
            eval_opponents: 3,
            episode_horizons: vec![20],
            world_seeds: vec![11, 12, 13, 14],
            evaluator_workers: 1,
            ..NeatConfig::default()
        };
        evaluate_population_shared(&mut left, &scenarios, &config.world_seeds, &config, 7, 0)
            .expect("canonical arena evaluation");
        evaluate_population_shared(&mut right, &scenarios, &config.world_seeds, &config, 7, 0)
            .expect("permuted arena evaluation");

        let scores = |population: &[Individual]| {
            population
                .iter()
                .map(|individual| {
                    (
                        bincode::serialize(&individual.genome).expect("serialize genome"),
                        individual.fitness.to_bits(),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(scores(&left), scores(&right));
    }
}
