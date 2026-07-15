//! Canonical, generational NEAT: the sole evolutionary system for NeuroGenesis.
//!
//! This module owns species and innovation history. The simulation
//! (`world_sim::Simulation`) is a deterministic fitness evaluator; NEAT is the outer
//! loop. Candidates are evaluated either as isolated clonal colonies or in
//! mixed-founder worlds under fixed world seeds. In-world reproduction does not
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
    action_gene_node_id, action_gene_node_index, is_hidden_gene_node_id, sensory_gene_node_id,
    sensory_gene_node_index, split_hidden_gene_node_id, ActionType, GeneNodeId, HiddenNodeGene,
    InnovationId, NeuronId, OrganismGenome, OrganismId, SensoryReceptor, SynapseGene,
};
use world_sim::{AttackOutcome, Simulation};

const WEIGHT_MIN_ABS: f32 = 0.001;
const WEIGHT_MAX_ABS: f32 = 1.5;
const BIAS_MAX_ABS: f32 = 1.0;
const BREED_SELECTION_DOMAIN: u64 = 0x4252_4545_445f_5345;
const CROSSOVER_DOMAIN: u64 = 0x4352_4f53_534f_5645;
const MUTATION_DOMAIN: u64 = 0x4d55_5441_5449_4f4e;
const OPPONENT_DOMAIN: u64 = 0x4f50_504f_4e45_4e54;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatConfig {
    pub population_size: usize,
    pub generations: u32,
    pub episode_horizons: Vec<u64>,
    /// Positive per-window weights applied across each episode from early to
    /// late ticks. `[1]` is uniform survival; `[1,2,4,8]` values later survival
    /// more while preserving a positive gradient for every alive tick.
    pub survival_window_weights: Vec<f64>,
    /// Competitive evaluation: total contemporary-opponent exposures per
    /// candidate. Exposures are grouped into worlds containing
    /// `eval_lineages_per_world` lineages and every lineage is scored from the
    /// same simulation. `0` restores isolated clonal-colony evaluation.
    pub eval_opponents: usize,
    /// Number of contemporary lineages sharing each competitive evaluator
    /// world. The canonical baseline is two; three enables a mini-ecosystem
    /// treatment with one focal lineage and two simultaneous opponents.
    pub eval_lineages_per_world: usize,
    /// Diagnostic oracle: in evaluator worlds, attacks only affect organisms
    /// from other founder-pool entries. This removes friendly fire to test
    /// whether identity ambiguity is the binding predation constraint.
    pub cross_pool_predation_only: bool,
    pub world_seeds: Vec<u64>,
    /// `0` freezes training layouts. Otherwise a fresh deterministic seed
    /// suite is derived every N generations.
    pub training_seed_rotation_period: u32,
    pub scenarios: Vec<ScenarioPreset>,
    /// Select on the mean of the worst-performing fraction of scenario/seed
    /// cases. `1.0` is the ordinary mean; `0.25` is lower-quartile CVaR.
    pub objective_cvar_fraction: f64,
    pub fitness_objective: FitnessObjective,
    pub selection_strategy: SelectionStrategy,
    pub novelty_k: usize,
    pub novelty_archive_additions_per_generation: usize,
    pub curriculum_enabled: bool,
    pub curriculum_promotion_threshold: f64,
    pub curriculum_promotion_patience: u32,
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
    pub stagnation_generations: u32,
    pub young_species_grace_generations: u32,
    pub min_young_species_offspring: usize,
    pub elitism_min_species_size: usize,
}

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 20,
            episode_horizons: vec![500],
            survival_window_weights: vec![1.0],
            eval_opponents: 8,
            eval_lineages_per_world: 2,
            cross_pool_predation_only: false,
            world_seeds: vec![11, 29, 47],
            training_seed_rotation_period: 0,
            scenarios: ScenarioPreset::ALL.to_vec(),
            objective_cvar_fraction: 1.0,
            fitness_objective: FitnessObjective::SurvivalTimesRelativeAdvantage,
            selection_strategy: SelectionStrategy::Fitness,
            novelty_k: 15,
            novelty_archive_additions_per_generation: 2,
            curriculum_enabled: false,
            curriculum_promotion_threshold: 0.35,
            curriculum_promotion_patience: 3,
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
            stagnation_generations: 15,
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
        if !(2..=3).contains(&self.eval_lineages_per_world) {
            bail!("eval_lineages_per_world must be 2 or 3");
        }
        if self.eval_opponents > 0 {
            let opponents_per_world = self.eval_lineages_per_world - 1;
            if !self.eval_opponents.is_multiple_of(opponents_per_world) {
                bail!(
                    "eval_opponents must be divisible by {} for {}-lineage evaluation",
                    opponents_per_world,
                    self.eval_lineages_per_world
                );
            }
            let memberships_per_genome = self.eval_opponents / opponents_per_world;
            if self.eval_lineages_per_world == 2 && self.eval_opponents >= self.population_size {
                bail!("pairwise eval_opponents must be smaller than population_size");
            }
            if !self
                .population_size
                .is_multiple_of(self.eval_lineages_per_world)
            {
                bail!(
                    "population_size must be divisible by eval_lineages_per_world for balanced evaluation"
                );
            }
            if !self
                .population_size
                .saturating_mul(memberships_per_genome)
                .is_multiple_of(self.eval_lineages_per_world)
            {
                bail!("competitive schedule cannot balance the requested world memberships");
            }
        }
        if self.eval_opponents > 0
            && self.eval_lineages_per_world == 2
            && !self.world_seeds.len().is_multiple_of(4)
        {
            bail!("balanced pairwise evaluation needs a multiple of four training world seeds");
        }
        if self.world_seeds.is_empty() || self.evaluator_workers == 0 {
            bail!("NEAT needs at least one world seed and evaluator worker");
        }
        if self.scenarios.is_empty() {
            bail!("NEAT needs at least one evaluation scenario");
        }
        if !(0.0..=1.0).contains(&self.objective_cvar_fraction)
            || self.objective_cvar_fraction == 0.0
        {
            bail!("objective_cvar_fraction must be in (0,1]");
        }
        if self.fitness_objective == FitnessObjective::SurvivalTimesRelativeAdvantage
            && self.eval_opponents == 0
        {
            bail!("survival_times_relative_advantage requires eval_opponents >= 1");
        }
        if self.curriculum_promotion_threshold < 0.0 || self.curriculum_promotion_patience == 0 {
            bail!("curriculum threshold must be nonnegative and patience must be >= 1");
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
        if self.novelty_k == 0 || self.novelty_archive_additions_per_generation == 0 {
            bail!("novelty_k and novelty archive additions must be >= 1");
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FitnessObjective {
    SurvivalFraction,
    LateWeightedSurvival,
    SurvivalTimesRelativeAdvantage,
}

impl FitnessObjective {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "survival_fraction" | "survival" => Ok(Self::SurvivalFraction),
            "late_weighted_survival" | "weighted_survival" => Ok(Self::LateWeightedSurvival),
            "survival_times_relative_advantage" | "sustainable_competition" => {
                Ok(Self::SurvivalTimesRelativeAdvantage)
            }
            other => bail!(
                "unknown fitness objective `{other}`; valid: survival_fraction late_weighted_survival survival_times_relative_advantage"
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::SurvivalFraction => "lower_tail_mean_survival_fraction",
            Self::LateWeightedSurvival => "lower_tail_mean_late_weighted_survival",
            Self::SurvivalTimesRelativeAdvantage => {
                "lower_tail_mean_survival_times_relative_advantage"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrophicRole {
    #[default]
    Nonconsumer,
    Forager,
    Scavenger,
    Predator,
    Omnivore,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TrophicRoleCounts {
    pub nonconsumer: usize,
    pub forager: usize,
    pub scavenger: usize,
    pub predator: usize,
    pub omnivore: usize,
}

impl TrophicRoleCounts {
    fn observe(&mut self, role: TrophicRole) {
        match role {
            TrophicRole::Nonconsumer => self.nonconsumer += 1,
            TrophicRole::Forager => self.forager += 1,
            TrophicRole::Scavenger => self.scavenger += 1,
            TrophicRole::Predator => self.predator += 1,
            TrophicRole::Omnivore => self.omnivore += 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Evaluation {
    /// The only selection objective in this baseline.
    pub mean_objective_score: f64,
    /// Ordinary mean across cases, retained beside the robust lower-tail
    /// selection objective as a diagnostic.
    pub mean_case_score: f64,
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
    /// indices represented across them. In the default pairwise evaluator this
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
    pub mean_plant_consumption_rate: Option<f64>,
    pub mean_prey_consumption_rate: Option<f64>,
    pub mean_mi_sa: Option<f64>,
    pub mean_learning_slope: Option<f64>,
    /// Coarse trophic description backed by the intake and attack channels
    /// below; use the continuous rates for comparisons.
    pub trophic_role: TrophicRole,
    pub plant_intake_fraction: Option<f64>,
    pub prey_intake_fraction: Option<f64>,
    /// Candidate-lineage energy-flow diagnostics only; these never contribute
    /// to fitness. Gross acquired energy is plant energy plus attack-transfer
    /// credits. Starting energy is excluded, and attack transfers are counted
    /// exactly once rather than also being valued as prey consumptions.
    pub mean_gross_energy_acquired: f64,
    pub mean_plant_energy_acquired: f64,
    pub mean_attack_energy_received: f64,
    pub mean_attack_energy_lost: f64,
    pub mean_attack_attempt_energy_cost: f64,
    pub mean_net_attack_energy_balance: f64,
    pub mean_consumptions: f64,
    pub mean_plant_consumptions: f64,
    pub mean_prey_consumptions: f64,
    pub mean_attack_no_organism_targets: f64,
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
    /// Fraction of all plant spawn events in the scored window that were
    /// consumed. Reported with standing-plant pressure because regrowth supply
    /// itself depends on successful harvests.
    pub mean_plant_capture_fraction: Option<f64>,
    pub mean_plant_consumptions_per_tick: f64,
    pub mean_realized_plant_supply_per_tick: f64,
    /// Mean fraction of food tiles occupied by a standing plant after each
    /// scored tick. Values near zero together with high sustained harvest rate
    /// indicate pressure against the ecological supply ceiling.
    pub mean_standing_plant_fraction: f64,
    pub mean_spatial_coverage: f64,
    /// First plant-consumption tick divided by episode length; no-consumption
    /// cases are right-censored to one tick beyond the episode.
    pub mean_normalized_time_to_first_plant: f64,
    /// Organism-tick action distribution in ActionType declaration order,
    /// including Idle. This is an observational behavior descriptor only.
    pub mean_action_fractions: [f64; 6],
    pub mean_world_final_population: f64,
}

#[derive(Debug, Clone, Default)]
struct LineageCaseDiagnostics {
    consumptions: u64,
    plant_consumptions: u64,
    plant_energy_acquired: f64,
    attack_no_organism_targets: u64,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseEvaluation {
    pub scenario: String,
    pub curriculum_level: u32,
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
    pub fixed_anchor_index: Option<usize>,
    pub founder_pool_size: usize,
    pub focal_pool_index: usize,
    pub world_founders: usize,
    pub candidate_founders: u64,
    pub candidate_end_survivors: u64,
    pub action_effectiveness: Option<f64>,
    pub plant_consumption_rate: Option<f64>,
    pub prey_consumption_rate: Option<f64>,
    pub mi_sa: Option<f64>,
    pub learning_slope: Option<f64>,
    /// Plant energy plus attack-transfer credits acquired by the focal lineage.
    /// Starting energy is excluded and each direct transfer is counted once.
    pub gross_energy_acquired: f64,
    pub plant_energy_acquired: f64,
    pub attack_energy_received: f64,
    pub attack_energy_lost: f64,
    pub attack_attempt_energy_cost: f64,
    pub net_attack_energy_balance: f64,
    pub consumptions: u64,
    pub plant_consumptions: u64,
    pub prey_consumptions: u64,
    pub attack_no_organism_targets: u64,
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
    pub plant_supply_events: u64,
    /// Plant instances that existed early enough to be consumed during this
    /// scoring window. Final-tick spawns are excluded because food regrowth is
    /// committed after interactions.
    pub actionable_plant_supply: u64,
    pub final_tick_plant_spawns: u64,
    pub final_standing_plants: u64,
    pub plant_capture_fraction: Option<f64>,
    pub plant_consumptions_per_tick: f64,
    pub realized_plant_supply_per_tick: f64,
    pub mean_standing_plant_fraction: f64,
    pub time_to_first_plant: Option<u64>,
    pub normalized_time_to_first_plant: f64,
    pub spatial_coverage: f64,
    pub action_fractions: [f64; 6],
    pub world_final_population: usize,
    /// Observation-only facts for every lineage in the shared world. They are
    /// retained in memory just long enough to construct accurate mirrored
    /// `CaseEvaluation`s and are not part of the result schema.
    #[serde(skip)]
    lineage_diagnostics_by_pool: Vec<LineageCaseDiagnostics>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    Fitness,
    NoveltyLocalCompetition,
}

impl SelectionStrategy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fitness" => Ok(Self::Fitness),
            "novelty_local_competition" | "novelty-local-competition" | "nslc" => {
                Ok(Self::NoveltyLocalCompetition)
            }
            other => bail!(
                "unknown selection strategy `{other}`; valid: fitness novelty_local_competition"
            ),
        }
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioPreset {
    Baseline,
    Scarcity,
    SparseSearch,
}

impl ScenarioPreset {
    pub const ALL: [Self; 3] = [Self::Baseline, Self::Scarcity, Self::SparseSearch];

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "scarcity" => Ok(Self::Scarcity),
            "sparse_search" | "sparse-search" => Ok(Self::SparseSearch),
            other => {
                bail!("unknown scenario `{other}`; valid: baseline scarcity sparse_search")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Scarcity => "scarcity",
            Self::SparseSearch => "sparse_search",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub name: String,
    pub curriculum_level: u32,
    pub world: WorldConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub id: u64,
    pub size: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub stagnant_generations: u32,
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

/// Parent population coordinates are sufficient because every evaluated
/// generation is persisted in full.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParentReference {
    pub generation: u32,
    pub population_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSummary {
    pub generation: u32,
    pub curriculum_level: u32,
    pub training_seed_epoch: u32,
    pub effective_training_seeds: Vec<u64>,
    pub eval_opponents: usize,
    pub evaluation_cases_per_genome: usize,
    /// Actual pairwise worlds simulated. Each world scores both contemporary
    /// lineages.
    pub evaluation_worlds: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub median_fitness: f64,
    pub best_absolute_survival_fraction: f64,
    pub best_candidate_alive_ticks: f64,
    pub best_late_weighted_survival_fraction: f64,
    pub best_relative_survival_advantage: f64,
    pub mean_absolute_survival_fraction: f64,
    pub mean_candidate_alive_ticks: f64,
    pub mean_late_weighted_survival_fraction: f64,
    pub mean_relative_survival_advantage: f64,
    pub best_mean_prey_consumptions: f64,
    pub best_trophic_role: TrophicRole,
    pub best_action_effectiveness: Option<f64>,
    pub best_plant_consumption_rate: Option<f64>,
    pub best_prey_consumption_rate: Option<f64>,
    pub best_mi_sa: Option<f64>,
    pub best_learning_slope: Option<f64>,
    pub best_plant_intake_fraction: Option<f64>,
    pub best_prey_intake_fraction: Option<f64>,
    pub best_mean_attack_kills: f64,
    pub mean_action_effectiveness: Option<f64>,
    pub mean_plant_consumption_rate: Option<f64>,
    pub mean_prey_consumption_rate: Option<f64>,
    pub population_trophic_roles: TrophicRoleCounts,
    /// Exact population distribution of gross acquired energy (plant energy +
    /// attack-transfer credits, excluding starting energy), plus summaries.
    pub best_gross_energy_acquired: f64,
    pub mean_gross_energy_acquired: f64,
    pub median_gross_energy_acquired: f64,
    pub gross_energy_acquired_distribution: Vec<f64>,
    pub champion_plant_energy_acquired: f64,
    pub mean_plant_energy_acquired: f64,
    pub champion_attack_energy_received: f64,
    pub mean_attack_energy_received: f64,
    pub champion_attack_energy_lost: f64,
    pub mean_attack_energy_lost: f64,
    pub champion_attack_attempt_energy_cost: f64,
    pub mean_attack_attempt_energy_cost: f64,
    pub champion_net_attack_energy_balance: f64,
    pub mean_net_attack_energy_balance: f64,
    pub champion_distinct_attack_victims: f64,
    pub mean_distinct_attack_victims: f64,
    pub champion_attack_repeat_hit_fraction: Option<f64>,
    pub mean_attack_repeat_hit_fraction: Option<f64>,
    pub champion_action_fractions: [f64; 6],
    pub mean_action_fractions: [f64; 6],
    pub champion_realized_plant_supply_per_tick: f64,
    pub mean_realized_plant_supply_per_tick: f64,
    pub champion_plant_capture_fraction: Option<f64>,
    pub mean_plant_capture_fraction: Option<f64>,
    pub champion_standing_plant_fraction: f64,
    pub mean_standing_plant_fraction: f64,
    pub champion_spatial_coverage: f64,
    pub mean_spatial_coverage: f64,
    pub best_end_survival_fraction: f64,
    pub mean_end_survival_fraction: f64,
    /// Across-member summary of opponent-context sensitivity. Inspect each
    /// member's profile in `population_checkpoint` for opponent identities and
    /// exact opponent means.
    pub mean_opponent_score_stddev: Option<f64>,
    pub max_opponent_score_stddev: Option<f64>,
    /// Generation champion persisted for historical cross-play.
    pub checkpoint_champion_genome: Option<OrganismGenome>,
    /// Complete evaluated population, making `(generation, population_index)`
    /// a durable lineage coordinate and retaining all diagnostic distributions.
    pub population_checkpoint: Vec<PopulationMemberResult>,
    pub selection_strategy: SelectionStrategy,
    pub best_novelty: Option<f64>,
    pub mean_novelty: Option<f64>,
    pub best_local_competition: Option<f64>,
    pub mean_local_competition: Option<f64>,
    pub novelty_archive_size: usize,
    pub compatibility_threshold: f64,
    pub best_hidden_nodes: usize,
    pub best_enabled_connections: usize,
    pub best_encoded_connections: usize,
    /// Enabled structure on at least one directed sensory-to-action path.
    pub best_expressed_hidden_nodes: usize,
    pub best_expressed_connections: usize,
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
    pub fitness: f64,
    pub evaluation: Evaluation,
    pub opponent_scores: OpponentScoreProfile,
    pub genome: OrganismGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenOuterLoopContract {
    pub fully_connected_initial_topology: bool,
    pub feed_forward_hidden_graph: bool,
    pub balanced_pairwise_evaluation: bool,
    pub symmetric_founder_slot_rotation: bool,
    pub eval_lineages_per_world: usize,
    pub runtime_plasticity_enabled: bool,
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
    pub force_random_actions: bool,
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
    pub food_energy: u32,
    pub replay_anchor_scenarios: Vec<ScenarioManifest>,
    pub final_training_scenarios: Vec<ScenarioManifest>,
    pub generations: Vec<GenerationSummary>,
    pub final_population: Vec<PopulationMemberResult>,
    pub champion_fitness: f64,
    pub champion_evaluation: Evaluation,
    pub champion_generation: u32,
    pub champion_curriculum_level: u32,
    pub champion_training_seed_epoch: u32,
    pub champion_genome: OrganismGenome,
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
    /// `None` identifies the shared minimal starting topology.
    pub origin_generation: Option<u32>,
    pub kind: InnovationKind,
    pub first_expressed_generation: Option<u32>,
    pub first_ten_percent_generation: Option<u32>,
    pub first_majority_generation: Option<u32>,
    pub last_present_generation: Option<u32>,
    pub max_encoded_frequency: f64,
    pub max_expressed_frequency: f64,
    /// Largest same-generation mean-fitness advantage of carriers over
    /// non-carriers while both groups existed. Descriptive, not causal.
    pub max_carrier_fitness_advantage: Option<f64>,
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
    novelty: f64,
    local_competition: f64,
    pareto_rank: usize,
}

#[derive(Clone)]
struct SpeciesRecord {
    id: u64,
    representative: OrganismGenome,
    members: Vec<usize>,
    best_fitness: f64,
    stagnant_generations: u32,
    created_generation: u32,
}

struct ActiveStructure {
    hidden_nodes: BTreeSet<GeneNodeId>,
    connections: BTreeSet<InnovationId>,
}

const BEHAVIOR_DESCRIPTOR_DIMENSIONS: usize = 8;

#[derive(Clone, Copy)]
struct BehaviorDescriptor([f64; BEHAVIOR_DESCRIPTOR_DIMENSIONS]);

impl BehaviorDescriptor {
    fn from_evaluation(evaluation: Evaluation) -> Self {
        let mut values = [0.0; BEHAVIOR_DESCRIPTOR_DIMENSIONS];
        values[..6].copy_from_slice(&evaluation.mean_action_fractions);
        values[6] = evaluation.mean_spatial_coverage.clamp(0.0, 1.0);
        values[7] = evaluation
            .mean_normalized_time_to_first_plant
            .clamp(0.0, 1.0);
        Self(values)
    }

    fn distance(self, other: Self) -> f64 {
        (self
            .0
            .iter()
            .zip(other.0)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            / BEHAVIOR_DESCRIPTOR_DIMENSIONS as f64)
            .sqrt()
    }
}

struct ComplexificationSnapshot {
    best_expressed_hidden_nodes: usize,
    best_expressed_connections: usize,
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
}

/// Run-owned historical markings. A structural event receives one monotonic
/// innovation number and every lineage encountering that event reuses it.
#[derive(Default)]
struct InnovationRegistry {
    next: u64,
    connections: HashMap<(GeneNodeId, GeneNodeId), InnovationId>,
    splits: HashMap<InnovationId, SplitRecord>,
    connection_history: BTreeMap<InnovationId, ConnectionInnovationRecord>,
    node_history: BTreeMap<GeneNodeId, NodeInnovationRecord>,
}

impl InnovationRegistry {
    fn connection(
        &mut self,
        pre: GeneNodeId,
        post: GeneNodeId,
        origin_generation: Option<u32>,
        kind: InnovationKind,
    ) -> InnovationId {
        if let Some(id) = self.connections.get(&(pre, post)) {
            return *id;
        }
        let id = InnovationId(self.next);
        self.next = self.next.saturating_add(1);
        self.connections.insert((pre, post), id);
        self.connection_history.insert(
            id,
            ConnectionInnovationRecord {
                innovation: id,
                pre_node_id: pre,
                post_node_id: post,
                origin_generation,
                kind,
                first_expressed_generation: None,
                first_ten_percent_generation: None,
                first_majority_generation: None,
                last_present_generation: None,
                max_encoded_frequency: 0.0,
                max_expressed_frequency: 0.0,
                max_carrier_fitness_advantage: None,
            },
        );
        id
    }

    fn split(
        &mut self,
        original: InnovationId,
        pre: GeneNodeId,
        post: GeneNodeId,
        origin_generation: u32,
    ) -> SplitRecord {
        if let Some(record) = self.splits.get(&original) {
            return *record;
        }
        let node = split_hidden_gene_node_id(original);
        let record = SplitRecord {
            node,
            incoming: self.connection(
                pre,
                node,
                Some(origin_generation),
                InnovationKind::SplitIncoming,
            ),
            outgoing: self.connection(
                node,
                post,
                Some(origin_generation),
                InnovationKind::SplitOutgoing,
            ),
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
    if world.force_random_actions {
        bail!(
            "NEAT requires genome-controlled actions; force_random_actions is reserved for explicit null-control simulations"
        );
    }
    configure_evaluation_world(&mut world);
    if config.eval_opponents > 0
        && !(world.num_organisms as usize).is_multiple_of(config.eval_lineages_per_world)
    {
        bail!(
            "competitive founder count {} must be divisible by {} lineages per world",
            world.num_organisms,
            config.eval_lineages_per_world
        );
    }
    if !world.leaky_neurons_enabled {
        config.mutate_time_constant_probability = 0.0;
    }
    let level_zero_scenarios = build_scenarios(&world, &config.scenarios, 0);
    let mut curriculum_level = 0u32;
    let mut curriculum_promotion_streak = 0u32;
    let mut training_scenarios = build_training_scenarios(
        &world,
        &config.scenarios,
        curriculum_level,
        &level_zero_scenarios,
    );
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
                innovation: innovations.connection(pre, post, None, InnovationKind::Initial),
                pre_node_id: pre,
                post_node_id: post,
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
        restrict_predation_genes(&mut genome, world.predation_enabled);
        population.push(Individual {
            genome,
            evaluation: Evaluation::default(),
            opponent_scores: OpponentScoreProfile::default(),
            reproduction: ReproductionKind::Initial,
            parents: Vec::new(),
            fitness: 0.0,
            selection_score: 0.0,
            novelty: 0.0,
            local_competition: 0.0,
            pareto_rank: 0,
        });
    }

    let frozen_outer_loop_contract = FrozenOuterLoopContract {
        fully_connected_initial_topology: true,
        feed_forward_hidden_graph: true,
        balanced_pairwise_evaluation: config.eval_opponents > 0
            && config.eval_lineages_per_world == 2,
        symmetric_founder_slot_rotation: config.eval_opponents > 0,
        eval_lineages_per_world: config.eval_lineages_per_world,
        runtime_plasticity_enabled: world.runtime_plasticity_enabled,
        leaky_neurons_enabled: world.leaky_neurons_enabled,
        predation_enabled: world.predation_enabled,
        force_random_actions: world.force_random_actions,
        cross_pool_predation_only: config.cross_pool_predation_only,
        intent_parallel_threads: world.intent_parallel_threads,
    };
    let world_width = world.world_width;
    let founder_cohort_size = world.num_organisms;
    let food_energy = world.food_energy;

    let mut previous_species: Vec<SpeciesRecord> = Vec::new();
    let mut next_species_id = 0u64;
    let mut compatibility_threshold = config.compatibility_threshold;
    let mut summaries = Vec::with_capacity(config.generations as usize);
    let mut champion: Option<(u32, u32, f64, Evaluation, u32, OrganismGenome)> = None;
    let mut incoming_breeding_telemetry = BreedingTelemetry::default();
    let mut novelty_archive = Vec::<BehaviorDescriptor>::new();

    for generation in 0..config.generations {
        let seed_epoch = training_seed_epoch(&config, generation);
        let effective_training_seeds =
            effective_training_seeds(&config.world_seeds, seed, seed_epoch);
        evaluate_population(
            &mut population,
            &training_scenarios,
            &effective_training_seeds,
            &config,
            seed,
            generation,
        )?;
        assign_selection_scores(&mut population, &novelty_archive, &config);
        let selection_best_index = best_selection_index(&population);
        let mut species = assign_species(
            &population,
            &previous_species,
            &config,
            compatibility_threshold,
            generation,
            &mut next_species_id,
        );
        let best_index = best_individual_index(&population);
        let best = &population[best_index];
        if champion.as_ref().is_none_or(
            |(level, champion_seed_epoch, fitness, evaluation, _, _)| {
                curriculum_level > *level
                    || (curriculum_level == *level
                        && (seed_epoch > *champion_seed_epoch
                            || (seed_epoch == *champion_seed_epoch
                                && (best.fitness > *fitness
                                    || (best.fitness == *fitness
                                        && best
                                            .evaluation
                                            .mean_plant_capture_fraction
                                            .unwrap_or(0.0)
                                            > evaluation
                                                .mean_plant_capture_fraction
                                                .unwrap_or(0.0))))))
            },
        ) {
            champion = Some((
                curriculum_level,
                seed_epoch,
                best.fitness,
                best.evaluation,
                generation,
                best.genome.clone(),
            ));
        }

        let checkpoint_champion_genome = Some(best.genome.clone());
        let cases_per_match = training_scenarios
            .len()
            .saturating_mul(effective_training_seeds.len())
            .saturating_mul(config.episode_horizons.len());
        let opponent_slots_per_world = config.eval_lineages_per_world.saturating_sub(1).max(1);
        let world_memberships_per_genome = config.eval_opponents / opponent_slots_per_world;
        let evaluation_cases_per_genome =
            world_memberships_per_genome.saturating_mul(cases_per_match);
        let evaluation_worlds = population
            .len()
            .saturating_mul(world_memberships_per_genome)
            .saturating_div(config.eval_lineages_per_world)
            .saturating_mul(cases_per_match);
        let complexification =
            observe_complexification(generation, &population, best_index, &mut innovations);

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
            best_index,
            offspring_crossovers,
            offspring_clones,
            complexification,
            compatibility_threshold,
            curriculum_level,
            seed_epoch,
            effective_training_seeds,
            config.eval_opponents,
            evaluation_cases_per_genome,
            evaluation_worlds,
            incoming_breeding_telemetry,
            checkpoint_champion_genome,
            config.selection_strategy,
            novelty_archive.len(),
        );
        on_generation(&summary);
        summaries.push(summary);
        update_novelty_archive(&population, &mut novelty_archive, &config);
        if species.len() > config.target_species {
            compatibility_threshold += config.compatibility_threshold_adjustment;
        } else if species.len() < config.target_species {
            compatibility_threshold =
                (compatibility_threshold - config.compatibility_threshold_adjustment).max(0.1);
        }
        if config.curriculum_enabled {
            if best.fitness >= config.curriculum_promotion_threshold {
                curriculum_promotion_streak = curriculum_promotion_streak.saturating_add(1);
            } else {
                curriculum_promotion_streak = 0;
            }
            if curriculum_promotion_streak >= config.curriculum_promotion_patience {
                curriculum_level = curriculum_level.saturating_add(1);
                curriculum_promotion_streak = 0;
                training_scenarios = build_training_scenarios(
                    &world,
                    &config.scenarios,
                    curriculum_level,
                    &level_zero_scenarios,
                );
            }
        }
        previous_species = species;
        if generation + 1 < config.generations {
            population = next_population;
            incoming_breeding_telemetry = next_breeding_telemetry;
        }
    }

    let (
        champion_curriculum_level,
        champion_training_seed_epoch,
        champion_fitness,
        champion_evaluation,
        champion_generation,
        champion_genome,
    ) = champion.ok_or_else(|| anyhow!("NEAT run produced no champion"))?;
    let final_generation = config.generations.saturating_sub(1);
    let final_population = population
        .into_iter()
        .enumerate()
        .map(|(population_index, individual)| PopulationMemberResult {
            generation: final_generation,
            population_index,
            reproduction: individual.reproduction,
            parents: individual.parents,
            fitness: individual.fitness,
            evaluation: individual.evaluation,
            opponent_scores: individual.opponent_scores,
            genome: individual.genome,
        })
        .collect();
    Ok(RunResult {
        result_schema_version: 22,
        algorithm: if config.eval_opponents > 0 {
            "competitive_NEAT"
        } else {
            "NEAT"
        }
        .to_string(),
        objective: config.fitness_objective.name().to_string(),
        seed,
        neat_config: config,
        frozen_outer_loop_contract,
        world_width,
        founder_cohort_size,
        food_energy,
        replay_anchor_scenarios: level_zero_scenarios,
        final_training_scenarios: training_scenarios,
        generations: summaries,
        final_population,
        champion_fitness,
        champion_evaluation,
        champion_generation,
        champion_curriculum_level,
        champion_training_seed_epoch,
        champion_genome,
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
        let mut carrier_fitness = 0.0;
        let mut noncarrier_fitness = 0.0;
        for (index, individual) in population.iter().enumerate() {
            let encoded = individual
                .genome
                .brain
                .edges
                .binary_search_by_key(&record.innovation, |edge| edge.innovation)
                .is_ok();
            if encoded {
                encoded_count += 1;
                carrier_fitness += individual.fitness;
            } else {
                noncarrier_fitness += individual.fitness;
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
        if encoded_count > 0 && encoded_count < population.len() {
            let carrier_mean = carrier_fitness / encoded_count as f64;
            let noncarrier_mean = noncarrier_fitness / (population.len() - encoded_count) as f64;
            let advantage = carrier_mean - noncarrier_mean;
            record.max_carrier_fitness_advantage = Some(
                record
                    .max_carrier_fitness_advantage
                    .map_or(advantage, |old| old.max(advantage)),
            );
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
        best_expressed_hidden_nodes: active[best_index].hidden_nodes.len(),
        best_expressed_connections: active[best_index].connections.len(),
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

fn build_scenarios(
    base: &WorldConfig,
    presets: &[ScenarioPreset],
    curriculum_level: u32,
) -> Vec<ScenarioManifest> {
    presets
        .iter()
        .copied()
        .map(|preset| {
            let mut world = base.clone();
            let level = curriculum_level as f32;
            world.food_tile_fraction = (world.food_tile_fraction - 0.0025 * level).max(0.05);
            match preset {
                ScenarioPreset::Baseline => {}
                ScenarioPreset::Scarcity => {
                    world.food_energy = ((world.food_energy as f32 * 0.75 * 0.985_f32.powf(level))
                        .round() as u32)
                        .max(1);
                    world.food_tile_fraction =
                        (world.food_tile_fraction - 0.12 - 0.004 * level).max(0.03);
                    world.food_regrowth_interval = ((world.food_regrowth_interval as f64
                        * (1.5 + 0.02 * f64::from(curriculum_level)))
                    .round() as u32)
                        .max(1);
                }
                ScenarioPreset::SparseSearch => {
                    world.world_width = ((world.world_width as f64 * 1.4).round() as u32)
                        .max(world.world_width.saturating_add(1))
                        .saturating_add(curriculum_level.saturating_mul(2));
                    // A large clonal cohort can cover a world by brute-force
                    // parallelism, and the old "sparse" treatment still put
                    // food on roughly one third of cells. Evaluate individual
                    // search instead: three clones, food just beyond reliable
                    // ray range, and slow regrowth that discourages camping on
                    // one discovered tile. No new sensors or ecology mechanics
                    // are introduced by this treatment.
                    world.num_organisms = world.num_organisms.div_ceil(10).max(1);
                    world.food_tile_fraction =
                        (world.food_tile_fraction * 0.075 * 0.97_f32.powf(level)).max(0.005);
                    world.food_regrowth_interval = ((world.food_regrowth_interval as f64
                        * (4.0 + 0.02 * f64::from(curriculum_level)))
                    .round() as u32)
                        .max(1);
                }
            }
            ScenarioManifest {
                name: preset.name().to_string(),
                curriculum_level,
                world,
            }
        })
        .collect()
}

fn build_training_scenarios(
    base: &WorldConfig,
    presets: &[ScenarioPreset],
    curriculum_level: u32,
    audit_scenarios: &[ScenarioManifest],
) -> Vec<ScenarioManifest> {
    if curriculum_level == 0 {
        return audit_scenarios.to_vec();
    }
    let frontier = build_scenarios(base, presets, curriculum_level);
    let mut replay = Vec::with_capacity(audit_scenarios.len() + frontier.len());
    replay.extend_from_slice(audit_scenarios);
    replay.extend(frontier);
    replay
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
        .sort_unstable_by_key(|edge| (edge.pre_node_id, edge.post_node_id));
    for edge in &mut genome.brain.edges {
        edge.innovation = registry.connection(
            edge.pre_node_id,
            edge.post_node_id,
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
    if config.eval_opponents == 0 {
        return evaluate_population_isolated(population, scenarios, training_seeds, config);
    }
    if config.eval_lineages_per_world == 3 {
        return evaluate_population_three_lineages(
            population,
            scenarios,
            training_seeds,
            config,
            run_seed,
            generation,
        );
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
                        config.fitness_objective,
                        config.cross_pool_predation_only,
                    );
                    results.lock().expect("evaluation result lock poisoned")[pair_index] =
                        Some(result);
                });
            }
        });
    }

    let mut all_cases = vec![Vec::<CaseEvaluation>::new(); population.len()];
    let mut diagnostic_cases = vec![Vec::<CaseEvaluation>::new(); population.len()];
    for result in results
        .into_inner()
        .expect("evaluation result lock poisoned")
    {
        let result = result.ok_or_else(|| anyhow!("missing pairwise NEAT evaluation result"))??;
        all_cases[result.left_index].extend(result.left_cases);
        all_cases[result.right_index].extend(result.right_cases);
        diagnostic_cases[result.left_index].extend(result.left_diagnostic_cases);
        diagnostic_cases[result.right_index].extend(result.right_diagnostic_cases);
    }
    for index in 0..population.len() {
        if all_cases[index].is_empty() || diagnostic_cases[index].is_empty() {
            bail!("balanced pairwise evaluator produced incomplete cases for genome {index}");
        }
        let mut evaluation =
            summarize_evaluation_cases(&all_cases[index], config.objective_cvar_fraction);
        let diagnostics =
            summarize_evaluation_cases(&diagnostic_cases[index], config.objective_cvar_fraction);
        copy_observational_diagnostics(&mut evaluation, diagnostics);
        population[index].opponent_scores = summarize_opponent_scores(&all_cases[index]);
        population[index].fitness = evaluation.mean_objective_score;
        population[index].evaluation = evaluation;
    }
    Ok(())
}

fn evaluate_population_three_lineages(
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
    let memberships_per_genome = config.eval_opponents / 2;
    let groups = balanced_three_lineage_groups(
        population.len(),
        memberships_per_genome,
        run_seed,
        generation,
    );
    let next = AtomicUsize::new(0);
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(groups.len())
            .collect::<Vec<_>>(),
    );
    let workers = config.evaluator_workers.min(groups.len()).max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let group_index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&member_indices) = groups.get(group_index) else {
                    break;
                };
                let genomes = member_indices.map(|index| &snapshot[index]);
                let result = evaluate_three_lineage_genomes(
                    member_indices,
                    genomes,
                    group_index,
                    scenarios,
                    &config.episode_horizons,
                    &config.survival_window_weights,
                    training_seeds,
                    config.objective_cvar_fraction,
                    config.fitness_objective,
                    config.cross_pool_predation_only,
                );
                results.lock().expect("evaluation result lock poisoned")[group_index] =
                    Some(result);
            });
        }
    });

    let mut all_cases = vec![Vec::<CaseEvaluation>::new(); population.len()];
    let mut diagnostic_cases = vec![Vec::<CaseEvaluation>::new(); population.len()];
    for result in results
        .into_inner()
        .expect("evaluation result lock poisoned")
    {
        let result = result.ok_or_else(|| anyhow!("missing three-lineage evaluation result"))??;
        for member_position in 0..3 {
            let population_index = result.member_indices[member_position];
            all_cases[population_index].extend(result.cases[member_position].iter().cloned());
            diagnostic_cases[population_index]
                .extend(result.diagnostic_cases[member_position].iter().cloned());
        }
    }
    for index in 0..population.len() {
        if all_cases[index].is_empty() {
            bail!("balanced three-lineage evaluator produced incomplete cases for genome {index}");
        }
        if diagnostic_cases[index].is_empty() {
            // A triadic membership records direct behavioral facts only for its
            // instrumented lineage. With fewer than three seed/scenario/horizon
            // cases per membership, a deterministic slot rotation can still
            // leave a genome uninstrumented across all of its otherwise valid
            // competitive cases. Fitness is complete and must not change here;
            // replay one of that genome's actual triples solely to recover its
            // observational action/ecology diagnostics.
            let member_indices = groups
                .iter()
                .copied()
                .find(|members| members.contains(&index))
                .expect("every balanced three-lineage genome has a membership");
            let focal_position = member_indices
                .iter()
                .position(|&member| member == index)
                .expect("selected membership contains the focal genome");
            let opponent_positions = (0..3)
                .filter(|&position| position != focal_position)
                .collect::<Vec<_>>();
            let opponents = opponent_positions
                .iter()
                .map(|&position| snapshot[member_indices[position]].clone())
                .collect::<Vec<_>>();
            let opponent_population_indices = opponent_positions
                .iter()
                .map(|&position| member_indices[position])
                .collect::<Vec<_>>();
            for &episode_ticks in &config.episode_horizons {
                let mut supplemental = evaluate_genome_on_seeds_detailed(
                    &snapshot[index],
                    scenarios,
                    episode_ticks,
                    &config.survival_window_weights,
                    training_seeds,
                    config.objective_cvar_fraction,
                    config.fitness_objective,
                    config.cross_pool_predation_only,
                    focal_position,
                    Some(&opponents),
                )?
                .cases;
                for case in &mut supplemental {
                    case.opponent_population_indices = opponent_population_indices.clone();
                }
                diagnostic_cases[index].extend(supplemental);
            }
        }
        let mut evaluation =
            summarize_evaluation_cases(&all_cases[index], config.objective_cvar_fraction);
        let diagnostics =
            summarize_evaluation_cases(&diagnostic_cases[index], config.objective_cvar_fraction);
        copy_observational_diagnostics(&mut evaluation, diagnostics);
        population[index].opponent_scores = summarize_opponent_scores(&all_cases[index]);
        population[index].fitness = evaluation.mean_objective_score;
        population[index].evaluation = evaluation;
    }
    Ok(())
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

fn evaluate_population_isolated(
    population: &mut [Individual],
    scenarios: &[ScenarioManifest],
    training_seeds: &[u64],
    config: &NeatConfig,
) -> Result<()> {
    let next = AtomicUsize::new(0);
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(population.len())
            .collect::<Vec<_>>(),
    );
    let workers = config.evaluator_workers.min(population.len()).max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= population.len() {
                    break;
                }
                let result = evaluate_genome(
                    &population[index].genome,
                    scenarios,
                    &config.episode_horizons,
                    &config.survival_window_weights,
                    training_seeds,
                    config.objective_cvar_fraction,
                    config.fitness_objective,
                    config.cross_pool_predation_only,
                );
                results.lock().expect("evaluation result lock poisoned")[index] = Some(result);
            });
        }
    });
    let results = results
        .into_inner()
        .expect("evaluation result lock poisoned");
    for (individual, result) in population.iter_mut().zip(results) {
        let evaluation = result.ok_or_else(|| anyhow!("missing NEAT evaluation result"))??;
        individual.fitness = evaluation.mean_objective_score;
        individual.evaluation = evaluation;
    }
    Ok(())
}

struct PairwiseGenomeEvaluation {
    left_index: usize,
    right_index: usize,
    left_cases: Vec<CaseEvaluation>,
    right_cases: Vec<CaseEvaluation>,
    left_diagnostic_cases: Vec<CaseEvaluation>,
    right_diagnostic_cases: Vec<CaseEvaluation>,
}

struct ThreeLineageGenomeEvaluation {
    member_indices: [usize; 3],
    cases: [Vec<CaseEvaluation>; 3],
    diagnostic_cases: [Vec<CaseEvaluation>; 3],
}

fn balanced_three_lineage_groups(
    population_size: usize,
    memberships_per_genome: usize,
    run_seed: u64,
    generation: u32,
) -> Vec<[usize; 3]> {
    debug_assert!(population_size.is_multiple_of(3));
    let mut groups = Vec::with_capacity(
        population_size
            .saturating_mul(memberships_per_genome)
            .saturating_div(3),
    );
    for round in 0..memberships_per_genome {
        let mut order = (0..population_size).collect::<Vec<_>>();
        let mut rng = event_rng(
            run_seed,
            generation,
            round,
            OPPONENT_DOMAIN ^ 0x335f_4c49_4e45_4147,
        );
        order.shuffle(&mut rng);
        groups.extend(
            order
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]]),
        );
    }
    debug_assert_eq!(groups.len() * 3, population_size * memberships_per_genome);
    groups
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
fn evaluate_three_lineage_genomes(
    member_indices: [usize; 3],
    genomes: [&OrganismGenome; 3],
    group_index: usize,
    scenarios: &[ScenarioManifest],
    episode_horizons: &[u64],
    survival_window_weights: &[f64],
    training_seeds: &[u64],
    objective_cvar_fraction: f64,
    fitness_objective: FitnessObjective,
    cross_pool_predation_only: bool,
) -> Result<ThreeLineageGenomeEvaluation> {
    let case_capacity = scenarios
        .len()
        .saturating_mul(training_seeds.len())
        .saturating_mul(episode_horizons.len());
    let mut cases: [Vec<CaseEvaluation>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(case_capacity));
    let mut diagnostic_cases: [Vec<CaseEvaluation>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(case_capacity / 3 + 1));
    let mut case_ordinal = 0usize;
    for &episode_ticks in episode_horizons {
        for scenario in scenarios {
            for &world_seed in training_seeds {
                let instrument_position = (case_ordinal + group_index) % 3;
                let slot_rotation = (case_ordinal / 3 + group_index) % 3;
                let pool_member_positions = [
                    slot_rotation,
                    (slot_rotation + 1) % 3,
                    (slot_rotation + 2) % 3,
                ];
                let focal_pool_index = pool_member_positions
                    .iter()
                    .position(|&position| position == instrument_position)
                    .expect("instrumented member is in the pool");
                let opponent_positions = pool_member_positions
                    .iter()
                    .copied()
                    .filter(|&position| position != instrument_position)
                    .collect::<Vec<_>>();
                let opponents = opponent_positions
                    .iter()
                    .map(|&position| (*genomes[position]).clone())
                    .collect::<Vec<_>>();
                let bundle = evaluate_genome_on_seeds_detailed(
                    genomes[instrument_position],
                    std::slice::from_ref(scenario),
                    episode_ticks,
                    survival_window_weights,
                    std::slice::from_ref(&world_seed),
                    objective_cvar_fraction,
                    fitness_objective,
                    cross_pool_predation_only,
                    focal_pool_index,
                    Some(&opponents),
                )?;
                let mut focal_case = bundle
                    .cases
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("three-lineage match produced no case"))?;
                focal_case.opponent_population_indices = opponent_positions
                    .iter()
                    .map(|&position| member_indices[position])
                    .collect();
                diagnostic_cases[instrument_position].push(focal_case.clone());
                cases[instrument_position].push(focal_case.clone());

                for (member_position, member_cases) in cases.iter_mut().enumerate() {
                    if member_position == instrument_position {
                        continue;
                    }
                    let target_pool_index = pool_member_positions
                        .iter()
                        .position(|&position| position == member_position)
                        .expect("member is in the pool");
                    let opponent_population_indices = (0..3)
                        .filter(|&position| position != member_position)
                        .map(|position| member_indices[position])
                        .collect::<Vec<_>>();
                    member_cases.push(mirrored_multilineage_case(
                        &focal_case,
                        target_pool_index,
                        opponent_population_indices,
                        fitness_objective,
                        survival_window_weights,
                    ));
                }
                case_ordinal += 1;
            }
        }
    }
    Ok(ThreeLineageGenomeEvaluation {
        member_indices,
        cases,
        diagnostic_cases,
    })
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
    fitness_objective: FitnessObjective,
    cross_pool_predation_only: bool,
) -> Result<PairwiseGenomeEvaluation> {
    let case_capacity = scenarios
        .len()
        .saturating_mul(training_seeds.len())
        .saturating_mul(episode_horizons.len());
    let mut left_cases = Vec::with_capacity(case_capacity);
    let mut right_cases = Vec::with_capacity(case_capacity);
    let mut left_diagnostic_cases = Vec::with_capacity(case_capacity / 2);
    let mut right_diagnostic_cases = Vec::with_capacity(case_capacity / 2);
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
                    fitness_objective,
                    cross_pool_predation_only,
                    focal_pool_index,
                    Some(std::slice::from_ref(opponent)),
                )?;
                let mut focal_case = bundle
                    .cases
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("pairwise match produced no case"))?;
                focal_case.opponent_population_indices = vec![opponent_index];
                let mut other_case = mirrored_pairwise_case(
                    &focal_case,
                    focal_population_index,
                    fitness_objective,
                    survival_window_weights,
                );
                other_case.opponent_population_indices = vec![focal_population_index];
                if instrument_left {
                    left_diagnostic_cases.push(focal_case.clone());
                    left_cases.push(focal_case);
                    right_cases.push(other_case);
                } else {
                    right_diagnostic_cases.push(focal_case.clone());
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
        left_diagnostic_cases,
        right_diagnostic_cases,
    })
}

fn mirrored_pairwise_case(
    source: &CaseEvaluation,
    opponent_population_index: usize,
    fitness_objective: FitnessObjective,
    survival_window_weights: &[f64],
) -> CaseEvaluation {
    debug_assert_eq!(source.founder_pool_size, 2);
    let focal_pool_index = 1usize.saturating_sub(source.focal_pool_index);
    mirrored_multilineage_case(
        source,
        focal_pool_index,
        vec![opponent_population_index],
        fitness_objective,
        survival_window_weights,
    )
}

fn mirrored_multilineage_case(
    source: &CaseEvaluation,
    focal_pool_index: usize,
    opponent_population_indices: Vec<usize>,
    fitness_objective: FitnessObjective,
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
    mirrored.objective_score = match fitness_objective {
        FitnessObjective::SurvivalFraction => absolute_survival_fraction,
        FitnessObjective::LateWeightedSurvival => late_weighted_survival_fraction,
        FitnessObjective::SurvivalTimesRelativeAdvantage => {
            absolute_survival_fraction * relative_survival_advantage
        }
    };
    mirrored.opponent_population_indices = opponent_population_indices;
    if let Some(diagnostics) = source.lineage_diagnostics_by_pool.get(focal_pool_index) {
        apply_lineage_case_diagnostics(&mut mirrored, diagnostics);
    }
    mirrored
}

fn apply_lineage_case_diagnostics(case: &mut CaseEvaluation, diagnostics: &LineageCaseDiagnostics) {
    case.consumptions = diagnostics.consumptions;
    case.plant_consumptions = diagnostics.plant_consumptions;
    case.prey_consumptions = diagnostics
        .consumptions
        .saturating_sub(diagnostics.plant_consumptions);
    case.plant_energy_acquired = diagnostics.plant_energy_acquired;
    case.attack_energy_received = diagnostics.attack_energy_received;
    case.attack_energy_lost = diagnostics.attack_energy_lost;
    case.attack_attempt_energy_cost = diagnostics.attack_attempt_energy_cost;
    case.net_attack_energy_balance = diagnostics.attack_energy_received
        - diagnostics.attack_energy_lost
        - diagnostics.attack_attempt_energy_cost;
    case.gross_energy_acquired =
        diagnostics.plant_energy_acquired + diagnostics.attack_energy_received;
    case.attack_no_organism_targets = diagnostics.attack_no_organism_targets;
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
    case.plant_capture_fraction = (case.actionable_plant_supply > 0)
        .then(|| diagnostics.plant_consumptions as f64 / case.actionable_plant_supply as f64);
    case.plant_consumptions_per_tick =
        diagnostics.plant_consumptions as f64 / case.episode_horizon as f64;
}

fn copy_observational_diagnostics(target: &mut Evaluation, source: Evaluation) {
    target.mean_action_effectiveness = source.mean_action_effectiveness;
    target.mean_plant_consumption_rate = source.mean_plant_consumption_rate;
    target.mean_prey_consumption_rate = source.mean_prey_consumption_rate;
    target.mean_mi_sa = source.mean_mi_sa;
    target.mean_learning_slope = source.mean_learning_slope;
    target.trophic_role = source.trophic_role;
    target.plant_intake_fraction = source.plant_intake_fraction;
    target.prey_intake_fraction = source.prey_intake_fraction;
    target.mean_gross_energy_acquired = source.mean_gross_energy_acquired;
    target.mean_plant_energy_acquired = source.mean_plant_energy_acquired;
    target.mean_attack_energy_received = source.mean_attack_energy_received;
    target.mean_attack_energy_lost = source.mean_attack_energy_lost;
    target.mean_attack_attempt_energy_cost = source.mean_attack_attempt_energy_cost;
    target.mean_net_attack_energy_balance = source.mean_net_attack_energy_balance;
    target.mean_consumptions = source.mean_consumptions;
    target.mean_plant_consumptions = source.mean_plant_consumptions;
    target.mean_prey_consumptions = source.mean_prey_consumptions;
    target.mean_attack_no_organism_targets = source.mean_attack_no_organism_targets;
    target.mean_attack_same_pool_blocked = source.mean_attack_same_pool_blocked;
    target.mean_attack_insufficient_energy = source.mean_attack_insufficient_energy;
    target.mean_attack_eligible_attempts = source.mean_attack_eligible_attempts;
    target.mean_attack_hits = source.mean_attack_hits;
    target.mean_attack_nonlethal_hits = source.mean_attack_nonlethal_hits;
    target.mean_attack_kills = source.mean_attack_kills;
    target.mean_attack_same_pair_followups = source.mean_attack_same_pair_followups;
    target.mean_distinct_attack_victims = source.mean_distinct_attack_victims;
    target.attack_repeat_hit_fraction = source.attack_repeat_hit_fraction;
    target.mean_plant_capture_fraction = source.mean_plant_capture_fraction;
    target.mean_plant_consumptions_per_tick = source.mean_plant_consumptions_per_tick;
    target.mean_realized_plant_supply_per_tick = source.mean_realized_plant_supply_per_tick;
    target.mean_standing_plant_fraction = source.mean_standing_plant_fraction;
    target.mean_spatial_coverage = source.mean_spatial_coverage;
    target.mean_normalized_time_to_first_plant = source.mean_normalized_time_to_first_plant;
    target.mean_action_fractions = source.mean_action_fractions;
    target.mean_world_final_population = source.mean_world_final_population;
}

fn assign_selection_scores(
    population: &mut [Individual],
    archive: &[BehaviorDescriptor],
    config: &NeatConfig,
) {
    if config.selection_strategy == SelectionStrategy::Fitness {
        for individual in population {
            individual.selection_score = individual.fitness.max(0.0);
            individual.novelty = 0.0;
            individual.local_competition = 0.0;
            individual.pareto_rank = 0;
        }
        return;
    }

    let descriptors = population
        .iter()
        .map(|individual| BehaviorDescriptor::from_evaluation(individual.evaluation))
        .collect::<Vec<_>>();
    for index in 0..population.len() {
        let mut novelty_distances = descriptors
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, descriptor)| descriptors[index].distance(*descriptor))
            .chain(
                archive
                    .iter()
                    .map(|descriptor| descriptors[index].distance(*descriptor)),
            )
            .collect::<Vec<_>>();
        novelty_distances.sort_by(f64::total_cmp);
        let novelty_neighbors = config.novelty_k.min(novelty_distances.len());
        population[index].novelty = if novelty_neighbors == 0 {
            0.0
        } else {
            novelty_distances[..novelty_neighbors].iter().sum::<f64>() / novelty_neighbors as f64
        };

        let mut local_neighbors = descriptors
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(other, descriptor)| (other, descriptors[index].distance(*descriptor)))
            .collect::<Vec<_>>();
        local_neighbors.sort_by(
            |(left_index, left_distance), (right_index, right_distance)| {
                left_distance
                    .total_cmp(right_distance)
                    .then_with(|| left_index.cmp(right_index))
            },
        );
        let local_neighbor_count = config.novelty_k.min(local_neighbors.len());
        population[index].local_competition = if local_neighbor_count == 0 {
            0.0
        } else {
            local_neighbors[..local_neighbor_count]
                .iter()
                .filter(|(other, _)| local_quality_better(&population[index], &population[*other]))
                .count() as f64
                / local_neighbor_count as f64
        };
    }

    let mut remaining = (0..population.len()).collect::<BTreeSet<_>>();
    let mut fronts = Vec::<Vec<usize>>::new();
    while !remaining.is_empty() {
        let front = remaining
            .iter()
            .copied()
            .filter(|&candidate| {
                !remaining.iter().copied().any(|other| {
                    other != candidate
                        && selection_dominates(&population[other], &population[candidate])
                })
            })
            .collect::<Vec<_>>();
        debug_assert!(!front.is_empty());
        for &index in &front {
            remaining.remove(&index);
        }
        fronts.push(front);
    }

    for (rank, front) in fronts.iter().enumerate() {
        let crowding = pareto_crowding(population, front);
        for (&index, crowding) in front.iter().zip(crowding) {
            population[index].pareto_rank = rank;
            population[index].selection_score = 1.0 / (rank + 1) as f64 + 0.01 * crowding.min(1.0);
        }
    }
}

fn local_quality_better(left: &Individual, right: &Individual) -> bool {
    left.fitness > right.fitness
        || (left.fitness == right.fitness
            && left.evaluation.mean_plant_capture_fraction.unwrap_or(0.0)
                > right.evaluation.mean_plant_capture_fraction.unwrap_or(0.0))
}

fn selection_dominates(left: &Individual, right: &Individual) -> bool {
    left.novelty >= right.novelty
        && left.local_competition >= right.local_competition
        && (left.novelty > right.novelty || left.local_competition > right.local_competition)
}

fn pareto_crowding(population: &[Individual], front: &[usize]) -> Vec<f64> {
    if front.len() <= 2 {
        return vec![1.0; front.len()];
    }
    let mut crowding = vec![0.0; front.len()];
    for objective in [
        |individual: &Individual| individual.novelty,
        |individual: &Individual| individual.local_competition,
    ] {
        let mut order = (0..front.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            objective(&population[front[left]])
                .total_cmp(&objective(&population[front[right]]))
                .then_with(|| front[left].cmp(&front[right]))
        });
        crowding[order[0]] = 1.0;
        crowding[*order.last().expect("nonempty Pareto front")] = 1.0;
        let min = objective(&population[front[order[0]]]);
        let max = objective(&population[front[*order.last().expect("nonempty Pareto front")]]);
        if max <= min {
            continue;
        }
        for window in order.windows(3) {
            let previous = objective(&population[front[window[0]]]);
            let next = objective(&population[front[window[2]]]);
            crowding[window[1]] += (next - previous) / (max - min);
        }
    }
    crowding
}

fn update_novelty_archive(
    population: &[Individual],
    archive: &mut Vec<BehaviorDescriptor>,
    config: &NeatConfig,
) {
    if config.selection_strategy != SelectionStrategy::NoveltyLocalCompetition {
        return;
    }
    let mut order = (0..population.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        population[right]
            .novelty
            .total_cmp(&population[left].novelty)
            .then_with(|| left.cmp(&right))
    });
    for index in order
        .into_iter()
        .take(config.novelty_archive_additions_per_generation)
    {
        let descriptor = BehaviorDescriptor::from_evaluation(population[index].evaluation);
        if archive
            .iter()
            .all(|existing| descriptor.distance(*existing) > 1e-9)
        {
            archive.push(descriptor);
        }
    }
}

struct FounderPool {
    genomes: Vec<OrganismGenome>,
    opponent_population_indices: Vec<usize>,
    fixed_anchor_index: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_genome(
    genome: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_horizons: &[u64],
    survival_window_weights: &[f64],
    training_seeds: &[u64],
    objective_cvar_fraction: f64,
    fitness_objective: FitnessObjective,
    cross_pool_predation_only: bool,
) -> Result<Evaluation> {
    let mut cases = Vec::with_capacity(
        scenarios
            .len()
            .saturating_mul(training_seeds.len())
            .saturating_mul(episode_horizons.len()),
    );
    for &episode_ticks in episode_horizons {
        cases.extend(
            evaluate_genome_on_seeds_detailed(
                genome,
                scenarios,
                episode_ticks,
                survival_window_weights,
                training_seeds,
                objective_cvar_fraction,
                fitness_objective,
                cross_pool_predation_only,
                0,
                None,
            )?
            .cases,
        );
    }
    Ok(summarize_evaluation_cases(&cases, objective_cvar_fraction))
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
    fitness_objective: FitnessObjective,
    cross_pool_predation_only: bool,
    focal_pool_index: usize,
    fixed_opponents: Option<&[OrganismGenome]>,
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
            // Isolated clonal colony by default; competitive mode shares the
            // world with sampled opponents (candidate is pool entry 0).
            let mut founder_pool = if let Some(opponents) = fixed_opponents {
                let mut genomes = Vec::with_capacity(opponents.len() + 1);
                genomes.extend_from_slice(opponents);
                genomes.insert(focal_pool_index.min(opponents.len()), genome.clone());
                FounderPool {
                    genomes,
                    opponent_population_indices: (0..opponents.len()).collect(),
                    fixed_anchor_index: None,
                }
            } else {
                FounderPool {
                    genomes: vec![genome.clone()],
                    opponent_population_indices: Vec::new(),
                    fixed_anchor_index: None,
                }
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
            if fitness_objective == FitnessObjective::SurvivalTimesRelativeAdvantage
                && pool_len != requested_pool_len
            {
                bail!(
                    "scenario `{}` can realize only {} of {} requested founder-pool entries",
                    scenario.name,
                    pool_len,
                    requested_pool_len
                );
            }
            let mut sim = Simulation::new_with_champion_pool(
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
            // (index 0 = candidate). We accumulate alive-ticks per pool entry and
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
            if fitness_objective == FitnessObjective::SurvivalTimesRelativeAdvantage
                && founder_count_by_pool
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
            let mut behavior_ledger = {
                let mut ledger = Ledger::new();
                for organism in sim.organisms().iter().filter(|organism| {
                    (organism.species_id.0 as usize) % pool_len == focal_pool_index
                }) {
                    ledger.birth(organism.id);
                }
                ledger
            };
            let mut alive_ticks_by_pool = vec![0u64; pool_len];
            let mut weighted_alive_ticks_by_pool = vec![0.0_f64; pool_len];
            let mut total_survival_tick_weight = 0.0_f64;
            let world_width = sim.config().world_width as usize;
            let mut visited = vec![false; world_width.saturating_mul(world_width)];
            record_candidate_visited_cells(
                sim.organisms(),
                pool_len,
                focal_pool_index,
                world_width,
                &mut visited,
            );
            let food_tiles = sim.food_tile_count() as u64;
            let initial_plants = sim.foods().iter().count() as u64;
            let mut plant_supply_events = initial_plants;
            let mut standing_plant_cell_turns = 0u64;
            let mut candidate_action_counts = [0u64; 6];
            let mut candidate_action_observations = 0u64;
            let mut candidate_time_to_first_plant = None;
            let mut lineage_diagnostics_by_pool = vec![LineageCaseDiagnostics::default(); pool_len];
            let mut distinct_attack_victims_by_pool = vec![BTreeSet::<OrganismId>::new(); pool_len];
            let mut last_hit_by_pair = HashMap::<(OrganismId, OrganismId), u64>::new();
            #[cfg(feature = "instrumentation")]
            let organism_pool_by_id = sim
                .organisms()
                .iter()
                .map(|organism| (organism.id, (organism.species_id.0 as usize) % pool_len))
                .collect::<HashMap<_, _>>();
            #[cfg(feature = "instrumentation")]
            let mut lineage_consumption_counts = sim
                .organisms()
                .iter()
                .map(|organism| {
                    (
                        organism.id,
                        (
                            organism.consumptions_count,
                            organism.plant_consumptions_count,
                        ),
                    )
                })
                .collect::<HashMap<_, _>>();
            #[cfg(not(feature = "instrumentation"))]
            let mut candidate_consumption_counts = HashMap::<OrganismId, (u64, u64)>::new();
            let mut final_tick_plant_spawns = 0u64;
            let mut world_plant_consumptions = 0u64;
            for _ in 0..episode_ticks {
                let delta = sim.tick();
                #[cfg(feature = "instrumentation")]
                {
                    metrics::ingest_tick(
                        &mut behavior_ledger,
                        delta.turn,
                        &delta,
                        sim.action_records(),
                    );
                    // Action records retain the post-commit cumulative intake
                    // counters even for organisms removed later in this tick.
                    // This makes lineage plant-energy accounting exact without
                    // changing the simulation event model.
                    for record in sim.action_records().iter().flatten() {
                        let pool_index = organism_pool_by_id[&record.organism_id];
                        let previous = lineage_consumption_counts
                            .insert(
                                record.organism_id,
                                (record.consumptions_count, record.plant_consumptions_count),
                            )
                            .unwrap_or((0, 0));
                        let consumption_delta =
                            record.consumptions_count.saturating_sub(previous.0);
                        let plant_delta =
                            record.plant_consumptions_count.saturating_sub(previous.1);
                        let diagnostics = &mut lineage_diagnostics_by_pool[pool_index];
                        diagnostics.consumptions = diagnostics
                            .consumptions
                            .checked_add(consumption_delta)
                            .ok_or_else(|| anyhow!("lineage consumption counter overflow"))?;
                        diagnostics.plant_consumptions = diagnostics
                            .plant_consumptions
                            .checked_add(plant_delta)
                            .ok_or_else(|| anyhow!("lineage plant counter overflow"))?;
                        if pool_index == focal_pool_index
                            && plant_delta > 0
                            && candidate_time_to_first_plant.is_none()
                        {
                            candidate_time_to_first_plant = Some(delta.turn);
                        }
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
                world_plant_consumptions = world_plant_consumptions
                    .checked_add(delta.metrics.plant_consumptions_last_turn)
                    .ok_or_else(|| anyhow!("plant-consumption counter overflow"))?;
                final_tick_plant_spawns = delta.food_spawned.len() as u64;
                plant_supply_events = plant_supply_events
                    .checked_add(final_tick_plant_spawns)
                    .ok_or_else(|| anyhow!("plant-supply counter overflow"))?;
                standing_plant_cell_turns = standing_plant_cell_turns
                    .checked_add(sim.foods().iter().count() as u64)
                    .ok_or_else(|| anyhow!("standing-plant counter overflow"))?;
                record_candidate_visited_cells(
                    sim.organisms(),
                    pool_len,
                    focal_pool_index,
                    world_width,
                    &mut visited,
                );
                for organism in sim.organisms() {
                    let pool_index = (organism.species_id.0 as usize) % pool_len;
                    if pool_index != focal_pool_index {
                        continue;
                    }
                    let action_index = organism.last_action_taken.index();
                    candidate_action_counts[action_index] = candidate_action_counts[action_index]
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("action counter overflow"))?;
                    candidate_action_observations = candidate_action_observations
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("action-observation counter overflow"))?;
                    alive_ticks_by_pool[focal_pool_index] += 1;
                    weighted_alive_ticks_by_pool[focal_pool_index] += survival_tick_weight;
                    #[cfg(not(feature = "instrumentation"))]
                    {
                        let previous = candidate_consumption_counts
                            .insert(
                                organism.id,
                                (
                                    organism.consumptions_count,
                                    organism.plant_consumptions_count,
                                ),
                            )
                            .unwrap_or((0, 0));
                        let consumption_delta =
                            organism.consumptions_count.saturating_sub(previous.0);
                        let plant_delta =
                            organism.plant_consumptions_count.saturating_sub(previous.1);
                        let diagnostics = &mut lineage_diagnostics_by_pool[focal_pool_index];
                        diagnostics.consumptions = diagnostics
                            .consumptions
                            .checked_add(consumption_delta)
                            .ok_or_else(|| anyhow!("candidate consumption counter overflow"))?;
                        diagnostics.plant_consumptions = diagnostics
                            .plant_consumptions
                            .checked_add(plant_delta)
                            .ok_or_else(|| {
                                anyhow!("candidate plant-consumption counter overflow")
                            })?;
                        if plant_delta > 0 && candidate_time_to_first_plant.is_none() {
                            candidate_time_to_first_plant = Some(delta.turn);
                        }
                    }
                }
                for (pool_index, alive_ticks) in alive_ticks_by_pool.iter_mut().enumerate() {
                    if pool_index == focal_pool_index {
                        continue;
                    }
                    let alive_count = sim
                        .organisms()
                        .iter()
                        .filter(|organism| {
                            (organism.species_id.0 as usize) % pool_len == pool_index
                        })
                        .count() as u64;
                    *alive_ticks += alive_count;
                    weighted_alive_ticks_by_pool[pool_index] +=
                        alive_count as f64 * survival_tick_weight;
                }
            }
            let final_standing_plants = sim.foods().iter().count() as u64;
            if sim.metrics().total_plant_consumptions != world_plant_consumptions {
                bail!(
                    "independent plant-consumption totals disagree: evaluator={} sim={}",
                    world_plant_consumptions,
                    sim.metrics().total_plant_consumptions
                );
            }
            debug_assert_eq!(
                plant_supply_events,
                world_plant_consumptions.saturating_add(final_standing_plants),
                "plant spawn/consumption accounting must conserve plant instances"
            );
            if sim.turn() != episode_ticks {
                bail!(
                    "candidate stopped at turn {}; requested {}",
                    sim.turn(),
                    episode_ticks
                );
            }
            // Candidate survival fraction: mean fraction of the episode its
            // founders stayed alive. In the shared scarce world this is
            // implicitly competitive — surviving means out-foraging and
            // out-lasting rivals for too-little food. Dense from generation 0
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
            let objective_score = match fitness_objective {
                FitnessObjective::SurvivalFraction => absolute_survival_fraction,
                FitnessObjective::LateWeightedSurvival => late_weighted_survival_fraction,
                FitnessObjective::SurvivalTimesRelativeAdvantage => {
                    absolute_survival_fraction * relative_survival_advantage
                }
            };
            // Founders still alive at the end, retained per pool so mirrored
            // and multi-lineage evaluations do not infer one pool by subtraction.
            let mut end_survivors_by_pool = vec![0u64; pool_len];
            for organism in sim.organisms() {
                end_survivors_by_pool[(organism.species_id.0 as usize) % pool_len] += 1;
            }
            let candidate_end_survivors = end_survivors_by_pool[focal_pool_index];
            #[cfg(feature = "instrumentation")]
            let (
                action_effectiveness,
                plant_consumption_rate,
                prey_consumption_rate,
                mi_sa,
                learning_slope,
            ) = {
                let row = behavior_ledger.take_behavior_interval(episode_ticks);
                let metrics = derive_interval_metrics(std::slice::from_ref(&row))
                    .into_iter()
                    .next()
                    .expect("one behavior interval produces one metric row");
                (
                    metrics.action_effectiveness,
                    metrics.plant_consumption_rate,
                    metrics.prey_consumption_rate,
                    metrics.mi_sa,
                    metrics.learning_slope,
                )
            };
            #[cfg(not(feature = "instrumentation"))]
            let (
                action_effectiveness,
                plant_consumption_rate,
                prey_consumption_rate,
                mi_sa,
                learning_slope,
            ) = (None, None, None, None, None);
            for (diagnostics, victims) in lineage_diagnostics_by_pool
                .iter_mut()
                .zip(&distinct_attack_victims_by_pool)
            {
                diagnostics.distinct_attack_victims = victims.len() as u64;
                diagnostics.plant_energy_acquired =
                    diagnostics.plant_consumptions as f64 * f64::from(scenario.world.food_energy);
            }
            let focal_diagnostics = &lineage_diagnostics_by_pool[focal_pool_index];
            let case_consumptions = focal_diagnostics.consumptions;
            let case_plant_consumptions = focal_diagnostics.plant_consumptions;
            let case_prey_consumptions = case_consumptions.saturating_sub(case_plant_consumptions);
            let plant_energy_acquired = focal_diagnostics.plant_energy_acquired;
            let attack_energy_received = focal_diagnostics.attack_energy_received;
            let attack_energy_lost = focal_diagnostics.attack_energy_lost;
            let attack_attempt_energy_cost = focal_diagnostics.attack_attempt_energy_cost;
            let net_attack_energy_balance =
                attack_energy_received - attack_energy_lost - attack_attempt_energy_cost;
            let gross_energy_acquired = plant_energy_acquired + attack_energy_received;
            let attack_repeat_hit_fraction = (focal_diagnostics.attack_hits > 0).then(|| {
                focal_diagnostics.attack_same_pair_followups as f64
                    / focal_diagnostics.attack_hits as f64
            });
            let actionable_plant_supply = plant_supply_events
                .checked_sub(final_tick_plant_spawns)
                .ok_or_else(|| anyhow!("final plant spawns exceed realized supply"))?;
            let actionable_leftovers =
                final_standing_plants
                    .checked_sub(final_tick_plant_spawns)
                    .ok_or_else(|| anyhow!("final-tick plant spawns are not all standing"))?;
            if actionable_plant_supply.checked_sub(world_plant_consumptions)
                != Some(actionable_leftovers)
            {
                bail!("world plant supply does not close at the scoring boundary");
            }
            let plant_capture_fraction = if actionable_plant_supply == 0 {
                None
            } else {
                Some(case_plant_consumptions as f64 / actionable_plant_supply as f64)
            };
            let mean_standing_plant_fraction = if food_tiles == 0 {
                0.0
            } else {
                standing_plant_cell_turns as f64 / (food_tiles as f64 * episode_ticks as f64)
            };
            let normalized_time_to_first_plant = candidate_time_to_first_plant
                .unwrap_or_else(|| episode_ticks.saturating_add(1))
                as f64
                / episode_ticks as f64;
            let spatial_coverage = visited.iter().filter(|&&seen| seen).count() as f64
                / sim.habitable_cell_count().max(1) as f64;
            let mut action_fractions = [0.0; 6];
            if candidate_action_observations > 0 {
                for (fraction, count) in action_fractions.iter_mut().zip(candidate_action_counts) {
                    *fraction = count as f64 / candidate_action_observations as f64;
                }
            }
            cases.push(CaseEvaluation {
                scenario: scenario.name.clone(),
                curriculum_level: scenario.curriculum_level,
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
                fixed_anchor_index: founder_pool.fixed_anchor_index,
                founder_pool_size: pool_len,
                focal_pool_index,
                world_founders,
                candidate_founders,
                candidate_end_survivors,
                action_effectiveness,
                plant_consumption_rate,
                prey_consumption_rate,
                mi_sa,
                learning_slope,
                gross_energy_acquired,
                plant_energy_acquired,
                attack_energy_received,
                attack_energy_lost,
                attack_attempt_energy_cost,
                net_attack_energy_balance,
                consumptions: case_consumptions,
                plant_consumptions: case_plant_consumptions,
                prey_consumptions: case_prey_consumptions,
                attack_no_organism_targets: focal_diagnostics.attack_no_organism_targets,
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
                plant_supply_events,
                actionable_plant_supply,
                final_tick_plant_spawns,
                final_standing_plants,
                plant_capture_fraction,
                plant_consumptions_per_tick: case_plant_consumptions as f64 / episode_ticks as f64,
                realized_plant_supply_per_tick: actionable_plant_supply as f64
                    / episode_ticks as f64,
                mean_standing_plant_fraction,
                time_to_first_plant: candidate_time_to_first_plant,
                normalized_time_to_first_plant,
                spatial_coverage,
                action_fractions,
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
    let mut mean_action_fractions = [0.0; 6];
    for case in cases {
        for (mean, value) in mean_action_fractions.iter_mut().zip(case.action_fractions) {
            *mean += value / n;
        }
    }
    let plant_consumptions = cases
        .iter()
        .map(|case| case.plant_consumptions)
        .sum::<u64>();
    let prey_consumptions = cases.iter().map(|case| case.prey_consumptions).sum::<u64>();
    let attack_kills = cases.iter().map(|case| case.attack_kills).sum::<u64>();
    let total_intake = plant_consumptions.saturating_add(prey_consumptions);
    let trophic_role = match (
        plant_consumptions > 0,
        prey_consumptions > 0,
        attack_kills > 0,
    ) {
        (false, false, _) => TrophicRole::Nonconsumer,
        (true, false, _) => TrophicRole::Forager,
        (false, true, false) => TrophicRole::Scavenger,
        (false, true, true) => TrophicRole::Predator,
        (true, true, _) => TrophicRole::Omnivore,
    };
    Evaluation {
        mean_objective_score,
        mean_case_score,
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
        mean_plant_consumption_rate: mean_optional(
            cases.iter().map(|case| case.plant_consumption_rate),
        ),
        mean_prey_consumption_rate: mean_optional(
            cases.iter().map(|case| case.prey_consumption_rate),
        ),
        mean_mi_sa: mean_optional(cases.iter().map(|case| case.mi_sa)),
        mean_learning_slope: mean_optional(cases.iter().map(|case| case.learning_slope)),
        trophic_role,
        plant_intake_fraction: (total_intake > 0)
            .then(|| plant_consumptions as f64 / total_intake as f64),
        prey_intake_fraction: (total_intake > 0)
            .then(|| prey_consumptions as f64 / total_intake as f64),
        mean_gross_energy_acquired: cases
            .iter()
            .map(|case| case.gross_energy_acquired)
            .sum::<f64>()
            / n,
        mean_plant_energy_acquired: cases
            .iter()
            .map(|case| case.plant_energy_acquired)
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
        mean_consumptions: cases
            .iter()
            .map(|case| case.consumptions as f64)
            .sum::<f64>()
            / n,
        mean_plant_consumptions: cases
            .iter()
            .map(|case| case.plant_consumptions as f64)
            .sum::<f64>()
            / n,
        mean_prey_consumptions: cases
            .iter()
            .map(|case| case.prey_consumptions as f64)
            .sum::<f64>()
            / n,
        mean_attack_no_organism_targets: cases
            .iter()
            .map(|case| case.attack_no_organism_targets as f64)
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
        mean_plant_capture_fraction: mean_optional(
            cases.iter().map(|case| case.plant_capture_fraction),
        ),
        mean_plant_consumptions_per_tick: cases
            .iter()
            .map(|case| case.plant_consumptions_per_tick)
            .sum::<f64>()
            / n,
        mean_realized_plant_supply_per_tick: cases
            .iter()
            .map(|case| case.realized_plant_supply_per_tick)
            .sum::<f64>()
            / n,
        mean_standing_plant_fraction: cases
            .iter()
            .map(|case| case.mean_standing_plant_fraction)
            .sum::<f64>()
            / n,
        mean_spatial_coverage: cases.iter().map(|case| case.spatial_coverage).sum::<f64>() / n,
        mean_normalized_time_to_first_plant: cases
            .iter()
            .map(|case| case.normalized_time_to_first_plant)
            .sum::<f64>()
            / n,
        mean_action_fractions,
        mean_world_final_population: cases
            .iter()
            .map(|case| case.world_final_population as f64)
            .sum::<f64>()
            / n,
    }
}

/// Evaluate one frozen focal genome against an explicit, common opponent panel.
/// Every scenario/seed case uses the same ordered opponent genomes, making the
/// relative-survival component comparable across focal champions.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_frozen_panel(
    focal: &OrganismGenome,
    opponents: &[OrganismGenome],
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    survival_window_weights: &[f64],
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
    fitness_objective: FitnessObjective,
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
        fitness_objective,
        cross_pool_predation_only,
        focal_pool_index,
        Some(opponents),
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
    fitness_objective: FitnessObjective,
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
        fitness_objective,
        cross_pool_predation_only,
    )?;
    let mut left_summary =
        summarize_evaluation_cases(&evaluated.left_cases, objective_cvar_fraction);
    let left_diagnostics =
        summarize_evaluation_cases(&evaluated.left_diagnostic_cases, objective_cvar_fraction);
    copy_observational_diagnostics(&mut left_summary, left_diagnostics);
    let mut right_summary =
        summarize_evaluation_cases(&evaluated.right_cases, objective_cvar_fraction);
    let right_diagnostics =
        summarize_evaluation_cases(&evaluated.right_diagnostic_cases, objective_cvar_fraction);
    copy_observational_diagnostics(&mut right_summary, right_diagnostics);
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

fn record_candidate_visited_cells(
    organisms: &[types::OrganismState],
    pool_len: usize,
    focal_pool_index: usize,
    world_width: usize,
    visited: &mut [bool],
) {
    for organism in organisms
        .iter()
        .filter(|organism| (organism.species_id.0 as usize) % pool_len == focal_pool_index)
    {
        let index = organism.r as usize * world_width + organism.q as usize;
        if let Some(cell) = visited.get_mut(index) {
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
            best_fitness: old.best_fitness,
            stagnant_generations: old.stagnant_generations,
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
                best_fitness: f64::NEG_INFINITY,
                stagnant_generations: 0,
                created_generation: generation,
            });
            *next_species_id = next_species_id.saturating_add(1);
        }
    }
    species.retain(|entry| !entry.members.is_empty());
    for entry in &mut species {
        entry.members.sort_unstable();
        entry.representative = population[entry.members[0]].genome.clone();
        let current_best = entry
            .members
            .iter()
            .map(|&index| population[index].selection_score)
            .fold(f64::NEG_INFINITY, f64::max);
        if current_best > entry.best_fitness {
            entry.best_fitness = current_best;
            entry.stagnant_generations = 0;
        } else {
            entry.stagnant_generations = entry.stagnant_generations.saturating_add(1);
        }
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
    let best_species_id = species
        .iter()
        .find(|entry| entry.members.contains(&global_best_index))
        .map(|entry| entry.id)
        .ok_or_else(|| anyhow!("global champion was not assigned to a species"))?;
    let active: Vec<usize> = species
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            let young = parent_generation.saturating_sub(entry.created_generation)
                <= config.young_species_grace_generations;
            entry.id == best_species_id
                || young
                || entry.stagnant_generations <= config.stagnation_generations
        })
        .map(|(index, _)| index)
        .collect();
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
    restrict_predation_genes(genome, predation_enabled);
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
            if brain::genome::connection_would_create_cycle(genome, pre, post) {
                continue;
            }
            if genome
                .brain
                .edges
                .iter()
                .any(|edge| edge.pre_node_id == pre && edge.post_node_id == post && edge.enabled)
            {
                continue;
            }
            candidates.push((pre, post));
        }
    }
    if candidates.is_empty() {
        return (false, false);
    }
    let (pre, post) = candidates[rng.random_range(0..candidates.len())];
    if let Some(edge) = genome
        .brain
        .edges
        .iter_mut()
        .find(|edge| edge.pre_node_id == pre && edge.post_node_id == post)
    {
        edge.enabled = true;
        return (true, false);
    }
    let before = innovations.connection_history.len();
    let innovation = innovations.connection(
        pre,
        post,
        Some(offspring_generation),
        InnovationKind::AddConnection,
    );
    genome.brain.edges.push(SynapseGene {
        innovation,
        pre_node_id: pre,
        post_node_id: post,
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
        weight: 1.0,
        enabled: true,
    });
    genome.brain.edges.push(SynapseGene {
        innovation: split.outgoing,
        pre_node_id: split.node,
        post_node_id: old.post_node_id,
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
    best_index: usize,
    offspring_crossovers: usize,
    offspring_clones: usize,
    complexification: ComplexificationSnapshot,
    compatibility_threshold: f64,
    curriculum_level: u32,
    training_seed_epoch: u32,
    effective_training_seeds: Vec<u64>,
    eval_opponents: usize,
    evaluation_cases_per_genome: usize,
    evaluation_worlds: usize,
    breeding_telemetry: BreedingTelemetry,
    checkpoint_champion_genome: Option<OrganismGenome>,
    selection_strategy: SelectionStrategy,
    novelty_archive_size: usize,
) -> GenerationSummary {
    let mut fitnesses: Vec<_> = population.iter().map(|item| item.fitness).collect();
    fitnesses.sort_by(f64::total_cmp);
    let mean_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
    let median_fitness = if fitnesses.len().is_multiple_of(2) {
        let high = fitnesses.len() / 2;
        (fitnesses[high - 1] + fitnesses[high]) / 2.0
    } else {
        fitnesses[fitnesses.len() / 2]
    };
    let best = &population[best_index];
    let mut gross_energy_acquired_distribution = population
        .iter()
        .map(|individual| individual.evaluation.mean_gross_energy_acquired)
        .collect::<Vec<_>>();
    gross_energy_acquired_distribution.sort_by(f64::total_cmp);
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
    let mut mean_action_fractions = [0.0; 6];
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
    let population_checkpoint = population
        .iter()
        .enumerate()
        .map(|(population_index, individual)| PopulationMemberResult {
            generation,
            population_index,
            reproduction: individual.reproduction,
            parents: individual.parents.clone(),
            fitness: individual.fitness,
            evaluation: individual.evaluation,
            opponent_scores: individual.opponent_scores.clone(),
            genome: individual.genome.clone(),
        })
        .collect();
    let mut population_trophic_roles = TrophicRoleCounts::default();
    for individual in population {
        population_trophic_roles.observe(individual.evaluation.trophic_role);
    }
    let species = species
        .iter()
        .map(|entry| {
            let values: Vec<_> = entry
                .members
                .iter()
                .map(|&index| population[index].fitness)
                .collect();
            SpeciesSummary {
                id: entry.id,
                size: values.len(),
                best_fitness: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                mean_fitness: values.iter().sum::<f64>() / values.len() as f64,
                stagnant_generations: entry.stagnant_generations,
            }
        })
        .collect();
    GenerationSummary {
        generation,
        curriculum_level,
        training_seed_epoch,
        effective_training_seeds,
        eval_opponents,
        evaluation_cases_per_genome,
        evaluation_worlds,
        best_fitness: best.fitness,
        mean_fitness,
        median_fitness,
        best_absolute_survival_fraction: best.evaluation.mean_absolute_survival_fraction,
        best_candidate_alive_ticks: best.evaluation.mean_candidate_alive_ticks,
        best_late_weighted_survival_fraction: best.evaluation.mean_late_weighted_survival_fraction,
        best_relative_survival_advantage: best.evaluation.mean_relative_survival_advantage,
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
        best_mean_prey_consumptions: best.evaluation.mean_prey_consumptions,
        best_trophic_role: best.evaluation.trophic_role,
        best_action_effectiveness: best.evaluation.mean_action_effectiveness,
        best_plant_consumption_rate: best.evaluation.mean_plant_consumption_rate,
        best_prey_consumption_rate: best.evaluation.mean_prey_consumption_rate,
        best_mi_sa: best.evaluation.mean_mi_sa,
        best_learning_slope: best.evaluation.mean_learning_slope,
        best_plant_intake_fraction: best.evaluation.plant_intake_fraction,
        best_prey_intake_fraction: best.evaluation.prey_intake_fraction,
        best_mean_attack_kills: best.evaluation.mean_attack_kills,
        mean_action_effectiveness: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.mean_action_effectiveness),
        ),
        mean_plant_consumption_rate: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.mean_plant_consumption_rate),
        ),
        mean_prey_consumption_rate: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.mean_prey_consumption_rate),
        ),
        population_trophic_roles,
        best_gross_energy_acquired: gross_energy_acquired_distribution
            .last()
            .copied()
            .unwrap_or(0.0),
        mean_gross_energy_acquired,
        median_gross_energy_acquired,
        gross_energy_acquired_distribution,
        champion_plant_energy_acquired: best.evaluation.mean_plant_energy_acquired,
        mean_plant_energy_acquired: population
            .iter()
            .map(|individual| individual.evaluation.mean_plant_energy_acquired)
            .sum::<f64>()
            / population.len() as f64,
        champion_attack_energy_received: best.evaluation.mean_attack_energy_received,
        mean_attack_energy_received: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_energy_received)
            .sum::<f64>()
            / population.len() as f64,
        champion_attack_energy_lost: best.evaluation.mean_attack_energy_lost,
        mean_attack_energy_lost: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_energy_lost)
            .sum::<f64>()
            / population.len() as f64,
        champion_attack_attempt_energy_cost: best.evaluation.mean_attack_attempt_energy_cost,
        mean_attack_attempt_energy_cost: population
            .iter()
            .map(|individual| individual.evaluation.mean_attack_attempt_energy_cost)
            .sum::<f64>()
            / population.len() as f64,
        champion_net_attack_energy_balance: best.evaluation.mean_net_attack_energy_balance,
        mean_net_attack_energy_balance: population
            .iter()
            .map(|individual| individual.evaluation.mean_net_attack_energy_balance)
            .sum::<f64>()
            / population.len() as f64,
        champion_distinct_attack_victims: best.evaluation.mean_distinct_attack_victims,
        mean_distinct_attack_victims: population
            .iter()
            .map(|individual| individual.evaluation.mean_distinct_attack_victims)
            .sum::<f64>()
            / population.len() as f64,
        champion_attack_repeat_hit_fraction: best.evaluation.attack_repeat_hit_fraction,
        mean_attack_repeat_hit_fraction: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.attack_repeat_hit_fraction),
        ),
        champion_action_fractions: best.evaluation.mean_action_fractions,
        mean_action_fractions,
        champion_realized_plant_supply_per_tick: best
            .evaluation
            .mean_realized_plant_supply_per_tick,
        mean_realized_plant_supply_per_tick: population
            .iter()
            .map(|individual| individual.evaluation.mean_realized_plant_supply_per_tick)
            .sum::<f64>()
            / population.len() as f64,
        champion_plant_capture_fraction: best.evaluation.mean_plant_capture_fraction,
        mean_plant_capture_fraction: mean_optional(
            population
                .iter()
                .map(|individual| individual.evaluation.mean_plant_capture_fraction),
        ),
        champion_standing_plant_fraction: best.evaluation.mean_standing_plant_fraction,
        mean_standing_plant_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_standing_plant_fraction)
            .sum::<f64>()
            / population.len() as f64,
        champion_spatial_coverage: best.evaluation.mean_spatial_coverage,
        mean_spatial_coverage: population
            .iter()
            .map(|individual| individual.evaluation.mean_spatial_coverage)
            .sum::<f64>()
            / population.len() as f64,
        best_end_survival_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_candidate_end_survival_fraction)
            .fold(f64::NEG_INFINITY, f64::max),
        mean_end_survival_fraction: population
            .iter()
            .map(|individual| individual.evaluation.mean_candidate_end_survival_fraction)
            .sum::<f64>()
            / population.len() as f64,
        mean_opponent_score_stddev: (!opponent_stddevs.is_empty())
            .then(|| opponent_stddevs.iter().sum::<f64>() / opponent_stddevs.len() as f64),
        max_opponent_score_stddev: opponent_stddevs.into_iter().max_by(f64::total_cmp),
        checkpoint_champion_genome,
        population_checkpoint,
        selection_strategy,
        best_novelty: (selection_strategy == SelectionStrategy::NoveltyLocalCompetition).then(
            || {
                population
                    .iter()
                    .map(|individual| individual.novelty)
                    .fold(f64::NEG_INFINITY, f64::max)
            },
        ),
        mean_novelty: (selection_strategy == SelectionStrategy::NoveltyLocalCompetition).then(
            || {
                population
                    .iter()
                    .map(|individual| individual.novelty)
                    .sum::<f64>()
                    / population.len() as f64
            },
        ),
        best_local_competition: (selection_strategy == SelectionStrategy::NoveltyLocalCompetition)
            .then(|| {
                population
                    .iter()
                    .map(|individual| individual.local_competition)
                    .fold(f64::NEG_INFINITY, f64::max)
            }),
        mean_local_competition: (selection_strategy == SelectionStrategy::NoveltyLocalCompetition)
            .then(|| {
                population
                    .iter()
                    .map(|individual| individual.local_competition)
                    .sum::<f64>()
                    / population.len() as f64
            }),
        novelty_archive_size,
        compatibility_threshold,
        best_hidden_nodes: best.genome.hidden_node_count(),
        best_enabled_connections: best.genome.enabled_connection_count(),
        best_encoded_connections: best.genome.encoded_connection_count(),
        best_expressed_hidden_nodes: complexification.best_expressed_hidden_nodes,
        best_expressed_connections: complexification.best_expressed_connections,
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

fn best_individual_index(population: &[Individual]) -> usize {
    population
        .iter()
        .enumerate()
        .max_by(|(ai, a), (bi, b)| {
            a.fitness
                .total_cmp(&b.fitness)
                .then_with(|| {
                    a.evaluation
                        .mean_plant_capture_fraction
                        .unwrap_or(0.0)
                        .total_cmp(&b.evaluation.mean_plant_capture_fraction.unwrap_or(0.0))
                })
                .then_with(|| bi.cmp(ai))
        })
        .map(|(index, _)| index)
        .expect("NEAT population is non-empty")
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
        novelty: 0.0,
        local_competition: 0.0,
        pareto_rank: 0,
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

fn restrict_predation_genes(genome: &mut OrganismGenome, predation_enabled: bool) {
    if predation_enabled {
        return;
    }
    genome.brain.edges.retain(|edge| {
        !node_is_predation_only(edge.pre_node_id) && !node_is_predation_only(edge.post_node_id)
    });
    let attack_index = ActionType::ALL
        .iter()
        .position(|action| *action == ActionType::Attack)
        .expect("Attack is a canonical action");
    if let Some(bias) = genome.brain.action_biases.get_mut(attack_index) {
        *bias = 0.0;
    }
}

fn node_is_predation_only(node_id: GeneNodeId) -> bool {
    if let Some(index) = sensory_gene_node_index(node_id) {
        return SensoryReceptor::from_neuron_id(NeuronId(index))
            .is_some_and(SensoryReceptor::is_predation_only);
    }
    action_gene_node_index(node_id)
        .and_then(|index| ActionType::ALL.get(index).copied())
        .is_some_and(|action| action == ActionType::Attack)
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
