//! Canonical, generational NEAT: the sole evolutionary system for NeuroGenesis.
//!
//! This module owns species and innovation history. The simulation
//! (`crate::Simulation`) is a deterministic fitness evaluator; NEAT is the outer
//! loop. Every candidate is evaluated as a clonal colony under fixed world seeds
//! — in-world variation does not exist (reproduction is clonal), so there is
//! nothing to freeze off.

use anyhow::{anyhow, bail, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use crate::Simulation;
use sim_config::WorldConfig;
use sim_types::{
    action_gene_node_id, action_gene_node_index, is_hidden_gene_node_id, sensory_gene_node_id,
    sensory_gene_node_index, split_hidden_gene_node_id, ActionType, FoodKind, GeneNodeId,
    HiddenNodeGene, InnovationId, NeuronId, OrganismGenome, SensoryReceptor, SynapseGene,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const WEIGHT_MIN_ABS: f32 = 0.001;
const WEIGHT_MAX_ABS: f32 = 1.5;
const BIAS_MAX_ABS: f32 = 1.0;
const BREED_SELECTION_DOMAIN: u64 = 0x4252_4545_445f_5345;
const CROSSOVER_DOMAIN: u64 = 0x4352_4f53_534f_5645;
const MUTATION_DOMAIN: u64 = 0x4d55_5441_5449_4f4e;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatConfig {
    pub population_size: usize,
    pub generations: u32,
    pub episode_ticks: u64,
    pub world_seeds: Vec<u64>,
    /// `0` freezes training layouts. Otherwise a fresh deterministic seed
    /// suite is derived every N generations; audit seeds never rotate.
    pub training_seed_rotation_period: u32,
    /// Development-audit seeds. Never used for selection, but evaluated at the
    /// configured audit cadence and therefore not considered sealed evidence.
    pub development_world_seeds: Vec<u64>,
    /// Final-only sealed seeds. These are never evaluated in a generation
    /// summary and therefore cannot influence run-time model selection.
    pub sealed_holdout_world_seeds: Vec<u64>,
    /// Fixed curriculum levels used for development audits and the final
    /// sealed evaluation. Unlike the moving training frontier, this grid makes
    /// competence comparable across generations.
    pub audit_curriculum_levels: Vec<u32>,
    /// Evaluate the generation champion on the fixed audit grid every N
    /// generations (and always on the final generation).
    pub development_audit_interval_generations: u32,
    pub scenarios: Vec<ScenarioPreset>,
    /// Select on the mean of the worst-performing fraction of scenario/seed
    /// cases. `1.0` is the ordinary mean; `0.25` is lower-quartile CVaR.
    pub objective_cvar_fraction: f64,
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
            episode_ticks: 500,
            world_seeds: vec![11, 29, 47],
            training_seed_rotation_period: 5,
            development_world_seeds: vec![61, 79, 97],
            sealed_holdout_world_seeds: Vec::new(),
            audit_curriculum_levels: vec![0],
            development_audit_interval_generations: 1,
            scenarios: ScenarioPreset::ALL.to_vec(),
            objective_cvar_fraction: 0.5,
            selection_strategy: SelectionStrategy::Fitness,
            novelty_k: 15,
            novelty_archive_additions_per_generation: 2,
            curriculum_enabled: false,
            curriculum_promotion_threshold: 0.35,
            curriculum_promotion_patience: 3,
            evaluator_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            compatibility_threshold: 3.0,
            target_species: 8,
            compatibility_threshold_adjustment: 0.1,
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.4,
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
        if self.generations == 0 || self.episode_ticks == 0 {
            bail!("NEAT generations and episode_ticks must be >= 1");
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
        if self.curriculum_promotion_threshold < 0.0 || self.curriculum_promotion_patience == 0 {
            bail!("curriculum threshold must be nonnegative and patience must be >= 1");
        }
        let mut seen = std::collections::BTreeSet::new();
        for seed in &self.world_seeds {
            if !seen.insert(*seed) {
                bail!("duplicate training world seed {seed}");
            }
        }
        for seed in &self.development_world_seeds {
            if !seen.insert(*seed) {
                bail!("development seed {seed} duplicates another evaluation seed");
            }
        }
        for seed in &self.sealed_holdout_world_seeds {
            if !seen.insert(*seed) {
                bail!("sealed world seed {seed} duplicates another evaluation seed");
            }
        }
        if self.audit_curriculum_levels.is_empty()
            || self.development_audit_interval_generations == 0
        {
            bail!("audit curriculum levels must be nonempty and audit interval must be >= 1");
        }
        let mut levels = BTreeSet::new();
        for &level in &self.audit_curriculum_levels {
            if !levels.insert(level) {
                bail!("duplicate audit curriculum level {level}");
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

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Evaluation {
    /// The only selection objective in this baseline.
    pub mean_objective_score: f64,
    /// Ordinary mean across cases, retained beside the robust lower-tail
    /// selection objective as a diagnostic.
    pub mean_case_score: f64,
    pub mean_maturity_reached_offspring: f64,
    /// Diagnostics only; these do not contribute to fitness.
    pub mean_successful_births: f64,
    pub mean_consumptions: f64,
    pub mean_plant_consumptions: f64,
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
    pub mean_action_fractions: [f64; 7],
    pub mean_final_population: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseEvaluation {
    pub scenario: String,
    pub curriculum_level: u32,
    pub world_seed: u64,
    pub objective_score: f64,
    pub founders: usize,
    pub maturity_reached_offspring: u64,
    pub successful_births: u64,
    pub consumptions: u64,
    pub plant_consumptions: u64,
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
    pub action_fractions: [f64; 7],
    pub final_population: usize,
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
pub struct FixedLevelEvaluation {
    pub curriculum_level: u32,
    pub evaluation: Evaluation,
    pub cases: Vec<CaseEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedSuiteEvaluation {
    /// Arithmetic mean of the independently computed per-level robust
    /// objectives. Each level gets equal weight; hard levels cannot erase the
    /// shape of the fixed difficulty curve by dominating one flattened CVaR.
    pub mean_level_objective_score: f64,
    pub levels: Vec<FixedLevelEvaluation>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSummary {
    pub generation: u32,
    pub curriculum_level: u32,
    pub training_seed_epoch: u32,
    pub effective_training_seeds: Vec<u64>,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub median_fitness: f64,
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
    /// Disjoint-seed evaluation of this generation's training champion. Never
    /// enters selection.
    pub champion_development_evaluation: Option<FixedSuiteEvaluation>,
    /// Same champion after removing every post-initial structural innovation
    /// while preserving the current enabled state of ancestral connections.
    pub champion_development_evolved_structure_knockout: Option<FixedSuiteEvaluation>,
    pub evolved_structure_development_knockout_delta: Option<f64>,
    /// Stronger ancestral-collapse counterfactual that also re-enables every
    /// initial connection. Kept separate from the pure structural knockout.
    pub champion_development_ancestral_collapse: Option<FixedSuiteEvaluation>,
    pub evolved_structure_development_ancestral_delta: Option<f64>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenOuterLoopContract {
    pub runtime_plasticity_enabled: bool,
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
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
    pub food_energy: f32,
    pub replay_anchor_scenarios: Vec<ScenarioManifest>,
    pub final_training_scenarios: Vec<ScenarioManifest>,
    pub fixed_audit_scenarios: Vec<ScenarioManifest>,
    /// Extra ticks after the scoring window used only to observe whether every
    /// scored-window offspring reaches its fixed maturity age.
    pub maturity_followup_ticks: u64,
    pub generations: Vec<GenerationSummary>,
    pub champion_fitness: f64,
    pub champion_evaluation: Evaluation,
    /// Fixed-grid development diagnostic only; never enters selection.
    pub champion_development_evaluation: Option<FixedSuiteEvaluation>,
    pub champion_development_evolved_structure_knockout: Option<FixedSuiteEvaluation>,
    pub evolved_structure_development_knockout_delta: Option<f64>,
    pub champion_development_ancestral_collapse: Option<FixedSuiteEvaluation>,
    pub evolved_structure_development_ancestral_delta: Option<f64>,
    /// Completely untouched until the run has ended.
    pub sealed_holdout_evaluation: Option<FixedSuiteEvaluation>,
    pub sealed_holdout_evolved_structure_knockout: Option<FixedSuiteEvaluation>,
    pub evolved_structure_sealed_knockout_delta: Option<f64>,
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
    pub frontier_champion_present: bool,
    pub frontier_champion_expressed: bool,
    /// Paired frozen-audit objective loss when this edge alone is disabled.
    /// Positive values identify adaptive causal contribution.
    pub frontier_champion_ablation_delta: Option<f64>,
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
    pub frontier_champion_present: bool,
    pub frontier_champion_expressed: bool,
    /// Paired frozen-audit objective loss when this node and its incident edges
    /// are removed.
    pub frontier_champion_ablation_delta: Option<f64>,
}

#[derive(Clone)]
struct Individual {
    genome: OrganismGenome,
    evaluation: Evaluation,
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

const BEHAVIOR_DESCRIPTOR_DIMENSIONS: usize = 9;

#[derive(Clone, Copy)]
struct BehaviorDescriptor([f64; BEHAVIOR_DESCRIPTOR_DIMENSIONS]);

impl BehaviorDescriptor {
    fn from_evaluation(evaluation: Evaluation) -> Self {
        let mut values = [0.0; BEHAVIOR_DESCRIPTOR_DIMENSIONS];
        values[..7].copy_from_slice(&evaluation.mean_action_fractions);
        values[7] = evaluation.mean_spatial_coverage.clamp(0.0, 1.0);
        values[8] = evaluation
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
    champion_development_evaluation: Option<FixedSuiteEvaluation>,
    champion_development_evolved_structure_knockout: Option<FixedSuiteEvaluation>,
    evolved_structure_development_knockout_delta: Option<f64>,
    champion_development_ancestral_collapse: Option<FixedSuiteEvaluation>,
    evolved_structure_development_ancestral_delta: Option<f64>,
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
                frontier_champion_present: false,
                frontier_champion_expressed: false,
                frontier_champion_ablation_delta: None,
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
                frontier_champion_present: false,
                frontier_champion_expressed: false,
                frontier_champion_ablation_delta: None,
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
    freeze_world_contract(&mut world);
    if !world.leaky_neurons_enabled {
        config.mutate_time_constant_probability = 0.0;
    }
    let level_zero_scenarios = build_scenarios(&world, &config.scenarios, 0);
    let audit_scenarios =
        build_audit_scenarios(&world, &config.scenarios, &config.audit_curriculum_levels);
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
    freeze_genome_contract(&mut template);
    template.brain.hidden_nodes.clear();
    template.brain.edges.retain(|edge| {
        !is_hidden_gene_node_id(edge.pre_node_id) && !is_hidden_gene_node_id(edge.post_node_id)
    });

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut innovations = InnovationRegistry::default();
    canonicalize_initial_markings(&mut template, &mut innovations);
    if template.brain.edges.is_empty() {
        let pre = sensory_gene_node_id(0);
        let post = action_gene_node_id(0);
        template.brain.edges.push(SynapseGene {
            innovation: innovations.connection(pre, post, None, InnovationKind::Initial),
            pre_node_id: pre,
            post_node_id: post,
            weight: random_weight(&mut rng),
            enabled: true,
        });
    }

    let mut population = Vec::with_capacity(config.population_size);
    for _ in 0..config.population_size {
        let mut genome = template.clone();
        randomize_parameters(&mut genome, &mut rng);
        restrict_predation_genes(&mut genome, world.predation_enabled);
        population.push(Individual {
            genome,
            evaluation: Evaluation::default(),
            fitness: 0.0,
            selection_score: 0.0,
            novelty: 0.0,
            local_competition: 0.0,
            pareto_rank: 0,
        });
    }

    let frozen_outer_loop_contract = FrozenOuterLoopContract {
        runtime_plasticity_enabled: world.runtime_plasticity_enabled,
        leaky_neurons_enabled: world.leaky_neurons_enabled,
        predation_enabled: world.predation_enabled,
        intent_parallel_threads: world.intent_parallel_threads,
    };
    let world_width = world.world_width;
    let founder_cohort_size = world.num_organisms;
    let food_energy = world.food_energy;
    let maturity_followup_ticks = u64::from(template.lifecycle.age_of_maturity);

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
        if champion
            .as_ref()
            .is_none_or(|(level, champion_seed_epoch, fitness, evaluation, _, _)| {
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
            })
        {
            champion = Some((
                curriculum_level,
                seed_epoch,
                best.fitness,
                best.evaluation,
                generation,
                best.genome.clone(),
            ));
        }

        let complexification = observe_complexification(
            generation,
            &population,
            best_index,
            &mut innovations,
            &audit_scenarios,
            &config,
            generation + 1 == config.generations
                || generation.is_multiple_of(config.development_audit_interval_generations),
        )?;

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
            incoming_breeding_telemetry,
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
    let champion_development_evaluation = if config.development_world_seeds.is_empty() {
        None
    } else {
        Some(evaluate_genome_on_fixed_suite(
            &champion_genome,
            &audit_scenarios,
            config.episode_ticks,
            &config.development_world_seeds,
            config.objective_cvar_fraction,
        )?)
    };
    let sealed_holdout_evaluation = if config.sealed_holdout_world_seeds.is_empty() {
        None
    } else {
        Some(evaluate_genome_on_fixed_suite(
            &champion_genome,
            &audit_scenarios,
            config.episode_ticks,
            &config.sealed_holdout_world_seeds,
            config.objective_cvar_fraction,
        )?)
    };
    let champion_has_evolved_structure = champion_genome.brain.edges.iter().any(|edge| {
        innovations
            .connection_history
            .get(&edge.innovation)
            .is_some_and(|record| record.origin_generation.is_some())
    });
    let evolved_structure_knockout =
        remove_evolved_structure(&champion_genome, &innovations, false);
    let ancestral_collapse = remove_evolved_structure(&champion_genome, &innovations, true);
    let champion_development_evolved_structure_knockout =
        if let Some(full) = champion_development_evaluation.as_ref() {
            if champion_has_evolved_structure {
                Some(evaluate_genome_on_fixed_suite(
                    &evolved_structure_knockout,
                    &audit_scenarios,
                    config.episode_ticks,
                    &config.development_world_seeds,
                    config.objective_cvar_fraction,
                )?)
            } else {
                Some(full.clone())
            }
        } else {
            None
        };
    let champion_development_ancestral_collapse =
        if let Some(full) = champion_development_evaluation.as_ref() {
            if champion_has_evolved_structure {
                Some(evaluate_genome_on_fixed_suite(
                    &ancestral_collapse,
                    &audit_scenarios,
                    config.episode_ticks,
                    &config.development_world_seeds,
                    config.objective_cvar_fraction,
                )?)
            } else {
                Some(full.clone())
            }
        } else {
            None
        };
    let evolved_structure_development_knockout_delta = champion_development_evaluation
        .as_ref()
        .zip(champion_development_evolved_structure_knockout.as_ref())
        .map(|(full, knockout)| {
            full.mean_level_objective_score - knockout.mean_level_objective_score
        });
    let evolved_structure_development_ancestral_delta = champion_development_evaluation
        .as_ref()
        .zip(champion_development_ancestral_collapse.as_ref())
        .map(|(full, ancestral)| {
            full.mean_level_objective_score - ancestral.mean_level_objective_score
        });
    let sealed_holdout_evolved_structure_knockout =
        if let Some(full) = sealed_holdout_evaluation.as_ref() {
            if champion_has_evolved_structure {
                Some(evaluate_genome_on_fixed_suite(
                    &evolved_structure_knockout,
                    &audit_scenarios,
                    config.episode_ticks,
                    &config.sealed_holdout_world_seeds,
                    config.objective_cvar_fraction,
                )?)
            } else {
                Some(full.clone())
            }
        } else {
            None
        };
    let evolved_structure_sealed_knockout_delta = sealed_holdout_evaluation
        .as_ref()
        .zip(sealed_holdout_evolved_structure_knockout.as_ref())
        .map(|(full, knockout)| {
            full.mean_level_objective_score - knockout.mean_level_objective_score
        });
    let (ablation_seeds, frontier_audit_evaluation) =
        if let Some(development) = champion_development_evaluation.as_ref() {
            (&config.development_world_seeds[..], development.clone())
        } else {
            (
                &config.world_seeds[..],
                evaluate_genome_on_fixed_suite(
                    &champion_genome,
                    &audit_scenarios,
                    config.episode_ticks,
                    &config.world_seeds,
                    config.objective_cvar_fraction,
                )?,
            )
        };
    audit_frontier_champion_structures(
        &champion_genome,
        frontier_audit_evaluation,
        &audit_scenarios,
        config.episode_ticks,
        ablation_seeds,
        config.objective_cvar_fraction,
        &mut innovations,
    )?;
    Ok(RunResult {
        result_schema_version: 2,
        algorithm: "NEAT".to_string(),
        objective: "lower_tail_mean_log_maturity_reached_offspring_per_founder".to_string(),
        seed,
        neat_config: config,
        frozen_outer_loop_contract,
        world_width,
        founder_cohort_size,
        food_energy,
        replay_anchor_scenarios: level_zero_scenarios,
        final_training_scenarios: training_scenarios,
        fixed_audit_scenarios: audit_scenarios,
        maturity_followup_ticks,
        generations: summaries,
        champion_fitness,
        champion_evaluation,
        champion_development_evaluation,
        champion_development_evolved_structure_knockout,
        evolved_structure_development_knockout_delta,
        champion_development_ancestral_collapse,
        evolved_structure_development_ancestral_delta,
        sealed_holdout_evaluation,
        sealed_holdout_evolved_structure_knockout,
        evolved_structure_sealed_knockout_delta,
        champion_generation,
        champion_curriculum_level,
        champion_training_seed_epoch,
        champion_genome,
        connection_innovation_history: innovations.connection_history.values().cloned().collect(),
        node_innovation_history: innovations.node_history.values().cloned().collect(),
    })
}

fn audit_frontier_champion_structures(
    champion: &OrganismGenome,
    full_evaluation: FixedSuiteEvaluation,
    audit_scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    audit_seeds: &[u64],
    objective_cvar_fraction: f64,
    innovations: &mut InnovationRegistry,
) -> Result<()> {
    let active = active_structure(champion);
    let connection_ids: Vec<_> = innovations.connection_history.keys().copied().collect();
    for innovation in connection_ids {
        let present = champion
            .brain
            .edges
            .binary_search_by_key(&innovation, |edge| edge.innovation)
            .is_ok();
        let expressed = active.connections.contains(&innovation);
        let is_evolved = innovations
            .connection_history
            .get(&innovation)
            .is_some_and(|record| record.origin_generation.is_some());
        let delta = if is_evolved && expressed {
            let mut ablated = champion.clone();
            if let Ok(index) = ablated
                .brain
                .edges
                .binary_search_by_key(&innovation, |edge| edge.innovation)
            {
                ablated.brain.edges[index].enabled = false;
            }
            let evaluation = evaluate_genome_on_fixed_suite(
                &ablated,
                audit_scenarios,
                episode_ticks,
                audit_seeds,
                objective_cvar_fraction,
            )?;
            Some(full_evaluation.mean_level_objective_score - evaluation.mean_level_objective_score)
        } else {
            None
        };
        if let Some(record) = innovations.connection_history.get_mut(&innovation) {
            record.frontier_champion_present = present;
            record.frontier_champion_expressed = expressed;
            record.frontier_champion_ablation_delta = delta;
        }
    }

    let node_ids: Vec<_> = innovations.node_history.keys().copied().collect();
    for node_id in node_ids {
        let present = champion
            .brain
            .hidden_nodes
            .binary_search_by_key(&node_id, |node| node.id)
            .is_ok();
        let expressed = active.hidden_nodes.contains(&node_id);
        let delta = if expressed {
            let mut ablated = champion.clone();
            ablated.brain.hidden_nodes.retain(|node| node.id != node_id);
            ablated
                .brain
                .edges
                .retain(|edge| edge.pre_node_id != node_id && edge.post_node_id != node_id);
            let evaluation = evaluate_genome_on_fixed_suite(
                &ablated,
                audit_scenarios,
                episode_ticks,
                audit_seeds,
                objective_cvar_fraction,
            )?;
            Some(full_evaluation.mean_level_objective_score - evaluation.mean_level_objective_score)
        } else {
            None
        };
        if let Some(record) = innovations.node_history.get_mut(&node_id) {
            record.frontier_champion_present = present;
            record.frontier_champion_expressed = expressed;
            record.frontier_champion_ablation_delta = delta;
        }
    }
    Ok(())
}

fn observe_complexification(
    generation: u32,
    population: &[Individual],
    best_index: usize,
    innovations: &mut InnovationRegistry,
    scenarios: &[ScenarioManifest],
    config: &NeatConfig,
    audit_due: bool,
) -> Result<ComplexificationSnapshot> {
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

    let champion_development_evaluation = if !audit_due || config.development_world_seeds.is_empty()
    {
        None
    } else {
        Some(evaluate_genome_on_fixed_suite(
            &population[best_index].genome,
            scenarios,
            config.episode_ticks,
            &config.development_world_seeds,
            config.objective_cvar_fraction,
        )?)
    };
    let has_evolved_structure = population[best_index]
        .genome
        .brain
        .edges
        .iter()
        .any(|edge| {
            innovations
                .connection_history
                .get(&edge.innovation)
                .is_some_and(|record| record.origin_generation.is_some())
        });
    let (champion_development_evolved_structure_knockout, champion_development_ancestral_collapse) =
        if !audit_due || config.development_world_seeds.is_empty() || !has_evolved_structure {
            (
                champion_development_evaluation.clone(),
                champion_development_evaluation.clone(),
            )
        } else {
            let knockout =
                remove_evolved_structure(&population[best_index].genome, innovations, false);
            let ancestral =
                remove_evolved_structure(&population[best_index].genome, innovations, true);
            (
                Some(evaluate_genome_on_fixed_suite(
                    &knockout,
                    scenarios,
                    config.episode_ticks,
                    &config.development_world_seeds,
                    config.objective_cvar_fraction,
                )?),
                Some(evaluate_genome_on_fixed_suite(
                    &ancestral,
                    scenarios,
                    config.episode_ticks,
                    &config.development_world_seeds,
                    config.objective_cvar_fraction,
                )?),
            )
        };
    let evolved_structure_development_knockout_delta = champion_development_evaluation
        .as_ref()
        .zip(champion_development_evolved_structure_knockout.as_ref())
        .map(|(full, knockout)| {
            full.mean_level_objective_score - knockout.mean_level_objective_score
        });
    let evolved_structure_development_ancestral_delta = champion_development_evaluation
        .as_ref()
        .zip(champion_development_ancestral_collapse.as_ref())
        .map(|(full, ancestral)| {
            full.mean_level_objective_score - ancestral.mean_level_objective_score
        });

    Ok(ComplexificationSnapshot {
        best_expressed_hidden_nodes: active[best_index].hidden_nodes.len(),
        best_expressed_connections: active[best_index].connections.len(),
        mean_expressed_hidden_nodes,
        mean_expressed_connections,
        champion_development_evaluation,
        champion_development_evolved_structure_knockout,
        evolved_structure_development_knockout_delta,
        champion_development_ancestral_collapse,
        evolved_structure_development_ancestral_delta,
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
    })
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

fn remove_evolved_structure(
    genome: &OrganismGenome,
    innovations: &InnovationRegistry,
    reenable_ancestral_connections: bool,
) -> OrganismGenome {
    let mut counterfactual = genome.clone();
    counterfactual.brain.edges.retain(|edge| {
        innovations
            .connection_history
            .get(&edge.innovation)
            .is_some_and(|record| record.origin_generation.is_none())
    });
    if reenable_ancestral_connections {
        for edge in &mut counterfactual.brain.edges {
            edge.enabled = true;
        }
    }
    counterfactual.brain.hidden_nodes.clear();
    counterfactual
}

fn freeze_world_contract(world: &mut WorldConfig) {
    world.runtime_plasticity_enabled = false;
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
            world.passive_metabolism_cost_per_unit *= 1.0 + 0.01 * level;
            world.food_tile_fraction = (world.food_tile_fraction - 0.0025 * level).max(0.05);
            match preset {
                ScenarioPreset::Baseline => {}
                ScenarioPreset::Scarcity => {
                    world.food_energy = (world.food_energy * 0.75 * 0.985_f32.powf(level)).max(1.0);
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

fn build_audit_scenarios(
    base: &WorldConfig,
    presets: &[ScenarioPreset],
    levels: &[u32],
) -> Vec<ScenarioManifest> {
    let mut scenarios = Vec::with_capacity(levels.len().saturating_mul(presets.len()));
    for &level in levels {
        scenarios.extend(build_scenarios(base, presets, level));
    }
    scenarios
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

fn freeze_genome_contract(genome: &mut OrganismGenome) {
    genome.plasticity.hebb_eta_gain = 0.0;
    genome.plasticity.juvenile_eta_scale = 0.0;
    genome.plasticity.eligibility_retention = 0.0;
    genome.plasticity.max_weight_delta_per_tick = 0.0;
    genome.plasticity.synapse_prune_threshold = 0.0;
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
                    config.episode_ticks,
                    training_seeds,
                    config.objective_cvar_fraction,
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
            && left
                .evaluation
                .mean_plant_capture_fraction
                .unwrap_or(0.0)
                > right
                    .evaluation
                    .mean_plant_capture_fraction
                    .unwrap_or(0.0))
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

fn evaluate_genome(
    genome: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    training_seeds: &[u64],
    objective_cvar_fraction: f64,
) -> Result<Evaluation> {
    evaluate_genome_on_seeds(
        genome,
        scenarios,
        episode_ticks,
        training_seeds,
        objective_cvar_fraction,
    )
}

fn evaluate_genome_on_seeds(
    genome: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
) -> Result<Evaluation> {
    Ok(evaluate_genome_on_seeds_detailed(
        genome,
        scenarios,
        episode_ticks,
        world_seeds,
        objective_cvar_fraction,
    )?
    .summary)
}

struct EvaluationBundle {
    summary: Evaluation,
    cases: Vec<CaseEvaluation>,
}

fn evaluate_genome_on_seeds_detailed(
    genome: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
) -> Result<EvaluationBundle> {
    if scenarios.is_empty() || world_seeds.is_empty() {
        bail!("a genome evaluation needs at least one scenario and world seed");
    }
    let mut cases = Vec::with_capacity(scenarios.len() * world_seeds.len());
    let mut case_scores = Vec::with_capacity(scenarios.len() * world_seeds.len());
    let maturity_age = genome.lifecycle.age_of_maturity;
    let followup_ticks = u64::from(maturity_age);
    for scenario in scenarios {
        for &world_seed in world_seeds {
            let mut sim = Simulation::new_with_champion_pool(
                scenario.world.clone(),
                world_seed,
                vec![genome.clone()],
            )
            .map_err(|e| {
                anyhow!(
                    "candidate evaluation failed in scenario `{}`: {e}",
                    scenario.name
                )
            })?;
            let founders = sim.organisms().len();
            if founders == 0 {
                bail!("scenario `{}` spawned no founders", scenario.name);
            }
            let world_width = sim.config().world_width as usize;
            let mut visited = vec![false; world_width.saturating_mul(world_width)];
            record_visited_cells(sim.organisms(), world_width, &mut visited);
            let food_tiles = sim.food_tile_count() as u64;
            let initial_plants = sim
                .foods()
                .iter()
                .filter(|food| food.kind == FoodKind::Plant)
                .count() as u64;
            let mut plant_supply_events = initial_plants;
            let mut standing_plant_cell_turns = 0u64;
            let mut action_counts = [0u64; 7];
            let mut action_observations = 0u64;
            let mut time_to_first_plant = None;
            let mut final_tick_plant_spawns = 0u64;
            let mut case_births = 0u64;
            let mut case_consumptions = 0u64;
            let mut case_plant_consumptions = 0u64;
            let mut case_maturity_reached = 0u64;
            let mut pending_offspring = BTreeSet::new();
            for _ in 0..episode_ticks {
                let delta = sim.tick();
                case_births = case_births
                    .checked_add(delta.spawned.len() as u64)
                    .ok_or_else(|| anyhow!("birth counter overflow"))?;
                pending_offspring.extend(delta.spawned.iter().map(|child| child.id));
                case_consumptions = case_consumptions
                    .checked_add(delta.metrics.consumptions_last_turn)
                    .ok_or_else(|| anyhow!("consumption counter overflow"))?;
                case_plant_consumptions = case_plant_consumptions
                    .checked_add(delta.metrics.plant_consumptions_last_turn)
                    .ok_or_else(|| anyhow!("plant-consumption counter overflow"))?;
                if delta.metrics.plant_consumptions_last_turn > 0 && time_to_first_plant.is_none() {
                    time_to_first_plant = Some(delta.turn);
                }
                final_tick_plant_spawns = delta
                    .food_spawned
                    .iter()
                    .filter(|food| food.kind == FoodKind::Plant)
                    .count() as u64;
                plant_supply_events = plant_supply_events
                    .checked_add(final_tick_plant_spawns)
                    .ok_or_else(|| anyhow!("plant-supply counter overflow"))?;
                standing_plant_cell_turns = standing_plant_cell_turns
                    .checked_add(
                        sim.foods()
                            .iter()
                            .filter(|food| food.kind == FoodKind::Plant)
                            .count() as u64,
                    )
                    .ok_or_else(|| anyhow!("standing-plant counter overflow"))?;
                record_visited_cells(sim.organisms(), world_width, &mut visited);
                for organism in sim.organisms() {
                    let action_index = organism.last_action_taken.index();
                    action_counts[action_index] = action_counts[action_index]
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("action counter overflow"))?;
                    action_observations = action_observations
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("action-observation counter overflow"))?;
                }
                case_maturity_reached = case_maturity_reached.saturating_add(
                    mark_mature_offspring(sim.organisms(), &mut pending_offspring, maturity_age)
                        as u64,
                );
            }
            let final_standing_plants = sim
                .foods()
                .iter()
                .filter(|food| food.kind == FoodKind::Plant)
                .count() as u64;
            if sim.metrics().total_plant_consumptions != case_plant_consumptions {
                bail!(
                    "independent plant-consumption totals disagree: evaluator={} sim={}",
                    case_plant_consumptions,
                    sim.metrics().total_plant_consumptions
                );
            }
            debug_assert_eq!(
                plant_supply_events,
                case_plant_consumptions.saturating_add(final_standing_plants),
                "plant spawn/consumption accounting must conserve plant instances"
            );
            for _ in 0..followup_ticks {
                let _ = sim.tick();
                case_maturity_reached = case_maturity_reached.saturating_add(
                    mark_mature_offspring(sim.organisms(), &mut pending_offspring, maturity_age)
                        as u64,
                );
            }
            let expected_final_turn = episode_ticks.saturating_add(followup_ticks);
            if sim.turn() != expected_final_turn {
                bail!(
                    "candidate stopped at turn {}; requested {}",
                    sim.turn(),
                    expected_final_turn
                );
            }
            let objective_score = (1.0 + case_maturity_reached as f64 / founders as f64).ln();
            case_scores.push(objective_score);
            let actionable_plant_supply = plant_supply_events
                .checked_sub(final_tick_plant_spawns)
                .ok_or_else(|| anyhow!("final plant spawns exceed realized supply"))?;
            let actionable_leftovers =
                final_standing_plants
                    .checked_sub(final_tick_plant_spawns)
                    .ok_or_else(|| anyhow!("final-tick plant spawns are not all standing"))?;
            if actionable_plant_supply.checked_sub(case_plant_consumptions)
                != Some(actionable_leftovers)
            {
                bail!("actionable plant supply does not close at the scoring boundary");
            }
            let plant_capture_fraction = if actionable_plant_supply == 0 {
                None
            } else {
                Some(case_plant_consumptions as f64 / actionable_plant_supply as f64)
            };
            if case_plant_consumptions > actionable_plant_supply {
                bail!("plant consumptions exceed actionable realized supply");
            }
            let mean_standing_plant_fraction = if food_tiles == 0 {
                0.0
            } else {
                standing_plant_cell_turns as f64 / (food_tiles as f64 * episode_ticks as f64)
            };
            let normalized_time_to_first_plant =
                time_to_first_plant.unwrap_or_else(|| episode_ticks.saturating_add(1)) as f64
                    / episode_ticks as f64;
            let spatial_coverage = visited.iter().filter(|&&seen| seen).count() as f64
                / sim.habitable_cell_count().max(1) as f64;
            let mut action_fractions = [0.0; 7];
            if action_observations > 0 {
                for (fraction, count) in action_fractions.iter_mut().zip(action_counts) {
                    *fraction = count as f64 / action_observations as f64;
                }
            }
            cases.push(CaseEvaluation {
                scenario: scenario.name.clone(),
                curriculum_level: scenario.curriculum_level,
                world_seed,
                objective_score,
                founders,
                maturity_reached_offspring: case_maturity_reached,
                successful_births: case_births,
                consumptions: case_consumptions,
                plant_consumptions: case_plant_consumptions,
                plant_supply_events,
                actionable_plant_supply,
                final_tick_plant_spawns,
                final_standing_plants,
                plant_capture_fraction,
                plant_consumptions_per_tick: case_plant_consumptions as f64 / episode_ticks as f64,
                realized_plant_supply_per_tick: actionable_plant_supply as f64
                    / episode_ticks as f64,
                mean_standing_plant_fraction,
                time_to_first_plant,
                normalized_time_to_first_plant,
                spatial_coverage,
                action_fractions,
                final_population: sim.organisms().len(),
            });
        }
    }
    let n = cases.len() as f64;
    case_scores.sort_by(f64::total_cmp);
    let cvar_count = ((case_scores.len() as f64 * objective_cvar_fraction).ceil() as usize)
        .clamp(1, case_scores.len());
    let mean_case_score = case_scores.iter().sum::<f64>() / case_scores.len() as f64;
    let mean_objective_score = case_scores[..cvar_count].iter().sum::<f64>() / cvar_count as f64;
    let mut mean_action_fractions = [0.0; 7];
    for case in &cases {
        for (mean, value) in mean_action_fractions.iter_mut().zip(case.action_fractions) {
            *mean += value / n;
        }
    }
    let summary = Evaluation {
        mean_objective_score,
        mean_case_score,
        mean_maturity_reached_offspring: cases
            .iter()
            .map(|case| case.maturity_reached_offspring as f64)
            .sum::<f64>()
            / n,
        mean_successful_births: cases
            .iter()
            .map(|case| case.successful_births as f64)
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
        mean_final_population: cases
            .iter()
            .map(|case| case.final_population as f64)
            .sum::<f64>()
            / n,
    };
    Ok(EvaluationBundle { summary, cases })
}

fn evaluate_genome_on_fixed_suite(
    genome: &OrganismGenome,
    scenarios: &[ScenarioManifest],
    episode_ticks: u64,
    world_seeds: &[u64],
    objective_cvar_fraction: f64,
) -> Result<FixedSuiteEvaluation> {
    let mut by_level = BTreeMap::<u32, Vec<ScenarioManifest>>::new();
    for scenario in scenarios {
        by_level
            .entry(scenario.curriculum_level)
            .or_default()
            .push(scenario.clone());
    }
    let mut levels = Vec::with_capacity(by_level.len());
    for (curriculum_level, level_scenarios) in by_level {
        let bundle = evaluate_genome_on_seeds_detailed(
            genome,
            &level_scenarios,
            episode_ticks,
            world_seeds,
            objective_cvar_fraction,
        )?;
        levels.push(FixedLevelEvaluation {
            curriculum_level,
            evaluation: bundle.summary,
            cases: bundle.cases,
        });
    }
    let mean_level_objective_score = levels
        .iter()
        .map(|level| level.evaluation.mean_objective_score)
        .sum::<f64>()
        / levels.len().max(1) as f64;
    Ok(FixedSuiteEvaluation {
        mean_level_objective_score,
        levels,
    })
}

fn record_visited_cells(
    organisms: &[sim_types::OrganismState],
    world_width: usize,
    visited: &mut [bool],
) {
    for organism in organisms {
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

fn mark_mature_offspring(
    organisms: &[sim_types::OrganismState],
    pending: &mut BTreeSet<sim_types::OrganismId>,
    maturity_age: u32,
) -> usize {
    let mut reached = 0usize;
    for organism in organisms {
        if organism.age_turns >= u64::from(maturity_age) && pending.remove(&organism.id) {
            reached += 1;
        }
    }
    reached
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
            next.push(blank_individual(population[ranked[0]].genome.clone()));
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
            next.push(blank_individual(child));
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
        next.push(blank_individual(genome));
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
    freeze_genome_contract(&mut child);
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
    freeze_genome_contract(genome);
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
    breeding_telemetry: BreedingTelemetry,
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
        best_fitness: best.fitness,
        mean_fitness,
        median_fitness,
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
        champion_development_evaluation: complexification.champion_development_evaluation,
        champion_development_evolved_structure_knockout: complexification
            .champion_development_evolved_structure_knockout,
        evolved_structure_development_knockout_delta: complexification
            .evolved_structure_development_knockout_delta,
        champion_development_ancestral_collapse: complexification
            .champion_development_ancestral_collapse,
        evolved_structure_development_ancestral_delta: complexification
            .evolved_structure_development_ancestral_delta,
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
                        .total_cmp(
                            &b.evaluation
                                .mean_plant_capture_fraction
                                .unwrap_or(0.0),
                        )
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

fn blank_individual(genome: OrganismGenome) -> Individual {
    Individual {
        genome,
        evaluation: Evaluation::default(),
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
