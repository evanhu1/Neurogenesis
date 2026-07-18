//! Task-agnostic generational NEAT.
//!
//! The outer loop owns population initialization, speciation, selection,
//! crossover, and mutation. Genome evaluation is supplied as an independent
//! [`task::EvaluationTask`]. Training tasks live under [`tasks`].

pub mod task;
pub mod tasks;

pub use task::{EvaluationTask, TaskWorkReport};

use anyhow::{anyhow, bail, Result};
use brain::genome::{
    align_genome_vectors, connection_would_create_cycle, generate_seed_genome, MAX_INTER_NEURONS,
    SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN,
};
use rand::{seq::IndexedRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;
use types::{
    action_gene_node_id, action_gene_node_index, connection_innovation_id, is_hidden_gene_node_id,
    sensory_gene_node_id, split_hidden_gene_node_id, GeneNodeId, HiddenNodeGene, InnovationId,
    OrganismGenome, SeedGenomeConfig, SensoryReceptor, Symbol, SynapseGene, SynapseTiming,
};

const BREED_DOMAIN: u64 = 0x4252_4545_445f_5359;
const WEIGHT_MIN_ABS: f32 = SYNAPSE_STRENGTH_MIN;
const WEIGHT_MAX_ABS: f32 = SYNAPSE_STRENGTH_MAX;
const BIAS_MAX_ABS: f32 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatConfig {
    pub population_size: usize,
    pub generations: u32,
    pub population_checkpoint_interval: u32,
    pub evaluation_workers: usize,
    pub compatibility_threshold: f64,
    pub target_species: usize,
    pub compatibility_threshold_adjustment: f64,
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub weight_coefficient: f64,
    pub learning_coefficient: f64,
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
    pub mutate_learning_rate_probability: f64,
    pub learning_rate_perturb_stddev: f32,
    pub mutate_plasticity_coefficient_probability: f64,
    pub plasticity_coefficient_perturb_stddev: f32,
    pub add_connection_probability: f64,
    pub add_node_probability: f64,
    pub disabled_inheritance_probability: f64,
    pub elitism_min_species_size: usize,
}

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            generations: 100,
            population_checkpoint_interval: 10,
            evaluation_workers: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
            compatibility_threshold: 3.0,
            target_species: 8,
            compatibility_threshold_adjustment: 0.1,
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.4,
            learning_coefficient: 0.4,
            survival_fraction: 0.2,
            crossover_probability: 0.75,
            interspecies_mate_probability: 0.01,
            mutate_weight_probability: 0.9,
            per_connection_weight_mutation_probability: 0.8,
            replace_weight_probability: 0.1,
            weight_perturb_stddev: 0.15,
            mutate_bias_probability: 0.25,
            bias_perturb_stddev: 0.15,
            mutate_time_constant_probability: 0.1,
            time_constant_perturb_stddev: 0.15,
            mutate_learning_rate_probability: 0.25,
            learning_rate_perturb_stddev: 0.05,
            mutate_plasticity_coefficient_probability: 0.25,
            plasticity_coefficient_perturb_stddev: 0.15,
            add_connection_probability: 0.08,
            add_node_probability: 0.03,
            disabled_inheritance_probability: 0.75,
            elitism_min_species_size: 5,
        }
    }
}

impl NeatConfig {
    pub fn validate(&self) -> Result<()> {
        if self.population_size < 2 {
            bail!("population_size must be at least 2");
        }
        if self.generations == 0 {
            bail!("generations must be at least 1");
        }
        if self.population_checkpoint_interval == 0 {
            bail!("population_checkpoint_interval must be at least 1");
        }
        if self.evaluation_workers == 0 {
            bail!("evaluation_workers must be at least 1");
        }
        if self.target_species == 0 {
            bail!("target_species must be at least 1");
        }
        positive_finite(self.compatibility_threshold, "compatibility_threshold")?;
        positive_finite(
            self.compatibility_threshold_adjustment,
            "compatibility_threshold_adjustment",
        )?;
        positive_finite(self.excess_coefficient, "excess_coefficient")?;
        positive_finite(self.disjoint_coefficient, "disjoint_coefficient")?;
        positive_finite(self.weight_coefficient, "weight_coefficient")?;
        positive_finite(self.learning_coefficient, "learning_coefficient")?;
        probability_open_closed(self.survival_fraction, "survival_fraction")?;
        for (name, value) in [
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
            (
                "mutate_learning_rate_probability",
                self.mutate_learning_rate_probability,
            ),
            (
                "mutate_plasticity_coefficient_probability",
                self.mutate_plasticity_coefficient_probability,
            ),
            ("add_node_probability", self.add_node_probability),
            (
                "disabled_inheritance_probability",
                self.disabled_inheritance_probability,
            ),
        ] {
            probability(value, name)?;
        }
        for (name, value) in [
            ("weight_perturb_stddev", self.weight_perturb_stddev),
            ("bias_perturb_stddev", self.bias_perturb_stddev),
            (
                "time_constant_perturb_stddev",
                self.time_constant_perturb_stddev,
            ),
            (
                "learning_rate_perturb_stddev",
                self.learning_rate_perturb_stddev,
            ),
            (
                "plasticity_coefficient_perturb_stddev",
                self.plasticity_coefficient_perturb_stddev,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and nonnegative");
            }
        }
        Ok(())
    }
}

fn positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!("{name} must be finite and greater than zero");
    }
    Ok(())
}

fn probability(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be in [0, 1]");
    }
    Ok(())
}

fn probability_open_closed(value: f64, name: &str) -> Result<()> {
    probability(value, name)?;
    if value == 0.0 {
        bail!("{name} must be greater than zero");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub id: u64,
    pub size: usize,
    pub best_fitness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationMemberResult<E> {
    pub population_index: usize,
    pub fitness: f64,
    pub species_id: u64,
    pub evaluation: E,
    pub genome: OrganismGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSummary<E> {
    pub generation: u32,
    pub winner_population_index: usize,
    pub winner_fitness: f64,
    pub winner_normalized_fitness: Option<f64>,
    pub winner_evaluation: E,
    pub winner_validation: Option<E>,
    pub winner_hidden_nodes: usize,
    pub winner_enabled_connections: usize,
    pub species: Vec<SpeciesSummary>,
    pub compatibility_threshold: f64,
    /// Kept empty in lifecycle-v1 results. Complete populations are standalone
    /// generation-boundary checkpoint artifacts.
    pub population_checkpoint: Vec<PopulationMemberResult<E>>,
    pub winner_genome: OrganismGenome,
    pub work: GenerationWork,
    pub wall_time_seconds: f64,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct WorkBreakdown {
    pub genome_evaluations: u64,
    pub brain_synapse_operations: u64,
}

impl WorkBreakdown {
    fn add(&mut self, other: Self) {
        self.genome_evaluations = self
            .genome_evaluations
            .saturating_add(other.genome_evaluations);
        self.brain_synapse_operations = self
            .brain_synapse_operations
            .saturating_add(other.brain_synapse_operations);
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct GenerationWork {
    pub population: WorkBreakdown,
    /// Diagnostic subset of `population`, not additional work.
    pub winner_training: WorkBreakdown,
    pub winner_validation: WorkBreakdown,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct RunWorkTotals {
    pub population: WorkBreakdown,
    /// Diagnostic subset of `population`, excluded from `total()`.
    pub winner_training: WorkBreakdown,
    pub winner_validation: WorkBreakdown,
    pub final_development: WorkBreakdown,
    pub final_sealed: WorkBreakdown,
}

impl RunWorkTotals {
    pub fn total(self) -> WorkBreakdown {
        let mut total = self.population;
        total.add(self.winner_validation);
        total.add(self.final_development);
        total.add(self.final_sealed);
        total
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessThresholdEvent {
    pub normalized_fitness_threshold: f64,
    pub generation: u32,
    pub cumulative_population_genome_evaluations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSessionTiming {
    pub session_index: u32,
    pub resumed_from_generation: u32,
    pub wall_time_seconds: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunTerminationStatus {
    Completed,
    EarlyStopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTermination {
    pub status: RunTerminationStatus,
    pub reason: Option<String>,
    pub evaluated_generations: u32,
    pub configured_generations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenGenomeArtifact<C, E> {
    pub frozen_genome_schema_version: u32,
    pub task: String,
    pub objective: String,
    pub task_config: C,
    pub run_seed: u64,
    pub source_generation: u32,
    pub source_population_index: usize,
    pub role: String,
    pub fitness: f64,
    pub normalized_fitness: Option<f64>,
    pub hidden_nodes: usize,
    pub enabled_connections: usize,
    pub training_evaluation: E,
    pub genome: OrganismGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSpecies {
    pub id: u64,
    pub representative: OrganismGenome,
}

/// Complete semantic state at a before-evaluation generation boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeatCheckpoint<C, E> {
    pub checkpoint_schema_version: u32,
    pub boundary: String,
    pub algorithm: String,
    pub task: String,
    pub objective: String,
    pub seed: u64,
    pub neat_config: NeatConfig,
    pub task_config: C,
    pub seed_genome_config: SeedGenomeConfig,
    pub next_generation: u32,
    pub population: Vec<OrganismGenome>,
    pub compatibility_threshold: f64,
    pub species: Vec<CheckpointSpecies>,
    pub next_species_id: u64,
    pub generations: Vec<GenerationSummary<E>>,
    pub thresholds: Vec<f64>,
    pub threshold_events: Vec<FitnessThresholdEvent>,
    pub historical_champion: Option<FrozenGenomeArtifact<C, E>>,
    pub deterministic_work: RunWorkTotals,
    pub session_timings: Vec<RunSessionTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult<C, E> {
    pub result_schema_version: u32,
    pub algorithm: String,
    pub task: String,
    pub objective: String,
    pub seed: u64,
    pub neat_config: NeatConfig,
    pub task_config: C,
    pub seed_genome_config: SeedGenomeConfig,
    pub generations: Vec<GenerationSummary<E>>,
    pub final_population: Vec<PopulationMemberResult<E>>,
    pub final_winner_population_index: usize,
    pub final_winner_validation: Option<E>,
    pub final_winner_final_evaluation: Option<E>,
    pub termination: RunTermination,
    pub thresholds: Vec<f64>,
    pub threshold_events: Vec<FitnessThresholdEvent>,
    pub deterministic_work: RunWorkTotals,
    pub total_work: WorkBreakdown,
    pub session_timings: Vec<RunSessionTiming>,
    pub total_wall_time_seconds: f64,
}

#[derive(Clone)]
struct Individual<E> {
    genome: OrganismGenome,
    evaluation: Option<E>,
    fitness: f64,
    species_id: u64,
}

impl<E> Individual<E> {
    fn new(genome: OrganismGenome) -> Self {
        Self {
            genome,
            evaluation: None,
            fitness: 0.0,
            species_id: 0,
        }
    }
}

#[derive(Clone)]
struct SpeciesRecord {
    id: u64,
    representative: OrganismGenome,
    members: Vec<usize>,
}

/// Explicitly sized pool for independent genome evaluations. Keeping it local
/// to one research run avoids global-pool contention and makes the machine-wide
/// worker budget part of the persisted NEAT contract.
pub struct EvaluationPool {
    pool: ThreadPool,
}

impl EvaluationPool {
    pub fn new(workers: usize) -> Result<Self> {
        if workers == 0 {
            bail!("evaluation workers must be at least 1");
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("neat-evaluator-{index}"))
            .build()
            .map_err(|error| anyhow!("failed to build NEAT evaluation pool: {error}"))?;
        Ok(Self { pool })
    }

    pub fn evaluate<T: EvaluationTask>(
        &self,
        task: &T,
        genomes: &[OrganismGenome],
    ) -> Result<Vec<(f64, T::Evaluation)>> {
        self.pool.install(|| {
            genomes
                .par_iter()
                .map(|genome| {
                    let evaluation = task.evaluate(genome)?;
                    let fitness = task.fitness(&evaluation);
                    if !fitness.is_finite() || fitness < 0.0 {
                        bail!(
                            "task `{}` produced invalid fitness {fitness}; fitness must be finite and nonnegative",
                            task.name()
                        );
                    }
                    Ok((fitness, evaluation))
                })
                .collect()
        })
    }
}

pub fn run_neat<T: EvaluationTask>(
    task: &T,
    config: NeatConfig,
    seed_genome_config: SeedGenomeConfig,
    seed: u64,
    on_generation: impl FnMut(&GenerationSummary<T::Evaluation>),
) -> Result<RunResult<T::Config, T::Evaluation>> {
    run_neat_controlled(
        task,
        config,
        seed_genome_config,
        seed,
        None,
        vec![0.2, 0.5, 0.8, 0.9],
        true,
        on_generation,
        |_| Ok(()),
        |_| Ok(()),
        || None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_neat_controlled<T: EvaluationTask>(
    task: &T,
    config: NeatConfig,
    seed_genome_config: SeedGenomeConfig,
    seed: u64,
    resume_from: Option<NeatCheckpoint<T::Config, T::Evaluation>>,
    mut thresholds: Vec<f64>,
    embed_population_checkpoints: bool,
    mut on_generation: impl FnMut(&GenerationSummary<T::Evaluation>),
    mut on_checkpoint: impl FnMut(&NeatCheckpoint<T::Config, T::Evaluation>) -> Result<()>,
    mut on_historical_champion: impl FnMut(
        &FrozenGenomeArtifact<T::Config, T::Evaluation>,
    ) -> Result<()>,
    stop_reason: impl Fn() -> Option<String>,
) -> Result<RunResult<T::Config, T::Evaluation>> {
    config.validate()?;
    task.validate()?;
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());
    if thresholds
        .iter()
        .any(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(threshold))
    {
        bail!("normalized fitness thresholds must be finite values in [0, 1]");
    }

    let session_started = Instant::now();
    let (
        start_generation,
        mut population,
        mut compatibility_threshold,
        mut species,
        mut next_species_id,
        mut generations,
        mut threshold_events,
        mut historical_champion,
        mut deterministic_work,
        previous_session_timings,
    ) = if let Some(checkpoint) = resume_from {
        if checkpoint.checkpoint_schema_version != 1 {
            bail!(
                "unsupported NEAT checkpoint schema {}; expected 1",
                checkpoint.checkpoint_schema_version
            );
        }
        if checkpoint.task != task.name() || checkpoint.objective != task.objective() {
            bail!(
                "checkpoint task `{}` / objective `{}` is incompatible with `{}` / `{}`",
                checkpoint.task,
                checkpoint.objective,
                task.name(),
                task.objective()
            );
        }
        if checkpoint.seed != seed {
            bail!(
                "checkpoint seed {} does not match requested seed {seed}",
                checkpoint.seed
            );
        }
        if serde_json::to_value(task.config())? != serde_json::to_value(&checkpoint.task_config)? {
            bail!("checkpoint task configuration does not match the resumed task");
        }
        if serde_json::to_value(&seed_genome_config)?
            != serde_json::to_value(&checkpoint.seed_genome_config)?
        {
            bail!("checkpoint seed-genome configuration does not match the resumed run");
        }
        let mut compatible_neat_config = checkpoint.neat_config.clone();
        compatible_neat_config.generations = config.generations;
        if serde_json::to_value(&config)? != serde_json::to_value(&compatible_neat_config)? {
            bail!(
                "checkpoint NEAT configuration is incompatible; only the terminal generation target may change"
            );
        }
        if checkpoint.population.len() != config.population_size {
            bail!(
                "checkpoint population has {} genomes but configuration requires {}",
                checkpoint.population.len(),
                config.population_size
            );
        }
        if checkpoint.next_generation >= config.generations {
            bail!(
                "resume target generations {} must exceed checkpoint next_generation {}",
                config.generations,
                checkpoint.next_generation
            );
        }
        if checkpoint.thresholds != thresholds {
            bail!("resume thresholds do not match the checkpoint contract");
        }
        (
            checkpoint.next_generation,
            checkpoint
                .population
                .into_iter()
                .map(Individual::<T::Evaluation>::new)
                .collect(),
            checkpoint.compatibility_threshold,
            checkpoint
                .species
                .into_iter()
                .map(|record| SpeciesRecord {
                    id: record.id,
                    representative: record.representative,
                    members: Vec::new(),
                })
                .collect(),
            checkpoint.next_species_id,
            checkpoint.generations,
            checkpoint.threshold_events,
            checkpoint.historical_champion,
            checkpoint.deterministic_work,
            checkpoint.session_timings,
        )
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut template = generate_seed_genome(&seed_genome_config, &mut rng);
        task.prepare_founder_genome(&mut template)?;
        let population = (0..config.population_size)
            .map(|_| {
                let mut genome = template.clone();
                randomize_parameters(&mut genome, &mut rng);
                Individual::<T::Evaluation>::new(genome)
            })
            .collect::<Vec<_>>();
        (
            0,
            population,
            config.compatibility_threshold,
            Vec::new(),
            1,
            Vec::with_capacity(config.generations as usize),
            Vec::new(),
            None,
            RunWorkTotals::default(),
            Vec::new(),
        )
    };

    let evaluation_pool = EvaluationPool::new(config.evaluation_workers)?;
    let mut termination_reason = None;

    for generation in start_generation..config.generations {
        let generation_started = Instant::now();
        let evaluations = evaluation_pool.pool.install(|| {
            population
                .par_iter()
                .map(|individual| {
                    let evaluation = task.evaluate(&individual.genome)?;
                    let fitness = task.fitness(&evaluation);
                    if !fitness.is_finite() || fitness < 0.0 {
                        bail!(
                            "task `{}` produced invalid fitness {fitness}; fitness must be finite and nonnegative",
                            task.name()
                        );
                    }
                    Ok((fitness, evaluation))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let population_work = WorkBreakdown {
            genome_evaluations: evaluations.len() as u64,
            brain_synapse_operations: evaluations.iter().fold(0_u64, |total, (_, evaluation)| {
                total.saturating_add(task.work_report(evaluation).brain_synapse_operations)
            }),
        };
        deterministic_work.population.add(population_work);
        for (individual, (fitness, evaluation)) in population.iter_mut().zip(evaluations) {
            individual.fitness = fitness;
            individual.evaluation = Some(evaluation);
        }

        species = assign_species(
            &mut population,
            &species,
            compatibility_threshold,
            &config,
            &mut next_species_id,
        );
        compatibility_threshold =
            adjusted_compatibility_threshold(compatibility_threshold, species.len(), &config);

        let winner_index = best_index(&population);
        let winner_validation = if task.validation_due(generation, config.generations) {
            task.validation_evaluation(&population[winner_index].genome)?
        } else {
            None
        };
        let validation_work = WorkBreakdown {
            genome_evaluations: u64::from(winner_validation.is_some()),
            brain_synapse_operations: winner_validation
                .as_ref()
                .map(|evaluation| task.work_report(evaluation).brain_synapse_operations)
                .unwrap_or(0),
        };
        deterministic_work.winner_validation.add(validation_work);
        let winner_training_work = WorkBreakdown {
            genome_evaluations: 1,
            brain_synapse_operations: task
                .work_report(
                    population[winner_index]
                        .evaluation
                        .as_ref()
                        .expect("winner is evaluated"),
                )
                .brain_synapse_operations,
        };
        deterministic_work.winner_training.add(winner_training_work);
        // Breeding belongs to the completed generation's wall-time envelope.
        // The child population is also the exact continuation state persisted
        // by any checkpoint written at this boundary.
        let next_population =
            breed_next_generation(task, &population, &species, &config, seed, generation)?;
        let summary = generation_summary(
            task,
            GenerationContext {
                generation,
                winner_index,
                compatibility_threshold,
                winner_validation,
                work: GenerationWork {
                    population: population_work,
                    winner_training: winner_training_work,
                    winner_validation: validation_work,
                },
                persist_population: embed_population_checkpoints
                    && (generation + 1 == config.generations
                        || generation.is_multiple_of(config.population_checkpoint_interval)),
                wall_time_seconds: generation_started.elapsed().as_secs_f64(),
            },
            &population,
            &species,
        );
        let normalized_fitness = summary.winner_normalized_fitness;
        for &threshold in &thresholds {
            if normalized_fitness.is_some_and(|fitness| fitness >= threshold)
                && !threshold_events
                    .iter()
                    .any(|event: &FitnessThresholdEvent| {
                        event
                            .normalized_fitness_threshold
                            .total_cmp(&threshold)
                            .is_eq()
                    })
            {
                threshold_events.push(FitnessThresholdEvent {
                    normalized_fitness_threshold: threshold,
                    generation,
                    cumulative_population_genome_evaluations: deterministic_work
                        .population
                        .genome_evaluations,
                });
            }
        }
        let is_new_champion = historical_champion
            .as_ref()
            .is_none_or(|champion| summary.winner_fitness > champion.fitness);
        if is_new_champion {
            let champion = frozen_genome_artifact(task, seed, &summary, "historical_champion");
            on_historical_champion(&champion)?;
            historical_champion = Some(champion);
        }
        on_generation(&summary);
        generations.push(summary);

        let next_generation = generation + 1;
        termination_reason = stop_reason();
        let checkpoint_due = termination_reason.is_some()
            || next_generation == config.generations
            || next_generation.is_multiple_of(config.population_checkpoint_interval);
        if checkpoint_due {
            let mut session_timings = previous_session_timings.clone();
            session_timings.push(RunSessionTiming {
                session_index: session_timings.len() as u32,
                resumed_from_generation: start_generation,
                wall_time_seconds: session_started.elapsed().as_secs_f64(),
            });
            let checkpoint = NeatCheckpoint {
                checkpoint_schema_version: 1,
                boundary: "before_generation_evaluation".to_owned(),
                algorithm: "generational_neat".to_owned(),
                task: task.name().to_owned(),
                objective: task.objective().to_owned(),
                seed,
                neat_config: config.clone(),
                task_config: task.config(),
                seed_genome_config: seed_genome_config.clone(),
                next_generation,
                population: next_population
                    .iter()
                    .map(|individual| individual.genome.clone())
                    .collect(),
                compatibility_threshold,
                species: species
                    .iter()
                    .map(|record| CheckpointSpecies {
                        id: record.id,
                        representative: record.representative.clone(),
                    })
                    .collect(),
                next_species_id,
                generations: generations.clone(),
                thresholds: thresholds.clone(),
                threshold_events: threshold_events.clone(),
                historical_champion: historical_champion.clone(),
                deterministic_work,
                session_timings,
            };
            on_checkpoint(&checkpoint)?;
        }
        if termination_reason.is_some() || next_generation == config.generations {
            break;
        }
        population = next_population;
    }

    let final_winner_population_index = best_index(&population);
    let mut final_winner_validation = generations
        .last()
        .expect("at least one validated generation was evaluated")
        .winner_validation
        .clone();
    if final_winner_validation.is_none() {
        final_winner_validation =
            task.validation_evaluation(&population[final_winner_population_index].genome)?;
        if let Some(evaluation) = &final_winner_validation {
            deterministic_work.final_development.add(WorkBreakdown {
                genome_evaluations: 1,
                brain_synapse_operations: task.work_report(evaluation).brain_synapse_operations,
            });
        }
    }
    let final_population = population_results(&population);
    let final_winner_final_evaluation =
        task.final_evaluation(&population[final_winner_population_index].genome)?;
    if let Some(evaluation) = &final_winner_final_evaluation {
        deterministic_work.final_sealed.add(WorkBreakdown {
            genome_evaluations: 1,
            brain_synapse_operations: task.work_report(evaluation).brain_synapse_operations,
        });
    }
    let mut session_timings = previous_session_timings;
    session_timings.push(RunSessionTiming {
        session_index: session_timings.len() as u32,
        resumed_from_generation: start_generation,
        wall_time_seconds: session_started.elapsed().as_secs_f64(),
    });
    let total_wall_time_seconds = session_timings
        .iter()
        .map(|timing| timing.wall_time_seconds)
        .sum();
    let evaluated_generations = generations.len() as u32;
    let termination = RunTermination {
        status: if termination_reason.is_some() {
            RunTerminationStatus::EarlyStopped
        } else {
            RunTerminationStatus::Completed
        },
        reason: termination_reason,
        evaluated_generations,
        configured_generations: config.generations,
    };
    let total_work = deterministic_work.total();
    Ok(RunResult {
        result_schema_version: 5,
        algorithm: "generational_neat".to_owned(),
        task: task.name().to_owned(),
        objective: task.objective().to_owned(),
        seed,
        neat_config: config,
        task_config: task.config(),
        seed_genome_config,
        generations,
        final_population,
        final_winner_population_index,
        final_winner_validation,
        final_winner_final_evaluation,
        termination,
        thresholds,
        threshold_events,
        deterministic_work,
        total_work,
        session_timings,
        total_wall_time_seconds,
    })
}

fn frozen_genome_artifact<T: EvaluationTask>(
    task: &T,
    seed: u64,
    summary: &GenerationSummary<T::Evaluation>,
    role: &str,
) -> FrozenGenomeArtifact<T::Config, T::Evaluation> {
    FrozenGenomeArtifact {
        frozen_genome_schema_version: 1,
        task: task.name().to_owned(),
        objective: task.objective().to_owned(),
        task_config: task.config(),
        run_seed: seed,
        source_generation: summary.generation,
        source_population_index: summary.winner_population_index,
        role: role.to_owned(),
        fitness: summary.winner_fitness,
        normalized_fitness: summary.winner_normalized_fitness,
        hidden_nodes: summary.winner_hidden_nodes,
        enabled_connections: summary.winner_enabled_connections,
        training_evaluation: summary.winner_evaluation.clone(),
        genome: summary.winner_genome.clone(),
    }
}

fn assign_species<E>(
    population: &mut [Individual<E>],
    previous: &[SpeciesRecord],
    threshold: f64,
    config: &NeatConfig,
    next_species_id: &mut u64,
) -> Vec<SpeciesRecord> {
    let mut species = previous
        .iter()
        .map(|record| SpeciesRecord {
            id: record.id,
            representative: record.representative.clone(),
            members: Vec::new(),
        })
        .collect::<Vec<_>>();

    for index in canonical_genome_order(population) {
        let genome = &population[index].genome;
        let assigned = species
            .iter()
            .enumerate()
            .filter_map(|(species_index, record)| {
                let distance = compatibility_distance(genome, &record.representative, config);
                (distance <= threshold).then_some((species_index, distance, record.id))
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.2.cmp(&right.2))
            })
            .map(|entry| entry.0);
        let species_index = assigned.unwrap_or_else(|| {
            let id = *next_species_id;
            *next_species_id += 1;
            species.push(SpeciesRecord {
                id,
                representative: genome.clone(),
                members: Vec::new(),
            });
            species.len() - 1
        });
        population[index].species_id = species[species_index].id;
        species[species_index].members.push(index);
    }
    species.retain(|record| !record.members.is_empty());
    for record in &mut species {
        let representative_index = *record
            .members
            .iter()
            .min()
            .expect("nonempty species has a representative");
        record.representative = population[representative_index].genome.clone();
    }
    species.sort_unstable_by_key(|record| record.id);
    species
}

fn canonical_genome_order<E>(population: &[Individual<E>]) -> Vec<usize> {
    let mut order = (0..population.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        genome_key(&population[left].genome)
            .cmp(&genome_key(&population[right].genome))
            .then_with(|| left.cmp(&right))
    });
    order
}

fn genome_key(genome: &OrganismGenome) -> Vec<(InnovationId, u32)> {
    genome
        .brain
        .edges
        .iter()
        .map(|edge| (edge.innovation, edge.weight.to_bits()))
        .collect()
}

fn adjusted_compatibility_threshold(
    threshold: f64,
    species_count: usize,
    config: &NeatConfig,
) -> f64 {
    if species_count > config.target_species {
        threshold + config.compatibility_threshold_adjustment
    } else if species_count < config.target_species {
        (threshold - config.compatibility_threshold_adjustment).max(0.1)
    } else {
        threshold
    }
}

fn compatibility_distance(
    left: &OrganismGenome,
    right: &OrganismGenome,
    config: &NeatConfig,
) -> f64 {
    let left_edges = &left.brain.edges;
    let right_edges = &right.brain.edges;
    let left_max = left_edges.last().map(|edge| edge.innovation);
    let right_max = right_edges.last().map(|edge| edge.innovation);
    let mut left_index = 0;
    let mut right_index = 0;
    let mut matching = 0_usize;
    let mut disjoint = 0_usize;
    let mut excess = 0_usize;
    let mut weight_difference = 0.0_f64;
    while left_index < left_edges.len() && right_index < right_edges.len() {
        let a = &left_edges[left_index];
        let b = &right_edges[right_index];
        if a.innovation == b.innovation {
            matching += 1;
            weight_difference += f64::from((a.weight - b.weight).abs());
            weight_difference +=
                f64::from((a.plasticity_coefficient - b.plasticity_coefficient).abs());
            left_index += 1;
            right_index += 1;
        } else if a.innovation < b.innovation {
            if right_max.is_some_and(|max| a.innovation > max) {
                excess += 1;
            } else {
                disjoint += 1;
            }
            left_index += 1;
        } else {
            if left_max.is_some_and(|max| b.innovation > max) {
                excess += 1;
            } else {
                disjoint += 1;
            }
            right_index += 1;
        }
    }
    excess += left_edges.len() - left_index + right_edges.len() - right_index;
    let normalization = left_edges.len().max(right_edges.len()).max(1) as f64;
    let mean_weight_difference = if matching == 0 {
        0.0
    } else {
        weight_difference / matching as f64
    };
    config.excess_coefficient * excess as f64 / normalization
        + config.disjoint_coefficient * disjoint as f64 / normalization
        + config.weight_coefficient * mean_weight_difference
        + config.learning_coefficient
            * f64::from(
                (left.plasticity.initial_learning_rate
                    - right.plasticity.initial_learning_rate)
                    .abs(),
            )
}

fn breed_next_generation<T: EvaluationTask>(
    task: &T,
    population: &[Individual<T::Evaluation>],
    species: &[SpeciesRecord],
    config: &NeatConfig,
    run_seed: u64,
    generation: u32,
) -> Result<Vec<Individual<T::Evaluation>>> {
    let mut rng = event_rng(run_seed, generation, BREED_DOMAIN);
    let quotas = offspring_quotas(population, species, config.population_size);
    let all_survivors = survivor_indices(population, species, config);
    let mut children = Vec::with_capacity(config.population_size);

    for (record, quota) in species.iter().zip(quotas) {
        if quota == 0 {
            continue;
        }
        let mut ranked = record.members.clone();
        ranked.sort_unstable_by(|&left, &right| {
            population[right]
                .fitness
                .total_cmp(&population[left].fitness)
                .then_with(|| left.cmp(&right))
        });
        let survivor_count = ((ranked.len() as f64 * config.survival_fraction).ceil() as usize)
            .clamp(1, ranked.len());
        let survivors = &ranked[..survivor_count];
        let mut produced = 0_usize;
        if ranked.len() >= config.elitism_min_species_size {
            children.push(Individual::new(population[ranked[0]].genome.clone()));
            produced += 1;
        }
        while produced < quota {
            let first = *survivors
                .choose(&mut rng)
                .ok_or_else(|| anyhow!("species has no breeding survivor"))?;
            let mut child = if rng.random_bool(config.crossover_probability) {
                let mate_pool = if rng.random_bool(config.interspecies_mate_probability) {
                    &all_survivors[..]
                } else {
                    survivors
                };
                let second = *mate_pool
                    .choose(&mut rng)
                    .ok_or_else(|| anyhow!("population has no breeding survivor"))?;
                crossover(&population[first], &population[second], config, &mut rng)
            } else {
                population[first].genome.clone()
            };
            mutate(task, &mut child, config, &mut rng);
            children.push(Individual::new(child));
            produced += 1;
        }
    }

    while children.len() < config.population_size {
        let parent = *all_survivors
            .choose(&mut rng)
            .ok_or_else(|| anyhow!("population has no survivors"))?;
        let mut child = population[parent].genome.clone();
        mutate(task, &mut child, config, &mut rng);
        children.push(Individual::new(child));
    }
    children.truncate(config.population_size);
    Ok(children)
}

fn survivor_indices<E>(
    population: &[Individual<E>],
    species: &[SpeciesRecord],
    config: &NeatConfig,
) -> Vec<usize> {
    let mut survivors = Vec::new();
    for record in species {
        let mut ranked = record.members.clone();
        ranked.sort_unstable_by(|&left, &right| {
            population[right]
                .fitness
                .total_cmp(&population[left].fitness)
                .then_with(|| left.cmp(&right))
        });
        let count = ((ranked.len() as f64 * config.survival_fraction).ceil() as usize)
            .clamp(1, ranked.len());
        survivors.extend_from_slice(&ranked[..count]);
    }
    survivors
}

fn offspring_quotas<E>(
    population: &[Individual<E>],
    species: &[SpeciesRecord],
    population_size: usize,
) -> Vec<usize> {
    let adjusted = species
        .iter()
        .map(|record| {
            record
                .members
                .iter()
                .map(|&index| population[index].fitness / record.members.len() as f64)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let total = adjusted.iter().sum::<f64>();
    let expected = if total > 0.0 {
        adjusted
            .iter()
            .map(|value| value / total * population_size as f64)
            .collect::<Vec<_>>()
    } else {
        vec![population_size as f64 / species.len().max(1) as f64; species.len()]
    };
    let mut quotas = expected
        .iter()
        .map(|value| value.floor() as usize)
        .collect::<Vec<_>>();
    let assigned = quotas.iter().sum::<usize>();
    let mut remainder_order = (0..species.len()).collect::<Vec<_>>();
    remainder_order.sort_unstable_by(|&left, &right| {
        expected[right]
            .fract()
            .total_cmp(&expected[left].fract())
            .then_with(|| species[left].id.cmp(&species[right].id))
    });
    for index in remainder_order
        .into_iter()
        .take(population_size.saturating_sub(assigned))
    {
        quotas[index] += 1;
    }
    quotas
}

fn crossover<E>(
    left: &Individual<E>,
    right: &Individual<E>,
    config: &NeatConfig,
    rng: &mut ChaCha8Rng,
) -> OrganismGenome {
    let (fitter, other, equal) = match left.fitness.total_cmp(&right.fitness) {
        std::cmp::Ordering::Greater => (&left.genome, &right.genome, false),
        std::cmp::Ordering::Less => (&right.genome, &left.genome, false),
        std::cmp::Ordering::Equal if rng.random_bool(0.5) => (&left.genome, &right.genome, true),
        std::cmp::Ordering::Equal => (&right.genome, &left.genome, true),
    };
    let other_by_innovation = other
        .brain
        .edges
        .iter()
        .map(|edge| (edge.innovation, edge))
        .collect::<BTreeMap<_, _>>();
    let mut child = fitter.clone();
    for edge in &mut child.brain.edges {
        if let Some(other_edge) = other_by_innovation.get(&edge.innovation) {
            if rng.random_bool(0.5) {
                *edge = **other_edge;
            }
            if (!edge.enabled || !other_edge.enabled)
                && rng.random_bool(config.disabled_inheritance_probability)
            {
                edge.enabled = false;
            }
        }
    }
    if equal {
        let inherited = child
            .brain
            .edges
            .iter()
            .map(|edge| edge.innovation)
            .collect::<BTreeSet<_>>();
        child.brain.edges.extend(
            other
                .brain
                .edges
                .iter()
                .filter(|edge| !inherited.contains(&edge.innovation) && rng.random_bool(0.5))
                .copied(),
        );
        let existing_hidden = child
            .brain
            .hidden_nodes
            .iter()
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        child.brain.hidden_nodes.extend(
            other
                .brain
                .hidden_nodes
                .iter()
                .filter(|node| !existing_hidden.contains(&node.id))
                .copied(),
        );
    }
    for (bias, other_bias) in child
        .brain
        .action_biases
        .iter_mut()
        .zip(&other.brain.action_biases)
    {
        if rng.random_bool(0.5) {
            *bias = *other_bias;
        }
    }
    if rng.random_bool(0.5) {
        child.plasticity = other.plasticity.clone();
    }
    align_genome_vectors(&mut child, rng);
    child
}

fn mutate<T: EvaluationTask>(
    task: &T,
    genome: &mut OrganismGenome,
    config: &NeatConfig,
    rng: &mut ChaCha8Rng,
) {
    if rng.random_bool(config.mutate_weight_probability) {
        for edge in &mut genome.brain.edges {
            if !rng.random_bool(config.per_connection_weight_mutation_probability) {
                continue;
            }
            edge.weight = if rng.random_bool(config.replace_weight_probability) {
                random_weight(rng)
            } else {
                constrain_weight(edge.weight + normal(rng) * config.weight_perturb_stddev)
            };
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
    if rng.random_bool(config.mutate_learning_rate_probability) {
        genome.plasticity.initial_learning_rate = (genome.plasticity.initial_learning_rate
            + normal(rng) * config.learning_rate_perturb_stddev)
            .clamp(0.0, 1.0);
    }
    for edge in &mut genome.brain.edges {
        if rng.random_bool(config.mutate_plasticity_coefficient_probability) {
            edge.plasticity_coefficient = (edge.plasticity_coefficient
                + normal(rng) * config.plasticity_coefficient_perturb_stddev)
                .clamp(0.0, 2.0);
        }
    }
    if rng.random_bool(config.add_connection_probability) {
        mutate_add_connection(task, genome, rng);
    }
    if rng.random_bool(config.add_node_probability) {
        mutate_add_node(genome, rng);
    }
    align_genome_vectors(genome, rng);
}

fn mutate_add_connection<T: EvaluationTask>(
    task: &T,
    genome: &mut OrganismGenome,
    rng: &mut ChaCha8Rng,
) {
    let sensors = SensoryReceptor::ordered()
        .filter(|sensor| task.sensor_enabled(*sensor))
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| sensory_gene_node_id(id.0))
        .collect::<Vec<_>>();
    let hidden = genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    let actions = Symbol::ALL
        .into_iter()
        .filter(|symbol| task.action_enabled(*symbol))
        .map(|symbol| action_gene_node_id(symbol.index()))
        .collect::<Vec<_>>();
    let mut candidates = Vec::<(GeneNodeId, GeneNodeId, SynapseTiming)>::new();
    for &pre in sensors.iter().chain(hidden.iter()) {
        for &post in hidden.iter().chain(actions.iter()) {
            collect_connection_candidate(
                genome,
                pre,
                post,
                SynapseTiming::CurrentTick,
                &mut candidates,
            );
        }
    }
    for &pre in &hidden {
        for &post in &hidden {
            collect_connection_candidate(
                genome,
                pre,
                post,
                SynapseTiming::PreviousTick,
                &mut candidates,
            );
        }
    }
    let Some(&(pre, post, timing)) = candidates.choose(rng) else {
        return;
    };
    let innovation = connection_innovation_id(pre, post, timing);
    if let Some(edge) = genome
        .brain
        .edges
        .iter_mut()
        .find(|edge| edge.innovation == innovation)
    {
        edge.enabled = true;
        return;
    }
    genome.brain.edges.push(SynapseGene {
        innovation,
        pre_node_id: pre,
        post_node_id: post,
        timing,
        weight: random_weight(rng),
        plasticity_coefficient: 1.0,
        enabled: true,
    });
}

fn collect_connection_candidate(
    genome: &OrganismGenome,
    pre: GeneNodeId,
    post: GeneNodeId,
    timing: SynapseTiming,
    candidates: &mut Vec<(GeneNodeId, GeneNodeId, SynapseTiming)>,
) {
    if action_gene_node_index(pre).is_some()
        || (timing == SynapseTiming::PreviousTick
            && (!is_hidden_gene_node_id(pre) || !is_hidden_gene_node_id(post)))
        || connection_would_create_cycle(genome, pre, post, timing)
    {
        return;
    }
    let innovation = connection_innovation_id(pre, post, timing);
    if genome
        .brain
        .edges
        .iter()
        .any(|edge| edge.innovation == innovation && edge.enabled)
    {
        return;
    }
    candidates.push((pre, post, timing));
}

fn mutate_add_node(genome: &mut OrganismGenome, rng: &mut ChaCha8Rng) {
    if genome.brain.hidden_nodes.len() >= MAX_INTER_NEURONS as usize {
        return;
    }
    let candidates = genome
        .brain
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.enabled && edge.timing == SynapseTiming::CurrentTick)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let Some(&edge_index) = candidates.choose(rng) else {
        return;
    };
    let original = genome.brain.edges[edge_index];
    let node_id = split_hidden_gene_node_id(original.innovation);
    if genome
        .brain
        .hidden_nodes
        .iter()
        .any(|node| node.id == node_id)
    {
        return;
    }
    genome.brain.edges[edge_index].enabled = false;
    genome.brain.hidden_nodes.push(HiddenNodeGene {
        id: node_id,
        bias: 0.0,
        log_time_constant: 0.0,
    });
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(
            original.pre_node_id,
            node_id,
            SynapseTiming::CurrentTick,
        ),
        pre_node_id: original.pre_node_id,
        post_node_id: node_id,
        timing: SynapseTiming::CurrentTick,
        weight: 1.0,
        plasticity_coefficient: original.plasticity_coefficient,
        enabled: true,
    });
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(
            node_id,
            original.post_node_id,
            SynapseTiming::CurrentTick,
        ),
        pre_node_id: node_id,
        post_node_id: original.post_node_id,
        timing: SynapseTiming::CurrentTick,
        weight: original.weight,
        plasticity_coefficient: original.plasticity_coefficient,
        enabled: true,
    });
}

struct GenerationContext<E> {
    generation: u32,
    winner_index: usize,
    compatibility_threshold: f64,
    winner_validation: Option<E>,
    work: GenerationWork,
    persist_population: bool,
    wall_time_seconds: f64,
}

fn generation_summary<T: EvaluationTask>(
    task: &T,
    context: GenerationContext<T::Evaluation>,
    population: &[Individual<T::Evaluation>],
    species: &[SpeciesRecord],
) -> GenerationSummary<T::Evaluation> {
    let winner = &population[context.winner_index];
    let winner_evaluation = winner
        .evaluation
        .as_ref()
        .expect("population is evaluated before summarization");
    GenerationSummary {
        generation: context.generation,
        winner_population_index: context.winner_index,
        winner_fitness: winner.fitness,
        winner_normalized_fitness: task.normalized_fitness(winner_evaluation),
        winner_evaluation: winner_evaluation.clone(),
        winner_validation: context.winner_validation,
        winner_hidden_nodes: winner.genome.hidden_node_count(),
        winner_enabled_connections: winner.genome.enabled_connection_count(),
        species: species
            .iter()
            .map(|record| SpeciesSummary {
                id: record.id,
                size: record.members.len(),
                best_fitness: record
                    .members
                    .iter()
                    .map(|&index| population[index].fitness)
                    .max_by(f64::total_cmp)
                    .unwrap_or(0.0),
            })
            .collect(),
        compatibility_threshold: context.compatibility_threshold,
        population_checkpoint: if context.persist_population {
            population_results(population)
        } else {
            Vec::new()
        },
        winner_genome: winner.genome.clone(),
        work: context.work,
        wall_time_seconds: context.wall_time_seconds,
    }
}

fn population_results<E: Clone>(population: &[Individual<E>]) -> Vec<PopulationMemberResult<E>> {
    population
        .iter()
        .enumerate()
        .map(|(population_index, individual)| PopulationMemberResult {
            population_index,
            fitness: individual.fitness,
            species_id: individual.species_id,
            evaluation: individual
                .evaluation
                .clone()
                .expect("population is evaluated before persistence"),
            genome: individual.genome.clone(),
        })
        .collect()
}

fn best_index<E>(population: &[Individual<E>]) -> usize {
    population
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.fitness
                .total_cmp(&right.fitness)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("a validated NEAT population is nonempty")
}

fn randomize_parameters(genome: &mut OrganismGenome, rng: &mut ChaCha8Rng) {
    for edge in &mut genome.brain.edges {
        edge.weight = random_weight(rng);
    }
    for bias in &mut genome.brain.action_biases {
        *bias = normal(rng).clamp(-BIAS_MAX_ABS, BIAS_MAX_ABS);
    }
    for node in &mut genome.brain.hidden_nodes {
        node.bias = normal(rng).clamp(-BIAS_MAX_ABS, BIAS_MAX_ABS);
    }
    genome.plasticity.initial_learning_rate = rng.random_range(0.0..=0.5);
    for edge in &mut genome.brain.edges {
        edge.plasticity_coefficient = rng.random_range(0.0..=2.0);
    }
}

fn random_weight(rng: &mut ChaCha8Rng) -> f32 {
    constrain_weight(normal(rng) * 0.5)
}

fn constrain_weight(weight: f32) -> f32 {
    if weight == 0.0 {
        return WEIGHT_MIN_ABS;
    }
    weight.signum() * weight.abs().clamp(WEIGHT_MIN_ABS, WEIGHT_MAX_ABS)
}

fn normal(rng: &mut ChaCha8Rng) -> f32 {
    StandardNormal.sample(rng)
}

fn event_rng(run_seed: u64, generation: u32, domain: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(mix64(run_seed ^ domain ^ (u64::from(generation) << 32)))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
