//! Bounded Causal Ecology PowerPlay vertical slice.
//!
//! This is an evaluation-owned outer loop over the ordinary deterministic
//! [`Simulation`].  A task is a causal chain of plant-energy captures: only the
//! first resource exists initially and consuming it releases the next resource
//! relative to the organism's current pose.  A solver may therefore collect a
//! later payoff only after collecting every predecessor payoff.
//!
//! The implementation is deliberately a *bounded pilot*, not a claim of
//! open-endedness.  It exercises four mechanisms needed by a future unbounded
//! system: generated payoff-bearing tasks, protected additive solver capacity,
//! predecessor-retention gates, and an exact causal module knockout.

use crate::{
    genome::generate_seed_genome, grid::hex_neighbor, grid::rotate_by_steps,
    progressive::ProtectedResidual, Simulation,
};
use anyhow::{anyhow, bail, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use sim_config::WorldConfig;
use sim_types::{
    action_gene_node_id, connection_innovation_id, seed_hidden_gene_node_id, sensory_gene_node_id,
    ActionType, FacingDirection, FoodId, FoodKind, FoodState, GeneNodeId, HiddenNodeGene,
    InnovationId, Occupant, OrganismGenome, SynapseGene,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;

const RESULT_SCHEMA_VERSION: u32 = 2;
const TASK_DOMAIN: u64 = 0x504f_5745_5250_4c59;
const TASK_ENERGY_EPSILON_MULTIPLIER: f32 = 32.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPlayConfig {
    pub run_seed: u64,
    pub max_depth: u32,
    pub population_size: usize,
    pub generations_per_depth: u32,
    pub module_width: usize,
    /// Mutable/search contexts. These never decide admission.
    pub search_seeds: Vec<u64>,
    /// Fixed admission contexts, untouched by module search.
    pub episode_seeds: Vec<u64>,
    pub ticks_per_stage: u64,
    pub pass_fraction: f64,
    pub predecessor_fail_max_fraction: f64,
    pub world_width: u32,
    pub food_energy: f32,
}

impl Default for PowerPlayConfig {
    fn default() -> Self {
        Self {
            run_seed: 7,
            max_depth: 4,
            population_size: 96,
            generations_per_depth: 160,
            module_width: 4,
            search_seeds: vec![
                1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
                1093, 1097,
            ],
            episode_seeds: vec![
                7, 11, 29, 42, 71, 101, 123, 131, 149, 181, 211, 2026, 2718, 31_415, 65_537, 99_991,
            ],
            ticks_per_stage: 18,
            pass_fraction: 14.0 / 16.0,
            predecessor_fail_max_fraction: 2.0 / 16.0,
            world_width: 15,
            food_energy: 10.0,
        }
    }
}

impl PowerPlayConfig {
    fn validate(&self) -> Result<()> {
        if self.max_depth == 0 || self.max_depth > 4 {
            bail!("powerplay pilot depth must be in 1..=4");
        }
        if self.population_size < 2 || self.generations_per_depth == 0 {
            bail!("population must be >= 2 and generations-per-depth must be >= 1");
        }
        if self.module_width == 0 || self.module_width > 16 {
            bail!("module-width must be in 1..=16");
        }
        if self.search_seeds.len() != 16
            || self.episode_seeds.len() != 16
            || self.ticks_per_stage == 0
        {
            bail!("the adversarial pilot gate requires exactly 16 search and admission seeds");
        }
        if self
            .search_seeds
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != 16
            || self
                .episode_seeds
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != 16
        {
            bail!("search and admission seed suites must each contain 16 unique seeds");
        }
        if self
            .search_seeds
            .iter()
            .any(|seed| self.episode_seeds.contains(seed))
        {
            bail!("search and admission seed suites must be disjoint");
        }
        if self.pass_fraction != 14.0 / 16.0 || self.predecessor_fail_max_fraction != 2.0 / 16.0 {
            bail!("the adversarial pilot gate is fixed at pass >=14/16 and fail <=2/16");
        }
        let minimum_stage_energy = self.food_energy / self.max_depth as f32;
        if self.world_width < 9
            || !self.food_energy.is_normal()
            || self.food_energy <= 0.0
            || !minimum_stage_energy.is_normal()
            || minimum_stage_energy <= 0.0
        {
            bail!(
                "world-width must be >= 9 and food-energy/max-depth must remain finite, normal, and positive"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceMotion {
    Static,
    /// Every third tick, drift one cell from the resource's current position
    /// in a direction coupled to the organism's current facing.
    FacingCoupledLeftDriftEveryThreeTicks,
    FacingCoupledRightDriftEveryThreeTicks,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct EcologyStage {
    /// Signed 60-degree turns from the organism's facing at release time.
    pub relative_turns: i8,
    pub distance: u8,
    pub motion: ResourceMotion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcologyProgram {
    pub stages: Vec<EcologyStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BehaviorStep {
    pub tick: u64,
    pub active_stage: usize,
    pub action: ActionType,
    pub q: i32,
    pub r: i32,
    pub facing: FacingDirection,
    pub food_q: Option<i32>,
    pub food_r: Option<i32>,
    pub energy: f32,
    pub stage_completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpisodeEvidence {
    pub seed: u64,
    /// Deterministic seed-conditioned reflection/distance perturbations used
    /// in this episode. The abstract task program remains unchanged.
    pub resolved_stages: Vec<EcologyStage>,
    pub completed_stages: usize,
    pub completion_ticks: Vec<u64>,
    pub plant_consumptions: u64,
    pub captured_energy: f32,
    pub standing_food_energy: f32,
    /// Constant across task depths: deeper programs split this escrow across
    /// more chronological releases instead of minting more payoff.
    pub fixed_episode_energy_escrow: f32,
    pub released_task_energy: f32,
    pub unreleased_energy_escrow: f32,
    pub resource_energy_closure_error: f32,
    pub initial_organism_energy: f32,
    pub final_organism_energy: f32,
    pub organism_energy_closure_error: f32,
    pub task_energy_residual_tolerance: f32,
    pub organism_transfer_residual_tolerance: f32,
    /// Largest absolute residual emitted by sim-core's independent fail-closed
    /// per-tick physical energy ledger during this episode.
    pub max_engine_energy_ledger_residual: f64,
    pub max_engine_energy_ledger_tolerance: f64,
    pub steps: Vec<BehaviorStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgramEvaluation {
    /// Success rate for each chronological task prefix under its own deadline.
    pub prefix_success_counts: Vec<usize>,
    pub prefix_success_rates: Vec<f64>,
    pub full_success_count: usize,
    pub full_success_rate: f64,
    pub mean_completed_fraction: f64,
    pub mean_captured_energy: f64,
    pub min_prefix_success_rate: f64,
    pub episodes: Vec<EpisodeEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationEvidence {
    pub generation: u32,
    pub best_full_success_rate: f64,
    pub best_min_prefix_success_rate: f64,
    pub best_mean_completed_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleEvidence {
    pub hidden_node_ids: Vec<GeneNodeId>,
    pub connection_innovations: Vec<InnovationId>,
    pub knockout_restores_predecessor_exactly: bool,
    pub zero_extension_matches_predecessor_behavior: bool,
    pub protected_seed_is_exact_predecessor: bool,
    pub protected_projection_verified: bool,
    pub protected_knockout_matches_module_knockout: bool,
    pub candidate_materializes_exactly: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverTaskEvaluation {
    pub solver_depth: u32,
    pub task_depth: u32,
    pub evaluation: ProgramEvaluation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGenerationCandidateEvaluation {
    pub stage: EcologyStage,
    pub checkpoint_evaluations: Vec<SolverTaskEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryMatrixRow {
    pub solver_depth: u32,
    pub tasks: Vec<SolverTaskEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthEvidence {
    pub depth: u32,
    pub generated_stage: EcologyStage,
    pub task_generation_candidates_checked: usize,
    pub predecessor_evaluation: ProgramEvaluation,
    /// Complete task-generator ranking evidence on the mutable search
    /// contexts, including all grammar candidates rather than only the winner.
    pub task_generation_search_evaluations: Vec<TaskGenerationCandidateEvaluation>,
    /// The proposed task evaluated against every checkpoint available before
    /// this depth on the sealed admission contexts. Every row must be <=2/16.
    pub historical_solver_novelty_evaluations: Vec<SolverTaskEvaluation>,
    pub generations: Vec<GenerationEvidence>,
    pub candidate_evaluation: Option<ProgramEvaluation>,
    pub candidate_genome: Option<OrganismGenome>,
    /// The new solver evaluated independently on every archived task prefix.
    /// Admission requires every cell to be >=14/16.
    pub candidate_archive_evaluations: Vec<SolverTaskEvaluation>,
    pub knockout_evaluation: Option<ProgramEvaluation>,
    pub predecessor_failed_new_task_gate: bool,
    pub all_historical_solvers_failed_new_task_gate: bool,
    pub predecessor_retention_gate: bool,
    pub candidate_retention_gate: bool,
    pub causal_knockout_gate: bool,
    pub module_integrity_gate: bool,
    pub accepted: bool,
    pub module: ModuleEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPlayResult {
    pub result_schema_version: u32,
    pub algorithm: String,
    pub claim_scope: String,
    pub limitations: Vec<String>,
    pub config: PowerPlayConfig,
    pub accepted_depth: u32,
    pub program: EcologyProgram,
    pub depths: Vec<DepthEvidence>,
    /// Final all-history solver x task matrix, including the depth-0 ancestor.
    pub all_history_matrix: Vec<HistoryMatrixRow>,
    pub champion_genome: OrganismGenome,
    pub champion_materializes_exactly: bool,
    pub stopped_reason: Option<String>,
}

#[derive(Clone)]
struct Candidate {
    genome: OrganismGenome,
    evaluation: ProgramEvaluation,
}

#[derive(Clone)]
struct ModuleSpec {
    hidden_node_ids: Vec<GeneNodeId>,
    connection_innovations: Vec<InnovationId>,
}

/// Run the bounded depth-1..4 PowerPlay pilot.
pub fn run_powerplay(
    mut base_world: WorldConfig,
    config: PowerPlayConfig,
) -> Result<PowerPlayResult> {
    config.validate()?;
    let search_config = selection_config(&config);
    configure_task_world(&mut base_world, &config);

    let mut seed_config = base_world.seed_genome_config.clone();
    seed_config.num_neurons = 0;
    seed_config.num_synapses = 0;
    seed_config.vision_distance = 5;
    seed_config.hebb_eta_gain = 0.0;
    let mut founder_rng = ChaCha8Rng::seed_from_u64(config.run_seed ^ TASK_DOMAIN);
    let mut champion = generate_seed_genome(&seed_config, false, &mut founder_rng);
    // An intentionally incapable ancestor makes the first task admission
    // falsifiable: absent a residual module, Idle wins by a wide margin.
    champion.brain.action_biases.fill(-4.0);

    // Canonicalize external-intake clamps once so the durable checkpoint is
    // exactly the genotype that Simulation expresses in every episode.
    champion = materialize_genome(&base_world, &champion, config.run_seed)?;

    let mut program = EcologyProgram { stages: Vec::new() };
    let mut depths = Vec::new();
    let mut stopped_reason = None;
    let mut checkpoints = vec![(0_u32, champion.clone())];

    for depth in 1..=config.max_depth {
        let previous_program = program.clone();
        let predecessor_old = if previous_program.stages.is_empty() {
            None
        } else {
            Some(evaluate_program(
                &base_world,
                &champion,
                &previous_program,
                &config,
                false,
            )?)
        };
        let predecessor_retention_gate = predecessor_old
            .as_ref()
            .is_none_or(|eval| passes_all_prefixes(eval, config.pass_fraction));
        if !predecessor_retention_gate {
            stopped_reason = Some(format!(
                "accepted solver failed predecessor retention before depth {depth}"
            ));
            break;
        }

        let (stage, task_generation_search_evaluations, candidates_checked) =
            match generate_failed_task(&base_world, &checkpoints, &program, &search_config)? {
                Some(generated) => generated,
                None => {
                    stopped_reason = Some(format!(
                        "task grammar exhausted: predecessor passed every generated extension at depth {depth}"
                    ));
                    break;
                }
            };
        program.stages.push(stage);
        let historical_solver_novelty_evaluations =
            evaluate_checkpoints_on_task(&base_world, &checkpoints, &program, &config, false)?;
        let predecessor_evaluation = historical_solver_novelty_evaluations
            .last()
            .map(|entry| entry.evaluation.clone())
            .ok_or_else(|| anyhow!("task generation returned no checkpoint evaluations"))?;
        let predecessor_failed_new_task_gate =
            predecessor_evaluation.full_success_rate <= config.predecessor_fail_max_fraction;
        let all_historical_solvers_failed_new_task_gate = historical_solver_novelty_evaluations
            .iter()
            .all(|entry| entry.evaluation.full_success_count <= 2);

        let protection = ProtectedResidual::seal(&champion).map_err(|error| {
            anyhow!("cannot seal protected controller at depth {depth}: {error}")
        })?;
        let protected_seed = protection.seed_extension();
        let protected_seed_is_exact_predecessor = protected_seed == champion;
        let (mut zero_extended, module) =
            add_zero_residual_module(&protected_seed, depth, &config)?;
        protection.project(&mut zero_extended);
        protection.verify(&zero_extended).map_err(|error| {
            anyhow!("zero residual violates protected controller at depth {depth}: {error}")
        })?;
        let knockout = knockout_module(&zero_extended, &module);
        let protected_knockout = protection.knockout_extension();
        let protected_knockout_matches_module_knockout = protected_knockout == knockout;
        let knockout_restores_predecessor_exactly =
            knockout == champion && protected_knockout == champion;
        let zero_eval = evaluate_program(&base_world, &zero_extended, &program, &config, true)?;
        let predecessor_behavior_eval =
            evaluate_program(&base_world, &champion, &program, &config, true)?;
        let zero_extension_matches_predecessor_behavior =
            same_behavior(&zero_eval, &predecessor_behavior_eval);
        let (candidate, generations) = search_module(
            &base_world,
            &zero_extended,
            &module,
            &protection,
            &program,
            depth,
            &search_config,
        )?;

        let mut candidate_evaluation = None;
        let mut candidate_genome = None;
        let mut candidate_archive_evaluations = Vec::new();
        let mut knockout_evaluation = None;
        let mut candidate_retention_gate = false;
        let mut causal_knockout_gate = false;
        let mut candidate_materializes_exactly = false;
        let mut module_integrity_gate = false;
        let accepted = if let Some(candidate) = candidate {
            protection.verify(&candidate.genome).map_err(|error| {
                anyhow!("candidate violates protected controller at depth {depth}: {error}")
            })?;
            candidate_materializes_exactly =
                materialize_genome(&base_world, &candidate.genome, config.run_seed)?
                    == candidate.genome;
            candidate_archive_evaluations = evaluate_solver_on_task_archive(
                &base_world,
                depth,
                &candidate.genome,
                &program,
                &config,
                false,
            )?;
            candidate_retention_gate = candidate_archive_evaluations
                .iter()
                .all(|entry| entry.evaluation.full_success_count >= 14);
            let sealed_candidate_eval =
                evaluate_program(&base_world, &candidate.genome, &program, &config, true)?;
            let ablated = knockout_module(&candidate.genome, &module);
            let ablated_eval = evaluate_program(&base_world, &ablated, &program, &config, true)?;
            causal_knockout_gate = knockout_restores_predecessor_exactly
                && ablated_eval.full_success_count <= 2
                && sealed_candidate_eval.full_success_count >= 14;
            module_integrity_gate = protected_seed_is_exact_predecessor
                && protected_knockout_matches_module_knockout
                && knockout_restores_predecessor_exactly
                && zero_extension_matches_predecessor_behavior
                && candidate_materializes_exactly;
            candidate_evaluation = Some(sealed_candidate_eval);
            candidate_genome = Some(candidate.genome.clone());
            knockout_evaluation = Some(ablated_eval);
            let accepted = predecessor_failed_new_task_gate
                && all_historical_solvers_failed_new_task_gate
                && predecessor_retention_gate
                && candidate_retention_gate
                && causal_knockout_gate
                && module_integrity_gate;
            if accepted {
                champion = candidate.genome;
                checkpoints.push((depth, champion.clone()));
            }
            accepted
        } else {
            false
        };

        depths.push(DepthEvidence {
            depth,
            generated_stage: stage,
            task_generation_candidates_checked: candidates_checked,
            predecessor_evaluation,
            task_generation_search_evaluations,
            historical_solver_novelty_evaluations,
            generations,
            candidate_evaluation,
            candidate_genome,
            candidate_archive_evaluations,
            knockout_evaluation,
            predecessor_failed_new_task_gate,
            all_historical_solvers_failed_new_task_gate,
            predecessor_retention_gate,
            candidate_retention_gate,
            causal_knockout_gate,
            module_integrity_gate,
            accepted,
            module: ModuleEvidence {
                hidden_node_ids: module.hidden_node_ids.clone(),
                connection_innovations: module.connection_innovations.clone(),
                knockout_restores_predecessor_exactly,
                zero_extension_matches_predecessor_behavior,
                protected_seed_is_exact_predecessor,
                protected_projection_verified: true,
                protected_knockout_matches_module_knockout,
                candidate_materializes_exactly,
            },
        });

        if !accepted {
            program = previous_program;
            stopped_reason = Some(format!(
                "solver search failed admission or causal gate at depth {depth}"
            ));
            break;
        }
    }

    let all_history_matrix = build_history_matrix(&base_world, &checkpoints, &program, &config)?;
    let champion_materializes_exactly =
        materialize_genome(&base_world, &champion, config.run_seed)? == champion;
    Ok(PowerPlayResult {
        result_schema_version: RESULT_SCHEMA_VERSION,
        algorithm: "bounded_causal_ecology_powerplay_pilot".to_string(),
        claim_scope: "bounded depth-1..4 vertical slice; not evidence of open-endedness"
            .to_string(),
        limitations: vec![
            "the task grammar and maximum depth are finite".to_string(),
            "resources are visible sequential targets, so the pilot does not require delayed conditional memory".to_string(),
            "captured energy is physically transferred and conserved but action/metabolic costs are zero, so payoff is not a survival gate".to_string(),
            "the pilot validates infrastructure and bounded checkpoint separation, not sustained tail novelty".to_string(),
        ],
        config,
        accepted_depth: program.stages.len() as u32,
        program,
        depths,
        all_history_matrix,
        champion_genome: champion,
        champion_materializes_exactly,
        stopped_reason,
    })
}

fn configure_task_world(world: &mut WorldConfig, config: &PowerPlayConfig) {
    world.world_width = config.world_width;
    world.num_organisms = 1;
    world.food_energy = config.food_energy;
    world.food_tile_fraction = 0.0;
    world.food_regrowth_interval = 1;
    world.food_regrowth_jitter = 0;
    world.terrain_threshold = 1.0;
    world.passive_metabolism_cost_per_unit = 0.0;
    world.body_mass_metabolic_cost_coeff = 0.0;
    world.move_action_energy_cost = 0.0;
    world.action_temperature = 0.05;
    world.intent_parallel_threads = 1;
    world.runtime_plasticity_enabled = false;
    world.leaky_neurons_enabled = false;
    world.predation_enabled = false;
    world.force_random_actions = false;
}

fn selection_config(config: &PowerPlayConfig) -> PowerPlayConfig {
    let mut selection = config.clone();
    selection.episode_seeds.clone_from(&config.search_seeds);
    // Search is deliberately stricter than the sealed 14/16 admission gate;
    // this reduces the chance that the single predeclared candidate fails its
    // one-shot admission audit. Audit failure still stops the run.
    selection.pass_fraction = 1.0;
    selection.predecessor_fail_max_fraction = 0.0;
    selection
}

fn materialize_genome(
    world: &WorldConfig,
    genome: &OrganismGenome,
    seed: u64,
) -> Result<OrganismGenome> {
    let sim = Simulation::new_with_champion_pool(world.clone(), seed, vec![genome.clone()])
        .map_err(|error| anyhow!("powerplay genome materialization failed: {error}"))?;
    sim.organisms()
        .first()
        .map(|organism| organism.genome.clone())
        .ok_or_else(|| anyhow!("powerplay genome materialization spawned no founder"))
}

fn same_behavior(left: &ProgramEvaluation, right: &ProgramEvaluation) -> bool {
    left == right
}

fn evaluate_solver_on_task_archive(
    world: &WorldConfig,
    solver_depth: u32,
    genome: &OrganismGenome,
    program: &EcologyProgram,
    config: &PowerPlayConfig,
    capture_steps: bool,
) -> Result<Vec<SolverTaskEvaluation>> {
    let mut evaluations = Vec::with_capacity(program.stages.len());
    for task_depth in 1..=program.stages.len() {
        let task = EcologyProgram {
            stages: program.stages[..task_depth].to_vec(),
        };
        evaluations.push(SolverTaskEvaluation {
            solver_depth,
            task_depth: task_depth as u32,
            evaluation: evaluate_program(world, genome, &task, config, capture_steps)?,
        });
    }
    Ok(evaluations)
}

fn evaluate_checkpoints_on_task(
    world: &WorldConfig,
    checkpoints: &[(u32, OrganismGenome)],
    task: &EcologyProgram,
    config: &PowerPlayConfig,
    capture_steps: bool,
) -> Result<Vec<SolverTaskEvaluation>> {
    checkpoints
        .iter()
        .map(|(solver_depth, genome)| {
            Ok(SolverTaskEvaluation {
                solver_depth: *solver_depth,
                task_depth: task.stages.len() as u32,
                evaluation: evaluate_program(world, genome, task, config, capture_steps)?,
            })
        })
        .collect()
}

fn build_history_matrix(
    world: &WorldConfig,
    checkpoints: &[(u32, OrganismGenome)],
    program: &EcologyProgram,
    config: &PowerPlayConfig,
) -> Result<Vec<HistoryMatrixRow>> {
    checkpoints
        .iter()
        .map(|(solver_depth, genome)| {
            Ok(HistoryMatrixRow {
                solver_depth: *solver_depth,
                tasks: evaluate_solver_on_task_archive(
                    world,
                    *solver_depth,
                    genome,
                    program,
                    config,
                    false,
                )?,
            })
        })
        .collect()
}

fn grammar() -> Vec<EcologyStage> {
    let mut stages = Vec::new();
    for motion in [
        ResourceMotion::Static,
        ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks,
        ResourceMotion::FacingCoupledRightDriftEveryThreeTicks,
    ] {
        for distance in 1..=3 {
            for relative_turns in [-2, -1, 0, 1, 2, 3] {
                stages.push(EcologyStage {
                    relative_turns,
                    distance,
                    motion,
                });
            }
        }
    }
    stages
}

fn generate_failed_task(
    world: &WorldConfig,
    checkpoints: &[(u32, OrganismGenome)],
    program: &EcologyProgram,
    config: &PowerPlayConfig,
) -> Result<Option<(EcologyStage, Vec<TaskGenerationCandidateEvaluation>, usize)>> {
    let mut best_index = None;
    let mut evidence: Vec<TaskGenerationCandidateEvaluation> = Vec::new();
    let candidates = grammar();
    for stage in &candidates {
        let mut proposed = program.clone();
        proposed.stages.push(*stage);
        let evaluations = checkpoints
            .iter()
            .map(|(solver_depth, genome)| {
                Ok(SolverTaskEvaluation {
                    solver_depth: *solver_depth,
                    task_depth: proposed.stages.len() as u32,
                    evaluation: evaluate_program(world, genome, &proposed, config, false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let max_success = evaluations
            .iter()
            .map(|entry| entry.evaluation.full_success_rate)
            .fold(0.0_f64, f64::max);
        let mean_partial = evaluations
            .iter()
            .map(|entry| entry.evaluation.mean_completed_fraction)
            .sum::<f64>()
            / evaluations.len() as f64;
        let ordering = best_index.map_or(Ordering::Less, |index: usize| {
            let incumbent = &evidence[index].checkpoint_evaluations;
            let incumbent_max = incumbent
                .iter()
                .map(|entry| entry.evaluation.full_success_rate)
                .fold(0.0_f64, f64::max);
            let incumbent_partial = incumbent
                .iter()
                .map(|entry| entry.evaluation.mean_completed_fraction)
                .sum::<f64>()
                / incumbent.len() as f64;
            max_success
                .total_cmp(&incumbent_max)
                .then_with(|| mean_partial.total_cmp(&incumbent_partial))
        });
        if ordering == Ordering::Less {
            best_index = Some(evidence.len());
        }
        evidence.push(TaskGenerationCandidateEvaluation {
            stage: *stage,
            checkpoint_evaluations: evaluations,
        });
    }
    Ok(best_index.and_then(|index| {
        let selected = &evidence[index];
        selected
            .checkpoint_evaluations
            .iter()
            .all(|entry| entry.evaluation.full_success_rate <= config.predecessor_fail_max_fraction)
            .then_some((selected.stage, evidence, candidates.len()))
    }))
}

fn add_zero_residual_module(
    predecessor: &OrganismGenome,
    depth: u32,
    config: &PowerPlayConfig,
) -> Result<(OrganismGenome, ModuleSpec)> {
    let mut genome = predecessor.clone();
    let mut hidden_node_ids = Vec::with_capacity(config.module_width);
    for slot in 0..config.module_width {
        let index = 10_000_u32
            .checked_add(depth.saturating_mul(32))
            .and_then(|value| value.checked_add(slot as u32))
            .ok_or_else(|| anyhow!("module node id overflow"))?;
        let id = seed_hidden_gene_node_id(index);
        if genome.brain.hidden_nodes.iter().any(|node| node.id == id) {
            bail!("module hidden node id collision at depth {depth}");
        }
        genome.brain.hidden_nodes.push(HiddenNodeGene {
            id,
            bias: 0.0,
            log_time_constant: 0.0,
        });
        hidden_node_ids.push(id);
    }
    genome
        .brain
        .hidden_nodes
        .sort_unstable_by_key(|node| node.id);

    let mut connection_innovations = Vec::new();
    let mut add_edge = |pre_node_id: GeneNodeId, post_node_id: GeneNodeId| -> Result<()> {
        if genome
            .brain
            .edges
            .iter()
            .any(|edge| edge.pre_node_id == pre_node_id && edge.post_node_id == post_node_id)
        {
            bail!("residual module attempted duplicate connection endpoint");
        }
        let innovation = connection_innovation_id(pre_node_id, post_node_id);
        genome.brain.edges.push(SynapseGene {
            innovation,
            pre_node_id,
            post_node_id,
            weight: 0.0,
            // External intake clamps an enabled zero weight to +0.001.
            // Disabled edges therefore provide the exact no-op extension;
            // search candidates explicitly enable them below.
            enabled: false,
        });
        connection_innovations.push(innovation);
        Ok(())
    };
    for &hidden in &hidden_node_ids {
        // Food rays + contact only. Excluding the Energy receptor is crucial:
        // fixed escrow means each stage's share shrinks as a program grows,
        // and a solver that sensed that share could see an old prefix change
        // merely because a later stage was appended.
        for sensor in 0..4 {
            add_edge(sensory_gene_node_id(sensor), hidden)?;
        }
    }
    for &pre in &hidden_node_ids {
        for &post in &hidden_node_ids {
            add_edge(pre, post)?;
        }
    }
    // Attack is disabled in this pilot, so only the four physical foraging
    // actions receive residual output edges.
    for &hidden in &hidden_node_ids {
        for action in 0..4 {
            add_edge(hidden, action_gene_node_id(action))?;
        }
    }
    genome
        .brain
        .edges
        .sort_unstable_by_key(|edge| edge.innovation);
    connection_innovations.sort_unstable();
    Ok((
        genome,
        ModuleSpec {
            hidden_node_ids,
            connection_innovations,
        },
    ))
}

fn knockout_module(genome: &OrganismGenome, module: &ModuleSpec) -> OrganismGenome {
    let mut knockout = genome.clone();
    knockout
        .brain
        .hidden_nodes
        .retain(|node| !module.hidden_node_ids.contains(&node.id));
    knockout.brain.edges.retain(|edge| {
        !module.hidden_node_ids.contains(&edge.pre_node_id)
            && !module.hidden_node_ids.contains(&edge.post_node_id)
    });
    knockout
}

fn randomize_module(genome: &mut OrganismGenome, module: &ModuleSpec, rng: &mut ChaCha8Rng) {
    for node in &mut genome.brain.hidden_nodes {
        if module.hidden_node_ids.contains(&node.id) {
            node.bias = rng.random_range(-1.0..=1.0);
        }
    }
    for edge in &mut genome.brain.edges {
        if module.connection_innovations.contains(&edge.innovation) {
            edge.enabled = true;
            edge.weight = nonzero_weight(rng.random_range(-1.5..=1.5));
        }
    }
}

fn mutate_module(
    genome: &mut OrganismGenome,
    module: &ModuleSpec,
    sigma: f32,
    rng: &mut ChaCha8Rng,
) {
    for node in &mut genome.brain.hidden_nodes {
        if module.hidden_node_ids.contains(&node.id) && rng.random_bool(0.45) {
            let delta: f32 = StandardNormal.sample(rng);
            node.bias = (node.bias + delta * sigma).clamp(-1.0, 1.0);
        }
    }
    for edge in &mut genome.brain.edges {
        if module.connection_innovations.contains(&edge.innovation) && rng.random_bool(0.35) {
            if rng.random_bool(0.05) {
                edge.weight = nonzero_weight(rng.random_range(-1.5..=1.5));
            } else {
                let delta: f32 = StandardNormal.sample(rng);
                edge.weight = nonzero_weight((edge.weight + delta * sigma).clamp(-1.5, 1.5));
            }
        }
    }
}

fn nonzero_weight(weight: f32) -> f32 {
    if weight.abs() < 0.001 {
        if weight.is_sign_negative() {
            -0.001
        } else {
            0.001
        }
    } else {
        weight
    }
}

fn search_module(
    world: &WorldConfig,
    zero_extended: &OrganismGenome,
    module: &ModuleSpec,
    protection: &ProtectedResidual,
    program: &EcologyProgram,
    depth: u32,
    config: &PowerPlayConfig,
) -> Result<(Option<Candidate>, Vec<GenerationEvidence>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(
        config.run_seed ^ TASK_DOMAIN ^ u64::from(depth).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    let mut genomes = Vec::with_capacity(config.population_size);
    genomes.push(zero_extended.clone());
    while genomes.len() < config.population_size {
        let mut genome = zero_extended.clone();
        randomize_module(&mut genome, module, &mut rng);
        protection.project(&mut genome);
        protection
            .verify(&genome)
            .map_err(|error| anyhow!("randomized residual violates protection: {error}"))?;
        genomes.push(genome);
    }

    let mut history = Vec::new();
    let elite_count = (config.population_size / 8).max(2);
    for generation in 0..config.generations_per_depth {
        let mut population = genomes
            .into_iter()
            .map(|genome| {
                let evaluation = evaluate_program(world, &genome, program, config, false)?;
                Ok(Candidate { genome, evaluation })
            })
            .collect::<Result<Vec<_>>>()?;
        population.sort_by(candidate_ordering);
        let best = &population[0];
        history.push(GenerationEvidence {
            generation,
            best_full_success_rate: best.evaluation.full_success_rate,
            best_min_prefix_success_rate: best.evaluation.min_prefix_success_rate,
            best_mean_completed_fraction: best.evaluation.mean_completed_fraction,
        });
        if passes_all_prefixes(&best.evaluation, config.pass_fraction) {
            return Ok((Some(best.clone()), history));
        }

        let elites = population.into_iter().take(elite_count).collect::<Vec<_>>();
        genomes = elites
            .iter()
            .map(|candidate| candidate.genome.clone())
            .collect();
        let progress = generation as f32 / config.generations_per_depth.max(1) as f32;
        let sigma = 1.25 * (1.0 - progress) + 0.15;
        while genomes.len() < config.population_size {
            let parent = &elites[rng.random_range(0..elites.len())];
            let mut child = parent.genome.clone();
            mutate_module(&mut child, module, sigma, &mut rng);
            protection.project(&mut child);
            protection
                .verify(&child)
                .map_err(|error| anyhow!("mutated residual violates protection: {error}"))?;
            genomes.push(child);
        }
    }
    Ok((None, history))
}

fn candidate_ordering(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .evaluation
        .min_prefix_success_rate
        .total_cmp(&left.evaluation.min_prefix_success_rate)
        .then_with(|| {
            right
                .evaluation
                .full_success_rate
                .total_cmp(&left.evaluation.full_success_rate)
        })
        .then_with(|| {
            right
                .evaluation
                .mean_completed_fraction
                .total_cmp(&left.evaluation.mean_completed_fraction)
        })
}

fn passes_all_prefixes(evaluation: &ProgramEvaluation, threshold: f64) -> bool {
    !evaluation.prefix_success_rates.is_empty()
        && evaluation
            .prefix_success_rates
            .iter()
            .all(|rate| *rate >= threshold)
}

fn evaluate_program(
    world: &WorldConfig,
    genome: &OrganismGenome,
    program: &EcologyProgram,
    config: &PowerPlayConfig,
    capture_steps: bool,
) -> Result<ProgramEvaluation> {
    if program.stages.is_empty() {
        bail!("cannot evaluate an empty ecology program");
    }
    let episodes = config
        .episode_seeds
        .iter()
        .copied()
        .map(|seed| run_episode(world, genome, program, config, seed, capture_steps))
        .collect::<Result<Vec<_>>>()?;
    let mut prefix_success_counts = Vec::with_capacity(program.stages.len());
    let mut prefix_success_rates = Vec::with_capacity(program.stages.len());
    for prefix in 1..=program.stages.len() {
        let deadline = config.ticks_per_stage.saturating_mul(prefix as u64);
        let successes = episodes
            .iter()
            .filter(|episode| {
                episode
                    .completion_ticks
                    .get(prefix - 1)
                    .is_some_and(|tick| *tick <= deadline)
            })
            .count();
        prefix_success_counts.push(successes);
        prefix_success_rates.push(successes as f64 / episodes.len() as f64);
    }
    let full_success_count = *prefix_success_counts.last().unwrap_or(&0);
    let full_success_rate = *prefix_success_rates.last().unwrap_or(&0.0);
    let mean_completed_fraction = episodes
        .iter()
        .map(|episode| episode.completed_stages as f64 / program.stages.len() as f64)
        .sum::<f64>()
        / episodes.len() as f64;
    let mean_captured_energy = episodes
        .iter()
        .map(|episode| f64::from(episode.captured_energy))
        .sum::<f64>()
        / episodes.len() as f64;
    let min_prefix_success_rate = prefix_success_rates.iter().copied().fold(1.0_f64, f64::min);
    Ok(ProgramEvaluation {
        prefix_success_counts,
        prefix_success_rates,
        full_success_count,
        full_success_rate,
        mean_completed_fraction,
        mean_captured_energy,
        min_prefix_success_rate,
        episodes,
    })
}

fn run_episode(
    world: &WorldConfig,
    genome: &OrganismGenome,
    program: &EcologyProgram,
    config: &PowerPlayConfig,
    seed: u64,
    capture_steps: bool,
) -> Result<EpisodeEvidence> {
    let mut sim = Simulation::new_with_champion_pool(world.clone(), seed, vec![genome.clone()])
        .map_err(|error| anyhow!("powerplay episode world construction failed: {error}"))?;
    prepare_episode_state(&mut sim)?;
    let initial_organism_energy = sim.organisms[0].energy;
    let horizon = config
        .ticks_per_stage
        .saturating_mul(program.stages.len() as u64);
    let resolved_stages = program
        .stages
        .iter()
        .copied()
        .enumerate()
        .map(|(index, stage)| resolve_stage_context(stage, seed, index))
        .collect::<Vec<_>>();
    let mut active_stage = 0usize;
    let mut completion_ticks = Vec::new();
    let stage_energy = config.food_energy / program.stages.len() as f32;
    if !stage_energy.is_normal() || stage_energy <= 0.0 {
        bail!("powerplay stage energy must remain finite, normal, and positive");
    }
    let mut task_energy_injected = 0.0_f32;
    let mut max_engine_energy_ledger_residual = 0.0_f64;
    let mut max_engine_energy_ledger_tolerance = 0.0_f64;
    let mut steps = Vec::new();
    spawn_stage_resource(&mut sim, resolved_stages[active_stage], stage_energy)?;
    task_energy_injected += stage_energy;

    for _ in 0..horizon {
        move_active_resource(&mut sim, resolved_stages[active_stage])?;
        let before = sim.metrics().total_plant_consumptions;
        let delta = sim.tick();
        let ledger = delta.metrics.energy_ledger_last_turn;
        for residual in [
            ledger.organism_residual,
            ledger.food_residual,
            ledger.total_residual,
            ledger.transfer_residual,
        ] {
            max_engine_energy_ledger_residual =
                max_engine_energy_ledger_residual.max(residual.abs());
        }
        max_engine_energy_ledger_tolerance =
            max_engine_energy_ledger_tolerance.max(ledger.residual_tolerance);
        let consumed = sim.metrics().total_plant_consumptions > before;
        if consumed {
            completion_ticks.push(delta.turn);
            active_stage += 1;
            if active_stage < program.stages.len() {
                spawn_stage_resource(&mut sim, resolved_stages[active_stage], stage_energy)?;
                task_energy_injected += stage_energy;
            }
        }
        if capture_steps {
            let organism = sim.organisms().first();
            let food = sim.foods().first();
            steps.push(BehaviorStep {
                tick: delta.turn,
                active_stage: active_stage.min(program.stages.len().saturating_sub(1)),
                action: organism.map_or(ActionType::Idle, |value| value.last_action_taken),
                q: organism.map_or(0, |value| value.q),
                r: organism.map_or(0, |value| value.r),
                facing: organism.map_or(FacingDirection::East, |value| value.facing),
                food_q: food.map(|value| value.q),
                food_r: food.map(|value| value.r),
                energy: organism.map_or(0.0, |value| value.energy),
                stage_completed: consumed,
            });
        }
        if active_stage == program.stages.len() {
            break;
        }
    }

    let completed_stages = completion_ticks.len();
    let captured_energy = completed_stages as f32 * stage_energy;
    let standing_food_energy = sim.foods().iter().map(|food| food.energy).sum::<f32>();
    let unreleased_energy_escrow = config.food_energy - task_energy_injected;
    let resource_energy_closure_error =
        config.food_energy - captured_energy - standing_food_energy - unreleased_energy_escrow;
    let final_organism_energy = sim.organisms().first().map_or(0.0, |value| value.energy);
    let organism_energy_closure_error =
        final_organism_energy - initial_organism_energy - captured_energy;
    let task_energy_scale = config.food_energy.abs()
        + task_energy_injected.abs()
        + captured_energy.abs()
        + standing_food_energy.abs()
        + unreleased_energy_escrow.abs();
    let task_energy_residual_tolerance =
        TASK_ENERGY_EPSILON_MULTIPLIER * f32::EPSILON * task_energy_scale.max(1.0);
    let observed_organism_gain = final_organism_energy - initial_organism_energy;
    let organism_transfer_scale = captured_energy.abs() + observed_organism_gain.abs();
    let organism_transfer_residual_tolerance =
        TASK_ENERGY_EPSILON_MULTIPLIER * f32::EPSILON * organism_transfer_scale.max(1.0);
    let energy_evidence = [
        config.food_energy,
        stage_energy,
        task_energy_injected,
        captured_energy,
        standing_food_energy,
        unreleased_energy_escrow,
        initial_organism_energy,
        final_organism_energy,
        resource_energy_closure_error,
        organism_energy_closure_error,
        task_energy_residual_tolerance,
        organism_transfer_residual_tolerance,
    ];
    if energy_evidence.iter().any(|value| !value.is_finite())
        || task_energy_injected < 0.0
        || captured_energy < 0.0
        || standing_food_energy < 0.0
        || unreleased_energy_escrow < -task_energy_residual_tolerance
        || task_energy_injected > config.food_energy + task_energy_residual_tolerance
        || resource_energy_closure_error.abs() > task_energy_residual_tolerance
        || organism_energy_closure_error.abs() > organism_transfer_residual_tolerance
        || (completed_stages > 0 && captured_energy <= 0.0)
        || (completed_stages > 0 && observed_organism_gain <= 0.0)
    {
        bail!(
            "powerplay task energy does not close: escrow={} released={} captured={} standing={} locked={} observed_gain={} resource_residual={} organism_residual={} resource_tolerance={} organism_tolerance={}",
            config.food_energy,
            task_energy_injected,
            captured_energy,
            standing_food_energy,
            unreleased_energy_escrow,
            observed_organism_gain,
            resource_energy_closure_error,
            organism_energy_closure_error,
            task_energy_residual_tolerance,
            organism_transfer_residual_tolerance,
        );
    }
    Ok(EpisodeEvidence {
        seed,
        resolved_stages,
        completed_stages,
        completion_ticks,
        plant_consumptions: sim.metrics().total_plant_consumptions,
        captured_energy,
        standing_food_energy,
        fixed_episode_energy_escrow: config.food_energy,
        released_task_energy: task_energy_injected,
        unreleased_energy_escrow,
        resource_energy_closure_error,
        initial_organism_energy,
        final_organism_energy,
        organism_energy_closure_error,
        task_energy_residual_tolerance,
        organism_transfer_residual_tolerance,
        max_engine_energy_ledger_residual,
        max_engine_energy_ledger_tolerance,
        steps,
    })
}

fn resolve_stage_context(stage: EcologyStage, seed: u64, stage_index: usize) -> EcologyStage {
    let mixed =
        mix64(seed ^ TASK_DOMAIN ^ (stage_index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
    let reflected = mixed & 1 == 1;
    let distance_delta = ((mixed >> 1) % 3) as i16 - 1;
    let distance = (i16::from(stage.distance) + distance_delta).clamp(1, 3) as u8;
    let motion = if reflected {
        match stage.motion {
            ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks => {
                ResourceMotion::FacingCoupledRightDriftEveryThreeTicks
            }
            ResourceMotion::FacingCoupledRightDriftEveryThreeTicks => {
                ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks
            }
            ResourceMotion::Static => ResourceMotion::Static,
        }
    } else {
        stage.motion
    };
    EcologyStage {
        relative_turns: if reflected {
            -stage.relative_turns
        } else {
            stage.relative_turns
        },
        distance,
        motion,
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn prepare_episode_state(sim: &mut Simulation) -> Result<()> {
    if sim.organisms.len() != 1 {
        bail!("powerplay episode requires exactly one founder");
    }
    if sim.terrain_map.iter().any(|blocked| *blocked) {
        bail!("powerplay task world unexpectedly contains blocked terrain");
    }
    sim.foods.clear();
    sim.food_tiles.fill(false);
    sim.food_regrowth_due_turn.fill(u64::MAX);
    sim.food_regrowth_schedule.clear();
    sim.occupancy.fill(None);
    let center = (sim.config.world_width / 2) as i32;
    let organism = &mut sim.organisms[0];
    organism.q = center;
    organism.r = center;
    organism.facing = FacingDirection::East;
    organism.energy_at_last_sensing = organism.energy;
    let idx = center as usize * sim.config.world_width as usize + center as usize;
    sim.occupancy[idx] = Some(Occupant::Organism(organism.id));
    Ok(())
}

fn spawn_stage_resource(sim: &mut Simulation, stage: EcologyStage, energy: f32) -> Result<()> {
    if !energy.is_normal() || energy <= 0.0 {
        bail!("powerplay resource energy must be finite, normal, and positive");
    }
    if !sim.foods.is_empty() {
        bail!("powerplay attempted to release a stage while another resource remained");
    }
    let organism = sim
        .organisms
        .first()
        .ok_or_else(|| anyhow!("powerplay organism died before resource release"))?;
    let direction = rotate_by_steps(organism.facing, stage.relative_turns);
    let mut position = (organism.q, organism.r);
    for _ in 0..stage.distance {
        position = hex_neighbor(position, direction, sim.config.world_width as i32);
    }
    let cell_idx = sim.cell_index(position.0, position.1);
    if sim.occupancy[cell_idx].is_some() {
        bail!("generated task resource target is occupied");
    }
    let id = FoodId(sim.next_food_id);
    sim.next_food_id = sim.next_food_id.saturating_add(1);
    let food = FoodState {
        id,
        q: position.0,
        r: position.1,
        energy,
        kind: FoodKind::Plant,
        visual: sim_types::food_visual(FoodKind::Plant),
    };
    sim.occupancy[cell_idx] = Some(Occupant::Food(id));
    sim.foods.push(food);
    Ok(())
}

fn move_active_resource(sim: &mut Simulation, stage: EcologyStage) -> Result<()> {
    let rotation = match stage.motion {
        ResourceMotion::Static => return Ok(()),
        ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks if sim.turn % 3 == 2 => -1,
        ResourceMotion::FacingCoupledRightDriftEveryThreeTicks if sim.turn % 3 == 2 => 1,
        ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks
        | ResourceMotion::FacingCoupledRightDriftEveryThreeTicks => return Ok(()),
    };
    let Some(food) = sim.foods.first().cloned() else {
        return Ok(());
    };
    let organism = sim
        .organisms
        .first()
        .ok_or_else(|| anyhow!("powerplay organism died while resource remained"))?;
    let old_idx = sim.cell_index(food.q, food.r);
    // This is deliberately controller-coupled drift, not a geometric orbit:
    // turning changes the direction in which the visible target moves.
    let base_direction = rotate_by_steps(organism.facing, stage.relative_turns);
    let target_direction = rotate_by_steps(base_direction, rotation);
    let target = hex_neighbor(
        (food.q, food.r),
        target_direction,
        sim.config.world_width as i32,
    );
    let target_idx = sim.cell_index(target.0, target.1);
    if sim.occupancy[target_idx].is_some() {
        return Ok(());
    }
    sim.occupancy[old_idx] = None;
    sim.occupancy[target_idx] = Some(Occupant::Food(food.id));
    sim.foods[0].q = target.0;
    sim.foods[0].r = target.1;
    Ok(())
}
