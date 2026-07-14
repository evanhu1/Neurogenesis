use crate::{run_output_path, DEFAULT_CONFIG, REPORT_EVERY};
use anyhow::{anyhow, bail, Result};
use evolution::{
    evaluate_frozen_pair, evaluate_frozen_panel, run_neat, FitnessObjective, NeatConfig, RunResult,
    ScenarioManifest, ScenarioPreset, SelectionStrategy,
};
use ring::digest::{digest, SHA256};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use types::OrganismGenome;
use views::{
    atomic_write, save_sidecar, save_world, sibling_metrics_path, start_recording,
    world_config_with_overrides,
};
use world_sim::Simulation;

const PARAMS: &str = "compatibility_threshold excess_coefficient disjoint_coefficient \
target_species compatibility_threshold_adjustment weight_coefficient survival_fraction \
crossover_probability interspecies_mate_probability \
curriculum_enabled curriculum_promotion_threshold curriculum_promotion_patience \
training_seed_rotation_period objective_cvar_fraction fitness_objective survival_window_weights \
selection_strategy novelty_k novelty_archive_additions_per_generation \
mutate_weight_probability replace_weight_probability weight_perturb_stddev \
per_connection_weight_mutation_probability mutate_bias_probability bias_perturb_stddev \
mutate_time_constant_probability time_constant_perturb_stddev add_connection_probability \
add_node_probability disabled_inheritance_probability stagnation_generations \
young_species_grace_generations min_young_species_offspring \
elitism_min_species_size eval_opponents eval_lineages_per_world cross_pool_predation_only";

/// Canonical generational NEAT runner. It owns no persistent world: each
/// candidate is evaluated in deterministic fixed-seed episodes, and the durable
/// artifacts contain the full effective algorithm/environment contract.
pub(crate) fn run_neat_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    if args.first() == Some(&"analyze") {
        return run_neat_analysis(&args[1..], out);
    }
    if args.first() == Some(&"evaluate-panel") {
        return run_neat_panel_evaluation(&args[1..], out);
    }
    if args.first() == Some(&"crossplay") {
        return run_neat_crossplay(&args[1..], out);
    }
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut run_seed = 0u64;
    let mut scale = None;
    let mut sets = Vec::new();
    let mut neat = NeatConfig::default();
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--config" => {
                config_path = value(args, i, "--config")?.to_string();
                i += 2;
            }
            "--seed" => {
                run_seed = value(args, i, "--seed")?.parse()?;
                i += 2;
            }
            "--population" => {
                neat.population_size = value(args, i, "--population")?.parse()?;
                i += 2;
            }
            "--generations" => {
                neat.generations = value(args, i, "--generations")?.parse()?;
                i += 2;
            }
            "--episode-horizons" => {
                neat.episode_horizons =
                    parse_u64_list(value(args, i, "--episode-horizons")?, "--episode-horizons")?;
                i += 2;
            }
            "--opponents" => {
                neat.eval_opponents = value(args, i, "--opponents")?.parse()?;
                i += 2;
            }
            "--world-seeds" => {
                neat.world_seeds =
                    parse_u64_list(value(args, i, "--world-seeds")?, "--world-seeds")?;
                i += 2;
            }
            "--scenarios" => {
                neat.scenarios = parse_scenarios(value(args, i, "--scenarios")?)?;
                i += 2;
            }
            "--workers" => {
                neat.evaluator_workers = value(args, i, "--workers")?.parse()?;
                i += 2;
            }
            "--scale" => {
                scale = Some(parse_scale(value(args, i, "--scale")?)?);
                i += 2;
            }
            "--no-scale" => {
                scale = None;
                i += 1;
            }
            "--set" => {
                sets.push(parse_assignment(value(args, i, "--set")?)?);
                i += 2;
            }
            "--param" => {
                let (key, value) = parse_assignment(value(args, i, "--param")?)?;
                apply_neat_param(&mut neat, &key, &value)?;
                i += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "neat options: --config P --seed N --population N --generations N \
                     --episode-horizons T[,T...] [--opponents N] --world-seeds N,N \
                     [--scenarios baseline,scarcity,sparse_search] \
                     --workers K --scale W,POP \
                     [--no-scale] [--set world_key=value] [--param neat_key=value]\n\
                     --opponents is the number of contemporary-opponent exposures per candidate;\n\
                     set --param eval_lineages_per_world=3 for simultaneous two-opponent mini-ecosystems;\n\
                     defaults: 5000 ticks, 4 training seeds, 8 pairwise opponent exposures, relative survival objective;\n\
                     default scale: the canonical world config (currently 50,100);\n\
                     analyze: neat analyze RESULT.json [RESULT2.json ...]\n\
                     frozen panel: neat evaluate-panel --focal RESULT.json --opponents RESULT.json,RESULT.json --horizons 500,1000 --world-seeds 131,149\n\
                     checkpoint crossplay: neat crossplay RESULT.json --checkpoints all --horizons 1000,2000,4000 --world-seeds 131,149\n\
                     NEAT params: {PARAMS}"
                )?;
                return Ok(());
            }
            other => bail!("unknown neat arg `{other}` (use `neat --help`)"),
        }
    }
    neat.validate()?;
    let mut world = world_config_with_overrides(&config_path, &sets)?;
    if let Some((width, founders)) = scale {
        world.world_width = width;
        world.num_organisms = founders;
    }

    eprintln!(
        "{}",
        json!({
            "event": "neat_started",
            "population": neat.population_size,
            "generations": neat.generations,
            "episode_horizons": neat.episode_horizons,
            "survival_window_weights": neat.survival_window_weights,
            "world_seeds": neat.world_seeds,
            "workers": neat.evaluator_workers,
            "world_width": world.world_width,
            "founder_cohort_size": world.num_organisms,
            "objective": neat.fitness_objective.name(),
            "fully_connected_initial_topology": true,
            "feed_forward_hidden_graph": true,
            "balanced_pairwise_evaluation": neat.eval_opponents > 0 && neat.eval_lineages_per_world == 2,
            "symmetric_founder_slot_rotation": neat.eval_opponents > 0,
            "eval_opponents": neat.eval_opponents,
            "eval_lineages_per_world": neat.eval_lineages_per_world,
            "cross_pool_predation_only": neat.cross_pool_predation_only,
            "scenarios": neat.scenarios,
        })
    );
    // Keep the base world config so the champion can be materialized into a
    // real, inspectable world once the run finishes.
    let champion_world_config = world.clone();
    let result = run_neat(neat, world, run_seed, |generation| {
        let behavior = json!({
            "best_trophic_role": generation.best_trophic_role,
            "best_action_effectiveness": generation.best_action_effectiveness,
            "best_plant_consumption_rate": generation.best_plant_consumption_rate,
            "best_prey_consumption_rate": generation.best_prey_consumption_rate,
            "best_mi_sa": generation.best_mi_sa,
            "best_learning_slope": generation.best_learning_slope,
            "best_plant_intake_fraction": generation.best_plant_intake_fraction,
            "best_prey_intake_fraction": generation.best_prey_intake_fraction,
            "best_mean_attack_kills": generation.best_mean_attack_kills,
            "mean_action_effectiveness": generation.mean_action_effectiveness,
            "mean_plant_consumption_rate": generation.mean_plant_consumption_rate,
            "mean_prey_consumption_rate": generation.mean_prey_consumption_rate,
            "population_trophic_roles": generation.population_trophic_roles,
        });
        eprintln!(
            "{}",
            json!({
                "event": "neat_generation",
                "generation": generation.generation,
                "curriculum_level": generation.curriculum_level,
                "training_seed_epoch": generation.training_seed_epoch,
                "effective_training_seeds": generation.effective_training_seeds,
                "eval_opponents": generation.eval_opponents,
                "evaluation_cases_per_genome": generation.evaluation_cases_per_genome,
                "evaluation_worlds": generation.evaluation_worlds,
                "best_fitness": generation.best_fitness,
                "mean_fitness": generation.mean_fitness,
                "best_absolute_survival": generation.best_absolute_survival_fraction,
                "best_candidate_alive_ticks": generation.best_candidate_alive_ticks,
                "best_late_weighted_survival": generation.best_late_weighted_survival_fraction,
                "best_relative_advantage": generation.best_relative_survival_advantage,
                "mean_absolute_survival": generation.mean_absolute_survival_fraction,
                "mean_candidate_alive_ticks": generation.mean_candidate_alive_ticks,
                "mean_late_weighted_survival": generation.mean_late_weighted_survival_fraction,
                "mean_relative_advantage": generation.mean_relative_survival_advantage,
                "best_mean_prey_consumptions": generation.best_mean_prey_consumptions,
                "behavior": behavior,
                "checkpoint_genome_persisted": generation.checkpoint_champion_genome.is_some(),
                "selection_strategy": generation.selection_strategy,
                "best_novelty": generation.best_novelty,
                "mean_novelty": generation.mean_novelty,
                "best_local_competition": generation.best_local_competition,
                "mean_local_competition": generation.mean_local_competition,
                "novelty_archive_size": generation.novelty_archive_size,
                "species": generation.species.len(),
                "compatibility_threshold": generation.compatibility_threshold,
                "hidden_nodes": generation.best_hidden_nodes,
                "enabled_connections": generation.best_enabled_connections,
                "expressed_hidden_nodes": generation.best_expressed_hidden_nodes,
                "expressed_connections": generation.best_expressed_connections,
                "new_connection_innovations": generation.new_connection_innovations,
                "new_node_innovations": generation.new_node_innovations,
                "new_origin_offspring_rate": generation.new_origin_offspring_rate,
                "innovations_reaching_ten_percent": generation.connection_innovations_reaching_ten_percent,
                "crossovers": generation.offspring_crossovers,
            })
        );
    })?;

    let result_path = run_output_path(out_dir, "neat")?;
    let result_path_string = result_path.to_string_lossy().into_owned();
    atomic_write(&result_path_string, |writer| {
        serde_json::to_writer_pretty(writer, &result).map_err(Into::into)
    })?;

    // Materialize the champion as a real world.bin (+ minted metric sidecar)
    // seeded as a clonal colony from the champion genome. It is then a first-
    // class world — `run-to`/`pillars`/`inspect`/`brain` all work directly, so
    // there is no separate champion serialization format to re-inject.
    let mut world_path = result_path.clone();
    world_path.set_extension("world.bin");
    let world_path_string = world_path.to_string_lossy().into_owned();
    let champion_world = Simulation::new_with_champion_pool(
        champion_world_config,
        run_seed,
        vec![result.champion_genome.clone()],
    )
    .map_err(|error| anyhow!("building champion world: {error}"))?;
    save_world(&champion_world, &world_path_string)?;
    let recorder = start_recording(&champion_world, REPORT_EVERY);
    let sidecar_path = sibling_metrics_path(&world_path_string);
    save_sidecar(REPORT_EVERY, &recorder, &sidecar_path)?;

    writeln!(
        out,
        "{}",
        json!({
            "wrote": result_path_string,
            "champion_world": world_path_string,
            "objective": result.objective,
            "champion_fitness": result.champion_fitness,
            "champion_generation": result.champion_generation,
            "champion_curriculum_level": result.champion_curriculum_level,
            "champion_training_seed_epoch": result.champion_training_seed_epoch,
            "champion_trophic_role": result.champion_evaluation.trophic_role,
            "champion_action_effectiveness": result.champion_evaluation.mean_action_effectiveness,
            "champion_plant_consumption_rate": result.champion_evaluation.mean_plant_consumption_rate,
            "champion_prey_consumption_rate": result.champion_evaluation.mean_prey_consumption_rate,
            "generations": result.generations.len(),
        })
    )
    .map_err(Into::into)
}

fn run_neat_panel_evaluation(args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut focal_path = None::<String>;
    let mut opponent_paths = Vec::<String>::new();
    let mut horizons = vec![500_u64, 1_000, 2_000, 4_000];
    let mut world_seeds = None::<Vec<u64>>;
    let mut levels = None::<Vec<u32>>;
    let mut objective = FitnessObjective::SurvivalTimesRelativeAdvantage;
    let mut objective_cvar_fraction = 0.5_f64;
    let mut survival_window_weights = vec![1.0_f64];
    let mut cross_pool_predation_only = false;
    let mut focal_pool_index = 0usize;
    let mut i = 0usize;
    while i < args.len() {
        match args[i] {
            "--focal" => {
                focal_path = Some(value(args, i, "--focal")?.to_string());
                i += 2;
            }
            "--opponents" => {
                opponent_paths = value(args, i, "--opponents")?
                    .split(',')
                    .filter(|path| !path.trim().is_empty())
                    .map(|path| path.trim().to_string())
                    .collect();
                i += 2;
            }
            "--horizons" => {
                horizons = parse_u64_list(value(args, i, "--horizons")?, "--horizons")?;
                i += 2;
            }
            "--world-seeds" => {
                world_seeds = Some(parse_u64_list(
                    value(args, i, "--world-seeds")?,
                    "--world-seeds",
                )?);
                i += 2;
            }
            "--levels" => {
                levels = Some(parse_u32_list(value(args, i, "--levels")?, "--levels")?);
                i += 2;
            }
            "--objective" => {
                objective = FitnessObjective::parse(value(args, i, "--objective")?)?;
                i += 2;
            }
            "--cvar" => {
                objective_cvar_fraction = value(args, i, "--cvar")?.parse()?;
                i += 2;
            }
            "--window-weights" => {
                survival_window_weights =
                    parse_f64_list(value(args, i, "--window-weights")?, "--window-weights")?;
                i += 2;
            }
            "--cross-pool-only" => {
                cross_pool_predation_only = true;
                i += 1;
            }
            "--focal-slot" => {
                focal_pool_index = value(args, i, "--focal-slot")?.parse()?;
                i += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "neat evaluate-panel --focal RESULT.json --opponents RESULT.json[,RESULT.json...] [--horizons N,N] [--window-weights W,W] [--world-seeds N,N] [--levels N,N] [--objective survival_fraction|late_weighted_survival|survival_times_relative_advantage] [--cvar F] [--cross-pool-only] [--focal-slot 0|1|2]"
                )?;
                return Ok(());
            }
            other => bail!("unknown neat evaluate-panel arg `{other}`"),
        }
    }
    let focal_path = focal_path.ok_or_else(|| anyhow!("evaluate-panel needs --focal"))?;
    if opponent_paths.is_empty() {
        bail!("evaluate-panel needs at least one --opponents result");
    }
    if focal_pool_index > opponent_paths.len() {
        bail!("evaluate-panel --focal-slot exceeds the realized founder-pool size");
    }
    if horizons.is_empty() || horizons.contains(&0) {
        bail!("evaluate-panel horizons must be nonempty and positive");
    }
    if !(0.0..=1.0).contains(&objective_cvar_fraction) || objective_cvar_fraction == 0.0 {
        bail!("evaluate-panel --cvar must be in (0,1]");
    }
    let focal = read_neat_result(&focal_path)?;
    let opponents = opponent_paths
        .iter()
        .map(|path| read_neat_result(path))
        .collect::<Result<Vec<_>>>()?;
    let opponent_genomes = opponents
        .iter()
        .map(|result| result.champion_genome.clone())
        .collect::<Vec<_>>();
    let world_seeds = world_seeds.unwrap_or_else(|| focal.neat_config.world_seeds.clone());
    let scenarios = focal
        .final_training_scenarios
        .iter()
        .filter(|scenario| {
            levels
                .as_ref()
                .is_none_or(|levels| levels.contains(&scenario.curriculum_level))
        })
        .cloned()
        .collect::<Vec<_>>();
    if scenarios.is_empty() {
        bail!("evaluate-panel selected no scenarios");
    }

    let mut evaluations = Vec::with_capacity(horizons.len());
    for horizon in horizons {
        let evaluation = evaluate_frozen_panel(
            &focal.champion_genome,
            &opponent_genomes,
            &scenarios,
            horizon,
            &survival_window_weights,
            &world_seeds,
            objective_cvar_fraction,
            objective,
            cross_pool_predation_only,
            focal_pool_index,
        )?;
        evaluations.push(json!({
            "horizon": horizon,
            "summary": evaluation.summary,
            "cases": evaluation.cases,
        }));
    }
    writeln!(
        out,
        "{}",
        json!({
            "schema_version": 2,
            "focal": focal_path,
            "focal_run_seed": focal.seed,
            "opponents": opponent_paths,
            "opponent_run_seeds": opponents.iter().map(|result| result.seed).collect::<Vec<_>>(),
            "objective": objective,
            "objective_cvar_fraction": objective_cvar_fraction,
            "survival_window_weights": survival_window_weights,
            "cross_pool_predation_only": cross_pool_predation_only,
            "focal_pool_index": focal_pool_index,
            "world_seeds": world_seeds,
            "curriculum_levels": levels,
            "evaluations": evaluations,
        })
    )
    .map_err(Into::into)
}

fn run_neat_crossplay(args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut result_path = None::<String>;
    let mut selected_generations = None::<Vec<u32>>;
    let mut horizons = None::<Vec<u64>>;
    let mut world_seeds = None::<Vec<u64>>;
    let mut levels = None::<Vec<u32>>;
    let mut scenario_names = None::<Vec<String>>;
    let mut objective = None::<FitnessObjective>;
    let mut objective_cvar_fraction = None::<f64>;
    let mut cross_pool_predation_only = None::<bool>;
    let mut i = 0usize;
    while i < args.len() {
        match args[i] {
            "--checkpoints" => {
                let raw = value(args, i, "--checkpoints")?;
                selected_generations = if raw == "all" {
                    None
                } else {
                    Some(parse_u32_list(raw, "--checkpoints")?)
                };
                i += 2;
            }
            "--horizons" => {
                horizons = Some(parse_u64_list(value(args, i, "--horizons")?, "--horizons")?);
                i += 2;
            }
            "--world-seeds" => {
                world_seeds = Some(parse_u64_list(
                    value(args, i, "--world-seeds")?,
                    "--world-seeds",
                )?);
                i += 2;
            }
            "--levels" => {
                levels = Some(parse_u32_list(value(args, i, "--levels")?, "--levels")?);
                i += 2;
            }
            "--scenarios" => {
                scenario_names = Some(
                    value(args, i, "--scenarios")?
                        .split(',')
                        .filter(|value| !value.trim().is_empty())
                        .map(|value| value.trim().to_string())
                        .collect(),
                );
                i += 2;
            }
            "--objective" => {
                objective = Some(FitnessObjective::parse(value(args, i, "--objective")?)?);
                i += 2;
            }
            "--cvar" => {
                objective_cvar_fraction = Some(value(args, i, "--cvar")?.parse()?);
                i += 2;
            }
            "--cross-pool-only" => {
                cross_pool_predation_only = Some(true);
                i += 1;
            }
            "--allow-same-pool" => {
                cross_pool_predation_only = Some(false);
                i += 1;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "neat crossplay RESULT.json [--checkpoints all|G,G] [--horizons N,N] [--world-seeds N,N] [--levels N,N] [--scenarios baseline,scarcity,sparse_search] [--objective survival_fraction|late_weighted_survival|survival_times_relative_advantage] [--cvar F] [--cross-pool-only|--allow-same-pool]\nEvery matchup uses two distinct genomes and the balanced training evaluator; clone-versus-clone self-play is never emitted. World seeds must be a multiple of four. Evaluation settings default to the source run's training contract."
                )?;
                return Ok(());
            }
            value if !value.starts_with('-') && result_path.is_none() => {
                result_path = Some(value.to_string());
                i += 1;
            }
            other => bail!("unknown neat crossplay arg `{other}`"),
        }
    }
    let result_path = result_path.ok_or_else(|| anyhow!("crossplay needs RESULT.json"))?;
    let source = read_neat_result(&result_path)?;
    let horizons = horizons.unwrap_or_else(|| source.neat_config.episode_horizons.clone());
    let objective = objective.unwrap_or(source.neat_config.fitness_objective);
    let objective_cvar_fraction =
        objective_cvar_fraction.unwrap_or(source.neat_config.objective_cvar_fraction);
    let cross_pool_predation_only =
        cross_pool_predation_only.unwrap_or(source.neat_config.cross_pool_predation_only);
    if horizons.is_empty() || horizons.contains(&0) {
        bail!("crossplay horizons must be nonempty and positive");
    }
    if !(0.0..=1.0).contains(&objective_cvar_fraction) || objective_cvar_fraction == 0.0 {
        bail!("crossplay --cvar must be in (0,1]");
    }
    let world_seeds = world_seeds.unwrap_or_else(|| source.neat_config.world_seeds.clone());
    let scenarios = source
        .final_training_scenarios
        .iter()
        .filter(|scenario| {
            levels
                .as_ref()
                .is_none_or(|levels| levels.contains(&scenario.curriculum_level))
                && scenario_names
                    .as_ref()
                    .is_none_or(|names| names.contains(&scenario.name))
        })
        .cloned()
        .collect::<Vec<_>>();
    if scenarios.is_empty() {
        bail!("crossplay selected no scenarios");
    }

    let mut checkpoints = source
        .generations
        .iter()
        .filter_map(|generation| {
            generation
                .checkpoint_champion_genome
                .as_ref()
                .map(|genome| (generation.generation, genome.clone()))
        })
        .filter(|(generation, _)| {
            selected_generations
                .as_ref()
                .is_none_or(|selected| selected.contains(generation))
        })
        .map(|(generation, genome)| CrossplayCheckpoint {
            generation,
            genome_hash: genome_sha256(&genome),
            genome,
            duplicate_of_generation: None,
        })
        .collect::<Vec<_>>();
    checkpoints.sort_by_key(|checkpoint| checkpoint.generation);
    if checkpoints.len() < 2 {
        bail!("crossplay needs at least two persisted checkpoint genomes");
    }
    let mut first_generation_by_hash = BTreeMap::<String, u32>::new();
    for checkpoint in &mut checkpoints {
        checkpoint.duplicate_of_generation = first_generation_by_hash
            .get(&checkpoint.genome_hash)
            .copied();
        first_generation_by_hash
            .entry(checkpoint.genome_hash.clone())
            .or_insert(checkpoint.generation);
    }
    if first_generation_by_hash.len() < 2 {
        bail!("crossplay needs at least two distinct persisted checkpoint genomes");
    }

    let mut cells = Vec::with_capacity(
        checkpoints
            .len()
            .saturating_mul(checkpoints.len().saturating_sub(1))
            .saturating_mul(horizons.len()),
    );
    for focal in &checkpoints {
        for opponent in &checkpoints {
            // Clone-versus-clone play is not a general competence measurement:
            // carnivorous strategies either prey on their own clone or lose the
            // distinct lineage their ecology depends on. Compare only genuinely
            // distinct genomes, including across different checkpoint numbers.
            if focal.genome_hash == opponent.genome_hash {
                continue;
            }
            for &horizon in &horizons {
                let evaluation = evaluate_frozen_pair(
                    &focal.genome,
                    &opponent.genome,
                    &scenarios,
                    &[horizon],
                    &source.neat_config.survival_window_weights,
                    &world_seeds,
                    objective_cvar_fraction,
                    objective,
                    cross_pool_predation_only,
                )?;
                cells.push(json!({
                    "focal_generation": focal.generation,
                    "focal_genome_hash": focal.genome_hash,
                    "opponent_generation": opponent.generation,
                    "opponent_genome_hash": opponent.genome_hash,
                    "horizon": horizon,
                    "summary": evaluation.left.summary,
                    "cases": evaluation.left.cases,
                    "opponent_summary": evaluation.right.summary,
                    "opponent_cases": evaluation.right.cases,
                }));
            }
        }
    }
    writeln!(
        out,
        "{}",
        json!({
            "schema_version": 3,
            "source": result_path,
            "source_run_seed": source.seed,
            "checkpoints": checkpoints.iter().map(|checkpoint| json!({
                "generation": checkpoint.generation,
                "genome_hash_sha256": checkpoint.genome_hash,
                "duplicate_of_generation": checkpoint.duplicate_of_generation,
            })).collect::<Vec<_>>(),
            "contract": {
                "horizons": horizons,
                "world_seeds": world_seeds,
                "curriculum_levels": levels,
                "scenario_names": scenario_names,
                "founder_pool_size": 2,
                "balanced_slot_rotation": true,
                "distinct_genomes_only": true,
                "objective": objective,
                "objective_cvar_fraction": objective_cvar_fraction,
                "cross_pool_predation_only": cross_pool_predation_only,
            },
            "cells": cells,
        })
    )
    .map_err(Into::into)
}

struct CrossplayCheckpoint {
    generation: u32,
    genome_hash: String,
    genome: OrganismGenome,
    duplicate_of_generation: Option<u32>,
}

fn genome_sha256(genome: &OrganismGenome) -> String {
    let bytes = bincode::serialize(genome).expect("genome serialization must be infallible");
    digest(&SHA256, &bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Deserialize)]
struct PanelResultSource {
    seed: u64,
    champion_genome: OrganismGenome,
    final_training_scenarios: Vec<ScenarioManifest>,
    neat_config: PanelSeedConfig,
    generations: Vec<PanelGenerationSource>,
}

#[derive(Deserialize)]
struct PanelGenerationSource {
    generation: u32,
    #[serde(default)]
    checkpoint_champion_genome: Option<OrganismGenome>,
}

#[derive(Deserialize)]
struct PanelSeedConfig {
    world_seeds: Vec<u64>,
    episode_horizons: Vec<u64>,
    survival_window_weights: Vec<f64>,
    objective_cvar_fraction: f64,
    fitness_objective: FitnessObjective,
    cross_pool_predation_only: bool,
}

fn read_neat_result(path: &str) -> Result<PanelResultSource> {
    let reader = File::open(path).map_err(|error| anyhow!("cannot open `{path}`: {error}"))?;
    serde_json::from_reader(reader)
        .map_err(|error| anyhow!("cannot parse NEAT result `{path}`: {error}"))
}

fn run_neat_analysis(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.is_empty() {
        bail!("neat analyze needs at least one result.json path");
    }
    let mut analyses = Vec::with_capacity(args.len());
    for path in args {
        let reader = File::open(path).map_err(|error| anyhow!("cannot open `{path}`: {error}"))?;
        let result: RunResult = serde_json::from_reader(reader)
            .map_err(|error| anyhow!("cannot parse NEAT result `{path}`: {error}"))?;
        analyses.push(analyze_result(path, &result));
    }
    let value = if analyses.len() == 1 {
        analyses.pop().expect("one analysis")
    } else {
        json!({ "runs": analyses })
    };
    writeln!(out, "{value}").map_err(Into::into)
}

fn analyze_result(path: &str, result: &RunResult) -> serde_json::Value {
    let generations = &result.generations;
    let split = generations.len().saturating_mul(2) / 3;
    let late = &generations[split..];
    let all_generation = generations
        .iter()
        .map(|generation| generation.generation as f64)
        .collect::<Vec<_>>();
    let late_generation = late
        .iter()
        .map(|generation| generation.generation as f64)
        .collect::<Vec<_>>();
    let late_start_generation = late
        .first()
        .map(|generation| generation.generation)
        .unwrap_or(0);
    let late_new_connection_origins = result
        .connection_innovation_history
        .iter()
        .filter(|record| {
            record
                .origin_generation
                .is_some_and(|g| g >= late_start_generation)
        })
        .count();
    let late_new_node_origins = result
        .node_innovation_history
        .iter()
        .filter(|record| record.origin_generation >= late_start_generation)
        .count();
    let adaptive_connections = result
        .connection_innovation_history
        .iter()
        .filter(|record| record.origin_generation.is_some())
        .filter(|record| record.max_expressed_frequency >= 0.10)
        .count();
    let adaptive_nodes = result
        .node_innovation_history
        .iter()
        .filter(|record| record.max_expressed_frequency >= 0.10)
        .count();

    json!({
        "path": path,
        "run_seed": result.seed,
        "generations": generations.len(),
        "complexification_enabled": result.neat_config.add_connection_probability > 0.0
            || result.neat_config.add_node_probability > 0.0,
        "champion": {
            "generation": result.champion_generation,
            "curriculum_level": result.champion_curriculum_level,
            "training_fitness": result.champion_fitness,
            "training_absolute_survival": result.champion_evaluation.mean_absolute_survival_fraction,
            "training_late_weighted_survival": result.champion_evaluation.mean_late_weighted_survival_fraction,
            "training_relative_advantage": result.champion_evaluation.mean_relative_survival_advantage,
            "training_cvar_absolute_survival": result.champion_evaluation.objective_cvar_absolute_survival_fraction,
            "training_cvar_late_weighted_survival": result.champion_evaluation.objective_cvar_late_weighted_survival_fraction,
            "training_cvar_relative_advantage": result.champion_evaluation.objective_cvar_relative_survival_advantage,
            "pair_seed_cases": result.champion_evaluation.pair_seed_cases,
            "unique_opponents": result.champion_evaluation.unique_opponents,
            "trophic_role": result.champion_evaluation.trophic_role,
            "action_effectiveness": result.champion_evaluation.mean_action_effectiveness,
            "plant_consumption_rate": result.champion_evaluation.mean_plant_consumption_rate,
            "prey_consumption_rate": result.champion_evaluation.mean_prey_consumption_rate,
            "plant_intake_fraction": result.champion_evaluation.plant_intake_fraction,
            "prey_intake_fraction": result.champion_evaluation.prey_intake_fraction,
            "mi_sa": result.champion_evaluation.mean_mi_sa,
            "learning_slope": result.champion_evaluation.mean_learning_slope,
            "training_attack_funnel": {
                "no_organism_targets": result.champion_evaluation.mean_attack_no_organism_targets,
                "same_pool_blocked": result.champion_evaluation.mean_attack_same_pool_blocked,
                "eligible_attempts": result.champion_evaluation.mean_attack_eligible_attempts,
                "hits": result.champion_evaluation.mean_attack_hits,
                "nonlethal_hits": result.champion_evaluation.mean_attack_nonlethal_hits,
                "kills": result.champion_evaluation.mean_attack_kills,
                "same_pair_followups": result.champion_evaluation.mean_attack_same_pair_followups,
                "energy_transferred": result.champion_evaluation.mean_attack_energy_transferred,
            },
        },
        "trends": {
            "all_best_fitness_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_fitness).collect::<Vec<_>>(),
            ),
            "all_best_absolute_survival_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_absolute_survival_fraction).collect::<Vec<_>>(),
            ),
            "all_best_late_weighted_survival_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_late_weighted_survival_fraction).collect::<Vec<_>>(),
            ),
            "all_best_relative_advantage_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_relative_survival_advantage).collect::<Vec<_>>(),
            ),
            "all_best_path_connected_hidden_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_expressed_hidden_nodes as f64).collect::<Vec<_>>(),
            ),
            "all_best_path_connected_connections_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_expressed_connections as f64).collect::<Vec<_>>(),
            ),
            "late_best_fitness_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_fitness).collect::<Vec<_>>(),
            ),
            "late_best_absolute_survival_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_absolute_survival_fraction).collect::<Vec<_>>(),
            ),
            "late_best_late_weighted_survival_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_late_weighted_survival_fraction).collect::<Vec<_>>(),
            ),
            "late_best_relative_advantage_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_relative_survival_advantage).collect::<Vec<_>>(),
            ),
            "late_best_path_connected_hidden_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_expressed_hidden_nodes as f64).collect::<Vec<_>>(),
            ),
            "late_best_path_connected_connections_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_expressed_connections as f64).collect::<Vec<_>>(),
            ),
        },
        "innovation": {
            "late_start_generation": late_start_generation,
            "late_new_connection_origins": late_new_connection_origins,
            "late_new_node_origins": late_new_node_origins,
            "late_longest_zero_origin_streak": longest_zero_origin_streak(late),
            "expressed_connections_reaching_ten_percent": adaptive_connections,
            "expressed_nodes_reaching_ten_percent": adaptive_nodes,
        },
        "historical_checkpoints": {
            "count": generations.iter().filter(|generation| generation.checkpoint_champion_genome.is_some()).count(),
            "generations": generations.iter().filter(|generation| generation.checkpoint_champion_genome.is_some()).map(|generation| generation.generation).collect::<Vec<_>>(),
        },
    })
}

fn linear_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let covariance = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance = x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    (variance > 0.0).then(|| covariance / variance)
}

fn longest_zero_origin_streak(generations: &[evolution::GenerationSummary]) -> usize {
    let mut longest = 0usize;
    let mut current = 0usize;
    for generation in generations {
        if generation.new_connection_innovations + generation.new_node_innovations == 0 {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

fn value<'a>(args: &[&'a str], i: usize, flag: &str) -> Result<&'a str> {
    args.get(i + 1)
        .copied()
        .ok_or_else(|| anyhow!("{flag} needs a value"))
}

fn parse_scale(raw: &str) -> Result<(u32, u32)> {
    let (width, founders) = raw
        .split_once(',')
        .ok_or_else(|| anyhow!("--scale wants WIDTH,POP"))?;
    let width: u32 = width.trim().parse()?;
    let founders: u32 = founders.trim().parse()?;
    if width == 0 || founders == 0 {
        bail!("--scale width and population must be >= 1");
    }
    Ok((width, founders))
}

fn parse_u64_list(raw: &str, flag: &str) -> Result<Vec<u64>> {
    let values: Vec<u64> = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    if values.is_empty() {
        bail!("{flag} needs at least one value");
    }
    Ok(values)
}

fn parse_u32_list(raw: &str, flag: &str) -> Result<Vec<u32>> {
    let values: Vec<u32> = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    if values.is_empty() {
        bail!("{flag} needs at least one value");
    }
    Ok(values)
}

fn parse_f64_list(raw: &str, flag: &str) -> Result<Vec<f64>> {
    let values = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("{flag} needs a nonempty list of finite numbers");
    }
    Ok(values)
}

fn parse_scenarios(raw: &str) -> Result<Vec<ScenarioPreset>> {
    let scenarios: Vec<_> = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| ScenarioPreset::parse(part.trim()))
        .collect::<Result<_>>()?;
    if scenarios.is_empty() {
        bail!("--scenarios needs at least one scenario");
    }
    let mut deduped = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        if !deduped.contains(&scenario) {
            deduped.push(scenario);
        }
    }
    Ok(deduped)
}

fn parse_assignment(raw: &str) -> Result<(String, String)> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("expected key=value, got `{raw}`"))?;
    let key = key.trim();
    let value = value.trim();
    if key.is_empty() || value.is_empty() {
        bail!("expected non-empty key=value, got `{raw}`");
    }
    Ok((key.to_string(), value.to_string()))
}

fn apply_neat_param(config: &mut NeatConfig, key: &str, value: &str) -> Result<()> {
    match key {
        "compatibility_threshold" => config.compatibility_threshold = value.parse()?,
        "target_species" => config.target_species = value.parse()?,
        "compatibility_threshold_adjustment" => {
            config.compatibility_threshold_adjustment = value.parse()?
        }
        "excess_coefficient" => config.excess_coefficient = value.parse()?,
        "disjoint_coefficient" => config.disjoint_coefficient = value.parse()?,
        "weight_coefficient" => config.weight_coefficient = value.parse()?,
        "survival_fraction" => config.survival_fraction = value.parse()?,
        "crossover_probability" => config.crossover_probability = value.parse()?,
        "interspecies_mate_probability" => config.interspecies_mate_probability = value.parse()?,
        "curriculum_enabled" => config.curriculum_enabled = value.parse()?,
        "curriculum_promotion_threshold" => {
            config.curriculum_promotion_threshold = value.parse()?
        }
        "curriculum_promotion_patience" => config.curriculum_promotion_patience = value.parse()?,
        "training_seed_rotation_period" => config.training_seed_rotation_period = value.parse()?,
        "objective_cvar_fraction" => config.objective_cvar_fraction = value.parse()?,
        "fitness_objective" => config.fitness_objective = FitnessObjective::parse(value)?,
        "survival_window_weights" => {
            config.survival_window_weights = parse_f64_list(value, "survival_window_weights")?
        }
        "selection_strategy" => config.selection_strategy = SelectionStrategy::parse(value)?,
        "novelty_k" => config.novelty_k = value.parse()?,
        "novelty_archive_additions_per_generation" => {
            config.novelty_archive_additions_per_generation = value.parse()?
        }
        "mutate_weight_probability" => config.mutate_weight_probability = value.parse()?,
        "per_connection_weight_mutation_probability" => {
            config.per_connection_weight_mutation_probability = value.parse()?
        }
        "replace_weight_probability" => config.replace_weight_probability = value.parse()?,
        "weight_perturb_stddev" => config.weight_perturb_stddev = value.parse()?,
        "mutate_bias_probability" => config.mutate_bias_probability = value.parse()?,
        "bias_perturb_stddev" => config.bias_perturb_stddev = value.parse()?,
        "mutate_time_constant_probability" => {
            config.mutate_time_constant_probability = value.parse()?
        }
        "time_constant_perturb_stddev" => config.time_constant_perturb_stddev = value.parse()?,
        "add_connection_probability" => config.add_connection_probability = value.parse()?,
        "add_node_probability" => config.add_node_probability = value.parse()?,
        "disabled_inheritance_probability" => {
            config.disabled_inheritance_probability = value.parse()?
        }
        "stagnation_generations" => config.stagnation_generations = value.parse()?,
        "young_species_grace_generations" => {
            config.young_species_grace_generations = value.parse()?
        }
        "min_young_species_offspring" => config.min_young_species_offspring = value.parse()?,
        "elitism_min_species_size" => config.elitism_min_species_size = value.parse()?,
        "eval_opponents" => config.eval_opponents = value.parse()?,
        "eval_lineages_per_world" => config.eval_lineages_per_world = value.parse()?,
        "cross_pool_predation_only" => config.cross_pool_predation_only = value.parse()?,
        _ => bail!("unknown NEAT parameter `{key}`; valid: {PARAMS}"),
    }
    Ok(())
}
