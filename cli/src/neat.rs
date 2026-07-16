use crate::{run_output_path, DEFAULT_CONFIG, REPORT_EVERY};
use anyhow::{anyhow, bail, Result};
use evolution::{
    evaluate_frozen_pair, evaluate_frozen_panel, run_neat, EvaluationTopology, NeatConfig,
    RunResult, ScenarioManifest, OBJECTIVE_NAME,
};
use ring::digest::{digest, SHA256};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;
use types::OrganismGenome;
use views::{
    atomic_write, save_sidecar, save_world, sibling_metrics_path, start_recording,
    world_config_with_overrides,
};
use world_sim::Simulation;

const PARAMS: &str = "compatibility_threshold excess_coefficient disjoint_coefficient \
target_species compatibility_threshold_adjustment weight_coefficient survival_fraction \
crossover_probability interspecies_mate_probability \
training_seed_rotation_period survival_window_weights \
mutate_weight_probability replace_weight_probability weight_perturb_stddev \
per_connection_weight_mutation_probability mutate_bias_probability bias_perturb_stddev \
mutate_time_constant_probability time_constant_perturb_stddev add_connection_probability \
add_node_probability disabled_inheritance_probability \
young_species_grace_generations min_young_species_offspring \
elitism_min_species_size cross_pool_predation_only";

struct NeatRunRequest {
    config_path: String,
    run_seed: u64,
    founders: Option<u32>,
    world_width: Option<u32>,
    sets: Vec<(String, String)>,
    neat: NeatConfig,
}

/// Canonical generational NEAT runner. It owns no persistent world: each
/// candidate is evaluated in deterministic fixed-seed episodes, and the durable
/// artifacts contain the full effective algorithm/environment contract.
pub(crate) fn run_neat_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    if args.first() == Some(&"run") {
        return run_neat_cli(&args[1..], out_dir, out);
    }
    if args.first() == Some(&"plan") {
        return run_neat_plan(&args[1..], out);
    }
    if args.first() == Some(&"batch") {
        return crate::experiment::run_batch_cli(&args[1..], out_dir, out);
    }
    if args.first() == Some(&"summarize") {
        return crate::experiment::run_summarize_cli(&args[1..], out);
    }
    if args.first() == Some(&"analyze") {
        return run_neat_analysis(&args[1..], out);
    }
    if args.first() == Some(&"evaluate-panel") {
        return run_neat_panel_evaluation(&args[1..], out);
    }
    if args.first() == Some(&"crossplay") {
        return run_neat_crossplay(&args[1..], out);
    }
    if args.first() == Some(&"materialize") {
        return run_neat_materialize(&args[1..], out);
    }
    let Some(request) = parse_neat_run(args, out)? else {
        return Ok(());
    };
    execute_neat_run(request, out_dir, out)
}

fn parse_neat_run(args: &[&str], out: &mut impl Write) -> Result<Option<NeatRunRequest>> {
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut run_seed = 0u64;
    let mut founders = None;
    let mut world_width = None;
    let mut sets = Vec::new();
    // Research runs use one deliberately short baseline evaluator unless the
    // caller explicitly changes it.
    let mut neat = NeatConfig {
        episode_horizons: vec![500],
        ..NeatConfig::default()
    };
    let mut opponents_per_genome = None::<usize>;
    let mut cases_per_genome = None::<usize>;
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
            "--population-checkpoint-interval" => {
                neat.population_checkpoint_interval =
                    value(args, i, "--population-checkpoint-interval")?.parse()?;
                i += 2;
            }
            "--horizon" => {
                neat.episode_horizons = vec![value(args, i, "--horizon")?.parse()?];
                i += 2;
            }
            "--evaluator" => {
                neat.evaluation_topology =
                    EvaluationTopology::parse(value(args, i, "--evaluator")?)?;
                i += 2;
            }
            "--opponents-per-genome" => {
                opponents_per_genome = Some(value(args, i, "--opponents-per-genome")?.parse()?);
                i += 2;
            }
            "--cases-per-genome" => {
                cases_per_genome = Some(value(args, i, "--cases-per-genome")?.parse()?);
                i += 2;
            }
            "--world-seeds" => {
                neat.world_seeds =
                    parse_u64_list(value(args, i, "--world-seeds")?, "--world-seeds")?;
                i += 2;
            }
            "--workers" => {
                neat.evaluator_workers = value(args, i, "--workers")?.parse()?;
                i += 2;
            }
            "--founders" => {
                founders = Some(value(args, i, "--founders")?.parse()?);
                i += 2;
            }
            "--world-width" => {
                world_width = Some(value(args, i, "--world-width")?.parse()?);
                i += 2;
            }
            "--cvar" => {
                neat.objective_cvar_fraction = value(args, i, "--cvar")?.parse()?;
                i += 2;
            }
            "--set" => {
                let assignment = parse_assignment(value(args, i, "--set")?)?;
                match assignment.0.as_str() {
                    "world_width" => bail!("use --world-width instead of --set world_width=..."),
                    "num_organisms" => bail!("use --founders instead of --set num_organisms=..."),
                    "intent_parallel_threads" => bail!(
                        "NEAT parallelizes evaluator worlds and fixes intent threads to 1; use --workers to control parallelism"
                    ),
                    _ => sets.push(assignment),
                }
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
                    "NEAT run: cli [run] [OPTIONS]\n\
                     Preflight only: cli plan [OPTIONS]\n\
                     \n\
                     --seed N                    evolutionary run seed\n\
                     --population N              genomes per generation\n\
                     --generations N             generations to evaluate\n\
                     --population-checkpoint-interval N\n\
                                                 persist every Nth full population (default 10)\n\
                     --horizon N                 evaluator ticks (default 500)\n\
                     --evaluator pairwise|shared_population\n\
                                                 evaluator topology (default shared_population)\n\
                     --opponents-per-genome N    contemporary opponents per genome\n\
                     --cases-per-genome N        alternative exact scored-case budget\n\
                     --world-seeds N,N           deterministic training layouts\n\
                     --cvar F                    worst-case fraction aggregated into the contextual score\n\
                     --workers N                 parallel evaluator groups\n\
                     --founders N                founders (shared arenas require one per genome)\n\
                     --world-width N             hex-world width\n\
                     --set world_key=value       explicit world override\n\
                     --config PATH               canonical world TOML\n\
                     \n\
                     Pairwise accepts exactly one of --opponents-per-genome and\n\
                     --cases-per-genome. Shared population includes every genome\n\
                     and derives its opponent count from --population.\n\
                     Advanced algorithm overrides: --param key=value (valid: {PARAMS})"
                )?;
                return Ok(None);
            }
            other => bail!("unknown NEAT argument `{other}` (use `cli --help`)"),
        }
    }
    if opponents_per_genome.is_some() && cases_per_genome.is_some() {
        bail!("use either --opponents-per-genome or --cases-per-genome, not both");
    }
    let cases_per_opponent = neat
        .world_seeds
        .len()
        .saturating_mul(neat.episode_horizons.len());
    neat.eval_opponents = match neat.evaluation_topology {
        EvaluationTopology::SharedPopulation => {
            if opponents_per_genome.is_some() || cases_per_genome.is_some() {
                bail!(
                    "shared-population evaluation includes every other genome; omit --opponents-per-genome and --cases-per-genome"
                );
            }
            neat.population_size.saturating_sub(1)
        }
        EvaluationTopology::Pairwise => {
            let opponents = if let Some(cases) = cases_per_genome {
                if cases == 0
                    || cases_per_opponent == 0
                    || !cases.is_multiple_of(cases_per_opponent)
                {
                    bail!(
                        "--cases-per-genome {cases} is not an exact multiple of {cases_per_opponent} cases per opponent ({} world seeds × {} horizons)",
                        neat.world_seeds.len(),
                        neat.episode_horizons.len()
                    );
                }
                cases / cases_per_opponent
            } else if let Some(opponents) = opponents_per_genome {
                opponents
            } else {
                neat.eval_opponents
            };
            if opponents == 0 {
                bail!("--opponents-per-genome must be at least 1");
            }
            opponents
        }
    };
    neat.validate()?;
    Ok(Some(NeatRunRequest {
        config_path,
        run_seed,
        founders,
        world_width,
        sets,
        neat,
    }))
}

fn resolve_neat_world(request: &NeatRunRequest) -> Result<types::WorldConfig> {
    let mut world = world_config_with_overrides(&request.config_path, &request.sets)?;
    if let Some(width) = request.world_width {
        world.world_width = width;
    }
    if let Some(founders) = request.founders {
        world.num_organisms = founders;
    } else if request.neat.evaluation_topology == EvaluationTopology::SharedPopulation {
        world.num_organisms =
            request.neat.population_size.try_into().map_err(|_| {
                anyhow!("shared-population size does not fit the world founder count")
            })?;
    }
    if world.world_width == 0 || world.num_organisms == 0 {
        bail!("world width and founder count must both be at least 1");
    }
    if request.neat.evaluation_topology == EvaluationTopology::SharedPopulation
        && world.num_organisms as usize != request.neat.population_size
    {
        bail!(
            "shared-population evaluation requires exactly one founder per genome: founders={}, population={}",
            world.num_organisms,
            request.neat.population_size
        );
    }
    let lineage_count = match request.neat.evaluation_topology {
        EvaluationTopology::Pairwise => 2,
        EvaluationTopology::SharedPopulation => request.neat.population_size,
    };
    if !(world.num_organisms as usize).is_multiple_of(lineage_count) {
        bail!(
            "founder count {} must be divisible by {lineage_count} for {:?} evaluation",
            world.num_organisms,
            request.neat.evaluation_topology,
        );
    }
    Ok(world)
}

fn run_neat_plan(args: &[&str], out: &mut impl Write) -> Result<()> {
    let Some(request) = parse_neat_run(args, out)? else {
        return Ok(());
    };
    let world = resolve_neat_world(&request)?;
    let lineages = match request.neat.evaluation_topology {
        EvaluationTopology::Pairwise => 2,
        EvaluationTopology::SharedPopulation => request.neat.population_size,
    };
    let opponents = request.neat.eval_opponents;
    let cases_per_opponent = request.neat.world_seeds.len() * request.neat.episode_horizons.len();
    let (cases_per_genome, pairings, worlds_per_generation) = match request.neat.evaluation_topology
    {
        EvaluationTopology::Pairwise => {
            let pairings = request.neat.population_size * opponents / lineages;
            (
                opponents * cases_per_opponent,
                pairings,
                pairings * cases_per_opponent,
            )
        }
        EvaluationTopology::SharedPopulation => (cases_per_opponent, 0, cases_per_opponent),
    };
    let total_worlds = worlds_per_generation * request.neat.generations as usize;
    let scored_cases_per_generation = request.neat.population_size * cases_per_genome;
    let horizon = request.neat.episode_horizons[0];
    let available_workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    writeln!(
        out,
        "{}",
        json!({
            "mode": "neat_plan",
            "valid": true,
            "run_seed": request.run_seed,
            "population": request.neat.population_size,
            "generations": request.neat.generations,
            "population_checkpoint_interval": request.neat.population_checkpoint_interval,
            "horizon": horizon,
            "score_semantics": "contextual_within_generation_only",
            "objective_score_name": OBJECTIVE_NAME,
            "objective_cvar_fraction": request.neat.objective_cvar_fraction,
            "evaluation_topology": request.neat.evaluation_topology,
            "lineages_per_world": lineages,
            "opponents_per_genome": opponents,
            "world_seeds": request.neat.world_seeds,
            "scenario": "combat_baseline",
            "cases_per_opponent": cases_per_opponent,
            "scored_cases_per_genome": cases_per_genome,
            "opponent_exposures_per_genome": opponents * cases_per_opponent,
            "pairings_per_generation": pairings,
            "evaluation_worlds_per_generation": worlds_per_generation,
            "scored_lineage_cases_per_generation": scored_cases_per_generation,
            "total_evaluation_worlds": total_worlds,
            "total_scored_lineage_cases": scored_cases_per_generation * request.neat.generations as usize,
            "total_world_ticks": (total_worlds as u128) * u128::from(horizon),
            "world": {
                "width": world.world_width,
                "founders": world.num_organisms,
                "founders_per_lineage": world.num_organisms as usize / lineages,
                "attack_attempt_cost": world.attack_attempt_cost,
                "attack_energy_transfer": world.attack_energy_transfer,
                "compositional_actions_enabled": world.compositional_actions_enabled,
            },
            "workers": {
                "requested_evaluator_workers": request.neat.evaluator_workers,
                "effective_evaluator_workers": request.neat.evaluator_workers
                    .min(worlds_per_generation)
                    .max(1),
                "available_parallelism": available_workers,
                "intent_threads_per_world": 1,
            },
        })
    )
    .map_err(Into::into)
}

fn execute_neat_run(request: NeatRunRequest, out_dir: &str, out: &mut impl Write) -> Result<()> {
    // Plan and execution enforce the identical founder-divisibility and
    // world-override contract.
    let world = resolve_neat_world(&request)?;
    let run_seed = request.run_seed;
    let neat = request.neat;

    eprintln!(
        "{}",
        json!({
            "event": "neat_started",
            "population": neat.population_size,
            "generations": neat.generations,
            "population_checkpoint_interval": neat.population_checkpoint_interval,
            "episode_horizons": neat.episode_horizons,
            "survival_window_weights": neat.survival_window_weights,
            "world_seeds": neat.world_seeds,
            "workers": neat.evaluator_workers,
            "world_width": world.world_width,
            "founder_cohort_size": world.num_organisms,
            "objective": OBJECTIVE_NAME,
            "fully_connected_initial_topology": true,
            "current_tick_hidden_graph_acyclic": true,
            "previous_tick_hidden_recurrence_enabled": true,
            "evaluation_topology": neat.evaluation_topology,
            "balanced_pairwise_evaluation": neat.evaluation_topology
                == EvaluationTopology::Pairwise,
            "symmetric_founder_slot_rotation": true,
            "eval_opponents": neat.eval_opponents,
            "lineages_per_world": match neat.evaluation_topology {
                EvaluationTopology::Pairwise => 2,
                EvaluationTopology::SharedPopulation => neat.population_size,
            },
            "cross_pool_predation_only": neat.cross_pool_predation_only,
            "scenario": "combat_baseline",
        })
    );
    // Keep the base world config so the final generation's contextual winner
    // can be materialized into a real, inspectable world.
    let final_winner_world_config = world.clone();
    let total_generations = neat.generations;
    let progress_started = Instant::now();
    let mut previous_generation_finished = progress_started;
    let result = run_neat(neat, world, run_seed, |generation| {
        let now = Instant::now();
        let generation_seconds = now
            .duration_since(previous_generation_finished)
            .as_secs_f64();
        previous_generation_finished = now;
        let elapsed_seconds = now.duration_since(progress_started).as_secs_f64();
        let completed_generations = generation.generation.saturating_add(1);
        let mean_seconds_per_generation = elapsed_seconds / f64::from(completed_generations);
        let remaining_generations = total_generations.saturating_sub(completed_generations);
        let eta_seconds = mean_seconds_per_generation * f64::from(remaining_generations);
        let progress = json!({
            "completed_generations": completed_generations,
            "total_generations": total_generations,
            "generation_seconds": generation_seconds,
            "elapsed_seconds": elapsed_seconds,
            "mean_seconds_per_generation": mean_seconds_per_generation,
            "eta_seconds": eta_seconds,
        });
        eprintln!(
            "{}",
            json!({
                "event": "neat_generation",
                "run_seed": run_seed,
                "generation": generation.generation,
                "progress": progress,
                "training_seed_epoch": generation.training_seed_epoch,
                "effective_training_seeds": generation.effective_training_seeds,
                "eval_opponents": generation.eval_opponents,
                "evaluation_cases_per_genome": generation.evaluation_cases_per_genome,
                "evaluation_worlds": generation.evaluation_worlds,
                "winner_contextual_score": generation.winner_contextual_score,
                "winner_case_score_stddev": generation.winner_case_score_stddev,
                "winner_observations": {
                    "absolute_survival_fraction": generation.winner_absolute_survival_fraction,
                    "candidate_alive_ticks": generation.winner_candidate_alive_ticks,
                    "late_weighted_survival_fraction": generation.winner_late_weighted_survival_fraction,
                    "relative_survival_advantage": generation.winner_relative_survival_advantage,
                    "action_effectiveness": generation.winner_action_effectiveness,
                    "successful_attack_rate": generation.winner_successful_attack_rate,
                    "mean_attack_kills": generation.winner_mean_attack_kills,
                    "gross_energy_acquired": generation.winner_gross_energy_acquired,
                    "net_energy_profit": generation.winner_net_energy_profit,
                    "commands_per_tick": generation.winner_commands_per_tick,
                    "multi_command_tick_fraction": generation.winner_multi_command_tick_fraction,
                    "attack_target_evaded": generation.winner_attack_target_evaded,
                },
                "population_observations": {
                    "absolute_survival_fraction": generation.mean_absolute_survival_fraction,
                    "candidate_alive_ticks": generation.mean_candidate_alive_ticks,
                    "late_weighted_survival_fraction": generation.mean_late_weighted_survival_fraction,
                    "relative_survival_advantage": generation.mean_relative_survival_advantage,
                    "case_score_stddev_mean": generation.mean_case_score_stddev,
                    "case_score_stddev_max": generation.max_case_score_stddev,
                    "gross_energy_acquired_mean": generation.mean_gross_energy_acquired,
                    "net_energy_profit_mean": generation.mean_net_energy_profit,
                    "action_effectiveness_mean": generation.mean_action_effectiveness,
                    "successful_attack_rate_mean": generation.mean_successful_attack_rate,
                },
                "crossplay_checkpoint_persisted": generation.crossplay_checkpoint_genome.is_some(),
                "population_checkpoint_persisted": !generation.population_checkpoint.is_empty(),
                "species": generation.species.len(),
                "compatibility_threshold": generation.compatibility_threshold,
                "hidden_nodes": generation.winner_hidden_nodes,
                "enabled_connections": generation.winner_enabled_connections,
                "expressed_hidden_nodes": generation.winner_expressed_hidden_nodes,
                "expressed_connections": generation.winner_expressed_connections,
                "new_connection_innovations": generation.new_connection_innovations,
                "new_node_innovations": generation.new_node_innovations,
                "new_origin_offspring_rate": generation.new_origin_offspring_rate,
                "innovations_reaching_ten_percent": generation.connection_innovations_reaching_ten_percent,
                "crossovers": generation.offspring_crossovers,
            })
        );
    })?;

    let mut result_path = run_output_path(out_dir, "neat")?;
    result_path.set_extension("json.zst");
    let result_path_string = result_path.to_string_lossy().into_owned();
    atomic_write(&result_path_string, |writer| {
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, &result)?;
        encoder.finish()?;
        Ok(())
    })?;

    let final_winner = result
        .final_population
        .iter()
        .max_by(|left, right| left.contextual_score.total_cmp(&right.contextual_score))
        .ok_or_else(|| anyhow!("NEAT result has no final population"))?;

    // Materialize the final contextual winner as a real world.bin (+ sidecar)
    // seeded as a clonal colony from its genome. It is then a first-
    // class world — `run-to`/`pillars`/`inspect`/`brain` all work directly.
    let mut world_path = result_path.clone();
    world_path.set_extension("world.bin");
    let world_path_string = world_path.to_string_lossy().into_owned();
    let final_winner_world = Simulation::new_with_founder_genome_pool(
        final_winner_world_config,
        run_seed,
        vec![final_winner.genome.clone()],
    )
    .map_err(|error| anyhow!("building final winner world: {error}"))?;
    save_world(&final_winner_world, &world_path_string)?;
    let recorder = start_recording(&final_winner_world, REPORT_EVERY);
    let sidecar_path = sibling_metrics_path(&world_path_string);
    save_sidecar(REPORT_EVERY, &recorder, &sidecar_path)?;

    writeln!(
        out,
        "{}",
        json!({
            "wrote": result_path_string,
            "final_winner_world": world_path_string,
            "objective": result.objective,
            "final_generation": final_winner.generation,
            "final_winner_population_index": final_winner.population_index,
            "final_winner_contextual_score": final_winner.contextual_score,
            "final_winner_action_effectiveness": final_winner.evaluation.mean_action_effectiveness,
            "final_winner_successful_attack_rate": final_winner.evaluation.mean_successful_attack_rate,
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
                    "cli evaluate-panel --focal RESULT.json.zst --opponents RESULT.json.zst[,RESULT.json.zst...] [--horizons N,N] [--window-weights W,W] [--world-seeds N,N] [--cvar F] [--cross-pool-only] [--focal-slot N]"
                )?;
                return Ok(());
            }
            other => bail!("unknown evaluate-panel argument `{other}`"),
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
        .map(|result| final_winner(result).map(|winner| winner.genome.clone()))
        .collect::<Result<Vec<_>>>()?;
    let world_seeds = world_seeds.unwrap_or_else(|| focal.neat_config.world_seeds.clone());
    let focal_winner = final_winner(&focal)?;

    let mut evaluations = Vec::with_capacity(horizons.len());
    for horizon in horizons {
        let evaluation = evaluate_frozen_panel(
            &focal_winner.genome,
            &opponent_genomes,
            &focal.evaluation_scenarios,
            horizon,
            &survival_window_weights,
            &world_seeds,
            objective_cvar_fraction,
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
            "schema_version": 3,
            "focal": focal_path,
            "focal_run_seed": focal.seed,
            "focal_final_winner_population_index": focal_winner.population_index,
            "opponents": opponent_paths,
            "opponent_run_seeds": opponents.iter().map(|result| result.seed).collect::<Vec<_>>(),
            "objective": OBJECTIVE_NAME,
            "objective_cvar_fraction": objective_cvar_fraction,
            "survival_window_weights": survival_window_weights,
            "cross_pool_predation_only": cross_pool_predation_only,
            "focal_pool_index": focal_pool_index,
            "world_seeds": world_seeds,
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
    let mut objective_cvar_fraction = None::<f64>;
    let mut cross_pool_predation_only = None::<bool>;
    let mut output_path = None::<String>;
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
            "--out" => {
                output_path = Some(value(args, i, "--out")?.to_string());
                i += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "cli crossplay RESULT.json.zst [--checkpoints all|G,G] [--horizons N,N] [--world-seeds N,N] [--cvar F] [--cross-pool-only|--allow-same-pool] [--out FILE]\nEvery distinct-genome matchup is evaluated in both founder slots; clone-versus-clone comparisons are omitted. World seeds must be a multiple of four. Scores are comparable only within this frozen crossplay contract. `--out` persists the full JSON atomically and prints only a completion record."
                )?;
                return Ok(());
            }
            value if !value.starts_with('-') && result_path.is_none() => {
                result_path = Some(value.to_string());
                i += 1;
            }
            other => bail!("unknown crossplay argument `{other}`"),
        }
    }
    let result_path = result_path.ok_or_else(|| anyhow!("crossplay needs RESULT.json.zst"))?;
    let source = read_neat_result(&result_path)?;
    let horizons = horizons.unwrap_or_else(|| source.neat_config.episode_horizons.clone());
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
    let mut checkpoints = source
        .generations
        .iter()
        .filter_map(|generation| {
            generation
                .crossplay_checkpoint_genome
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
    let crossplay_scenarios = source
        .evaluation_scenarios
        .iter()
        .map(|scenario| {
            let mut scenario = scenario.clone();
            let source_cells = u64::from(scenario.world.world_width).pow(2).max(1);
            let source_density = f64::from(scenario.world.num_organisms) / source_cells as f64;
            scenario.world.num_organisms = 2;
            scenario.world.world_width = ((2.0 / source_density).sqrt().round() as u32).max(3);
            scenario
        })
        .collect::<Vec<_>>();

    let mut cells = Vec::with_capacity(
        checkpoints
            .len()
            .saturating_mul(checkpoints.len().saturating_sub(1))
            .saturating_mul(horizons.len()),
    );
    for focal in &checkpoints {
        for opponent in &checkpoints {
            // Clone-versus-clone play is not a general competence measurement:
            // identical strategies can simply exchange attacks. Compare only
            // genuinely distinct genomes, including across checkpoint numbers.
            if focal.genome_hash == opponent.genome_hash {
                continue;
            }
            for &horizon in &horizons {
                let evaluation = evaluate_frozen_pair(
                    &focal.genome,
                    &opponent.genome,
                    &crossplay_scenarios,
                    &[horizon],
                    &source.neat_config.survival_window_weights,
                    &world_seeds,
                    objective_cvar_fraction,
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
    let payload = json!({
        "schema_version": 5,
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
            "scenario_names": source.evaluation_scenarios.iter().map(|scenario| &scenario.name).collect::<Vec<_>>(),
            "founder_pool_size": 2,
            "one_founder_per_genome": true,
            "density_matched_worlds": crossplay_scenarios.iter().map(|scenario| json!({
                "name": scenario.name,
                "world_width": scenario.world.world_width,
                "founders": scenario.world.num_organisms,
            })).collect::<Vec<_>>(),
            "balanced_slot_rotation": true,
            "distinct_genomes_only": true,
            "objective": OBJECTIVE_NAME,
            "objective_cvar_fraction": objective_cvar_fraction,
            "cross_pool_predation_only": cross_pool_predation_only,
        },
        "cells": cells,
    });
    if let Some(output_path) = output_path {
        atomic_write(&output_path, |writer| {
            serde_json::to_writer_pretty(writer, &payload).map_err(Into::into)
        })?;
        writeln!(
            out,
            "{}",
            json!({
                "wrote": output_path,
                "schema_version": 5,
                "checkpoints": checkpoints.len(),
                "cells": payload["cells"].as_array().map_or(0, Vec::len),
            })
        )
        .map_err(Into::into)
    } else {
        writeln!(out, "{payload}").map_err(Into::into)
    }
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
    evaluation_scenarios: Vec<ScenarioManifest>,
    neat_config: PanelSeedConfig,
    generations: Vec<PanelGenerationSource>,
    final_population: Vec<PanelPopulationSource>,
}

#[derive(Deserialize)]
struct PanelGenerationSource {
    generation: u32,
    #[serde(default)]
    crossplay_checkpoint_genome: Option<OrganismGenome>,
}

#[derive(Deserialize)]
struct PanelPopulationSource {
    population_index: usize,
    contextual_score: f64,
    genome: OrganismGenome,
}

#[derive(Deserialize)]
struct PanelSeedConfig {
    world_seeds: Vec<u64>,
    episode_horizons: Vec<u64>,
    survival_window_weights: Vec<f64>,
    objective_cvar_fraction: f64,
    cross_pool_predation_only: bool,
}

fn read_neat_result(path: &str) -> Result<PanelResultSource> {
    serde_json::from_reader(result_reader(path)?)
        .map_err(|error| anyhow!("cannot parse NEAT result `{path}`: {error}"))
}

fn final_winner(source: &PanelResultSource) -> Result<&PanelPopulationSource> {
    source
        .final_population
        .iter()
        .max_by(|left, right| left.contextual_score.total_cmp(&right.contextual_score))
        .ok_or_else(|| anyhow!("NEAT result has no final population"))
}

fn run_neat_materialize(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.iter().any(|arg| matches!(*arg, "--help" | "-h")) {
        writeln!(
            out,
            "cli materialize RESULT.json.zst --generation N [--seed N] --out WORLD.bin"
        )?;
        return Ok(());
    }
    let result_path = args
        .first()
        .ok_or_else(|| anyhow!("materialize needs a result path"))?;
    let mut generation = None;
    let mut world_seed = None;
    let mut output_path = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i] {
            "--generation" => {
                generation = Some(value(args, i, "--generation")?.parse::<u32>()?);
                i += 2;
            }
            "--seed" => {
                world_seed = Some(value(args, i, "--seed")?.parse::<u64>()?);
                i += 2;
            }
            "--out" => {
                output_path = Some(value(args, i, "--out")?.to_string());
                i += 2;
            }
            other => bail!("unknown materialize argument `{other}`"),
        }
    }
    let generation = generation.ok_or_else(|| anyhow!("materialize needs --generation N"))?;
    let output_path = output_path.ok_or_else(|| anyhow!("materialize needs --out WORLD.bin"))?;
    let source = read_neat_result(result_path)?;
    let genome = source
        .generations
        .iter()
        .find(|checkpoint| checkpoint.generation == generation)
        .and_then(|checkpoint| checkpoint.crossplay_checkpoint_genome.clone())
        .ok_or_else(|| anyhow!("result has no crossplay checkpoint for generation {generation}"))?;
    let scenario = source
        .evaluation_scenarios
        .first()
        .ok_or_else(|| anyhow!("result has no evaluation scenario"))?;
    let world_seed = world_seed.unwrap_or(source.seed);
    let simulation =
        Simulation::new_with_founder_genome_pool(scenario.world.clone(), world_seed, vec![genome])
            .map_err(|error| anyhow!("materializing checkpoint world: {error}"))?;
    save_world(&simulation, &output_path)?;
    let recorder = start_recording(&simulation, REPORT_EVERY);
    let metrics_path = sibling_metrics_path(&output_path);
    save_sidecar(REPORT_EVERY, &recorder, &metrics_path)?;
    writeln!(
        out,
        "{}",
        json!({
            "source": result_path,
            "generation": generation,
            "world_seed": world_seed,
            "world": output_path,
            "metrics": metrics_path,
        })
    )
    .map_err(Into::into)
}

fn run_neat_analysis(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.is_empty() {
        bail!("neat analyze needs at least one result.json.zst path");
    }
    let mut analyses = Vec::with_capacity(args.len());
    for path in args {
        let result: RunResult = serde_json::from_reader(result_reader(path)?)
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

fn result_reader(path: &str) -> Result<Box<dyn Read>> {
    let file = File::open(path).map_err(|error| anyhow!("cannot open `{path}`: {error}"))?;
    if path.ends_with(".zst") {
        Ok(Box::new(zstd::stream::read::Decoder::new(file)?))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn analyze_result(path: &str, result: &RunResult) -> serde_json::Value {
    let generations = &result.generations;
    let final_winner = result
        .final_population
        .iter()
        .max_by(|left, right| left.contextual_score.total_cmp(&right.contextual_score));
    let final_winner = final_winner.map(|winner| {
        let evaluation = &winner.evaluation;
        let attack_attempts = evaluation.mean_attack_no_organism_targets
            + evaluation.mean_attack_target_evaded
            + evaluation.mean_attack_same_pool_blocked
            + evaluation.mean_attack_insufficient_energy
            + evaluation.mean_attack_eligible_attempts;
        json!({
            "generation": winner.generation,
            "population_index": winner.population_index,
            "contextual_score": winner.contextual_score,
            "case_score_stddev": evaluation.case_score_stddev,
            "case_score_range": [evaluation.min_case_score, evaluation.max_case_score],
            "opponent_score_profile": winner.opponent_scores,
            "survival": {
                "absolute_fraction": evaluation.mean_absolute_survival_fraction,
                "alive_ticks": evaluation.mean_candidate_alive_ticks,
                "late_weighted_fraction": evaluation.mean_late_weighted_survival_fraction,
                "relative_advantage": evaluation.mean_relative_survival_advantage,
                "end_survivor_fraction": evaluation.mean_candidate_end_survival_fraction,
            },
            "behavior": {
                "action_effectiveness": evaluation.mean_action_effectiveness,
                "successful_attack_rate": evaluation.mean_successful_attack_rate,
                "action_fractions": evaluation.mean_action_fractions,
                "commands_per_tick": evaluation.mean_commands_per_tick,
                "multi_command_tick_fraction": evaluation.mean_multi_command_tick_fraction,
                "spatial_coverage": evaluation.mean_spatial_coverage,
            },
            "attack_funnel": {
                "no_organism_targets": evaluation.mean_attack_no_organism_targets,
                "target_evaded": evaluation.mean_attack_target_evaded,
                "same_pool_blocked": evaluation.mean_attack_same_pool_blocked,
                "insufficient_energy": evaluation.mean_attack_insufficient_energy,
                "eligible_attempts": evaluation.mean_attack_eligible_attempts,
                "hits": evaluation.mean_attack_hits,
                "nonlethal_hits": evaluation.mean_attack_nonlethal_hits,
                "kills": evaluation.mean_attack_kills,
                "same_pair_followups": evaluation.mean_attack_same_pair_followups,
                "distinct_victims": evaluation.mean_distinct_attack_victims,
                "precision": (attack_attempts > 0.0).then(|| evaluation.mean_attack_hits / attack_attempts),
                "repeat_hit_fraction": evaluation.attack_repeat_hit_fraction,
            },
            "energy_flow": {
                "gross_acquired": evaluation.mean_gross_energy_acquired,
                "attack_received": evaluation.mean_attack_energy_received,
                "attack_lost": evaluation.mean_attack_energy_lost,
                "attack_attempt_cost": evaluation.mean_attack_attempt_energy_cost,
                "net_attack_balance": evaluation.mean_net_attack_energy_balance,
            },
        })
    });

    json!({
        "path": path,
        "run_seed": result.seed,
        "generations": generations.len(),
        "score_semantics": "contextual_within_generation_only",
        "longitudinal_validation": "use_crossplay",
        "complexification_enabled": result.neat_config.add_connection_probability > 0.0
            || result.neat_config.add_node_probability > 0.0,
        "final_generation_winner": final_winner,
        "generation_contexts": generations.iter().map(|generation| json!({
            "generation": generation.generation,
            "winner_contextual_score": generation.winner_contextual_score,
            "winner_case_score_stddev": generation.winner_case_score_stddev,
            "evaluation_cases_per_genome": generation.evaluation_cases_per_genome,
            "evaluation_worlds": generation.evaluation_worlds,
            "species": generation.species.len(),
            "winner_expressed_hidden_nodes": generation.winner_expressed_hidden_nodes,
            "winner_expressed_connections": generation.winner_expressed_connections,
            "new_connection_innovations": generation.new_connection_innovations,
            "new_node_innovations": generation.new_node_innovations,
        })).collect::<Vec<_>>(),
        "innovation": {
            "connection_records": result.connection_innovation_history.len(),
            "node_records": result.node_innovation_history.len(),
        },
        "crossplay_checkpoints": {
            "count": generations.iter().filter(|generation| generation.crossplay_checkpoint_genome.is_some()).count(),
            "generations": generations.iter().filter(|generation| generation.crossplay_checkpoint_genome.is_some()).map(|generation| generation.generation).collect::<Vec<_>>(),
        },
    })
}

fn value<'a>(args: &[&'a str], i: usize, flag: &str) -> Result<&'a str> {
    args.get(i + 1)
        .copied()
        .ok_or_else(|| anyhow!("{flag} needs a value"))
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
        "training_seed_rotation_period" => config.training_seed_rotation_period = value.parse()?,
        "survival_window_weights" => {
            config.survival_window_weights = parse_f64_list(value, "survival_window_weights")?
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
        "young_species_grace_generations" => {
            config.young_species_grace_generations = value.parse()?
        }
        "min_young_species_offspring" => config.min_young_species_offspring = value.parse()?,
        "elitism_min_species_size" => config.elitism_min_species_size = value.parse()?,
        "cross_pool_predation_only" => config.cross_pool_predation_only = value.parse()?,
        _ => bail!("unknown NEAT parameter `{key}`; valid: {PARAMS}"),
    }
    Ok(())
}
