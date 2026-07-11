use crate::{run_output_path, DEFAULT_CONFIG, REPORT_EVERY};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sim_core::{run_neat, NeatConfig, RunResult, ScenarioPreset, SelectionStrategy, Simulation};
use sim_views::{
    atomic_write, save_sidecar, save_world, sibling_metrics_path, start_recording,
    world_config_with_overrides,
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;

const PARAMS: &str = "compatibility_threshold excess_coefficient disjoint_coefficient \
target_species compatibility_threshold_adjustment weight_coefficient survival_fraction \
crossover_probability interspecies_mate_probability \
curriculum_enabled curriculum_promotion_threshold curriculum_promotion_patience \
training_seed_rotation_period objective_cvar_fraction \
selection_strategy novelty_k novelty_archive_additions_per_generation \
mutate_weight_probability replace_weight_probability weight_perturb_stddev \
per_connection_weight_mutation_probability mutate_bias_probability bias_perturb_stddev \
mutate_time_constant_probability time_constant_perturb_stddev add_connection_probability \
add_node_probability disabled_inheritance_probability stagnation_generations \
young_species_grace_generations min_young_species_offspring \
elitism_min_species_size eval_opponents";

/// Canonical generational NEAT runner. It owns no persistent world: each
/// candidate is evaluated in deterministic fixed-seed episodes, and the durable
/// artifacts contain the full effective algorithm/environment contract.
pub(crate) fn run_neat_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    if args.first() == Some(&"analyze") {
        return run_neat_analysis(&args[1..], out);
    }
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut run_seed = 0u64;
    let mut scale = Some((25u32, 30u32));
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
            "--episode-ticks" => {
                neat.episode_ticks = value(args, i, "--episode-ticks")?.parse()?;
                i += 2;
            }
            "--world-seeds" => {
                neat.world_seeds =
                    parse_u64_list(value(args, i, "--world-seeds")?, "--world-seeds")?;
                i += 2;
            }
            "--audit-seeds" => {
                neat.development_world_seeds =
                    parse_u64_list(value(args, i, "--audit-seeds")?, "--audit-seeds")?;
                i += 2;
            }
            "--holdout-seeds" => {
                neat.sealed_holdout_world_seeds =
                    parse_u64_list(value(args, i, "--holdout-seeds")?, "--holdout-seeds")?;
                i += 2;
            }
            "--audit-levels" => {
                neat.audit_curriculum_levels =
                    parse_u32_list(value(args, i, "--audit-levels")?, "--audit-levels")?;
                i += 2;
            }
            "--audit-every" => {
                neat.development_audit_interval_generations =
                    value(args, i, "--audit-every")?.parse()?;
                i += 2;
            }
            "--scenarios" => {
                neat.scenarios = parse_scenarios(value(args, i, "--scenarios")?)?;
                i += 2;
            }
            "--no-holdout" => {
                neat.sealed_holdout_world_seeds.clear();
                i += 1;
            }
            "--no-audit" => {
                neat.development_world_seeds.clear();
                i += 1;
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
                     --episode-ticks T --world-seeds N,N [--audit-seeds N,N|--no-audit] \
                     [--holdout-seeds N,N|--no-holdout] [--audit-levels N,N] [--audit-every N] \
                     [--scenarios baseline,scarcity,sparse_search] \
                     --workers K --scale W,POP \
                     [--no-scale] [--set world_key=value] [--param neat_key=value]\n\
                     default scale: 25,30; objective: lower-tail robust offspring maturity;\n\
                     analyze: neat analyze RESULT.json [RESULT2.json ...]\n\
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
            "episode_ticks": neat.episode_ticks,
            "world_seeds": neat.world_seeds,
            "development_world_seeds": neat.development_world_seeds,
            "sealed_holdout_world_seeds": neat.sealed_holdout_world_seeds,
            "audit_curriculum_levels": neat.audit_curriculum_levels,
            "development_audit_interval_generations": neat.development_audit_interval_generations,
            "workers": neat.evaluator_workers,
            "world_width": world.world_width,
            "founder_cohort_size": world.num_organisms,
            "objective": if neat.eval_opponents > 0 {
                "lower_tail_mean_competitive_survival_fraction"
            } else {
                "lower_tail_mean_survival_fraction"
            },
            "eval_opponents": neat.eval_opponents,
            "scenarios": neat.scenarios,
        })
    );
    // Keep the base world config so the champion can be materialized into a
    // real, inspectable world once the run finishes.
    let champion_world_config = world.clone();
    let result = run_neat(neat, world, run_seed, |generation| {
        let development_audit_by_level = generation
            .champion_development_evaluation
            .as_ref()
            .map(|suite| {
                suite
                    .levels
                    .iter()
                    .map(|level| {
                        json!({
                            "curriculum_level": level.curriculum_level,
                            "objective": level.evaluation.mean_objective_score,
                            "plant_capture_fraction": level.evaluation.mean_plant_capture_fraction,
                            "plant_consumptions_per_tick": level.evaluation.mean_plant_consumptions_per_tick,
                            "standing_plant_fraction": level.evaluation.mean_standing_plant_fraction,
                            "spatial_coverage": level.evaluation.mean_spatial_coverage,
                        })
                    })
                    .collect::<Vec<_>>()
            });
        eprintln!(
            "{}",
            json!({
                "event": "neat_generation",
                "generation": generation.generation,
                "curriculum_level": generation.curriculum_level,
                "training_seed_epoch": generation.training_seed_epoch,
                "effective_training_seeds": generation.effective_training_seeds,
                "best_fitness": generation.best_fitness,
                "mean_fitness": generation.mean_fitness,
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
                "development_audit_score": generation.champion_development_evaluation.as_ref().map(|value| value.mean_level_objective_score),
                "development_audit_by_level": development_audit_by_level,
                "evolved_structure_development_knockout_delta": generation.evolved_structure_development_knockout_delta,
                "evolved_structure_development_ancestral_delta": generation.evolved_structure_development_ancestral_delta,
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
    let recorder = start_recording(&champion_world);
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
            "development_audit_score": result.champion_development_evaluation.as_ref().map(|value| value.mean_level_objective_score),
            "evolved_structure_development_knockout_delta": result.evolved_structure_development_knockout_delta,
            "sealed_holdout_score": result.sealed_holdout_evaluation.as_ref().map(|value| value.mean_level_objective_score),
            "evolved_structure_sealed_knockout_delta": result.evolved_structure_sealed_knockout_delta,
            "generations": result.generations.len(),
        })
    )
    .map_err(Into::into)
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
    let development_points = generations
        .iter()
        .filter_map(|generation| {
            generation
                .champion_development_evaluation
                .as_ref()
                .map(|suite| {
                    (
                        generation.generation as f64,
                        suite.mean_level_objective_score,
                    )
                })
        })
        .collect::<Vec<_>>();
    let late_development_points = development_points
        .iter()
        .copied()
        .filter(|(generation, _)| *generation >= split as f64)
        .collect::<Vec<_>>();

    let mut audit_levels = BTreeMap::<u32, Vec<(f64, f64, Option<f64>, f64, f64)>>::new();
    for generation in generations {
        let Some(suite) = generation.champion_development_evaluation.as_ref() else {
            continue;
        };
        for level in &suite.levels {
            audit_levels
                .entry(level.curriculum_level)
                .or_default()
                .push((
                    generation.generation as f64,
                    level.evaluation.mean_objective_score,
                    level.evaluation.mean_plant_capture_fraction,
                    level.evaluation.mean_plant_consumptions_per_tick,
                    level.evaluation.mean_standing_plant_fraction,
                ));
        }
    }
    let audit_level_trends = audit_levels
        .into_iter()
        .map(|(level, points)| {
            let x = points.iter().map(|point| point.0).collect::<Vec<_>>();
            let competence = points.iter().map(|point| point.1).collect::<Vec<_>>();
            let capture_points = points
                .iter()
                .filter_map(|point| point.2.map(|capture| (point.0, capture)))
                .collect::<Vec<_>>();
            let capture_x = capture_points
                .iter()
                .map(|point| point.0)
                .collect::<Vec<_>>();
            let capture_y = capture_points
                .iter()
                .map(|point| point.1)
                .collect::<Vec<_>>();
            let final_point = points.last().copied();
            json!({
                "curriculum_level": level,
                "checkpoints": points.len(),
                "competence_slope": linear_slope(&x, &competence),
                "plant_capture_slope": linear_slope(&capture_x, &capture_y),
                "final_objective": final_point.map(|point| point.1),
                "final_plant_capture_fraction": final_point.and_then(|point| point.2),
                "final_plant_consumptions_per_tick": final_point.map(|point| point.3),
                "final_standing_plant_fraction": final_point.map(|point| point.4),
            })
        })
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
        .filter(|record| {
            record
                .frontier_champion_ablation_delta
                .is_some_and(|delta| delta > 0.0)
        })
        .count();
    let adaptive_nodes = result
        .node_innovation_history
        .iter()
        .filter(|record| record.max_expressed_frequency >= 0.10)
        .filter(|record| {
            record
                .frontier_champion_ablation_delta
                .is_some_and(|delta| delta > 0.0)
        })
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
            "development_score": result.champion_development_evaluation.as_ref().map(|suite| suite.mean_level_objective_score),
            "development_levels": result.champion_development_evaluation.as_ref().map(suite_level_summary),
            "sealed_score": result.sealed_holdout_evaluation.as_ref().map(|suite| suite.mean_level_objective_score),
            "sealed_levels": result.sealed_holdout_evaluation.as_ref().map(suite_level_summary),
            "development_structure_knockout_delta": result.evolved_structure_development_knockout_delta,
            "sealed_structure_knockout_delta": result.evolved_structure_sealed_knockout_delta,
        },
        "trends": {
            "all_best_fitness_slope": linear_slope(
                &all_generation,
                &generations.iter().map(|generation| generation.best_fitness).collect::<Vec<_>>(),
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
            "late_best_path_connected_hidden_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_expressed_hidden_nodes as f64).collect::<Vec<_>>(),
            ),
            "late_best_path_connected_connections_slope": linear_slope(
                &late_generation,
                &late.iter().map(|generation| generation.best_expressed_connections as f64).collect::<Vec<_>>(),
            ),
            "development_score_slope": slope_of_points(&development_points),
            "late_development_score_slope": slope_of_points(&late_development_points),
        },
        "fixed_audit_levels": audit_level_trends,
        "innovation": {
            "late_start_generation": late_start_generation,
            "late_new_connection_origins": late_new_connection_origins,
            "late_new_node_origins": late_new_node_origins,
            "late_longest_zero_origin_streak": longest_zero_origin_streak(late),
            "adaptive_connections_reaching_ten_percent": adaptive_connections,
            "adaptive_nodes_reaching_ten_percent": adaptive_nodes,
        },
    })
}

fn suite_level_summary(suite: &sim_core::FixedSuiteEvaluation) -> Vec<serde_json::Value> {
    suite
        .levels
        .iter()
        .map(|level| {
            json!({
                "curriculum_level": level.curriculum_level,
                "objective": level.evaluation.mean_objective_score,
                "plant_capture_fraction": level.evaluation.mean_plant_capture_fraction,
                "plant_consumptions_per_tick": level.evaluation.mean_plant_consumptions_per_tick,
                "standing_plant_fraction": level.evaluation.mean_standing_plant_fraction,
                "spatial_coverage": level.evaluation.mean_spatial_coverage,
            })
        })
        .collect()
}

fn slope_of_points(points: &[(f64, f64)]) -> Option<f64> {
    let x = points.iter().map(|point| point.0).collect::<Vec<_>>();
    let y = points.iter().map(|point| point.1).collect::<Vec<_>>();
    linear_slope(&x, &y)
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

fn longest_zero_origin_streak(generations: &[sim_core::GenerationSummary]) -> usize {
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
        _ => bail!("unknown NEAT parameter `{key}`; valid: {PARAMS}"),
    }
    Ok(())
}
