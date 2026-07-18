use crate::run_output_path;
use anyhow::{anyhow, bail, Result};
use evolution::{
    run_neat,
    tasks::symbol_copy::{SymbolCopyRunResult, SymbolCopyTask, OBJECTIVE_NAME, TASK_NAME},
    NeatConfig,
};
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;
use types::{SeedGenomeConfig, Symbol};
use views::atomic_write;

const DEFAULT_SEED_CONFIG: &str = config::CANONICAL_SEED_GENOME_CONFIG_PATH;
pub(crate) const PARAMS: &str = "compatibility_threshold target_species \
compatibility_threshold_adjustment excess_coefficient disjoint_coefficient \
weight_coefficient learning_coefficient survival_fraction crossover_probability \
interspecies_mate_probability mutate_weight_probability \
per_connection_weight_mutation_probability replace_weight_probability \
weight_perturb_stddev mutate_bias_probability bias_perturb_stddev \
mutate_time_constant_probability time_constant_perturb_stddev \
mutate_learning_rate_probability learning_rate_perturb_stddev \
mutate_plasticity_coefficient_probability plasticity_coefficient_perturb_stddev \
add_connection_probability add_node_probability \
disabled_inheritance_probability elitism_min_species_size";

struct NeatRunRequest {
    seed_config_path: String,
    run_seed: u64,
    neat: NeatConfig,
    task: SymbolCopyTask,
}

pub(crate) fn run_neat_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    if args.first() == Some(&"hidden-string") {
        return crate::neat_hidden_string::run_cli(&args[1..], out_dir, out);
    }
    if args.first() == Some(&"run") {
        return run_neat_cli(&args[1..], out_dir, out);
    }
    if args.first() == Some(&"plan") {
        return run_neat_plan(&args[1..], out);
    }
    if args.first() == Some(&"analyze") {
        return run_neat_analysis(&args[1..], out);
    }
    let Some(request) = parse_neat_run(args, out)? else {
        return Ok(());
    };
    execute_neat_run(request, out_dir, out)
}

fn parse_neat_run(args: &[&str], out: &mut impl Write) -> Result<Option<NeatRunRequest>> {
    let mut seed_config_path = DEFAULT_SEED_CONFIG.to_owned();
    let mut run_seed = 0_u64;
    let mut streams = Vec::new();
    let mut neat = NeatConfig::default();
    let mut task = SymbolCopyTask::default();
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--seed-config" => {
                seed_config_path = value(args, index, "--seed-config")?.to_owned();
                index += 2;
            }
            "--seed" => {
                run_seed = value(args, index, "--seed")?.parse()?;
                index += 2;
            }
            "--population" => {
                neat.population_size = value(args, index, "--population")?.parse()?;
                index += 2;
            }
            "--generations" => {
                neat.generations = value(args, index, "--generations")?.parse()?;
                index += 2;
            }
            "--population-checkpoint-interval" => {
                neat.population_checkpoint_interval =
                    value(args, index, "--population-checkpoint-interval")?.parse()?;
                index += 2;
            }
            "--workers" => {
                neat.evaluation_workers = value(args, index, "--workers")?.parse()?;
                index += 2;
            }
            "--stream" => {
                streams.push(parse_stream(value(args, index, "--stream")?)?);
                index += 2;
            }
            "--leaky-neurons" => {
                task.config.leaky_neurons_enabled = true;
                index += 1;
            }
            "--param" => {
                let (key, raw) = parse_assignment(value(args, index, "--param")?)?;
                apply_neat_param(&mut neat, &key, &raw)?;
                index += 2;
            }
            "--help" | "-h" => {
                write_help(out)?;
                return Ok(None);
            }
            other => bail!("unknown NEAT argument `{other}` (use `cli --help`)"),
        }
    }
    if !streams.is_empty() {
        task.config.training_streams = streams;
    }
    neat.validate()?;
    task.config.validate()?;
    Ok(Some(NeatRunRequest {
        seed_config_path,
        run_seed,
        neat,
        task,
    }))
}

fn write_help(out: &mut impl Write) -> Result<()> {
    writeln!(
        out,
        "NEAT symbol-copy run: cli [run] [OPTIONS]\n\
         Preflight only: cli plan [OPTIONS]\n\
         \n\
         --seed N                    evolutionary run seed\n\
         --population N              genomes per generation\n\
         --generations N             generations to evaluate\n\
         --population-checkpoint-interval N\n\
                                     persist every Nth full population\n\
         --workers N                 parallel genome evaluations\n\
         --stream SYMBOLS            replace default training corpus; repeat for cases\n\
                                     examples: abc  or  a,b,c,end\n\
                                     `end` is appended when omitted\n\
         --leaky-neurons             enable leaky hidden state\n\
         --seed-config PATH          seed genome TOML\n\
         --param key=value           NEAT override (valid: {PARAMS})"
    )
    .map_err(Into::into)
}

fn run_neat_plan(args: &[&str], out: &mut impl Write) -> Result<()> {
    let Some(request) = parse_neat_run(args, out)? else {
        return Ok(());
    };
    let seed_config = load_seed_config(&request.seed_config_path)?;
    let evaluations = request.neat.population_size as u128 * request.neat.generations as u128;
    let training_symbol_comparisons =
        evaluations * u128::from(request.task.config.training_symbols_per_genome());
    let holdout_symbol_comparisons = u128::from(request.neat.generations)
        * u128::from(request.task.config.holdout_symbols_per_winner());
    writeln!(
        out,
        "{}",
        json!({
            "mode": "neat_plan",
            "valid": true,
            "task": TASK_NAME,
            "objective": OBJECTIVE_NAME,
            "fitness": "number_of_exact_symbol_matches",
            "run_seed": request.run_seed,
            "population": request.neat.population_size,
            "generations": request.neat.generations,
            "training_streams": request.task.config.training_streams,
            "holdout_streams": request.task.config.holdout_streams,
            "training_stream_count": request.task.config.training_streams.len(),
            "holdout_stream_count": request.task.config.holdout_streams.len(),
            "training_symbols_per_genome": request.task.config.training_symbols_per_genome(),
            "holdout_symbols_per_winner": request.task.config.holdout_symbols_per_winner(),
            "genome_evaluations": evaluations,
            "training_symbol_comparisons": training_symbol_comparisons,
            "holdout_symbol_comparisons": holdout_symbol_comparisons,
            "multiplayer": false,
            "seed_genome": seed_config,
        })
    )?;
    Ok(())
}

fn execute_neat_run(request: NeatRunRequest, out_dir: &str, out: &mut impl Write) -> Result<()> {
    let seed_genome_config = load_seed_config(&request.seed_config_path)?;
    let total_generations = request.neat.generations;
    let started = Instant::now();
    eprintln!(
        "{}",
        json!({
            "event": "neat_started",
            "task": TASK_NAME,
            "population": request.neat.population_size,
            "generations": total_generations,
            "training_stream_count": request.task.config.training_streams.len(),
            "holdout_stream_count": request.task.config.holdout_streams.len(),
            "training_symbols_per_genome": request.task.config.training_symbols_per_genome(),
            "holdout_symbols_per_winner": request.task.config.holdout_symbols_per_winner(),
            "objective": OBJECTIVE_NAME,
            "multiplayer": false,
        })
    );
    let result = run_neat(
        &request.task,
        request.neat,
        seed_genome_config,
        request.run_seed,
        |generation| {
            let training = &generation.winner_evaluation;
            let holdout = generation
                .winner_validation
                .as_ref()
                .expect("symbol-copy task always supplies holdout evaluation");
            let elapsed = started.elapsed().as_secs_f64();
            let completed = generation.generation + 1;
            let mean_seconds = elapsed / f64::from(completed);
            let eta = mean_seconds * f64::from(total_generations.saturating_sub(completed));
            eprintln!(
                "{}",
                json!({
                    "event": "neat_generation",
                    "generation": generation.generation,
                    "training_fitness": generation.winner_fitness,
                    "training_accuracy": training.accuracy,
                    "holdout_correct": holdout.correct,
                    "holdout_total": holdout.total,
                    "holdout_accuracy": holdout.accuracy,
                    "training_example": training.streams.first(),
                    "holdout_example": holdout.streams.first(),
                    "species": generation.species.len(),
                    "hidden_nodes": generation.winner_hidden_nodes,
                    "enabled_connections": generation.winner_enabled_connections,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                })
            );
        },
    )?;

    let mut result_path = run_output_path(out_dir, "neat-symbol-copy")?;
    result_path.set_extension("json.zst");
    let result_path_string = result_path.to_string_lossy().into_owned();
    atomic_write(&result_path_string, |writer| {
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, &result)?;
        encoder.finish()?;
        Ok(())
    })?;
    let winner = result
        .final_population
        .get(result.final_winner_population_index)
        .ok_or_else(|| anyhow!("NEAT result has no final winner"))?;
    let validation = result
        .final_winner_validation
        .as_ref()
        .ok_or_else(|| anyhow!("symbol-copy result has no holdout evaluation"))?;
    writeln!(
        out,
        "{}",
        json!({
            "wrote": result_path_string,
            "objective": result.objective,
            "final_winner_population_index": winner.population_index,
            "final_training_fitness": winner.fitness,
            "final_training_total": winner.evaluation.total,
            "final_training_accuracy": winner.evaluation.accuracy,
            "final_holdout_correct": validation.correct,
            "final_holdout_total": validation.total,
            "final_holdout_accuracy": validation.accuracy,
            "training_example": winner.evaluation.streams.first(),
            "holdout_example": validation.streams.first(),
            "generations": result.generations.len(),
        })
    )?;
    Ok(())
}

fn run_neat_analysis(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.is_empty() {
        bail!("analyze needs at least one result.json.zst path");
    }
    let analyses = args
        .iter()
        .map(|path| {
            let result: SymbolCopyRunResult = serde_json::from_reader(result_reader(path)?)
                .map_err(|error| anyhow!("cannot parse NEAT result `{path}`: {error}"))?;
            let final_winner = result
                .final_population
                .get(result.final_winner_population_index)
                .ok_or_else(|| anyhow!("NEAT result `{path}` has no final winner"))?;
            let validation = result
                .final_winner_validation
                .as_ref()
                .ok_or_else(|| anyhow!("symbol-copy result `{path}` has no holdout evaluation"))?;
            Ok(json!({
                "path": path,
                "run_seed": result.seed,
                "task": result.task,
                "objective": result.objective,
                "training_stream_count": result.task_config.training_streams.len(),
                "holdout_stream_count": result.task_config.holdout_streams.len(),
                "generations": result.generations.len(),
                "final_training_fitness": final_winner.fitness,
                "final_training_total": final_winner.evaluation.total,
                "final_training_accuracy": final_winner.evaluation.accuracy,
                "final_holdout_correct": validation.correct,
                "final_holdout_total": validation.total,
                "final_holdout_accuracy": validation.accuracy,
                "trajectory": result.generations.iter().map(|generation| json!({
                    "generation": generation.generation,
                    "training_fitness": generation.winner_fitness,
                    "training_accuracy": generation.winner_evaluation.accuracy,
                    "holdout_accuracy": generation.winner_validation.as_ref().map(|evaluation| evaluation.accuracy),
                    "species": generation.species.len(),
                })).collect::<Vec<_>>(),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let value = if analyses.len() == 1 {
        analyses.into_iter().next().expect("one analysis")
    } else {
        json!({ "runs": analyses })
    };
    writeln!(out, "{value}")?;
    Ok(())
}

fn load_seed_config(path: &str) -> Result<SeedGenomeConfig> {
    config::load_seed_genome_config_from_path(Path::new(path))
}

fn result_reader(path: &str) -> Result<Box<dyn Read>> {
    let file = File::open(path).map_err(|error| anyhow!("cannot open `{path}`: {error}"))?;
    if path.ends_with(".zst") {
        Ok(Box::new(zstd::stream::read::Decoder::new(file)?))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn parse_stream(raw: &str) -> Result<Vec<Symbol>> {
    let normalized = raw.replace("->", ",");
    let mut symbols = if normalized.contains(',') || normalized.contains(char::is_whitespace) {
        normalized
            .split(|character: char| character == ',' || character.is_whitespace())
            .filter(|token| !token.is_empty())
            .map(parse_symbol)
            .collect::<Result<Vec<_>>>()?
    } else if normalized == "end" {
        vec![Symbol::End]
    } else {
        normalized
            .chars()
            .map(|character| parse_symbol(&character.to_string()))
            .collect::<Result<Vec<_>>>()?
    };
    if symbols.last() != Some(&Symbol::End) {
        symbols.push(Symbol::End);
    }
    if symbols[..symbols.len() - 1].contains(&Symbol::End) {
        bail!("`end` may appear only at the end of a stream");
    }
    Ok(symbols)
}

fn parse_symbol(raw: &str) -> Result<Symbol> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "a" => Ok(Symbol::A),
        "b" => Ok(Symbol::B),
        "c" => Ok(Symbol::C),
        "d" => Ok(Symbol::D),
        "e" => Ok(Symbol::E),
        "f" => Ok(Symbol::F),
        "g" => Ok(Symbol::G),
        "h" => Ok(Symbol::H),
        "end" => Ok(Symbol::End),
        other => bail!("unknown symbol `{other}`; valid: a b c d e f g h end"),
    }
}

fn value<'a>(args: &[&'a str], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index + 1)
        .copied()
        .ok_or_else(|| anyhow!("{flag} needs a value"))
}

fn parse_assignment(raw: &str) -> Result<(String, String)> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("expected key=value, got `{raw}`"))?;
    if key.trim().is_empty() || value.trim().is_empty() {
        bail!("expected non-empty key=value, got `{raw}`");
    }
    Ok((key.trim().to_owned(), value.trim().to_owned()))
}

pub(crate) fn apply_neat_param(config: &mut NeatConfig, key: &str, value: &str) -> Result<()> {
    match key {
        "compatibility_threshold" => config.compatibility_threshold = value.parse()?,
        "target_species" => config.target_species = value.parse()?,
        "compatibility_threshold_adjustment" => {
            config.compatibility_threshold_adjustment = value.parse()?
        }
        "excess_coefficient" => config.excess_coefficient = value.parse()?,
        "disjoint_coefficient" => config.disjoint_coefficient = value.parse()?,
        "weight_coefficient" => config.weight_coefficient = value.parse()?,
        "learning_coefficient" => config.learning_coefficient = value.parse()?,
        "survival_fraction" => config.survival_fraction = value.parse()?,
        "crossover_probability" => config.crossover_probability = value.parse()?,
        "interspecies_mate_probability" => config.interspecies_mate_probability = value.parse()?,
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
        "mutate_learning_rate_probability" => {
            config.mutate_learning_rate_probability = value.parse()?
        }
        "learning_rate_perturb_stddev" => config.learning_rate_perturb_stddev = value.parse()?,
        "mutate_plasticity_coefficient_probability" => {
            config.mutate_plasticity_coefficient_probability = value.parse()?
        }
        "plasticity_coefficient_perturb_stddev" => {
            config.plasticity_coefficient_perturb_stddev = value.parse()?
        }
        "add_connection_probability" => config.add_connection_probability = value.parse()?,
        "add_node_probability" => config.add_node_probability = value.parse()?,
        "disabled_inheritance_probability" => {
            config.disabled_inheritance_probability = value.parse()?
        }
        "elitism_min_species_size" => config.elitism_min_species_size = value.parse()?,
        _ => bail!("unknown NEAT parameter `{key}`; valid: {PARAMS}"),
    }
    Ok(())
}
