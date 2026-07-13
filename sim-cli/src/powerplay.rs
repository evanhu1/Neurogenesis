use crate::{run_output_path, DEFAULT_CONFIG};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sim_config::load_world_config_from_path;
use sim_core::powerplay::{
    run_powerplay, run_public_decoder_probe, run_public_preamble_probe, PowerPlayConfig,
    PublicDecoderProbeConfig, PublicPreambleProbeConfig,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub(crate) fn run_powerplay_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut config = PowerPlayConfig::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i] {
            "--config" => {
                config_path = value(args, i, "--config")?.to_string();
                i += 2;
            }
            "--seed" => {
                config.run_seed = value(args, i, "--seed")?.parse()?;
                i += 2;
            }
            "--depth" => {
                config.max_depth = value(args, i, "--depth")?.parse()?;
                i += 2;
            }
            "--population" => {
                config.population_size = value(args, i, "--population")?.parse()?;
                i += 2;
            }
            "--generations" => {
                config.generations_per_depth = value(args, i, "--generations")?.parse()?;
                i += 2;
            }
            "--module-width" => {
                config.module_width = value(args, i, "--module-width")?.parse()?;
                i += 2;
            }
            "--ticks-per-stage" => {
                config.ticks_per_stage = value(args, i, "--ticks-per-stage")?.parse()?;
                i += 2;
            }
            "--world-width" => {
                config.world_width = value(args, i, "--world-width")?.parse()?;
                i += 2;
            }
            "--food-energy" => {
                config.food_energy = value(args, i, "--food-energy")?.parse()?;
                i += 2;
            }
            "--episode-seeds" => {
                config.episode_seeds = parse_seed_suite(value(args, i, "--episode-seeds")?)?;
                i += 2;
            }
            "--search-seeds" => {
                config.search_seeds = parse_seed_suite(value(args, i, "--search-seeds")?)?;
                i += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "powerplay options: --config P --seed N --depth 1..4 --population N \
                     --generations N --module-width N --ticks-per-stage N \
                     --world-width N --food-energy F --search-seeds N,... \
                     --episode-seeds N,...\n\
                     This is a bounded causal-ecology vertical slice, not an \
                     open-endedness claim. Admission is fixed at >=14/16 seeds; \
                     every earlier checkpoint must fail a new task on <=2/16."
                )?;
                return Ok(());
            }
            other => bail!("unknown powerplay arg `{other}` (use `powerplay --help`)"),
        }
    }
    if config.episode_seeds.len() != 16 {
        bail!("powerplay's adversarial gate requires exactly 16 episode seeds");
    }

    let world = load_world_config_from_path(Path::new(&config_path))?;
    eprintln!(
        "{}",
        json!({
            "event": "powerplay_started",
            "claim_scope": "bounded depth-1..4 pilot; not open-endedness evidence",
            "run_seed": config.run_seed,
            "max_depth": config.max_depth,
            "population_size": config.population_size,
            "generations_per_depth": config.generations_per_depth,
            "module_width": config.module_width,
            "search_seeds": config.search_seeds,
            "episode_seeds": config.episode_seeds,
            "admission_pass_count": 14,
            "historical_solver_fail_max_count": 2,
        })
    );
    let result = run_powerplay(world, config)?;
    let path = run_output_path(out_dir, "powerplay")?;
    let file = File::create(&path)
        .map_err(|error| anyhow!("cannot create `{}`: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &result)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writeln!(
        out,
        "{}",
        json!({
            "result": path,
            "accepted_depth": result.accepted_depth,
            "requested_depth": result.config.max_depth,
            "stopped_reason": result.stopped_reason,
            "claim_scope": result.claim_scope,
        })
    )?;
    Ok(())
}

pub(crate) fn run_public_preamble_probe_cli(
    args: &[&str],
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut config = PublicPreambleProbeConfig::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i] {
            "--config" => {
                config_path = value(args, i, "--config")?.to_string();
                i += 2;
            }
            "--run-seeds" => {
                config.source_run_seeds = parse_seed_suite(value(args, i, "--run-seeds")?)?;
                i += 2;
            }
            "--population" => {
                config.source_powerplay.population_size =
                    value(args, i, "--population")?.parse()?;
                i += 2;
            }
            "--generations" => {
                config.source_powerplay.generations_per_depth =
                    value(args, i, "--generations")?.parse()?;
                i += 2;
            }
            "--module-width" => {
                config.source_powerplay.module_width = value(args, i, "--module-width")?.parse()?;
                i += 2;
            }
            "--ticks-per-stage" => {
                config.source_powerplay.ticks_per_stage =
                    value(args, i, "--ticks-per-stage")?.parse()?;
                i += 2;
            }
            "--world-width" => {
                config.source_powerplay.world_width = value(args, i, "--world-width")?.parse()?;
                i += 2;
            }
            "--food-energy" => {
                config.source_powerplay.food_energy = value(args, i, "--food-energy")?.parse()?;
                i += 2;
            }
            "--episode-seeds" => {
                config.source_powerplay.episode_seeds =
                    parse_seed_suite(value(args, i, "--episode-seeds")?)?;
                i += 2;
            }
            "--search-seeds" => {
                config.source_powerplay.search_seeds =
                    parse_seed_suite(value(args, i, "--search-seeds")?)?;
                i += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "public-preamble-probe options: --config P --run-seeds N,... \
                     --population N --generations N --module-width N \
                     --ticks-per-stage N --world-width N --food-energy F \
                     --search-seeds N,... --episode-seeds N,...\n\
                     Reconstructs exact accepted depth-1/depth-2 PowerPlay \
                     solvers, then compares a fixed task-program FoodRay \
                     preamble with blank and cue-permuted controls on 16 \
                     disjoint contexts. This checks only zero-shot compatibility \
                     of legacy checkpoints; it is never evidence about trained \
                     decoder capacity, transfer, or open-endedness."
                )?;
                return Ok(());
            }
            other => bail!(
                "unknown public-preamble-probe arg `{other}` (use `public-preamble-probe --help`)"
            ),
        }
    }

    let world = load_world_config_from_path(Path::new(&config_path))?;
    eprintln!(
        "{}",
        json!({
            "event": "public_preamble_probe_started",
            "claim_scope": "evaluator-owned zero-shot legacy-checkpoint compatibility; not trained-capacity, transfer, or open-endedness evidence",
            "source_run_seeds": config.source_run_seeds,
            "source_depth": config.source_powerplay.max_depth,
            "population_size": config.source_powerplay.population_size,
            "generations_per_depth": config.source_powerplay.generations_per_depth,
            "strict_gate": {
                "meaningful_minimum": 14,
                "blank_maximum": 2,
                "permuted_maximum": 2,
            },
        })
    );
    let result = run_public_preamble_probe(world, config)?;
    let path = run_output_path(out_dir, "public-preamble-probe")?;
    let file = File::create(&path)
        .map_err(|error| anyhow!("cannot create `{}`: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &result)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writeln!(
        out,
        "{}",
        json!({
            "result": path,
            "verdict": result.verdict,
            "verdict_reason": result.verdict_reason,
            "branch_transfer_status": result.branch_transfer_status,
            "requested_pair_count": result.requested_pair_count,
            "evaluated_pair_count": result.evaluated_pair_count,
            "passed_pair_count": result.passed_pair_count,
            "evaluator_owned": result.evaluator_owned,
            "evidentiary": result.evidentiary,
            "open_endedness_demonstrated": result.open_endedness_demonstrated,
        })
    )?;
    Ok(())
}

pub(crate) fn run_public_decoder_probe_cli(
    args: &[&str],
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut config = PublicDecoderProbeConfig::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i] {
            "--config" => {
                config_path = value(args, i, "--config")?.to_string();
                i += 2;
            }
            "--source-seed" => {
                config.source_powerplay.run_seed = value(args, i, "--source-seed")?.parse()?;
                i += 2;
            }
            "--source-population" => {
                config.source_powerplay.population_size =
                    value(args, i, "--source-population")?.parse()?;
                i += 2;
            }
            "--source-generations" => {
                config.source_powerplay.generations_per_depth =
                    value(args, i, "--source-generations")?.parse()?;
                i += 2;
            }
            "--source-module-width" => {
                config.source_powerplay.module_width =
                    value(args, i, "--source-module-width")?.parse()?;
                i += 2;
            }
            "--decoder-population" => {
                config.decoder_population_size = value(args, i, "--decoder-population")?.parse()?;
                i += 2;
            }
            "--decoder-generations" => {
                config.decoder_generations = value(args, i, "--decoder-generations")?.parse()?;
                i += 2;
            }
            "--decoder-module-width" => {
                config.decoder_module_width = value(args, i, "--decoder-module-width")?.parse()?;
                i += 2;
            }
            "--ticks-per-stage" => {
                config.source_powerplay.ticks_per_stage =
                    value(args, i, "--ticks-per-stage")?.parse()?;
                i += 2;
            }
            "--world-width" => {
                config.source_powerplay.world_width = value(args, i, "--world-width")?.parse()?;
                i += 2;
            }
            "--food-energy" => {
                config.source_powerplay.food_energy = value(args, i, "--food-energy")?.parse()?;
                i += 2;
            }
            "--search-seeds" => {
                config.source_powerplay.search_seeds =
                    parse_seed_suite(value(args, i, "--search-seeds")?)?;
                i += 2;
            }
            "--episode-seeds" => {
                config.source_powerplay.episode_seeds =
                    parse_seed_suite(value(args, i, "--episode-seeds")?)?;
                i += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "public-decoder-probe options: --config P --source-seed N \
                     --source-population N --source-generations N \
                     --source-module-width N --decoder-population N \
                     --decoder-generations N --decoder-module-width N \
                     --ticks-per-stage N --world-width N --food-energy F \
                     --search-seeds N,... --episode-seeds N,...\n\
                     Trains one protected residual to decode valid two-stage \
                     public programs into an identical-scene declaration, \
                     performs one sealed held-out recombination audit with \
                     blank/polarity/code-swap/source/knockout controls and \
                     ordinary-task retention, then attempts exact module \
                     exact reuse in the descendant depth-2 checkpoint only if the \
                     decoder gate passes. Evaluator-owned; not TCPE or \
                     open-endedness evidence."
                )?;
                return Ok(());
            }
            other => bail!(
                "unknown public-decoder-probe arg `{other}` (use `public-decoder-probe --help`)"
            ),
        }
    }

    let world = load_world_config_from_path(Path::new(&config_path))?;
    eprintln!(
        "{}",
        json!({
            "event": "public_decoder_probe_started",
            "claim_scope": "evaluator-owned protected decoder and descendant-checkpoint reuse falsifier",
            "source_seed": config.source_powerplay.run_seed,
            "source_population": config.source_powerplay.population_size,
            "source_generations": config.source_powerplay.generations_per_depth,
            "decoder_population": config.decoder_population_size,
            "decoder_generations": config.decoder_generations,
            "decoder_module_width": config.decoder_module_width,
            "gate": {
                "meaningful_minimum": config.meaningful_minimum,
                "code_swap_minimum": config.code_swap_minimum,
                "control_maximum": config.control_maximum,
                "ordinary_retention_minimum": config.ordinary_retention_minimum,
            },
        })
    );
    let result = run_public_decoder_probe(world, config)?;
    let path = run_output_path(out_dir, "public-decoder-probe")?;
    let file = File::create(&path)
        .map_err(|error| anyhow!("cannot create `{}`: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &result)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writeln!(
        out,
        "{}",
        json!({
            "result": path,
            "verdict": result.verdict,
            "verdict_reason": result.verdict_reason,
            "decoder_gate_passed": result.training.sealed_decoder_gate_passed,
            "descendant_checkpoint_reuse_status": result.descendant_checkpoint_reuse.status,
            "descendant_checkpoint_reuse_passed": result.descendant_checkpoint_reuse.passed,
            "evaluator_owned": result.evaluator_owned,
            "evidentiary": result.evidentiary,
            "open_endedness_demonstrated": result.open_endedness_demonstrated,
        })
    )?;
    Ok(())
}

fn value<'a>(args: &'a [&str], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index + 1)
        .copied()
        .ok_or_else(|| anyhow!("{flag} needs a value"))
}

fn parse_seed_suite(raw: &str) -> Result<Vec<u64>> {
    let seeds = raw
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim().parse::<u64>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    if seeds.is_empty() {
        bail!("--episode-seeds must be nonempty");
    }
    Ok(seeds)
}
