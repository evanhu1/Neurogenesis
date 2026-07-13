use crate::{run_output_path, DEFAULT_CONFIG};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sim_config::load_world_config_from_path;
use sim_core::powerplay::{run_powerplay, PowerPlayConfig};
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
