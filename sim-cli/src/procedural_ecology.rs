use crate::{run_output_path, DEFAULT_CONFIG};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sim_config::load_world_config_from_path;
use sim_core::procedural_ecology::{run_procedural_ecology_stage0, ProceduralEcologyStage0Config};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub(crate) fn run_procedural_ecology_stage0_cli(
    args: &[&str],
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let mut config_path = DEFAULT_CONFIG.to_owned();
    let mut config = ProceduralEcologyStage0Config::default();
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--config" => {
                config_path = value(args, index, "--config")?.to_owned();
                index += 2;
            }
            "--run-seeds" => {
                config.run_seeds = parse_seed_suite(value(args, index, "--run-seeds")?)?;
                index += 2;
            }
            "--horizon" => {
                config.horizon_ticks = value(args, index, "--horizon")?.parse()?;
                index += 2;
            }
            "--release-interval" => {
                config.release_interval_ticks =
                    value(args, index, "--release-interval")?.parse()?;
                index += 2;
            }
            "--plant-energy" => {
                config.plant_energy = value(args, index, "--plant-energy")?.parse()?;
                index += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "procedural-ecology-stage0 options: --config P --run-seeds N,... \
                     --horizon N --release-interval N --plant-energy F\n\
                     Runs stationary, moving-front, consumption-responsive, \
                     behavior-input-clamped, translated, and duplicate-replay \
                     cases with a fixed release schedule and explicit ecology \
                     energy escrow. This is a mechanics gate, never an \
                     evolutionary or open-endedness result."
                )?;
                return Ok(());
            }
            other => bail!(
                "unknown procedural-ecology-stage0 arg `{other}` (use `procedural-ecology-stage0 --help`)"
            ),
        }
    }

    let world = load_world_config_from_path(Path::new(&config_path))?;
    eprintln!(
        "{}",
        json!({
            "event": "procedural_ecology_stage0_started",
            "claim_scope": "evaluator-owned mechanics gate; not an evolutionary or open-endedness result",
            "run_seeds": config.run_seeds,
            "horizon_ticks": config.horizon_ticks,
            "release_interval_ticks": config.release_interval_ticks,
            "plant_energy": config.plant_energy,
            "policies": ["stationary", "moving_front", "consumption_responsive"],
            "controls": ["behavior_input_clamp", "translation", "duplicate_replay", "empty_disabled_hook_noop"],
        })
    );
    let result = run_procedural_ecology_stage0(world, config)?;
    let path = run_output_path(out_dir, "procedural-ecology-stage0")?;
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
            "all_gates_passed": result.gates.all_passed,
            "gates": result.gates,
            "stage_1_authorized": result.stage_1_authorized,
            "open_endedness_demonstrated": result.open_endedness_demonstrated,
            "result_fingerprint": result.result_fingerprint,
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
        bail!("--run-seeds must be nonempty");
    }
    Ok(seeds)
}
