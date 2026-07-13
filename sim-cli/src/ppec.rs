use crate::{run_output_path, DEFAULT_CONFIG};
use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sim_config::load_world_config_from_path;
use sim_core::ppec::{run_ppec_mechanism_experiment, PpecMechanismExperimentConfig};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub(crate) fn run_ppec_mechanism_cli(
    args: &[&str],
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let mut config_path = DEFAULT_CONFIG.to_owned();
    let mut config = PpecMechanismExperimentConfig::default();
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
            "--contexts" => {
                config.contexts_per_seed = value(args, index, "--contexts")?.parse()?;
                index += 2;
            }
            "--persistence-ticks" => {
                config.persistence_ticks = value(args, index, "--persistence-ticks")?.parse()?;
                index += 2;
            }
            "--cache-fraction" => {
                config.cache_energy_fraction = value(args, index, "--cache-fraction")?.parse()?;
                index += 2;
            }
            "--interaction-cost" => {
                config.protocol_interaction_energy_cost =
                    value(args, index, "--interaction-cost")?.parse()?;
                index += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "ppec-mechanism options: --config P --run-seeds N,... \
                     --contexts N --persistence-ticks N --cache-fraction F \
                     --interaction-cost F\n\
                     Runs an evaluator-owned Stage-0 persistent public cache \
                     experiment with own/foreign use and no-payoff, code-\
                     permutation, challenge-permutation, artifact-knockout, \
                     constant-0/1/2/3, and random-response \
                     controls. It is mechanism engagement, never open-endedness evidence."
                )?;
                return Ok(());
            }
            other => bail!("unknown ppec-mechanism arg `{other}` (use `ppec-mechanism --help`)"),
        }
    }

    let world = load_world_config_from_path(Path::new(&config_path))?;
    eprintln!(
        "{}",
        json!({
            "event": "ppec_mechanism_started",
            "claim_scope": "evaluator-owned Stage-0 mechanism engagement; not open-endedness evidence",
            "run_seeds": config.run_seeds,
            "contexts_per_seed": config.contexts_per_seed,
            "persistence_ticks": config.persistence_ticks,
            "controls": [
                "own_protocol",
                "foreign_protocol",
                "no_payoff",
                "code_permutation",
                "challenge_permutation",
                "artifact_knockout",
                "constant_response_0",
                "constant_response_1",
                "constant_response_2",
                "constant_response_3",
                "random_response"
            ],
        })
    );
    let result = run_ppec_mechanism_experiment(world, config)?;
    let path = run_output_path(out_dir, "ppec-mechanism")?;
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
            "evaluator_owned": result.evaluator_owned,
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
