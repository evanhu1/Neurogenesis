use crate::run_output_path;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::json;
use sim_core::conditional::{run_conditional_program_experiment, ConditionalProgramConfig};
use std::io::Write;

pub(crate) fn run_conditional_program_cli(
    args: &[&str],
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let mut config = ConditionalProgramConfig::default();
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--outer-seeds" => {
                let raw = args
                    .get(index + 1)
                    .ok_or_else(|| anyhow!("--outer-seeds needs N,N,..."))?;
                config.outer_seeds = raw
                    .split(',')
                    .map(|part| {
                        part.trim()
                            .parse::<u64>()
                            .with_context(|| format!("invalid outer seed `{part}`"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                index += 2;
            }
            "--stages" => {
                config.stage_budget = parse_next(args, index, "--stages")?;
                index += 2;
            }
            "--search-budget" => {
                config.search_budget = parse_next(args, index, "--search-budget")?;
                index += 2;
            }
            "--starting-rank" => {
                config.starting_rank = parse_next(args, index, "--starting-rank")?;
                index += 2;
            }
            "--delay" => {
                config.empty_delay_ticks = parse_next(args, index, "--delay")?;
                index += 2;
            }
            "--escrow" => {
                config.escrow_energy = parse_next(args, index, "--escrow")?;
                index += 2;
            }
            "--ecology-horizon" => {
                config.ecology_horizon = parse_next(args, index, "--ecology-horizon")?;
                index += 2;
            }
            "--help" | "-h" => {
                writeln!(
                    out,
                    "conditional-program options:\n  --outer-seeds N,N  unique independent outer seeds (default 7,42,123)\n  --stages N          runtime stage budget; not a task-grammar depth cap\n  --search-budget N   deterministic proposals per frozen task\n  --starting-rank N   first cue/response sequence length (minimum 4)\n  --delay N           mandatory empty pose-reset ticks (minimum 1)\n  --escrow E          fixed all-or-nothing episode reward (minimum 1)\n  --ecology-horizon N per-seed paired fixed-ecology noninferiority horizon\n  --out-dir D         durable result directory (global flag)"
                )?;
                return Ok(());
            }
            other => bail!("unknown conditional-program arg `{other}`"),
        }
    }

    let result =
        run_conditional_program_experiment(config).map_err(|error| anyhow!(error.to_string()))?;
    let path = run_output_path(out_dir, "conditional-program")?;
    let file = std::fs::File::create(&path)
        .with_context(|| format!("cannot create `{}`", path.display()))?;
    serde_json::to_writer_pretty(file, &result)
        .with_context(|| format!("cannot write `{}`", path.display()))?;
    writeln!(
        out,
        "{}",
        json!({
            "result": path,
            "schema": result.schema,
            "outer_seeds": result.summary.outer_seed_count,
            "qualifying_discoveries": result.summary.qualifying_discoveries,
            "maximum_accepted_rank": result.summary.maximum_accepted_rank,
            "seeds_with_at_least_one_discovery": result.summary.seeds_with_at_least_one_discovery,
            "open_endedness_demonstrated": result.summary.open_endedness_demonstrated,
            "terminal_failure_modes": result.summary.terminal_failure_modes,
        })
    )?;
    Ok(())
}

fn parse_next<T>(args: &[&str], index: usize, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.get(index + 1)
        .ok_or_else(|| anyhow!("{flag} needs a value"))?
        .parse::<T>()
        .map_err(|error| anyhow!("invalid value for {flag}: {error}"))
}
