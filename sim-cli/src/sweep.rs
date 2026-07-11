//! `sweep` run mode: run a cartesian grid of config overrides × seeds, score
//! each cell's pillars (mean ± spread across the seed cohort, matching how the
//! eval scores), and write a run-result file. Jobs run in parallel, bounded to
//! the machine's parallelism; each job uses a single intent thread so the
//! parallelism comes from running many independent worlds at once.

use crate::run_output_path;
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use sim_core::Simulation;
use sim_metrics::{
    compute_pillar_scores, derive_interval_metrics, ingest_tick, register_founders, Ledger,
    PillarScores, TickSummaryRow,
};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The raw windowed metrics a sweep ranks on (no [0,1] interpretation).
const METRIC_KEYS: [&str; 5] = [
    "plant_consumption_rate",
    "prey_consumption_rate",
    "action_effectiveness",
    "mi_sa",
    "learning_slope",
];

fn metric(p: &PillarScores, key: &str) -> f64 {
    let value = match key {
        "plant_consumption_rate" => p.mean_plant_consumption_rate,
        "prey_consumption_rate" => p.mean_prey_consumption_rate,
        "action_effectiveness" => p.mean_action_effectiveness,
        "mi_sa" => p.mean_mi_sa,
        "learning_slope" => p.mean_learning_slope,
        other => unreachable!("metric() called with key outside METRIC_KEYS: {other}"),
    };
    value.unwrap_or(f64::NAN)
}

struct Job {
    cell_idx: usize,
    overrides: Vec<(String, String)>,
    seed: u64,
}

struct JobOutcome {
    cell_idx: usize,
    seed: u64,
    pillars: Option<PillarScores>,
    error: Option<String>,
}

pub fn run_sweep(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    let mut grid: Vec<(String, Vec<String>)> = Vec::new();
    let mut seeds: Vec<u64> = Vec::new();
    let mut to: Option<u64> = None;
    let mut config_path = crate::DEFAULT_CONFIG.to_string();
    let mut report_every: u64 = crate::REPORT_EVERY;
    let mut baseline: Option<Vec<(String, String)>> = None;
    let mut threads: u32 = 1;
    let mut jobs_cap: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--grid" => {
                // Consume KEY=v1,v2[,..] tokens until the next `--flag`.
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    let (k, vlist) = args[i]
                        .split_once('=')
                        .ok_or_else(|| anyhow!("--grid wants KEY=v1,v2 (got `{}`)", args[i]))?;
                    let values: Vec<String> =
                        vlist.split(',').map(|s| s.trim().to_string()).collect();
                    if values.iter().any(|v| v.is_empty()) {
                        bail!("--grid `{k}` has an empty value");
                    }
                    let key = k.trim().to_string();
                    if grid.iter().any(|(existing, _)| *existing == key) {
                        bail!("--grid key `{key}` given twice");
                    }
                    grid.push((key, values));
                    i += 1;
                }
            }
            "--seeds" => {
                let spec = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--seeds needs a list"))?;
                for s in spec.split(',') {
                    seeds.push(s.trim().parse().map_err(|_| anyhow!("bad seed `{s}`"))?);
                }
                i += 2;
            }
            "--to" => {
                to = Some(
                    args.get(i + 1)
                        .ok_or_else(|| anyhow!("--to needs a tick"))?
                        .parse()?,
                );
                i += 2;
            }
            "--config" => {
                config_path = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--config needs a path"))?
                    .to_string();
                i += 2;
            }
            "--report-every" => {
                report_every = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--report-every needs a value"))?
                    .parse()?;
                if report_every == 0 {
                    bail!("--report-every must be >= 1");
                }
                i += 2;
            }
            "--baseline" => {
                let spec = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--baseline needs KEY=v[,KEY2=v]"))?;
                baseline = Some(parse_overrides(spec)?);
                i += 2;
            }
            "--threads" => {
                threads = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--threads needs a value"))?
                    .parse()?;
                if threads == 0 {
                    bail!("--threads must be >= 1");
                }
                i += 2;
            }
            "--jobs" => {
                jobs_cap = Some(
                    args.get(i + 1)
                        .ok_or_else(|| anyhow!("--jobs needs a value"))?
                        .parse()?,
                );
                i += 2;
            }
            "--json" | "--text" => i += 1, // output is a file; format flag is a no-op here
            other => bail!("unknown sweep arg `{other}`"),
        }
    }

    if grid.is_empty() {
        bail!("sweep needs --grid KEY=v1,v2 [KEY2=...]");
    }
    if seeds.is_empty() {
        bail!("sweep needs --seeds N[,N...]");
    }
    let to = to.ok_or_else(|| anyhow!("sweep needs --to TICK"))?;

    // Read the base config once; each cell patches it with its overrides.
    let world_raw = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("reading config `{config_path}`: {e}"))?;
    let seed_genome_path = Path::new(&config_path).with_file_name("seed_genome.toml");
    let seed_genome_raw = std::fs::read_to_string(&seed_genome_path)
        .map_err(|e| anyhow!("reading {}: {e}", seed_genome_path.display()))?;

    // Cartesian product of the grid → one override-set per cell.
    let cells = cartesian(&grid);
    // A --baseline that matches no cell is a user error (typo / wrong key set);
    // fail loudly rather than silently computing deltas against cell 0.
    let baseline_idx = match baseline.as_ref() {
        Some(b) => cells
            .iter()
            .position(|c| same_overrides(c, b))
            .ok_or_else(|| anyhow!("--baseline {:?} matches no cell in the grid", b))?,
        None => 0,
    };

    // Flatten to (cell, seed) jobs and run them in a bounded parallel pool.
    let jobs: Vec<Job> = cells
        .iter()
        .enumerate()
        .flat_map(|(cell_idx, overrides)| {
            seeds.iter().map(move |&seed| Job {
                cell_idx,
                overrides: overrides.clone(),
                seed,
            })
        })
        .collect();

    let parallelism = jobs_cap.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let total_jobs = jobs.len();
    writeln!(
        out,
        "sweep: {} cells × {} seeds = {} runs to tick {to}, {} parallel (threads/run={threads})…",
        cells.len(),
        seeds.len(),
        total_jobs,
        parallelism.min(total_jobs).max(1),
    )?;
    out.flush()?;

    let next = AtomicUsize::new(0);
    let outcomes: Mutex<Vec<JobOutcome>> = Mutex::new(Vec::with_capacity(total_jobs));
    let n_workers = parallelism.clamp(1, total_jobs.max(1));

    std::thread::scope(|scope| {
        for _ in 0..n_workers {
            scope.spawn(|| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= jobs.len() {
                    break;
                }
                let job = &jobs[idx];
                let outcome = match run_cell_seed(
                    &world_raw,
                    &seed_genome_raw,
                    &job.overrides,
                    job.seed,
                    to,
                    report_every,
                    threads,
                ) {
                    Ok(pillars) => JobOutcome {
                        cell_idx: job.cell_idx,
                        seed: job.seed,
                        pillars: Some(pillars),
                        error: None,
                    },
                    Err(e) => JobOutcome {
                        cell_idx: job.cell_idx,
                        seed: job.seed,
                        pillars: None,
                        error: Some(e.to_string()),
                    },
                };
                outcomes.lock().unwrap().push(outcome);
            });
        }
    });

    let outcomes = outcomes.into_inner().unwrap();
    let result = build_result(&cells, baseline_idx, &seeds, to, &config_path, &outcomes);

    // Persist the run-result file and tell the caller where it landed.
    let path = run_output_path(out_dir, "sweep")?;
    let file = std::fs::File::create(&path).map_err(|e| anyhow!("writing {path:?}: {e}"))?;
    serde_json::to_writer_pretty(file, &result)?;

    writeln!(
        out,
        "{}",
        json!({ "wrote": path.to_string_lossy(), "runs": total_jobs })
    )?;
    print_table(out, &result)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_cell_seed(
    world_raw: &str,
    seed_genome_raw: &str,
    overrides: &[(String, String)],
    seed: u64,
    to: u64,
    report_every: u64,
    threads: u32,
) -> Result<PillarScores> {
    let mut config =
        sim_views::world_config_from_raw_overrides(world_raw, seed_genome_raw, overrides)?;
    config.intent_parallel_threads = threads;
    let mut sim = Simulation::new(config, seed).map_err(|e| anyhow!("{e}"))?;

    let mut ledger = Ledger::new();
    register_founders(&mut ledger, sim.organisms());
    let mut tick_summary: Vec<TickSummaryRow> = Vec::new();
    let mut lifetimes = Vec::new();
    for _ in 0..to {
        let delta = sim.tick();
        let turn = sim.turn();
        lifetimes.extend(ingest_tick(&mut ledger, turn, &delta, sim.action_records()));
        tick_summary.push(TickSummaryRow {
            tick: turn,
            population: ledger.population(),
        });
    }
    // Include organisms still alive at `to` (the eval drains survivors before
    // scoring), so sweep pillar values match an eval run of the same config/seed,
    // not just rank consistently.
    lifetimes.extend(ledger.drain_survivors());
    let intervals = derive_interval_metrics(&tick_summary, &lifetimes, report_every, to);
    Ok(compute_pillar_scores(&intervals))
}

/// Cartesian product of `[(key, [values])]` → `[[(key, value)]]`.
fn cartesian(grid: &[(String, Vec<String>)]) -> Vec<Vec<(String, String)>> {
    let mut out: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for (key, values) in grid {
        let mut next = Vec::with_capacity(out.len() * values.len());
        for base in &out {
            for v in values {
                let mut row = base.clone();
                row.push((key.clone(), v.clone()));
                next.push(row);
            }
        }
        out = next;
    }
    out
}

fn parse_overrides(spec: &str) -> Result<Vec<(String, String)>> {
    spec.split(',')
        .map(|kv| {
            kv.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .ok_or_else(|| anyhow!("override wants KEY=value (got `{kv}`)"))
        })
        .collect()
}

fn same_overrides(a: &[(String, String)], b: &[(String, String)]) -> bool {
    a.len() == b.len()
        && b.iter()
            .all(|(k, v)| a.iter().any(|(ak, av)| ak == k && av == v))
}

/// Aggregate per cell across seeds: mean/min/max for each metric, plus delta of
/// the mean vs the baseline cell.
fn build_result(
    cells: &[Vec<(String, String)>],
    baseline_idx: usize,
    seeds: &[u64],
    to: u64,
    config_path: &str,
    outcomes: &[JobOutcome],
) -> Value {
    // Per-cell collected pillar scores.
    let mut by_cell: Vec<Vec<&PillarScores>> = vec![Vec::new(); cells.len()];
    let mut errors: Vec<Value> = Vec::new();
    for o in outcomes {
        match &o.pillars {
            Some(p) => by_cell[o.cell_idx].push(p),
            None => errors.push(json!({
                "cell": overrides_obj(&cells[o.cell_idx]),
                "seed": o.seed,
                "error": o.error.clone().unwrap_or_default(),
            })),
        }
    }

    let mean_of = |scores: &[&PillarScores], key: &str| -> Option<f64> {
        let vals: Vec<f64> = scores
            .iter()
            .map(|p| metric(p, key))
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    };

    let baseline_means: Vec<Option<f64>> = METRIC_KEYS
        .iter()
        .map(|k| mean_of(&by_cell[baseline_idx], k))
        .collect();

    let cell_values: Vec<Value> = cells
        .iter()
        .enumerate()
        .map(|(idx, overrides)| {
            let scores = &by_cell[idx];
            let mut metrics = serde_json::Map::new();
            let mut delta = serde_json::Map::new();
            for (mi, key) in METRIC_KEYS.iter().enumerate() {
                let vals: Vec<f64> = scores
                    .iter()
                    .map(|p| metric(p, key))
                    .filter(|v| v.is_finite())
                    .collect();
                let mean = if vals.is_empty() {
                    None
                } else {
                    Some(vals.iter().sum::<f64>() / vals.len() as f64)
                };
                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                metrics.insert(
                    key.to_string(),
                    json!({
                        "mean": mean,
                        "min": if vals.is_empty() { Value::Null } else { json!(min) },
                        "max": if vals.is_empty() { Value::Null } else { json!(max) },
                        "n": vals.len(),
                    }),
                );
                if let (Some(m), Some(b)) = (mean, baseline_means[mi]) {
                    delta.insert(key.to_string(), json!(m - b));
                }
            }
            json!({
                "overrides": overrides_obj(overrides),
                "is_baseline": idx == baseline_idx,
                "metrics": metrics,
                "delta_vs_baseline": delta,
            })
        })
        .collect();

    json!({
        "mode": "sweep",
        "config": config_path,
        "to": to,
        "seeds": seeds,
        "baseline": overrides_obj(&cells[baseline_idx]),
        "cells": cell_values,
        "errors": errors,
    })
}

fn overrides_obj(overrides: &[(String, String)]) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in overrides {
        m.insert(k.clone(), json!(v));
    }
    Value::Object(m)
}

/// Compact stdout summary so the agent sees the ranking without opening the file.
fn print_table(out: &mut impl Write, result: &Value) -> Result<()> {
    let cells = result["cells"].as_array().cloned().unwrap_or_default();
    writeln!(
        out,
        "{:<32} {:>9} {:>9} {:>9} {:>9} {:>11} {:>12}",
        "cell", "plant", "prey", "act_eff", "mi_sa", "learn_slope", "Δplant_rate"
    )?;
    for c in &cells {
        let label = c["overrides"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        let m = |key: &str| c["metrics"][key]["mean"].as_f64();
        let fmt = |v: Option<f64>| v.map(|x| format!("{x:.4}")).unwrap_or_else(|| "NA".into());
        let fmt_slope =
            |v: Option<f64>| v.map(|x| format!("{x:.6}")).unwrap_or_else(|| "NA".into());
        let dplant = c["delta_vs_baseline"]["plant_consumption_rate"].as_f64();
        let star = if c["is_baseline"].as_bool().unwrap_or(false) {
            "*"
        } else {
            " "
        };
        writeln!(
            out,
            "{star}{:<31} {:>9} {:>9} {:>9} {:>9} {:>11} {:>12}",
            label,
            fmt(m("plant_consumption_rate")),
            fmt(m("prey_consumption_rate")),
            fmt(m("action_effectiveness")),
            fmt(m("mi_sa")),
            fmt_slope(m("learning_slope")),
            dplant
                .map(|x| format!("{x:+.4}"))
                .unwrap_or_else(|| "—".into()),
        )?;
    }
    Ok(())
}
