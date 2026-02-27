mod ledger;
mod metrics;
mod report;

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use ledger::Ledger;
use metrics::{compute_interval_metrics, IntervalMetrics};
use report::{write_html_report, HtmlReportMeta, Reporter};
use serde::Serialize;
use sim_config::load_world_config_from_path;
use sim_core::Simulation;
use sim_types::{EntityId, OrganismState, WorldConfig};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_CONFIG_PATH: &str = "sim-config/config.toml";

#[derive(Debug, Clone, Parser)]
#[command(name = "sim-validation")]
#[command(about = "Headless validation harness for the deterministic simulation")]
struct Cli {
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    config: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 50_000)]
    ticks: u64,
    #[arg(long, default_value_t = 2_500, value_parser = clap::value_parser!(u64).range(1..))]
    report_every: u64,
    #[arg(long, default_value_t = 30)]
    min_lifetime: u64,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    baseline: bool,
}

#[derive(Debug, Clone)]
struct RunOptions {
    seed: u64,
    ticks: u64,
    report_every: u64,
    min_lifetime: u64,
    out_dir: PathBuf,
    baseline: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationSummary {
    seed: u64,
    ticks: u64,
    baseline: bool,
    total_time_seconds: f64,
    aggregate_score: AggregateScore,
    state_hash: String,
    timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateScore {
    score: f64,
    window_start_tick: u64,
    window_end_tick: u64,
    mean_p_fwd_food: Option<f64>,
    mean_mi_sa: Option<f64>,
    mean_h_action: Option<f64>,
    p_component: f64,
    mi_component: f64,
    entropy_component: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: running sim-validation in debug mode; use `cargo run -p sim-validation --release -- ...` for much faster runs"
        );
    }
    let mut config = load_world_config_from_path(&cli.config)?;
    if cli.baseline {
        config.force_random_actions = true;
    }

    let out_dir = cli
        .out
        .clone()
        .unwrap_or_else(|| default_output_dir(cli.seed));
    let options = RunOptions {
        seed: cli.seed,
        ticks: cli.ticks,
        report_every: cli.report_every,
        min_lifetime: cli.min_lifetime,
        out_dir,
        baseline: cli.baseline,
    };

    let summary = run_with_config(config, &options)?;
    let report_path = options.out_dir.join("report.html");
    println!("wrote artifacts to {}", options.out_dir.display());
    println!("html_report: {}", report_path.display());
    println!("browser_url: {}", browser_file_url(&report_path));
    println!("aggregate_score: {:.2}", summary.aggregate_score.score);
    println!("total_time_seconds: {:.3}", summary.total_time_seconds);
    Ok(())
}

fn browser_file_url(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        }
    });
    #[cfg(windows)]
    {
        return format!("file:///{}", absolute.to_string_lossy().replace('\\', "/"));
    }
    #[cfg(not(windows))]
    {
        format!("file://{}", absolute.to_string_lossy())
    }
}

fn run_with_config(config: WorldConfig, options: &RunOptions) -> Result<ValidationSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let mut reporter = Reporter::new(&options.out_dir)?;
    let mut sim = Simulation::new(config, options.seed)?;
    let mut ledger = Ledger::new(options.min_lifetime);

    for organism in sim.organisms() {
        ledger.birth(organism.id, 0);
    }

    let mut current_food_count = sim.snapshot().foods.len() as u64;
    let mut interval_births = 0_u64;
    let mut interval_deaths = 0_u64;
    let mut timeseries = Vec::new();

    for tick in 1..=options.ticks {
        let delta = sim.tick();
        let records = sim.drain_action_records();

        for record in records {
            ledger.update(record);
        }

        interval_births = interval_births.saturating_add(delta.spawned.len() as u64);
        for spawned in &delta.spawned {
            ledger.birth(spawned.id, tick);
        }

        for removed in &delta.removed_positions {
            match removed.entity_id {
                EntityId::Organism(id) => {
                    interval_deaths = interval_deaths.saturating_add(1);
                    ledger.death(id, tick);
                }
                EntityId::Food(_) => {
                    current_food_count = current_food_count.saturating_sub(1);
                }
            }
        }
        current_food_count = current_food_count.saturating_add(delta.food_spawned.len() as u64);

        if tick % options.report_every == 0 || tick == options.ticks {
            let fraction = tick as f64 / options.ticks as f64;
            println!(
                "progress: {tick}/{total} ({fraction:.3})",
                total = options.ticks
            );
            let interval = compute_interval_metrics(
                tick,
                delta.metrics.organisms,
                interval_births,
                interval_deaths,
                current_food_count,
                ledger.recently_deceased(),
                sim.organisms(),
            );
            reporter.emit(&interval)?;
            timeseries.push(interval);

            interval_births = 0;
            interval_deaths = 0;
            ledger.clear_interval();
        }
    }

    reporter.flush()?;
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let aggregate_score = compute_aggregate_score(&timeseries);

    let summary = ValidationSummary {
        seed: options.seed,
        ticks: options.ticks,
        baseline: options.baseline,
        total_time_seconds,
        aggregate_score,
        state_hash: state_hash(sim.organisms()),
        timeseries,
    };
    write_summary(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &HtmlReportMeta {
            seed: summary.seed,
            ticks: summary.ticks,
            report_every: options.report_every,
            min_lifetime: options.min_lifetime,
            baseline: summary.baseline,
            total_time_seconds: summary.total_time_seconds,
            aggregate_score: summary.aggregate_score.score,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_p_component: summary.aggregate_score.p_component,
            aggregate_mi_component: summary.aggregate_score.mi_component,
            aggregate_entropy_component: summary.aggregate_score.entropy_component,
            aggregate_mean_p_fwd_food: summary.aggregate_score.mean_p_fwd_food,
            aggregate_mean_mi_sa: summary.aggregate_score.mean_mi_sa,
            aggregate_mean_h_action: summary.aggregate_score.mean_h_action,
        },
        &summary.timeseries,
    )?;
    Ok(summary)
}

fn write_summary(out_dir: &Path, summary: &ValidationSummary) -> Result<()> {
    let summary_path = out_dir.join("summary.json");
    let json = serde_json::to_vec_pretty(summary)?;
    fs::write(summary_path, json)?;
    Ok(())
}

fn default_output_dir(seed: u64) -> PathBuf {
    let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    PathBuf::from(format!("artifacts/validation/{}_seed_{seed}", timestamp))
}

fn state_hash(organisms: &[OrganismState]) -> String {
    let population_count = organisms.len() as u64;
    let sum_ids = organisms
        .iter()
        .map(|organism| organism.id.0)
        .fold(0_u64, |acc, value| acc.wrapping_add(value));
    let total_energy = organisms
        .iter()
        .map(|organism| organism.energy as f64)
        .sum::<f64>();

    let mut hasher = DefaultHasher::new();
    population_count.hash(&mut hasher);
    sum_ids.hash(&mut hasher);
    total_energy.to_bits().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn compute_aggregate_score(timeseries: &[IntervalMetrics]) -> AggregateScore {
    let window_len = (timeseries.len() / 5).max(1);
    let start_idx = timeseries.len().saturating_sub(window_len);
    let window = &timeseries[start_idx..];
    let window_start_tick = window.first().map(|row| row.tick).unwrap_or(0);
    let window_end_tick = window.last().map(|row| row.tick).unwrap_or(0);

    let mean_p_fwd_food = mean_option(window.iter().map(|row| row.p_fwd_food));
    let mean_mi_sa = mean_option(window.iter().map(|row| row.mi_sa));
    let mean_h_action = mean_option(window.iter().map(|row| row.h_action));

    let p_baseline = metrics::action_baseline_probability();
    let h_baseline = metrics::action_baseline_entropy();
    let strong_foraging_reference = 0.55;
    let p_component = mean_p_fwd_food
        .map(|value| {
            clamp01(
                (value - p_baseline) / (strong_foraging_reference - p_baseline).max(f64::EPSILON),
            )
        })
        .unwrap_or(0.0);
    let mi_component = mean_mi_sa.map(|value| clamp01(value / 0.10)).unwrap_or(0.0);
    let entropy_target = 0.60 * h_baseline;
    let entropy_width = 0.60 * h_baseline;
    let entropy_component = mean_h_action
        .map(|value| 1.0 - clamp01((value - entropy_target).abs() / entropy_width.max(1e-6)))
        .unwrap_or(0.0);

    let score = 100.0 * (0.50 * p_component + 0.35 * mi_component + 0.15 * entropy_component);

    AggregateScore {
        score,
        window_start_tick,
        window_end_tick,
        mean_p_fwd_food,
        mean_mi_sa,
        mean_h_action,
        p_component,
        mi_component,
        entropy_component,
    }
}

fn mean_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_u64;
    for value in values.flatten() {
        if value.is_finite() {
            sum += value;
            count = count.saturating_add(1);
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_yields_same_summary_hash() {
        let mut cfg = WorldConfig::default();
        cfg.world_width = 40;
        cfg.num_organisms = 300;
        cfg.periodic_injection_interval_turns = 0;
        cfg.periodic_injection_count = 0;
        cfg.force_random_actions = false;

        let out_a = test_output_dir("a");
        let out_b = test_output_dir("b");
        let options_a = RunOptions {
            seed: 2026,
            ticks: 100,
            report_every: 50,
            min_lifetime: 10,
            out_dir: out_a.clone(),
            baseline: false,
        };
        let options_b = RunOptions {
            out_dir: out_b.clone(),
            ..options_a.clone()
        };

        let summary_a = run_with_config(cfg.clone(), &options_a).expect("first run should succeed");
        let summary_b = run_with_config(cfg, &options_b).expect("second run should succeed");

        assert_eq!(summary_a.state_hash, summary_b.state_hash);
        assert_eq!(
            serde_json::to_string(&summary_a.timeseries).expect("serialize first timeseries"),
            serde_json::to_string(&summary_b.timeseries).expect("serialize second timeseries")
        );

        let _ = fs::remove_dir_all(out_a);
        let _ = fs::remove_dir_all(out_b);
    }

    fn test_output_dir(suffix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("sim-validation-test-{suffix}-{nanos}"))
    }
}
