mod ledger;
mod metrics;
mod report;

use anyhow::{anyhow, Result};
use chrono::Utc;
use clap::Parser;
use ledger::Ledger;
use metrics::{compute_interval_metrics, IntervalMetrics};
use report::{write_html_report, HtmlReportMeta, PerSeedReportRow, Reporter};
use serde::Serialize;
use sim_config::load_world_config_from_path;
use sim_core::Simulation;
use sim_types::{EntityId, OrganismState, WorldConfig};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

const DEFAULT_CONFIG_PATH: &str = "sim-validation/config.toml";
const DEFAULT_SEEDS: &str = "42,123,7,2026";

#[derive(Debug, Clone, Parser)]
#[command(name = "sim-validation")]
#[command(about = "Headless validation harness for the deterministic simulation")]
struct Cli {
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    config: PathBuf,
    #[arg(
        long = "seed",
        value_delimiter = ',',
        num_args = 1..,
        default_value = DEFAULT_SEEDS
    )]
    seeds: Vec<u64>,
    #[arg(long, default_value_t = 50_000)]
    ticks: u64,
    #[arg(long, default_value_t = 2_500, value_parser = clap::value_parser!(u64).range(1..))]
    report_every: u64,
    #[arg(long, default_value_t = 30)]
    min_lifetime: u64,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    title: Option<String>,
    #[arg(long, default_value_t = false)]
    baseline: bool,
}

#[derive(Debug, Clone)]
struct HarnessRunOptions {
    seeds: Vec<u64>,
    ticks: u64,
    report_every: u64,
    min_lifetime: u64,
    out_dir: PathBuf,
    title: Option<String>,
    baseline: bool,
}

#[derive(Debug, Clone)]
struct SeedRunOptions {
    seed: u64,
    ticks: u64,
    report_every: u64,
    min_lifetime: u64,
    out_dir: PathBuf,
    title: Option<String>,
    baseline: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SeedValidationSummary {
    title: Option<String>,
    seed: u64,
    ticks: u64,
    baseline: bool,
    total_time_seconds: f64,
    aggregate_score: AggregateScore,
    state_hash: String,
    timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationSummary {
    title: Option<String>,
    seeds: Vec<u64>,
    ticks: u64,
    baseline: bool,
    total_time_seconds: f64,
    aggregate_score: AggregateScore,
    seed_summaries: Vec<SeedRunSummary>,
    timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct SeedRunSummary {
    seed: u64,
    out_dir: PathBuf,
    total_time_seconds: f64,
    aggregate_score: AggregateScore,
    state_hash: String,
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

    let seeds = normalize_seeds(cli.seeds);
    if seeds.is_empty() {
        return Err(anyhow!("sim-validation requires at least one seed"));
    }

    let out_dir = cli
        .out
        .clone()
        .unwrap_or_else(|| default_output_dir(&seeds));
    let options = HarnessRunOptions {
        seeds,
        ticks: cli.ticks,
        report_every: cli.report_every,
        min_lifetime: cli.min_lifetime,
        out_dir,
        title: cli.title,
        baseline: cli.baseline,
    };

    let summary = run_validation_across_seeds(config, &options)?;
    let report_path = options.out_dir.join("report.html");
    println!("wrote artifacts to {}", options.out_dir.display());
    println!("html_report: {}", report_path.display());
    println!("browser_url: {}", browser_file_url(&report_path));
    println!("seeds: {}", format_seed_list(&summary.seeds));
    for seed_summary in &summary.seed_summaries {
        println!(
            "seed_score[{}]: {:.2}",
            seed_summary.seed, seed_summary.aggregate_score.score
        );
    }
    println!("aggregate_score: {:.2}", summary.aggregate_score.score);
    println!("total_time_seconds: {:.3}", summary.total_time_seconds);
    Ok(())
}

fn run_validation_across_seeds(
    config: WorldConfig,
    options: &HarnessRunOptions,
) -> Result<ValidationSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let mut handles = Vec::with_capacity(options.seeds.len());
    for &seed in &options.seeds {
        let config = config.clone();
        let seed_options = SeedRunOptions {
            seed,
            ticks: options.ticks,
            report_every: options.report_every,
            min_lifetime: options.min_lifetime,
            out_dir: options.out_dir.join(format!("seed_{seed}")),
            title: options
                .title
                .as_ref()
                .map(|title| format!("{title} (seed {seed})")),
            baseline: options.baseline,
        };
        handles.push((
            seed,
            thread::spawn(move || run_single_seed_validation(config, seed_options)),
        ));
    }

    let mut seed_summaries = Vec::with_capacity(handles.len());
    for (seed, handle) in handles {
        let summary = handle
            .join()
            .map_err(|_| anyhow!("seed run {seed} panicked"))??;
        seed_summaries.push(summary);
    }
    seed_summaries.sort_by_key(|summary| summary.seed);

    let averaged_timeseries = average_timeseries(&seed_summaries);
    write_timeseries_csv(&options.out_dir, &averaged_timeseries)?;

    let aggregate_score = average_aggregate_scores(&seed_summaries);
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    let seed_run_summaries = seed_summaries
        .iter()
        .map(|summary| SeedRunSummary {
            seed: summary.seed,
            out_dir: PathBuf::from(format!("seed_{}", summary.seed)),
            total_time_seconds: summary.total_time_seconds,
            aggregate_score: summary.aggregate_score.clone(),
            state_hash: summary.state_hash.clone(),
        })
        .collect::<Vec<_>>();

    let summary = ValidationSummary {
        title: options.title.clone(),
        seeds: options.seeds.clone(),
        ticks: options.ticks,
        baseline: options.baseline,
        total_time_seconds,
        aggregate_score: aggregate_score.clone(),
        seed_summaries: seed_run_summaries.clone(),
        timeseries: averaged_timeseries.clone(),
    };

    write_summary_json(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &HtmlReportMeta {
            title: summary.title.clone(),
            seed_label: format_seed_list(&summary.seeds),
            seed_count: summary.seeds.len(),
            ticks: summary.ticks,
            report_every: options.report_every,
            min_lifetime: options.min_lifetime,
            baseline: summary.baseline,
            total_time_seconds: summary.total_time_seconds,
            generated_at_utc,
            aggregate_score: summary.aggregate_score.score,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_p_component: summary.aggregate_score.p_component,
            aggregate_mi_component: summary.aggregate_score.mi_component,
            aggregate_entropy_component: summary.aggregate_score.entropy_component,
            aggregate_mean_p_fwd_food: summary.aggregate_score.mean_p_fwd_food,
            aggregate_mean_mi_sa: summary.aggregate_score.mean_mi_sa,
            aggregate_mean_h_action: summary.aggregate_score.mean_h_action,
            timeseries_label: "mean across seeds".to_owned(),
            per_seed_rows: seed_run_summaries
                .iter()
                .map(|seed_summary| PerSeedReportRow {
                    seed: seed_summary.seed,
                    score: seed_summary.aggregate_score.score,
                    total_time_seconds: seed_summary.total_time_seconds,
                    state_hash: seed_summary.state_hash.clone(),
                    report_href: format!("seed_{}/report.html", seed_summary.seed),
                })
                .collect(),
        },
        &summary.timeseries,
    )?;

    Ok(summary)
}

fn run_single_seed_validation(
    config: WorldConfig,
    options: SeedRunOptions,
) -> Result<SeedValidationSummary> {
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
                "progress[seed={}]: {tick}/{total} ({fraction:.3})",
                options.seed,
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
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let aggregate_score = compute_aggregate_score(&timeseries);

    let summary = SeedValidationSummary {
        title: options.title.clone(),
        seed: options.seed,
        ticks: options.ticks,
        baseline: options.baseline,
        total_time_seconds,
        aggregate_score: aggregate_score.clone(),
        state_hash: state_hash(sim.organisms()),
        timeseries,
    };
    write_summary_json(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &HtmlReportMeta {
            title: summary.title.clone(),
            seed_label: summary.seed.to_string(),
            seed_count: 1,
            ticks: summary.ticks,
            report_every: options.report_every,
            min_lifetime: options.min_lifetime,
            baseline: summary.baseline,
            total_time_seconds: summary.total_time_seconds,
            generated_at_utc,
            aggregate_score: summary.aggregate_score.score,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_p_component: summary.aggregate_score.p_component,
            aggregate_mi_component: summary.aggregate_score.mi_component,
            aggregate_entropy_component: summary.aggregate_score.entropy_component,
            aggregate_mean_p_fwd_food: summary.aggregate_score.mean_p_fwd_food,
            aggregate_mean_mi_sa: summary.aggregate_score.mean_mi_sa,
            aggregate_mean_h_action: summary.aggregate_score.mean_h_action,
            timeseries_label: "per-seed timeseries".to_owned(),
            per_seed_rows: Vec::new(),
        },
        &summary.timeseries,
    )?;

    Ok(summary)
}

fn write_summary_json<T: Serialize>(out_dir: &Path, summary: &T) -> Result<()> {
    let summary_path = out_dir.join("summary.json");
    let json = serde_json::to_vec_pretty(summary)?;
    fs::write(summary_path, json)?;
    Ok(())
}

fn write_timeseries_csv(out_dir: &Path, rows: &[IntervalMetrics]) -> Result<()> {
    let mut reporter = Reporter::new(out_dir)?;
    for row in rows {
        reporter.emit(row)?;
    }
    reporter.flush()?;
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

fn default_output_dir(seeds: &[u64]) -> PathBuf {
    let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    if seeds.len() == 1 {
        return PathBuf::from(format!("artifacts/validation/{}_seed_{}", timestamp, seeds[0]));
    }
    PathBuf::from(format!(
        "artifacts/validation/{}_seeds_{}",
        timestamp,
        seed_slug(seeds)
    ))
}

fn seed_slug(seeds: &[u64]) -> String {
    const MAX_LISTED_SEEDS: usize = 4;
    let listed = seeds
        .iter()
        .take(MAX_LISTED_SEEDS)
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("_");
    if seeds.len() <= MAX_LISTED_SEEDS {
        listed
    } else {
        format!("{listed}_plus{}", seeds.len() - MAX_LISTED_SEEDS)
    }
}

fn format_seed_list(seeds: &[u64]) -> String {
    seeds.iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn normalize_seeds(seeds: Vec<u64>) -> Vec<u64> {
    let mut unique = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if !unique.contains(&seed) {
            unique.push(seed);
        }
    }
    unique
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

fn average_aggregate_scores(seed_summaries: &[SeedValidationSummary]) -> AggregateScore {
    let first = seed_summaries
        .first()
        .expect("multi-seed validation requires at least one seed");
    AggregateScore {
        score: mean_f64(seed_summaries.iter().map(|summary| summary.aggregate_score.score)),
        window_start_tick: first.aggregate_score.window_start_tick,
        window_end_tick: first.aggregate_score.window_end_tick,
        mean_p_fwd_food: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_p_fwd_food),
        ),
        mean_mi_sa: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_mi_sa),
        ),
        mean_h_action: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_h_action),
        ),
        p_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.p_component),
        ),
        mi_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mi_component),
        ),
        entropy_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.entropy_component),
        ),
    }
}

fn average_timeseries(seed_summaries: &[SeedValidationSummary]) -> Vec<IntervalMetrics> {
    let Some(first_summary) = seed_summaries.first() else {
        return Vec::new();
    };
    let row_count = first_summary.timeseries.len();
    let mut averaged = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        let tick = first_summary.timeseries[row_idx].tick;
        debug_assert!(seed_summaries.iter().all(|summary| {
            summary.timeseries.len() == row_count && summary.timeseries[row_idx].tick == tick
        }));

        averaged.push(IntervalMetrics {
            tick,
            pop: mean_round_u32(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].pop),
            ),
            births: mean_round_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].births),
            ),
            deaths: mean_round_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].deaths),
            ),
            food: mean_round_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].food),
            ),
            max_generation: mean_option_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].max_generation),
            ),
            life_mean: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].life_mean),
            ),
            life_max: mean_option_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].life_max),
            ),
            ate_pct: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].ate_pct),
            ),
            cons_mean: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].cons_mean),
            ),
            brain_size: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size),
            ),
            p_fwd_food: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].p_fwd_food),
            ),
            mi_sa: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa),
            ),
            h_action: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].h_action),
            ),
            util: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].util),
            ),
        });
    }

    averaged
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

fn mean_option_u64(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    mean_option(values.map(|value| value.map(|inner| inner as f64)))
        .map(|value| value.round() as u64)
}

fn mean_round_u64(values: impl Iterator<Item = u64>) -> u64 {
    mean_f64(values.map(|value| value as f64)).round() as u64
}

fn mean_round_u32(values: impl Iterator<Item = u32>) -> u32 {
    mean_f64(values.map(|value| value as f64)).round() as u32
}

fn mean_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_u64;
    for value in values {
        if value.is_finite() {
            sum += value;
            count = count.saturating_add(1);
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
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
        let options_a = SeedRunOptions {
            seed: 2026,
            ticks: 100,
            report_every: 50,
            min_lifetime: 10,
            out_dir: out_a.clone(),
            title: None,
            baseline: false,
        };
        let options_b = SeedRunOptions {
            out_dir: out_b.clone(),
            ..options_a.clone()
        };

        let summary_a =
            run_single_seed_validation(cfg.clone(), options_a).expect("first run should succeed");
        let summary_b =
            run_single_seed_validation(cfg, options_b).expect("second run should succeed");

        assert_eq!(summary_a.state_hash, summary_b.state_hash);
        assert_eq!(
            serde_json::to_string(&summary_a.timeseries).expect("serialize first timeseries"),
            serde_json::to_string(&summary_b.timeseries).expect("serialize second timeseries")
        );

        let _ = fs::remove_dir_all(out_a);
        let _ = fs::remove_dir_all(out_b);
    }

    #[test]
    fn multi_seed_summary_uses_mean_seed_score() {
        let mut cfg = WorldConfig::default();
        cfg.world_width = 40;
        cfg.num_organisms = 300;
        cfg.periodic_injection_interval_turns = 0;
        cfg.periodic_injection_count = 0;
        cfg.force_random_actions = false;

        let out_dir = test_output_dir("multi");
        let options = HarnessRunOptions {
            seeds: vec![2026, 7, 123, 42],
            ticks: 100,
            report_every: 50,
            min_lifetime: 10,
            out_dir: out_dir.clone(),
            title: None,
            baseline: false,
        };

        let summary =
            run_validation_across_seeds(cfg, &options).expect("multi-seed run should succeed");
        let expected = summary
            .seed_summaries
            .iter()
            .map(|seed_summary| seed_summary.aggregate_score.score)
            .sum::<f64>()
            / summary.seed_summaries.len() as f64;
        assert!((summary.aggregate_score.score - expected).abs() < 1.0e-9);
        assert_eq!(summary.seed_summaries.len(), 4);
        assert_eq!(summary.seeds, vec![2026, 7, 123, 42]);

        let _ = fs::remove_dir_all(out_dir);
    }

    fn test_output_dir(suffix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("sim-validation-test-{suffix}-{nanos}"))
    }
}
