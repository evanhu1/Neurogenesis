mod ledger;
mod metrics;
mod report;

use anyhow::{anyhow, Result};
use chrono::Utc;
use clap::Parser;
use ledger::{Ledger, N_ACTIONS};
use metrics::{compute_interval_metrics, jensen_shannon_divergence, IntervalMetrics};
use report::{
    write_comparison_html_report, write_html_report, ComparisonHtmlReportMeta,
    ComparisonMetricRow, HtmlReportMeta, PerSeedComparisonRow, PerSeedReportRow, Reporter,
};
use serde::Serialize;
use sim_config::load_world_config_from_path;
use sim_core::Simulation;
use sim_types::{EntityId, OrganismState, WorldConfig};
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

const DEFAULT_CONFIG_PATH: &str = "sim-validation/config.toml";
const DEFAULT_SEEDS: &str = "42,123,7,2026,99,314,2718,4242,9001,65537";

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
    #[arg(long, default_value_t = 100_000)]
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
    #[arg(long, default_value_t = false)]
    compare: bool,
    #[arg(long, default_value_t = false)]
    disable_plasticity: bool,
    #[arg(long)]
    executed_action_credit: Option<bool>,
    #[arg(long)]
    explicit_idle_softmax: Option<bool>,
    #[arg(long)]
    juvenile_plasticity: Option<bool>,
    #[arg(long)]
    split_attack: Option<bool>,
}

#[derive(Debug, Clone, Default)]
struct FeatureOverrides {
    disable_plasticity: bool,
    executed_action_credit: Option<bool>,
    explicit_idle_softmax: Option<bool>,
    juvenile_plasticity: Option<bool>,
    split_attack: Option<bool>,
}

impl FeatureOverrides {
    fn has_overrides(&self) -> bool {
        self.disable_plasticity
            || self.executed_action_credit.is_some()
            || self.explicit_idle_softmax.is_some()
            || self.juvenile_plasticity.is_some()
            || self.split_attack.is_some()
    }

    fn label(&self) -> String {
        let mut parts = Vec::new();
        if self.disable_plasticity {
            parts.push("disable-plasticity".to_owned());
        }
        if let Some(value) = self.executed_action_credit {
            parts.push(format!("executed-action-credit={value}"));
        }
        if let Some(value) = self.explicit_idle_softmax {
            parts.push(format!("explicit-idle-softmax={value}"));
        }
        if let Some(value) = self.juvenile_plasticity {
            parts.push(format!("juvenile-plasticity={value}"));
        }
        if let Some(value) = self.split_attack {
            parts.push(format!("split-attack={value}"));
        }
        if parts.is_empty() {
            "treatment".to_owned()
        } else {
            parts.join(", ")
        }
    }
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
    reward_reversal_tick: Option<u64>,
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
    worker_threads: usize,
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
    score_median: f64,
    score_stddev: f64,
    score_min: f64,
    score_max: f64,
    window_start_tick: u64,
    window_end_tick: u64,
    mean_p_fwd_food: Option<f64>,
    mean_mi_sa: Option<f64>,
    mean_mi_sa_juvenile: Option<f64>,
    mean_mi_sa_adult: Option<f64>,
    mean_h_action: Option<f64>,
    mean_predation_rate: Option<f64>,
    mean_foraging_rate: Option<f64>,
    mean_attack_attempt_rate: Option<f64>,
    mean_attack_success_rate: Option<f64>,
    mean_idle_fraction: Option<f64>,
    mean_reproduction_efficiency: Option<f64>,
    mean_lineage_diversity: Option<f64>,
    mean_damage_avoidance: Option<f64>,
    mean_reward_reversal_shift: Option<f64>,
    mean_action_histogram: [f64; N_ACTIONS],
    reward_reversal_adaptation_ticks: Option<u64>,
    p_component: f64,
    mi_component: f64,
    entropy_component: f64,
    predation_component: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonSummary {
    title: Option<String>,
    seeds: Vec<u64>,
    ticks: u64,
    control_label: String,
    treatment_label: String,
    total_time_seconds: f64,
    control: ValidationSummary,
    treatment: ValidationSummary,
    metric_rows: Vec<ComparisonMetricRow>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: running sim-validation in debug mode; use `cargo run -p sim-validation --release -- ...` for much faster runs"
        );
    }

    let mut control_config = load_world_config_from_path(&cli.config)?;
    if cli.baseline {
        control_config.force_random_actions = true;
    }
    let overrides = FeatureOverrides {
        disable_plasticity: cli.disable_plasticity,
        executed_action_credit: cli.executed_action_credit,
        explicit_idle_softmax: cli.explicit_idle_softmax,
        juvenile_plasticity: cli.juvenile_plasticity,
        split_attack: cli.split_attack,
    };

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

    let run_comparison = cli.compare || overrides.has_overrides();
    if run_comparison {
        let treatment_config = apply_feature_overrides(control_config.clone(), &overrides);
        let comparison = run_comparison_validation(control_config, treatment_config, &options, &overrides)?;
        let report_path = options.out_dir.join("comparison_report.html");
        println!("wrote artifacts to {}", options.out_dir.display());
        println!("comparison_html_report: {}", report_path.display());
        println!("browser_url: {}", browser_file_url(&report_path));
        println!("seeds: {}", format_seed_list(&comparison.seeds));
        println!("control_label: {}", comparison.control_label);
        println!("treatment_label: {}", comparison.treatment_label);
        for row in &comparison.metric_rows {
            println!(
                "compare[{label}]: control={control} treatment={treatment} diff={diff} ci95=[{low},{high}]",
                label = row.label,
                control = fmt_option(row.control_mean, 4),
                treatment = fmt_option(row.treatment_mean, 4),
                diff = fmt_option(row.mean_diff, 4),
                low = fmt_option(row.ci_low, 4),
                high = fmt_option(row.ci_high, 4),
            );
        }
        println!("total_time_seconds: {:.3}", comparison.total_time_seconds);
    } else {
        let summary = run_validation_across_seeds(control_config, &options)?;
        let report_path = options.out_dir.join("report.html");
        println!("wrote artifacts to {}", options.out_dir.display());
        println!("html_report: {}", report_path.display());
        println!("browser_url: {}", browser_file_url(&report_path));
        println!("seeds: {}", format_seed_list(&summary.seeds));
        println!("worker_threads: {}", summary.worker_threads);
        for seed_summary in &summary.seed_summaries {
            println!(
                "seed_score[{}]: {:.2}",
                seed_summary.seed, seed_summary.aggregate_score.score
            );
        }
        println!("aggregate_score: {:.2}", summary.aggregate_score.score);
        println!(
            "aggregate_score_median: {:.2}",
            summary.aggregate_score.score_median
        );
        println!(
            "aggregate_score_stddev: {:.2}",
            summary.aggregate_score.score_stddev
        );
        println!(
            "aggregate_score_min: {:.2}",
            summary.aggregate_score.score_min
        );
        println!(
            "aggregate_score_max: {:.2}",
            summary.aggregate_score.score_max
        );
        println!(
            "aggregate_predation_component: {:.3}",
            summary.aggregate_score.predation_component
        );
        println!("total_time_seconds: {:.3}", summary.total_time_seconds);
    }
    Ok(())
}

fn apply_feature_overrides(mut config: WorldConfig, overrides: &FeatureOverrides) -> WorldConfig {
    if overrides.disable_plasticity {
        config.runtime_plasticity_enabled = false;
    }
    if let Some(value) = overrides.executed_action_credit {
        config.executed_action_credit = value;
    }
    if let Some(value) = overrides.explicit_idle_softmax {
        config.explicit_idle_softmax = value;
    }
    if let Some(value) = overrides.juvenile_plasticity {
        config.juvenile_plasticity_enabled = value;
    }
    if let Some(value) = overrides.split_attack {
        config.split_attack_actions = value;
    }
    config
}

fn reward_reversal_tick_for_run(ticks: u64) -> Option<u64> {
    if ticks < 2 {
        None
    } else {
        Some((ticks / 2).max(1))
    }
}

fn fmt_option(value: Option<f64>, decimals: usize) -> String {
    value
        .map(|value| format!("{value:.decimals$}"))
        .unwrap_or_else(|| "NA".to_owned())
}

fn run_comparison_validation(
    control_config: WorldConfig,
    treatment_config: WorldConfig,
    options: &HarnessRunOptions,
    overrides: &FeatureOverrides,
) -> Result<ComparisonSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let control = run_validation_across_seeds(
        control_config,
        &HarnessRunOptions {
            out_dir: options.out_dir.join("control"),
            title: options
                .title
                .as_ref()
                .map(|title| format!("{title} [control]")),
            ..options.clone()
        },
    )?;
    let treatment = run_validation_across_seeds(
        treatment_config,
        &HarnessRunOptions {
            out_dir: options.out_dir.join("treatment"),
            title: options
                .title
                .as_ref()
                .map(|title| format!("{title} [treatment]")),
            ..options.clone()
        },
    )?;

    let control_label = if options.baseline {
        "control (baseline)".to_owned()
    } else {
        "control".to_owned()
    };
    let treatment_label = overrides.label();
    let metric_rows = comparison_metric_rows(&control, &treatment);
    let per_seed_rows = control
        .seed_summaries
        .iter()
        .zip(&treatment.seed_summaries)
        .map(|(control_seed, treatment_seed)| PerSeedComparisonRow {
            seed: control_seed.seed,
            control_score: control_seed.aggregate_score.score,
            treatment_score: treatment_seed.aggregate_score.score,
            diff_score: treatment_seed.aggregate_score.score - control_seed.aggregate_score.score,
            control_report_href: format!("control/seed_{}/report.html", control_seed.seed),
            treatment_report_href: format!("treatment/seed_{}/report.html", treatment_seed.seed),
        })
        .collect::<Vec<_>>();
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let comparison = ComparisonSummary {
        title: options.title.clone(),
        seeds: options.seeds.clone(),
        ticks: options.ticks,
        control_label: control_label.clone(),
        treatment_label: treatment_label.clone(),
        total_time_seconds,
        control: control.clone(),
        treatment: treatment.clone(),
        metric_rows: metric_rows.clone(),
    };
    write_summary_json(&options.out_dir, &comparison)?;
    write_comparison_html_report(
        &options.out_dir,
        &ComparisonHtmlReportMeta {
            title: options.title.clone(),
            seed_label: format_seed_list(&options.seeds),
            ticks: options.ticks,
            control_label,
            treatment_label,
            total_time_seconds,
            metric_rows,
            per_seed_rows,
            control_report_href: "control/report.html".to_owned(),
            treatment_report_href: "treatment/report.html".to_owned(),
        },
    )?;
    Ok(comparison)
}

fn comparison_metric_rows(
    control: &ValidationSummary,
    treatment: &ValidationSummary,
) -> Vec<ComparisonMetricRow> {
    vec![
        paired_metric_row(
            "aggregate_score",
            control
                .seed_summaries
                .iter()
                .map(|seed| Some(seed.aggregate_score.score))
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| Some(seed.aggregate_score.score))
                .collect(),
        ),
        paired_metric_row(
            "idle_fraction",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_idle_fraction)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_idle_fraction)
                .collect(),
        ),
        paired_metric_row(
            "action_entropy",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_h_action)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_h_action)
                .collect(),
        ),
        paired_metric_row(
            "p_fwd_food",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_p_fwd_food)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_p_fwd_food)
                .collect(),
        ),
        paired_metric_row(
            "mi_sa",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa)
                .collect(),
        ),
        paired_metric_row(
            "mi_sa_juvenile",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_juvenile)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_juvenile)
                .collect(),
        ),
        paired_metric_row(
            "mi_sa_adult",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_adult)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_adult)
                .collect(),
        ),
        paired_metric_row(
            "reproduction_efficiency",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reproduction_efficiency)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reproduction_efficiency)
                .collect(),
        ),
        paired_metric_row(
            "foraging_rate",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_foraging_rate)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_foraging_rate)
                .collect(),
        ),
        paired_metric_row(
            "attack_attempt_rate",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_attempt_rate)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_attempt_rate)
                .collect(),
        ),
        paired_metric_row(
            "attack_success_rate",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_success_rate)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_success_rate)
                .collect(),
        ),
        paired_metric_row(
            "damage_avoidance",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_damage_avoidance)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_damage_avoidance)
                .collect(),
        ),
        paired_metric_row(
            "reward_reversal_shift",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reward_reversal_shift)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reward_reversal_shift)
                .collect(),
        ),
        paired_metric_row(
            "reward_reversal_adaptation_ticks",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.reward_reversal_adaptation_ticks.map(|v| v as f64))
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.reward_reversal_adaptation_ticks.map(|v| v as f64))
                .collect(),
        ),
    ]
}

fn paired_metric_row(
    label: &str,
    control_values: Vec<Option<f64>>,
    treatment_values: Vec<Option<f64>>,
) -> ComparisonMetricRow {
    let control_mean = mean_option(control_values.iter().copied());
    let treatment_mean = mean_option(treatment_values.iter().copied());
    let diffs = control_values
        .iter()
        .zip(&treatment_values)
        .filter_map(|(control, treatment)| match (*control, *treatment) {
            (Some(control), Some(treatment)) => Some(treatment - control),
            _ => None,
        })
        .collect::<Vec<_>>();
    let (mean_diff, ci_low, ci_high) = diff_confidence_interval(&diffs);
    ComparisonMetricRow {
        label: label.to_owned(),
        control_mean,
        treatment_mean,
        mean_diff,
        ci_low,
        ci_high,
    }
}

fn diff_confidence_interval(diffs: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if diffs.is_empty() {
        return (None, None, None);
    }
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    if diffs.len() == 1 {
        return (Some(mean), Some(mean), Some(mean));
    }
    let variance = diffs
        .iter()
        .map(|diff| {
            let delta = *diff - mean;
            delta * delta
        })
        .sum::<f64>()
        / (diffs.len() as f64 - 1.0);
    let se = variance.sqrt() / (diffs.len() as f64).sqrt();
    let margin = 1.96 * se;
    (Some(mean), Some(mean - margin), Some(mean + margin))
}

fn run_validation_across_seeds(
    config: WorldConfig,
    options: &HarnessRunOptions,
) -> Result<ValidationSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;
    let worker_threads = default_worker_threads(options.seeds.len());
    let seed_queue = Arc::new(Mutex::new(VecDeque::from(options.seeds.clone())));
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::with_capacity(worker_threads);

    for _ in 0..worker_threads {
        let config = config.clone();
        let seed_queue = Arc::clone(&seed_queue);
        let tx = tx.clone();
        let out_dir = options.out_dir.clone();
        let title = options.title.clone();
        let ticks = options.ticks;
        let report_every = options.report_every;
        let min_lifetime = options.min_lifetime;
        let baseline = options.baseline;

        handles.push(thread::spawn(move || loop {
            let seed = match seed_queue.lock() {
                Ok(mut queue) => queue.pop_front(),
                Err(_) => None,
            };
            let Some(seed) = seed else {
                break;
            };

            let seed_options = SeedRunOptions {
                seed,
                ticks,
                report_every,
                min_lifetime,
                out_dir: out_dir.join(format!("seed_{seed}")),
                title: title
                    .as_ref()
                    .map(|run_title| format!("{run_title} (seed {seed})")),
                baseline,
                reward_reversal_tick: reward_reversal_tick_for_run(ticks),
            };
            let result = run_single_seed_validation(config.clone(), seed_options);
            if tx.send((seed, result)).is_err() {
                break;
            }
        }));
    }
    drop(tx);

    let mut seed_summaries = Vec::with_capacity(options.seeds.len());
    for _ in 0..options.seeds.len() {
        let (_seed, result) = rx
            .recv()
            .map_err(|_| anyhow!("validation worker exited before reporting all seeds"))?;
        let summary = result?;
        seed_summaries.push(summary);
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("validation worker panicked"))?;
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
        worker_threads,
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
            aggregate_score_median: summary.aggregate_score.score_median,
            aggregate_score_stddev: summary.aggregate_score.score_stddev,
            aggregate_score_min: summary.aggregate_score.score_min,
            aggregate_score_max: summary.aggregate_score.score_max,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_p_component: summary.aggregate_score.p_component,
            aggregate_mi_component: summary.aggregate_score.mi_component,
            aggregate_entropy_component: summary.aggregate_score.entropy_component,
            aggregate_predation_component: summary.aggregate_score.predation_component,
            aggregate_mean_p_fwd_food: summary.aggregate_score.mean_p_fwd_food,
            aggregate_mean_mi_sa: summary.aggregate_score.mean_mi_sa,
            aggregate_mean_mi_sa_juvenile: summary.aggregate_score.mean_mi_sa_juvenile,
            aggregate_mean_mi_sa_adult: summary.aggregate_score.mean_mi_sa_adult,
            aggregate_mean_h_action: summary.aggregate_score.mean_h_action,
            aggregate_mean_predation_rate: summary.aggregate_score.mean_predation_rate,
            aggregate_mean_foraging_rate: summary.aggregate_score.mean_foraging_rate,
            aggregate_mean_attack_attempt_rate: summary.aggregate_score.mean_attack_attempt_rate,
            aggregate_mean_attack_success_rate: summary.aggregate_score.mean_attack_success_rate,
            aggregate_mean_idle_fraction: summary.aggregate_score.mean_idle_fraction,
            aggregate_mean_reproduction_efficiency: summary
                .aggregate_score
                .mean_reproduction_efficiency,
            aggregate_mean_lineage_diversity: summary.aggregate_score.mean_lineage_diversity,
            aggregate_mean_damage_avoidance: summary.aggregate_score.mean_damage_avoidance,
            aggregate_mean_reward_reversal_shift: summary
                .aggregate_score
                .mean_reward_reversal_shift,
            aggregate_reward_reversal_adaptation_ticks: summary
                .aggregate_score
                .reward_reversal_adaptation_ticks,
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
    let mut interval_consumptions = 0_u64;
    let mut interval_predations = 0_u64;
    let mut interval_population_exposure = 0_u64;
    let mut pre_reversal_histogram: Option<[f64; N_ACTIONS]> = None;
    let mut timeseries = Vec::new();

    for tick in 1..=options.ticks {
        if options
            .reward_reversal_tick
            .is_some_and(|reversal_tick| tick > reversal_tick)
        {
            sim.set_reward_signal_multiplier(-1.0);
        }
        interval_population_exposure =
            interval_population_exposure.saturating_add(sim.organisms().len() as u64);
        let delta = sim.tick();
        let records = sim.drain_action_records();
        interval_consumptions =
            interval_consumptions.saturating_add(delta.metrics.consumptions_last_turn);
        interval_predations =
            interval_predations.saturating_add(delta.metrics.predations_last_turn);

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
            let mut interval = compute_interval_metrics(
                tick,
                delta.metrics.organisms,
                interval_births,
                interval_deaths,
                current_food_count,
                interval_consumptions,
                interval_predations,
                interval_population_exposure,
                ledger.recently_deceased(),
                sim.organisms(),
                ledger.interval_action_stats(),
                sim.config().food_energy,
            );
            if options
                .reward_reversal_tick
                .is_some_and(|reversal_tick| tick <= reversal_tick)
            {
                pre_reversal_histogram = Some(interval.action_histogram);
            } else if let Some(reference) = pre_reversal_histogram.as_ref() {
                interval.reward_reversal_shift =
                    jensen_shannon_divergence(&interval.action_histogram, reference);
            }
            reporter.emit(&interval)?;
            timeseries.push(interval);

            interval_births = 0;
            interval_deaths = 0;
            interval_consumptions = 0;
            interval_predations = 0;
            interval_population_exposure = 0;
            ledger.clear_interval();
        }
    }

    reporter.flush()?;
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let aggregate_score = compute_aggregate_score(&timeseries, options.reward_reversal_tick);

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
            aggregate_score_median: summary.aggregate_score.score_median,
            aggregate_score_stddev: summary.aggregate_score.score_stddev,
            aggregate_score_min: summary.aggregate_score.score_min,
            aggregate_score_max: summary.aggregate_score.score_max,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_p_component: summary.aggregate_score.p_component,
            aggregate_mi_component: summary.aggregate_score.mi_component,
            aggregate_entropy_component: summary.aggregate_score.entropy_component,
            aggregate_predation_component: summary.aggregate_score.predation_component,
            aggregate_mean_p_fwd_food: summary.aggregate_score.mean_p_fwd_food,
            aggregate_mean_mi_sa: summary.aggregate_score.mean_mi_sa,
            aggregate_mean_mi_sa_juvenile: summary.aggregate_score.mean_mi_sa_juvenile,
            aggregate_mean_mi_sa_adult: summary.aggregate_score.mean_mi_sa_adult,
            aggregate_mean_h_action: summary.aggregate_score.mean_h_action,
            aggregate_mean_predation_rate: summary.aggregate_score.mean_predation_rate,
            aggregate_mean_foraging_rate: summary.aggregate_score.mean_foraging_rate,
            aggregate_mean_attack_attempt_rate: summary.aggregate_score.mean_attack_attempt_rate,
            aggregate_mean_attack_success_rate: summary.aggregate_score.mean_attack_success_rate,
            aggregate_mean_idle_fraction: summary.aggregate_score.mean_idle_fraction,
            aggregate_mean_reproduction_efficiency: summary
                .aggregate_score
                .mean_reproduction_efficiency,
            aggregate_mean_lineage_diversity: summary.aggregate_score.mean_lineage_diversity,
            aggregate_mean_damage_avoidance: summary.aggregate_score.mean_damage_avoidance,
            aggregate_mean_reward_reversal_shift: summary
                .aggregate_score
                .mean_reward_reversal_shift,
            aggregate_reward_reversal_adaptation_ticks: summary
                .aggregate_score
                .reward_reversal_adaptation_ticks,
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
        return PathBuf::from(format!(
            "artifacts/validation/{}_seed_{}",
            timestamp, seeds[0]
        ));
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
    seeds
        .iter()
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

fn compute_aggregate_score(
    timeseries: &[IntervalMetrics],
    reward_reversal_tick: Option<u64>,
) -> AggregateScore {
    let window_len = (timeseries.len() / 5).max(1);
    let start_idx = timeseries.len().saturating_sub(window_len);
    let window = &timeseries[start_idx..];
    let window_start_tick = window.first().map(|row| row.tick).unwrap_or(0);
    let window_end_tick = window.last().map(|row| row.tick).unwrap_or(0);

    let mean_p_fwd_food = mean_option(window.iter().map(|row| row.p_fwd_food));
    let mean_mi_sa = mean_option(window.iter().map(|row| row.mi_sa));
    let mean_mi_sa_juvenile = mean_option(window.iter().map(|row| row.mi_sa_juvenile));
    let mean_mi_sa_adult = mean_option(window.iter().map(|row| row.mi_sa_adult));
    let mean_h_action = mean_option(window.iter().map(|row| row.h_action));
    let mean_predation_rate = mean_option(window.iter().map(|row| row.predation_rate));
    let mean_foraging_rate = mean_option(window.iter().map(|row| row.foraging_rate));
    let mean_attack_attempt_rate = mean_option(window.iter().map(|row| row.attack_attempt_rate));
    let mean_attack_success_rate = mean_option(window.iter().map(|row| row.attack_success_rate));
    let mean_idle_fraction = mean_option(window.iter().map(|row| row.idle_fraction));
    let mean_reproduction_efficiency =
        mean_option(window.iter().map(|row| row.reproduction_efficiency));
    let mean_lineage_diversity = mean_option(window.iter().map(|row| row.lineage_diversity));
    let mean_damage_avoidance = mean_option(window.iter().map(|row| row.damage_avoidance));
    let mean_reward_reversal_shift =
        mean_option(window.iter().map(|row| row.reward_reversal_shift));
    let mean_action_histogram = mean_action_histogram(window);
    let reward_reversal_adaptation_ticks =
        reward_reversal_adaptation_ticks(timeseries, reward_reversal_tick);

    let p_baseline = metrics::action_baseline_probability();
    let h_baseline = metrics::action_baseline_entropy();
    let strong_foraging_reference = 0.55;
    let competitive_predation_reference = 0.002;
    let p_component = mean_p_fwd_food
        .map(|value| {
            clamp01(
                (value - p_baseline) / (strong_foraging_reference - p_baseline).max(f64::EPSILON),
            )
        })
        .unwrap_or(0.0);
    let mi_component = mean_mi_sa.map(|value| clamp01(value / 0.10)).unwrap_or(0.0);
    let entropy_component = mean_h_action
        .map(|value| entropy_component_score(value, h_baseline))
        .unwrap_or(0.0);
    let predation_component = mean_predation_rate
        .map(|value| clamp01(value / competitive_predation_reference))
        .unwrap_or(0.0);

    let score = 100.0
        * (0.40 * p_component
            + 0.25 * mi_component
            + 0.10 * entropy_component
            + 0.25 * predation_component);

    AggregateScore {
        score,
        score_median: score,
        score_stddev: 0.0,
        score_min: score,
        score_max: score,
        window_start_tick,
        window_end_tick,
        mean_p_fwd_food,
        mean_mi_sa,
        mean_mi_sa_juvenile,
        mean_mi_sa_adult,
        mean_h_action,
        mean_predation_rate,
        mean_foraging_rate,
        mean_attack_attempt_rate,
        mean_attack_success_rate,
        mean_idle_fraction,
        mean_reproduction_efficiency,
        mean_lineage_diversity,
        mean_damage_avoidance,
        mean_reward_reversal_shift,
        mean_action_histogram,
        reward_reversal_adaptation_ticks,
        p_component,
        mi_component,
        entropy_component,
        predation_component,
    }
}

fn average_aggregate_scores(seed_summaries: &[SeedValidationSummary]) -> AggregateScore {
    let first = seed_summaries
        .first()
        .expect("multi-seed validation requires at least one seed");
    let score_stats = score_stats(
        seed_summaries
            .iter()
            .map(|summary| summary.aggregate_score.score),
    );
    AggregateScore {
        score: score_stats.mean,
        score_median: score_stats.median,
        score_stddev: score_stats.stddev,
        score_min: score_stats.min,
        score_max: score_stats.max,
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
        mean_mi_sa_juvenile: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_mi_sa_juvenile),
        ),
        mean_mi_sa_adult: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_mi_sa_adult),
        ),
        mean_h_action: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_h_action),
        ),
        mean_predation_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_predation_rate),
        ),
        mean_foraging_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_foraging_rate),
        ),
        mean_attack_attempt_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_attack_attempt_rate),
        ),
        mean_attack_success_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_attack_success_rate),
        ),
        mean_idle_fraction: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_idle_fraction),
        ),
        mean_reproduction_efficiency: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_reproduction_efficiency),
        ),
        mean_lineage_diversity: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_lineage_diversity),
        ),
        mean_damage_avoidance: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_damage_avoidance),
        ),
        mean_reward_reversal_shift: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_reward_reversal_shift),
        ),
        mean_action_histogram: mean_histogram(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_action_histogram),
        ),
        reward_reversal_adaptation_ticks: mean_option_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.reward_reversal_adaptation_ticks),
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
        predation_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.predation_component),
        ),
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoreStats {
    mean: f64,
    median: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

fn score_stats(values: impl Iterator<Item = f64>) -> ScoreStats {
    let mut scores = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if scores.is_empty() {
        return ScoreStats {
            mean: 0.0,
            median: 0.0,
            stddev: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    scores.sort_by(|a, b| a.total_cmp(b));
    let len = scores.len();
    let mean = scores.iter().sum::<f64>() / len as f64;
    let median = if len % 2 == 0 {
        (scores[len / 2 - 1] + scores[len / 2]) / 2.0
    } else {
        scores[len / 2]
    };
    let variance = scores
        .iter()
        .map(|score| {
            let delta = *score - mean;
            delta * delta
        })
        .sum::<f64>()
        / len as f64;

    ScoreStats {
        mean,
        median,
        stddev: variance.sqrt(),
        min: scores[0],
        max: scores[len - 1],
    }
}

fn default_worker_threads(seed_count: usize) -> usize {
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, seed_count.max(1))
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
            predation_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].predation_rate),
            ),
            foraging_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].foraging_rate),
            ),
            attack_attempt_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].attack_attempt_rate),
            ),
            attack_success_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].attack_success_rate),
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
            brain_size_stddev: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_stddev),
            ),
            brain_size_p10: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_p10),
            ),
            brain_size_p50: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_p50),
            ),
            brain_size_p90: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_p90),
            ),
            lineage_diversity: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].lineage_diversity),
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
            mi_sa_juvenile: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa_juvenile),
            ),
            mi_sa_adult: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa_adult),
            ),
            h_action: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].h_action),
            ),
            idle_fraction: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].idle_fraction),
            ),
            reproduction_efficiency: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].reproduction_efficiency),
            ),
            damage_avoidance: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].damage_avoidance),
            ),
            reward_reversal_shift: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].reward_reversal_shift),
            ),
            util: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].util),
            ),
            action_histogram: mean_histogram(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].action_histogram),
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

fn mean_histogram(values: impl Iterator<Item = [f64; N_ACTIONS]>) -> [f64; N_ACTIONS] {
    let mut sums = [0.0; N_ACTIONS];
    let mut count = 0.0;
    for value in values {
        for idx in 0..N_ACTIONS {
            if value[idx].is_finite() {
                sums[idx] += value[idx];
            }
        }
        count += 1.0;
    }
    if count == 0.0 {
        return [0.0; N_ACTIONS];
    }
    for sum in &mut sums {
        *sum /= count;
    }
    sums
}

fn mean_action_histogram(window: &[IntervalMetrics]) -> [f64; N_ACTIONS] {
    mean_histogram(window.iter().map(|row| row.action_histogram))
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
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn entropy_component_score(mean_h_action: f64, h_baseline: f64) -> f64 {
    // Prefer a small but nonzero action repertoire: competent agents should be
    // decisive, but a lifetime histogram of exactly one action usually means
    // collapse rather than adaptive sequential behavior.
    let entropy_target = 0.35 * h_baseline;
    let high_entropy_width = 0.45 * h_baseline;

    if mean_h_action <= entropy_target {
        let progress = clamp01(mean_h_action / entropy_target.max(f64::EPSILON));
        0.25 + 0.75 * progress * progress
    } else {
        let excess =
            clamp01((mean_h_action - entropy_target) / high_entropy_width.max(f64::EPSILON));
        1.0 - excess
    }
}

fn reward_reversal_adaptation_ticks(
    timeseries: &[IntervalMetrics],
    reward_reversal_tick: Option<u64>,
) -> Option<u64> {
    const REVERSAL_SHIFT_THRESHOLD: f64 = 0.12;
    let reward_reversal_tick = reward_reversal_tick?;
    timeseries
        .iter()
        .find(|row| {
            row.tick > reward_reversal_tick
                && row
                    .reward_reversal_shift
                    .is_some_and(|shift| shift >= REVERSAL_SHIFT_THRESHOLD)
        })
        .map(|row| row.tick.saturating_sub(reward_reversal_tick))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::N_ACTIONS;
    use sim_types::ActionType;

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
            reward_reversal_tick: reward_reversal_tick_for_run(100),
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
    fn entropy_component_prefers_small_purposeful_repertoires() {
        let h_baseline = metrics::action_baseline_entropy();

        let collapsed = entropy_component_score(
            entropy_from_actions(&repeated(ActionType::Forward, 24)),
            h_baseline,
        );
        let purposeful = entropy_component_score(
            entropy_from_actions(&sequence_counts(&[
                (ActionType::Forward, 15),
                (ActionType::Eat, 3),
                (ActionType::TurnLeft, 2),
            ])),
            h_baseline,
        );
        let exploratory = entropy_component_score(
            entropy_from_actions(&sequence_counts(&[
                (ActionType::Forward, 10),
                (ActionType::Eat, 5),
                (ActionType::TurnLeft, 5),
                (ActionType::TurnRight, 5),
            ])),
            h_baseline,
        );
        let random_like = entropy_component_score(
            entropy_from_actions(&sequence_counts(&[
                (ActionType::Idle, 4),
                (ActionType::TurnLeft, 4),
                (ActionType::TurnRight, 4),
                (ActionType::Forward, 4),
                (ActionType::Eat, 4),
                (ActionType::Attack, 4),
                (ActionType::Reproduce, 4),
            ])),
            h_baseline,
        );

        assert!(purposeful > collapsed);
        assert!(purposeful > exploratory);
        assert!(collapsed > random_like);
        assert!(exploratory > random_like);
    }

    fn test_output_dir(suffix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("sim-validation-test-{suffix}-{nanos}"))
    }

    fn repeated(action: ActionType, count: usize) -> Vec<ActionType> {
        let mut actions = Vec::with_capacity(count);
        for _ in 0..count {
            actions.push(action);
        }
        actions
    }

    fn sequence_counts(counts: &[(ActionType, usize)]) -> Vec<ActionType> {
        let total = counts.iter().map(|(_, count)| *count).sum();
        let mut actions = Vec::with_capacity(total);
        for (action, count) in counts {
            for _ in 0..*count {
                actions.push(*action);
            }
        }
        actions
    }

    fn entropy_from_actions(actions: &[ActionType]) -> f64 {
        let mut counts = [0_u64; N_ACTIONS];
        for action in actions {
            let idx = match action {
                ActionType::Idle => 0,
                ActionType::TurnLeft => 1,
                ActionType::TurnRight => 2,
                ActionType::Forward => 3,
                ActionType::Eat => 4,
                ActionType::Attack => 5,
                ActionType::Reproduce => 6,
            };
            counts[idx] = counts[idx].saturating_add(1);
        }

        let total: u64 = counts.iter().sum();
        assert!(total > 0, "entropy test sequences must be non-empty");

        let mut entropy = 0.0;
        for count in counts {
            if count == 0 {
                continue;
            }
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
        entropy
    }
}
