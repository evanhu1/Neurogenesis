use crate::types::{ComparisonSummary, EvaluationSummary, IntervalMetrics};
use anyhow::Result;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// Numeric mean helpers moved to `sim-metrics`; re-exported under the historical
// `crate::output::…` path so callers (comparison, averaging) are unchanged.
pub(crate) use sim_metrics::{mean_option, mean_round_u32};

pub(crate) fn write_summary_json<T: Serialize>(out_dir: &Path, summary: &T) -> Result<()> {
    let summary_path = out_dir.join("summary.json");
    let json = serde_json::to_vec_pretty(summary)?;
    fs::write(summary_path, json)?;
    Ok(())
}

pub(crate) fn write_timeseries_csv(out_dir: &Path, rows: &[IntervalMetrics]) -> Result<()> {
    let csv_path = out_dir.join("timeseries.csv");
    let mut csv = BufWriter::new(File::create(csv_path)?);
    writeln!(
        csv,
        "start_tick,tick,pop,action_effectiveness,plant_consumption_rate,prey_consumption_rate,mi_sa,learning_slope"
    )?;
    for metrics in rows {
        writeln!(
            csv,
            "{start_tick},{tick},{pop},{action_effectiveness},{plant_consumption_rate},{prey_consumption_rate},{mi_sa},{learning_slope}",
            start_tick = metrics.start_tick,
            tick = metrics.tick,
            pop = metrics.pop,
            action_effectiveness = csv_opt(metrics.action_effectiveness),
            plant_consumption_rate = csv_opt(metrics.plant_consumption_rate),
            prey_consumption_rate = csv_opt(metrics.prey_consumption_rate),
            mi_sa = csv_opt(metrics.mi_sa),
            learning_slope = csv_opt(metrics.learning_slope),
        )?;
    }
    csv.flush()?;
    Ok(())
}

fn csv_opt(value: Option<f64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

pub(crate) fn default_output_dir(seeds: &[u64]) -> PathBuf {
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    if seeds.len() == 1 {
        return PathBuf::from(format!(
            "artifacts/evaluation/{}_seed_{}",
            timestamp, seeds[0]
        ));
    }
    PathBuf::from(format!(
        "artifacts/evaluation/{}_seeds_{}",
        timestamp,
        seed_slug(seeds)
    ))
}

pub(crate) fn format_seed_list(seeds: &[u64]) -> String {
    seeds
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn normalize_seeds(seeds: Vec<u64>) -> Vec<u64> {
    let mut unique = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if !unique.contains(&seed) {
            unique.push(seed);
        }
    }
    unique
}

pub(crate) fn print_comparison_summary(out_dir: &Path, comparison: &ComparisonSummary) {
    let report_path = out_dir.join("comparison_report.html");
    println!("wrote artifacts to {}", out_dir.display());
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
}

pub(crate) fn print_evaluation_summary(out_dir: &Path, summary: &EvaluationSummary) {
    let report_path = out_dir.join("report.html");
    println!("wrote artifacts to {}", out_dir.display());
    println!("html_report: {}", report_path.display());
    println!("browser_url: {}", browser_file_url(&report_path));
    println!("seeds: {}", format_seed_list(&summary.seeds));
    println!("worker_threads: {}", summary.worker_threads);
    let pillars = &summary.pillars;
    println!(
        "metric_coverage: action={}/{} mi={}/{} plant={}/{} prey={}/{} learning={}/{}",
        pillars.coverage.action_effectiveness,
        pillars.coverage.runs_total,
        pillars.coverage.mi_sa,
        pillars.coverage.runs_total,
        pillars.coverage.plant_consumption_rate,
        pillars.coverage.runs_total,
        pillars.coverage.prey_consumption_rate,
        pillars.coverage.runs_total,
        pillars.coverage.learning_slope,
        pillars.coverage.runs_total,
    );
    println!(
        "plant_consumption_rate: {}",
        fmt_option(pillars.mean_plant_consumption_rate, 4),
    );
    println!(
        "prey_consumption_rate: {}",
        fmt_option(pillars.mean_prey_consumption_rate, 4),
    );
    println!(
        "action_effectiveness: {} | mi_sa: {}",
        fmt_option(pillars.mean_action_effectiveness, 4),
        fmt_option(pillars.mean_mi_sa, 4),
    );
    println!(
        "learning_slope: {}",
        fmt_option(pillars.mean_learning_slope, 6),
    );
    for seed_summary in &summary.seed_summaries {
        let p = &seed_summary.pillars;
        println!(
            "seed[{}]: plant={} prey={} effectiveness={} mi={} learning={}",
            seed_summary.seed,
            fmt_option(p.mean_plant_consumption_rate, 4),
            fmt_option(p.mean_prey_consumption_rate, 4),
            fmt_option(p.mean_action_effectiveness, 4),
            fmt_option(p.mean_mi_sa, 4),
            fmt_option(p.mean_learning_slope, 6),
        );
    }
    println!("total_time_seconds: {:.3}", summary.total_time_seconds);
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

pub(crate) fn fmt_option(value: Option<f64>, decimals: usize) -> String {
    value
        .map(|value| format!("{value:.decimals$}"))
        .unwrap_or_else(|| "NA".to_owned())
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
