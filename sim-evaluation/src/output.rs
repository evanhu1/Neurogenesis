use crate::{
    dataset::ACTION_COUNT,
    types::{ComparisonSummary, EvaluationSummary, IntervalMetrics},
};
use anyhow::Result;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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
        "tick,pop,action_effectiveness,plant_consumption_rate,prey_consumption_rate,mi_sa,learning_slope"
    )?;
    for metrics in rows {
        writeln!(
            csv,
            "{tick},{pop},{action_effectiveness},{plant_consumption_rate},{prey_consumption_rate},{mi_sa},{learning_slope}",
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
        "foraging_pillar: {:.3} [plant_consumption_rate={}]",
        pillars.foraging_pillar,
        fmt_option(pillars.mean_plant_consumption_rate, 4),
    );
    println!(
        "predation_pillar: {:.3} [prey_consumption_rate={}]",
        pillars.predation_pillar,
        fmt_option(pillars.mean_prey_consumption_rate, 4),
    );
    println!(
        "intelligence_pillar: {:.3} [effectiveness={:.3} mi={:.3}]",
        pillars.intelligence_pillar,
        pillars.intelligence_effectiveness_component,
        pillars.intelligence_mi_component,
    );
    println!(
        "learning_pillar: {:.3} [mean_slope={}]",
        pillars.learning_pillar,
        fmt_option(pillars.mean_learning_slope, 6),
    );
    for seed_summary in &summary.seed_summaries {
        let p = &seed_summary.pillars;
        println!(
            "seed[{}]: foraging={:.3} predation={:.3} intelligence={:.3} learning={:.3}",
            seed_summary.seed,
            p.foraging_pillar,
            p.predation_pillar,
            p.intelligence_pillar,
            p.learning_pillar,
        );
    }
    println!("total_time_seconds: {:.3}", summary.total_time_seconds);
}

pub(crate) fn mean_histogram(
    values: impl Iterator<Item = [f64; ACTION_COUNT]>,
) -> [f64; ACTION_COUNT] {
    let mut sums = [0.0; ACTION_COUNT];
    let mut count = 0.0;
    for value in values {
        for idx in 0..ACTION_COUNT {
            if value[idx].is_finite() {
                sums[idx] += value[idx];
            }
        }
        count += 1.0;
    }
    if count == 0.0 {
        return [0.0; ACTION_COUNT];
    }
    for sum in &mut sums {
        *sum /= count;
    }
    sums
}

pub(crate) fn mean_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
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

pub(crate) fn mean_round_u32(values: impl Iterator<Item = u32>) -> u32 {
    mean_f64(values.map(|value| value as f64)).round() as u32
}

pub(crate) fn mean_f64(values: impl Iterator<Item = f64>) -> f64 {
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
