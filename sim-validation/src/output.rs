use crate::{
    ledger::N_ACTIONS,
    metrics::IntervalMetrics,
    report::Reporter,
    types::{ComparisonSummary, ValidationSummary},
};
use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn write_summary_json<T: Serialize>(out_dir: &Path, summary: &T) -> Result<()> {
    let summary_path = out_dir.join("summary.json");
    let json = serde_json::to_vec_pretty(summary)?;
    fs::write(summary_path, json)?;
    Ok(())
}

pub(crate) fn write_timeseries_csv(out_dir: &Path, rows: &[IntervalMetrics]) -> Result<()> {
    let mut reporter = Reporter::new(out_dir)?;
    for row in rows {
        reporter.emit(row)?;
    }
    reporter.flush()?;
    Ok(())
}

pub(crate) fn default_output_dir(seeds: &[u64]) -> PathBuf {
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
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

pub(crate) fn print_validation_summary(out_dir: &Path, summary: &ValidationSummary) {
    let report_path = out_dir.join("report.html");
    println!("wrote artifacts to {}", out_dir.display());
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
        "aggregate_viability_pillar: {:.3} [life={:.3} reproduction={:.3} damage={:.3}]",
        summary.aggregate_score.viability_pillar,
        summary.aggregate_score.viability_life_component,
        summary.aggregate_score.viability_reproduction_component,
        summary.aggregate_score.viability_damage_component
    );
    println!(
        "aggregate_foraging_pillar: {:.3} [p_fwd_food={:.3} foraging_rate={:.3}]",
        summary.aggregate_score.foraging_pillar,
        summary.aggregate_score.foraging_p_fwd_food_component,
        summary.aggregate_score.foraging_rate_component
    );
    println!(
        "aggregate_control_pillar: {:.3} [adult_mi={:.3} entropy={:.3} anti_idle={:.3} util={:.3}]",
        summary.aggregate_score.control_pillar,
        summary.aggregate_score.control_adult_mi_component,
        summary.aggregate_score.control_entropy_component,
        summary.aggregate_score.control_anti_idle_component,
        summary.aggregate_score.control_util_component
    );
    println!(
        "aggregate_competition_pillar: {:.3} [predation={:.3} attack_success={:.3} attack_attempts={:.3}]",
        summary.aggregate_score.competition_pillar,
        summary.aggregate_score.competition_predation_component,
        summary.aggregate_score.competition_attack_success_component,
        summary.aggregate_score.competition_attack_attempt_component
    );
    println!(
        "aggregate_adaptation_pillar: {:.3} [reversal={:.3} juvenile_mi={:.3} diversity={:.3}]",
        summary.aggregate_score.adaptation_pillar,
        summary.aggregate_score.adaptation_reversal_component,
        summary.aggregate_score.adaptation_juvenile_mi_component,
        summary.aggregate_score.adaptation_diversity_component
    );
    println!("total_time_seconds: {:.3}", summary.total_time_seconds);
}

pub(crate) fn mean_histogram(values: impl Iterator<Item = [f64; N_ACTIONS]>) -> [f64; N_ACTIONS] {
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

pub(crate) fn mean_option_u64(values: impl Iterator<Item = Option<u64>>) -> Option<u64> {
    mean_option(values.map(|value| value.map(|inner| inner as f64)))
        .map(|value| value.round() as u64)
}

pub(crate) fn mean_round_u64(values: impl Iterator<Item = u64>) -> u64 {
    mean_f64(values.map(|value| value as f64)).round() as u64
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

fn fmt_option(value: Option<f64>, decimals: usize) -> String {
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
