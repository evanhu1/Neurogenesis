//! Derive interpretable metrics from a persisted dataset. This is the only
//! place that knows how to turn raw rows into `IntervalMetrics` and pillar
//! scores. The sim-run layer emits raw data; the reporting layer consumes
//! analysis output; this module is the pivot.

pub mod averaging;
pub mod cli;
pub mod intervals;
pub mod pillars;

pub use averaging::{average_pillar_scores, average_timeseries};
pub use intervals::derive_interval_metrics;
pub use pillars::compute_pillar_scores;

use crate::dataset::DatasetReader;
use crate::output::{write_summary_json, write_timeseries_csv};
use crate::report::{write_html_report, HtmlReportContext, PerSeedReportRow};
use crate::types::{EvaluationSummary, IntervalMetrics, PillarScores, SeedEvaluationSummary};
use anyhow::Result;
use chrono::Utc;
use std::path::Path;

pub struct AnalysisOptions {
    pub report_every: u64,
    pub total_ticks: u64,
}

pub struct AnalysisOutput {
    pub timeseries: Vec<IntervalMetrics>,
    pub pillars: PillarScores,
}

pub fn analyze(dataset: &DatasetReader, options: &AnalysisOptions) -> AnalysisOutput {
    let timeseries = derive_interval_metrics(dataset, options.report_every, options.total_ticks);
    let pillars = compute_pillar_scores(&timeseries);
    AnalysisOutput {
        timeseries,
        pillars,
    }
}

/// Write the three per-seed artifacts (`summary.json`, `timeseries.csv`,
/// `report.html`) that the evaluation CLI and the `analyze` subcommand both
/// produce. `timeseries_label` is the human label shown above the table.
pub fn write_per_seed_artifacts(
    out_dir: &Path,
    summary: &SeedEvaluationSummary,
    report_every: u64,
    timeseries_label: &str,
) -> Result<()> {
    write_summary_json(out_dir, summary)?;
    write_timeseries_csv(out_dir, &summary.timeseries)?;
    write_html_report(
        out_dir,
        &summary.pillars,
        &HtmlReportContext {
            title: summary.title.clone(),
            ticks: summary.ticks,
            report_every,
            control: summary.control,
            total_time_seconds: summary.total_time_seconds,
            generated_at_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            timeseries_label: timeseries_label.to_owned(),
            per_seed_rows: Vec::new(),
        },
        &summary.timeseries,
    )
}

/// Write the three run-level aggregate artifacts (`summary.json`,
/// `timeseries.csv`, `report.html`) at the run root — the "mean across seeds"
/// view. Shared between the live evaluation harness and the re-analysis path.
pub fn write_aggregate_artifacts(
    out_dir: &Path,
    summary: &EvaluationSummary,
    report_every: u64,
    generated_at_utc: &str,
) -> Result<()> {
    write_timeseries_csv(out_dir, &summary.timeseries)?;
    write_summary_json(out_dir, summary)?;
    let per_seed_rows = summary
        .seed_summaries
        .iter()
        .map(|seed_summary| PerSeedReportRow {
            seed: seed_summary.seed,
            total_time_seconds: seed_summary.total_time_seconds,
            foraging_pillar: seed_summary.pillars.foraging_pillar,
            predation_pillar: seed_summary.pillars.predation_pillar,
            intelligence_pillar: seed_summary.pillars.intelligence_pillar,
            learning_pillar: seed_summary.pillars.learning_pillar,
            report_href: format!("seed_{}/report.html", seed_summary.seed),
        })
        .collect();
    write_html_report(
        out_dir,
        &summary.pillars,
        &HtmlReportContext {
            title: summary.title.clone(),
            ticks: summary.ticks,
            report_every,
            control: summary.control,
            total_time_seconds: summary.total_time_seconds,
            generated_at_utc: generated_at_utc.to_owned(),
            timeseries_label: "mean across seeds".to_owned(),
            per_seed_rows,
        },
        &summary.timeseries,
    )
}
