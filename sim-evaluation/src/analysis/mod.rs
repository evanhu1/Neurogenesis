//! Derive interpretable metrics from a persisted dataset. This is the only
//! place that knows how to turn raw rows into `IntervalMetrics`, pillar
//! scores, and demographic analytics. The sim-run layer emits raw data; the
//! reporting layer consumes analysis output; this module is the pivot.

pub mod averaging;
pub mod cli;
pub mod demographics;
pub mod intervals;
pub mod pillars;

pub use averaging::{average_demographic_analytics, average_pillar_scores, average_timeseries};
pub use demographics::compute_demographic_analytics;
pub use intervals::derive_interval_metrics;
pub use pillars::{compute_pillar_scores, ScoringWindow};

use crate::dataset::DatasetReader;
use crate::output::{write_summary_json, write_timeseries_csv};
use crate::report::{write_html_report, HtmlReportContext, HtmlReportMeta};
use crate::types::{DemographicAnalytics, IntervalMetrics, PillarScores, SeedEvaluationSummary};
use anyhow::Result;
use chrono::Utc;
use std::path::Path;

pub struct AnalysisOptions {
    pub report_every: u64,
    pub total_ticks: u64,
    pub min_lifetime: u64,
    pub scoring_window: ScoringWindow,
}

pub struct AnalysisOutput {
    pub timeseries: Vec<IntervalMetrics>,
    pub pillars: PillarScores,
    pub demographics: DemographicAnalytics,
}

pub fn analyze(dataset: &DatasetReader, options: &AnalysisOptions) -> AnalysisOutput {
    let timeseries = derive_interval_metrics(
        dataset,
        options.report_every,
        options.total_ticks,
        options.min_lifetime,
    );
    let pillars = compute_pillar_scores(&timeseries, options.scoring_window);
    let demographics = compute_demographic_analytics(dataset, options.total_ticks);
    AnalysisOutput {
        timeseries,
        pillars,
        demographics,
    }
}

/// Write the three per-seed artifacts (`summary.json`, `timeseries.csv`,
/// `report.html`) that the evaluation CLI and the `analyze` subcommand both
/// produce. `timeseries_label` is the human label shown above the table.
pub fn write_per_seed_artifacts(
    out_dir: &Path,
    summary: &SeedEvaluationSummary,
    report_every: u64,
    min_lifetime: u64,
    timeseries_label: &str,
) -> Result<()> {
    write_summary_json(out_dir, summary)?;
    write_timeseries_csv(out_dir, &summary.timeseries)?;
    write_html_report(
        out_dir,
        &HtmlReportMeta::from_pillars(
            &summary.pillars,
            HtmlReportContext {
                title: summary.title.clone(),
                ticks: summary.ticks,
                report_every,
                min_lifetime,
                control: summary.control,
                total_time_seconds: summary.total_time_seconds,
                generated_at_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                timeseries_label: timeseries_label.to_owned(),
                per_seed_rows: Vec::new(),
            },
        ),
        &summary.timeseries,
    )
}
