//! `sim-evaluation analyze <dataset-dir>` entrypoint. Re-derives metrics,
//! pillars and reproduction analytics from a persisted seed dataset and
//! rewrites the human-readable artifacts (`summary.json`, `timeseries.csv`,
//! `report.html`) without re-running the simulation.

use super::{analyze, AnalysisOptions, ScoringWindow};
use crate::dataset::{DatasetReader, Manifest};
use crate::output::{write_summary_json, write_timeseries_csv};
use crate::report::{write_html_report, HtmlReportContext, HtmlReportMeta};
use crate::types::SeedEvaluationSummary;
use anyhow::Result;
use chrono::Utc;
use std::path::Path;

const DEFAULT_MIN_LIFETIME: u64 = 30;

pub fn analyze_dataset_dir(dataset_dir: &Path) -> Result<()> {
    let manifest = Manifest::read(dataset_dir)?;
    let dataset = DatasetReader::load(dataset_dir)?;
    let analysis = analyze(
        &dataset,
        &AnalysisOptions {
            report_every: manifest.report_every,
            total_ticks: manifest.total_ticks,
            min_lifetime: DEFAULT_MIN_LIFETIME,
            scoring_window: ScoringWindow::default(),
        },
    );

    let summary = SeedEvaluationSummary {
        title: None,
        seed: manifest.seed,
        ticks: manifest.total_ticks,
        control: false,
        total_time_seconds: 0.0,
        pillars: analysis.pillars.clone(),
        experiment_readouts: analysis.reproduction.clone(),
        state_hash: String::new(),
        timeseries: analysis.timeseries.clone(),
    };
    write_summary_json(dataset_dir, &summary)?;
    write_timeseries_csv(dataset_dir, &summary.timeseries)?;
    write_html_report(
        dataset_dir,
        &HtmlReportMeta::from_pillars(
            &analysis.pillars,
            HtmlReportContext {
                title: None,
                ticks: manifest.total_ticks,
                report_every: manifest.report_every,
                min_lifetime: DEFAULT_MIN_LIFETIME,
                control: false,
                total_time_seconds: 0.0,
                generated_at_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                timeseries_label: "re-analyzed from dataset".to_owned(),
                per_seed_rows: Vec::new(),
            },
        ),
        &summary.timeseries,
    )?;
    println!("re-analyzed dataset: {}", dataset_dir.display());
    Ok(())
}
