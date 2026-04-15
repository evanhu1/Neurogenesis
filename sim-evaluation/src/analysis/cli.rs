//! `sim-evaluation analyze <dataset-dir>` entrypoint. Re-derives metrics,
//! pillars and reproduction analytics from a persisted seed dataset and
//! rewrites the human-readable artifacts (`summary.json`, `timeseries.csv`,
//! `report.html`) without re-running the simulation.

use super::{analyze, write_per_seed_artifacts, AnalysisOptions, ScoringWindow};
use crate::dataset::{DatasetReader, Manifest};
use crate::types::SeedEvaluationSummary;
use anyhow::Result;
use std::path::Path;

pub fn analyze_dataset_dir(dataset_dir: &Path) -> Result<()> {
    let manifest = Manifest::read(dataset_dir)?;
    let dataset = DatasetReader::load(dataset_dir)?;
    let analysis = analyze(
        &dataset,
        &AnalysisOptions {
            report_every: manifest.report_every,
            total_ticks: manifest.total_ticks,
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
        demographics: analysis.demographics.clone(),
        state_hash: String::new(),
        timeseries: analysis.timeseries.clone(),
    };
    write_per_seed_artifacts(
        dataset_dir,
        &summary,
        manifest.report_every,
        "re-analyzed from dataset",
    )?;
    println!("re-analyzed dataset: {}", dataset_dir.display());
    Ok(())
}
