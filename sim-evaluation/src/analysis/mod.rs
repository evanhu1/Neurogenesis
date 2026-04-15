//! Derive interpretable metrics from a persisted dataset. This is the only
//! place that knows how to turn raw rows into `IntervalMetrics`, pillar
//! scores, and reproduction analytics. The sim-run layer emits raw data; the
//! reporting layer consumes analysis output; this module is the pivot.

pub mod averaging;
pub mod cli;
pub mod intervals;
pub mod pillars;
pub mod reproduction;

pub use averaging::{average_pillar_scores, average_reproduction_analytics, average_timeseries};
pub use intervals::derive_interval_metrics;
pub use pillars::{compute_pillar_scores, ScoringWindow};
pub use reproduction::compute_reproduction_analytics;

use crate::dataset::DatasetReader;
use crate::types::{IntervalMetrics, PillarScores, ReproductionAnalytics};

pub struct AnalysisOptions {
    pub report_every: u64,
    pub total_ticks: u64,
    pub min_lifetime: u64,
    pub scoring_window: ScoringWindow,
}

pub struct AnalysisOutput {
    pub timeseries: Vec<IntervalMetrics>,
    pub pillars: PillarScores,
    pub reproduction: ReproductionAnalytics,
}

pub fn analyze(dataset: &DatasetReader, options: &AnalysisOptions) -> AnalysisOutput {
    let timeseries = derive_interval_metrics(
        dataset,
        options.report_every,
        options.total_ticks,
        options.min_lifetime,
    );
    let pillars = compute_pillar_scores(&timeseries, options.scoring_window);
    let reproduction = compute_reproduction_analytics(dataset, options.total_ticks);
    AnalysisOutput {
        timeseries,
        pillars,
        reproduction,
    }
}
