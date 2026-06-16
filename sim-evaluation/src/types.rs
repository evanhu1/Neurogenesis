use crate::report::ComparisonMetricRow;
use serde::Serialize;
use std::path::PathBuf;

// Metric readouts are owned by `sim-metrics`; re-exported under the historical
// `crate::types::…` path so the summary/report/comparison code is unchanged.
pub(crate) use sim_metrics::{IntervalMetrics, PillarScores};

#[derive(Debug, Clone)]
pub(crate) struct HarnessRunOptions {
    pub(crate) seeds: Vec<u64>,
    pub(crate) ticks: u64,
    pub(crate) report_every: u64,
    pub(crate) out_dir: PathBuf,
    pub(crate) title: Option<String>,
    pub(crate) control: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SeedRunOptions {
    pub(crate) seed: u64,
    pub(crate) ticks: u64,
    pub(crate) report_every: u64,
    pub(crate) out_dir: PathBuf,
    pub(crate) title: Option<String>,
    pub(crate) control: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SeedEvaluationSummary {
    pub(crate) title: Option<String>,
    pub(crate) seed: u64,
    pub(crate) ticks: u64,
    pub(crate) control: bool,
    pub(crate) total_time_seconds: f64,
    pub(crate) pillars: PillarScores,
    pub(crate) state_hash: String,
    pub(crate) timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EvaluationSummary {
    pub(crate) title: Option<String>,
    pub(crate) seeds: Vec<u64>,
    pub(crate) ticks: u64,
    pub(crate) control: bool,
    pub(crate) worker_threads: usize,
    pub(crate) total_time_seconds: f64,
    pub(crate) pillars: PillarScores,
    pub(crate) seed_summaries: Vec<SeedRunSummary>,
    pub(crate) timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SeedRunSummary {
    pub(crate) seed: u64,
    pub(crate) out_dir: PathBuf,
    pub(crate) total_time_seconds: f64,
    pub(crate) pillars: PillarScores,
    pub(crate) state_hash: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ComparisonSummary {
    pub(crate) title: Option<String>,
    pub(crate) seeds: Vec<u64>,
    pub(crate) ticks: u64,
    pub(crate) control_label: String,
    pub(crate) treatment_label: String,
    pub(crate) total_time_seconds: f64,
    pub(crate) control: EvaluationSummary,
    pub(crate) treatment: EvaluationSummary,
    pub(crate) metric_rows: Vec<ComparisonMetricRow>,
}
