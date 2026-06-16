use crate::report::ComparisonMetricRow;
use serde::Serialize;
use std::path::PathBuf;

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

/// One row per reporting interval — the primary timeseries format consumed by
/// reports, comparisons, and the CSV export. Derived by `analysis::intervals`
/// from the raw dataset by pooling descendant lifetime rows per death-tick
/// interval. Every rate uses total actions taken as the denominator.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct IntervalMetrics {
    pub tick: u64,
    /// Descendant population (viability context, not a competence score).
    pub pop: u32,
    /// Successful contingent actions / total actions. Idling and spinning
    /// self-penalise because they inflate the denominator without succeeding.
    pub action_effectiveness: Option<f64>,
    /// Plant consumptions / total actions — foraging "consume success".
    pub plant_consumption_rate: Option<f64>,
    /// Prey/corpse consumptions / total actions — predation competence.
    pub prey_consumption_rate: Option<f64>,
    /// Miller-Madow MI between food-visibility context and selected action.
    pub mi_sa: Option<f64>,
    /// Mean within-life success-vs-age slope over descendants — in-life learning.
    pub learning_slope: Option<f64>,
}

/// Per-pillar evaluation readout. There is deliberately no single aggregate
/// score here — each axis stands on its own. Every signal is chosen to be
/// hard to game: foraging/predation reward real consumption per action,
/// intelligence rewards successful action and sensing-conditioned behaviour,
/// and learning rewards within-life improvement (≈0 under the random control).
#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct PillarScores {
    pub(crate) window_start_tick: u64,
    pub(crate) window_end_tick: u64,
    pub(crate) mean_action_effectiveness: Option<f64>,
    pub(crate) mean_mi_sa: Option<f64>,
    pub(crate) mean_plant_consumption_rate: Option<f64>,
    pub(crate) mean_prey_consumption_rate: Option<f64>,
    pub(crate) mean_learning_slope: Option<f64>,
    pub(crate) intelligence_effectiveness_component: f64,
    pub(crate) intelligence_mi_component: f64,
    pub(crate) foraging_pillar: f64,
    pub(crate) predation_pillar: f64,
    pub(crate) intelligence_pillar: f64,
    pub(crate) learning_pillar: f64,
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
