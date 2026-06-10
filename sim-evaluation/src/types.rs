use crate::dataset::ACTION_COUNT;
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
    pub(crate) demographics: DemographicAnalytics,
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
    pub(crate) demographics: DemographicAnalytics,
    pub(crate) seed_summaries: Vec<SeedRunSummary>,
    pub(crate) timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SeedRunSummary {
    pub(crate) seed: u64,
    pub(crate) out_dir: PathBuf,
    pub(crate) total_time_seconds: f64,
    pub(crate) pillars: PillarScores,
    pub(crate) demographics: DemographicAnalytics,
    pub(crate) state_hash: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct DemographicAnalytics {
    pub(crate) successful_births: u64,
    pub(crate) blocked_births: u64,
    pub(crate) parent_died_during_reproduction: u64,
    pub(crate) survived_to_30: u64,
    pub(crate) survived_to_maturity: u64,
    pub(crate) mean_parent_energy_after_successful_birth: Option<f64>,
    pub(crate) mean_age_at_first_successful_reproduction: Option<f64>,
    pub(crate) mean_successful_birth_interval: Option<f64>,
}

/// One row per reporting interval — the primary timeseries format consumed by
/// reports, comparisons, and the CSV export. Derived by `analysis::intervals`
/// from the raw dataset.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct IntervalMetrics {
    pub tick: u64,
    pub pop: u32,
    pub births: u64,
    pub deaths: u64,
    pub food: u64,
    pub max_generation: Option<u64>,
    pub attack_attempt_rate: Option<f64>,
    pub attack_success_rate: Option<f64>,
    pub failed_action_rate: Option<f64>,
    pub ate_pct: Option<f64>,
    pub cons_mean: Option<f64>,
    pub neurons: Option<f64>,
    pub synapses: Option<f64>,
    pub p_fwd_food: Option<f64>,
    pub mi_sa: Option<f64>,
    pub idle_fraction: Option<f64>,
    pub util: Option<f64>,
    pub generation_time: Option<f64>,
    pub abs_td_error: Option<f64>,
    pub age_correlated_competence: Option<f64>,
    pub action_histogram: [f64; ACTION_COUNT],
}

/// Per-pillar evaluation readout. There is deliberately no single aggregate
/// score here — each pillar stands on its own. The intelligence pillar is
/// composed of niche-agnostic behavioural measures so predators and foragers
/// can still be compared on the same axis.
#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct PillarScores {
    pub(crate) window_start_tick: u64,
    pub(crate) window_end_tick: u64,
    pub(crate) mean_p_fwd_food: Option<f64>,
    pub(crate) mean_mi_sa: Option<f64>,
    pub(crate) mean_attack_attempt_rate: Option<f64>,
    pub(crate) mean_attack_success_rate: Option<f64>,
    pub(crate) mean_failed_action_rate: Option<f64>,
    pub(crate) mean_idle_fraction: Option<f64>,
    pub(crate) mean_util: Option<f64>,
    pub(crate) mean_action_histogram: [f64; ACTION_COUNT],
    pub(crate) mean_abs_td_error: Option<f64>,
    pub(crate) mean_age_correlated_competence: Option<f64>,
    pub(crate) foraging_p_fwd_food_component: f64,
    pub(crate) intelligence_mi_component: f64,
    pub(crate) intelligence_action_effectiveness_component: f64,
    pub(crate) intelligence_anti_idle_component: f64,
    pub(crate) intelligence_util_component: f64,
    pub(crate) competition_attack_success_component: f64,
    pub(crate) competition_attack_attempt_component: f64,
    pub(crate) foraging_pillar: f64,
    pub(crate) intelligence_pillar: f64,
    pub(crate) competition_pillar: f64,
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
