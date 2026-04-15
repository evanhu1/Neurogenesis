use crate::{ledger::N_ACTIONS, metrics::IntervalMetrics, report::ComparisonMetricRow};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(crate) struct HarnessRunOptions {
    pub(crate) seeds: Vec<u64>,
    pub(crate) ticks: u64,
    pub(crate) report_every: u64,
    pub(crate) min_lifetime: u64,
    pub(crate) out_dir: PathBuf,
    pub(crate) title: Option<String>,
    pub(crate) control: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SeedRunOptions {
    pub(crate) seed: u64,
    pub(crate) ticks: u64,
    pub(crate) report_every: u64,
    pub(crate) min_lifetime: u64,
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
    pub(crate) experiment_readouts: ReproductionAnalytics,
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
    pub(crate) experiment_readouts: ReproductionAnalytics,
    pub(crate) seed_summaries: Vec<SeedRunSummary>,
    pub(crate) timeseries: Vec<IntervalMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SeedRunSummary {
    pub(crate) seed: u64,
    pub(crate) out_dir: PathBuf,
    pub(crate) total_time_seconds: f64,
    pub(crate) pillars: PillarScores,
    pub(crate) experiment_readouts: ReproductionAnalytics,
    pub(crate) state_hash: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct ReproductionAnalytics {
    pub(crate) births: u64,
    pub(crate) successful_births: u64,
    pub(crate) blocked_births: u64,
    pub(crate) parent_died_during_reproduction: u64,
    pub(crate) survived_to_30: u64,
    pub(crate) survived_to_maturity: u64,
    pub(crate) mean_parent_energy_after_successful_birth: Option<f64>,
    pub(crate) mean_age_at_first_successful_reproduction: Option<f64>,
    pub(crate) mean_successful_birth_interval: Option<f64>,
}

/// Per-pillar evaluation readout. There is deliberately no single aggregate
/// score here — see PERF_LOG.md / EVALUATION_GUIDANCE.md for why. Each pillar
/// is reported in its own right, and the intelligence pillar is composed of
/// niche-agnostic behavioural measures so predators, foragers, and
/// r-strategists can all be compared on the same axis.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PillarScores {
    pub(crate) window_start_tick: u64,
    pub(crate) window_end_tick: u64,
    pub(crate) mean_life_mean: Option<f64>,
    pub(crate) mean_p_fwd_food: Option<f64>,
    pub(crate) mean_mi_sa: Option<f64>,
    pub(crate) mean_mi_sa_juvenile: Option<f64>,
    pub(crate) mean_mi_sa_adult: Option<f64>,
    pub(crate) mean_predation_rate: Option<f64>,
    pub(crate) mean_foraging_rate: Option<f64>,
    pub(crate) mean_attack_attempt_rate: Option<f64>,
    pub(crate) mean_attack_success_rate: Option<f64>,
    pub(crate) mean_failed_action_rate: Option<f64>,
    pub(crate) mean_idle_fraction: Option<f64>,
    /// Average parent age across every successful reproduction event in the
    /// scoring window. Shorter = faster generational turnover.
    pub(crate) mean_generation_time: Option<f64>,
    pub(crate) mean_lineage_diversity: Option<f64>,
    pub(crate) mean_util: Option<f64>,
    pub(crate) mean_action_histogram: [f64; N_ACTIONS],
    pub(crate) viability_life_component: f64,
    pub(crate) viability_reproduction_component: f64,
    pub(crate) foraging_p_fwd_food_component: f64,
    pub(crate) foraging_rate_component: f64,
    pub(crate) intelligence_adult_mi_component: f64,
    pub(crate) intelligence_action_effectiveness_component: f64,
    pub(crate) intelligence_anti_idle_component: f64,
    pub(crate) intelligence_util_component: f64,
    pub(crate) competition_predation_component: f64,
    pub(crate) competition_attack_success_component: f64,
    pub(crate) competition_attack_attempt_component: f64,
    pub(crate) adaptation_juvenile_mi_component: f64,
    pub(crate) adaptation_diversity_component: f64,
    pub(crate) viability_pillar: f64,
    pub(crate) foraging_pillar: f64,
    pub(crate) intelligence_pillar: f64,
    pub(crate) competition_pillar: f64,
    pub(crate) adaptation_pillar: f64,
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
