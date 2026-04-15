//! Pillar scoring — the five niche-agnostic competence axes printed on the
//! report card. Reads a derived `Vec<IntervalMetrics>` (no direct dataset
//! dependency) so it can be run on any analysis output.

use crate::output::{mean_histogram, mean_option};
use crate::types::{IntervalMetrics, PillarScores};

const MEAN_MI_SATURATION: f64 = 0.16;
const ADULT_MI_SATURATION: f64 = 0.20;
const JUVENILE_MI_SATURATION: f64 = 0.16;
const FORAGING_RATE_SATURATION: f64 = 0.025;
const UTILIZATION_SATURATION: f64 = 0.60;
/// Lineages reproducing by ~150 ticks of age saturate the viability
/// reproduction component; slower turnover scales down as `target / T`.
const GENERATION_TIME_TARGET: f64 = 150.0;

/// Which slice of the timeseries feeds pillar computation. Defaults to the
/// last 20% of the run. `Last { fraction: 1.0 }` uses the whole timeseries.
#[derive(Debug, Clone, Copy)]
pub enum ScoringWindow {
    LastFraction(f64),
}

impl Default for ScoringWindow {
    fn default() -> Self {
        Self::LastFraction(0.20)
    }
}

impl ScoringWindow {
    fn window_len(self, total: usize) -> usize {
        match self {
            ScoringWindow::LastFraction(fraction) => {
                let fraction = fraction.clamp(0.0, 1.0);
                ((total as f64 * fraction).ceil() as usize)
                    .max(1)
                    .min(total)
            }
        }
    }
}

pub fn compute_pillar_scores(
    timeseries: &[IntervalMetrics],
    window: ScoringWindow,
) -> PillarScores {
    if timeseries.is_empty() {
        return PillarScores::default();
    }
    let window_len = window.window_len(timeseries.len());
    let start_idx = timeseries.len().saturating_sub(window_len);
    let slice = &timeseries[start_idx..];
    let window_start_tick = slice.first().map(|row| row.tick).unwrap_or(0);
    let window_end_tick = slice.last().map(|row| row.tick).unwrap_or(0);

    let mean_life_mean = mean_option(slice.iter().map(|row| row.life_mean));
    let mean_p_fwd_food = mean_option(slice.iter().map(|row| row.p_fwd_food));
    let mean_mi_sa = mean_option(slice.iter().map(|row| row.mi_sa));
    let mean_mi_sa_juvenile = mean_option(slice.iter().map(|row| row.mi_sa_juvenile));
    let mean_mi_sa_adult = mean_option(slice.iter().map(|row| row.mi_sa_adult));
    let mean_predation_rate = mean_option(slice.iter().map(|row| row.predation_rate));
    let mean_foraging_rate = mean_option(slice.iter().map(|row| row.foraging_rate));
    let mean_attack_attempt_rate = mean_option(slice.iter().map(|row| row.attack_attempt_rate));
    let mean_attack_success_rate = mean_option(slice.iter().map(|row| row.attack_success_rate));
    let mean_failed_action_rate = mean_option(slice.iter().map(|row| row.failed_action_rate));
    let mean_idle_fraction = mean_option(slice.iter().map(|row| row.idle_fraction));
    let mean_generation_time = mean_option(slice.iter().map(|row| row.generation_time));
    let mean_lineage_diversity = mean_option(slice.iter().map(|row| row.lineage_diversity));
    let mean_util = mean_option(slice.iter().map(|row| row.util));
    let mean_action_histogram = mean_histogram(slice.iter().map(|row| row.action_histogram));

    let p_baseline = super::intervals::action_baseline_probability();
    let p_fwd_food_component = mean_p_fwd_food
        .map(|value| clamp01((value - p_baseline) / (0.55 - p_baseline).max(f64::EPSILON)))
        .unwrap_or(0.0);
    let mean_mi_component = mean_mi_sa
        .map(|value| clamp01(value / MEAN_MI_SATURATION))
        .unwrap_or(0.0);
    let adult_mi_component = mean_mi_sa_adult
        .map(|value| clamp01(value / ADULT_MI_SATURATION))
        .unwrap_or(mean_mi_component);
    let action_effectiveness_component = mean_failed_action_rate
        .map(|value| clamp01(1.0 - value))
        .unwrap_or(0.0);
    let juvenile_mi_component = mean_mi_sa_juvenile
        .map(|value| clamp01(value / JUVENILE_MI_SATURATION))
        .unwrap_or(0.0);
    let predation_component = mean_predation_rate
        .map(|value| clamp01(value / 0.002))
        .unwrap_or(0.0);
    let life_component = mean_life_mean
        .map(|value| clamp01(value / 250.0))
        .unwrap_or(0.0);
    let reproduction_component = mean_generation_time
        .map(|value| clamp01(GENERATION_TIME_TARGET / value.max(f64::EPSILON)))
        .unwrap_or(0.0);
    let foraging_rate_component = mean_foraging_rate
        .map(|value| clamp01(value / FORAGING_RATE_SATURATION))
        .unwrap_or(0.0);
    let anti_idle_component = mean_idle_fraction
        .map(|value| clamp01(1.0 - value / 0.60))
        .unwrap_or(0.0);
    let util_component = mean_util
        .map(|value| clamp01(value / UTILIZATION_SATURATION))
        .unwrap_or(0.0);
    let attack_success_component = mean_attack_success_rate
        .map(|value| clamp01(value / 0.35))
        .unwrap_or(0.0);
    let attack_attempt_component = mean_attack_attempt_rate
        .map(|value| clamp01(value / 0.01))
        .unwrap_or(0.0);
    let diversity_component = mean_lineage_diversity
        .map(|value| clamp01(value / 2.0))
        .unwrap_or(0.0);

    let viability_pillar =
        weighted_geometric_mean(&[(life_component, 0.55), (reproduction_component, 0.45)]);
    let foraging_pillar = weighted_geometric_mean(&[
        (p_fwd_food_component, 0.65),
        (foraging_rate_component, 0.35),
    ]);
    let intelligence_pillar = weighted_geometric_mean(&[
        (action_effectiveness_component, 0.50),
        (adult_mi_component, 0.28),
        (anti_idle_component, 0.11),
        (util_component, 0.11),
    ]);
    let competition_pillar = weighted_geometric_mean(&[
        (predation_component, 0.50),
        (attack_success_component, 0.30),
        (attack_attempt_component, 0.20),
    ]);
    let adaptation_pillar =
        weighted_geometric_mean(&[(juvenile_mi_component, 0.60), (diversity_component, 0.40)]);

    PillarScores {
        window_start_tick,
        window_end_tick,
        mean_life_mean,
        mean_p_fwd_food,
        mean_mi_sa,
        mean_mi_sa_juvenile,
        mean_mi_sa_adult,
        mean_predation_rate,
        mean_foraging_rate,
        mean_attack_attempt_rate,
        mean_attack_success_rate,
        mean_failed_action_rate,
        mean_idle_fraction,
        mean_generation_time,
        mean_lineage_diversity,
        mean_util,
        mean_action_histogram,
        viability_life_component: life_component,
        viability_reproduction_component: reproduction_component,
        foraging_p_fwd_food_component: p_fwd_food_component,
        foraging_rate_component,
        intelligence_adult_mi_component: adult_mi_component,
        intelligence_action_effectiveness_component: action_effectiveness_component,
        intelligence_anti_idle_component: anti_idle_component,
        intelligence_util_component: util_component,
        competition_predation_component: predation_component,
        competition_attack_success_component: attack_success_component,
        competition_attack_attempt_component: attack_attempt_component,
        adaptation_juvenile_mi_component: juvenile_mi_component,
        adaptation_diversity_component: diversity_component,
        viability_pillar,
        foraging_pillar,
        intelligence_pillar,
        competition_pillar,
        adaptation_pillar,
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn weighted_geometric_mean(components: &[(f64, f64)]) -> f64 {
    let mut total_weight = 0.0;
    let mut weighted_log_sum = 0.0;
    for (value, weight) in components {
        if !value.is_finite() || !weight.is_finite() || *weight <= 0.0 {
            continue;
        }
        total_weight += *weight;
        let softened = 0.05 + 0.95 * clamp01(*value);
        weighted_log_sum += *weight * softened.ln();
    }
    if total_weight <= 0.0 {
        0.0
    } else {
        (weighted_log_sum / total_weight).exp()
    }
}
