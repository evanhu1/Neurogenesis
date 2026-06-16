//! Pillar scoring — the niche-agnostic competence axes printed on the report
//! card. Reads a derived `Vec<IntervalMetrics>` (no direct dataset dependency)
//! so it can be run on any analysis output.
//!
//! Saturation constants below map a raw rate onto [0, 1]; they are deliberate,
//! tunable anchors (calibrate against the `--control` random-action baseline,
//! which should score ≈0 on every axis).

use crate::output::mean_option;
use crate::types::{IntervalMetrics, PillarScores};

/// MI(S;A) at which the intelligence MI component saturates to 1.0.
const MEAN_MI_SATURATION: f64 = 0.16;
/// Plant consumptions per action at which foraging saturates (≈ eat once per
/// five actions = excellent forager).
const FORAGE_SATURATION: f64 = 0.20;
/// Prey consumptions per action at which predation saturates (predation is
/// rarer than foraging, so the bar is lower).
const PREDATION_SATURATION: f64 = 0.05;
/// Within-life success-probability gain per tick of age at which the learning
/// pillar saturates. Negative slopes (forgetting) clamp to 0.
const LEARNING_SATURATION: f64 = 0.001;

/// Fraction of the timeseries (taken from the end) that feeds pillar
/// computation: the last 10% of the run.
const SCORING_WINDOW_FRACTION: f64 = 0.10;

fn window_len(total: usize) -> usize {
    ((total as f64 * SCORING_WINDOW_FRACTION).ceil() as usize)
        .max(1)
        .min(total)
}

pub fn compute_pillar_scores(timeseries: &[IntervalMetrics]) -> PillarScores {
    if timeseries.is_empty() {
        return PillarScores::default();
    }
    let window_len = window_len(timeseries.len());
    let start_idx = timeseries.len().saturating_sub(window_len);
    let slice = &timeseries[start_idx..];
    let window_start_tick = slice.first().map(|row| row.tick).unwrap_or(0);
    let window_end_tick = slice.last().map(|row| row.tick).unwrap_or(0);

    let mean_action_effectiveness = mean_option(slice.iter().map(|row| row.action_effectiveness));
    let mean_mi_sa = mean_option(slice.iter().map(|row| row.mi_sa));
    let mean_plant_consumption_rate =
        mean_option(slice.iter().map(|row| row.plant_consumption_rate));
    let mean_prey_consumption_rate = mean_option(slice.iter().map(|row| row.prey_consumption_rate));
    let mean_learning_slope = mean_option(slice.iter().map(|row| row.learning_slope));

    let intelligence_effectiveness_component =
        mean_action_effectiveness.map(clamp01).unwrap_or(0.0);
    let intelligence_mi_component = mean_mi_sa
        .map(|value| clamp01(value / MEAN_MI_SATURATION))
        .unwrap_or(0.0);

    let foraging_pillar = mean_plant_consumption_rate
        .map(|value| clamp01(value / FORAGE_SATURATION))
        .unwrap_or(0.0);
    let predation_pillar = mean_prey_consumption_rate
        .map(|value| clamp01(value / PREDATION_SATURATION))
        .unwrap_or(0.0);
    let intelligence_pillar = weighted_geometric_mean(&[
        (intelligence_effectiveness_component, 0.5),
        (intelligence_mi_component, 0.5),
    ]);
    let learning_pillar = mean_learning_slope
        .map(|value| clamp01(value / LEARNING_SATURATION))
        .unwrap_or(0.0);

    PillarScores {
        window_start_tick,
        window_end_tick,
        mean_action_effectiveness,
        mean_mi_sa,
        mean_plant_consumption_rate,
        mean_prey_consumption_rate,
        mean_learning_slope,
        intelligence_effectiveness_component,
        intelligence_mi_component,
        foraging_pillar,
        predation_pillar,
        intelligence_pillar,
        learning_pillar,
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
