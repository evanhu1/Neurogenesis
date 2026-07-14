//! Windowed competence metrics — the niche-agnostic behavioural axes reported
//! on the card. Reads a derived `&[IntervalMetrics]` (no dataset dependency) so
//! it can be run on any analysis output, batch or live.
//!
//! These are the **raw** windowed means of each signal (foraging/predation
//! consumption rates, action effectiveness, MI(S;A), within-life learning
//! slope). There is deliberately no [0,1] saturation/normalisation or composite
//! scoring layer — consumers report the raw values and apply their own
//! thresholds if they want them.

use crate::intervals::IntervalMetrics;
use serde::Serialize;

/// Fraction of the timeseries (taken from the end) that feeds the windowed
/// means: the last 10% of the run.
const SCORING_WINDOW_FRACTION: f64 = 0.10;

/// Per-axis behavioural readout over the scoring window. Each field is the raw
/// windowed mean of its signal — no interpretation. Every signal is chosen to
/// be hard to game: foraging/predation are real consumption per action,
/// action-effectiveness and MI(S;A) capture successful and sensing-conditioned
/// behaviour, and the learning slope captures within-life improvement (≈0 under
/// the random-action control).
#[derive(Debug, Clone, Serialize, Default)]
pub struct PillarScores {
    pub window_start_tick: u64,
    pub window_end_tick: u64,
    /// Number of runs contributing to each optional metric. A multi-seed mean
    /// is conditional on these counts; extinct/no-action seeds are never
    /// silently presented as if every requested seed contributed.
    pub coverage: PillarCoverage,
    /// Successful contingent actions / total actions.
    pub mean_action_effectiveness: Option<f64>,
    /// Miller-Madow MI between food-visibility context and selected action.
    pub mean_mi_sa: Option<f64>,
    /// Plant consumptions / total actions (foraging).
    pub mean_plant_consumption_rate: Option<f64>,
    /// Successful attack-energy transfers / total actions (predation).
    pub mean_prey_consumption_rate: Option<f64>,
    /// Action-time success-vs-age slope (observational learning diagnostic).
    pub mean_learning_slope: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct PillarCoverage {
    pub runs_total: usize,
    pub action_effectiveness: usize,
    pub mi_sa: usize,
    pub plant_consumption_rate: usize,
    pub prey_consumption_rate: usize,
    pub learning_slope: usize,
}

pub fn compute_pillar_scores(timeseries: &[IntervalMetrics]) -> PillarScores {
    if timeseries.is_empty() {
        return PillarScores::default();
    }

    // Reporting intervals need not all have the same duration: the last row is
    // partial whenever the horizon is not a multiple of `report_every`. Select
    // the tail by recorded ticks, not row count, or a one-tick partial row can
    // replace a full interval in the advertised "last 10%" score.
    let total_duration: u64 = timeseries
        .iter()
        .map(|row| row.tick.saturating_sub(row.start_tick))
        .sum();
    let target_duration = ((total_duration as f64 * SCORING_WINDOW_FRACTION).ceil() as u64)
        .max(1)
        .min(total_duration.max(1));
    let mut remaining = target_duration;
    let mut weighted_rows = Vec::new();
    for row in timeseries.iter().rev() {
        if remaining == 0 {
            break;
        }
        let duration = row.tick.saturating_sub(row.start_tick);
        let included = duration.min(remaining);
        if included > 0 {
            weighted_rows.push((row, included));
            remaining -= included;
        }
    }
    let window_start_tick = weighted_rows
        .last()
        .map(|(row, included)| row.tick.saturating_sub(*included))
        .unwrap_or(0);
    let window_end_tick = weighted_rows.first().map(|(row, _)| row.tick).unwrap_or(0);

    let weighted_mean =
        |value: fn(&IntervalMetrics) -> Option<f64>| {
            let (weighted_sum, weight) = weighted_rows.iter().fold(
                (0.0, 0_u64),
                |(sum, weight), (row, duration)| match value(row) {
                    Some(value) => (sum + value * *duration as f64, weight + *duration),
                    None => (sum, weight),
                },
            );
            (weight > 0).then(|| weighted_sum / weight as f64)
        };

    let mean_action_effectiveness = weighted_mean(|row| row.action_effectiveness);
    let mean_mi_sa = weighted_mean(|row| row.mi_sa);
    let mean_plant_consumption_rate = weighted_mean(|row| row.plant_consumption_rate);
    let mean_prey_consumption_rate = weighted_mean(|row| row.prey_consumption_rate);
    let mean_learning_slope = weighted_mean(|row| row.learning_slope);

    PillarScores {
        window_start_tick,
        window_end_tick,
        coverage: PillarCoverage {
            runs_total: 1,
            action_effectiveness: usize::from(mean_action_effectiveness.is_some()),
            mi_sa: usize::from(mean_mi_sa.is_some()),
            plant_consumption_rate: usize::from(mean_plant_consumption_rate.is_some()),
            prey_consumption_rate: usize::from(mean_prey_consumption_rate.is_some()),
            learning_slope: usize::from(mean_learning_slope.is_some()),
        },
        mean_action_effectiveness,
        mean_mi_sa,
        mean_plant_consumption_rate,
        mean_prey_consumption_rate,
        mean_learning_slope,
    }
}
