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
use crate::stats::mean_option;
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
    /// Successful contingent actions / total actions.
    pub mean_action_effectiveness: Option<f64>,
    /// Miller-Madow MI between food-visibility context and selected action.
    pub mean_mi_sa: Option<f64>,
    /// Plant consumptions / total actions (foraging).
    pub mean_plant_consumption_rate: Option<f64>,
    /// Prey/corpse consumptions / total actions (predation).
    pub mean_prey_consumption_rate: Option<f64>,
    /// Mean within-life success-vs-age slope over descendants (learning).
    pub mean_learning_slope: Option<f64>,
}

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

    PillarScores {
        window_start_tick,
        window_end_tick,
        mean_action_effectiveness: mean_option(slice.iter().map(|row| row.action_effectiveness)),
        mean_mi_sa: mean_option(slice.iter().map(|row| row.mi_sa)),
        mean_plant_consumption_rate: mean_option(
            slice.iter().map(|row| row.plant_consumption_rate),
        ),
        mean_prey_consumption_rate: mean_option(slice.iter().map(|row| row.prey_consumption_rate)),
        mean_learning_slope: mean_option(slice.iter().map(|row| row.learning_slope)),
    }
}
