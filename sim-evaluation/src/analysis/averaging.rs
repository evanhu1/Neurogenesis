//! Cross-seed averaging of analysis outputs. Multi-seed runs fold per-seed
//! `IntervalMetrics` / `PillarScores` into a single "mean across seeds" view
//! for reports.

use crate::output::{mean_option, mean_round_u32};
use crate::types::{IntervalMetrics, PillarCoverage, PillarScores, SeedEvaluationSummary};

pub fn average_pillar_scores(seed_summaries: &[SeedEvaluationSummary]) -> PillarScores {
    let Some(first) = seed_summaries.first() else {
        return PillarScores::default();
    };
    let pillars = || seed_summaries.iter().map(|s| &s.pillars);
    PillarScores {
        window_start_tick: first.pillars.window_start_tick,
        window_end_tick: first.pillars.window_end_tick,
        coverage: PillarCoverage {
            runs_total: seed_summaries.len(),
            action_effectiveness: pillars()
                .filter(|p| p.mean_action_effectiveness.is_some())
                .count(),
            mi_sa: pillars().filter(|p| p.mean_mi_sa.is_some()).count(),
            plant_consumption_rate: pillars()
                .filter(|p| p.mean_plant_consumption_rate.is_some())
                .count(),
            prey_consumption_rate: pillars()
                .filter(|p| p.mean_prey_consumption_rate.is_some())
                .count(),
            learning_slope: pillars()
                .filter(|p| p.mean_learning_slope.is_some())
                .count(),
        },
        mean_action_effectiveness: mean_option(pillars().map(|p| p.mean_action_effectiveness)),
        mean_mi_sa: mean_option(pillars().map(|p| p.mean_mi_sa)),
        mean_plant_consumption_rate: mean_option(pillars().map(|p| p.mean_plant_consumption_rate)),
        mean_prey_consumption_rate: mean_option(pillars().map(|p| p.mean_prey_consumption_rate)),
        mean_learning_slope: mean_option(pillars().map(|p| p.mean_learning_slope)),
    }
}

pub fn average_timeseries(seed_summaries: &[SeedEvaluationSummary]) -> Vec<IntervalMetrics> {
    let Some(first_summary) = seed_summaries.first() else {
        return Vec::new();
    };
    // Per-seed timeseries can disagree in length or tick alignment (e.g. a
    // seed interrupted or re-run with different tick settings). Average only
    // the aligned prefix — rows present in every seed with matching ticks —
    // instead of indexing unconditionally and panicking.
    let min_len = seed_summaries
        .iter()
        .map(|summary| summary.timeseries.len())
        .min()
        .unwrap_or(0);
    let row_count = (0..min_len)
        .take_while(|&row_idx| {
            let start_tick = first_summary.timeseries[row_idx].start_tick;
            let tick = first_summary.timeseries[row_idx].tick;
            seed_summaries.iter().all(|summary| {
                summary.timeseries[row_idx].start_tick == start_tick
                    && summary.timeseries[row_idx].tick == tick
            })
        })
        .count();
    if row_count < first_summary.timeseries.len() {
        eprintln!(
            "warning: seed timeseries are misaligned; averaging only the first {row_count} aligned rows"
        );
    }
    let mut averaged = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        let start_tick = first_summary.timeseries[row_idx].start_tick;
        let tick = first_summary.timeseries[row_idx].tick;
        averaged.push(IntervalMetrics {
            start_tick,
            tick,
            pop: mean_round_u32(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].pop),
            ),
            action_effectiveness: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].action_effectiveness),
            ),
            plant_consumption_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].plant_consumption_rate),
            ),
            prey_consumption_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].prey_consumption_rate),
            ),
            mi_sa: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa),
            ),
            learning_slope: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].learning_slope),
            ),
        });
    }

    averaged
}
