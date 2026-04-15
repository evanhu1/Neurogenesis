//! Cross-seed averaging of analysis outputs. Multi-seed runs fold per-seed
//! `IntervalMetrics` / `PillarScores` / `DemographicAnalytics` into a single
//! "mean across seeds" view for reports.

use crate::output::{
    mean_f64, mean_histogram, mean_option, mean_option_u64, mean_round_u32, mean_round_u64,
};
use crate::types::{DemographicAnalytics, IntervalMetrics, PillarScores, SeedEvaluationSummary};

pub fn average_pillar_scores(seed_summaries: &[SeedEvaluationSummary]) -> PillarScores {
    let Some(first) = seed_summaries.first() else {
        return PillarScores::default();
    };
    let pillars = |f: fn(&SeedEvaluationSummary) -> &PillarScores| seed_summaries.iter().map(f);
    PillarScores {
        window_start_tick: first.pillars.window_start_tick,
        window_end_tick: first.pillars.window_end_tick,
        mean_p_fwd_food: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_p_fwd_food)),
        mean_mi_sa: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_mi_sa)),
        mean_attack_attempt_rate: mean_option(
            pillars(|s| &s.pillars).map(|p| p.mean_attack_attempt_rate),
        ),
        mean_attack_success_rate: mean_option(
            pillars(|s| &s.pillars).map(|p| p.mean_attack_success_rate),
        ),
        mean_failed_action_rate: mean_option(
            pillars(|s| &s.pillars).map(|p| p.mean_failed_action_rate),
        ),
        mean_idle_fraction: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_idle_fraction)),
        mean_util: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_util)),
        mean_action_histogram: mean_histogram(
            pillars(|s| &s.pillars).map(|p| p.mean_action_histogram),
        ),
        foraging_p_fwd_food_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.foraging_p_fwd_food_component),
        ),
        intelligence_mi_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.intelligence_mi_component),
        ),
        intelligence_action_effectiveness_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.intelligence_action_effectiveness_component),
        ),
        intelligence_anti_idle_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.intelligence_anti_idle_component),
        ),
        intelligence_util_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.intelligence_util_component),
        ),
        competition_attack_success_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.competition_attack_success_component),
        ),
        competition_attack_attempt_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.competition_attack_attempt_component),
        ),
        foraging_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.foraging_pillar)),
        intelligence_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.intelligence_pillar)),
        competition_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.competition_pillar)),
    }
}

pub fn average_demographic_analytics(
    seed_summaries: &[SeedEvaluationSummary],
) -> DemographicAnalytics {
    DemographicAnalytics {
        births: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.births),
        ),
        successful_births: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.successful_births),
        ),
        blocked_births: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.blocked_births),
        ),
        parent_died_during_reproduction: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.parent_died_during_reproduction),
        ),
        survived_to_30: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.survived_to_30),
        ),
        survived_to_maturity: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.survived_to_maturity),
        ),
        mean_parent_energy_after_successful_birth: mean_option(seed_summaries.iter().map(
            |summary| {
                summary
                    .demographics
                    .mean_parent_energy_after_successful_birth
            },
        )),
        mean_age_at_first_successful_reproduction: mean_option(seed_summaries.iter().map(
            |summary| {
                summary
                    .demographics
                    .mean_age_at_first_successful_reproduction
            },
        )),
        mean_successful_birth_interval: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.demographics.mean_successful_birth_interval),
        ),
    }
}

pub fn average_timeseries(seed_summaries: &[SeedEvaluationSummary]) -> Vec<IntervalMetrics> {
    let Some(first_summary) = seed_summaries.first() else {
        return Vec::new();
    };
    let row_count = first_summary.timeseries.len();
    let mut averaged = Vec::with_capacity(row_count);

    for row_idx in 0..row_count {
        let tick = first_summary.timeseries[row_idx].tick;
        debug_assert!(seed_summaries.iter().all(|summary| {
            summary.timeseries.len() == row_count && summary.timeseries[row_idx].tick == tick
        }));

        averaged.push(IntervalMetrics {
            tick,
            pop: mean_round_u32(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].pop),
            ),
            births: mean_round_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].births),
            ),
            deaths: mean_round_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].deaths),
            ),
            food: mean_round_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].food),
            ),
            max_generation: mean_option_u64(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].max_generation),
            ),
            attack_attempt_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].attack_attempt_rate),
            ),
            attack_success_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].attack_success_rate),
            ),
            failed_action_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].failed_action_rate),
            ),
            ate_pct: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].ate_pct),
            ),
            cons_mean: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].cons_mean),
            ),
            brain_size: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size),
            ),
            brain_size_stddev: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_stddev),
            ),
            brain_size_p10: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_p10),
            ),
            brain_size_p50: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_p50),
            ),
            brain_size_p90: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].brain_size_p90),
            ),
            p_fwd_food: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].p_fwd_food),
            ),
            mi_sa: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa),
            ),
            idle_fraction: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].idle_fraction),
            ),
            util: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].util),
            ),
            action_histogram: mean_histogram(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].action_histogram),
            ),
        });
    }

    averaged
}
