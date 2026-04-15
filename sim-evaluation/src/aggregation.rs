use crate::{
    metrics::{self, IntervalMetrics},
    output::{mean_f64, mean_histogram, mean_option, mean_option_u64, mean_round_u32, mean_round_u64},
    types::{PillarScores, ReproductionAnalytics, SeedEvaluationSummary},
};
use sim_types::OrganismState;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const MEAN_MI_SATURATION: f64 = 0.16;
const ADULT_MI_SATURATION: f64 = 0.20;
const JUVENILE_MI_SATURATION: f64 = 0.16;
const FORAGING_RATE_SATURATION: f64 = 0.025;
const UTILIZATION_SATURATION: f64 = 0.60;
/// Lineages reproducing by ~150 ticks of age saturate the viability
/// reproduction component; slower turnover scales down as `target / T`.
const GENERATION_TIME_TARGET: f64 = 150.0;

pub(crate) fn state_hash(organisms: &[OrganismState]) -> String {
    let population_count = organisms.len() as u64;
    let sum_ids = organisms
        .iter()
        .map(|organism| organism.id.0)
        .fold(0_u64, |acc, value| acc.wrapping_add(value));
    let total_energy = organisms
        .iter()
        .map(|organism| organism.energy as f64)
        .sum::<f64>();

    let mut hasher = DefaultHasher::new();
    population_count.hash(&mut hasher);
    sum_ids.hash(&mut hasher);
    total_energy.to_bits().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub(crate) fn compute_pillar_scores(timeseries: &[IntervalMetrics]) -> PillarScores {
    let window_len = (timeseries.len() / 5).max(1);
    let start_idx = timeseries.len().saturating_sub(window_len);
    let window = &timeseries[start_idx..];
    let window_start_tick = window.first().map(|row| row.tick).unwrap_or(0);
    let window_end_tick = window.last().map(|row| row.tick).unwrap_or(0);

    let mean_life_mean = mean_option(window.iter().map(|row| row.life_mean));
    let mean_p_fwd_food = mean_option(window.iter().map(|row| row.p_fwd_food));
    let mean_mi_sa = mean_option(window.iter().map(|row| row.mi_sa));
    let mean_mi_sa_juvenile = mean_option(window.iter().map(|row| row.mi_sa_juvenile));
    let mean_mi_sa_adult = mean_option(window.iter().map(|row| row.mi_sa_adult));
    let mean_predation_rate = mean_option(window.iter().map(|row| row.predation_rate));
    let mean_foraging_rate = mean_option(window.iter().map(|row| row.foraging_rate));
    let mean_attack_attempt_rate = mean_option(window.iter().map(|row| row.attack_attempt_rate));
    let mean_attack_success_rate = mean_option(window.iter().map(|row| row.attack_success_rate));
    let mean_failed_action_rate = mean_option(window.iter().map(|row| row.failed_action_rate));
    let mean_idle_fraction = mean_option(window.iter().map(|row| row.idle_fraction));
    let mean_generation_time = mean_option(window.iter().map(|row| row.generation_time));
    let mean_lineage_diversity = mean_option(window.iter().map(|row| row.lineage_diversity));
    let mean_util = mean_option(window.iter().map(|row| row.util));
    let mean_action_histogram = mean_histogram(window.iter().map(|row| row.action_histogram));

    let p_baseline = metrics::action_baseline_probability();
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

    // Viability: life expectancy and generational turnover.
    // reproduction_component is now derived from generation_time (shorter =
    // better) instead of from the retired reproduction_efficiency metric.
    let viability_pillar = weighted_geometric_mean(&[
        (life_component, 0.55),
        (reproduction_component, 0.45),
    ]);
    let foraging_pillar = weighted_geometric_mean(&[
        (p_fwd_food_component, 0.65),
        (foraging_rate_component, 0.35),
    ]);
    // Intelligence: niche-agnostic behavioural competence — whether the
    // policy is conditional on state, avoids wasting actions, keeps inter
    // neurons active, and isn't stuck idling. The H(action)-derived entropy
    // component was retired for being a pooled-over-deceased artifact that
    // didn't distinguish intelligent policies from lucky ones. Remaining
    // weights renormalised from (0.45, 0.25, 0.10, 0.10, 0.10) -> 0.90 total
    // to (0.50, 0.28, 0.11, 0.11).
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
    // Adaptation: juvenile MI and lineage diversity. The reward-reversal
    // component was retired along with the reversal orchestration machinery.
    let adaptation_pillar = weighted_geometric_mean(&[
        (juvenile_mi_component, 0.60),
        (diversity_component, 0.40),
    ]);

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

pub(crate) fn average_pillar_scores(seed_summaries: &[SeedEvaluationSummary]) -> PillarScores {
    let first = seed_summaries
        .first()
        .expect("multi-seed evaluation requires at least one seed");
    let pillars = |f: fn(&SeedEvaluationSummary) -> &PillarScores| {
        seed_summaries.iter().map(f)
    };
    PillarScores {
        window_start_tick: first.pillars.window_start_tick,
        window_end_tick: first.pillars.window_end_tick,
        mean_life_mean: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_life_mean)),
        mean_p_fwd_food: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_p_fwd_food)),
        mean_mi_sa: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_mi_sa)),
        mean_mi_sa_juvenile: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_mi_sa_juvenile)),
        mean_mi_sa_adult: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_mi_sa_adult)),
        mean_predation_rate: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_predation_rate)),
        mean_foraging_rate: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_foraging_rate)),
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
        mean_generation_time: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_generation_time)),
        mean_lineage_diversity: mean_option(
            pillars(|s| &s.pillars).map(|p| p.mean_lineage_diversity),
        ),
        mean_util: mean_option(pillars(|s| &s.pillars).map(|p| p.mean_util)),
        mean_action_histogram: mean_histogram(
            pillars(|s| &s.pillars).map(|p| p.mean_action_histogram),
        ),
        viability_life_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.viability_life_component),
        ),
        viability_reproduction_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.viability_reproduction_component),
        ),
        foraging_p_fwd_food_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.foraging_p_fwd_food_component),
        ),
        foraging_rate_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.foraging_rate_component),
        ),
        intelligence_adult_mi_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.intelligence_adult_mi_component),
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
        competition_predation_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.competition_predation_component),
        ),
        competition_attack_success_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.competition_attack_success_component),
        ),
        competition_attack_attempt_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.competition_attack_attempt_component),
        ),
        adaptation_juvenile_mi_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.adaptation_juvenile_mi_component),
        ),
        adaptation_diversity_component: mean_f64(
            pillars(|s| &s.pillars).map(|p| p.adaptation_diversity_component),
        ),
        viability_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.viability_pillar)),
        foraging_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.foraging_pillar)),
        intelligence_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.intelligence_pillar)),
        competition_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.competition_pillar)),
        adaptation_pillar: mean_f64(pillars(|s| &s.pillars).map(|p| p.adaptation_pillar)),
    }
}

pub(crate) fn average_reproduction_analytics(
    seed_summaries: &[SeedEvaluationSummary],
) -> ReproductionAnalytics {
    ReproductionAnalytics {
        births: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.births),
        ),
        successful_births: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.successful_births),
        ),
        blocked_births: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.blocked_births),
        ),
        parent_died_during_reproduction: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.parent_died_during_reproduction),
        ),
        survived_to_30: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.survived_to_30),
        ),
        survived_to_maturity: mean_round_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.survived_to_maturity),
        ),
        mean_parent_energy_after_successful_birth: mean_option(seed_summaries.iter().map(
            |summary| {
                summary
                    .experiment_readouts
                    .mean_parent_energy_after_successful_birth
            },
        )),
        mean_age_at_first_successful_reproduction: mean_option(seed_summaries.iter().map(
            |summary| {
                summary
                    .experiment_readouts
                    .mean_age_at_first_successful_reproduction
            },
        )),
        mean_successful_birth_interval: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.experiment_readouts.mean_successful_birth_interval),
        ),
    }
}

pub(crate) fn average_timeseries(seed_summaries: &[SeedEvaluationSummary]) -> Vec<IntervalMetrics> {
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
            life_mean: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].life_mean),
            ),
            predation_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].predation_rate),
            ),
            foraging_rate: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].foraging_rate),
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
            lineage_diversity: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].lineage_diversity),
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
            mi_sa_juvenile: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa_juvenile),
            ),
            mi_sa_adult: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].mi_sa_adult),
            ),
            idle_fraction: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].idle_fraction),
            ),
            generation_time: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].generation_time),
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_failed_action_rate_improves_intelligence_pillar() {
        let low_failure = compute_pillar_scores(&[
            metrics_row(20, 0.05),
            metrics_row(40, 0.05),
            metrics_row(60, 0.05),
            metrics_row(80, 0.05),
            metrics_row(100, 0.05),
        ]);
        let high_failure = compute_pillar_scores(&[
            metrics_row(20, 0.65),
            metrics_row(40, 0.65),
            metrics_row(60, 0.65),
            metrics_row(80, 0.65),
            metrics_row(100, 0.65),
        ]);

        assert!(
            low_failure.intelligence_action_effectiveness_component
                > high_failure.intelligence_action_effectiveness_component
        );
        assert!(low_failure.intelligence_pillar > high_failure.intelligence_pillar);
    }

    fn metrics_row(tick: u64, failed_action_rate: f64) -> IntervalMetrics {
        IntervalMetrics {
            tick,
            pop: 100,
            births: 10,
            deaths: 5,
            food: 200,
            max_generation: Some(8),
            life_mean: Some(180.0),
            predation_rate: Some(0.003),
            foraging_rate: Some(0.02),
            attack_attempt_rate: Some(0.015),
            attack_success_rate: Some(0.4),
            failed_action_rate: Some(failed_action_rate),
            ate_pct: Some(60.0),
            cons_mean: Some(3.0),
            brain_size: Some(24.0),
            brain_size_stddev: Some(2.0),
            brain_size_p10: Some(20.0),
            brain_size_p50: Some(24.0),
            brain_size_p90: Some(28.0),
            lineage_diversity: Some(1.0),
            p_fwd_food: Some(0.45),
            mi_sa: Some(0.08),
            mi_sa_juvenile: Some(0.05),
            mi_sa_adult: Some(0.08),
            idle_fraction: Some(0.15),
            generation_time: Some(120.0),
            util: Some(0.3),
            action_histogram: [0.05, 0.1, 0.1, 0.4, 0.2, 0.1, 0.05],
        }
    }
}
