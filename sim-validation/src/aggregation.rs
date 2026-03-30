use crate::{
    metrics::{self, IntervalMetrics},
    output::{
        mean_f64, mean_histogram, mean_option, mean_option_u64, mean_round_u32, mean_round_u64,
    },
    types::{AggregateScore, SeedValidationSummary},
};
use sim_types::OrganismState;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

pub(crate) fn compute_aggregate_score(
    timeseries: &[IntervalMetrics],
    reward_reversal_tick: Option<u64>,
) -> AggregateScore {
    let window_len = (timeseries.len() / 5).max(1);
    let start_idx = timeseries.len().saturating_sub(window_len);
    let window = &timeseries[start_idx..];
    let window_start_tick = window.first().map(|row| row.tick).unwrap_or(0);
    let window_end_tick = window.last().map(|row| row.tick).unwrap_or(0);

    let mean_p_fwd_food = mean_option(window.iter().map(|row| row.p_fwd_food));
    let mean_mi_sa = mean_option(window.iter().map(|row| row.mi_sa));
    let mean_mi_sa_juvenile = mean_option(window.iter().map(|row| row.mi_sa_juvenile));
    let mean_mi_sa_adult = mean_option(window.iter().map(|row| row.mi_sa_adult));
    let mean_h_action = mean_option(window.iter().map(|row| row.h_action));
    let mean_predation_rate = mean_option(window.iter().map(|row| row.predation_rate));
    let mean_foraging_rate = mean_option(window.iter().map(|row| row.foraging_rate));
    let mean_attack_attempt_rate = mean_option(window.iter().map(|row| row.attack_attempt_rate));
    let mean_attack_success_rate = mean_option(window.iter().map(|row| row.attack_success_rate));
    let mean_idle_fraction = mean_option(window.iter().map(|row| row.idle_fraction));
    let mean_reproduction_efficiency =
        mean_option(window.iter().map(|row| row.reproduction_efficiency));
    let mean_lineage_diversity = mean_option(window.iter().map(|row| row.lineage_diversity));
    let mean_damage_avoidance = mean_option(window.iter().map(|row| row.damage_avoidance));
    let mean_reward_reversal_shift =
        mean_option(window.iter().map(|row| row.reward_reversal_shift));
    let mean_action_histogram = mean_histogram(window.iter().map(|row| row.action_histogram));
    let reward_reversal_adaptation_ticks =
        reward_reversal_adaptation_ticks(timeseries, reward_reversal_tick);

    let p_baseline = metrics::action_baseline_probability();
    let h_baseline = metrics::action_baseline_entropy();
    let strong_foraging_reference = 0.55;
    let competitive_predation_reference = 0.002;
    let p_component = mean_p_fwd_food
        .map(|value| {
            clamp01(
                (value - p_baseline) / (strong_foraging_reference - p_baseline).max(f64::EPSILON),
            )
        })
        .unwrap_or(0.0);
    let mi_component = mean_mi_sa.map(|value| clamp01(value / 0.10)).unwrap_or(0.0);
    let entropy_component = mean_h_action
        .map(|value| entropy_component_score(value, h_baseline))
        .unwrap_or(0.0);
    let predation_component = mean_predation_rate
        .map(|value| clamp01(value / competitive_predation_reference))
        .unwrap_or(0.0);

    let score = 100.0
        * (0.40 * p_component
            + 0.25 * mi_component
            + 0.10 * entropy_component
            + 0.25 * predation_component);

    AggregateScore {
        score,
        score_median: score,
        score_stddev: 0.0,
        score_min: score,
        score_max: score,
        window_start_tick,
        window_end_tick,
        mean_p_fwd_food,
        mean_mi_sa,
        mean_mi_sa_juvenile,
        mean_mi_sa_adult,
        mean_h_action,
        mean_predation_rate,
        mean_foraging_rate,
        mean_attack_attempt_rate,
        mean_attack_success_rate,
        mean_idle_fraction,
        mean_reproduction_efficiency,
        mean_lineage_diversity,
        mean_damage_avoidance,
        mean_reward_reversal_shift,
        mean_action_histogram,
        reward_reversal_adaptation_ticks,
        p_component,
        mi_component,
        entropy_component,
        predation_component,
    }
}

pub(crate) fn average_aggregate_scores(seed_summaries: &[SeedValidationSummary]) -> AggregateScore {
    let first = seed_summaries
        .first()
        .expect("multi-seed validation requires at least one seed");
    let score_stats = score_stats(
        seed_summaries
            .iter()
            .map(|summary| summary.aggregate_score.score),
    );
    AggregateScore {
        score: score_stats.mean,
        score_median: score_stats.median,
        score_stddev: score_stats.stddev,
        score_min: score_stats.min,
        score_max: score_stats.max,
        window_start_tick: first.aggregate_score.window_start_tick,
        window_end_tick: first.aggregate_score.window_end_tick,
        mean_p_fwd_food: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_p_fwd_food),
        ),
        mean_mi_sa: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_mi_sa),
        ),
        mean_mi_sa_juvenile: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_mi_sa_juvenile),
        ),
        mean_mi_sa_adult: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_mi_sa_adult),
        ),
        mean_h_action: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_h_action),
        ),
        mean_predation_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_predation_rate),
        ),
        mean_foraging_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_foraging_rate),
        ),
        mean_attack_attempt_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_attack_attempt_rate),
        ),
        mean_attack_success_rate: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_attack_success_rate),
        ),
        mean_idle_fraction: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_idle_fraction),
        ),
        mean_reproduction_efficiency: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_reproduction_efficiency),
        ),
        mean_lineage_diversity: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_lineage_diversity),
        ),
        mean_damage_avoidance: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_damage_avoidance),
        ),
        mean_reward_reversal_shift: mean_option(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_reward_reversal_shift),
        ),
        mean_action_histogram: mean_histogram(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mean_action_histogram),
        ),
        reward_reversal_adaptation_ticks: mean_option_u64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.reward_reversal_adaptation_ticks),
        ),
        p_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.p_component),
        ),
        mi_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.mi_component),
        ),
        entropy_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.entropy_component),
        ),
        predation_component: mean_f64(
            seed_summaries
                .iter()
                .map(|summary| summary.aggregate_score.predation_component),
        ),
    }
}

pub(crate) fn average_timeseries(seed_summaries: &[SeedValidationSummary]) -> Vec<IntervalMetrics> {
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
            h_action: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].h_action),
            ),
            idle_fraction: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].idle_fraction),
            ),
            reproduction_efficiency: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].reproduction_efficiency),
            ),
            damage_avoidance: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].damage_avoidance),
            ),
            reward_reversal_shift: mean_option(
                seed_summaries
                    .iter()
                    .map(|summary| summary.timeseries[row_idx].reward_reversal_shift),
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

pub(crate) fn entropy_component_score(mean_h_action: f64, h_baseline: f64) -> f64 {
    let entropy_target = 0.35 * h_baseline;
    let high_entropy_width = 0.45 * h_baseline;

    if mean_h_action <= entropy_target {
        let progress = clamp01(mean_h_action / entropy_target.max(f64::EPSILON));
        0.25 + 0.75 * progress * progress
    } else {
        let excess =
            clamp01((mean_h_action - entropy_target) / high_entropy_width.max(f64::EPSILON));
        1.0 - excess
    }
}

fn reward_reversal_adaptation_ticks(
    timeseries: &[IntervalMetrics],
    reward_reversal_tick: Option<u64>,
) -> Option<u64> {
    const REVERSAL_SHIFT_THRESHOLD: f64 = 0.12;
    let reward_reversal_tick = reward_reversal_tick?;
    timeseries
        .iter()
        .find(|row| {
            row.tick > reward_reversal_tick
                && row
                    .reward_reversal_shift
                    .is_some_and(|shift| shift >= REVERSAL_SHIFT_THRESHOLD)
        })
        .map(|row| row.tick.saturating_sub(reward_reversal_tick))
}

#[derive(Debug, Clone, Copy)]
struct ScoreStats {
    mean: f64,
    median: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

fn score_stats(values: impl Iterator<Item = f64>) -> ScoreStats {
    let mut scores = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if scores.is_empty() {
        return ScoreStats {
            mean: 0.0,
            median: 0.0,
            stddev: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    scores.sort_by(|a, b| a.total_cmp(b));
    let len = scores.len();
    let mean = scores.iter().sum::<f64>() / len as f64;
    let median = if len % 2 == 0 {
        (scores[len / 2 - 1] + scores[len / 2]) / 2.0
    } else {
        scores[len / 2]
    };
    let variance = scores
        .iter()
        .map(|score| {
            let delta = *score - mean;
            delta * delta
        })
        .sum::<f64>()
        / len as f64;

    ScoreStats {
        mean,
        median,
        stddev: variance.sqrt(),
        min: scores[0],
        max: scores[len - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::N_ACTIONS;
    use sim_types::ActionType;

    #[test]
    fn entropy_component_prefers_small_purposeful_repertoires() {
        let h_baseline = metrics::action_baseline_entropy();

        let collapsed = entropy_component_score(
            entropy_from_actions(&repeated(ActionType::Forward, 24)),
            h_baseline,
        );
        let purposeful = entropy_component_score(
            entropy_from_actions(&sequence_counts(&[
                (ActionType::Forward, 15),
                (ActionType::Eat, 3),
                (ActionType::TurnLeft, 2),
            ])),
            h_baseline,
        );
        let exploratory = entropy_component_score(
            entropy_from_actions(&sequence_counts(&[
                (ActionType::Forward, 10),
                (ActionType::Eat, 5),
                (ActionType::TurnLeft, 5),
                (ActionType::TurnRight, 5),
            ])),
            h_baseline,
        );
        let random_like = entropy_component_score(
            entropy_from_actions(&sequence_counts(&[
                (ActionType::Idle, 4),
                (ActionType::TurnLeft, 4),
                (ActionType::TurnRight, 4),
                (ActionType::Forward, 4),
                (ActionType::Eat, 4),
                (ActionType::Attack, 4),
                (ActionType::Reproduce, 4),
            ])),
            h_baseline,
        );

        assert!(purposeful > collapsed);
        assert!(purposeful > exploratory);
        assert!(collapsed > random_like);
        assert!(exploratory > random_like);
    }

    fn repeated(action: ActionType, count: usize) -> Vec<ActionType> {
        let mut actions = Vec::with_capacity(count);
        for _ in 0..count {
            actions.push(action);
        }
        actions
    }

    fn sequence_counts(counts: &[(ActionType, usize)]) -> Vec<ActionType> {
        let total = counts.iter().map(|(_, count)| *count).sum();
        let mut actions = Vec::with_capacity(total);
        for (action, count) in counts {
            for _ in 0..*count {
                actions.push(*action);
            }
        }
        actions
    }

    fn entropy_from_actions(actions: &[ActionType]) -> f64 {
        let mut counts = [0_u64; N_ACTIONS];
        for action in actions {
            let idx = match action {
                ActionType::Idle => 0,
                ActionType::TurnLeft => 1,
                ActionType::TurnRight => 2,
                ActionType::Forward => 3,
                ActionType::Eat => 4,
                ActionType::Attack => 5,
                ActionType::Reproduce => 6,
            };
            counts[idx] = counts[idx].saturating_add(1);
        }

        let total: u64 = counts.iter().sum();
        assert!(total > 0, "entropy test sequences must be non-empty");

        let mut entropy = 0.0;
        for count in counts {
            if count == 0 {
                continue;
            }
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
        entropy
    }
}
