use crate::ledger::{IntervalActionStats, IntervalLifetimeSummary, N_ACTIONS, SENSORY_BIN_COUNT};
use serde::Serialize;
use sim_types::{offspring_transfer_energy, OrganismState};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct IntervalMetrics {
    pub tick: u64,
    pub pop: u32,
    pub births: u64,
    pub deaths: u64,
    pub food: u64,
    pub max_generation: Option<u64>,
    pub life_mean: Option<f64>,
    pub predation_rate: Option<f64>,
    pub foraging_rate: Option<f64>,
    pub attack_attempt_rate: Option<f64>,
    pub attack_success_rate: Option<f64>,
    pub ate_pct: Option<f64>,
    pub cons_mean: Option<f64>,
    pub brain_size: Option<f64>,
    pub brain_size_stddev: Option<f64>,
    pub brain_size_p10: Option<f64>,
    pub brain_size_p50: Option<f64>,
    pub brain_size_p90: Option<f64>,
    pub lineage_diversity: Option<f64>,
    pub p_fwd_food: Option<f64>,
    pub mi_sa: Option<f64>,
    pub mi_sa_juvenile: Option<f64>,
    pub mi_sa_adult: Option<f64>,
    pub h_action: Option<f64>,
    pub idle_fraction: Option<f64>,
    pub reproduction_efficiency: Option<f64>,
    pub mean_gestation_ticks: Option<f64>,
    pub mean_offspring_transfer_energy: Option<f64>,
    pub damage_avoidance: Option<f64>,
    pub reward_reversal_shift: Option<f64>,
    pub util: Option<f64>,
    pub action_histogram: [f64; N_ACTIONS],
}

pub fn compute_interval_metrics(
    tick: u64,
    pop: u32,
    births: u64,
    deaths: u64,
    food: u64,
    interval_consumptions: u64,
    interval_predations: u64,
    interval_population_exposure: u64,
    deceased: &IntervalLifetimeSummary,
    living: &[OrganismState],
    action_stats: &IntervalActionStats,
    food_energy: f32,
) -> IntervalMetrics {
    let brain_stats = living_brain_stats(living);
    let max_generation = living.iter().map(|organism| organism.generation).max();
    let predation_rate = event_rate(interval_predations, interval_population_exposure);
    let foraging_rate = event_rate(interval_consumptions, interval_population_exposure);

    let total_actions = action_stats.total_actions();
    let attack_attempts = action_stats.action_counts[5];
    let attack_attempt_rate = event_rate(attack_attempts, interval_population_exposure);
    let attack_success_rate = if attack_attempts == 0 {
        None
    } else {
        Some(interval_predations as f64 / attack_attempts as f64)
    };
    let reproduction_efficiency = if action_stats.reproduction_attempts == 0 {
        None
    } else {
        Some(births as f64 / action_stats.reproduction_attempts as f64)
    };
    let damage_avoidance = if interval_population_exposure == 0 || food_energy <= 0.0 {
        None
    } else {
        let mean_damage = action_stats.total_damage_taken / interval_population_exposure as f64;
        Some((1.0 - (mean_damage / food_energy as f64)).clamp(0.0, 1.0))
    };
    let action_histogram = action_histogram(action_stats);
    let idle_fraction = if total_actions == 0 {
        None
    } else {
        Some(action_histogram[0])
    };
    let mean_gestation_ticks = mean_gestation_ticks(living);
    let mean_offspring_transfer_energy = mean_offspring_transfer_energy(living);

    let (life_mean, ate_pct, cons_mean, p_fwd_food, mi_sa, h_action, util) = if deceased.count == 0
    {
        (None, None, None, None, None, None, None)
    } else {
        (
            Some(deceased.lifetime_sum as f64 / deceased.count as f64),
            Some(100.0 * deceased.ate_count as f64 / deceased.count as f64),
            Some(deceased.consumptions_sum as f64 / deceased.count as f64),
            pooled_p_fwd_food(deceased),
            pooled_mi_sa(deceased),
            pooled_action_entropy(deceased),
            Some(deceased.utilization_sum / deceased.count as f64),
        )
    };

    IntervalMetrics {
        tick,
        pop,
        births,
        deaths,
        food,
        max_generation,
        life_mean,
        predation_rate,
        foraging_rate,
        attack_attempt_rate,
        attack_success_rate,
        ate_pct,
        cons_mean,
        brain_size: brain_stats.mean,
        brain_size_stddev: brain_stats.stddev,
        brain_size_p10: brain_stats.p10,
        brain_size_p50: brain_stats.p50,
        brain_size_p90: brain_stats.p90,
        lineage_diversity: lineage_diversity(living),
        p_fwd_food,
        mi_sa,
        mi_sa_juvenile: pooled_mi_sa_from_u64(&action_stats.juvenile_joint),
        mi_sa_adult: pooled_mi_sa_from_u64(&action_stats.adult_joint),
        h_action,
        idle_fraction,
        reproduction_efficiency,
        mean_gestation_ticks,
        mean_offspring_transfer_energy,
        damage_avoidance,
        reward_reversal_shift: None,
        util,
        action_histogram,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct BrainStats {
    mean: Option<f64>,
    stddev: Option<f64>,
    p10: Option<f64>,
    p50: Option<f64>,
    p90: Option<f64>,
}

fn event_rate(events: u64, interval_population_exposure: u64) -> Option<f64> {
    if interval_population_exposure == 0 {
        return None;
    }
    Some(events as f64 / interval_population_exposure as f64)
}

fn living_brain_stats(living: &[OrganismState]) -> BrainStats {
    if living.is_empty() {
        return BrainStats::default();
    }

    let mut sizes = living
        .iter()
        .map(|organism| (organism.genome.num_neurons + organism.brain.synapse_count) as f64)
        .collect::<Vec<_>>();
    sizes.sort_by(|a, b| a.total_cmp(b));

    let len = sizes.len() as f64;
    let mean = sizes.iter().sum::<f64>() / len;
    let variance = sizes
        .iter()
        .map(|size| {
            let delta = *size - mean;
            delta * delta
        })
        .sum::<f64>()
        / len;

    BrainStats {
        mean: Some(mean),
        stddev: Some(variance.sqrt()),
        p10: percentile(&sizes, 0.10),
        p50: percentile(&sizes, 0.50),
        p90: percentile(&sizes, 0.90),
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted.get(idx).copied()
}

fn action_histogram(action_stats: &IntervalActionStats) -> [f64; N_ACTIONS] {
    let total = action_stats.total_actions();
    if total == 0 {
        return [0.0; N_ACTIONS];
    }

    let mut histogram = [0.0; N_ACTIONS];
    for (idx, count) in action_stats.action_counts.iter().enumerate() {
        histogram[idx] = *count as f64 / total as f64;
    }
    histogram
}

fn lineage_diversity(living: &[OrganismState]) -> Option<f64> {
    if living.is_empty() {
        return None;
    }

    let mut counts = HashMap::new();
    for organism in living {
        *counts.entry(organism.species_id).or_insert(0_u64) += 1;
    }
    let total = living.len() as f64;
    let mut shannon = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total;
        shannon -= p * p.log2();
    }
    Some(shannon)
}

fn mean_gestation_ticks(living: &[OrganismState]) -> Option<f64> {
    if living.is_empty() {
        return None;
    }
    Some(
        living
            .iter()
            .map(|organism| f64::from(organism.genome.gestation_ticks))
            .sum::<f64>()
            / living.len() as f64,
    )
}

fn mean_offspring_transfer_energy(living: &[OrganismState]) -> Option<f64> {
    if living.is_empty() {
        return None;
    }
    Some(
        living
            .iter()
            .map(|organism| f64::from(offspring_transfer_energy(organism.genome.gestation_ticks)))
            .sum::<f64>()
            / living.len() as f64,
    )
}

fn pooled_p_fwd_food(deceased: &IntervalLifetimeSummary) -> Option<f64> {
    if deceased.food_ahead_ticks_sum == 0 {
        return None;
    }
    Some(deceased.fwd_when_food_ahead_sum as f64 / deceased.food_ahead_ticks_sum as f64)
}

fn pooled_action_entropy(deceased: &IntervalLifetimeSummary) -> Option<f64> {
    action_entropy_from_counts(&deceased.action_counts)
}

fn action_entropy_from_counts(counts: &[u64; N_ACTIONS]) -> Option<f64> {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return None;
    }

    let mut entropy = 0.0;
    for count in counts {
        if *count == 0 {
            continue;
        }
        let p = *count as f64 / total as f64;
        entropy -= p * p.log2();
    }
    Some(entropy)
}

fn pooled_mi_sa(deceased: &IntervalLifetimeSummary) -> Option<f64> {
    pooled_mi_sa_from_u64(&deceased.joint)
}

fn pooled_mi_sa_from_u64(pooled_joint: &[[u64; N_ACTIONS]; SENSORY_BIN_COUNT]) -> Option<f64> {
    let total_obs: u64 = pooled_joint.iter().flatten().sum();
    if total_obs == 0 {
        return None;
    }

    let mut p_s = [0_u64; SENSORY_BIN_COUNT];
    let mut p_a = [0_u64; N_ACTIONS];
    let mut nonzero_cells: u64 = 0;
    for sensory_idx in 0..SENSORY_BIN_COUNT {
        for (action_idx, count) in pooled_joint[sensory_idx].iter().enumerate() {
            p_s[sensory_idx] = p_s[sensory_idx].saturating_add(*count);
            p_a[action_idx] = p_a[action_idx].saturating_add(*count);
            if *count > 0 {
                nonzero_cells = nonzero_cells.saturating_add(1);
            }
        }
    }

    let n = total_obs as f64;
    let mut mi = 0.0;
    for sensory_idx in 0..SENSORY_BIN_COUNT {
        for action_idx in 0..N_ACTIONS {
            let joint_count = pooled_joint[sensory_idx][action_idx];
            if joint_count == 0 {
                continue;
            }
            let p_sa = joint_count as f64 / n;
            let p_s = p_s[sensory_idx] as f64 / n;
            let p_a = p_a[action_idx] as f64 / n;
            mi += p_sa * (p_sa / (p_s * p_a)).log2();
        }
    }

    let correction = if nonzero_cells > 1 {
        (nonzero_cells as f64 - 1.0) / (2.0 * n * std::f64::consts::LN_2)
    } else {
        0.0
    };

    Some((mi - correction).max(0.0))
}

pub fn action_baseline_probability() -> f64 {
    1.0 / N_ACTIONS as f64
}

pub fn action_baseline_entropy() -> f64 {
    (N_ACTIONS as f64).log2()
}

pub fn jensen_shannon_divergence(a: &[f64; N_ACTIONS], b: &[f64; N_ACTIONS]) -> Option<f64> {
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if !sum_a.is_finite() || !sum_b.is_finite() || sum_a <= 0.0 || sum_b <= 0.0 {
        return None;
    }

    let mut m = [0.0; N_ACTIONS];
    for idx in 0..N_ACTIONS {
        m[idx] = 0.5 * (a[idx] + b[idx]);
    }

    Some(0.5 * kl_divergence(a, &m) + 0.5 * kl_divergence(b, &m))
}

fn kl_divergence(a: &[f64; N_ACTIONS], b: &[f64; N_ACTIONS]) -> f64 {
    let mut kl = 0.0;
    for idx in 0..N_ACTIONS {
        let p = a[idx];
        let q = b[idx];
        if p <= 0.0 || q <= 0.0 {
            continue;
        }
        kl += p * (p / q).log2();
    }
    kl
}
