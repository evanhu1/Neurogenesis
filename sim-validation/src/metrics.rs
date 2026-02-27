use crate::ledger::{CompletedLifetime, N_ACTIONS, SENSORY_BIN_COUNT};
use serde::Serialize;
use sim_types::OrganismState;

#[derive(Debug, Clone, Serialize)]
pub struct IntervalMetrics {
    pub tick: u64,
    pub pop: u32,
    pub births: u64,
    pub deaths: u64,
    pub food: u64,
    pub max_generation: Option<u64>,
    pub life_mean: Option<f64>,
    pub life_max: Option<u64>,
    pub ate_pct: Option<f64>,
    pub cons_mean: Option<f64>,
    pub brain_size: Option<f64>,
    pub p_fwd_food: Option<f64>,
    pub mi_sa: Option<f64>,
    pub h_action: Option<f64>,
    pub util: Option<f64>,
}

pub fn compute_interval_metrics(
    tick: u64,
    pop: u32,
    births: u64,
    deaths: u64,
    food: u64,
    deceased: &[CompletedLifetime],
    living: &[OrganismState],
) -> IntervalMetrics {
    let brain_size = mean_living_brain_size(living);
    let max_generation = living.iter().map(|organism| organism.generation).max();

    let (life_mean, life_max, ate_pct, cons_mean, p_fwd_food, mi_sa, h_action, util) =
        if deceased.is_empty() {
            (None, None, None, None, None, None, None, None)
        } else {
            let life_sum: u64 = deceased.iter().map(|entry| entry.lifetime).sum();
            let life_max = deceased.iter().map(|entry| entry.lifetime).max();
            let ate_count = deceased
                .iter()
                .filter(|entry| entry.consumptions > 0)
                .count() as f64;
            let cons_sum: u64 = deceased.iter().map(|entry| entry.consumptions).sum();
            let util_mean = deceased
                .iter()
                .map(|entry| entry.utilization as f64)
                .sum::<f64>()
                / deceased.len() as f64;

            (
                Some(life_sum as f64 / deceased.len() as f64),
                life_max,
                Some(100.0 * ate_count / deceased.len() as f64),
                Some(cons_sum as f64 / deceased.len() as f64),
                pooled_p_fwd_food(deceased),
                pooled_mi_sa(deceased),
                pooled_action_entropy(deceased),
                Some(util_mean),
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
        life_max,
        ate_pct,
        cons_mean,
        brain_size,
        p_fwd_food,
        mi_sa,
        h_action,
        util,
    }
}

fn mean_living_brain_size(living: &[OrganismState]) -> Option<f64> {
    if living.is_empty() {
        return None;
    }

    let total = living
        .iter()
        .map(|organism| (organism.genome.num_neurons + organism.brain.synapse_count) as f64)
        .sum::<f64>();
    Some(total / living.len() as f64)
}

fn pooled_p_fwd_food(deceased: &[CompletedLifetime]) -> Option<f64> {
    let numerator: u64 = deceased
        .iter()
        .map(|entry| u64::from(entry.fwd_when_food_ahead))
        .sum();
    let denominator: u64 = deceased
        .iter()
        .map(|entry| u64::from(entry.food_ahead_ticks))
        .sum();
    if denominator == 0 {
        return None;
    }
    Some(numerator as f64 / denominator as f64)
}

fn pooled_action_entropy(deceased: &[CompletedLifetime]) -> Option<f64> {
    let mut pooled = [0_u64; N_ACTIONS];
    for entry in deceased {
        for (idx, count) in entry.action_counts.iter().enumerate() {
            pooled[idx] = pooled[idx].saturating_add(u64::from(*count));
        }
    }

    let total: u64 = pooled.iter().sum();
    if total == 0 {
        return None;
    }

    let mut entropy = 0.0;
    for count in pooled {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total as f64;
        entropy -= p * p.log2();
    }
    Some(entropy)
}

fn pooled_mi_sa(deceased: &[CompletedLifetime]) -> Option<f64> {
    let mut pooled_joint = [[0_u64; N_ACTIONS]; SENSORY_BIN_COUNT];
    for entry in deceased {
        for (sensory_idx, row) in entry.joint.iter().enumerate() {
            for (action_idx, count) in row.iter().enumerate() {
                pooled_joint[sensory_idx][action_idx] =
                    pooled_joint[sensory_idx][action_idx].saturating_add(u64::from(*count));
            }
        }
    }

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
