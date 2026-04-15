//! Derive `ReproductionAnalytics` (aggregate lineage-tracking readouts) from
//! the `reproduction_events` and `organism_lifetimes` tables.
//!
//! "Descendants" are organisms with `parent_id.is_some()` — everyone born
//! from an observed reproduction event, i.e. not a founder or periodic
//! injection.

use crate::dataset::{DatasetReader, ReproductionOutcome};
use crate::types::ReproductionAnalytics;
use std::collections::HashMap;

const SURVIVAL_AGE_30: u64 = 30;

pub fn compute_reproduction_analytics(
    dataset: &DatasetReader,
    total_ticks: u64,
) -> ReproductionAnalytics {
    // ---- reproduction_events rollup -------------------------------------
    let mut successful_births = 0_u64;
    let mut blocked_births = 0_u64;
    let mut parent_died_during_reproduction = 0_u64;
    let mut parent_energy_sum = 0.0_f64;
    let mut parent_energy_count = 0_u64;
    let mut first_success_age: HashMap<u64, u64> = HashMap::new();
    let mut last_success_tick: HashMap<u64, u64> = HashMap::new();
    let mut interval_sum = 0.0_f64;
    let mut interval_count = 0_u64;
    for event in &dataset.reproduction_events {
        let outcome = event.outcome;
        if outcome == ReproductionOutcome::Success.code() {
            successful_births = successful_births.saturating_add(1);
            parent_energy_sum += f64::from(event.parent_energy_after);
            parent_energy_count = parent_energy_count.saturating_add(1);
            first_success_age
                .entry(event.parent_id)
                .and_modify(|age| *age = (*age).min(event.parent_age_turns))
                .or_insert(event.parent_age_turns);
            if let Some(prev_tick) = last_success_tick.insert(event.parent_id, event.tick) {
                interval_sum += event.tick.saturating_sub(prev_tick) as f64;
                interval_count = interval_count.saturating_add(1);
            }
        } else if outcome == ReproductionOutcome::BlockedBirth.code() {
            blocked_births = blocked_births.saturating_add(1);
        } else if outcome == ReproductionOutcome::ParentDied.code() {
            parent_died_during_reproduction = parent_died_during_reproduction.saturating_add(1);
        }
    }
    let mean_parent_energy = mean_or_none(parent_energy_sum, parent_energy_count);
    let mean_age_at_first = mean_or_none(
        first_success_age
            .values()
            .copied()
            .map(|v| v as f64)
            .sum::<f64>(),
        first_success_age.len() as u64,
    );
    let mean_birth_interval = mean_or_none(interval_sum, interval_count);

    // ---- lifetime survival thresholds -----------------------------------
    let mut survived_to_30 = 0_u64;
    let mut survived_to_maturity = 0_u64;
    for row in &dataset.organism_lifetimes {
        // Only count descendants — we care about whether newly produced
        // lineages made it past the threshold, not founders.
        if row.parent_id.is_none() {
            continue;
        }
        let lifetime = match row.death_tick {
            Some(death) => death.saturating_sub(row.birth_tick),
            None => total_ticks.saturating_sub(row.birth_tick),
        };
        if lifetime >= SURVIVAL_AGE_30 {
            survived_to_30 = survived_to_30.saturating_add(1);
        }
        if lifetime >= u64::from(row.age_of_maturity) {
            survived_to_maturity = survived_to_maturity.saturating_add(1);
        }
    }

    ReproductionAnalytics {
        births: successful_births,
        successful_births,
        blocked_births,
        parent_died_during_reproduction,
        survived_to_30,
        survived_to_maturity,
        mean_parent_energy_after_successful_birth: mean_parent_energy,
        mean_age_at_first_successful_reproduction: mean_age_at_first,
        mean_successful_birth_interval: mean_birth_interval,
    }
}

fn mean_or_none(sum: f64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}
