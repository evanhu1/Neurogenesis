//! Raw fact vocabulary shared by every metric consumer. These structs ARE the
//! dataset schema — `evolution` derives its Arrow/Parquet columns from
//! their `serde` shape, and `cli` accumulates the same rows in memory.
//!
//! The dataset is deliberately compact: per-tick population, per-reporting-
//! interval behavior, and per-organism lifetime rows carry every fact the
//! analysis consumes.

use serde::{Deserialize, Serialize};
/// One row per tick. The only per-tick fact still consumed is the living
/// population, used for the `pop` viability-context line in the timeseries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickSummaryRow {
    pub tick: u64,
    /// Living organisms at this tick.
    pub population: u32,
}

/// Action-time facts accumulated over one reporting interval `(start_tick,
/// end_tick]`. Unlike lifetime cohorts, these rows include actions taken by
/// organisms that are still alive at the interval boundary, so they are the
/// source of truth for behavioral tail metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorIntervalRow {
    pub start_tick: u64,
    pub end_tick: u64,
    /// Living organisms at `end_tick`.
    pub population: u32,
    pub total_actions: u64,
    pub contingent_actions: u64,
    pub failed_actions: u64,
    pub successful_attacks: u64,
    /// Organism-fixed-effect OLS sufficient statistics for contingent-action
    /// success vs age. Centering within each organism prevents mixed-age cohort
    /// composition from masquerading as lifetime improvement.
    pub learning_samples: u64,
    pub learning_within_numerator: f64,
    pub learning_within_denominator: f64,
}

/// One row per organism, emitted at death (or at end-of-run for survivors).
/// These rows support lifecycle/cohort analyses; behavioral interval metrics
/// are derived from [`BehaviorIntervalRow`] so live survivors are not omitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganismLifetimeRow {
    pub id: u64,
    pub death_tick: Option<u64>,
    /// Every action taken over the lifetime (includes Idle and Turns).
    pub total_actions: u64,
    /// Contingent actions taken (Forward/Attack — those that can fail).
    /// `contingent_actions - failed_actions` is the successful count.
    pub contingent_actions: u64,
    pub failed_actions: u64,
    /// Lifetime successful attack energy transfers.
    pub successful_attacks: u64,
    /// Within-life regression slope of success (0/1) vs age over contingent
    /// actions (Forward/Attack). Positive ⇒ the organism got
    /// better at acting over its life — the in-life-learning signal. `None`
    /// when the organism took too few such actions to estimate a slope.
    pub learning_slope: Option<f32>,
}
