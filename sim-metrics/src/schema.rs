//! Raw fact vocabulary shared by every metric consumer. These structs ARE the
//! dataset schema — `sim-evaluation` derives its Arrow/Parquet columns from
//! their `serde` shape, and `sim-cli` accumulates the same rows in memory.
//!
//! The dataset is deliberately compact: per-tick population and per-organism
//! lifetime rows carry every fact the analysis consumes.

use serde::{Deserialize, Serialize};
use sim_types::{ActionType, SensoryReceptor};

/// Number of action variants in the joint histograms: Idle plus every
/// contingent action. The analysis layer reshapes flat `Vec<u64>` histograms
/// with this constant.
pub const ACTION_COUNT: usize = ActionType::ALL.len() + 1;

/// Number of sensory context bins used by the joint histograms. One bin for
/// "no food visible" plus one bin per vision ray.
pub const SENSORY_BIN_COUNT: usize = 1 + SensoryReceptor::VISION_RAY_OFFSETS.len();

/// Length of the flattened joint-sensory-action histogram.
pub const JOINT_LEN: usize = SENSORY_BIN_COUNT * ACTION_COUNT;

/// One row per tick. The only per-tick fact still consumed is the living
/// population, used for the `pop` viability-context line in the timeseries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickSummaryRow {
    pub tick: u64,
    /// Living organisms at this tick.
    pub population: u32,
}

/// One row per organism, emitted at death (or at end-of-run for survivors).
/// Every behavioural metric is derived by bucketing these rows by `death_tick`
/// into reporting intervals and pooling over the whole population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganismLifetimeRow {
    pub id: u64,
    pub death_tick: Option<u64>,
    /// Every action taken over the lifetime (includes Idle and Turns).
    pub total_actions: u64,
    /// Contingent actions taken (Forward/Eat/Attack — those that can fail).
    /// `contingent_actions - failed_actions` is the successful count.
    pub contingent_actions: u64,
    pub failed_actions: u64,
    /// Lifetime plant (foraging) consumptions.
    pub plant_consumptions: u64,
    /// Lifetime prey/corpse (predation) consumptions.
    pub prey_consumptions: u64,
    /// Row-major flattened `[SENSORY_BIN_COUNT][ACTION_COUNT]` across the whole
    /// lifetime — feeds MI(S;A).
    pub joint_sensory_action: Vec<u64>,
    /// Within-life regression slope of success (0/1) vs age over contingent
    /// actions (Forward/Eat/Attack). Positive ⇒ the organism got
    /// better at acting over its life — the in-life-learning signal. `None`
    /// when the organism took too few such actions to estimate a slope.
    pub learning_slope: Option<f32>,
}
