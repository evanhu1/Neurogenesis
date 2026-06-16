//! Raw fact vocabulary shared by every metric consumer. These structs ARE the
//! dataset schema — `sim-evaluation` derives its Arrow/Parquet columns from
//! their `serde` shape, and `sim-cli` accumulates the same rows in memory.
//!
//! The dataset is deliberately minimal: a per-tick population line and a
//! per-organism lifetime row carry every fact the analysis layer consumes.

use serde::{Deserialize, Serialize};
use sim_types::{ActionType, SensoryReceptor};

/// Number of action variants in the joint histograms: Idle plus every
/// contingent action. The analysis layer reshapes flat `Vec<u64>` histograms
/// with this constant.
pub const ACTION_COUNT: usize = ActionType::ALL.len() + 1;

/// How an organism first appeared in the world. The analysis layer filters to
/// `Descendant` so that founder behaviour and periodic-injection bursts don't
/// contaminate evolution metrics; the other two variants remain in the dataset
/// for completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrganismOrigin {
    /// Part of the population spawned at tick 0.
    InitialFounder = 0,
    /// Spawned by the periodic seed-genome injection mechanism after tick 0.
    PeriodicInjection = 1,
    /// Born via a successful in-world reproduction event.
    Descendant = 2,
}

impl OrganismOrigin {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

pub const DESCENDANT_CODE: u8 = OrganismOrigin::Descendant.code();

/// Number of sensory context bins used by the joint histograms. One bin for
/// "no food visible" plus one bin per vision ray.
pub const SENSORY_BIN_COUNT: usize = 1 + SensoryReceptor::VISION_RAY_OFFSETS.len();

/// Length of the flattened joint-sensory-action histogram.
pub const JOINT_LEN: usize = SENSORY_BIN_COUNT * ACTION_COUNT;

/// One row per tick. The only per-tick fact still consumed is the descendant
/// population, used for the `pop` viability-context line in the timeseries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickSummaryRow {
    pub tick: u64,
    /// Living organisms whose origin is `Descendant`.
    pub descendant_population: u32,
}

/// One row per organism, emitted at death (or at end-of-run for survivors).
/// Every behavioural metric is derived by bucketing these rows by `death_tick`
/// into reporting intervals and pooling over descendants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganismLifetimeRow {
    pub id: u64,
    /// `OrganismOrigin` discriminant. Descendants alone feed pillars and
    /// timeseries; founder/injection rows are retained for completeness.
    pub origin: u8,
    pub death_tick: Option<u64>,
    /// Every action taken over the lifetime (includes Idle and Turns).
    pub total_actions: u64,
    /// Contingent actions taken (Forward/Eat/Attack/Reproduce — those that can
    /// fail). `contingent_actions - failed_actions` is the successful count.
    pub contingent_actions: u64,
    pub failed_actions: u64,
    /// Lifetime plant (foraging) consumptions.
    pub plant_consumptions: u64,
    /// Lifetime prey/corpse (predation) consumptions.
    pub prey_consumptions: u64,
    /// Row-major flattened `[SENSORY_BIN_COUNT][ACTION_COUNT]` across the whole
    /// lifetime — feeds MI(S;A).
    pub joint_sensory_action: Vec<u64>,
    /// Within-life regression slope of success (0/1) vs age over non-Reproduce
    /// contingent actions (Forward/Eat/Attack). Positive ⇒ the organism got
    /// better at acting over its life — the in-life-learning signal. `None`
    /// when the organism took too few such actions to estimate a slope.
    pub learning_slope: Option<f32>,
}
