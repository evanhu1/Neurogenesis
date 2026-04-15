//! Row structs for every dataset table. These structs ARE the schema — Arrow
//! field definitions are derived from their `serde` shape via `serde_arrow`,
//! so adding/removing a column means editing exactly one struct.

use serde::{Deserialize, Serialize};
use sim_types::SensoryReceptor;

/// Number of action variants in the joint histograms and `action_counts`
/// table. Matches `sim_types::ActionType::ALL.len() + 1` (Idle + 6 contingent
/// actions), and the analysis layer reshapes flat Vec<u64> histograms with
/// this constant.
pub const ACTION_COUNT: usize = 7;

/// Number of sensory context bins used by the joint histograms. One bin for
/// "no food visible" plus one bin per vision ray.
pub const SENSORY_BIN_COUNT: usize = 1 + SensoryReceptor::VISION_RAY_OFFSETS.len();

/// Length of the flattened joint-sensory-action histogram.
pub const JOINT_LEN: usize = SENSORY_BIN_COUNT * ACTION_COUNT;

/// One row per tick. All fields are cheap scalars; expensive population
/// statistics (brain-size distribution, lineage diversity) live in
/// `population_snapshots` instead because they require iterating every living
/// organism.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickSummaryRow {
    pub tick: u64,
    pub population: u32,
    pub max_generation: Option<u64>,
    pub births: u32,
    pub deaths: u32,
    pub food_count: u32,
    pub consumptions: u32,
    pub predations: u32,
    pub food_spawned: u32,
}

/// Expensive per-population statistics sampled at each flush boundary. Sparse
/// by design — there is one row per reporting interval, not one per tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationSnapshotRow {
    pub tick: u64,
    pub brain_size_mean: Option<f64>,
    pub brain_size_stddev: Option<f64>,
    pub brain_size_p10: Option<f64>,
    pub brain_size_p50: Option<f64>,
    pub brain_size_p90: Option<f64>,
    pub lineage_diversity: Option<f64>,
}

/// Long-format action counters: one row per (tick, action_type). Adding new
/// action types doesn't reshape the schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionCountRow {
    pub tick: u64,
    /// `ActionType` encoded as its discriminant index. Kept as u8 for compact
    /// storage; reverse map lives in `sim_types::ActionType::ALL`.
    pub action_type: u8,
    pub count: u64,
    pub failed_count: u64,
    pub juvenile_count: u64,
    pub adult_count: u64,
}

/// One row per organism, emitted at death (or at end-of-run for survivors).
/// Joints are stored split by maturity stage so the analysis layer can derive
/// `mi_sa_juvenile`/`mi_sa_adult` by pooling across organisms that died (or
/// were still alive at run end) inside each reporting interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganismLifetimeRow {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub species_id: u64,
    pub birth_tick: u64,
    pub death_tick: Option<u64>,
    pub generation: u64,
    pub age_of_maturity: u32,
    pub num_offspring: u32,
    pub total_consumptions: u64,
    pub total_actions: u64,
    /// Length `ACTION_COUNT`. Counts across the whole lifetime.
    pub action_histogram: Vec<u64>,
    pub utilization: f32,
    pub food_ahead_ticks: u32,
    pub fwd_when_food_ahead: u32,
    /// Row-major flattened `[SENSORY_BIN_COUNT][ACTION_COUNT]`, juvenile-only
    /// samples (ages strictly below `age_of_maturity`).
    pub joint_juvenile: Vec<u64>,
    /// Row-major flattened `[SENSORY_BIN_COUNT][ACTION_COUNT]`, adult-only.
    pub joint_adult: Vec<u64>,
}

/// One row per reproduction attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReproductionEventRow {
    pub tick: u64,
    pub parent_id: u64,
    pub parent_species_id: u64,
    pub parent_generation: u64,
    pub parent_age_turns: u64,
    pub child_id: Option<u64>,
    pub investment_energy: f32,
    pub parent_energy_after: f32,
    pub outcome: u8,
}

/// Encoded into `ReproductionEventRow::outcome` so readers don't need the
/// `sim_types` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReproductionOutcome {
    Success = 0,
    BlockedBirth = 1,
    ParentDied = 2,
}

impl ReproductionOutcome {
    pub fn code(self) -> u8 {
        self as u8
    }
}

/// Index row pointing at a serialized genome blob. One per flush interval
/// where at least one organism has reproduced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeSnapshotIndexRow {
    pub snapshot_id: u64,
    pub tick: u64,
    pub organism_id: u64,
    pub species_id: u64,
    pub generation: u64,
    pub num_offspring: u32,
    /// Path relative to the seed's dataset directory, e.g. `genomes/t001000.bin`.
    pub file_path: String,
}
