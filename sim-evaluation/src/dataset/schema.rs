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

/// Number of origin classes; used to size per-origin tick aggregates.
pub const ORIGIN_COUNT: usize = 3;

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

/// One row per tick. Most fields are whole-world totals; the `descendant_*`
/// fields carry the descendants-only slice the analysis layer uses so pillars
/// and the timeseries ignore founder/injection contributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickSummaryRow {
    pub tick: u64,
    pub population: u32,
    /// Subset of `population` whose origin is `Descendant`.
    pub descendant_population: u32,
    pub max_generation: Option<u64>,
    pub births: u32,
    /// Successful reproduction events this tick. Excludes periodic-injection
    /// spawns, which are counted in `births` but not in this field.
    pub descendant_births: u32,
    pub deaths: u32,
    /// Deaths restricted to organisms whose origin is `Descendant`.
    pub descendant_deaths: u32,
    pub food_count: u32,
    pub consumptions: u32,
    pub predations: u32,
    pub food_spawned: u32,
    pub descendant_abs_dopamine_sum: f64,
    pub descendant_abs_dopamine_count: u32,
}

/// One row per living organism at a flush boundary. This is raw snapshot data;
/// analysis derives population-level statistics such as brain-size
/// percentiles later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationSnapshotRow {
    pub tick: u64,
    pub organism_id: u64,
    pub parent_id: Option<u64>,
    /// `OrganismOrigin` discriminant.
    pub origin: u8,
    pub species_id: u64,
    pub generation: u64,
    pub birth_tick: u64,
    pub age_turns: u64,
    pub age_of_maturity: u32,
    pub max_organism_age: u32,
    pub num_neurons: u32,
    pub synapse_count: u32,
    pub contingent_action_count: u64,
    pub failed_action_count: u64,
}

/// Long-format action counters: one row per (tick, origin, action_type).
/// Adding new action types doesn't reshape the schema. The `origin` column
/// lets the analysis layer drop founder/injection buckets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionCountRow {
    pub tick: u64,
    /// `OrganismOrigin` discriminant. Pairs with `action_type` to form the
    /// row key.
    pub origin: u8,
    /// `ActionType` encoded as its discriminant index. Kept as u8 for compact
    /// storage; reverse map lives in `sim_types::ActionType::ALL`.
    pub action_type: u8,
    pub count: u64,
    pub failed_count: u64,
    pub pre_maturity_count: u64,
    pub post_maturity_count: u64,
}

/// One row per organism, emitted at death (or at end-of-run for survivors).
/// Stores the organism-specific maturity cutoff plus a compact pre/post
/// maturity behavior summary so derived views can compare early vs late life
/// without requiring full per-tick action logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganismLifetimeRow {
    pub id: u64,
    pub parent_id: Option<u64>,
    /// `OrganismOrigin` discriminant. Descendants alone feed pillars and
    /// timeseries; founder/injection rows are retained for completeness.
    pub origin: u8,
    pub species_id: u64,
    pub birth_tick: u64,
    pub death_tick: Option<u64>,
    pub generation: u64,
    /// Mature from `maturity_tick` onward. Ages strictly below
    /// `age_of_maturity` are pre-maturity.
    pub age_of_maturity: u32,
    pub maturity_tick: u64,
    pub num_offspring: u32,
    pub total_consumptions: u64,
    pub total_actions: u64,
    /// Length `ACTION_COUNT`. Counts across the whole lifetime.
    pub action_histogram: Vec<u64>,
    pub utilization: f32,
    pub food_ahead_ticks: u32,
    pub fwd_when_food_ahead: u32,
    /// Row-major flattened `[SENSORY_BIN_COUNT][ACTION_COUNT]` across the
    /// whole lifetime.
    pub joint_sensory_action: Vec<u64>,
    pub pre_maturity_actions: u64,
    pub post_maturity_actions: u64,
    /// Length `ACTION_COUNT`. Counts for samples with age strictly below
    /// `age_of_maturity`.
    pub pre_maturity_action_histogram: Vec<u64>,
    /// Length `ACTION_COUNT`. Counts for samples at or after `maturity_tick`.
    pub post_maturity_action_histogram: Vec<u64>,
    pub pre_maturity_consumptions: u64,
    pub post_maturity_consumptions: u64,
    pub pre_maturity_food_ahead_ticks: u32,
    pub post_maturity_food_ahead_ticks: u32,
    pub pre_maturity_fwd_when_food_ahead: u32,
    pub post_maturity_fwd_when_food_ahead: u32,
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
