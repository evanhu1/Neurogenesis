//! Dataset-specific row structs. The metric vocabulary (the per-tick and
//! per-organism lifetime rows plus the histogram-shape constants) now lives in
//! `sim-metrics` and is re-exported through `dataset::mod`; this file keeps only
//! the rows that are purely a function of on-disk persistence.
//!
//! These structs ARE the dataset-specific part of the schema — Arrow field
//! definitions are derived from their `serde` shape via `serde_arrow`.

use serde::{Deserialize, Serialize};

/// Index row pointing at a serialized genome blob. One per flush interval
/// where at least one organism has consumed food.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeSnapshotIndexRow {
    pub snapshot_id: u64,
    pub tick: u64,
    pub organism_id: u64,
    pub species_id: u64,
    pub generation: u64,
    pub total_consumptions: u64,
    /// Path relative to the seed's dataset directory, e.g. `genomes/t001000.bin`.
    pub file_path: String,
}
