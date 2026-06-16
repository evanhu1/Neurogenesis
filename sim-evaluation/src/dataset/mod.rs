//! Central experiment dataset: raw per-tick population and per-organism
//! lifetime facts emitted during a sim run, written as partitioned Parquet
//! files plus a JSON manifest, then read back by the analysis layer.
//!
//! The dataset is the single source of truth. The sim loop only produces raw
//! data; every derived metric or report reads from here.

pub mod manifest;
pub mod reader;
pub mod schema;
pub mod writer;

pub use manifest::{Manifest, SCHEMA_VERSION};
pub use reader::DatasetReader;
pub use writer::PartitionedParquetWriter;

// The metric vocabulary is owned by `sim-metrics`; re-export the row types this
// crate persists under the historical `crate::dataset::…` path. The histogram
// constants/enum are consumed inside `sim-metrics` itself, so they are imported
// directly from there where still needed (e.g. dataset tests).
pub use sim_metrics::{OrganismLifetimeRow, TickSummaryRow};
