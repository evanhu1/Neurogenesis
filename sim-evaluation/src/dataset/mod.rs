//! Central experiment dataset: raw tick/event/lifetime facts emitted during a
//! sim run, written as partitioned Parquet files plus a JSON manifest, then
//! read back by the analysis layer.
//!
//! The dataset is the single source of truth. The sim loop only produces raw
//! data; every derived metric or report reads from here.

pub mod manifest;
pub mod reader;
pub mod schema;
pub mod writer;

pub use manifest::{Manifest, SCHEMA_VERSION};
pub use reader::DatasetReader;
pub use schema::{
    OrganismLifetimeRow, OrganismOrigin, TickSummaryRow, ACTION_COUNT, DESCENDANT_CODE, JOINT_LEN,
    SENSORY_BIN_COUNT,
};
pub use writer::PartitionedParquetWriter;
