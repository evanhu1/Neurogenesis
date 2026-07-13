//! `derive_interval_metrics` over a loaded dataset. The pure, storage-agnostic
//! computation lives in `sim-metrics`; this wrapper adapts a `DatasetReader` so
//! the analysis layer keeps its dataset-shaped call.

use crate::dataset::DatasetReader;
use crate::types::IntervalMetrics;

pub fn derive_interval_metrics(
    dataset: &DatasetReader,
    _report_every: u64,
    _total_ticks: u64,
) -> Vec<IntervalMetrics> {
    sim_metrics::derive_interval_metrics(&dataset.behavior_intervals)
}
