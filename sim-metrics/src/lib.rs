//! Shared metric computation for the NeuroGenesis engine.
//!
//! This crate owns the single source of truth for every derived evolution
//! signal: the per-organism lifetime [`Ledger`], the per-reporting-interval
//! [`IntervalMetrics`], and the [`PillarScores`] competence axes. Both the
//! batch evaluation harness (`sim-evaluation`, which persists raw facts to
//! Parquet then re-derives metrics post-hoc) and the interactive research CLI
//! (`sim-cli`, which keeps a live in-memory ledger) consume these same
//! functions, so live and offline numbers can never drift.
//!
//! Layering:
//! - [`schema`] — the raw fact vocabulary (row structs, origin enum, the
//!   histogram-shape constants).
//! - [`ledger`] — accumulates raw facts per organism, producing
//!   [`OrganismLifetimeRow`]s at death.
//! - [`ingest`] — drives the ledger from live simulation tick data; shared by
//!   the eval orchestration and the CLI recorder.
//! - [`intervals`] — pools lifetime rows into per-interval timeseries.
//! - [`pillars`] — scores a timeseries window into competence axes.
//! - [`stats`] — small numeric helpers shared across the above.

pub mod ingest;
pub mod intervals;
pub mod ledger;
pub mod pillars;
pub mod schema;
pub mod stats;

pub use ingest::{ingest_tick, register_existing, register_founders};
pub use intervals::{derive_interval_metrics, IntervalMetrics};
pub use ledger::{Ledger, OrganismEntry};
pub use pillars::{compute_pillar_scores, PillarScores};
pub use schema::{OrganismLifetimeRow, TickSummaryRow, ACTION_COUNT, JOINT_LEN, SENSORY_BIN_COUNT};
pub use stats::{mean_f64, mean_option, mean_round_u32};
