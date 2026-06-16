//! The per-organism lifetime ledger moved to `sim-metrics` so the interactive
//! CLI can accumulate the same rows live. Re-exported here so existing
//! `crate::ledger::…` paths keep resolving.

pub use sim_metrics::Ledger;
