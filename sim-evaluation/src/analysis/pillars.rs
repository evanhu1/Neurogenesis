//! Pillar scoring moved to `sim-metrics` so the live CLI scores identically.
//! Re-exported here so `analysis::compute_pillar_scores` keeps resolving.

pub use sim_metrics::compute_pillar_scores;
