//! Shared output primitives for the CLI: the text/JSON format switch, compact
//! distribution stats, and token-cheap sparklines. Every command renders
//! through these so output is uniform and agent-parseable.

use serde_json::{json, Value};

/// Default output format for commands that support both. Flipped globally with
/// the `format` command or per-command with `--json` / `--text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Format {
    #[default]
    Text,
    Json,
}

impl Format {
    pub(crate) fn is_json(self) -> bool {
        self == Format::Json
    }
}

/// Five-number summary of a scalar field across the population.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stats {
    pub n: usize,
    pub min: f64,
    pub p50: f64,
    pub mean: f64,
    pub p90: f64,
    pub max: f64,
}

impl Stats {
    /// Compute over `vals` (does not need to be pre-sorted). `None` if empty.
    pub(crate) fn of(vals: &[f64]) -> Option<Stats> {
        if vals.is_empty() {
            return None;
        }
        let mut v: Vec<f64> = vals.to_vec();
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        Some(Stats {
            n,
            min: v[0],
            p50: percentile(&v, 0.50),
            mean,
            p90: percentile(&v, 0.90),
            max: v[n - 1],
        })
    }

    pub(crate) fn text(&self) -> String {
        format!(
            "min={:.2} p50={:.2} mean={:.2} p90={:.2} max={:.2}",
            self.min, self.p50, self.mean, self.p90, self.max
        )
    }

    pub(crate) fn json(&self) -> Value {
        json!({
            "n": self.n,
            "min": self.min,
            "p50": self.p50,
            "mean": self.mean,
            "p90": self.p90,
            "max": self.max,
        })
    }
}

/// Nearest-rank percentile over an already-sorted slice (`q` in [0,1]).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 * q) as usize).min(sorted.len() - 1);
    sorted[idx]
}

const SPARK_LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Render a numeric series as a unicode block sparkline (one char per sample).
/// Scales to the series' own [min,max]; a flat series renders as the low block.
pub(crate) fn sparkline(vals: &[f64]) -> String {
    if vals.is_empty() {
        return String::new();
    }
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = max - min;
    vals.iter()
        .map(|&v| {
            let level = if span <= 0.0 {
                0
            } else {
                (((v - min) / span) * (SPARK_LEVELS.len() - 1) as f64).round() as usize
            };
            SPARK_LEVELS[level.min(SPARK_LEVELS.len() - 1)]
        })
        .collect()
}

/// Format an optional metric, printing `NA` when the signal is absent.
pub(crate) fn opt(value: Option<f64>, decimals: usize) -> String {
    value
        .map(|v| format!("{v:.decimals$}"))
        .unwrap_or_else(|| "NA".to_string())
}

/// JSON-encode an optional metric (null when absent).
pub(crate) fn opt_json(value: Option<f64>) -> Value {
    match value {
        Some(v) => json!(v),
        None => Value::Null,
    }
}
