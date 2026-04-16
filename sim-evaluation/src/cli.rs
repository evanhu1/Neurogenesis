use clap::{Parser, Subcommand};
use std::path::PathBuf;

const DEFAULT_CONFIG_PATH: &str = "sim-evaluation/config.toml";
const DEFAULT_SEEDS: &str = "42,123,7,2026";

#[derive(Debug, Clone, Parser)]
#[command(name = "sim-evaluation")]
#[command(about = "Headless evaluation harness for the deterministic simulation")]
pub(crate) struct Cli {
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    pub(crate) config: PathBuf,
    #[arg(
        long = "seed",
        value_delimiter = ',',
        num_args = 1..,
        default_value = DEFAULT_SEEDS
    )]
    pub(crate) seeds: Vec<u64>,
    #[arg(long, default_value_t = 500_000)]
    pub(crate) ticks: u64,
    #[arg(long, default_value_t = 2_500, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) report_every: u64,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    #[arg(long)]
    pub(crate) title: Option<String>,
    /// Force organisms to act randomly — a degenerate control run used to
    /// measure the random-policy floor of the evaluation scoring system.
    #[arg(long, default_value_t = false)]
    pub(crate) control: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) compare: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) disable_plasticity: bool,

    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum Command {
    /// Run a fresh evaluation — sim + dataset emit + analysis + reports.
    /// This is the default when no subcommand is given.
    Run,
    /// Re-run analysis against an existing evaluation dataset and rewrite the
    /// `summary.json`, `timeseries.csv`, and `report.html` artifacts. Does not
    /// re-run the simulation.
    ///
    /// The argument can be: a path to a run root (with `seed_*` subdirs), a
    /// path to a single seed dataset, a timestamp prefix resolved under
    /// `artifacts/evaluation/`, or the literal `latest`.
    Analyze {
        /// Run identifier — path, timestamp prefix, or `latest`.
        run: String,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FeatureOverrides {
    pub(crate) disable_plasticity: bool,
}

impl FeatureOverrides {
    pub(crate) fn has_overrides(&self) -> bool {
        self.disable_plasticity
    }

    pub(crate) fn label(&self) -> String {
        let mut parts = Vec::new();
        if self.disable_plasticity {
            parts.push("disable-plasticity".to_owned());
        }
        if parts.is_empty() {
            "treatment".to_owned()
        } else {
            parts.join(", ")
        }
    }
}
