use clap::Parser;
use std::path::PathBuf;

const DEFAULT_CONFIG_PATH: &str = "sim-validation/config.toml";
const DEFAULT_SEEDS: &str = "42,123,7,2026,99,314,2718,4242,9001,65537";

#[derive(Debug, Clone, Parser)]
#[command(name = "sim-validation")]
#[command(about = "Headless validation harness for the deterministic simulation")]
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
    #[arg(long, default_value_t = 100_000)]
    pub(crate) ticks: u64,
    #[arg(long, default_value_t = 2_500, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) report_every: u64,
    #[arg(long, default_value_t = 30)]
    pub(crate) min_lifetime: u64,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    #[arg(long)]
    pub(crate) title: Option<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) baseline: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) compare: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) disable_plasticity: bool,
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
