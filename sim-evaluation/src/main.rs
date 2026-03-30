mod aggregation;
mod cli;
mod comparison;
mod ledger;
mod metrics;
mod orchestration;
mod output;
mod report;
mod types;

use anyhow::{anyhow, Result};
use clap::Parser;
use cli::{Cli, FeatureOverrides};
use comparison::{apply_feature_overrides, run_comparison_evaluation};
use orchestration::run_evaluation_across_seeds;
use output::{
    default_output_dir, normalize_seeds, print_comparison_summary, print_evaluation_summary,
};
use sim_config::load_world_config_from_path;
use types::HarnessRunOptions;

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: running sim-evaluation in debug mode; use `cargo run -p sim-evaluation --release -- ...` for much faster runs"
        );
    }

    let mut control_config = load_world_config_from_path(&cli.config)?;
    if cli.baseline {
        control_config.force_random_actions = true;
    }
    let overrides = FeatureOverrides {
        disable_plasticity: cli.disable_plasticity,
    };

    let seeds = normalize_seeds(cli.seeds);
    if seeds.is_empty() {
        return Err(anyhow!("sim-evaluation requires at least one seed"));
    }

    let options = HarnessRunOptions {
        seeds: seeds.clone(),
        ticks: cli.ticks,
        report_every: cli.report_every,
        min_lifetime: cli.min_lifetime,
        out_dir: cli.out.unwrap_or_else(|| default_output_dir(&seeds)),
        title: cli.title,
        baseline: cli.baseline,
    };

    if cli.compare || overrides.has_overrides() {
        let treatment_config = apply_feature_overrides(control_config.clone(), &overrides);
        let comparison =
            run_comparison_evaluation(control_config, treatment_config, &options, &overrides)?;
        print_comparison_summary(&options.out_dir, &comparison);
    } else {
        let summary = run_evaluation_across_seeds(control_config, &options)?;
        print_evaluation_summary(&options.out_dir, &summary);
    }

    Ok(())
}
