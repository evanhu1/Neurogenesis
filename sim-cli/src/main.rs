use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use sim_core::Simulation;
use sim_protocol::{MetricsSnapshot, WorldConfig, WorldSnapshot};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "sim-cli")]
#[command(about = "NeuroGenesis simulation CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 100)]
        epochs: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        format: OutputFormat,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Step {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        ticks: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = false)]
        print_state: bool,
    },
    Benchmark {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 20)]
        epochs: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        organisms: Option<u32>,
        #[arg(long)]
        neurons: Option<u32>,
        #[arg(long)]
        synapses: Option<u32>,
    },
    Export {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 50)]
        epochs: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum, default_value_t = ExportFormat::Jsonl)]
        format: ExportFormat,
        #[arg(long)]
        out: PathBuf,
    },
    Replay {
        #[arg(long)]
        input: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Pretty,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExportFormat {
    Jsonl,
    Json,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    epochs: u32,
    seed: u64,
    final_epoch: u64,
    organism_count: usize,
    survivors_last_epoch: u32,
}

#[derive(Debug, Serialize)]
struct StepSummary {
    ticks: u32,
    final_tick_in_epoch: u32,
    actions_applied_last_tick: u64,
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    epochs: u32,
    elapsed_ms: u128,
    avg_ms_per_epoch: f64,
    avg_ms_per_tick: f64,
    normalized_us_per_unit: f64,
    final_metrics: MetricsSnapshot,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            epochs,
            seed,
            format,
            out,
        } => run_command(config, epochs, seed, format, out),
        Commands::Step {
            config,
            ticks,
            seed,
            print_state,
        } => step_command(config, ticks, seed, print_state),
        Commands::Benchmark {
            config,
            epochs,
            seed,
            organisms,
            neurons,
            synapses,
        } => benchmark_command(config, epochs, seed, organisms, neurons, synapses),
        Commands::Export {
            config,
            epochs,
            seed,
            format,
            out,
        } => export_command(config, epochs, seed, format, out),
        Commands::Replay { input } => replay_command(input),
    }
}

fn run_command(
    config_path: Option<PathBuf>,
    epochs: u32,
    seed: u64,
    format: OutputFormat,
    out: Option<PathBuf>,
) -> Result<()> {
    let cfg = load_config(config_path)?;
    let mut sim = Simulation::new(cfg, seed)?;
    sim.epoch_n(epochs);
    let snapshot = sim.snapshot();

    let summary = RunSummary {
        epochs,
        seed,
        final_epoch: snapshot.epoch,
        organism_count: snapshot.organisms.len(),
        survivors_last_epoch: snapshot.metrics.survivors_last_epoch,
    };

    match format {
        OutputFormat::Pretty => {
            let text = format!(
                "epochs={} seed={} final_epoch={} organisms={} survivors_last_epoch={}",
                summary.epochs,
                summary.seed,
                summary.final_epoch,
                summary.organism_count,
                summary.survivors_last_epoch
            );
            write_output(text, out)?;
        }
        OutputFormat::Json => {
            let text = serde_json::to_string_pretty(&summary)?;
            write_output(text, out)?;
        }
    }

    Ok(())
}

fn step_command(
    config_path: Option<PathBuf>,
    ticks: u32,
    seed: u64,
    print_state: bool,
) -> Result<()> {
    let cfg = load_config(config_path)?;
    let mut sim = Simulation::new(cfg, seed)?;
    sim.step_n(ticks.max(1));
    let snapshot = sim.snapshot();

    let summary = StepSummary {
        ticks: ticks.max(1),
        final_tick_in_epoch: snapshot.tick_in_epoch,
        actions_applied_last_tick: snapshot.metrics.actions_applied_last_tick,
    };

    println!("{}", serde_json::to_string_pretty(&summary)?);
    if print_state {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
    }

    Ok(())
}

fn benchmark_command(
    config_path: Option<PathBuf>,
    epochs: u32,
    seed: u64,
    organisms: Option<u32>,
    neurons: Option<u32>,
    synapses: Option<u32>,
) -> Result<()> {
    let mut cfg = load_config(config_path)?;
    if let Some(v) = organisms {
        cfg.num_organisms = v;
    }
    if let Some(v) = neurons {
        cfg.num_neurons = v;
    }
    if let Some(v) = synapses {
        cfg.num_synapses = v;
    }

    let mut sim = Simulation::new(cfg.clone(), seed)?;
    let start = Instant::now();
    sim.epoch_n(epochs.max(1));
    let elapsed = start.elapsed();
    let snapshot = sim.snapshot();

    let total_ticks = (epochs.max(1) as u128) * (cfg.steps_per_epoch as u128);
    let complexity = (cfg.num_organisms.max(1) as f64)
        * ((cfg.num_neurons + cfg.num_synapses).max(1) as f64)
        * (cfg.steps_per_epoch.max(1) as f64)
        * (epochs.max(1) as f64);

    let summary = BenchmarkSummary {
        epochs: epochs.max(1),
        elapsed_ms: elapsed.as_millis(),
        avg_ms_per_epoch: elapsed.as_secs_f64() * 1000.0 / epochs.max(1) as f64,
        avg_ms_per_tick: elapsed.as_secs_f64() * 1000.0 / total_ticks.max(1) as f64,
        normalized_us_per_unit: elapsed.as_secs_f64() * 1_000_000.0 / complexity.max(1.0),
        final_metrics: snapshot.metrics,
    };

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn export_command(
    config_path: Option<PathBuf>,
    epochs: u32,
    seed: u64,
    format: ExportFormat,
    out: PathBuf,
) -> Result<()> {
    let cfg = load_config(config_path)?;
    let mut sim = Simulation::new(cfg, seed)?;
    let lines = sim.export_trace_jsonl(epochs);

    let payload = match format {
        ExportFormat::Jsonl => lines.join("\n"),
        ExportFormat::Json => {
            let snapshots: Vec<WorldSnapshot> = lines
                .iter()
                .map(|line| serde_json::from_str::<WorldSnapshot>(line))
                .collect::<std::result::Result<_, _>>()?;
            serde_json::to_string_pretty(&snapshots)?
        }
    };

    fs::write(&out, payload)
        .with_context(|| format!("failed writing export to {}", out.display()))?;
    println!("exported trace to {}", out.display());
    Ok(())
}

fn replay_command(input: PathBuf) -> Result<()> {
    let content = fs::read_to_string(&input)
        .with_context(|| format!("failed to read replay input {}", input.display()))?;

    let snapshots: Vec<WorldSnapshot> = if input
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
    {
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str::<WorldSnapshot>)
            .collect::<std::result::Result<_, _>>()?
    } else {
        serde_json::from_str::<Vec<WorldSnapshot>>(&content)
            .or_else(|_| {
                content
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(serde_json::from_str::<WorldSnapshot>)
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .context("input is neither valid JSON array nor JSONL snapshots")?
    };

    let last = snapshots.last().context("replay input is empty")?;
    let summary = RunSummary {
        epochs: last.epoch as u32,
        seed: last.rng_seed,
        final_epoch: last.epoch,
        organism_count: last.organisms.len(),
        survivors_last_epoch: last.metrics.survivors_last_epoch,
    };

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn load_config(path: Option<PathBuf>) -> Result<WorldConfig> {
    if let Some(path) = path {
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let cfg: WorldConfig = toml::from_str(&raw)
            .with_context(|| format!("failed to parse TOML config {}", path.display()))?;
        Ok(cfg)
    } else {
        Ok(WorldConfig::default())
    }
}

fn write_output(text: String, out: Option<PathBuf>) -> Result<()> {
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed creating output directory {}", parent.display())
            })?;
        }
        fs::write(&path, text).with_context(|| format!("failed writing {}", path.display()))?;
        println!("wrote output to {}", path.display());
    } else {
        println!("{text}");
    }
    Ok(())
}
