//! Drives the simulation and funnels everything it emits into the on-disk
//! dataset: per-tick population, per-reproduction, and per-organism lifetime
//! facts. Pillars, comparisons and reports are produced afterwards by the
//! analysis layer reading the same dataset.

use crate::analysis::{
    analyze, average_pillar_scores, average_timeseries, write_aggregate_artifacts,
    write_per_seed_artifacts, AnalysisOptions,
};
use crate::dataset::{
    DatasetReader, Manifest, PartitionedParquetWriter, TickSummaryRow, SCHEMA_VERSION,
};
use crate::ledger::Ledger;
use crate::types::{
    EvaluationSummary, HarnessRunOptions, SeedEvaluationSummary, SeedRunOptions, SeedRunSummary,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use ring::digest::{digest, SHA256};
use sim_core::Simulation;
use sim_metrics::{ingest_tick, register_founders};
use sim_types::WorldConfig;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

enum WorkerMessage {
    Progress {
        seed: u64,
        tick: u64,
        total_ticks: u64,
        elapsed: Duration,
    },
    Finished {
        result: Box<Result<SeedEvaluationSummary>>,
    },
}

#[derive(Clone, Copy)]
struct SeedProgress {
    tick: u64,
    total_ticks: u64,
    elapsed: Duration,
}

pub(crate) fn run_evaluation_across_seeds(
    config: WorldConfig,
    options: &HarnessRunOptions,
) -> Result<EvaluationSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;
    let worker_threads = default_worker_threads(options.seeds.len());
    let seed_queue = Arc::new(Mutex::new(VecDeque::from(options.seeds.clone())));
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::with_capacity(worker_threads);
    let mut progress_rows_rendered = 0_usize;
    let mut latest_progress = HashMap::with_capacity(options.seeds.len());

    // When multiple seed workers share the machine, split the available cores
    // across them so each seed's simulation gets a dedicated slice of threads
    // rather than fighting over a single shared rayon pool.
    let mut config = config;
    config.intent_parallel_threads = per_seed_intent_threads(worker_threads);

    for _ in 0..worker_threads {
        let config = config.clone();
        let seed_queue = Arc::clone(&seed_queue);
        let tx = tx.clone();
        let out_dir = options.out_dir.clone();
        let title = options.title.clone();
        let ticks = options.ticks;
        let report_every = options.report_every;
        let control = options.control;

        handles.push(thread::spawn(move || loop {
            let seed = match seed_queue.lock() {
                Ok(mut queue) => queue.pop_front(),
                Err(_) => None,
            };
            let Some(seed) = seed else {
                break;
            };
            let seed_options = SeedRunOptions {
                seed,
                ticks,
                report_every,
                out_dir: out_dir.join(format!("seed_{seed}")),
                title: title
                    .as_ref()
                    .map(|run_title| format!("{run_title} (seed {seed})")),
                control,
            };
            let result = run_single_seed_evaluation(config.clone(), seed_options, &tx);
            if tx
                .send(WorkerMessage::Finished {
                    result: Box::new(result),
                })
                .is_err()
            {
                break;
            }
        }));
    }
    drop(tx);

    let mut seed_summaries = Vec::with_capacity(options.seeds.len());
    while seed_summaries.len() < options.seeds.len() {
        match rx
            .recv()
            .map_err(|_| anyhow!("evaluation worker exited before reporting all seeds"))?
        {
            WorkerMessage::Progress {
                seed,
                tick,
                total_ticks,
                elapsed,
            } => {
                latest_progress.insert(
                    seed,
                    SeedProgress {
                        tick,
                        total_ticks,
                        elapsed,
                    },
                );
                render_seed_progress(
                    &options.seeds,
                    options.ticks,
                    &latest_progress,
                    &mut progress_rows_rendered,
                )?;
            }
            WorkerMessage::Finished { result } => {
                seed_summaries.push((*result)?);
            }
        }
    }
    if progress_rows_rendered > 0 {
        eprintln!();
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("evaluation worker panicked"))?;
    }
    seed_summaries.sort_by_key(|summary| summary.seed);

    let averaged_timeseries = average_timeseries(&seed_summaries);
    let pillars = average_pillar_scores(&seed_summaries);
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    let seed_run_summaries = seed_summaries
        .iter()
        .map(|summary| SeedRunSummary {
            seed: summary.seed,
            out_dir: PathBuf::from(format!("seed_{}", summary.seed)),
            total_time_seconds: summary.total_time_seconds,
            pillars: summary.pillars.clone(),
            state_hash: summary.state_hash.clone(),
        })
        .collect::<Vec<_>>();

    let summary = EvaluationSummary {
        title: options.title.clone(),
        seeds: options.seeds.clone(),
        ticks: options.ticks,
        control: options.control,
        worker_threads,
        total_time_seconds,
        pillars,
        seed_summaries: seed_run_summaries,
        timeseries: averaged_timeseries,
    };

    write_aggregate_artifacts(
        &options.out_dir,
        &summary,
        options.report_every,
        &generated_at_utc,
    )?;

    Ok(summary)
}

fn run_single_seed_evaluation(
    config: WorldConfig,
    options: SeedRunOptions,
    progress_tx: &mpsc::Sender<WorkerMessage>,
) -> Result<SeedEvaluationSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let mut sim = Simulation::new(config.clone(), options.seed)?;
    let mut ledger = Ledger::new();
    let mut writer = PartitionedParquetWriter::new(&options.out_dir)?;

    register_founders(&mut ledger, sim.organisms());

    for tick in 1..=options.ticks {
        let delta = sim.tick();

        for row in ingest_tick(&mut ledger, tick, &delta, sim.action_records()) {
            writer.emit_organism_lifetime(row);
        }

        writer.emit_tick(TickSummaryRow {
            tick,
            descendant_population: ledger.descendant_population(),
        });

        let flush_tick = tick % options.report_every == 0 || tick == options.ticks;
        if flush_tick {
            // Genome snapshot of the top reproducer only at flush boundaries —
            // iterating every organism is expensive.
            if let Some(top) = ledger.top_reproducer() {
                if let Ok(idx) = sim
                    .organisms()
                    .binary_search_by_key(&sim_types::OrganismId(top.id), |o| o.id)
                {
                    let organism = &sim.organisms()[idx];
                    writer.emit_genome_snapshot(
                        tick,
                        organism.id.0,
                        organism.species_id.0,
                        organism.generation,
                        top.num_offspring,
                        &organism.genome,
                    )?;
                }
            }

            let _ = progress_tx.send(WorkerMessage::Progress {
                seed: options.seed,
                tick,
                total_ticks: options.ticks,
                elapsed: run_started.elapsed(),
            });
            writer.flush()?;
        }
    }

    // Drain remaining live organisms to `organism_lifetimes` with death_tick = None.
    for row in ledger.drain_survivors() {
        writer.emit_organism_lifetime(row);
    }
    let state_hash = state_hash(&sim)?;
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        seed: options.seed,
        total_ticks: options.ticks,
        report_every: options.report_every,
        created_at_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        world_config: config.clone(),
    };
    manifest.write(&options.out_dir)?;
    writer.finalize()?;

    // Analysis phase — derive metrics/pillars from the dataset we just wrote.
    let dataset = DatasetReader::load(&options.out_dir)?;
    let analysis = analyze(
        &dataset,
        &AnalysisOptions {
            report_every: options.report_every,
            total_ticks: options.ticks,
        },
    );

    let total_time_seconds = run_started.elapsed().as_secs_f64();

    let summary = SeedEvaluationSummary {
        title: options.title.clone(),
        seed: options.seed,
        ticks: options.ticks,
        control: options.control,
        total_time_seconds,
        pillars: analysis.pillars.clone(),
        state_hash,
        timeseries: analysis.timeseries.clone(),
    };
    write_per_seed_artifacts(
        &options.out_dir,
        &summary,
        options.report_every,
        "per-seed timeseries",
    )?;

    Ok(summary)
}

fn render_seed_progress(
    seeds: &[u64],
    default_total_ticks: u64,
    latest_progress: &HashMap<u64, SeedProgress>,
    rows_rendered: &mut usize,
) -> Result<()> {
    if seeds.is_empty() || !io::stderr().is_terminal() {
        return Ok(());
    }

    let mut stderr = io::stderr().lock();
    if *rows_rendered > 0 {
        write!(stderr, "\x1b[{}A", *rows_rendered)?;
    }
    for seed in seeds {
        let progress = latest_progress.get(seed).copied().unwrap_or(SeedProgress {
            tick: 0,
            total_ticks: default_total_ticks,
            elapsed: Duration::ZERO,
        });
        write!(
            stderr,
            "\r\x1b[2K{}\n",
            format_seed_progress_line(*seed, progress)
        )?;
    }
    stderr.flush()?;
    *rows_rendered = seeds.len();
    Ok(())
}

fn format_seed_progress_line(seed: u64, progress: SeedProgress) -> String {
    const BAR_WIDTH: usize = 30;

    let total_ticks = progress.total_ticks.max(1);
    let tick = progress.tick.min(total_ticks);
    let fraction = tick as f64 / total_ticks as f64;
    let filled = ((fraction * BAR_WIDTH as f64).round() as usize).min(BAR_WIDTH);
    let mut bar = String::with_capacity(BAR_WIDTH);
    for idx in 0..BAR_WIDTH {
        bar.push(if idx < filled { '#' } else { '-' });
    }
    let eta = estimate_eta(progress);

    format!(
        "seed {seed:<8} [{bar}] {:>6.1}% {tick}/{total_ticks} eta {}",
        fraction * 100.0,
        format_duration_compact(eta)
    )
}

fn estimate_eta(progress: SeedProgress) -> Duration {
    let total_ticks = progress.total_ticks.max(1);
    let tick = progress.tick.min(total_ticks);
    if tick == 0 || progress.elapsed.is_zero() {
        return Duration::ZERO;
    }
    let remaining_ticks = total_ticks.saturating_sub(tick);
    if remaining_ticks == 0 {
        return Duration::ZERO;
    }

    Duration::from_secs_f64(progress.elapsed.as_secs_f64() * remaining_ticks as f64 / tick as f64)
}

fn format_duration_compact(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

/// SHA-256 fingerprint of the complete persisted simulation state.
///
/// `Simulation::save` is the canonical world serialization used by sim-cli;
/// it includes the config, turn, RNG, id counters, organisms and genomes,
/// pending actions, ecology, occupancy, and metrics while intentionally
/// excluding behavior-neutral transient instrumentation/threading buffers.
/// Hashing those bytes catches divergence that the former population/id/energy
/// aggregate could not observe.
pub(crate) fn state_hash(sim: &Simulation) -> Result<String> {
    let mut bytes = Vec::new();
    sim.save(&mut bytes)
        .map_err(|error| anyhow!("serializing final simulation state for hashing: {error}"))?;

    let hash = digest(&SHA256, &bytes);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(hash.as_ref().len() * 2);
    for &byte in hash.as_ref() {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

fn default_worker_threads(seed_count: usize) -> usize {
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, seed_count.max(1))
}

/// Choose how many intent threads each sim should use given that
/// `worker_threads` sims will be running concurrently on this machine.
///
/// Target = cores / worker_threads (rounded up), so all physical parallelism
/// is used but no seed's rayon pool competes with another seed's pool for the
/// same core. Capped at 4 to avoid over-splitting the per-tick work for small
/// runs — empirically, throughput plateaus around 4 threads for a single
/// 5000-organism simulation and excess workers add rayon scheduling overhead.
fn per_seed_intent_threads(worker_threads: usize) -> u32 {
    const PER_SEED_CAP: usize = 4;
    let cores = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let per_seed = cores.div_ceil(worker_threads.max(1));
    per_seed.clamp(1, PER_SEED_CAP) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::WorldConfig;
    use std::sync::mpsc;

    #[test]
    fn same_seed_yields_same_summary_hash() {
        let cfg = WorldConfig {
            world_width: 40,
            num_organisms: 300,
            force_random_actions: false,
            ..Default::default()
        };

        let out_a = test_output_dir("a");
        let out_b = test_output_dir("b");
        let options_a = SeedRunOptions {
            seed: 2026,
            ticks: 100,
            report_every: 50,
            out_dir: out_a.clone(),
            title: None,
            control: false,
        };
        let options_b = SeedRunOptions {
            out_dir: out_b.clone(),
            ..options_a.clone()
        };

        let (tx, _rx) = mpsc::channel();
        let summary_a = run_single_seed_evaluation(cfg.clone(), options_a, &tx)
            .expect("first run should succeed");
        let summary_b =
            run_single_seed_evaluation(cfg, options_b, &tx).expect("second run should succeed");

        assert_eq!(summary_a.state_hash, summary_b.state_hash);
        assert_eq!(
            rounded_json(&summary_a.timeseries),
            rounded_json(&summary_b.timeseries)
        );

        let _ = fs::remove_dir_all(out_a);
        let _ = fs::remove_dir_all(out_b);
    }

    fn test_output_dir(suffix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("sim-evaluation-test-{suffix}-{nanos}"))
    }

    fn rounded_json<T: serde::Serialize>(value: &T) -> serde_json::Value {
        let mut json = serde_json::to_value(value).expect("test value should serialize");
        round_json_numbers(&mut json);
        json
    }

    fn round_json_numbers(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Number(number) => {
                if let Some(float) = number.as_f64() {
                    let rounded = (float * 1_000_000_000_000.0).round() / 1_000_000_000_000.0;
                    *number = serde_json::Number::from_f64(rounded)
                        .expect("rounded finite float should remain serializable");
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    round_json_numbers(item);
                }
            }
            serde_json::Value::Object(entries) => {
                for entry in entries.values_mut() {
                    round_json_numbers(entry);
                }
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::String(_) => {
            }
        }
    }
}
