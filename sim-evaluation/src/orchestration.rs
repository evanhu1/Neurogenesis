//! Drives the simulation and funnels everything it emits into the on-disk
//! dataset. Every per-tick row, per-action count, per-death lifetime, and
//! per-reproduction event is written as raw data. Pillars, comparisons and
//! reports are produced afterwards by the analysis layer reading the same
//! dataset.

use crate::analysis::{
    analyze, average_demographic_analytics, average_pillar_scores, average_timeseries,
    write_per_seed_artifacts, AnalysisOptions, ScoringWindow,
};
use crate::dataset::{
    DatasetReader, DatasetWriter as DatasetWriterTrait, Manifest, PartitionedParquetWriter,
    PopulationSnapshotRow, TickSummaryRow, SCHEMA_VERSION,
};
use crate::ledger::Ledger;
use crate::output::{write_summary_json, write_timeseries_csv};
use crate::report::{write_html_report, HtmlReportContext, HtmlReportMeta, PerSeedReportRow};
use crate::types::{
    EvaluationSummary, HarnessRunOptions, SeedEvaluationSummary, SeedRunOptions, SeedRunSummary,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use sim_core::Simulation;
use sim_types::{EntityId, OrganismState, WorldConfig};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
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
        _seed: u64,
        result: Result<SeedEvaluationSummary>,
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
        let min_lifetime = options.min_lifetime;
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
                min_lifetime,
                out_dir: out_dir.join(format!("seed_{seed}")),
                title: title
                    .as_ref()
                    .map(|run_title| format!("{run_title} (seed {seed})")),
                control,
            };
            let result = run_single_seed_evaluation(config.clone(), seed_options, &tx);
            if tx
                .send(WorkerMessage::Finished {
                    _seed: seed,
                    result,
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
            WorkerMessage::Finished { _seed: _, result } => {
                seed_summaries.push(result?);
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
    write_timeseries_csv(&options.out_dir, &averaged_timeseries)?;

    let pillars = average_pillar_scores(&seed_summaries);
    let demographics = average_demographic_analytics(&seed_summaries);
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    let seed_run_summaries = seed_summaries
        .iter()
        .map(|summary| SeedRunSummary {
            seed: summary.seed,
            out_dir: PathBuf::from(format!("seed_{}", summary.seed)),
            total_time_seconds: summary.total_time_seconds,
            pillars: summary.pillars.clone(),
            demographics: summary.demographics.clone(),
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
        pillars: pillars.clone(),
        demographics,
        seed_summaries: seed_run_summaries.clone(),
        timeseries: averaged_timeseries.clone(),
    };

    write_summary_json(&options.out_dir, &summary)?;
    let per_seed_rows = seed_run_summaries
        .iter()
        .map(|seed_summary| PerSeedReportRow {
            seed: seed_summary.seed,
            total_time_seconds: seed_summary.total_time_seconds,
            state_hash: seed_summary.state_hash.clone(),
            report_href: format!("seed_{}/report.html", seed_summary.seed),
        })
        .collect();
    write_html_report(
        &options.out_dir,
        &HtmlReportMeta::from_pillars(
            &summary.pillars,
            HtmlReportContext {
                title: summary.title.clone(),
                ticks: summary.ticks,
                report_every: options.report_every,
                min_lifetime: options.min_lifetime,
                control: summary.control,
                total_time_seconds: summary.total_time_seconds,
                generated_at_utc,
                timeseries_label: "mean across seeds".to_owned(),
                per_seed_rows,
            },
        ),
        &summary.timeseries,
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

    for organism in sim.organisms() {
        ledger.birth(
            organism.id,
            organism.species_id.0,
            organism.generation,
            0,
            organism.genome.lifecycle.age_of_maturity,
        );
    }

    let mut current_food_count = sim.snapshot().foods.len() as u64;
    // Running max across all organisms ever observed. Generation is monotonic
    // in reproduction order, so folding over `delta.spawned` is sufficient —
    // iterating every living organism each tick would be O(population) for
    // the same answer.
    let mut max_generation = sim
        .organisms()
        .iter()
        .map(|o| o.generation)
        .max()
        .unwrap_or(0);

    for tick in 1..=options.ticks {
        let delta = sim.tick();

        for record in sim.action_records().iter().flatten() {
            ledger.record_action(record);
        }
        for event in delta.reproduction_events.iter().copied() {
            let row = ledger.record_reproduction(tick, event);
            writer.emit_reproduction_event(row);
        }

        let births = delta.spawned.len() as u32;
        for spawned in &delta.spawned {
            max_generation = max_generation.max(spawned.generation);
            ledger.birth(
                spawned.id,
                spawned.species_id.0,
                spawned.generation,
                tick,
                spawned.genome.lifecycle.age_of_maturity,
            );
        }

        let mut deaths = 0_u32;
        for removed in &delta.removed_positions {
            match removed.entity_id {
                EntityId::Organism(id) => {
                    deaths = deaths.saturating_add(1);
                    if let Some(row) = ledger.death(id, tick) {
                        writer.emit_organism_lifetime(row);
                    }
                }
                EntityId::Food(_) => {
                    current_food_count = current_food_count.saturating_sub(1);
                }
            }
        }
        let food_spawned = delta.food_spawned.len() as u32;
        current_food_count = current_food_count.saturating_add(u64::from(food_spawned));

        let population = delta.metrics.organisms;

        writer.emit_tick(TickSummaryRow {
            tick,
            population,
            max_generation: if population > 0 {
                Some(max_generation)
            } else {
                None
            },
            births,
            deaths,
            food_count: current_food_count as u32,
            consumptions: delta.metrics.consumptions_last_turn as u32,
            predations: delta.metrics.predations_last_turn as u32,
            food_spawned,
        });

        for row in ledger.take_tick_aggregates().into_rows(tick) {
            writer.emit_action_count(row);
        }

        let flush_tick = tick % options.report_every == 0 || tick == options.ticks;
        if flush_tick {
            // Population-wide snapshot (brain stats, lineage diversity) only at
            // flush boundaries — iterating every organism is expensive.
            let snapshot = compute_population_snapshot(tick, sim.organisms());
            writer.emit_population_snapshot(snapshot);

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
    let state_hash = state_hash(sim.organisms());
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        seed: options.seed,
        total_ticks: options.ticks,
        report_every: options.report_every,
        snapshot_interval: options.report_every,
        created_at_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        world_config: config.clone(),
    };
    manifest.write(&options.out_dir)?;
    Box::new(writer).finalize()?;

    // Analysis phase — derive metrics/pillars from the dataset we just wrote.
    let dataset = DatasetReader::load(&options.out_dir)?;
    let analysis = analyze(
        &dataset,
        &AnalysisOptions {
            report_every: options.report_every,
            total_ticks: options.ticks,
            min_lifetime: options.min_lifetime,
            scoring_window: ScoringWindow::default(),
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
        demographics: analysis.demographics.clone(),
        state_hash,
        timeseries: analysis.timeseries.clone(),
    };
    write_per_seed_artifacts(
        &options.out_dir,
        &summary,
        options.report_every,
        options.min_lifetime,
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

fn compute_population_snapshot(tick: u64, organisms: &[OrganismState]) -> PopulationSnapshotRow {
    if organisms.is_empty() {
        return PopulationSnapshotRow {
            tick,
            brain_size_mean: None,
            brain_size_stddev: None,
            brain_size_p10: None,
            brain_size_p50: None,
            brain_size_p90: None,
            lineage_diversity: None,
        };
    }

    let mut sizes: Vec<f64> = organisms
        .iter()
        .map(|o| (o.genome.topology.num_neurons + o.brain.synapse_count) as f64)
        .collect();
    sizes.sort_by(|a, b| a.total_cmp(b));
    let len = sizes.len() as f64;
    let mean = sizes.iter().sum::<f64>() / len;
    let variance = sizes
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / len;

    let percentile = |fraction: f64| -> Option<f64> {
        let idx = ((sizes.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
        sizes.get(idx).copied()
    };

    let mut counts = std::collections::HashMap::new();
    for organism in organisms {
        *counts.entry(organism.species_id).or_insert(0_u64) += 1;
    }
    let total = organisms.len() as f64;
    let mut shannon = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total;
        shannon -= p * p.log2();
    }

    PopulationSnapshotRow {
        tick,
        brain_size_mean: Some(mean),
        brain_size_stddev: Some(variance.sqrt()),
        brain_size_p10: percentile(0.10),
        brain_size_p50: percentile(0.50),
        brain_size_p90: percentile(0.90),
        lineage_diversity: Some(shannon),
    }
}

pub(crate) fn state_hash(organisms: &[OrganismState]) -> String {
    let population_count = organisms.len() as u64;
    let sum_ids = organisms
        .iter()
        .map(|o| o.id.0)
        .fold(0_u64, |acc, value| acc.wrapping_add(value));
    let total_energy = organisms.iter().map(|o| o.energy as f64).sum::<f64>();

    let mut hasher = DefaultHasher::new();
    population_count.hash(&mut hasher);
    sum_ids.hash(&mut hasher);
    total_energy.to_bits().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
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
            periodic_injection_interval_turns: 0,
            periodic_injection_count: 0,
            force_random_actions: false,
            ..Default::default()
        };

        let out_a = test_output_dir("a");
        let out_b = test_output_dir("b");
        let options_a = SeedRunOptions {
            seed: 2026,
            ticks: 100,
            report_every: 50,
            min_lifetime: 10,
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
