use crate::{
    aggregation::{
        average_pillar_scores, average_reproduction_analytics, average_timeseries,
        compute_pillar_scores, state_hash,
    },
    ledger::Ledger,
    metrics::compute_interval_metrics,
    output::{write_summary_json, write_timeseries_csv},
    report::{write_html_report, HtmlReportMeta, PerSeedReportRow, Reporter},
    types::{
        EvaluationSummary, HarnessRunOptions, SeedEvaluationSummary, SeedRunOptions, SeedRunSummary,
    },
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use sim_core::Simulation;
use sim_types::{EntityId, WorldConfig};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

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

    // When multiple seed workers share the machine, split the available cores
    // across them so each seed's simulation gets a dedicated slice of threads
    // rather than fighting over a single shared rayon pool. Without this each
    // sim was requesting 8 workers concurrently, oversubscribing the cores
    // and serializing par_iter_mut work through a contended pool.
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
            let result = run_single_seed_evaluation(config.clone(), seed_options);
            if tx.send((seed, result)).is_err() {
                break;
            }
        }));
    }
    drop(tx);

    let mut seed_summaries = Vec::with_capacity(options.seeds.len());
    for _ in 0..options.seeds.len() {
        let (_seed, result) = rx
            .recv()
            .map_err(|_| anyhow!("evaluation worker exited before reporting all seeds"))?;
        seed_summaries.push(result?);
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
    let experiment_readouts = average_reproduction_analytics(&seed_summaries);
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    let seed_run_summaries = seed_summaries
        .iter()
        .map(|summary| SeedRunSummary {
            seed: summary.seed,
            out_dir: PathBuf::from(format!("seed_{}", summary.seed)),
            total_time_seconds: summary.total_time_seconds,
            pillars: summary.pillars.clone(),
            experiment_readouts: summary.experiment_readouts.clone(),
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
        experiment_readouts,
        seed_summaries: seed_run_summaries.clone(),
        timeseries: averaged_timeseries.clone(),
    };

    write_summary_json(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &html_report_meta(
            &summary.pillars,
            summary.title.clone(),
            summary.ticks,
            options.report_every,
            options.min_lifetime,
            summary.control,
            summary.total_time_seconds,
            generated_at_utc,
            "mean across seeds".to_owned(),
            seed_run_summaries
                .iter()
                .map(|seed_summary| PerSeedReportRow {
                    seed: seed_summary.seed,
                    total_time_seconds: seed_summary.total_time_seconds,
                    state_hash: seed_summary.state_hash.clone(),
                    report_href: format!("seed_{}/report.html", seed_summary.seed),
                })
                .collect(),
        ),
        &summary.timeseries,
    )?;

    Ok(summary)
}

pub(crate) fn run_single_seed_evaluation(
    config: WorldConfig,
    options: SeedRunOptions,
) -> Result<SeedEvaluationSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let mut reporter = Reporter::new(&options.out_dir)?;
    let mut sim = Simulation::new(config, options.seed)?;
    let mut ledger = Ledger::new(options.min_lifetime);

    for organism in sim.organisms() {
        ledger.birth(organism.id, 0, organism.genome.lifecycle.age_of_maturity);
    }

    let mut current_food_count = sim.snapshot().foods.len() as u64;
    let mut interval_births = 0_u64;
    let mut interval_deaths = 0_u64;
    let mut interval_consumptions = 0_u64;
    let mut interval_predations = 0_u64;
    let mut interval_population_exposure = 0_u64;
    let mut timeseries = Vec::new();

    for tick in 1..=options.ticks {
        interval_population_exposure =
            interval_population_exposure.saturating_add(sim.organisms().len() as u64);
        let delta = sim.tick();
        interval_consumptions =
            interval_consumptions.saturating_add(delta.metrics.consumptions_last_turn);
        interval_predations =
            interval_predations.saturating_add(delta.metrics.predations_last_turn);

        for record in sim.action_records().iter().flatten() {
            ledger.update(record);
        }
        for event in delta.reproduction_events.iter().copied() {
            ledger.handle_reproduction_event(tick, event);
        }

        interval_births = interval_births.saturating_add(delta.spawned.len() as u64);
        for spawned in &delta.spawned {
            ledger.birth(spawned.id, tick, spawned.genome.lifecycle.age_of_maturity);
        }
        ledger.update_survival_thresholds(sim.organisms());

        for removed in &delta.removed_positions {
            match removed.entity_id {
                EntityId::Organism(id) => {
                    interval_deaths = interval_deaths.saturating_add(1);
                    ledger.death(id, tick);
                }
                EntityId::Food(_) => {
                    current_food_count = current_food_count.saturating_sub(1);
                }
            }
        }
        current_food_count = current_food_count.saturating_add(delta.food_spawned.len() as u64);

        if tick % options.report_every == 0 || tick == options.ticks {
            let fraction = tick as f64 / options.ticks as f64;
            println!(
                "progress[seed={}]: {tick}/{total} ({fraction:.3})",
                options.seed,
                total = options.ticks
            );
            let interval = compute_interval_metrics(
                tick,
                delta.metrics.organisms,
                interval_births,
                interval_deaths,
                current_food_count,
                interval_consumptions,
                interval_predations,
                interval_population_exposure,
                ledger.recently_deceased(),
                sim.organisms(),
                ledger.interval_action_stats(),
                ledger.interval_generation_time(),
            );
            reporter.emit(&interval)?;
            timeseries.push(interval);

            interval_births = 0;
            interval_deaths = 0;
            interval_consumptions = 0;
            interval_predations = 0;
            interval_population_exposure = 0;
            ledger.clear_interval();
        }
    }

    reporter.flush()?;
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let pillars = compute_pillar_scores(&timeseries);
    let experiment_readouts = ledger.reproduction_analytics();

    let summary = SeedEvaluationSummary {
        title: options.title.clone(),
        seed: options.seed,
        ticks: options.ticks,
        control: options.control,
        total_time_seconds,
        pillars: pillars.clone(),
        experiment_readouts,
        state_hash: state_hash(sim.organisms()),
        timeseries,
    };
    write_summary_json(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &html_report_meta(
            &summary.pillars,
            summary.title.clone(),
            summary.ticks,
            options.report_every,
            options.min_lifetime,
            summary.control,
            summary.total_time_seconds,
            generated_at_utc,
            "per-seed timeseries".to_owned(),
            Vec::new(),
        ),
        &summary.timeseries,
    )?;

    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn html_report_meta(
    pillars: &crate::types::PillarScores,
    title: Option<String>,
    ticks: u64,
    report_every: u64,
    min_lifetime: u64,
    control: bool,
    total_time_seconds: f64,
    generated_at_utc: String,
    timeseries_label: String,
    per_seed_rows: Vec<PerSeedReportRow>,
) -> HtmlReportMeta {
    HtmlReportMeta {
        title,
        ticks,
        report_every,
        min_lifetime,
        control,
        total_time_seconds,
        generated_at_utc,
        pillar_window_start_tick: pillars.window_start_tick,
        pillar_window_end_tick: pillars.window_end_tick,
        viability_pillar: pillars.viability_pillar,
        foraging_pillar: pillars.foraging_pillar,
        intelligence_pillar: pillars.intelligence_pillar,
        competition_pillar: pillars.competition_pillar,
        adaptation_pillar: pillars.adaptation_pillar,
        viability_life_component: pillars.viability_life_component,
        viability_reproduction_component: pillars.viability_reproduction_component,
        foraging_p_fwd_food_component: pillars.foraging_p_fwd_food_component,
        foraging_rate_component: pillars.foraging_rate_component,
        intelligence_adult_mi_component: pillars.intelligence_adult_mi_component,
        intelligence_action_effectiveness_component: pillars
            .intelligence_action_effectiveness_component,
        intelligence_anti_idle_component: pillars.intelligence_anti_idle_component,
        intelligence_util_component: pillars.intelligence_util_component,
        competition_predation_component: pillars.competition_predation_component,
        competition_attack_success_component: pillars.competition_attack_success_component,
        competition_attack_attempt_component: pillars.competition_attack_attempt_component,
        adaptation_juvenile_mi_component: pillars.adaptation_juvenile_mi_component,
        adaptation_diversity_component: pillars.adaptation_diversity_component,
        timeseries_label,
        per_seed_rows,
    }
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
/// runs — empirically, throughput plateaus around 4 threads for a single 5000-
/// organism simulation and excess workers add rayon scheduling overhead.
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

        let summary_a =
            run_single_seed_evaluation(cfg.clone(), options_a).expect("first run should succeed");
        let summary_b =
            run_single_seed_evaluation(cfg, options_b).expect("second run should succeed");

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
