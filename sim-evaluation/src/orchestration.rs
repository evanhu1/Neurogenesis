use crate::{
    aggregation::{
        average_aggregate_scores, average_reproduction_analytics, average_timeseries,
        compute_aggregate_score, state_hash,
    },
    ledger::{Ledger, N_ACTIONS},
    metrics::{compute_interval_metrics, jensen_shannon_divergence},
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

    for _ in 0..worker_threads {
        let config = config.clone();
        let seed_queue = Arc::clone(&seed_queue);
        let tx = tx.clone();
        let out_dir = options.out_dir.clone();
        let title = options.title.clone();
        let ticks = options.ticks;
        let report_every = options.report_every;
        let min_lifetime = options.min_lifetime;
        let baseline = options.baseline;

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
                baseline,
                reward_reversal_tick: reward_reversal_tick_for_run(ticks),
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

    let aggregate_score = average_aggregate_scores(&seed_summaries);
    let experiment_readouts = average_reproduction_analytics(&seed_summaries);
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();

    let seed_run_summaries = seed_summaries
        .iter()
        .map(|summary| SeedRunSummary {
            seed: summary.seed,
            out_dir: PathBuf::from(format!("seed_{}", summary.seed)),
            total_time_seconds: summary.total_time_seconds,
            aggregate_score: summary.aggregate_score.clone(),
            experiment_readouts: summary.experiment_readouts.clone(),
            state_hash: summary.state_hash.clone(),
        })
        .collect::<Vec<_>>();

    let summary = EvaluationSummary {
        title: options.title.clone(),
        seeds: options.seeds.clone(),
        ticks: options.ticks,
        baseline: options.baseline,
        worker_threads,
        total_time_seconds,
        aggregate_score: aggregate_score.clone(),
        experiment_readouts,
        seed_summaries: seed_run_summaries.clone(),
        timeseries: averaged_timeseries.clone(),
    };

    write_summary_json(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &HtmlReportMeta {
            title: summary.title.clone(),
            seed_count: summary.seeds.len(),
            ticks: summary.ticks,
            report_every: options.report_every,
            min_lifetime: options.min_lifetime,
            baseline: summary.baseline,
            total_time_seconds: summary.total_time_seconds,
            generated_at_utc,
            aggregate_score: summary.aggregate_score.score,
            aggregate_score_median: summary.aggregate_score.score_median,
            aggregate_score_stddev: summary.aggregate_score.score_stddev,
            aggregate_score_min: summary.aggregate_score.score_min,
            aggregate_score_max: summary.aggregate_score.score_max,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_viability_pillar: summary.aggregate_score.viability_pillar,
            aggregate_foraging_pillar: summary.aggregate_score.foraging_pillar,
            aggregate_control_pillar: summary.aggregate_score.control_pillar,
            aggregate_competition_pillar: summary.aggregate_score.competition_pillar,
            aggregate_adaptation_pillar: summary.aggregate_score.adaptation_pillar,
            aggregate_viability_life_component: summary.aggregate_score.viability_life_component,
            aggregate_viability_reproduction_component: summary
                .aggregate_score
                .viability_reproduction_component,
            aggregate_viability_damage_component: summary
                .aggregate_score
                .viability_damage_component,
            aggregate_foraging_p_fwd_food_component: summary
                .aggregate_score
                .foraging_p_fwd_food_component,
            aggregate_foraging_rate_component: summary.aggregate_score.foraging_rate_component,
            aggregate_control_adult_mi_component: summary
                .aggregate_score
                .control_adult_mi_component,
            aggregate_control_entropy_component: summary.aggregate_score.control_entropy_component,
            aggregate_control_anti_idle_component: summary
                .aggregate_score
                .control_anti_idle_component,
            aggregate_control_util_component: summary.aggregate_score.control_util_component,
            aggregate_competition_predation_component: summary
                .aggregate_score
                .competition_predation_component,
            aggregate_competition_attack_success_component: summary
                .aggregate_score
                .competition_attack_success_component,
            aggregate_competition_attack_attempt_component: summary
                .aggregate_score
                .competition_attack_attempt_component,
            aggregate_adaptation_reversal_component: summary
                .aggregate_score
                .adaptation_reversal_component,
            aggregate_adaptation_juvenile_mi_component: summary
                .aggregate_score
                .adaptation_juvenile_mi_component,
            aggregate_adaptation_diversity_component: summary
                .aggregate_score
                .adaptation_diversity_component,
            timeseries_label: "mean across seeds".to_owned(),
            per_seed_rows: seed_run_summaries
                .iter()
                .map(|seed_summary| PerSeedReportRow {
                    seed: seed_summary.seed,
                    score: seed_summary.aggregate_score.score,
                    total_time_seconds: seed_summary.total_time_seconds,
                    state_hash: seed_summary.state_hash.clone(),
                    report_href: format!("seed_{}/report.html", seed_summary.seed),
                })
                .collect(),
        },
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
        ledger.birth(organism.id, 0, organism.genome.age_of_maturity);
    }

    let mut current_food_count = sim.snapshot().foods.len() as u64;
    let mut interval_births = 0_u64;
    let mut interval_deaths = 0_u64;
    let mut interval_consumptions = 0_u64;
    let mut interval_predations = 0_u64;
    let mut interval_population_exposure = 0_u64;
    let mut pre_reversal_histogram: Option<[f64; N_ACTIONS]> = None;
    let mut timeseries = Vec::new();

    for tick in 1..=options.ticks {
        if options
            .reward_reversal_tick
            .is_some_and(|reversal_tick| tick > reversal_tick)
        {
            sim.set_reward_signal_multiplier(-1.0);
        }
        interval_population_exposure =
            interval_population_exposure.saturating_add(sim.organisms().len() as u64);
        let delta = sim.tick();
        let records = sim.drain_action_records();
        interval_consumptions =
            interval_consumptions.saturating_add(delta.metrics.consumptions_last_turn);
        interval_predations =
            interval_predations.saturating_add(delta.metrics.predations_last_turn);

        for record in records {
            ledger.update(record);
        }
        for event in delta.reproduction_events.iter().copied() {
            ledger.handle_reproduction_event(tick, event);
        }

        interval_births = interval_births.saturating_add(delta.spawned.len() as u64);
        for spawned in &delta.spawned {
            ledger.birth(spawned.id, tick, spawned.genome.age_of_maturity);
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
            let mut interval = compute_interval_metrics(
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
                sim.config().food_energy,
            );
            if options
                .reward_reversal_tick
                .is_some_and(|reversal_tick| tick <= reversal_tick)
            {
                pre_reversal_histogram = Some(interval.action_histogram);
            } else if let Some(reference) = pre_reversal_histogram.as_ref() {
                interval.reward_reversal_shift =
                    jensen_shannon_divergence(&interval.action_histogram, reference);
            }
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
    let aggregate_score = compute_aggregate_score(&timeseries, options.reward_reversal_tick);
    let experiment_readouts = ledger.reproduction_analytics();

    let summary = SeedEvaluationSummary {
        title: options.title.clone(),
        seed: options.seed,
        ticks: options.ticks,
        baseline: options.baseline,
        total_time_seconds,
        aggregate_score: aggregate_score.clone(),
        experiment_readouts,
        state_hash: state_hash(sim.organisms()),
        timeseries,
    };
    write_summary_json(&options.out_dir, &summary)?;
    write_html_report(
        &options.out_dir,
        &HtmlReportMeta {
            title: summary.title.clone(),
            seed_count: 1,
            ticks: summary.ticks,
            report_every: options.report_every,
            min_lifetime: options.min_lifetime,
            baseline: summary.baseline,
            total_time_seconds: summary.total_time_seconds,
            generated_at_utc,
            aggregate_score: summary.aggregate_score.score,
            aggregate_score_median: summary.aggregate_score.score_median,
            aggregate_score_stddev: summary.aggregate_score.score_stddev,
            aggregate_score_min: summary.aggregate_score.score_min,
            aggregate_score_max: summary.aggregate_score.score_max,
            aggregate_window_start_tick: summary.aggregate_score.window_start_tick,
            aggregate_window_end_tick: summary.aggregate_score.window_end_tick,
            aggregate_viability_pillar: summary.aggregate_score.viability_pillar,
            aggregate_foraging_pillar: summary.aggregate_score.foraging_pillar,
            aggregate_control_pillar: summary.aggregate_score.control_pillar,
            aggregate_competition_pillar: summary.aggregate_score.competition_pillar,
            aggregate_adaptation_pillar: summary.aggregate_score.adaptation_pillar,
            aggregate_viability_life_component: summary.aggregate_score.viability_life_component,
            aggregate_viability_reproduction_component: summary
                .aggregate_score
                .viability_reproduction_component,
            aggregate_viability_damage_component: summary
                .aggregate_score
                .viability_damage_component,
            aggregate_foraging_p_fwd_food_component: summary
                .aggregate_score
                .foraging_p_fwd_food_component,
            aggregate_foraging_rate_component: summary.aggregate_score.foraging_rate_component,
            aggregate_control_adult_mi_component: summary
                .aggregate_score
                .control_adult_mi_component,
            aggregate_control_entropy_component: summary.aggregate_score.control_entropy_component,
            aggregate_control_anti_idle_component: summary
                .aggregate_score
                .control_anti_idle_component,
            aggregate_control_util_component: summary.aggregate_score.control_util_component,
            aggregate_competition_predation_component: summary
                .aggregate_score
                .competition_predation_component,
            aggregate_competition_attack_success_component: summary
                .aggregate_score
                .competition_attack_success_component,
            aggregate_competition_attack_attempt_component: summary
                .aggregate_score
                .competition_attack_attempt_component,
            aggregate_adaptation_reversal_component: summary
                .aggregate_score
                .adaptation_reversal_component,
            aggregate_adaptation_juvenile_mi_component: summary
                .aggregate_score
                .adaptation_juvenile_mi_component,
            aggregate_adaptation_diversity_component: summary
                .aggregate_score
                .adaptation_diversity_component,
            timeseries_label: "per-seed timeseries".to_owned(),
            per_seed_rows: Vec::new(),
        },
        &summary.timeseries,
    )?;

    Ok(summary)
}

pub(crate) fn reward_reversal_tick_for_run(ticks: u64) -> Option<u64> {
    if ticks < 2 {
        None
    } else {
        Some((ticks / 2).max(1))
    }
}

fn default_worker_threads(seed_count: usize) -> usize {
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, seed_count.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::WorldConfig;

    #[test]
    fn same_seed_yields_same_summary_hash() {
        let mut cfg = WorldConfig::default();
        cfg.world_width = 40;
        cfg.num_organisms = 300;
        cfg.periodic_injection_interval_turns = 0;
        cfg.periodic_injection_count = 0;
        cfg.force_random_actions = false;

        let out_a = test_output_dir("a");
        let out_b = test_output_dir("b");
        let options_a = SeedRunOptions {
            seed: 2026,
            ticks: 100,
            report_every: 50,
            min_lifetime: 10,
            out_dir: out_a.clone(),
            title: None,
            baseline: false,
            reward_reversal_tick: reward_reversal_tick_for_run(100),
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
            serde_json::to_string(&summary_a.timeseries).expect("serialize first timeseries"),
            serde_json::to_string(&summary_b.timeseries).expect("serialize second timeseries")
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
}
