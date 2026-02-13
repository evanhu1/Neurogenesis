use super::*;
use std::time::{Duration, Instant};

const PERF_SEED: u64 = 42;
const WARMUP_SAMPLES: usize = 3;
const MEASURED_SAMPLES: usize = 9;
const STEADY_STATE_WARMUP_TURNS: u32 = 400;
const SHORT_SAMPLE_TURNS: u32 = 300;
const LONG_SAMPLE_TURNS: u32 = 600;
const DEFAULT_BUDGET_NS_PER_TURN: f64 = 130_000.0;

fn stable_perf_config() -> WorldConfig {
    WorldConfig {
        world_width: 100,
        steps_per_second: 5,
        num_organisms: 2_000,
        starting_energy: 10_000.0,
        food_energy: 50.0,
        reproduction_energy_cost: 1_000_000_000.0,
        move_action_energy_cost: 0.0,
        turn_energy_cost: 0.0,
        food_coverage_divisor: u32::MAX,
        max_organism_age: u32::MAX,
        max_num_neurons: 50,
        speciation_threshold: 50.0,
        seed_genome_config: SeedGenomeConfig {
            num_neurons: 20,
            num_synapses: 80,
            vision_distance: 10,
            hebb_eta_baseline: 0.0,
            hebb_eta_gain: 0.0,
            eligibility_decay_lambda: 0.9,
            synapse_prune_threshold: 0.01,
            mutation_rate_vision_distance: 0.04,
            mutation_rate_weight: 0.25,
            mutation_rate_add_edge: 0.03,
            mutation_rate_remove_edge: 0.02,
            mutation_rate_split_edge: 0.005,
            mutation_rate_inter_bias: 0.2,
            mutation_rate_inter_update_rate: 0.12,
            mutation_rate_action_bias: 0.2,
            mutation_rate_eligibility_decay_lambda: 0.05,
            mutation_rate_synapse_prune_threshold: 0.05,
        },
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn median_duration_seconds(config: &WorldConfig, turns: u32) -> f64 {
    let total = WARMUP_SAMPLES + MEASURED_SAMPLES;
    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
    for sample_idx in 0..total {
        let mut sim = Simulation::new(config.clone(), PERF_SEED).expect("simulation init");
        let _ = sim.step_n(STEADY_STATE_WARMUP_TURNS);
        let start = Instant::now();
        let _ = sim.step_n(turns);
        let elapsed = start.elapsed().as_secs_f64();
        if sample_idx >= WARMUP_SAMPLES {
            samples.push(elapsed);
        }
    }
    median(&mut samples)
}

fn median_ns_per_turn(config: &WorldConfig, turns: u32) -> f64 {
    median_duration_seconds(config, turns) * 1_000_000_000.0 / turns as f64
}

#[test]
#[ignore = "performance-sensitive; run with --release --ignored"]
fn turn_tick_scales_linearly_with_turn_count() {
    let cfg = stable_perf_config();
    let short = median_duration_seconds(&cfg, SHORT_SAMPLE_TURNS);
    let long = median_duration_seconds(&cfg, LONG_SAMPLE_TURNS);
    let ratio = long / short;

    assert!(
        (1.65..=2.30).contains(&ratio),
        "tick scaling regressed: short={:.3?} long={:.3?} ratio={ratio:.3}",
        Duration::from_secs_f64(short),
        Duration::from_secs_f64(long),
    );
}

#[test]
#[ignore = "performance-sensitive; run with --release --ignored"]
fn turn_tick_median_ns_per_turn_within_budget() {
    if cfg!(debug_assertions) {
        panic!("run this performance test with --release");
    }

    let budget_ns_per_turn = std::env::var("SIM_CORE_TICK_BUDGET_NS_PER_TURN")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_BUDGET_NS_PER_TURN);

    let measured = median_ns_per_turn(&stable_perf_config(), SHORT_SAMPLE_TURNS);
    assert!(
        measured <= budget_ns_per_turn,
        "tick throughput regressed: measured={measured:.0}ns/turn budget={budget_ns_per_turn:.0}ns/turn (override with SIM_CORE_TICK_BUDGET_NS_PER_TURN)"
    );
}
