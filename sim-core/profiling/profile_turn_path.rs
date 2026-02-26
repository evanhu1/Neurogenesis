#[cfg(feature = "profiling")]
use sim_core::profiling::{self, PhaseCounterSnapshot, ProfilingSnapshot};
#[cfg(feature = "profiling")]
use sim_core::Simulation;
#[cfg(feature = "profiling")]
use sim_types::{SeedGenomeConfig, WorldConfig};
#[cfg(feature = "profiling")]
use std::time::{Duration, Instant};

#[cfg(feature = "profiling")]
#[derive(Clone, Copy)]
struct Args {
    turns: u32,
    warmup_turns: u32,
    seed: u64,
}

#[cfg(feature = "profiling")]
fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "profiling"))]
fn main() {
    eprintln!(
        "This example requires the `profiling` feature.\n\
         Run: cargo run -p sim-core --release --features profiling --example profile_turn_path -- --help"
    );
    std::process::exit(1);
}

#[cfg(feature = "profiling")]
fn run() -> Result<(), String> {
    let args = parse_args()?;
    let mut sim = Simulation::new(stable_perf_config(), args.seed)
        .map_err(|err| format!("failed to initialize simulation: {err}"))?;

    if args.warmup_turns > 0 {
        sim.advance_n(args.warmup_turns);
    }

    profiling::reset();
    let started = Instant::now();
    sim.advance_n(args.turns);
    let elapsed = started.elapsed();
    let snapshot = profiling::snapshot();

    print_report(args, &snapshot, elapsed);
    Ok(())
}

#[cfg(feature = "profiling")]
fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        turns: 1_000,
        warmup_turns: 200,
        seed: 42,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--turns" => args.turns = parse_value(&mut iter, "--turns")?,
            "--warmup-turns" => args.warmup_turns = parse_value(&mut iter, "--warmup-turns")?,
            "--seed" => args.seed = parse_value(&mut iter, "--seed")?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("unknown flag `{flag}`. Use `--help` for usage."));
            }
        }
    }

    if args.turns == 0 {
        return Err("`--turns` must be >= 1".to_owned());
    }

    Ok(args)
}

#[cfg(feature = "profiling")]
fn parse_value<T>(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = iter
        .next()
        .ok_or_else(|| format!("missing value for `{flag}`"))?;
    raw.parse::<T>()
        .map_err(|err| format!("invalid value `{raw}` for `{flag}`: {err}"))
}

#[cfg(feature = "profiling")]
fn print_help() {
    println!("Profiles the per-turn hot path with phase-level and brain-stage timing.");
    println!();
    println!("Usage:");
    println!(
        "  cargo run -p sim-core --release --features profiling --example profile_turn_path -- [options]"
    );
    println!();
    println!("Options:");
    println!("  --turns <u32>                Number of measured turns (default: 1000)");
    println!("  --warmup-turns <u32>         Warmup turns before reset (default: 200)");
    println!("  --seed <u64>                 Simulation seed (default: 42)");
}

#[cfg(feature = "profiling")]
fn print_report(args: Args, snapshot: &ProfilingSnapshot, wall_elapsed: Duration) {
    let tick_ns = snapshot.tick_total.total_ns.max(1);
    let wall_ns = duration_to_ns(wall_elapsed);

    println!("== sim-core turn profile ==");
    println!("turns: {}", args.turns);
    println!("warmup_turns: {}", args.warmup_turns);
    println!("seed: {}", args.seed);
    println!("wall_time_ms: {:.3}", ns_to_ms(wall_ns));
    println!(
        "avg_wall_us_per_turn: {:.3}",
        wall_ns as f64 / args.turns as f64 / 1_000.0
    );
    println!(
        "instrumented_tick_total_ms: {:.3}",
        ns_to_ms(snapshot.tick_total.total_ns)
    );
    println!();

    let mut turn_rows = vec![
        ("intents", snapshot.intents),
        ("commit", snapshot.commit),
        ("reproduction", snapshot.reproduction),
        ("move_resolution", snapshot.move_resolution),
        ("snapshot", snapshot.snapshot),
        ("lifecycle", snapshot.lifecycle),
        ("spawn", snapshot.spawn),
        ("prune_species", snapshot.prune_species),
        ("consistency_check", snapshot.consistency_check),
        ("metrics_and_delta", snapshot.metrics_and_delta),
        ("age", snapshot.age),
    ];
    turn_rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

    println!("-- Turn Phase Breakdown (sorted by total time) --");
    println!(
        "{:<24} {:>12} {:>10} {:>14}",
        "phase", "total_ms", "%tick", "avg_us/call"
    );
    for (label, counter) in turn_rows {
        println!(
            "{:<24} {:>12.3} {:>9.2}% {:>14.3}",
            label,
            ns_to_ms(counter.total_ns),
            pct(counter.total_ns, tick_ns),
            avg_us(counter)
        );
    }
    println!();

    println!("-- Brain Totals --");
    println!(
        "{:<24} {:>12} {:>10} {:>14} {:>10}",
        "section", "total_ms", "%intents", "avg_us/call", "calls"
    );
    println!(
        "{:<24} {:>12.3} {:>9.2}% {:>14.3} {:>10}",
        "evaluate_brain_total",
        ns_to_ms(snapshot.brain_eval_total.total_ns),
        pct(
            snapshot.brain_eval_total.total_ns,
            snapshot.intents.total_ns.max(1)
        ),
        avg_us(snapshot.brain_eval_total),
        snapshot.brain_eval_total.calls
    );
    println!(
        "{:<24} {:>12.3} {:>9.2}% {:>14.3} {:>10}",
        "apply_plasticity_total",
        ns_to_ms(snapshot.brain_plasticity_total.total_ns),
        pct(
            snapshot.brain_plasticity_total.total_ns,
            snapshot.intents.total_ns.max(1)
        ),
        avg_us(snapshot.brain_plasticity_total),
        snapshot.brain_plasticity_total.calls
    );
    println!();

    let mut brain_rows = vec![
        ("inter_accumulation", snapshot.brain_inter_accumulation),
        ("action_accumulation", snapshot.brain_action_accumulation),
        (
            "plasticity_inter_tuning",
            snapshot.brain_plasticity_inter_tuning,
        ),
        (
            "plasticity_sensory_tuning",
            snapshot.brain_plasticity_sensory_tuning,
        ),
        ("inter_activation", snapshot.brain_inter_activation),
        (
            "action_activation_resolve",
            snapshot.brain_action_activation_resolve,
        ),
        ("scan_ahead", snapshot.brain_scan_ahead),
        ("sensory_encoding", snapshot.brain_sensory_encoding),
        ("inter_setup", snapshot.brain_inter_setup),
        ("plasticity_setup", snapshot.brain_plasticity_setup),
        ("plasticity_prune", snapshot.brain_plasticity_prune),
    ];
    brain_rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

    println!("-- Brain Stage Breakdown (sorted by total time) --");
    println!(
        "{:<28} {:>12} {:>10} {:>14} {:>10}",
        "stage", "total_ms", "%brain", "avg_us/call", "calls"
    );
    let brain_total_ns = brain_rows
        .iter()
        .map(|(_, counter)| counter.total_ns)
        .sum::<u64>()
        .max(1);
    for (label, counter) in brain_rows {
        println!(
            "{:<28} {:>12.3} {:>9.2}% {:>14.3} {:>10}",
            label,
            ns_to_ms(counter.total_ns),
            pct(counter.total_ns, brain_total_ns),
            avg_us(counter),
            counter.calls
        );
    }
}

#[cfg(feature = "profiling")]
fn avg_us(counter: PhaseCounterSnapshot) -> f64 {
    if counter.calls == 0 {
        return 0.0;
    }
    counter.total_ns as f64 / counter.calls as f64 / 1_000.0
}

#[cfg(feature = "profiling")]
fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

#[cfg(feature = "profiling")]
fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 * 100.0 / whole as f64
}

#[cfg(feature = "profiling")]
fn duration_to_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

#[cfg(feature = "profiling")]
fn stable_perf_config() -> WorldConfig {
    WorldConfig {
        world_width: 100,
        steps_per_second: 5,
        num_organisms: 2_000,
        periodic_injection_interval_turns: 0,
        periodic_injection_count: 0,
        food_energy: 50.0,
        move_action_energy_cost: 0.0,
        action_temperature: 0.5,
        plant_growth_speed: 1.0,
        food_regrowth_interval: 10,
        food_fertility_noise_scale: 0.045,
        food_fertility_exponent: 1.8,
        food_fertility_floor: 0.04,
        terrain_noise_scale: 0.02,
        terrain_threshold: 1.0,
        max_organism_age: u32::MAX,
        global_mutation_rate_modifier: 1.0,
        runtime_plasticity_enabled: true,
        seed_genome_config: SeedGenomeConfig {
            num_neurons: 20,
            num_synapses: 80,
            spatial_prior_sigma: 3.5,
            vision_distance: 10,
            starting_energy: 1_000_000_000.0,
            age_of_maturity: 0,
            hebb_eta_gain: 0.0,
            eligibility_retention: 0.9,
            synapse_prune_threshold: 0.01,
            mutation_rate_age_of_maturity: 0.05,
            mutation_rate_vision_distance: 0.04,
            mutation_rate_inter_bias: 0.2,
            mutation_rate_inter_update_rate: 0.12,
            mutation_rate_action_bias: 0.2,
            mutation_rate_eligibility_retention: 0.05,
            mutation_rate_synapse_prune_threshold: 0.05,
            mutation_rate_neuron_location: 0.02,
            mutation_rate_synapse_weight_perturbation: 0.1,
            mutation_rate_add_neuron_split_edge: 0.05,
        },
    }
}
