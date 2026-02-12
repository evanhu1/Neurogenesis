use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sim_core::Simulation;
use sim_types::WorldConfig;

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
        seed_genome_config: sim_types::SeedGenomeConfig {
            num_neurons: 20,
            num_synapses: 80,
            vision_distance: 10,
            mutation_rate_vision_distance: 0.04,
            mutation_rate_weight: 0.25,
            mutation_rate_add_edge: 0.03,
            mutation_rate_remove_edge: 0.02,
            mutation_rate_split_edge: 0.005,
            mutation_rate_inter_bias: 0.2,
            mutation_rate_inter_update_rate: 0.12,
            mutation_rate_action_bias: 0.2,
        },
    }
}

fn bench_1000_turns(c: &mut Criterion) {
    let config = stable_perf_config();
    c.bench_function(
        "turn throughput / 1000 turns (stable workload, seed 42)",
        |b| {
            b.iter_batched(
                || Simulation::new(config.clone(), 42).expect("simulation init"),
                |mut sim| black_box(sim.step_n(1000)),
                criterion::BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, bench_1000_turns);
criterion_main!(benches);
