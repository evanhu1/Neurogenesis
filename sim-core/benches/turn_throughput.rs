use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sim_core::Simulation;
use sim_types::WorldConfig;

fn bench_1000_turns(c: &mut Criterion) {
    let config = WorldConfig::perf_fixture();
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

fn bench_1000_turns_headless(c: &mut Criterion) {
    let config = WorldConfig::perf_fixture();
    c.bench_function(
        "turn throughput / 1000 turns headless (advance_n, seed 42)",
        |b| {
            b.iter_batched(
                || Simulation::new(config.clone(), 42).expect("simulation init"),
                |mut sim| {
                    sim.advance_n(1000);
                    black_box(&sim);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
}

fn bench_1000_turns_single_thread(c: &mut Criterion) {
    let mut config = WorldConfig::perf_fixture();
    config.intent_parallel_threads = 1;
    c.bench_function(
        "turn throughput / 1000 turns single-thread (seed 42)",
        |b| {
            b.iter_batched(
                || Simulation::new(config.clone(), 42).expect("simulation init"),
                |mut sim| {
                    sim.advance_n(1000);
                    black_box(&sim);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(
    benches,
    bench_1000_turns,
    bench_1000_turns_headless,
    bench_1000_turns_single_thread,
);
criterion_main!(benches);
