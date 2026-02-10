use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sim_core::Simulation;
use sim_protocol::WorldConfig;

fn bench_1000_turns(c: &mut Criterion) {
    c.bench_function("1000 turns (seed 42)", |b| {
        b.iter_batched(
            || Simulation::new(WorldConfig::default(), 42).expect("simulation init"),
            |mut sim| black_box(sim.step_n(1000)),
            criterion::BatchSize::PerIteration,
        );
    });
}

criterion_group!(benches, bench_1000_turns);
criterion_main!(benches);
