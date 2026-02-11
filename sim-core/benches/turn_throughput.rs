use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sim_core::Simulation;
use sim_types::WorldConfig;

fn bench_500_turns(c: &mut Criterion) {
    c.bench_function("500 turns (seed 42)", |b| {
        b.iter_batched(
            || Simulation::new(WorldConfig::default(), 42).expect("simulation init"),
            |mut sim| black_box(sim.step_n(500)),
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_500_turns);
criterion_main!(benches);
