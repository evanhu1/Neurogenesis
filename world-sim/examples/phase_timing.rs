use std::time::Instant;
use types::WorldConfig;
use world_sim::Simulation;

fn main() {
    let config = WorldConfig::perf_fixture();
    let mut sim = Simulation::new(config, 42).expect("init");
    sim.advance_n(200); // warmup

    eprintln!("Organisms after warmup: {}", sim.metrics().organisms);

    // Measure total
    let started = Instant::now();
    sim.advance_n(1000);
    let elapsed = started.elapsed();
    eprintln!(
        "1000 turns: {:.1}ms ({:.1} us/turn)",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_micros() as f64 / 1000.0
    );
    eprintln!("Organisms after run: {}", sim.metrics().organisms);
}
