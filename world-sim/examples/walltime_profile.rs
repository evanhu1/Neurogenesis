use std::time::Instant;
use types::WorldConfig;
use world_sim::Simulation;

fn main() {
    let threads: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut config = WorldConfig::perf_fixture();
    config.intent_parallel_threads = threads;
    let mut sim = Simulation::new(config, 42).expect("init");
    sim.advance_n(200); // warmup
    let started = Instant::now();
    sim.advance_n(10000);
    let elapsed = started.elapsed();
    eprintln!(
        "10000 turns ({} threads): {:.1}ms ({:.1} us/turn)",
        threads,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_micros() as f64 / 10000.0
    );
}
