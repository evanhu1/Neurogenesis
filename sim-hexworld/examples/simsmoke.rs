//! Smoke test for the `HexSim` wrapper: tick, bincode round-trip, resume, and
//! extinction termination (no periodic injection — a dead world stays dead).

use sim_hexworld::{HexConfig, HexSim};

fn main() {
    // 1. Healthy world ticks and self-sustains; save/load round-trips exactly.
    let mut sim = HexSim::new(HexConfig::default(), 42);
    for _ in 0..500 {
        sim.tick();
    }
    let bytes = bincode::serialize(&sim).expect("serialize HexSim");
    let mut restored: HexSim = bincode::deserialize(&bytes).expect("deserialize HexSim");
    assert_eq!(restored.turn(), sim.turn());
    assert_eq!(restored.alive_count(), sim.alive_count());
    // Continue both; they must stay identical (determinism across save/load).
    for _ in 0..200 {
        sim.tick();
        restored.tick();
    }
    assert_eq!(sim.turn(), restored.turn());
    assert_eq!(sim.alive_count(), restored.alive_count());
    assert!(sim.alive_count() > 0, "healthy world should still be alive");
    println!(
        "healthy: turn={} alive={} (save/load identical)",
        sim.turn(),
        sim.alive_count()
    );

    // 2. Hostile world: few founders, no injection — must go extinct and stop.
    let hostile = HexConfig {
        world_width: 8,
        num_founders: 3,
        founder_energy: 20.0,
        ..HexConfig::default()
    };
    let mut doomed = HexSim::new(hostile, 1);
    let mut ticks = 0u64;
    while doomed.tick() {
        ticks += 1;
        if ticks > 100_000 {
            panic!("hostile world never went extinct");
        }
    }
    assert!(doomed.is_extinct(), "world should be extinct");
    assert_eq!(doomed.extinct_at, Some(doomed.turn()));
    let t = doomed.turn();
    assert!(!doomed.tick());
    assert_eq!(doomed.turn(), t, "extinct world must not advance");
    println!("hostile: extinct at turn {} (no injection, stays dead)", doomed.turn());
    println!("OK: HexSim ticks, round-trips, and terminates on extinction");
}
