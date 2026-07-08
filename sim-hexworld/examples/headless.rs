//! Phase-2 proof-of-life: tick the real hex ecology, driven by the substrate
//! brain, with embodied sexual mating. A living, reproducing population must
//! persist deterministically across identical-seed runs.

use sim_hexworld::catalog::build_catalog;
use sim_hexworld::HexWorld;
use sim_substrate::seed::seed_genome;
use sim_substrate::{DriverConfig, PopulationDriver};

fn run(seed: u64, ticks: u64) -> (usize, u64, u64) {
    let mut env = HexWorld::new(32, seed);
    let catalog = build_catalog();
    let genome = seed_genome(&catalog);
    let baseline = genome.header.mutation_rates;
    let config = DriverConfig {
        founder_energy: 400.0,
        ..DriverConfig::default()
    };
    let mut driver = PopulationDriver::new(seed, baseline, config);
    driver.seed_population(&mut env, 200, &genome);

    for t in 0..ticks {
        driver.tick(&mut env);
        if t % 200 == 0 {
            eprintln!(
                "  t={t:5} alive={:4} born={}",
                driver.alive_count(),
                driver.total_ever()
            );
        }
    }

    let mut fp: u64 = 0x1234_ABCD;
    let mut alive = 0usize;
    for b in &driver.bodies {
        if !b.alive {
            continue;
        }
        alive += 1;
        let q = (b.energy * 100.0) as i64 as u64;
        fp = sim_substrate::rng::mix_u64(fp ^ b.id ^ q.rotate_left(21));
    }
    (alive, driver.total_ever(), fp)
}

fn main() {
    let ticks = 2000;
    let (alive, born, fp) = run(42, ticks);
    let (alive2, born2, fp2) = run(42, ticks);
    println!("run A: alive={alive} born={born} fp={fp:#018x}");
    println!("run B: alive={alive2} born={born2} fp={fp2:#018x}");
    assert_eq!(
        (alive, born, fp),
        (alive2, born2, fp2),
        "determinism violated on the hex world"
    );
    assert!(alive > 0, "hex population went extinct");
    assert!(born > 200, "no births occurred in the hex world");
    println!("OK: hex proof-of-life — {alive} alive after {ticks} ticks, {born} ever born (deterministic)");
}
