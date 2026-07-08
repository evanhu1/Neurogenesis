//! Decoupling proof: the Chemotaxis Ribbon runs on the **unchanged**
//! `sim-substrate`. A living, reproducing population on a non-hex world with a
//! different sensor/actuator/morphology schema demonstrates the substrate
//! carries no environment assumptions.

use sim_substrate::seed::seed_genome;
use sim_substrate::{DriverConfig, PopulationDriver};
use sim_toyenv::ChemotaxisRibbon;

fn run(seed: u64, ticks: u64) -> (usize, u64, u64) {
    let mut env = ChemotaxisRibbon::new(256, seed);
    let genome = seed_genome(env_catalog(&env));
    let baseline = genome.header.mutation_rates;
    let config = DriverConfig {
        founder_energy: 200.0,
        ..DriverConfig::default()
    };
    let mut driver = PopulationDriver::new(seed, baseline, config);
    driver.seed_population(&mut env, 120, &genome);
    for t in 0..ticks {
        driver.tick(&mut env);
        if t % 200 == 0 {
            eprintln!("  t={t:5} alive={:4} born={}", driver.alive_count(), driver.total_ever());
        }
    }
    let mut fp: u64 = 0x5151_5151;
    let mut alive = 0usize;
    for b in &driver.bodies {
        if !b.alive {
            continue;
        }
        alive += 1;
        fp = sim_substrate::rng::mix_u64(
            fp ^ b.id ^ ((b.energy * 100.0) as i64 as u64).rotate_left(21),
        );
    }
    (alive, driver.total_ever(), fp)
}

fn env_catalog(env: &ChemotaxisRibbon) -> &sim_substrate::SubstrateCatalog {
    use sim_substrate::Environment;
    env.catalog()
}

fn main() {
    let ticks = 2000;
    let (alive, born, fp) = run(9, ticks);
    let (alive2, born2, fp2) = run(9, ticks);
    println!("run A: alive={alive} born={born} fp={fp:#018x}");
    println!("run B: alive={alive2} born={born2} fp={fp2:#018x}");
    assert_eq!((alive, born, fp), (alive2, born2, fp2), "determinism violated on toy env");
    assert!(alive > 0, "toy-env population went extinct");
    assert!(born > 120, "no births in toy env");
    println!("OK: decoupling proven — Chemotaxis Ribbon lives on unchanged sim-substrate ({alive} alive, {born} born)");
}
