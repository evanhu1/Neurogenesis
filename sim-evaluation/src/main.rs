//! sim-evaluation — headless multi-seed evolution harness for the new substrate.
//!
//! With no fitness function, progress is measured by **Quality-Diversity**:
//! after running each seed to a tick budget (or extinction), we build a
//! MAP-Elites archive over a behavior descriptor and report coverage / QD-score
//! alongside population and lineage stats. There is no periodic injection — a
//! seed that goes extinct ends there (its extinction turn is recorded).
//!
//! Usage: sim-evaluation [--seeds a,b,c] [--ticks N] [--width W] [--founders F]
//!                       [--out summary.json]

use sim_hexworld::{HexConfig, HexSim};
use sim_substrate::QdArchive;

const DEFAULT_SEEDS: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const QD_RESOLUTION: usize = 8;

#[derive(serde::Serialize)]
struct SeedReport {
    seed: u64,
    final_turn: u64,
    alive: usize,
    total_ever: u64,
    extinct_at: Option<u64>,
    max_generation: u64,
    mean_neurons: f32,
    mean_edges: f32,
    qd_coverage: usize,
    qd_score: f32,
}

#[derive(serde::Serialize)]
struct Summary {
    ticks: u64,
    seeds: Vec<SeedReport>,
    mean_alive: f32,
    mean_qd_coverage: f32,
    extinctions: usize,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags = parse_flags(&args);

    let seeds: Vec<u64> = flags
        .get("seeds")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| DEFAULT_SEEDS.to_vec());
    let ticks: u64 = flags.get("ticks").and_then(|s| s.parse().ok()).unwrap_or(5000);
    let width: u32 = flags.get("width").and_then(|s| s.parse().ok()).unwrap_or(32);
    let founders: usize = flags.get("founders").and_then(|s| s.parse().ok()).unwrap_or(200);

    let mut reports = Vec::new();
    for &seed in &seeds {
        let config = HexConfig {
            world_width: width,
            num_founders: founders,
            ..HexConfig::default()
        };
        let mut sim = HexSim::new(config, seed);
        while sim.turn() < ticks {
            if !sim.tick() {
                break;
            }
        }

        // Build a QD archive from the final living population.
        let mut archive = QdArchive::new(QD_RESOLUTION);
        for body in sim.living_bodies() {
            let descriptor = sim.behavior_descriptor(body);
            // Quality proxy: reproductive success ~ generational depth + energy.
            let quality = body.generation as f32 + body.energy / 100.0;
            archive.insert(body.genome.clone(), descriptor, quality);
        }
        let stats = sim.population_stats();
        eprintln!(
            "seed {seed}: turn={} alive={} born={} extinct_at={:?} qd_cov={}",
            stats.turn, stats.alive, stats.total_ever, stats.extinct_at, archive.coverage()
        );
        reports.push(SeedReport {
            seed,
            final_turn: stats.turn,
            alive: stats.alive,
            total_ever: stats.total_ever,
            extinct_at: stats.extinct_at,
            max_generation: stats.max_generation,
            mean_neurons: stats.mean_neurons,
            mean_edges: stats.mean_edges,
            qd_coverage: archive.coverage(),
            qd_score: archive.qd_score(),
        });
    }

    let n = reports.len().max(1) as f32;
    let summary = Summary {
        ticks,
        mean_alive: reports.iter().map(|r| r.alive as f32).sum::<f32>() / n,
        mean_qd_coverage: reports.iter().map(|r| r.qd_coverage as f32).sum::<f32>() / n,
        extinctions: reports.iter().filter(|r| r.extinct_at.is_some()).count(),
        seeds: reports,
    };

    let json = serde_json::to_string_pretty(&summary)?;
    if let Some(out) = flags.get("out") {
        std::fs::write(out, &json)?;
        eprintln!("wrote {out}");
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_flags(args: &[String]) -> std::collections::BTreeMap<String, String> {
    let mut flags = std::collections::BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            let val = args.get(i + 1).cloned().unwrap_or_default();
            flags.insert(key.to_string(), val);
            i += 2;
        } else {
            i += 1;
        }
    }
    flags
}
