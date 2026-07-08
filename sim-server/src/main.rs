//! sim-server — lean HTTP backend for the new-substrate hex world.
//!
//! Serves a single in-memory `HexSim` (auto-ticked in the background, stopping
//! on extinction — no periodic injection) plus a Quality-Diversity champion
//! archive (MAP-Elites over a behavior descriptor). REST + JSON; a permissive
//! CORS layer lets the Vite dev client talk to it.
//!
//! Flags: --seed N --width W --founders F --port P
//!        --champion-pool-path <p.json>  --seed-genome-snapshot <g.bin>

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use sim_hexworld::{HexConfig, HexSim};
use sim_substrate::{Genome, QdArchive};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

const CHAMPION_POOL_SCHEMA_VERSION: u32 = 5;
const QD_RESOLUTION: usize = 8;

struct AppState {
    sim: RwLock<HexSim>,
    archive: RwLock<QdArchive>,
    pool_path: Option<PathBuf>,
    running: AtomicBool,
}

#[derive(Serialize)]
struct ChampionPoolFile {
    schema_version: u32,
    archive: QdArchive,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let flags = parse_flags();
    let seed: u64 = flags.get("seed").and_then(|s| s.parse().ok()).unwrap_or(0);
    let port: u16 = flags.get("port").and_then(|s| s.parse().ok()).unwrap_or(8080);
    let config = HexConfig {
        world_width: flags.get("width").and_then(|s| s.parse().ok()).unwrap_or(32),
        num_founders: flags.get("founders").and_then(|s| s.parse().ok()).unwrap_or(200),
        ..HexConfig::default()
    };

    // Optional: seed every founder from a single champion genome snapshot.
    let sim = match flags.get("seed-genome-snapshot") {
        Some(path) => {
            let bytes = std::fs::read(path)?;
            let genome: Genome = bincode::deserialize(&bytes)?;
            HexSim::new_from_pool(config, seed, std::slice::from_ref(&genome))
        }
        None => HexSim::new(config, seed),
    };

    let pool_path = flags
        .get("champion-pool-path")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("champion_pool.json")));
    let archive = load_archive(pool_path.as_deref()).unwrap_or_else(|| QdArchive::new(QD_RESOLUTION));

    let state = Arc::new(AppState {
        sim: RwLock::new(sim),
        archive: RwLock::new(archive),
        pool_path,
        running: AtomicBool::new(true),
    });

    // Background ticker: advance while running (HexSim::tick no-ops once extinct).
    let tick_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(30));
        loop {
            interval.tick().await;
            if tick_state.running.load(Ordering::Relaxed) {
                let mut sim = tick_state.sim.write().await;
                sim.tick();
            }
        }
    });

    let app = Router::new()
        .route("/api/state", get(get_state))
        .route("/api/snapshot", get(get_snapshot))
        .route("/api/organism/{id}", get(get_organism))
        .route("/api/champions", get(get_champions))
        .route("/api/champions/{id}", post(save_champion))
        .route("/api/control/{cmd}", post(control))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{port}");
    println!("sim-server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn get_state(State(s): State<Arc<AppState>>) -> Json<Value> {
    let sim = s.sim.read().await;
    Json(serde_json::to_value(sim.population_stats()).unwrap())
}

async fn get_snapshot(State(s): State<Arc<AppState>>) -> Json<Value> {
    let sim = s.sim.read().await;
    Json(serde_json::to_value(sim.render_snapshot()).unwrap())
}

async fn get_organism(
    State(s): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, StatusCode> {
    let sim = s.sim.read().await;
    let (_, body) = sim.body_by_id(id).ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(json!({
        "id": body.id,
        "energy": body.energy,
        "health": body.health,
        "age_turns": body.age_turns,
        "generation": body.generation,
        "morphology": body.morphology,
        "genome": body.genome,
        "brain": body.brain,
    })))
}

async fn get_champions(State(s): State<Arc<AppState>>) -> Json<Value> {
    let archive = s.archive.read().await;
    let entries: Vec<Value> = archive
        .entries()
        .map(|e| json!({ "quality": e.quality, "descriptor": e.descriptor, "genome": e.genome }))
        .collect();
    Json(json!({
        "schema_version": CHAMPION_POOL_SCHEMA_VERSION,
        "coverage": archive.coverage(),
        "qd_score": archive.qd_score(),
        "entries": entries,
    }))
}

async fn save_champion(
    State(s): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, StatusCode> {
    let (genome, descriptor, quality) = {
        let sim = s.sim.read().await;
        let (_, body) = sim.body_by_id(id).ok_or(StatusCode::NOT_FOUND)?;
        (
            body.genome.clone(),
            sim.behavior_descriptor(body),
            body.generation as f32 + body.energy / 100.0,
        )
    };
    let inserted = {
        let mut archive = s.archive.write().await;
        let ins = archive.insert(genome, descriptor, quality);
        persist_archive(s.pool_path.as_deref(), &archive);
        ins
    };
    Ok(Json(json!({ "saved": id, "became_elite": inserted })))
}

async fn control(State(s): State<Arc<AppState>>, Path(cmd): Path<String>) -> Json<Value> {
    match cmd.as_str() {
        "play" => s.running.store(true, Ordering::Relaxed),
        "pause" => s.running.store(false, Ordering::Relaxed),
        "step" => {
            let mut sim = s.sim.write().await;
            sim.tick();
        }
        _ => {}
    }
    let sim = s.sim.read().await;
    Json(json!({ "running": s.running.load(Ordering::Relaxed), "turn": sim.turn() }))
}

fn load_archive(path: Option<&std::path::Path>) -> Option<QdArchive> {
    let path = path?;
    let bytes = std::fs::read(path).ok()?;
    let file: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    if file.get("schema_version").and_then(|v| v.as_u64())
        != Some(CHAMPION_POOL_SCHEMA_VERSION as u64)
    {
        return None;
    }
    serde_json::from_value(file.get("archive")?.clone()).ok()
}

fn persist_archive(path: Option<&std::path::Path>, archive: &QdArchive) {
    let Some(path) = path else { return };
    let file = ChampionPoolFile {
        schema_version: CHAMPION_POOL_SCHEMA_VERSION,
        archive: archive.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&file) {
        let _ = std::fs::write(path, json);
    }
}

fn parse_flags() -> BTreeMap<String, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut flags = BTreeMap::new();
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
