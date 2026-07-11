//! sim-server — a thin, stateless file-server over `world.bin` files.
//!
//! A world is a file on disk under `--world-root`; every request loads it, runs
//! one command (mirroring the sim-cli verbs via the shared `sim-views` crate),
//! and — for mutating commands — saves it back. The one stateful surface is the
//! `/worlds/{name}/stream` WebSocket, which holds a world resident only for the
//! duration of a live animation feed and persists it on disconnect. There is no
//! session registry: the file IS the durable state.

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use clap::Parser;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use sim_core::{SimError, Simulation};
use sim_server::protocol::{
    ApiError, ChampionPoolEntry, ChampionPoolResponse, OrganismDetail, StreamFrame,
    WorldSnapshotView,
};
use sim_types::{OrganismGenome, OrganismState};
use sim_views::{ReadCtx, Recorder};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

/// Reporting-interval width minted for freshly-created worlds and assumed for
/// worlds whose sidecar is absent. Matches the sim-cli / eval default.
const DEFAULT_REPORT_EVERY: u64 = 10_000;

#[derive(Clone)]
struct AppState {
    world_root: Arc<PathBuf>,
    champion_pool: Arc<ChampionPoolStore>,
}

// ---------------------------------------------------------------------------
// World-file addressing
// ---------------------------------------------------------------------------

/// Validate a world name is a safe single path segment (no traversal, no
/// separators). Names map to `<world_root>/<name>.bin`.
fn valid_world_name(name: &str) -> Result<(), AppError> {
    if name.is_empty() || name.len() > 128 {
        return Err(AppError::BadRequest(
            "world name must be 1..=128 chars".to_owned(),
        ));
    }
    if name.contains("..") {
        return Err(AppError::BadRequest(
            "world name may not contain `..`".to_owned(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(AppError::BadRequest(
            "world name may only contain [A-Za-z0-9._-]".to_owned(),
        ));
    }
    Ok(())
}

fn world_bin_path(root: &FsPath, name: &str) -> PathBuf {
    root.join(format!("{name}.bin"))
}

/// A world loaded from disk together with its metric sidecar (if present).
struct Loaded {
    sim: Simulation,
    recorder: Option<Recorder>,
    report_every: u64,
}

fn load_bundle(root: &FsPath, name: &str) -> Result<Loaded, AppError> {
    valid_world_name(name)?;
    let path = world_bin_path(root, name);
    if !path.exists() {
        return Err(AppError::NotFound(format!("world `{name}` not found")));
    }
    let path_str = path.to_string_lossy();
    let sim = sim_views::load_world(&path_str)
        .map_err(|e| AppError::BadRequest(format!("loading world `{name}`: {e}")))?;
    let sidecar = sim_views::sibling_metrics_path(&path_str);
    let (report_every, recorder) = if FsPath::new(&sidecar).exists() {
        let (report_every, recorder) = sim_views::load_sidecar(&sidecar)
            .map_err(|e| AppError::Internal(format!("loading metrics for `{name}`: {e}")))?;
        (report_every, Some(recorder))
    } else {
        (DEFAULT_REPORT_EVERY, None)
    };
    Ok(Loaded {
        sim,
        recorder,
        report_every,
    })
}

fn save_bundle(
    root: &FsPath,
    name: &str,
    sim: &Simulation,
    recorder: Option<&Recorder>,
    report_every: u64,
) -> Result<(), AppError> {
    let path = world_bin_path(root, name);
    let path_str = path.to_string_lossy();
    sim_views::save_world(sim, &path_str)
        .map_err(|e| AppError::Internal(format!("saving world `{name}`: {e}")))?;
    if let Some(rec) = recorder {
        let sidecar = sim_views::sibling_metrics_path(&path_str);
        sim_views::save_sidecar(report_every, rec, &sidecar)
            .map_err(|e| AppError::Internal(format!("saving metrics for `{name}`: {e}")))?;
    }
    Ok(())
}

/// Build a JSON `Response` from bytes a `sim-views` read wrote (already a
/// complete JSON document + trailing newline).
fn json_bytes_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

/// Run a `sim-views` read against a loaded world, capturing its JSON output.
fn run_read(
    loaded: &Loaded,
    f: impl FnOnce(&ReadCtx, &mut Vec<u8>) -> anyhow::Result<()>,
) -> Result<Vec<u8>, AppError> {
    let ctx = ReadCtx {
        sim: &loaded.sim,
        recorder: loaded.recorder.as_ref(),
        report_every: loaded.report_every,
        format: sim_views::output::Format::Json,
        scaled: false,
    };
    let mut buf = Vec::new();
    f(&ctx, &mut buf).map_err(|e| AppError::BadRequest(e.to_string()))?;
    Ok(buf)
}

/// Load a world on the blocking pool and run a read closure, returning a JSON
/// response. Reads are pure but touch the filesystem + CPU, so they run off the
/// async runtime.
async fn blocking_read(
    root: Arc<PathBuf>,
    name: String,
    f: impl FnOnce(&Loaded) -> Result<Vec<u8>, AppError> + Send + 'static,
) -> Result<Response, AppError> {
    let bytes = tokio::task::spawn_blocking(move || {
        let loaded = load_bundle(&root, &name)?;
        f(&loaded)
    })
    .await
    .map_err(|e| AppError::Internal(format!("read worker join error: {e}")))??;
    Ok(json_bytes_response(bytes))
}

// ---------------------------------------------------------------------------
// Champion pool (persisted set of the best genomes seen; unchanged semantics)
// ---------------------------------------------------------------------------

const CHAMPION_POOL_SCHEMA_VERSION: u32 = 5;
const CHAMPION_POOL_MAX_GENOMES: usize = 32;
const CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChampionPoolFile {
    schema_version: u32,
    updated_at_unix_ms: u128,
    entries: Vec<ChampionGenomeRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ChampionGenomeRecord {
    genome: OrganismGenome,
    source_turn: u64,
    source_created_at_unix_ms: u128,
    generation: u64,
    age_turns: u64,
    consumptions_count: u64,
    energy: f32,
}

struct ChampionPoolStore {
    /// `None` for ephemeral stores (e.g. `--seed-genome-snapshot` mode); in that
    /// case any attempt to persist silently no-ops so the pool stays read-only.
    pool_path: Option<PathBuf>,
    entries: StdRwLock<Vec<ChampionGenomeRecord>>,
}

fn now_unix_ms() -> Result<u128, AppError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| AppError::Internal(e.to_string()))
        .map(|duration| duration.as_millis())
}

fn champion_pool_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("champion_pool.json")
}

fn compare_champion_records(
    left: &ChampionGenomeRecord,
    right: &ChampionGenomeRecord,
) -> std::cmp::Ordering {
    right
        .generation
        .cmp(&left.generation)
        .then_with(|| right.consumptions_count.cmp(&left.consumptions_count))
        .then_with(|| right.energy.total_cmp(&left.energy))
        .then_with(|| right.age_turns.cmp(&left.age_turns))
        .then_with(|| {
            right
                .source_created_at_unix_ms
                .cmp(&left.source_created_at_unix_ms)
        })
}

fn champion_record_from_organism(
    source_created_at_unix_ms: u128,
    source_turn: u64,
    organism: &OrganismState,
) -> ChampionGenomeRecord {
    ChampionGenomeRecord {
        genome: organism.genome.clone(),
        source_turn,
        source_created_at_unix_ms,
        generation: organism.generation,
        age_turns: organism.age_turns,
        consumptions_count: organism.consumptions_count,
        energy: organism.energy,
    }
}

fn select_champion_candidates(
    source_created_at_unix_ms: u128,
    source_turn: u64,
    organisms: &[OrganismState],
) -> Vec<ChampionGenomeRecord> {
    let mut candidates: Vec<ChampionGenomeRecord> = organisms
        .iter()
        .map(|organism| {
            champion_record_from_organism(source_created_at_unix_ms, source_turn, organism)
        })
        .collect();
    candidates.sort_by(compare_champion_records);

    let mut unique =
        Vec::with_capacity(CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD.min(candidates.len()));
    for candidate in candidates {
        if unique
            .iter()
            .any(|existing: &ChampionGenomeRecord| existing.genome == candidate.genome)
        {
            continue;
        }
        unique.push(candidate);
        if unique.len() >= CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD {
            break;
        }
    }

    unique
}

fn merge_champion_entries(
    existing: &[ChampionGenomeRecord],
    candidates: &[ChampionGenomeRecord],
) -> Vec<ChampionGenomeRecord> {
    let mut merged = Vec::with_capacity(existing.len() + candidates.len());
    merged.extend(existing.iter().cloned());
    merged.extend(candidates.iter().cloned());
    merged.sort_by(compare_champion_records);

    let mut unique = Vec::with_capacity(CHAMPION_POOL_MAX_GENOMES.min(merged.len()));
    for entry in merged {
        if unique
            .iter()
            .any(|existing_entry: &ChampionGenomeRecord| existing_entry.genome == entry.genome)
        {
            continue;
        }
        unique.push(entry);
        if unique.len() >= CHAMPION_POOL_MAX_GENOMES {
            break;
        }
    }

    unique
}

impl ChampionPoolStore {
    fn bootstrap(pool_path: PathBuf) -> Result<Self, AppError> {
        if let Some(parent) = pool_path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                AppError::Internal(format!(
                    "failed to create champion pool directory {}: {err}",
                    parent.display()
                ))
            })?;
        }

        let entries = match std::fs::read(&pool_path) {
            Ok(bytes) => match serde_json::from_slice::<ChampionPoolFile>(&bytes) {
                Ok(file) if file.schema_version == CHAMPION_POOL_SCHEMA_VERSION => file.entries,
                Ok(file) => {
                    warn!(
                        "ignoring champion pool {} with schema version {} (expected {})",
                        pool_path.display(),
                        file.schema_version,
                        CHAMPION_POOL_SCHEMA_VERSION
                    );
                    Vec::new()
                }
                Err(err) => {
                    warn!(
                        "failed to parse champion pool {}: {err}",
                        pool_path.display()
                    );
                    Vec::new()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(err) => {
                return Err(AppError::Internal(format!(
                    "failed to read champion pool {}: {err}",
                    pool_path.display()
                )));
            }
        };

        Ok(Self {
            pool_path: Some(pool_path),
            entries: StdRwLock::new(entries),
        })
    }

    /// Build a read-only pool containing a single genome loaded from a
    /// bincode-encoded evaluation snapshot. Every initial organism will start
    /// with this genome; champion-save endpoints no-op against disk.
    fn from_snapshot_file(snapshot_path: &FsPath) -> Result<Self, AppError> {
        let bytes = std::fs::read(snapshot_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to read seed genome snapshot {}: {err}",
                snapshot_path.display()
            ))
        })?;
        let genome: OrganismGenome = bincode::deserialize(&bytes).map_err(|err| {
            AppError::Internal(format!(
                "failed to decode seed genome snapshot {}: {err}",
                snapshot_path.display()
            ))
        })?;
        let record = ChampionGenomeRecord {
            genome,
            source_turn: snapshot_tick_from_filename(snapshot_path).unwrap_or(0),
            source_created_at_unix_ms: now_unix_ms().unwrap_or(0),
            generation: 0,
            age_turns: 0,
            consumptions_count: 0,
            energy: 0.0,
        };
        Ok(Self {
            pool_path: None,
            entries: StdRwLock::new(vec![record]),
        })
    }

    fn snapshot_genomes(&self) -> Result<Vec<OrganismGenome>, AppError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        Ok(entries.iter().map(|entry| entry.genome.clone()).collect())
    }

    fn snapshot_entries(&self) -> Result<Vec<ChampionGenomeRecord>, AppError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        Ok(entries.clone())
    }

    fn clear(&self) -> Result<(), AppError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        self.persist_entries(&[])?;
        entries.clear();
        Ok(())
    }

    fn delete_entry(&self, index: usize) -> Result<Vec<ChampionGenomeRecord>, AppError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        if index >= entries.len() {
            return Err(AppError::NotFound(format!(
                "champion pool entry {index} not found"
            )));
        }
        entries.remove(index);
        self.persist_entries(entries.as_slice())?;
        Ok(entries.clone())
    }

    fn update_from_simulation(&self, simulation: &Simulation) -> Result<(), AppError> {
        let source_created_at_unix_ms = now_unix_ms()?;
        let candidates = select_champion_candidates(
            source_created_at_unix_ms,
            simulation.turn(),
            simulation.organisms(),
        );
        if candidates.is_empty() {
            return Ok(());
        }

        let mut entries = self
            .entries
            .write()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        let merged = merge_champion_entries(entries.as_slice(), &candidates);
        self.persist_entries(&merged)?;
        *entries = merged;
        Ok(())
    }

    fn persist_entries(&self, entries: &[ChampionGenomeRecord]) -> Result<(), AppError> {
        let Some(pool_path) = self.pool_path.as_ref() else {
            return Ok(());
        };
        let file = ChampionPoolFile {
            schema_version: CHAMPION_POOL_SCHEMA_VERSION,
            updated_at_unix_ms: now_unix_ms()?,
            entries: entries.to_vec(),
        };
        let encoded = serde_json::to_vec_pretty(&file).map_err(|err| {
            AppError::Internal(format!(
                "failed to encode champion pool {}: {err}",
                pool_path.display()
            ))
        })?;

        let temp_path = pool_path.with_extension("json.tmp");
        std::fs::write(&temp_path, encoded).map_err(|err| {
            AppError::Internal(format!(
                "failed to write champion pool temp file {}: {err}",
                temp_path.display()
            ))
        })?;
        std::fs::rename(&temp_path, pool_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to finalize champion pool {}: {err}",
                pool_path.display()
            ))
        })?;
        Ok(())
    }
}

fn snapshot_tick_from_filename(path: &FsPath) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let digits = stem.strip_prefix('t')?;
    digits.parse().ok()
}

impl From<ChampionGenomeRecord> for ChampionPoolEntry {
    fn from(entry: ChampionGenomeRecord) -> Self {
        Self {
            genome: entry.genome,
            source_turn: entry.source_turn,
            source_created_at_unix_ms: entry.source_created_at_unix_ms,
            generation: entry.generation,
            age_turns: entry.age_turns,
            consumptions_count: entry.consumptions_count,
            energy: entry.energy,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum AppError {
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::NotFound(message)
            | AppError::BadRequest(message)
            | AppError::Internal(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AppError {}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal", msg),
        };
        let error = ApiError {
            code: code.to_owned(),
            message,
        };
        (status, Json(error)).into_response()
    }
}

impl From<SimError> for AppError {
    fn from(value: SimError) -> Self {
        AppError::BadRequest(value.to_string())
    }
}

// ---------------------------------------------------------------------------
// Request / response bodies
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

/// Result of a mutating world command: the world's name + its fresh render
/// snapshot, so the client can update the canvas immediately.
#[derive(Serialize)]
struct WorldResponse {
    name: String,
    snapshot: WorldSnapshotView,
}

#[derive(Deserialize)]
struct NewWorldRequest {
    name: Option<String>,
    seed: Option<u64>,
    config: Option<String>,
    /// Inline `key=value` config overrides (same vocabulary as `sim-cli --set`).
    #[serde(default)]
    set: Vec<String>,
    /// `[world_width, num_organisms]` scale override (marks the world non-canonical).
    scale: Option<[u32; 2]>,
    threads: Option<u32>,
    report_every: Option<u64>,
}

#[derive(Deserialize)]
struct StepRequest {
    count: u32,
}

#[derive(Deserialize)]
struct RunToRequest {
    turn: u64,
}

#[derive(Deserialize)]
struct TopQuery {
    n: Option<usize>,
}

#[derive(Deserialize)]
struct BrainQuery {
    view: Option<String>,
}

#[derive(Deserialize)]
struct FindQuery {
    expr: String,
    limit: Option<usize>,
    fields: Option<String>,
}

#[derive(Deserialize)]
struct GenomeQuery {
    gene: Option<String>,
    #[serde(default)]
    drift: bool,
}

#[derive(Deserialize)]
struct TimeseriesQuery {
    cols: Option<String>,
    last: Option<usize>,
}

#[derive(Deserialize)]
struct StreamQuery {
    /// Ticks per second to stream; 0 (default) means as fast as possible.
    tps: Option<u32>,
}

// ---------------------------------------------------------------------------
// CLI + startup
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(name = "sim-server")]
#[command(about = "Thin file-server over world.bin files")]
struct Cli {
    /// Directory holding world files (`<name>.bin` + `<name>.metrics`).
    #[arg(long, default_value = "artifacts/worlds")]
    world_root: PathBuf,
    /// Override the default champion pool JSON path.
    #[arg(long)]
    champion_pool_path: Option<PathBuf>,
    /// Bincode-encoded `OrganismGenome` from an evaluation snapshot
    /// (`artifacts/evaluation/.../seed_<seed>/genomes/tNNNNNN.bin`). When set,
    /// every initial organism spawns with this genome and champion-save
    /// endpoints do not touch disk.
    #[arg(long)]
    seed_genome_snapshot: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "sim_server=info,axum=info,tower_http=info".to_owned()),
        )
        .init();

    let app = build_app(new_state(&cli)?);

    let addr = std::env::var("SIM_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("sim-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn new_state(cli: &Cli) -> Result<AppState, AppError> {
    std::fs::create_dir_all(&cli.world_root).map_err(|err| {
        AppError::Internal(format!(
            "failed to create world root {}: {err}",
            cli.world_root.display()
        ))
    })?;

    let champion_pool = if let Some(snapshot_path) = &cli.seed_genome_snapshot {
        info!(
            "seeding champion pool from snapshot {} (persistence disabled)",
            snapshot_path.display()
        );
        ChampionPoolStore::from_snapshot_file(snapshot_path)?
    } else {
        let pool_path = cli
            .champion_pool_path
            .clone()
            .unwrap_or_else(champion_pool_path);
        ChampionPoolStore::bootstrap(pool_path)?
    };
    Ok(AppState {
        world_root: Arc::new(cli.world_root.clone()),
        champion_pool: Arc::new(champion_pool),
    })
}

fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/worlds", get(list_worlds).post(create_world))
        .route("/worlds/{name}/snapshot", get(get_snapshot))
        .route("/worlds/{name}/organism/{id}", get(get_organism))
        .route("/worlds/{name}/step", post(step_world))
        .route("/worlds/{name}/run-to", post(run_to_world))
        .route("/worlds/{name}/stream", get(stream_world))
        .route("/worlds/{name}/champions", post(save_world_champions))
        .route("/worlds/{name}/state", get(read_state))
        .route("/worlds/{name}/turn", get(read_turn))
        .route("/worlds/{name}/pillars", get(read_pillars))
        .route("/worlds/{name}/eco", get(read_eco))
        .route("/worlds/{name}/lineage", get(read_lineage))
        .route("/worlds/{name}/genome", get(read_genome))
        .route("/worlds/{name}/timeseries", get(read_timeseries))
        .route("/worlds/{name}/food", get(read_food))
        .route("/worlds/{name}/inspect/{id}", get(read_inspect))
        .route("/worlds/{name}/brain/{id}", get(read_brain))
        .route("/worlds/{name}/decide/{id}", get(read_decide))
        .route("/worlds/{name}/top/{field}", get(read_top))
        .route("/worlds/{name}/hist/{field}", get(read_hist))
        .route("/worlds/{name}/find", get(read_find))
        .route(
            "/champions",
            get(get_champion_pool).delete(clear_champion_pool),
        )
        .route("/champions/{index}", delete(delete_champion_pool_entry))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

// ---------------------------------------------------------------------------
// World lifecycle + render views
// ---------------------------------------------------------------------------

async fn list_worlds(State(state): State<AppState>) -> Result<Json<Vec<String>>, AppError> {
    let root = state.world_root.clone();
    let names = tokio::task::spawn_blocking(move || -> Result<Vec<String>, AppError> {
        let mut names = Vec::new();
        match std::fs::read_dir(root.as_ref().as_path()) {
            Ok(read_dir) => {
                for entry in read_dir.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            names.push(stem.to_owned());
                        }
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(AppError::Internal(err.to_string())),
        }
        names.sort();
        Ok(names)
    })
    .await
    .map_err(|e| AppError::Internal(format!("list worker join error: {e}")))??;
    Ok(Json(names))
}

async fn create_world(
    State(state): State<AppState>,
    Json(req): Json<NewWorldRequest>,
) -> Result<Json<WorldResponse>, AppError> {
    let root = state.world_root.clone();
    let pool = state.champion_pool.clone();
    let response = tokio::task::spawn_blocking(move || build_new_world(&root, &pool, req))
        .await
        .map_err(|e| AppError::Internal(format!("create worker join error: {e}")))??;
    Ok(Json(response))
}

fn build_new_world(
    root: &FsPath,
    pool: &ChampionPoolStore,
    req: NewWorldRequest,
) -> Result<WorldResponse, AppError> {
    let name = req
        .name
        .unwrap_or_else(|| format!("world-{}", Uuid::new_v4().simple()));
    valid_world_name(&name)?;

    let config_path = req.config.unwrap_or_else(|| {
        sim_server::default_world_config_path()
            .to_string_lossy()
            .into_owned()
    });
    let sets: Vec<(String, String)> = req
        .set
        .iter()
        .map(|kv| {
            kv.split_once('=')
                .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
                .ok_or_else(|| AppError::BadRequest(format!("--set wants key=value, got `{kv}`")))
        })
        .collect::<Result<_, _>>()?;

    let mut config = sim_views::world_config_with_overrides(&config_path, &sets)
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    if let Some(threads) = req.threads {
        config.intent_parallel_threads = threads;
    }
    if let Some([width, pop]) = req.scale {
        config.world_width = width;
        config.num_organisms = pop;
    }
    let report_every = req.report_every.unwrap_or(DEFAULT_REPORT_EVERY);
    let champions = pool.snapshot_genomes()?;
    let sim = Simulation::new_with_champion_pool(config, req.seed.unwrap_or(0), champions)?;
    let recorder = sim_views::start_recording(&sim);
    save_bundle(root, &name, &sim, Some(&recorder), report_every)?;
    let snapshot = sim.snapshot().into();
    Ok(WorldResponse { name, snapshot })
}

async fn get_snapshot(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let root = state.world_root.clone();
    let view = tokio::task::spawn_blocking(move || -> Result<WorldSnapshotView, AppError> {
        let loaded = load_bundle(&root, &name)?;
        Ok(loaded.sim.snapshot().into())
    })
    .await
    .map_err(|e| AppError::Internal(format!("snapshot worker join error: {e}")))??;
    Ok(Json(view))
}

async fn get_organism(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<AppState>,
) -> Result<Json<OrganismDetail>, AppError> {
    let root = state.world_root.clone();
    let detail = tokio::task::spawn_blocking(move || -> Result<OrganismDetail, AppError> {
        let loaded = load_bundle(&root, &name)?;
        let organism = loaded
            .sim
            .focused_organism(sim_types::OrganismId(id))
            .ok_or_else(|| AppError::NotFound(format!("no live organism with id {id}")))?;
        let active_action_neuron_id = organism.last_action_taken.neuron_id();
        Ok(OrganismDetail {
            turn: loaded.sim.turn(),
            organism,
            active_action_neuron_id,
        })
    })
    .await
    .map_err(|e| AppError::Internal(format!("organism worker join error: {e}")))??;
    Ok(Json(detail))
}

async fn step_world(
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(req): Json<StepRequest>,
) -> Result<Json<WorldResponse>, AppError> {
    let root = state.world_root.clone();
    let response = tokio::task::spawn_blocking(move || -> Result<WorldResponse, AppError> {
        let mut loaded = load_bundle(&root, &name)?;
        sim_views::advance(
            &mut loaded.sim,
            loaded.recorder.as_mut(),
            req.count.max(1) as u64,
        );
        save_bundle(
            &root,
            &name,
            &loaded.sim,
            loaded.recorder.as_ref(),
            loaded.report_every,
        )?;
        Ok(WorldResponse {
            name,
            snapshot: loaded.sim.snapshot().into(),
        })
    })
    .await
    .map_err(|e| AppError::Internal(format!("step worker join error: {e}")))??;
    Ok(Json(response))
}

async fn run_to_world(
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(req): Json<RunToRequest>,
) -> Result<Json<WorldResponse>, AppError> {
    let root = state.world_root.clone();
    let response = tokio::task::spawn_blocking(move || -> Result<WorldResponse, AppError> {
        let mut loaded = load_bundle(&root, &name)?;
        let current = loaded.sim.turn();
        if req.turn > current {
            sim_views::advance(
                &mut loaded.sim,
                loaded.recorder.as_mut(),
                req.turn - current,
            );
            save_bundle(
                &root,
                &name,
                &loaded.sim,
                loaded.recorder.as_ref(),
                loaded.report_every,
            )?;
        }
        Ok(WorldResponse {
            name,
            snapshot: loaded.sim.snapshot().into(),
        })
    })
    .await
    .map_err(|e| AppError::Internal(format!("run-to worker join error: {e}")))??;
    Ok(Json(response))
}

// ---------------------------------------------------------------------------
// CLI-parity reads (forwarded from sim-views as raw JSON)
// ---------------------------------------------------------------------------

async fn read_state(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| sim_views::state(c, &[], b))
    })
    .await
}

async fn read_turn(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| sim_views::turn(c, &[], b))
    })
    .await
}

async fn read_pillars(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| sim_views::pillars(c, &[], b))
    })
    .await
}

async fn read_eco(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| sim_views::eco(c, &[], b))
    })
    .await
}

async fn read_lineage(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| sim_views::lineage(c, &[], b))
    })
    .await
}

async fn read_food(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| sim_views::food(c, &[], b))
    })
    .await
}

async fn read_genome(
    Path(name): Path<String>,
    Query(q): Query<GenomeQuery>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let mut args: Vec<&str> = Vec::new();
        if let Some(gene) = q.gene.as_deref() {
            args.push("--gene");
            args.push(gene);
        }
        if q.drift {
            args.push("--drift");
        }
        run_read(l, |c, b| sim_views::genome(c, &args, b))
    })
    .await
}

async fn read_timeseries(
    Path(name): Path<String>,
    Query(q): Query<TimeseriesQuery>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let last = q.last.map(|n| n.to_string());
        let mut args: Vec<&str> = Vec::new();
        if let Some(cols) = q.cols.as_deref() {
            args.push("--cols");
            args.push(cols);
        }
        if let Some(last) = last.as_deref() {
            args.push("--last");
            args.push(last);
        }
        run_read(l, |c, b| sim_views::timeseries(c, &args, b))
    })
    .await
}

async fn read_inspect(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let id = id.to_string();
        run_read(l, |c, b| sim_views::inspect(c, &[&id], b))
    })
    .await
}

async fn read_decide(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let id = id.to_string();
        run_read(l, |c, b| sim_views::decide(c, &[&id], b))
    })
    .await
}

async fn read_brain(
    Path((name, id)): Path<(String, u64)>,
    Query(q): Query<BrainQuery>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let id = id.to_string();
        let view = q.view;
        let mut args: Vec<&str> = vec![&id];
        if let Some(view) = view.as_deref() {
            args.push("--view");
            args.push(view);
        }
        run_read(l, |c, b| sim_views::brain(c, &args, b))
    })
    .await
}

async fn read_top(
    Path((name, field)): Path<(String, String)>,
    Query(q): Query<TopQuery>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let n = q.n.map(|n| n.to_string());
        let mut args: Vec<&str> = vec![&field];
        if let Some(n) = n.as_deref() {
            args.push(n);
        }
        run_read(l, |c, b| sim_views::top(c, &args, b))
    })
    .await
}

async fn read_hist(
    Path((name, field)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        run_read(l, |c, b| sim_views::hist(c, &[&field], b))
    })
    .await
}

async fn read_find(
    Path(name): Path<String>,
    Query(q): Query<FindQuery>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let limit = q.limit.map(|n| n.to_string());
        let mut args: Vec<&str> = q.expr.split_whitespace().collect();
        if let Some(limit) = limit.as_deref() {
            args.push("--limit");
            args.push(limit);
        }
        if let Some(fields) = q.fields.as_deref() {
            args.push("--fields");
            args.push(fields);
        }
        run_read(l, |c, b| sim_views::find(c, &args, b))
    })
    .await
}

// ---------------------------------------------------------------------------
// Champions
// ---------------------------------------------------------------------------

async fn get_champion_pool(
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    let entries = state.champion_pool.snapshot_entries()?;
    Ok(Json(ChampionPoolResponse {
        entries: entries.into_iter().map(ChampionPoolEntry::from).collect(),
    }))
}

async fn clear_champion_pool(
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    state.champion_pool.clear()?;
    Ok(Json(ChampionPoolResponse {
        entries: Vec::new(),
    }))
}

async fn delete_champion_pool_entry(
    Path(index): Path<usize>,
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    let entries = state.champion_pool.delete_entry(index)?;
    Ok(Json(ChampionPoolResponse {
        entries: entries.into_iter().map(ChampionPoolEntry::from).collect(),
    }))
}

async fn save_world_champions(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    let root = state.world_root.clone();
    let pool = state.champion_pool.clone();
    let entries =
        tokio::task::spawn_blocking(move || -> Result<Vec<ChampionGenomeRecord>, AppError> {
            let loaded = load_bundle(&root, &name)?;
            if let Err(err) = pool.update_from_simulation(&loaded.sim) {
                warn!("failed to update champion pool from world `{name}`: {err}");
            }
            pool.snapshot_entries()
        })
        .await
        .map_err(|e| AppError::Internal(format!("champion-save worker join error: {e}")))??;

    Ok(Json(ChampionPoolResponse {
        entries: entries.into_iter().map(ChampionPoolEntry::from).collect(),
    }))
}

// ---------------------------------------------------------------------------
// Live animation stream (the one resident, transient stateful surface)
// ---------------------------------------------------------------------------

async fn stream_world(
    ws: WebSocketUpgrade,
    Path(name): Path<String>,
    Query(q): Query<StreamQuery>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    valid_world_name(&name)?;
    // Fail before the upgrade if the world is missing/unreadable.
    let root = state.world_root.clone();
    let name_check = name.clone();
    let root_check = root.clone();
    tokio::task::spawn_blocking(move || load_bundle(&root_check, &name_check).map(|_| ()))
        .await
        .map_err(|e| AppError::Internal(format!("stream preflight join error: {e}")))??;

    let tps = q.tps.unwrap_or(0);
    Ok(ws.on_upgrade(move |socket| stream_loop(socket, root, name, tps)))
}

async fn stream_loop(socket: WebSocket, root: Arc<PathBuf>, name: String, tps: u32) {
    let (root_load, name_load) = (root.clone(), name.clone());
    let loaded =
        match tokio::task::spawn_blocking(move || load_bundle(&root_load, &name_load)).await {
            Ok(Ok(loaded)) => loaded,
            Ok(Err(err)) => {
                warn!("stream `{name}` failed to load world: {err}");
                return;
            }
            Err(err) => {
                error!("stream `{name}` load worker join error: {err}");
                return;
            }
        };
    let mut sim = loaded.sim;
    let mut recorder = loaded.recorder;
    let report_every = loaded.report_every;

    let (mut sender, mut receiver) = socket.split();
    let initial = StreamFrame::StateSnapshot(sim.snapshot().into());
    if send_frame(&mut sender, &initial).await.is_err() {
        return;
    }

    let step_delay = if tps > 0 {
        Duration::from_millis((1000_u64 / tps as u64).max(1))
    } else {
        Duration::ZERO
    };

    loop {
        tokio::select! {
            incoming = receiver.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                }
            }
            _ = tokio::time::sleep(step_delay) => {
                let delta = sim_views::tick_recording(&mut sim, recorder.as_mut());
                if send_frame(&mut sender, &StreamFrame::TickDelta(delta.into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    }

    // Persist the advanced world + sidecar so the stream leaves the file at its
    // current turn, exactly as a `step`/`run-to` would.
    let _ = tokio::task::spawn_blocking(move || {
        if let Err(err) = save_bundle(&root, &name, &sim, recorder.as_ref(), report_every) {
            warn!("stream `{name}` failed to persist world on close: {err}");
        }
    })
    .await;
}

async fn send_frame(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    frame: &StreamFrame,
) -> Result<(), ()> {
    match serde_json::to_string(frame) {
        Ok(text) => sender
            .send(Message::Text(text.into()))
            .await
            .map_err(|_| ()),
        Err(err) => {
            error!("failed to serialize stream frame: {err}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{
        action_gene_node_id, connection_innovation_id, seed_hidden_gene_node_id,
        sensory_gene_node_id, ActionType, BrainState, FacingDirection, HiddenNodeGene, OrganismId,
        SensoryReceptor, SpeciesId, SynapseGene,
    };

    fn make_record(generation: u64, consumptions: u64, energy: f32) -> ChampionGenomeRecord {
        let mut genome = OrganismGenome::test_fixture();
        genome.lifecycle.max_organism_age = u32::MAX;
        genome.topology.vision_distance = generation as u32 + 2;
        genome.brain.hidden_nodes = (0..generation as u32 + 1)
            .map(|index| HiddenNodeGene {
                id: seed_hidden_gene_node_id(index),
                bias: generation as f32,
                log_time_constant: generation as f32 * 0.1,
            })
            .collect();

        ChampionGenomeRecord {
            genome,
            source_turn: generation * 10,
            source_created_at_unix_ms: generation as u128,
            generation,
            age_turns: generation * 2,
            consumptions_count: consumptions,
            energy,
        }
    }

    #[test]
    fn merge_champion_entries_prefers_higher_ranked_unique_genomes() {
        let weak = make_record(3, 1, 10.0);
        let strong = make_record(8, 9, 80.0);
        let duplicate_strong = strong.clone();

        let merged = merge_champion_entries(
            std::slice::from_ref(&weak),
            &[duplicate_strong, strong.clone()],
        );

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], strong);
        assert_eq!(merged[1], weak);
    }

    #[test]
    fn select_champion_candidates_deduplicates_identical_genomes() {
        let strong = make_record(10, 2, 40.0);
        let weaker_duplicate = ChampionGenomeRecord {
            energy: 5.0,
            generation: 2,
            ..strong.clone()
        };
        let empty_brain = BrainState {
            sensory: Vec::new(),
            inter: Vec::new(),
            action: Vec::new(),
            synapse_count: 0,
            sensory_mean_activation: Vec::new(),
            inter_mean_activation: Vec::new(),
            action_mean_activation: Vec::new(),
            means_initialized: false,
        };

        let organisms = vec![
            OrganismState::new(
                OrganismId(1),
                SpeciesId(1),
                0,
                0,
                strong.generation,
                strong.age_turns,
                FacingDirection::East,
                strong.energy,
                strong.energy.max(1.0),
                strong.energy.max(1.0),
                0.0,
                strong.consumptions_count,
                0,
                0,
                ActionType::Idle,
                empty_brain.clone(),
                strong.genome.clone(),
            ),
            OrganismState::new(
                OrganismId(2),
                SpeciesId(2),
                1,
                0,
                weaker_duplicate.generation,
                weaker_duplicate.age_turns,
                FacingDirection::East,
                weaker_duplicate.energy,
                weaker_duplicate.energy.max(1.0),
                weaker_duplicate.energy.max(1.0),
                0.0,
                weaker_duplicate.consumptions_count,
                0,
                0,
                ActionType::Idle,
                empty_brain,
                weaker_duplicate.genome,
            ),
        ];

        let candidates = select_champion_candidates(1, 100, &organisms);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].generation, strong.generation);
        assert_eq!(candidates[0].energy, strong.energy);
    }

    #[test]
    fn champion_persistence_round_trip_preserves_brain_layout_and_edges() {
        let food_receptor = SensoryReceptor::FoodRay { ray_offset: 0 };
        let organism_receptor = SensoryReceptor::OrganismRay { ray_offset: 0 };
        let forward_action = ActionType::Forward;
        let attack_action = ActionType::Attack;

        let mut record = make_record(4, 3, 25.0);
        record.genome.brain.hidden_nodes = vec![
            HiddenNodeGene {
                id: seed_hidden_gene_node_id(0),
                bias: 0.1,
                log_time_constant: 0.0,
            },
            HiddenNodeGene {
                id: seed_hidden_gene_node_id(1),
                bias: -0.2,
                log_time_constant: 0.3,
            },
        ];
        let edge = |pre_node_id, post_node_id, weight| SynapseGene {
            innovation: connection_innovation_id(pre_node_id, post_node_id),
            pre_node_id,
            post_node_id,
            weight,
            enabled: true,
        };
        let action_index = |action| {
            ActionType::ALL
                .iter()
                .position(|candidate| *candidate == action)
                .unwrap()
        };
        record.genome.brain.edges = vec![
            edge(
                sensory_gene_node_id(food_receptor.current_index().unwrap() as u32),
                seed_hidden_gene_node_id(0),
                0.75,
            ),
            edge(
                sensory_gene_node_id(organism_receptor.current_index().unwrap() as u32),
                action_gene_node_id(action_index(attack_action)),
                -0.5,
            ),
            edge(
                seed_hidden_gene_node_id(0),
                action_gene_node_id(action_index(forward_action)),
                1.25,
            ),
        ];
        record
            .genome
            .brain
            .edges
            .sort_unstable_by_key(|edge| edge.innovation);

        let json = serde_json::to_vec(&record).expect("record should serialize");
        let decoded: ChampionGenomeRecord =
            serde_json::from_slice(&json).expect("record should deserialize");

        assert_eq!(decoded, record);
    }
}
