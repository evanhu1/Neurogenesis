use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use sim_core::{derive_active_action_neuron_id, SimError, Simulation};
use sim_server::{
    load_default_world_config,
    protocol::{
        ApiError, ArchivedWorldSource, ArchivedWorldSummary, BatchAggregateStats, BatchRunStatus,
        BatchRunStatusResponse, ClientCommand, CountRequest, CreateBatchRunRequest,
        CreateBatchRunResponse, CreateSessionRequest, CreateSessionResponse, FocusBrainData,
        FocusRequest, ListArchivedWorldsResponse, ResetRequest, ServerEvent, SessionMetadata,
        StepProgressData, WorldSnapshotView,
    },
};
use sim_types::{OrganismGenome, OrganismState, WorldSnapshot};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::task::{JoinHandle, JoinSet};
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    sessions: Arc<RwLock<HashMap<Uuid, Arc<Session>>>>,
    batch_runs: Arc<RwLock<HashMap<Uuid, Arc<BatchRun>>>>,
    world_archive: Arc<WorldArchiveStore>,
    champion_pool: Arc<ChampionPoolStore>,
}

struct Session {
    metadata: SessionMetadata,
    simulation: Mutex<Simulation>,
    events: broadcast::Sender<ServerEvent>,
    runtime: Mutex<RuntimeState>,
}

#[derive(Default)]
struct RuntimeState {
    running: bool,
    ticks_per_second: u32,
    runner: Option<JoinHandle<()>>,
}

struct BatchRun {
    id: Uuid,
    created_at_unix_ms: u128,
    total_worlds: u32,
    inner: Mutex<BatchRunInner>,
}

struct BatchRunInner {
    completed_worlds: u32,
    status: BatchRunStatus,
    worlds: Vec<ArchivedWorldSummary>,
    aggregate: Option<BatchAggregateStats>,
    error: Option<String>,
}

const WORLD_ARCHIVE_SCHEMA_VERSION: u32 = 1;
const CHAMPION_POOL_SCHEMA_VERSION: u32 = 1;
const CHAMPION_POOL_MAX_GENOMES: usize = 32;
const CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchivedWorldFile {
    schema_version: u32,
    world_id: Uuid,
    created_at_unix_ms: u128,
    source: ArchivedWorldSource,
    summary: ArchivedWorldSummary,
    simulation: Simulation,
}

#[derive(Debug, Clone)]
struct ArchivedWorldIndexEntry {
    summary: ArchivedWorldSummary,
    file_name: String,
}

struct WorldArchiveStore {
    worlds_dir: PathBuf,
    by_id: StdRwLock<HashMap<Uuid, ArchivedWorldIndexEntry>>,
}

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
    reproductions_count: u64,
    consumptions_count: u64,
    energy: f32,
}

struct ChampionPoolStore {
    pool_path: PathBuf,
    entries: StdRwLock<Vec<ChampionGenomeRecord>>,
}

const STEP_PROGRESS_TARGET_BATCHES: u32 = 48;
const STEP_PROGRESS_MIN_BATCH_SIZE: u32 = 32;
const STEP_PROGRESS_MAX_BATCH_SIZE: u32 = 2_048;
const STEP_PROGRESS_TARGET_UPDATES: u32 = 64;
const DEFAULT_BATCH_TICKS_PER_WORLD: u64 = 1_000;
const UNBOUNDED_TICKS_PER_SECOND: u32 = 0;

fn step_batch_size(total_count: u32) -> u32 {
    let target = (total_count / STEP_PROGRESS_TARGET_BATCHES).max(1);
    target
        .clamp(STEP_PROGRESS_MIN_BATCH_SIZE, STEP_PROGRESS_MAX_BATCH_SIZE)
        .min(total_count.max(1))
}

fn step_progress_stride(total_count: u32) -> u32 {
    total_count.saturating_add(STEP_PROGRESS_TARGET_UPDATES.saturating_sub(1))
        / STEP_PROGRESS_TARGET_UPDATES.max(1)
}

fn now_unix_ms() -> Result<u128, AppError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| AppError::Internal(e.to_string()))
        .map(|duration| duration.as_millis())
}

fn archive_worlds_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("worlds")
}

fn champion_pool_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("champion_pool.json")
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn world_seed(universe_seed: u64, world_index: u32) -> u64 {
    splitmix64(universe_seed ^ (world_index as u64).wrapping_mul(0x9E37_79B9))
}

fn available_cpu_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .max(1)
}

fn batch_parallelism_plan(world_count: u32) -> usize {
    let total_cpu = available_cpu_parallelism();
    (world_count as usize).min(total_cpu).max(1)
}

fn load_runtime_default_world_config() -> Result<sim_types::WorldConfig, AppError> {
    load_default_world_config().map_err(|err| {
        AppError::Internal(format!(
            "failed to load {}: {err}",
            sim_server::default_world_config_path().display()
        ))
    })
}

fn simulate_ticks(simulation: &mut Simulation, ticks: u64) {
    let mut remaining = ticks;
    while remaining > 0 {
        let next_batch = remaining.min(u32::MAX as u64) as u32;
        simulation.advance_n(next_batch);
        remaining -= next_batch as u64;
    }
}

fn compute_batch_aggregate(worlds: &[ArchivedWorldSummary]) -> Option<BatchAggregateStats> {
    if worlds.is_empty() {
        return None;
    }

    let mut total_organisms_alive = 0_u64;
    let mut min_organisms_alive = u32::MAX;
    let mut max_organisms_alive = 0_u32;

    for world in worlds {
        total_organisms_alive = total_organisms_alive.saturating_add(world.organisms_alive as u64);
        min_organisms_alive = min_organisms_alive.min(world.organisms_alive);
        max_organisms_alive = max_organisms_alive.max(world.organisms_alive);
    }

    let world_count = worlds.len() as f64;
    Some(BatchAggregateStats {
        total_organisms_alive,
        mean_organisms_alive: total_organisms_alive as f64 / world_count,
        min_organisms_alive,
        max_organisms_alive,
    })
}

fn compare_champion_records(
    left: &ChampionGenomeRecord,
    right: &ChampionGenomeRecord,
) -> std::cmp::Ordering {
    right
        .generation
        .cmp(&left.generation)
        .then_with(|| right.reproductions_count.cmp(&left.reproductions_count))
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
        reproductions_count: organism.reproductions_count,
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

impl WorldArchiveStore {
    fn bootstrap(worlds_dir: PathBuf) -> Result<Self, AppError> {
        fs::create_dir_all(&worlds_dir).map_err(|err| {
            AppError::Internal(format!(
                "failed to create worlds archive directory {}: {err}",
                worlds_dir.display()
            ))
        })?;

        let mut by_id = HashMap::new();
        let read_dir = fs::read_dir(&worlds_dir).map_err(|err| {
            AppError::Internal(format!(
                "failed to read worlds archive directory {}: {err}",
                worlds_dir.display()
            ))
        })?;

        for entry in read_dir {
            let entry = match entry {
                Ok(item) => item,
                Err(err) => {
                    warn!("failed to read worlds archive entry: {err}");
                    continue;
                }
            };
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(err) => {
                    warn!("failed to read archive file type: {err}");
                    continue;
                }
            };
            if !file_type.is_file() {
                continue;
            }
            if entry.path().extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            let file_name = entry.file_name().to_string_lossy().to_string();
            let bytes = match fs::read(entry.path()) {
                Ok(contents) => contents,
                Err(err) => {
                    warn!("failed to read archive file {}: {err}", file_name);
                    continue;
                }
            };
            let archived_world = match serde_json::from_slice::<ArchivedWorldFile>(&bytes) {
                Ok(world) => world,
                Err(err) => {
                    warn!("failed to parse archive file {}: {err}", file_name);
                    continue;
                }
            };
            if archived_world.schema_version != WORLD_ARCHIVE_SCHEMA_VERSION {
                warn!(
                    "skipping archive {} with schema version {} (expected {})",
                    file_name, archived_world.schema_version, WORLD_ARCHIVE_SCHEMA_VERSION
                );
                continue;
            }

            by_id.insert(
                archived_world.world_id,
                ArchivedWorldIndexEntry {
                    summary: archived_world.summary,
                    file_name,
                },
            );
        }

        Ok(Self {
            worlds_dir,
            by_id: StdRwLock::new(by_id),
        })
    }

    fn path_for(&self, file_name: &str) -> PathBuf {
        self.worlds_dir.join(file_name)
    }

    fn load_world_file(&self, world_id: Uuid) -> Result<ArchivedWorldFile, AppError> {
        let entry =
            {
                let by_id = self.by_id.read().map_err(|_| {
                    AppError::Internal("failed to lock world archive index".to_owned())
                })?;
                by_id.get(&world_id).cloned().ok_or_else(|| {
                    AppError::NotFound(format!("archived world {world_id} not found"))
                })?
            };
        let bytes = fs::read(self.path_for(&entry.file_name)).map_err(|err| {
            AppError::Internal(format!(
                "failed to read archived world {} from disk: {err}",
                entry.file_name
            ))
        })?;
        let archived_world: ArchivedWorldFile = serde_json::from_slice(&bytes).map_err(|err| {
            AppError::Internal(format!(
                "failed to parse archived world {}: {err}",
                entry.file_name
            ))
        })?;
        if archived_world.schema_version != WORLD_ARCHIVE_SCHEMA_VERSION {
            return Err(AppError::BadRequest(format!(
                "archived world schema version {} is unsupported (expected {})",
                archived_world.schema_version, WORLD_ARCHIVE_SCHEMA_VERSION
            )));
        }
        Ok(archived_world)
    }

    fn persist_batch_world(
        &self,
        run_id: Uuid,
        world_index: u32,
        universe_seed: u64,
        world_seed: u64,
        ticks_simulated: u64,
        simulation: Simulation,
    ) -> Result<ArchivedWorldSummary, AppError> {
        let world_id = Uuid::new_v4();
        let created_at_unix_ms = now_unix_ms()?;
        let snapshot = simulation.snapshot();
        let summary = ArchivedWorldSummary {
            world_id,
            created_at_unix_ms,
            turn: snapshot.turn,
            organisms_alive: snapshot.metrics.organisms,
            source: ArchivedWorldSource::BatchRun {
                run_id,
                world_index,
                universe_seed,
                world_seed,
                ticks_simulated,
            },
        };
        let archived_world = ArchivedWorldFile {
            schema_version: WORLD_ARCHIVE_SCHEMA_VERSION,
            world_id,
            created_at_unix_ms,
            source: summary.source.clone(),
            summary: summary.clone(),
            simulation,
        };
        let file_name = format!("{created_at_unix_ms}_{world_id}.json");
        let temp_file_name = format!("{file_name}.tmp");
        let temp_path = self.path_for(&temp_file_name);
        let final_path = self.path_for(&file_name);

        let encoded = serde_json::to_vec_pretty(&archived_world).map_err(|err| {
            AppError::Internal(format!("failed to encode archived world {world_id}: {err}"))
        })?;
        fs::write(&temp_path, encoded).map_err(|err| {
            AppError::Internal(format!(
                "failed to write archived world temp file {}: {err}",
                temp_path.display()
            ))
        })?;
        fs::rename(&temp_path, &final_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to finalize archived world {}: {err}",
                final_path.display()
            ))
        })?;

        let mut by_id = self
            .by_id
            .write()
            .map_err(|_| AppError::Internal("failed to lock world archive index".to_owned()))?;
        by_id.insert(
            world_id,
            ArchivedWorldIndexEntry {
                summary: summary.clone(),
                file_name,
            },
        );
        Ok(summary)
    }

    fn persist_session_world(
        &self,
        session_id: Uuid,
        simulation: Simulation,
    ) -> Result<ArchivedWorldSummary, AppError> {
        let world_id = Uuid::new_v4();
        let created_at_unix_ms = now_unix_ms()?;
        let snapshot = simulation.snapshot();
        let summary = ArchivedWorldSummary {
            world_id,
            created_at_unix_ms,
            turn: snapshot.turn,
            organisms_alive: snapshot.metrics.organisms,
            source: ArchivedWorldSource::Session { session_id },
        };
        let archived_world = ArchivedWorldFile {
            schema_version: WORLD_ARCHIVE_SCHEMA_VERSION,
            world_id,
            created_at_unix_ms,
            source: summary.source.clone(),
            summary: summary.clone(),
            simulation,
        };
        let file_name = format!("{created_at_unix_ms}_{world_id}.json");
        let temp_file_name = format!("{file_name}.tmp");
        let temp_path = self.path_for(&temp_file_name);
        let final_path = self.path_for(&file_name);

        let encoded = serde_json::to_vec_pretty(&archived_world).map_err(|err| {
            AppError::Internal(format!("failed to encode archived world {world_id}: {err}"))
        })?;
        fs::write(&temp_path, encoded).map_err(|err| {
            AppError::Internal(format!(
                "failed to write archived world temp file {}: {err}",
                temp_path.display()
            ))
        })?;
        fs::rename(&temp_path, &final_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to finalize archived world {}: {err}",
                final_path.display()
            ))
        })?;

        let mut by_id = self
            .by_id
            .write()
            .map_err(|_| AppError::Internal("failed to lock world archive index".to_owned()))?;
        by_id.insert(
            world_id,
            ArchivedWorldIndexEntry {
                summary: summary.clone(),
                file_name,
            },
        );
        Ok(summary)
    }

    fn list_worlds(&self) -> Result<Vec<ArchivedWorldSummary>, AppError> {
        let by_id = self
            .by_id
            .read()
            .map_err(|_| AppError::Internal("failed to lock world archive index".to_owned()))?;
        let mut worlds: Vec<ArchivedWorldSummary> =
            by_id.values().map(|entry| entry.summary.clone()).collect();
        worlds.sort_by(|a, b| b.created_at_unix_ms.cmp(&a.created_at_unix_ms));
        Ok(worlds)
    }

    fn delete_world(&self, world_id: Uuid) -> Result<ArchivedWorldSummary, AppError> {
        let removed_entry = {
            let mut by_id = self
                .by_id
                .write()
                .map_err(|_| AppError::Internal("failed to lock world archive index".to_owned()))?;
            by_id
                .remove(&world_id)
                .ok_or_else(|| AppError::NotFound(format!("archived world {world_id} not found")))?
        };

        let path = self.path_for(&removed_entry.file_name);
        match fs::remove_file(&path) {
            Ok(()) => Ok(removed_entry.summary),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(removed_entry.summary),
            Err(err) => {
                let mut by_id = self.by_id.write().map_err(|_| {
                    AppError::Internal("failed to lock world archive index".to_owned())
                })?;
                by_id.insert(world_id, removed_entry);
                Err(AppError::Internal(format!(
                    "failed to delete archived world {}: {err}",
                    path.display()
                )))
            }
        }
    }
}

impl ChampionPoolStore {
    fn bootstrap(pool_path: PathBuf) -> Result<Self, AppError> {
        if let Some(parent) = pool_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                AppError::Internal(format!(
                    "failed to create champion pool directory {}: {err}",
                    parent.display()
                ))
            })?;
        }

        let entries = match fs::read(&pool_path) {
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
                )))
            }
        };

        Ok(Self {
            pool_path,
            entries: StdRwLock::new(entries),
        })
    }

    fn snapshot_genomes(&self) -> Result<Vec<OrganismGenome>, AppError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        Ok(entries.iter().map(|entry| entry.genome.clone()).collect())
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
        let file = ChampionPoolFile {
            schema_version: CHAMPION_POOL_SCHEMA_VERSION,
            updated_at_unix_ms: now_unix_ms()?,
            entries: entries.to_vec(),
        };
        let encoded = serde_json::to_vec_pretty(&file).map_err(|err| {
            AppError::Internal(format!(
                "failed to encode champion pool {}: {err}",
                self.pool_path.display()
            ))
        })?;

        let temp_path = self.pool_path.with_extension("json.tmp");
        fs::write(&temp_path, encoded).map_err(|err| {
            AppError::Internal(format!(
                "failed to write champion pool temp file {}: {err}",
                temp_path.display()
            ))
        })?;
        fs::rename(&temp_path, &self.pool_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to finalize champion pool {}: {err}",
                self.pool_path.display()
            ))
        })?;
        Ok(())
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct StepResponse {
    snapshot: WorldSnapshot,
}

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "sim_server=info,axum=info,tower_http=info".to_owned()),
        )
        .init();

    let app = build_app(new_state()?);

    let addr = std::env::var("SIM_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("sim-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn new_state() -> Result<AppState, AppError> {
    let world_archive = Arc::new(WorldArchiveStore::bootstrap(archive_worlds_dir())?);
    let champion_pool = Arc::new(ChampionPoolStore::bootstrap(champion_pool_path())?);
    Ok(AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        batch_runs: Arc::new(RwLock::new(HashMap::new())),
        world_archive,
        champion_pool,
    })
}

fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/world-runs", post(create_batch_run))
        .route("/v1/world-runs/{id}", get(get_batch_run_status))
        .route("/v1/worlds", get(list_archived_worlds))
        .route("/v1/worlds/{id}", delete(delete_archived_world))
        .route("/v1/worlds/{id}/sessions", post(create_session_from_world))
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/{id}", get(get_session_metadata))
        .route("/v1/sessions/{id}/state", get(get_state))
        .route("/v1/sessions/{id}/step", post(step_session))
        .route("/v1/sessions/{id}/reset", post(reset_session))
        .route("/v1/sessions/{id}/archive", post(save_session_world))
        .route("/v1/sessions/{id}/focus", post(set_focus))
        .route("/v1/sessions/{id}/stream", get(stream_session))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn create_batch_run(
    State(state): State<AppState>,
    Json(req): Json<CreateBatchRunRequest>,
) -> Result<Json<CreateBatchRunResponse>, AppError> {
    if req.world_count == 0 {
        return Err(AppError::BadRequest(
            "world_count must be greater than zero".to_owned(),
        ));
    }

    let ticks_per_world = if req.ticks_per_world == 0 {
        DEFAULT_BATCH_TICKS_PER_WORLD
    } else {
        req.ticks_per_world
    };
    let run_id = Uuid::new_v4();
    let created_at_unix_ms = now_unix_ms()?;
    let batch_run = Arc::new(BatchRun {
        id: run_id,
        created_at_unix_ms,
        total_worlds: req.world_count,
        inner: Mutex::new(BatchRunInner {
            completed_worlds: 0,
            status: BatchRunStatus::Running,
            worlds: Vec::with_capacity(req.world_count as usize),
            aggregate: None,
            error: None,
        }),
    });

    {
        let mut runs = state.batch_runs.write().await;
        runs.insert(run_id, batch_run.clone());
    }

    let config = load_runtime_default_world_config()?;
    let champion_pool = state.champion_pool.snapshot_genomes()?;

    tokio::spawn(run_batch_simulations(
        state.clone(),
        batch_run,
        config,
        champion_pool,
        req.world_count,
        req.universe_seed,
        ticks_per_world,
    ));

    Ok(Json(CreateBatchRunResponse { run_id }))
}

async fn get_batch_run_status(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<BatchRunStatusResponse>, AppError> {
    let run = {
        let runs = state.batch_runs.read().await;
        runs.get(&id)
            .cloned()
            .ok_or_else(|| AppError::NotFound(format!("world run {id} not found")))?
    };
    let inner = run.inner.lock().await;
    Ok(Json(BatchRunStatusResponse {
        run_id: run.id,
        created_at_unix_ms: run.created_at_unix_ms,
        status: inner.status.clone(),
        total_worlds: run.total_worlds,
        completed_worlds: inner.completed_worlds,
        aggregate: inner.aggregate.clone(),
        worlds: inner.worlds.clone(),
        error: inner.error.clone(),
    }))
}

async fn list_archived_worlds(
    State(state): State<AppState>,
) -> Result<Json<ListArchivedWorldsResponse>, AppError> {
    let worlds = state.world_archive.list_worlds()?;
    Ok(Json(ListArchivedWorldsResponse { worlds }))
}

async fn create_session_from_world(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let archived_world = state.world_archive.load_world_file(id)?;
    archived_world.simulation.validate_state()?;
    let response = create_runtime_session(&state, archived_world.simulation).await?;
    Ok(Json(response))
}

async fn delete_archived_world(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<ArchivedWorldSummary>, AppError> {
    let world_archive = state.world_archive.clone();
    let summary = tokio::task::spawn_blocking(move || world_archive.delete_world(id))
        .await
        .map_err(|err| AppError::Internal(format!("archive delete worker join error: {err}")))??;
    Ok(Json(summary))
}

async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let config = load_runtime_default_world_config()?;
    let champion_pool = state.champion_pool.snapshot_genomes()?;
    let simulation = Simulation::new_with_champion_pool(config, req.seed, champion_pool)?;
    let response = create_runtime_session(&state, simulation).await?;
    Ok(Json(response))
}

async fn create_runtime_session(
    state: &AppState,
    simulation: Simulation,
) -> Result<CreateSessionResponse, AppError> {
    let now_ms = now_unix_ms()?;
    let id = Uuid::new_v4();
    let snapshot = simulation.snapshot();
    let ticks_per_second = UNBOUNDED_TICKS_PER_SECOND;
    let metadata = SessionMetadata {
        id,
        created_at_unix_ms: now_ms,
        config: simulation.config().clone(),
        running: false,
        ticks_per_second,
    };

    let (events_tx, _events_rx) = broadcast::channel(1024);
    let session = Arc::new(Session {
        metadata: metadata.clone(),
        simulation: Mutex::new(simulation),
        events: events_tx,
        runtime: Mutex::new(RuntimeState {
            running: false,
            ticks_per_second,
            runner: None,
        }),
    });

    let mut sessions = state.sessions.write().await;
    sessions.insert(id, session);

    Ok(CreateSessionResponse {
        metadata,
        snapshot: snapshot.into(),
    })
}

async fn run_batch_simulations(
    state: AppState,
    run: Arc<BatchRun>,
    config: sim_types::WorldConfig,
    champion_pool: Vec<OrganismGenome>,
    world_count: u32,
    universe_seed: u64,
    ticks_per_world: u64,
) {
    let max_world_workers = batch_parallelism_plan(world_count);
    let mut join_set = JoinSet::new();

    for world_index in 0..world_count {
        let run_id = run.id;
        let world_archive = state.world_archive.clone();
        let champion_pool_store = state.champion_pool.clone();
        let world_config = config.clone();
        let champion_pool = champion_pool.clone();
        let seed = world_seed(universe_seed, world_index);
        join_set.spawn(async move {
            tokio::task::spawn_blocking(move || -> Result<ArchivedWorldSummary, AppError> {
                let mut simulation =
                    Simulation::new_with_champion_pool(world_config, seed, champion_pool)?;
                simulate_ticks(&mut simulation, ticks_per_world);
                if let Err(err) = champion_pool_store.update_from_simulation(&simulation) {
                    warn!("failed to update champion pool from batch world {world_index}: {err}");
                }
                world_archive.persist_batch_world(
                    run_id,
                    world_index,
                    universe_seed,
                    seed,
                    ticks_per_world,
                    simulation,
                )
            })
            .await
            .map_err(|err| AppError::Internal(format!("world worker join error: {err}")))?
        });

        if join_set.len() >= max_world_workers {
            let completed = join_set.join_next().await;
            if handle_batch_world_result(run.clone(), completed)
                .await
                .is_err()
            {
                join_set.abort_all();
                return;
            }
        }
    }

    while !join_set.is_empty() {
        let completed = join_set.join_next().await;
        if handle_batch_world_result(run.clone(), completed)
            .await
            .is_err()
        {
            join_set.abort_all();
            return;
        }
    }

    let mut inner = run.inner.lock().await;
    if inner.error.is_none() {
        inner.worlds.sort_by(|a, b| match (&a.source, &b.source) {
            (
                ArchivedWorldSource::BatchRun {
                    world_index: left_index,
                    ..
                },
                ArchivedWorldSource::BatchRun {
                    world_index: right_index,
                    ..
                },
            ) => left_index.cmp(right_index),
            _ => std::cmp::Ordering::Equal,
        });
        inner.aggregate = compute_batch_aggregate(&inner.worlds);
        inner.status = BatchRunStatus::Completed;
    }
}

async fn handle_batch_world_result(
    run: Arc<BatchRun>,
    completed: Option<Result<Result<ArchivedWorldSummary, AppError>, tokio::task::JoinError>>,
) -> Result<(), ()> {
    let Some(result) = completed else {
        return Ok(());
    };
    match result {
        Ok(Ok(summary)) => {
            let mut inner = run.inner.lock().await;
            inner.completed_worlds = inner.completed_worlds.saturating_add(1);
            inner.worlds.push(summary);
            Ok(())
        }
        Ok(Err(err)) => {
            let mut inner = run.inner.lock().await;
            inner.status = BatchRunStatus::Failed;
            inner.error = Some(err.to_string());
            Err(())
        }
        Err(err) => {
            let mut inner = run.inner.lock().await;
            inner.status = BatchRunStatus::Failed;
            inner.error = Some(format!("world worker task failed: {err}"));
            Err(())
        }
    }
}

async fn get_session_metadata(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<SessionMetadata>, AppError> {
    let session = get_session(&state, id).await?;
    let mut metadata = session.metadata.clone();
    let runtime = session.runtime.lock().await;
    metadata.running = runtime.running;
    metadata.ticks_per_second = runtime.ticks_per_second;
    Ok(Json(metadata))
}

async fn get_state(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let session = get_session(&state, id).await?;
    let sim = session.simulation.lock().await;
    Ok(Json(sim.snapshot().into()))
}

async fn step_session(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<CountRequest>,
) -> Result<Json<StepResponse>, AppError> {
    let session = get_session(&state, id).await?;
    let mut sim = session.simulation.lock().await;
    sim.advance_n(req.count.max(1));
    let snapshot = sim.snapshot();
    drop(sim);

    let _ = session
        .events
        .send(ServerEvent::StateSnapshot(snapshot.clone().into()));

    Ok(Json(StepResponse { snapshot }))
}

async fn reset_session(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<ResetRequest>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let session = get_session(&state, id).await?;
    let previous_sim = {
        let sim = session.simulation.lock().await;
        sim.clone()
    };
    let champion_pool_store = state.champion_pool.clone();
    let champion_pool = tokio::task::spawn_blocking(move || {
        if let Err(err) = champion_pool_store.update_from_simulation(&previous_sim) {
            warn!("failed to update champion pool before reset for session {id}: {err}");
        }
        champion_pool_store.snapshot_genomes()
    })
    .await
    .map_err(|err| AppError::Internal(format!("session reset worker join error: {err}")))??;

    let mut sim = session.simulation.lock().await;
    sim.reset_with_champion_pool(req.seed, champion_pool);
    let snapshot = sim.snapshot();
    drop(sim);

    let _ = session
        .events
        .send(ServerEvent::StateSnapshot(snapshot.clone().into()));

    Ok(Json(snapshot.into()))
}

async fn save_session_world(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<ArchivedWorldSummary>, AppError> {
    let session = get_session(&state, id).await?;
    let sim = {
        let sim_guard = session.simulation.lock().await;
        sim_guard.clone()
    };

    let world_archive = state.world_archive.clone();
    let champion_pool = state.champion_pool.clone();
    let summary = tokio::task::spawn_blocking(move || {
        if let Err(err) = champion_pool.update_from_simulation(&sim) {
            warn!("failed to update champion pool from session {id}: {err}");
        }
        world_archive.persist_session_world(id, sim)
    })
    .await
    .map_err(|err| AppError::Internal(format!("session archive worker join error: {err}")))??;
    Ok(Json(summary))
}

async fn set_focus(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<FocusRequest>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let session = get_session(&state, id).await?;
    let sim = session.simulation.lock().await;
    if let Some(org) = sim.focused_organism(req.organism_id) {
        let active = derive_active_action_neuron_id(&org);
        let _ = session.events.send(ServerEvent::FocusBrain(FocusBrainData {
            turn: sim.turn(),
            organism: org,
            active_action_neuron_id: active,
        }));
    }
    let snapshot = sim.snapshot();
    drop(sim);

    Ok(Json(snapshot.into()))
}

async fn stream_session(
    ws: WebSocketUpgrade,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    let session = get_session(&state, id).await?;
    Ok(ws
        .on_upgrade(move |socket| socket_loop(socket, session))
        .into_response())
}

async fn socket_loop(socket: WebSocket, session: Arc<Session>) {
    let mut rx = session.events.subscribe();
    let (mut sender, mut receiver) = socket.split();

    {
        let sim = session.simulation.lock().await;
        let event = ServerEvent::StateSnapshot(sim.snapshot().into());
        if let Ok(text) = serde_json::to_string(&event) {
            if sender.send(Message::Text(text.into())).await.is_err() {
                return;
            }
        }
    }

    let session_for_send = session.clone();
    let send_task = tokio::spawn(async move {
        loop {
            let event = match rx.recv().await {
                Ok(event) => event,
                Err(RecvError::Lagged(skipped)) => {
                    error!("ws receiver lagged by {skipped} events; sending state snapshot");
                    let snapshot_event = {
                        let sim = session_for_send.simulation.lock().await;
                        ServerEvent::StateSnapshot(sim.snapshot().into())
                    };
                    if send_ws_event(&mut sender, &snapshot_event).await.is_err() {
                        break;
                    }
                    continue;
                }
                Err(RecvError::Closed) => break,
            };

            if send_ws_event(&mut sender, &event).await.is_err() {
                break;
            }
        }
    });

    while let Some(message) = receiver.next().await {
        match message {
            Ok(Message::Text(text)) => {
                let command = match serde_json::from_str::<ClientCommand>(&text) {
                    Ok(cmd) => cmd,
                    Err(err) => {
                        let _ = session.events.send(ServerEvent::Error(ApiError {
                            code: "bad_command".to_owned(),
                            message: format!("failed to parse command: {err}"),
                        }));
                        continue;
                    }
                };

                if let Err(err) = handle_command(command, session.clone()).await {
                    let _ = session.events.send(ServerEvent::Error(ApiError {
                        code: "command_error".to_owned(),
                        message: err,
                    }));
                }
            }
            Ok(Message::Binary(_)) => {}
            Ok(Message::Ping(_)) => {}
            Ok(Message::Pong(_)) => {}
            Ok(Message::Close(_)) => break,
            Err(err) => {
                error!("ws receive error: {err}");
                break;
            }
        }
    }

    send_task.abort();
}

async fn handle_command(command: ClientCommand, session: Arc<Session>) -> Result<(), String> {
    match command {
        ClientCommand::Start { ticks_per_second } => {
            session_start(session, ticks_per_second).await;
            Ok(())
        }
        ClientCommand::Pause => {
            session_pause(&session).await;
            Ok(())
        }
        ClientCommand::Step { count } => {
            let requested_count = count.max(1);
            let _ = session
                .events
                .send(ServerEvent::StepProgress(StepProgressData {
                    requested_count,
                    completed_count: 0,
                }));

            let batch_size = step_batch_size(requested_count);
            let progress_stride = step_progress_stride(requested_count).max(1);
            let mut sim = session.simulation.lock().await;
            let mut completed_count = 0;
            let mut next_progress_emit = progress_stride;
            while completed_count < requested_count {
                let remaining = requested_count.saturating_sub(completed_count);
                let batch_count = remaining.min(batch_size);
                sim.advance_n(batch_count);
                completed_count = completed_count.saturating_add(batch_count);

                let is_final_batch = completed_count == requested_count;
                if is_final_batch || completed_count >= next_progress_emit {
                    next_progress_emit = completed_count.saturating_add(progress_stride);
                    let _ = session
                        .events
                        .send(ServerEvent::StepProgress(StepProgressData {
                            requested_count,
                            completed_count,
                        }));
                    tokio::task::yield_now().await;
                }
            }
            let snapshot = sim.snapshot();
            drop(sim);
            let _ = session
                .events
                .send(ServerEvent::StateSnapshot(snapshot.into()));
            Ok(())
        }
        ClientCommand::SetFocus { organism_id } => {
            let sim = session.simulation.lock().await;
            if let Some(organism) = sim.focused_organism(organism_id) {
                let active = derive_active_action_neuron_id(&organism);
                let _ = session.events.send(ServerEvent::FocusBrain(FocusBrainData {
                    turn: sim.turn(),
                    organism,
                    active_action_neuron_id: active,
                }));
            }
            Ok(())
        }
    }
}

async fn session_start(session: Arc<Session>, ticks_per_second: u32) {
    let mut runtime = session.runtime.lock().await;
    runtime.ticks_per_second = ticks_per_second;

    if runtime.running {
        return;
    }

    runtime.running = true;
    let session_for_task = session.clone();
    runtime.runner = Some(tokio::spawn(async move {
        loop {
            let tps = {
                let rt = session_for_task.runtime.lock().await;
                if !rt.running {
                    break;
                }
                rt.ticks_per_second
            };

            let delta = {
                let mut sim = session_for_task.simulation.lock().await;
                sim.step_n(1).into_iter().next()
            };

            if let Some(delta) = delta {
                let _ = session_for_task
                    .events
                    .send(ServerEvent::TickDelta(delta.into()));
            }

            if tps > 0 {
                tokio::time::sleep(Duration::from_millis((1000_u64 / tps as u64).max(1))).await;
            } else {
                tokio::task::yield_now().await;
            }
        }

        let mut rt = session_for_task.runtime.lock().await;
        rt.running = false;
        rt.runner = None;
    }));
}

async fn send_ws_event(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    event: &ServerEvent,
) -> Result<(), ()> {
    match serde_json::to_string(event) {
        Ok(text) => sender
            .send(Message::Text(text.into()))
            .await
            .map_err(|_| ()),
        Err(err) => {
            error!("failed to serialize server event: {err}");
            Ok(())
        }
    }
}

async fn session_pause(session: &Arc<Session>) {
    let mut runtime = session.runtime.lock().await;
    runtime.running = false;
    if let Some(handle) = runtime.runner.take() {
        handle.abort();
    }
}

async fn get_session(state: &AppState, id: Uuid) -> Result<Arc<Session>, AppError> {
    let sessions = state.sessions.read().await;
    sessions
        .get(&id)
        .cloned()
        .ok_or_else(|| AppError::NotFound(format!("session {id} not found")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{ActionType, BrainState, FacingDirection, OrganismId, SpeciesId};

    fn make_record(
        generation: u64,
        reproductions: u64,
        consumptions: u64,
        energy: f32,
    ) -> ChampionGenomeRecord {
        let mut genome = OrganismGenome {
            num_neurons: 1,
            num_synapses: 0,
            spatial_prior_sigma: 3.5,
            vision_distance: 2,
            starting_energy: energy.max(1.0),
            age_of_maturity: 0,
            hebb_eta_gain: 0.0,
            eligibility_retention: 0.9,
            synapse_prune_threshold: 0.01,
            mutation_rate_age_of_maturity: 0.0,
            mutation_rate_vision_distance: 0.0,
            mutation_rate_inter_bias: 0.0,
            mutation_rate_inter_update_rate: 0.0,
            mutation_rate_eligibility_retention: 0.0,
            mutation_rate_synapse_prune_threshold: 0.0,
            mutation_rate_neuron_location: 0.0,
            mutation_rate_synapse_weight_perturbation: 0.0,
            mutation_rate_add_synapse: 0.0,
            mutation_rate_remove_synapse: 0.0,
            mutation_rate_remove_neuron: 0.0,
            mutation_rate_add_neuron_split_edge: 0.0,
            inter_biases: vec![0.0],
            inter_log_time_constants: vec![0.0],
            sensory_locations: vec![sim_types::BrainLocation { x: 0.0, y: 0.0 }; 2],
            inter_locations: vec![sim_types::BrainLocation { x: 1.0, y: 1.0 }],
            action_locations: vec![sim_types::BrainLocation { x: 2.0, y: 2.0 }; 5],
            edges: Vec::new(),
        };
        genome.num_neurons = generation as u32 + 1;
        genome.vision_distance = generation as u32 + 2;
        genome.starting_energy = energy.max(1.0);
        genome.inter_biases = vec![generation as f32];
        genome.inter_log_time_constants = vec![generation as f32 * 0.1];
        genome.inter_locations = vec![sim_types::BrainLocation {
            x: generation as f32,
            y: generation as f32,
        }];

        ChampionGenomeRecord {
            genome,
            source_turn: generation * 10,
            source_created_at_unix_ms: generation as u128,
            generation,
            age_turns: generation * 2,
            reproductions_count: reproductions,
            consumptions_count: consumptions,
            energy,
        }
    }

    #[test]
    fn merge_champion_entries_prefers_higher_ranked_unique_genomes() {
        let weak = make_record(3, 1, 1, 10.0);
        let strong = make_record(8, 5, 9, 80.0);
        let duplicate_strong = strong.clone();

        let merged = merge_champion_entries(&[weak.clone()], &[duplicate_strong, strong.clone()]);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], strong);
        assert_eq!(merged[1], weak);
    }

    #[test]
    fn select_champion_candidates_deduplicates_identical_genomes() {
        let strong = make_record(10, 4, 2, 40.0);
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
        };

        let organisms = vec![
            OrganismState {
                id: OrganismId(1),
                species_id: SpeciesId(1),
                q: 0,
                r: 0,
                generation: strong.generation,
                age_turns: strong.age_turns,
                facing: FacingDirection::East,
                energy: strong.energy,
                energy_prev: strong.energy,
                dopamine: 0.0,
                consumptions_count: strong.consumptions_count,
                reproductions_count: strong.reproductions_count,
                last_action_taken: ActionType::Idle,
                brain: empty_brain.clone(),
                genome: strong.genome.clone(),
            },
            OrganismState {
                id: OrganismId(2),
                species_id: SpeciesId(2),
                q: 1,
                r: 0,
                generation: weaker_duplicate.generation,
                age_turns: weaker_duplicate.age_turns,
                facing: FacingDirection::East,
                energy: weaker_duplicate.energy,
                energy_prev: weaker_duplicate.energy,
                dopamine: 0.0,
                consumptions_count: weaker_duplicate.consumptions_count,
                reproductions_count: weaker_duplicate.reproductions_count,
                last_action_taken: ActionType::Idle,
                brain: empty_brain,
                genome: weaker_duplicate.genome,
            },
        ];

        let candidates = select_champion_candidates(1, 100, &organisms);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].generation, strong.generation);
        assert_eq!(candidates[0].energy, strong.energy);
    }
}
