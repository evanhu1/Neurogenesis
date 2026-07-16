//! sim-server — a thin, stateless file-server over `world.bin` files.
//!
//! A world is a file on disk under `--world-root`; every request loads it, runs
//! one command (mirroring the cli verbs via the shared `views` crate),
//! and — for mutating commands — saves it back. The one stateful surface is the
//! `/worlds/{name}/stream` WebSocket, which holds a world resident only for the
//! duration of a live animation feed and persists it on disconnect. There is no
//! session registry: the file IS the durable state.

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use sim_server::protocol::{ApiError, OrganismDetail, StreamFrame, WorldSnapshotView};
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use types::OrganismGenome;
use uuid::Uuid;
use views::{ReadCtx, Recorder};
use world_sim::{SimError, Simulation};

/// Reporting-interval width minted for freshly-created worlds and assumed for
/// worlds whose sidecar is absent. Matches the cli / eval default.
const DEFAULT_REPORT_EVERY: u64 = 10_000;

#[derive(Clone)]
struct AppState {
    world_root: Arc<PathBuf>,
    founder_genome_pool: Arc<Vec<OrganismGenome>>,
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
    let sim = views::load_world(&path_str)
        .map_err(|e| AppError::BadRequest(format!("loading world `{name}`: {e}")))?;
    let sidecar = views::sibling_metrics_path(&path_str);
    let (report_every, recorder) = if FsPath::new(&sidecar).exists() {
        let (report_every, recorder) = views::load_sidecar(&sidecar)
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
    views::save_world(sim, &path_str)
        .map_err(|e| AppError::Internal(format!("saving world `{name}`: {e}")))?;
    if let Some(rec) = recorder {
        let sidecar = views::sibling_metrics_path(&path_str);
        views::save_sidecar(report_every, rec, &sidecar)
            .map_err(|e| AppError::Internal(format!("saving metrics for `{name}`: {e}")))?;
    }
    Ok(())
}

/// Build a JSON `Response` from bytes a `views` read wrote (already a
/// complete JSON document + trailing newline).
fn json_bytes_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

/// Run a `views` read against a loaded world, capturing its JSON output.
fn run_read(
    loaded: &Loaded,
    f: impl FnOnce(&ReadCtx, &mut Vec<u8>) -> anyhow::Result<()>,
) -> Result<Vec<u8>, AppError> {
    let ctx = ReadCtx {
        sim: &loaded.sim,
        recorder: loaded.recorder.as_ref(),
        report_every: loaded.report_every,
        format: views::output::Format::Json,
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
    /// Inline `key=value` config overrides (same vocabulary as `cli --set`).
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
    /// Bincode-encoded `OrganismGenome` snapshot. When set,
    /// every initial organism spawns with this genome.
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

    let founder_genome_pool = if let Some(snapshot_path) = &cli.seed_genome_snapshot {
        let bytes = std::fs::read(snapshot_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to read seed genome snapshot {}: {err}",
                snapshot_path.display()
            ))
        })?;
        let genome = bincode::deserialize(&bytes).map_err(|err| {
            AppError::Internal(format!(
                "failed to decode seed genome snapshot {}: {err}",
                snapshot_path.display()
            ))
        })?;
        info!("seeding worlds from snapshot {}", snapshot_path.display());
        vec![genome]
    } else {
        Vec::new()
    };
    Ok(AppState {
        world_root: Arc::new(cli.world_root.clone()),
        founder_genome_pool: Arc::new(founder_genome_pool),
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
        .route("/worlds/{name}/state", get(read_state))
        .route("/worlds/{name}/turn", get(read_turn))
        .route("/worlds/{name}/pillars", get(read_pillars))
        .route("/worlds/{name}/eco", get(read_eco))
        .route("/worlds/{name}/lineage", get(read_lineage))
        .route("/worlds/{name}/genome", get(read_genome))
        .route("/worlds/{name}/timeseries", get(read_timeseries))
        .route("/worlds/{name}/inspect/{id}", get(read_inspect))
        .route("/worlds/{name}/brain/{id}", get(read_brain))
        .route("/worlds/{name}/decide/{id}", get(read_decide))
        .route("/worlds/{name}/top/{field}", get(read_top))
        .route("/worlds/{name}/hist/{field}", get(read_hist))
        .route("/worlds/{name}/find", get(read_find))
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
    let founder_genome_pool = state.founder_genome_pool.clone();
    let response =
        tokio::task::spawn_blocking(move || build_new_world(&root, &founder_genome_pool, req))
            .await
            .map_err(|e| AppError::Internal(format!("create worker join error: {e}")))??;
    Ok(Json(response))
}

fn build_new_world(
    root: &FsPath,
    founder_genome_pool: &[OrganismGenome],
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

    let mut config = views::world_config_with_overrides(&config_path, &sets)
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    if let Some(threads) = req.threads {
        config.intent_parallel_threads = threads;
    }
    if let Some([width, pop]) = req.scale {
        config.world_width = width;
        config.num_organisms = pop;
    }
    let report_every = req.report_every.unwrap_or(DEFAULT_REPORT_EVERY);
    if report_every == 0 {
        return Err(AppError::BadRequest(
            "report_every must be greater than zero".to_owned(),
        ));
    }
    let sim = Simulation::new_with_founder_genome_pool(
        config,
        req.seed.unwrap_or(0),
        founder_genome_pool.to_vec(),
    )?;
    let recorder = views::start_recording(&sim, report_every);
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
            .focused_organism(types::OrganismId(id))
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
        if loaded
            .recorder
            .as_ref()
            .is_some_and(|rec| rec.recorded_through_turn != loaded.sim.turn())
        {
            loaded.recorder = Some(views::start_recording(&loaded.sim, loaded.report_every));
        }
        views::advance(
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
        if loaded
            .recorder
            .as_ref()
            .is_some_and(|rec| rec.recorded_through_turn != loaded.sim.turn())
        {
            loaded.recorder = Some(views::start_recording(&loaded.sim, loaded.report_every));
        }
        let current = loaded.sim.turn();
        if req.turn > current {
            views::advance(
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
// CLI-parity reads (forwarded from views as raw JSON)
// ---------------------------------------------------------------------------

async fn read_state(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| views::state(c, &[], b))
    })
    .await
}

async fn read_turn(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| views::turn(c, &[], b))
    })
    .await
}

async fn read_pillars(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| views::pillars(c, &[], b))
    })
    .await
}

async fn read_eco(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| views::eco(c, &[], b))
    })
    .await
}

async fn read_lineage(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, |l| {
        run_read(l, |c, b| views::lineage(c, &[], b))
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
        run_read(l, |c, b| views::genome(c, &args, b))
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
        run_read(l, |c, b| views::timeseries(c, &args, b))
    })
    .await
}

async fn read_inspect(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let id = id.to_string();
        run_read(l, |c, b| views::inspect(c, &[&id], b))
    })
    .await
}

async fn read_decide(
    Path((name, id)): Path<(String, u64)>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        let id = id.to_string();
        run_read(l, |c, b| views::decide(c, &[&id], b))
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
        run_read(l, |c, b| views::brain(c, &args, b))
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
        run_read(l, |c, b| views::top(c, &args, b))
    })
    .await
}

async fn read_hist(
    Path((name, field)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    blocking_read(state.world_root.clone(), name, move |l| {
        run_read(l, |c, b| views::hist(c, &[&field], b))
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
        run_read(l, |c, b| views::find(c, &args, b))
    })
    .await
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
    if recorder
        .as_ref()
        .is_some_and(|rec| rec.recorded_through_turn != sim.turn())
    {
        recorder = Some(views::start_recording(&sim, report_every));
    }

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
                let delta = views::tick_recording(&mut sim, recorder.as_mut());
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
