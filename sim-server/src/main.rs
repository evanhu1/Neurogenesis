use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use sim_core::{derive_active_neuron_ids, SimError, Simulation};
use sim_server::protocol::{
    ApiError, ClientCommand, CountRequest, CreateSessionRequest, CreateSessionResponse,
    FocusBrainData, FocusRequest, ResetRequest, ServerEvent, SessionMetadata, StepProgressData,
    WorldSnapshotView,
};
use sim_types::WorldSnapshot;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tracing::{error, info};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    sessions: Arc<RwLock<HashMap<Uuid, Arc<Session>>>>,
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

const STEP_PROGRESS_TARGET_BATCHES: u32 = 48;
const STEP_PROGRESS_MIN_BATCH_SIZE: u32 = 32;
const STEP_PROGRESS_MAX_BATCH_SIZE: u32 = 2_048;
const STEP_PROGRESS_TARGET_UPDATES: u32 = 64;

fn step_batch_size(total_count: u32) -> u32 {
    let target = (total_count / STEP_PROGRESS_TARGET_BATCHES).max(1);
    target
        .clamp(STEP_PROGRESS_MIN_BATCH_SIZE, STEP_PROGRESS_MAX_BATCH_SIZE)
        .min(total_count.max(1))
}

fn step_progress_stride(total_count: u32) -> u32 {
    total_count
        .saturating_add(STEP_PROGRESS_TARGET_UPDATES.saturating_sub(1))
        / STEP_PROGRESS_TARGET_UPDATES.max(1)
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

    let app = build_app(new_state());

    let addr = std::env::var("SIM_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("sim-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn new_state() -> AppState {
    AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
    }
}

fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/{id}", get(get_session_metadata))
        .route("/v1/sessions/{id}/state", get(get_state))
        .route("/v1/sessions/{id}/step", post(step_session))
        .route("/v1/sessions/{id}/reset", post(reset_session))
        .route("/v1/sessions/{id}/focus", post(set_focus))
        .route("/v1/sessions/{id}/stream", get(stream_session))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| AppError::Internal(e.to_string()))?
        .as_millis();
    let id = Uuid::new_v4();

    let simulation = Simulation::new(req.config.clone(), req.seed)?;
    let snapshot = simulation.snapshot();

    let metadata = SessionMetadata {
        id,
        created_at_unix_ms: now_ms,
        config: req.config,
        running: false,
        ticks_per_second: simulation.config().steps_per_second,
    };

    let (events_tx, _events_rx) = broadcast::channel(1024);
    let session = Arc::new(Session {
        metadata: metadata.clone(),
        simulation: Mutex::new(simulation),
        events: events_tx,
        runtime: Mutex::new(RuntimeState {
            running: false,
            ticks_per_second: metadata.config.steps_per_second,
            runner: None,
        }),
    });

    let mut sessions = state.sessions.write().await;
    sessions.insert(id, session);

    Ok(Json(CreateSessionResponse {
        metadata,
        snapshot: snapshot.into(),
    }))
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
    let mut sim = session.simulation.lock().await;
    sim.reset(req.seed);
    let snapshot = sim.snapshot();
    drop(sim);

    let _ = session
        .events
        .send(ServerEvent::StateSnapshot(snapshot.clone().into()));

    Ok(Json(snapshot.into()))
}

async fn set_focus(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<FocusRequest>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let session = get_session(&state, id).await?;
    let sim = session.simulation.lock().await;
    if let Some(org) = sim.focused_organism(req.organism_id) {
        let active = derive_active_neuron_ids(&org.brain);
        let _ = session.events.send(ServerEvent::FocusBrain(FocusBrainData {
            organism: org,
            active_neuron_ids: active,
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
            session_start(session, ticks_per_second.max(1)).await;
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
                let active = derive_active_neuron_ids(&organism.brain);
                let _ = session.events.send(ServerEvent::FocusBrain(FocusBrainData {
                    organism,
                    active_neuron_ids: active,
                }));
            }
            Ok(())
        }
    }
}

async fn session_start(session: Arc<Session>, ticks_per_second: u32) {
    let mut runtime = session.runtime.lock().await;
    runtime.ticks_per_second = ticks_per_second.max(1);

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
                rt.ticks_per_second.max(1)
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

            tokio::time::sleep(Duration::from_millis((1000_u64 / tps as u64).max(1))).await;
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
