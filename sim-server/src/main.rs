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
    FocusBrainData, FocusRequest, ResetRequest, ServerEvent, SessionMetadata, WorldSnapshotView,
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

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct StepResponse {
    deltas: Vec<sim_types::TickDelta>,
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
    Ok(Json(session.metadata.clone()))
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
    let deltas = sim.step_n(req.count.max(1));
    let snapshot = sim.snapshot();
    drop(sim);

    for delta in &deltas {
        let _ = session
            .events
            .send(ServerEvent::TickDelta(delta.clone().into()));
    }

    Ok(Json(StepResponse { deltas, snapshot }))
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
            let mut sim = session.simulation.lock().await;
            let deltas = sim.step_n(count.max(1));
            drop(sim);
            for delta in deltas {
                let _ = session.events.send(ServerEvent::TickDelta(delta.into()));
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{SinkExt, StreamExt};
    use reqwest::Client;
    use sim_server::protocol::{CreateSessionRequest, ServerEvent};
    use sim_types::WorldConfig;
    use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

    async fn spawn_test_server() -> (String, tokio::task::JoinHandle<()>) {
        let app = build_app(new_state());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("read local addr");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn api_create_and_step_flow() {
        let (base, handle) = spawn_test_server().await;
        let client = Client::new();

        let create = client
            .post(format!("{base}/v1/sessions"))
            .json(&CreateSessionRequest {
                config: WorldConfig::default(),
                seed: 42,
            })
            .send()
            .await
            .expect("create session request should succeed");
        assert_eq!(create.status(), StatusCode::OK);

        let payload: CreateSessionResponse = create
            .json()
            .await
            .expect("create session response should deserialize");
        let session_id = payload.metadata.id;

        let state = client
            .get(format!("{base}/v1/sessions/{session_id}/state"))
            .send()
            .await
            .expect("state request should succeed");
        assert_eq!(state.status(), StatusCode::OK);

        let step = client
            .post(format!("{base}/v1/sessions/{session_id}/step"))
            .json(&CountRequest { count: 2 })
            .send()
            .await
            .expect("step request should succeed");
        assert_eq!(step.status(), StatusCode::OK);

        handle.abort();
    }

    #[tokio::test]
    async fn websocket_receives_snapshot_and_tick_delta() {
        let (base, handle) = spawn_test_server().await;
        let client = Client::new();

        let create = client
            .post(format!("{base}/v1/sessions"))
            .json(&CreateSessionRequest {
                config: WorldConfig::default(),
                seed: 7,
            })
            .send()
            .await
            .expect("create session request should succeed");
        let payload: CreateSessionResponse = create
            .json()
            .await
            .expect("create session response should deserialize");
        let session_id = payload.metadata.id;

        let ws_url = format!(
            "{}/v1/sessions/{session_id}/stream",
            base.replace("http://", "ws://")
        );
        let (mut stream, _) = connect_async(ws_url)
            .await
            .expect("websocket connect should succeed");

        let first = stream
            .next()
            .await
            .expect("expected first ws message")
            .expect("first ws message should be ok");
        let text = match first {
            WsMessage::Text(t) => t,
            other => panic!("unexpected ws message: {other:?}"),
        };
        let event: ServerEvent =
            serde_json::from_str(&text).expect("initial server event should deserialize");
        assert!(matches!(event, ServerEvent::StateSnapshot(_)));

        let cmd = ClientCommand::Step { count: 1 };
        stream
            .send(WsMessage::Text(
                serde_json::to_string(&cmd)
                    .expect("serialize ws command")
                    .into(),
            ))
            .await
            .expect("send step command");

        let mut saw_delta = false;
        for _ in 0..4 {
            if let Some(Ok(WsMessage::Text(t))) = stream.next().await {
                let event: ServerEvent =
                    serde_json::from_str(&t).expect("ws server event should deserialize");
                if matches!(event, ServerEvent::TickDelta(_)) {
                    saw_delta = true;
                    break;
                }
            }
        }

        assert!(saw_delta, "expected tick delta event after step command");
        handle.abort();
    }
}
