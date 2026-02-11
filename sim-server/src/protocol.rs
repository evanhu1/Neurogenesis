use serde::{Deserialize, Serialize};
use sim_types::{
    MetricsSnapshot, NeuronId, OrganismId, OrganismState, TickDelta, WorldConfig, WorldSnapshot,
};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionMetadata {
    pub id: Uuid,
    pub created_at_unix_ms: u128,
    pub config: WorldConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateSessionRequest {
    pub config: WorldConfig,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateSessionResponse {
    pub metadata: SessionMetadata,
    pub snapshot: WorldSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CountRequest {
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResetRequest {
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FocusRequest {
    pub organism_id: OrganismId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
pub enum ClientCommand {
    Start { ticks_per_second: u32 },
    Pause,
    Step { count: u32 },
    SetFocus { organism_id: OrganismId },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FocusBrainData {
    pub organism: OrganismState,
    pub active_neuron_ids: Vec<NeuronId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
pub enum ServerEvent {
    StateSnapshot(WorldSnapshot),
    TickDelta(TickDelta),
    FocusBrain(FocusBrainData),
    Metrics(MetricsSnapshot),
    Error(ApiError),
}
