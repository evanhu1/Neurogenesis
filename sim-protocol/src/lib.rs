use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrganismId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Envelope<T> {
    pub protocol_version: u32,
    pub payload: T,
}

impl<T> Envelope<T> {
    pub fn new(payload: T) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            payload,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    MoveForward,
    TurnLeft,
    TurnRight,
}

impl ActionType {
    pub const ALL: [ActionType; 3] = [
        ActionType::MoveForward,
        ActionType::TurnLeft,
        ActionType::TurnRight,
    ];
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FacingDirection {
    East,
    NorthEast,
    NorthWest,
    West,
    SouthWest,
    SouthEast,
}

impl FacingDirection {
    pub const ALL: [FacingDirection; 6] = [
        FacingDirection::East,
        FacingDirection::NorthEast,
        FacingDirection::NorthWest,
        FacingDirection::West,
        FacingDirection::SouthWest,
        FacingDirection::SouthEast,
    ];
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NeuronType {
    Sensory,
    Inter,
    Action,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SensoryReceptorType {
    Look,
}

impl SensoryReceptorType {
    pub const ALL: [SensoryReceptorType; 1] = [SensoryReceptorType::Look];
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub steps_per_second: u32,
    pub num_organisms: u32,
    pub num_neurons: u32,
    pub max_num_neurons: u32,
    pub num_synapses: u32,
    pub turns_to_starve: u32,
    pub mutation_chance: f32,
    pub mutation_magnitude: f32,
    pub center_spawn_min_fraction: f32,
    pub center_spawn_max_fraction: f32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        toml::from_str(include_str!("../../config/default.toml"))
            .expect("default world config TOML must parse")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseEdge {
    pub post_neuron_id: NeuronId,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuronState {
    pub neuron_id: NeuronId,
    pub neuron_type: NeuronType,
    pub bias: f32,
    pub activation: f32,
    pub parent_ids: Vec<NeuronId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensoryNeuronState {
    pub neuron: NeuronState,
    pub receptor_type: SensoryReceptorType,
    pub synapses: Vec<SynapseEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InterNeuronState {
    pub neuron: NeuronState,
    pub synapses: Vec<SynapseEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionNeuronState {
    pub neuron: NeuronState,
    pub action_type: ActionType,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainState {
    pub sensory: Vec<SensoryNeuronState>,
    pub inter: Vec<InterNeuronState>,
    pub action: Vec<ActionNeuronState>,
    pub synapse_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismState {
    pub id: OrganismId,
    pub q: i32,
    pub r: i32,
    pub facing: FacingDirection,
    pub turns_since_last_meal: u32,
    pub meals_eaten: u64,
    pub brain: BrainState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MetricsSnapshot {
    pub turns: u64,
    pub organisms: u32,
    pub synapse_ops_last_turn: u64,
    pub actions_applied_last_turn: u64,
    pub meals_last_turn: u64,
    pub starvations_last_turn: u64,
    pub births_last_turn: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub organisms: Vec<OrganismState>,
    pub occupancy: Vec<OccupancyCell>,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OccupancyCell {
    pub q: i32,
    pub r: i32,
    pub organism_ids: Vec<OrganismId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismMove {
    pub id: OrganismId,
    pub from: (i32, i32),
    pub to: (i32, i32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TickDelta {
    pub turn: u64,
    pub moves: Vec<OrganismMove>,
    pub removed: Vec<OrganismId>,
    pub spawned: Vec<OrganismState>,
    pub metrics: MetricsSnapshot,
}

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
#[serde(tag = "type", content = "data")]
pub enum ServerEvent {
    StateSnapshot(WorldSnapshot),
    TickDelta(TickDelta),
    FocusBrain(OrganismState),
    Metrics(MetricsSnapshot),
    Error(ApiError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_roundtrip() {
        let cfg = WorldConfig::default();
        let wrapped = Envelope::new(cfg.clone());
        let json = serde_json::to_string(&wrapped).expect("serialize envelope");
        let parsed: Envelope<WorldConfig> =
            serde_json::from_str(&json).expect("deserialize envelope");
        assert_eq!(parsed.payload, cfg);
        assert_eq!(parsed.protocol_version, PROTOCOL_VERSION);
    }
}
