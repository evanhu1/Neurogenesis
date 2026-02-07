use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 1;

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SurvivalRule {
    CenterBandX {
        min_fraction: f32,
        max_fraction: f32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
}

impl ActionType {
    pub const ALL: [ActionType; 4] = [
        ActionType::MoveUp,
        ActionType::MoveDown,
        ActionType::MoveLeft,
        ActionType::MoveRight,
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
    LookLeft,
    LookRight,
    LookUp,
    LookDown,
    X,
    Y,
}

impl SensoryReceptorType {
    pub const ALL: [SensoryReceptorType; 6] = [
        SensoryReceptorType::LookLeft,
        SensoryReceptorType::LookRight,
        SensoryReceptorType::LookUp,
        SensoryReceptorType::LookDown,
        SensoryReceptorType::X,
        SensoryReceptorType::Y,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldConfig {
    pub columns: u32,
    pub rows: u32,
    pub steps_per_epoch: u32,
    pub steps_per_second: u32,
    pub num_organisms: u32,
    pub num_neurons: u32,
    pub max_num_neurons: u32,
    pub num_synapses: u32,
    pub vision_depth: u32,
    pub action_potential_length: u32,
    pub mutation_chance: f32,
    pub mutation_magnitude: f32,
    pub unfit_kill_probability: f32,
    pub offspring_fill_ratio: f32,
    pub survival_rule: SurvivalRule,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            columns: 20,
            rows: 20,
            steps_per_epoch: 20,
            steps_per_second: 5,
            num_organisms: 500,
            num_neurons: 2,
            max_num_neurons: 20,
            num_synapses: 4,
            vision_depth: 3,
            action_potential_length: 1,
            mutation_chance: 0.04,
            mutation_magnitude: 1.0,
            unfit_kill_probability: 0.95,
            offspring_fill_ratio: 0.2,
            survival_rule: SurvivalRule::CenterBandX {
                min_fraction: 0.25,
                max_fraction: 0.75,
            },
        }
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
    pub is_inverted: bool,
    pub action_potential_threshold: f32,
    pub resting_potential: f32,
    pub potential: f32,
    pub incoming_current: f32,
    pub potential_decay_rate: f32,
    pub action_potential_length: u32,
    pub action_potential_time: Option<u32>,
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
    pub x: i32,
    pub y: i32,
    pub brain: BrainState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MetricsSnapshot {
    pub ticks: u64,
    pub epochs: u64,
    pub survivors_last_epoch: u32,
    pub organisms: u32,
    pub synapse_ops_last_tick: u64,
    pub actions_applied_last_tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub epoch: u64,
    pub tick_in_epoch: u32,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub organisms: Vec<OrganismState>,
    pub occupancy: Vec<OccupancyCell>,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OccupancyCell {
    pub x: i32,
    pub y: i32,
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
    pub tick_in_epoch: u32,
    pub epoch: u64,
    pub moves: Vec<OrganismMove>,
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
    Epoch { count: u32 },
    SetFocus { organism_id: OrganismId },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
pub enum ServerEvent {
    StateSnapshot(WorldSnapshot),
    TickDelta(TickDelta),
    EpochCompleted(MetricsSnapshot),
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

    #[test]
    fn command_roundtrip() {
        let cmd = ClientCommand::Step { count: 3 };
        let json = serde_json::to_string(&cmd).expect("serialize command");
        let parsed: ClientCommand = serde_json::from_str(&json).expect("deserialize command");
        assert_eq!(parsed, cmd);
    }
}
