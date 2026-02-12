use serde::{Deserialize, Serialize};
use sim_types::{
    FacingDirection, FoodState, MetricsSnapshot, NeuronId, OrganismFacing, OrganismId,
    OrganismMove, OrganismState, RemovedEntityPosition, SpeciesId, TickDelta, WorldConfig,
    WorldSnapshot,
};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionMetadata {
    pub id: Uuid,
    pub created_at_unix_ms: u128,
    pub config: WorldConfig,
    pub running: bool,
    pub ticks_per_second: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateSessionRequest {
    pub config: WorldConfig,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateSessionResponse {
    pub metadata: SessionMetadata,
    pub snapshot: WorldSnapshotView,
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
pub struct StepProgressData {
    pub requested_count: u32,
    pub completed_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldOrganismState {
    pub id: OrganismId,
    pub species_id: SpeciesId,
    pub q: i32,
    pub r: i32,
    pub age_turns: u64,
    pub facing: FacingDirection,
    pub energy: f32,
    pub consumptions_count: u64,
    pub reproductions_count: u64,
}

impl From<&OrganismState> for WorldOrganismState {
    fn from(organism: &OrganismState) -> Self {
        Self {
            id: organism.id,
            species_id: organism.species_id,
            q: organism.q,
            r: organism.r,
            age_turns: organism.age_turns,
            facing: organism.facing,
            energy: organism.energy,
            consumptions_count: organism.consumptions_count,
            reproductions_count: organism.reproductions_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshotView {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub organisms: Vec<WorldOrganismState>,
    pub foods: Vec<FoodState>,
    pub metrics: MetricsSnapshot,
}

impl From<WorldSnapshot> for WorldSnapshotView {
    fn from(snapshot: WorldSnapshot) -> Self {
        Self {
            turn: snapshot.turn,
            rng_seed: snapshot.rng_seed,
            config: snapshot.config,
            organisms: snapshot
                .organisms
                .iter()
                .map(WorldOrganismState::from)
                .collect(),
            foods: snapshot.foods,
            metrics: snapshot.metrics,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TickDeltaView {
    pub turn: u64,
    pub moves: Vec<OrganismMove>,
    pub facing_updates: Vec<OrganismFacing>,
    pub removed_positions: Vec<RemovedEntityPosition>,
    pub spawned: Vec<WorldOrganismState>,
    pub food_spawned: Vec<FoodState>,
    pub metrics: MetricsSnapshot,
}

impl From<TickDelta> for TickDeltaView {
    fn from(delta: TickDelta) -> Self {
        Self {
            turn: delta.turn,
            moves: delta.moves,
            facing_updates: delta.facing_updates,
            removed_positions: delta.removed_positions,
            spawned: delta.spawned.iter().map(WorldOrganismState::from).collect(),
            food_spawned: delta.food_spawned,
            metrics: delta.metrics,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
pub enum ServerEvent {
    StateSnapshot(WorldSnapshotView),
    TickDelta(TickDeltaView),
    StepProgress(StepProgressData),
    FocusBrain(FocusBrainData),
    Metrics(MetricsSnapshot),
    Error(ApiError),
}
