//! Wire schema for the thin file-server. A world is a file on disk; these are
//! the client-facing views the server serializes over HTTP + the WebSocket
//! animation stream. Render-shaped organism/food/terrain data only — the
//! CLI-parity research reads (state/pillars/inspect/brain/…) are produced by
//! `sim-views` and forwarded as raw JSON, so they have no structs here.

use serde::{Deserialize, Serialize};
use sim_types::{
    organism_visual, FacingDirection, FoodState, MetricsSnapshot, OrganismFacing, OrganismGenome,
    OrganismId, OrganismMove, OrganismState, RemovedEntityPosition, ReproductionEvent, SpeciesId,
    TerrainCell, TickDelta, VisualProperties, WorldSnapshot,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

/// Render-shaped organism: position + display-relevant scalar state + the
/// genome-derived visual properties, without the full brain/genome payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldOrganismState {
    pub id: OrganismId,
    pub species_id: SpeciesId,
    pub q: i32,
    pub r: i32,
    pub generation: u64,
    pub age_turns: u64,
    pub facing: FacingDirection,
    pub energy: f32,
    pub health: f32,
    pub max_health: f32,
    pub damage_taken_last_turn: f32,
    pub is_gestating: bool,
    pub consumptions_count: u64,
    pub plant_consumptions_count: u64,
    pub prey_consumptions_count: u64,
    pub reproductions_count: u64,
    pub visual: VisualProperties,
}

impl From<&OrganismState> for WorldOrganismState {
    fn from(organism: &OrganismState) -> Self {
        Self {
            id: organism.id,
            species_id: organism.species_id,
            q: organism.q,
            r: organism.r,
            generation: organism.generation,
            age_turns: organism.age_turns,
            facing: organism.facing,
            energy: organism.energy,
            health: organism.health,
            max_health: organism.max_health,
            damage_taken_last_turn: organism.damage_taken_last_turn,
            is_gestating: organism.is_gestating,
            consumptions_count: organism.consumptions_count,
            plant_consumptions_count: organism.plant_consumptions_count,
            prey_consumptions_count: organism.prey_consumptions_count,
            reproductions_count: organism.reproductions_count,
            visual: organism_visual(),
        }
    }
}

/// Full render snapshot of a world (the canvas feed): every organism, food, and
/// terrain cell plus the current metrics. Produced from `Simulation::snapshot`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshotView {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: sim_types::WorldConfig,
    pub organisms: Vec<WorldOrganismState>,
    pub foods: Vec<FoodState>,
    pub terrain: Vec<TerrainCell>,
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
            terrain: snapshot.terrain,
            metrics: snapshot.metrics,
        }
    }
}

/// One tick's incremental change, applied by the client-side renderer to
/// animate between full snapshots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TickDeltaView {
    pub turn: u64,
    pub moves: Vec<OrganismMove>,
    pub facing_updates: Vec<OrganismFacing>,
    pub removed_positions: Vec<RemovedEntityPosition>,
    pub spawned: Vec<WorldOrganismState>,
    pub reproduction_events: Vec<ReproductionEvent>,
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
            reproduction_events: delta.reproduction_events,
            food_spawned: delta.food_spawned,
            metrics: delta.metrics,
        }
    }
}

/// Frames pushed over the `/worlds/{name}/stream` WebSocket: an initial full
/// snapshot, then one delta per advanced tick.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data")]
#[allow(clippy::large_enum_variant)]
pub enum StreamFrame {
    StateSnapshot(WorldSnapshotView),
    TickDelta(TickDeltaView),
}

/// The full detail of a single organism (brain + genome), for the inspector
/// panel's brain visualization. `active_action_neuron_id` is the action neuron
/// the organism most recently fired.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismDetail {
    pub turn: u64,
    pub organism: OrganismState,
    pub active_action_neuron_id: Option<sim_types::NeuronId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChampionPoolEntry {
    pub genome: OrganismGenome,
    pub source_turn: u64,
    pub source_created_at_unix_ms: u128,
    pub generation: u64,
    pub age_turns: u64,
    pub reproductions_count: u64,
    pub consumptions_count: u64,
    pub energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChampionPoolResponse {
    pub entries: Vec<ChampionPoolEntry>,
}
