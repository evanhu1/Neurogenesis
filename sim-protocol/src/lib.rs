use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 11;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrganismId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronId(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeciesId(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FoodId(pub u64);

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
    Reproduce,
}

impl ActionType {
    pub const ALL: [ActionType; 4] = [
        ActionType::MoveForward,
        ActionType::TurnLeft,
        ActionType::TurnRight,
        ActionType::Reproduce,
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
pub enum LookTarget {
    Food,
    Organism,
    OutOfBounds,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "receptor_type")]
pub enum SensoryReceptor {
    Look { look_target: LookTarget },
    Energy,
}

impl SensoryReceptor {
    /// Number of look-based sensory neurons (Food, Organism, OutOfBounds).
    pub const LOOK_NEURON_COUNT: u32 = 3;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenomeEdge {
    pub pre: NeuronId,
    pub post: NeuronId,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
    pub num_neurons: u32,
    pub max_num_neurons: u32,
    pub vision_distance: u32,
    pub mutation_rate: f32,
    pub inter_biases: Vec<f32>,
    pub edges: Vec<GenomeEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub max_num_neurons: u32,
    pub num_synapses: u32,
    #[serde(alias = "mutation_chance")]
    pub mutation_rate: f32,
    pub vision_distance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub steps_per_second: u32,
    pub num_organisms: u32,
    pub center_spawn_min_fraction: f32,
    pub center_spawn_max_fraction: f32,
    pub starting_energy: f32,
    #[serde(default = "default_food_energy")]
    pub food_energy: f32,
    pub reproduction_energy_cost: f32,
    pub move_action_energy_cost: f32,
    #[serde(default = "default_turn_energy_cost")]
    pub turn_energy_cost: f32,
    #[serde(default = "default_food_coverage_divisor")]
    pub food_coverage_divisor: u32,
    #[serde(default = "default_max_organism_age")]
    pub max_organism_age: u32,
    #[serde(default = "default_speciation_threshold")]
    pub speciation_threshold: f32,
    #[serde(
        default = "default_seed_genome_config",
        alias = "seed_species_config",
        alias = "seed_config"
    )]
    pub seed_genome_config: SeedGenomeConfig,
}

impl Default for WorldConfig {
    fn default() -> Self {
        default_world_config()
    }
}

fn default_world_config() -> WorldConfig {
    toml::from_str(include_str!("../../config/default.toml"))
        .expect("default world config TOML must parse")
}

fn default_seed_genome_config() -> SeedGenomeConfig {
    default_world_config().seed_genome_config
}

fn default_food_energy() -> f32 {
    default_world_config().food_energy
}

fn default_turn_energy_cost() -> f32 {
    default_world_config().turn_energy_cost
}

fn default_food_coverage_divisor() -> u32 {
    default_world_config().food_coverage_divisor
}

fn default_max_organism_age() -> u32 {
    default_world_config().max_organism_age
}

fn default_speciation_threshold() -> f32 {
    default_world_config().speciation_threshold
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
    #[serde(flatten)]
    pub receptor: SensoryReceptor,
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
    pub species_id: SpeciesId,
    pub q: i32,
    pub r: i32,
    pub age_turns: u64,
    pub facing: FacingDirection,
    pub energy: f32,
    pub consumptions_count: u64,
    pub reproductions_count: u64,
    pub brain: BrainState,
    pub genome: OrganismGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FoodState {
    pub id: FoodId,
    pub q: i32,
    pub r: i32,
    pub energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MetricsSnapshot {
    pub turns: u64,
    pub organisms: u32,
    pub synapse_ops_last_turn: u64,
    pub actions_applied_last_turn: u64,
    pub consumptions_last_turn: u64,
    pub total_consumptions: u64,
    pub reproductions_last_turn: u64,
    pub starvations_last_turn: u64,
    pub species_counts: BTreeMap<SpeciesId, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub species_registry: BTreeMap<SpeciesId, OrganismGenome>,
    pub organisms: Vec<OrganismState>,
    #[serde(default)]
    pub foods: Vec<FoodState>,
    pub occupancy: Vec<OccupancyCell>,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OccupancyCell {
    pub q: i32,
    pub r: i32,
    pub organism_ids: Vec<OrganismId>,
    #[serde(default)]
    pub food_ids: Vec<FoodId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismMove {
    pub id: OrganismId,
    pub from: (i32, i32),
    pub to: (i32, i32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemovedOrganismPosition {
    pub id: OrganismId,
    pub q: i32,
    pub r: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemovedFoodPosition {
    pub id: FoodId,
    pub q: i32,
    pub r: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TickDelta {
    pub turn: u64,
    pub moves: Vec<OrganismMove>,
    pub removed_positions: Vec<RemovedOrganismPosition>,
    pub spawned: Vec<OrganismState>,
    #[serde(default)]
    pub food_removed_positions: Vec<RemovedFoodPosition>,
    #[serde(default)]
    pub food_spawned: Vec<FoodState>,
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
