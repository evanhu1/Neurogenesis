use serde::{Deserialize, Serialize};
pub use sim_config::{world_config_from_toml_str, SeedGenomeConfig, WorldConfig};
use strum::VariantArray;

macro_rules! id_newtype {
    ($name:ident, $inner:ty) => {
        #[derive(
            Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord,
        )]
        pub struct $name(pub $inner);
    };
}

id_newtype!(OrganismId, u64);
id_newtype!(NeuronId, u32);
id_newtype!(FoodId, u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, VariantArray)]
pub enum ActionType {
    Idle,
    TurnLeft,
    TurnRight,
    Forward,
    Consume,
    Reproduce,
}
impl ActionType {
    pub const ALL: &'static [ActionType] = Self::VARIANTS;
}

impl Default for ActionType {
    fn default() -> Self {
        Self::Idle
    }
}

#[cfg(feature = "instrumentation")]
#[derive(Debug, Clone)]
pub struct ActionRecord {
    pub organism_id: OrganismId,
    pub selected_action: ActionType,
    pub food_ahead: bool,
    pub food_left: bool,
    pub food_right: bool,
    pub food_behind: bool,
    pub inter_activations: Vec<f32>,
    pub consumptions_count: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, VariantArray)]
pub enum FacingDirection {
    East,
    NorthEast,
    NorthWest,
    West,
    SouthWest,
    SouthEast,
}

impl FacingDirection {
    pub const ALL: &'static [FacingDirection] = Self::VARIANTS;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NeuronType {
    Sensory,
    Inter,
    Action,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BrainLocation {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EntityType {
    Food,
    Organism,
    Wall,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "entity_type", content = "id")]
pub enum EntityId {
    Organism(OrganismId),
    Food(FoodId),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "receptor_type")]
pub enum SensoryReceptor {
    LookRay {
        ray_offset: i8,
        look_target: EntityType,
    },
    Energy,
}

impl SensoryReceptor {
    /// Fixed relative ray offsets around facing direction.
    pub const LOOK_RAY_OFFSETS: [i8; 6] = [-2, -1, 0, 1, 2, 3];
    /// Number of entity classes available to look-ray sensors.
    pub const LOOK_TARGET_COUNT: u32 = 3;
    /// Number of look-based sensory neurons (ray count x object type count).
    pub const LOOK_NEURON_COUNT: u32 =
        (Self::LOOK_RAY_OFFSETS.len() as u32) * Self::LOOK_TARGET_COUNT;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub spatial_prior_sigma: f32,
    pub vision_distance: u32,
    #[serde(default = "default_starting_energy")]
    pub starting_energy: f32,
    #[serde(default = "default_age_of_maturity")]
    pub age_of_maturity: u32,
    #[serde(default)]
    pub hebb_eta_gain: f32,
    #[serde(default = "default_eligibility_retention")]
    pub eligibility_retention: f32,
    #[serde(default)]
    pub synapse_prune_threshold: f32,
    #[serde(default)]
    pub mutation_rate_age_of_maturity: f32,
    #[serde(default)]
    pub mutation_rate_vision_distance: f32,
    #[serde(default)]
    pub mutation_rate_inter_bias: f32,
    #[serde(default)]
    pub mutation_rate_inter_update_rate: f32,
    #[serde(default)]
    pub mutation_rate_action_bias: f32,
    #[serde(default)]
    pub mutation_rate_eligibility_retention: f32,
    #[serde(default)]
    pub mutation_rate_synapse_prune_threshold: f32,
    #[serde(default)]
    pub mutation_rate_neuron_location: f32,
    #[serde(default)]
    pub mutation_rate_synapse_weight_perturbation: f32,
    #[serde(default)]
    pub mutation_rate_add_synapse: f32,
    #[serde(default)]
    pub mutation_rate_remove_synapse: f32,
    #[serde(default)]
    pub mutation_rate_add_neuron_split_edge: f32,
    pub inter_biases: Vec<f32>,
    pub inter_log_time_constants: Vec<f32>,
    pub action_biases: Vec<f32>,
    pub sensory_locations: Vec<BrainLocation>,
    pub inter_locations: Vec<BrainLocation>,
    pub action_locations: Vec<BrainLocation>,
    pub edges: Vec<SynapseEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseEdge {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub weight: f32,
    #[serde(default, skip_serializing)]
    pub eligibility: f32,
    #[serde(default, skip_serializing)]
    pub pending_coactivation: f32,
}

fn default_eligibility_retention() -> f32 {
    0.95
}

fn default_age_of_maturity() -> u32 {
    0
}

fn default_starting_energy() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuronState {
    pub neuron_id: NeuronId,
    pub neuron_type: NeuronType,
    pub bias: f32,
    pub x: f32,
    pub y: f32,
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
    pub alpha: f32,
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
    pub q: i32,
    pub r: i32,
    #[serde(default)]
    pub generation: u64,
    pub age_turns: u64,
    pub facing: FacingDirection,
    pub energy: f32,
    #[serde(default)]
    pub energy_prev: f32,
    #[serde(default)]
    pub dopamine: f32,
    pub consumptions_count: u64,
    pub reproductions_count: u64,
    #[serde(default)]
    pub last_action_taken: ActionType,
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
    pub predations_last_turn: u64,
    pub total_consumptions: u64,
    pub reproductions_last_turn: u64,
    pub starvations_last_turn: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub organisms: Vec<OrganismState>,
    pub foods: Vec<FoodState>,
    pub occupancy: Vec<OccupancyCell>,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "id")]
pub enum Occupant {
    Organism(OrganismId),
    Food(FoodId),
    Wall,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct OccupancyCell {
    pub q: i32,
    pub r: i32,
    pub occupant: Occupant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismMove {
    pub id: OrganismId,
    pub from: (i32, i32),
    pub to: (i32, i32),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrganismFacing {
    pub id: OrganismId,
    pub facing: FacingDirection,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RemovedEntityPosition {
    pub entity_id: EntityId,
    pub q: i32,
    pub r: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TickDelta {
    pub turn: u64,
    pub moves: Vec<OrganismMove>,
    pub facing_updates: Vec<OrganismFacing>,
    pub removed_positions: Vec<RemovedEntityPosition>,
    pub spawned: Vec<OrganismState>,
    pub food_spawned: Vec<FoodState>,
    pub metrics: MetricsSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_roundtrip() {
        let cfg = WorldConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize config");
        let parsed: WorldConfig = serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn legacy_coverage_fields_are_ignored() {
        let cfg = WorldConfig::default();
        let mut value = serde_json::to_value(&cfg).expect("serialize config to value");
        let object = value
            .as_object_mut()
            .expect("world config JSON value must be an object");
        object.insert("plant_target_coverage".to_owned(), json!(0.05));
        object.insert("food_coverage_divisor".to_owned(), json!(20));

        let parsed: WorldConfig =
            serde_json::from_value(value).expect("deserialize legacy world config");
        assert_eq!(parsed, cfg);
    }
}
