use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrganismId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronId(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeciesId(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FoodId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    MoveForward,
    Turn,
    Consume,
    Reproduce,
    Dopamine,
}
impl ActionType {
    pub const ALL: [ActionType; 5] = [
        ActionType::MoveForward,
        ActionType::Turn,
        ActionType::Consume,
        ActionType::Reproduce,
        ActionType::Dopamine,
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
pub enum InterNeuronType {
    Excitatory,
    Inhibitory,
}

impl Default for InterNeuronType {
    fn default() -> Self {
        Self::Excitatory
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EntityType {
    Food,
    Organism,
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
    Look { look_target: EntityType },
    Energy,
}

impl SensoryReceptor {
    /// Number of look-based sensory neurons (Food, Organism).
    pub const LOOK_NEURON_COUNT: u32 = 2;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
    pub num_neurons: u32,
    pub vision_distance: u32,
    #[serde(default = "default_age_of_maturity")]
    pub age_of_maturity: u32,
    #[serde(default)]
    pub hebb_eta_baseline: f32,
    #[serde(default)]
    pub hebb_eta_gain: f32,
    #[serde(default = "default_eligibility_decay_lambda")]
    pub eligibility_decay_lambda: f32,
    #[serde(default)]
    pub synapse_prune_threshold: f32,
    #[serde(default)]
    pub mutation_rate_age_of_maturity: f32,
    #[serde(default)]
    pub mutation_rate_vision_distance: f32,
    #[serde(default)]
    pub mutation_rate_add_edge: f32,
    #[serde(default)]
    pub mutation_rate_remove_edge: f32,
    #[serde(default)]
    pub mutation_rate_split_edge: f32,
    #[serde(default)]
    pub mutation_rate_inter_bias: f32,
    #[serde(default)]
    pub mutation_rate_inter_update_rate: f32,
    #[serde(default)]
    pub mutation_rate_action_bias: f32,
    #[serde(default)]
    pub mutation_rate_eligibility_decay_lambda: f32,
    #[serde(default)]
    pub mutation_rate_synapse_prune_threshold: f32,
    pub inter_biases: Vec<f32>,
    pub inter_log_taus: Vec<f32>,
    #[serde(default)]
    pub interneuron_types: Vec<InterNeuronType>,
    pub action_biases: Vec<f32>,
    pub edges: Vec<SynapseEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub vision_distance: u32,
    pub age_of_maturity: u32,
    pub hebb_eta_baseline: f32,
    pub hebb_eta_gain: f32,
    pub eligibility_decay_lambda: f32,
    pub synapse_prune_threshold: f32,
    pub mutation_rate_age_of_maturity: f32,
    pub mutation_rate_vision_distance: f32,
    pub mutation_rate_add_edge: f32,
    pub mutation_rate_remove_edge: f32,
    pub mutation_rate_split_edge: f32,
    pub mutation_rate_inter_bias: f32,
    pub mutation_rate_inter_update_rate: f32,
    pub mutation_rate_action_bias: f32,
    pub mutation_rate_eligibility_decay_lambda: f32,
    pub mutation_rate_synapse_prune_threshold: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub steps_per_second: u32,
    pub num_organisms: u32,
    pub starting_energy: f32,
    pub food_energy: f32,
    pub reproduction_energy_cost: f32,
    pub move_action_energy_cost: f32,
    pub turn_energy_cost: f32,
    pub plant_growth_speed: f32,
    #[serde(default = "default_food_regrowth_interval")]
    pub food_regrowth_interval: u32,
    #[serde(default = "default_food_fertility_noise_scale")]
    pub food_fertility_noise_scale: f32,
    #[serde(default = "default_food_fertility_exponent")]
    pub food_fertility_exponent: f32,
    #[serde(default = "default_food_fertility_floor")]
    pub food_fertility_floor: f32,
    pub max_organism_age: u32,
    pub max_num_neurons: u32,
    pub speciation_threshold: f32,
    pub seed_genome_config: SeedGenomeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldConfigDeserialize {
    world_width: u32,
    steps_per_second: u32,
    num_organisms: u32,
    starting_energy: f32,
    food_energy: f32,
    reproduction_energy_cost: f32,
    move_action_energy_cost: f32,
    turn_energy_cost: f32,
    #[serde(default)]
    plant_growth_speed: Option<f32>,
    #[serde(default)]
    _legacy_plant_target_coverage: Option<f32>,
    #[serde(default)]
    _legacy_food_coverage_divisor: Option<u32>,
    #[serde(default = "default_food_regrowth_interval")]
    food_regrowth_interval: u32,
    #[serde(default = "default_food_fertility_noise_scale")]
    food_fertility_noise_scale: f32,
    #[serde(default = "default_food_fertility_exponent")]
    food_fertility_exponent: f32,
    #[serde(default = "default_food_fertility_floor")]
    food_fertility_floor: f32,
    max_organism_age: u32,
    max_num_neurons: u32,
    speciation_threshold: f32,
    seed_genome_config: SeedGenomeConfig,
}

impl<'de> Deserialize<'de> for WorldConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = WorldConfigDeserialize::deserialize(deserializer)?;
        let plant_growth_speed = raw
            .plant_growth_speed
            .unwrap_or_else(default_plant_growth_speed);
        Ok(Self {
            world_width: raw.world_width,
            steps_per_second: raw.steps_per_second,
            num_organisms: raw.num_organisms,
            starting_energy: raw.starting_energy,
            food_energy: raw.food_energy,
            reproduction_energy_cost: raw.reproduction_energy_cost,
            move_action_energy_cost: raw.move_action_energy_cost,
            turn_energy_cost: raw.turn_energy_cost,
            plant_growth_speed,
            food_regrowth_interval: raw.food_regrowth_interval,
            food_fertility_noise_scale: raw.food_fertility_noise_scale,
            food_fertility_exponent: raw.food_fertility_exponent,
            food_fertility_floor: raw.food_fertility_floor,
            max_organism_age: raw.max_organism_age,
            max_num_neurons: raw.max_num_neurons,
            speciation_threshold: raw.speciation_threshold,
            seed_genome_config: raw.seed_genome_config,
        })
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        default_world_config()
    }
}

fn default_world_config() -> WorldConfig {
    let mut value: toml::Value = toml::from_str(include_str!("../../config/default.toml"))
        .expect("default world config TOML must parse");
    let table = value
        .as_table_mut()
        .expect("default world config root must be a table");

    let world_width = table
        .get("world_width")
        .and_then(toml::Value::as_integer)
        .expect("default world config world_width must be an integer");
    table
        .entry("starting_energy")
        .or_insert_with(|| toml::Value::Float(world_width as f64));

    let starting_energy = table
        .get("starting_energy")
        .and_then(|value| match value {
            toml::Value::Float(v) => Some(*v),
            toml::Value::Integer(v) => Some(*v as f64),
            _ => None,
        })
        .expect("default world config starting_energy must be numeric");
    table
        .entry("reproduction_energy_cost")
        .or_insert_with(|| toml::Value::Float(starting_energy));

    table
        .entry("max_organism_age")
        .or_insert_with(|| toml::Value::Integer(world_width.saturating_mul(4)));

    value
        .try_into()
        .expect("default world config TOML must deserialize")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseEdge {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub weight: f32,
    #[serde(default)]
    pub eligibility: f32,
}

fn default_eligibility_decay_lambda() -> f32 {
    0.9
}

fn default_age_of_maturity() -> u32 {
    0
}

fn default_food_regrowth_interval() -> u32 {
    10
}

fn default_food_fertility_noise_scale() -> f32 {
    0.045
}

fn default_food_fertility_exponent() -> f32 {
    1.8
}

fn default_food_fertility_floor() -> f32 {
    0.04
}

fn default_plant_growth_speed() -> f32 {
    1.0
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
    #[serde(default)]
    pub interneuron_type: InterNeuronType,
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
    pub predations_last_turn: u64,
    pub total_consumptions: u64,
    pub reproductions_last_turn: u64,
    pub starvations_last_turn: u64,
    pub total_species_created: u32,
    pub species_counts: BTreeMap<SpeciesId, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub species_registry: BTreeMap<SpeciesId, OrganismGenome>,
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
    fn legacy_coverage_config_deserializes_with_default_growth_speed() {
        let cfg = WorldConfig::default();
        let mut value = serde_json::to_value(&cfg).expect("serialize config to value");
        let object = value
            .as_object_mut()
            .expect("world config JSON value must be an object");
        object.remove("plant_growth_speed");
        object.insert("plant_target_coverage".to_owned(), json!(0.05));
        object.insert("food_coverage_divisor".to_owned(), json!(20));

        let parsed: WorldConfig =
            serde_json::from_value(value).expect("deserialize legacy world config");
        assert!((parsed.plant_growth_speed - 1.0).abs() < f32::EPSILON);
    }
}
