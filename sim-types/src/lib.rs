use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrganismId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronId(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FoodId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionType {
    Idle,
    TurnLeft,
    TurnRight,
    Forward,
    Consume,
    Reproduce,
}
impl ActionType {
    pub const ALL: [ActionType; 6] = [
        ActionType::Idle,
        ActionType::TurnLeft,
        ActionType::TurnRight,
        ActionType::Forward,
        ActionType::Consume,
        ActionType::Reproduce,
    ];
}

impl Default for ActionType {
    fn default() -> Self {
        Self::Idle
    }
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
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub spatial_prior_sigma: f32,
    pub vision_distance: u32,
    #[serde(default = "default_starting_energy")]
    pub starting_energy: f32,
    pub age_of_maturity: u32,
    pub hebb_eta_gain: f32,
    pub eligibility_retention: f32,
    pub synapse_prune_threshold: f32,
    pub mutation_rate_age_of_maturity: f32,
    pub mutation_rate_vision_distance: f32,
    pub mutation_rate_inter_bias: f32,
    pub mutation_rate_inter_update_rate: f32,
    pub mutation_rate_action_bias: f32,
    pub mutation_rate_eligibility_retention: f32,
    pub mutation_rate_synapse_prune_threshold: f32,
    pub mutation_rate_neuron_location: f32,
    #[serde(default)]
    pub mutation_rate_synapse_weight_perturbation: f32,
    #[serde(default)]
    pub mutation_rate_add_neuron_split_edge: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub steps_per_second: u32,
    pub num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    pub periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    pub periodic_injection_count: u32,
    pub food_energy: f32,
    pub move_action_energy_cost: f32,
    #[serde(default = "default_action_temperature")]
    pub action_temperature: f32,
    #[serde(default = "default_food_regrowth_interval")]
    pub food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    pub food_regrowth_jitter: u32,
    #[serde(default = "default_food_fertility_noise_scale")]
    pub food_fertility_noise_scale: f32,
    #[serde(default = "default_food_fertility_threshold")]
    pub food_fertility_threshold: f32,
    #[serde(default = "default_terrain_noise_scale")]
    pub terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    pub terrain_threshold: f32,
    pub max_organism_age: u32,
    #[serde(default = "default_global_mutation_rate_modifier")]
    pub global_mutation_rate_modifier: f32,
    #[serde(default = "default_runtime_plasticity_enabled")]
    pub runtime_plasticity_enabled: bool,
    pub seed_genome_config: SeedGenomeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldConfigDeserialize {
    world_width: u32,
    steps_per_second: u32,
    num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    periodic_injection_count: u32,
    food_energy: f32,
    move_action_energy_cost: f32,
    #[serde(default = "default_action_temperature")]
    action_temperature: f32,
    #[serde(default = "default_food_regrowth_interval")]
    food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    food_regrowth_jitter: u32,
    #[serde(default = "default_food_fertility_noise_scale")]
    food_fertility_noise_scale: f32,
    #[serde(default = "default_food_fertility_threshold")]
    food_fertility_threshold: f32,
    #[serde(default = "default_terrain_noise_scale")]
    terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    terrain_threshold: f32,
    max_organism_age: u32,
    #[serde(default = "default_global_mutation_rate_modifier")]
    global_mutation_rate_modifier: f32,
    #[serde(default = "default_runtime_plasticity_enabled")]
    runtime_plasticity_enabled: bool,
    seed_genome_config: SeedGenomeConfig,
}

impl<'de> Deserialize<'de> for WorldConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = WorldConfigDeserialize::deserialize(deserializer)?;
        Ok(Self {
            world_width: raw.world_width,
            steps_per_second: raw.steps_per_second,
            num_organisms: raw.num_organisms,
            periodic_injection_interval_turns: raw.periodic_injection_interval_turns,
            periodic_injection_count: raw.periodic_injection_count,
            food_energy: raw.food_energy,
            move_action_energy_cost: raw.move_action_energy_cost,
            action_temperature: raw.action_temperature,
            food_regrowth_interval: raw.food_regrowth_interval,
            food_regrowth_jitter: raw.food_regrowth_jitter,
            food_fertility_noise_scale: raw.food_fertility_noise_scale,
            food_fertility_threshold: raw.food_fertility_threshold,
            terrain_noise_scale: raw.terrain_noise_scale,
            terrain_threshold: raw.terrain_threshold,
            max_organism_age: raw.max_organism_age,
            global_mutation_rate_modifier: raw.global_mutation_rate_modifier,
            runtime_plasticity_enabled: raw.runtime_plasticity_enabled,
            seed_genome_config: raw.seed_genome_config,
        })
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        default_world_config()
    }
}

pub fn world_config_from_toml_str(raw: &str) -> Result<WorldConfig, toml::de::Error> {
    let mut value: toml::Value = toml::from_str(raw)?;
    normalize_world_config_toml(&mut value);
    value.try_into()
}

fn default_world_config() -> WorldConfig {
    world_config_from_toml_str(include_str!("../../config/default.toml"))
        .expect("default world config TOML must deserialize")
}

fn normalize_world_config_toml(value: &mut toml::Value) {
    let Some(table) = value.as_table_mut() else {
        return;
    };

    let world_width = table
        .get("world_width")
        .and_then(toml::Value::as_integer)
        .and_then(|v| u32::try_from(v).ok());
    let legacy_starting_energy = table
        .get("starting_energy")
        .and_then(|value| match value {
            toml::Value::Float(v) => Some(*v),
            toml::Value::Integer(v) => Some(*v as f64),
            _ => None,
        })
        .or_else(|| world_width.map(|w| w as f64));

    if let Some(seed_genome_table) = table
        .entry("seed_genome_config")
        .or_insert_with(|| toml::Value::Table(Default::default()))
        .as_table_mut()
    {
        if let Some(legacy_starting_energy) = legacy_starting_energy {
            seed_genome_table
                .entry("starting_energy")
                .or_insert_with(|| toml::Value::Float(legacy_starting_energy));
        }
    }

    if let Some(world_width) = world_width {
        table
            .entry("max_organism_age")
            .or_insert_with(|| toml::Value::Integer(i64::from(world_width.saturating_mul(10))));
    }
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

fn default_food_regrowth_interval() -> u32 {
    10
}

fn default_food_regrowth_jitter() -> u32 {
    2
}

fn default_periodic_injection_interval_turns() -> u32 {
    100
}

fn default_periodic_injection_count() -> u32 {
    100
}

fn default_food_fertility_noise_scale() -> f32 {
    0.045
}

fn default_food_fertility_threshold() -> f32 {
    0.83
}

fn default_terrain_noise_scale() -> f32 {
    0.02
}

fn default_terrain_threshold() -> f32 {
    0.86
}

fn default_action_temperature() -> f32 {
    0.5
}

fn default_global_mutation_rate_modifier() -> f32 {
    1.0
}

fn default_runtime_plasticity_enabled() -> bool {
    true
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
