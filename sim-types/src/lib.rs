use serde::{Deserialize, Serialize};
pub use sim_config::{SeedGenomeConfig, WorldConfig};
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
id_newtype!(SpeciesId, u64);
id_newtype!(NeuronId, u32);
id_newtype!(FoodId, u64);

pub const INTER_NEURON_ID_BASE: u32 = 1000;
pub const ACTION_NEURON_ID_BASE: u32 = 2000;
pub const MAX_GESTATION_TICKS: u8 = 4;
pub const BASE_OFFSPRING_TRANSFER_ENERGY: f32 = 100.0;
pub const GESTATION_TRANSFER_ENERGY_STEP: f32 = 100.0;
pub const ORGANISM_VISUAL_OPACITY: f32 = 0.8;
pub const FOOD_VISUAL_OPACITY: f32 = 0.8;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum FoodKind {
    #[default]
    Plant,
    Corpse,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, VariantArray, Default)]
pub enum ActionType {
    #[default]
    Idle,
    TurnLeft,
    TurnRight,
    Forward,
    Eat,
    Attack,
    Reproduce,
}
impl ActionType {
    pub const ALL: &'static [ActionType] = &[
        ActionType::TurnLeft,
        ActionType::TurnRight,
        ActionType::Forward,
        ActionType::Eat,
        ActionType::Attack,
        ActionType::Reproduce,
    ];

    pub fn neuron_id(self) -> Option<NeuronId> {
        let idx = Self::ALL.iter().position(|candidate| *candidate == self)?;
        Some(NeuronId(ACTION_NEURON_ID_BASE + idx as u32))
    }

    pub fn from_neuron_id(id: NeuronId) -> Option<Self> {
        let idx = id.0.checked_sub(ACTION_NEURON_ID_BASE)? as usize;
        Self::ALL.get(idx).copied()
    }
}

#[cfg(feature = "instrumentation")]
#[derive(Debug, Clone)]
pub struct ActionRecord {
    pub organism_id: OrganismId,
    pub selected_action: ActionType,
    pub action_failed: bool,
    pub food_visible: [bool; SensoryReceptor::VISION_RAY_OFFSETS.len()],
    pub damage_taken_last_turn: f32,
    pub age_turns: u64,
    pub utilization: f32,
    pub consumptions_count: u64,
}

#[cfg(feature = "instrumentation")]
impl ActionRecord {
    pub fn food_visible_at_offset(&self, ray_offset: i8) -> bool {
        let Some(ray_idx) = SensoryReceptor::VISION_RAY_OFFSETS
            .iter()
            .position(|candidate| *candidate == ray_offset)
        else {
            return false;
        };
        self.food_visible[ray_idx]
    }
}

#[cfg(feature = "instrumentation")]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OrganismInstrumentationState {
    #[serde(default, skip)]
    pub inter_ema: Vec<f32>,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct RgbColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RgbColor {
    pub fn clamped(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct VisualProperties {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub opacity: f32,
}

impl VisualProperties {
    pub fn clamped(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            opacity: self.opacity.clamp(0.0, 1.0),
        }
    }
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
    Spikes,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TerrainType {
    Spikes,
    Mountain,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "entity_type", content = "id")]
pub enum EntityId {
    Organism(OrganismId),
    Food(FoodId),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReproductionFailureCause {
    BlockedBirth,
    ParentDied,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ReproductionEvent {
    pub parent_id: OrganismId,
    pub parent_species_id: SpeciesId,
    pub parent_age_turns: u64,
    pub parent_generation: u64,
    pub investment_energy: f32,
    pub parent_energy_after_event: f32,
    pub child_id: Option<OrganismId>,
    pub failure_cause: Option<ReproductionFailureCause>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VisionChannel {
    Red,
    Green,
    Blue,
}

impl VisionChannel {
    pub const ALL: [VisionChannel; 3] = [
        VisionChannel::Red,
        VisionChannel::Green,
        VisionChannel::Blue,
    ];
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "receptor_type")]
pub enum SensoryReceptor {
    VisionRay {
        ray_offset: i8,
        channel: VisionChannel,
    },
    ContactAhead,
    Energy,
    Health,
}

impl SensoryReceptor {
    /// Fixed relative ray offsets around facing direction.
    pub const VISION_RAY_OFFSETS: [i8; 1] = [0];
    pub const VISION_CHANNEL_COUNT: u32 = VisionChannel::ALL.len() as u32;
    pub const SCALAR_NEURON_COUNT: u32 = 3;
    pub const VISION_NEURON_COUNT: u32 =
        (Self::VISION_RAY_OFFSETS.len() as u32) * Self::VISION_CHANNEL_COUNT;
    pub const TOTAL_NEURON_COUNT: u32 = Self::VISION_NEURON_COUNT + Self::SCALAR_NEURON_COUNT;

    pub fn ordered() -> impl Iterator<Item = Self> {
        Self::VISION_RAY_OFFSETS
            .into_iter()
            .flat_map(|ray_offset| {
                VisionChannel::ALL
                    .into_iter()
                    .map(move |channel| Self::VisionRay {
                        ray_offset,
                        channel,
                    })
            })
            .chain([Self::ContactAhead, Self::Energy, Self::Health])
    }

    pub fn current_index(self) -> Option<usize> {
        Self::ordered().position(|candidate| candidate == self)
    }

    pub fn neuron_id(self) -> Option<NeuronId> {
        self.current_index().map(|idx| NeuronId(idx as u32))
    }

    pub fn from_neuron_id(id: NeuronId) -> Option<Self> {
        Self::ordered().nth(id.0 as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::{SensoryReceptor, VisionChannel};

    #[test]
    fn sensory_receptors_include_forward_rgb_plus_scalar_sensors() {
        let receptors = SensoryReceptor::ordered().collect::<Vec<_>>();
        assert_eq!(receptors.len(), 6);
        assert_eq!(
            receptors,
            vec![
                SensoryReceptor::VisionRay {
                    ray_offset: 0,
                    channel: VisionChannel::Red,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: 0,
                    channel: VisionChannel::Green,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: 0,
                    channel: VisionChannel::Blue,
                },
                SensoryReceptor::ContactAhead,
                SensoryReceptor::Energy,
                SensoryReceptor::Health,
            ]
        );
    }
}

pub fn inter_neuron_id(index: u32) -> NeuronId {
    NeuronId(INTER_NEURON_ID_BASE + index)
}

pub fn inter_neuron_index(id: NeuronId, num_neurons: u32) -> Option<u32> {
    let idx = id.0.checked_sub(INTER_NEURON_ID_BASE)?;
    (idx < num_neurons).then_some(idx)
}

pub fn offspring_transfer_energy(gestation_ticks: u8) -> f32 {
    BASE_OFFSPRING_TRANSFER_ENERGY
        + GESTATION_TRANSFER_ENERGY_STEP * f32::from(gestation_ticks.min(MAX_GESTATION_TICKS))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub spatial_prior_sigma: f32,
    pub vision_distance: u32,
    pub body_color: RgbColor,
    #[serde(default = "default_max_health")]
    pub max_health: f32,
    #[serde(default = "default_age_of_maturity")]
    pub age_of_maturity: u32,
    #[serde(default = "default_gestation_ticks")]
    pub gestation_ticks: u8,
    #[serde(default = "default_max_organism_age")]
    pub max_organism_age: u32,
    #[serde(default = "default_plasticity_start_age")]
    pub plasticity_start_age: u32,
    #[serde(default)]
    pub hebb_eta_gain: f32,
    #[serde(default = "default_juvenile_eta_scale")]
    pub juvenile_eta_scale: f32,
    #[serde(default = "default_eligibility_retention")]
    pub eligibility_retention: f32,
    #[serde(default = "default_max_weight_delta_per_tick")]
    pub max_weight_delta_per_tick: f32,
    #[serde(default)]
    pub synapse_prune_threshold: f32,
    #[serde(default)]
    pub mutation_rate_age_of_maturity: f32,
    #[serde(default)]
    pub mutation_rate_gestation_ticks: f32,
    #[serde(default)]
    pub mutation_rate_max_organism_age: f32,
    #[serde(default)]
    pub mutation_rate_vision_distance: f32,
    #[serde(default)]
    pub mutation_rate_max_health: f32,
    #[serde(default)]
    pub mutation_rate_inter_bias: f32,
    #[serde(default)]
    pub mutation_rate_inter_update_rate: f32,
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
    pub mutation_rate_remove_neuron: f32,
    #[serde(default)]
    pub mutation_rate_add_neuron_split_edge: f32,
    pub inter_biases: Vec<f32>,
    pub inter_log_time_constants: Vec<f32>,
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

fn default_gestation_ticks() -> u8 {
    0
}

fn default_max_organism_age() -> u32 {
    u32::MAX
}

fn default_plasticity_start_age() -> u32 {
    0
}

fn default_max_health() -> f32 {
    1.0
}

fn default_juvenile_eta_scale() -> f32 {
    0.5
}

fn default_max_weight_delta_per_tick() -> f32 {
    0.05
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
    #[serde(default)]
    pub state: f32,
    pub alpha: f32,
    pub synapses: Vec<SynapseEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionNeuronState {
    pub neuron_id: NeuronId,
    pub x: f32,
    pub y: f32,
    pub logit: f32,
    pub parent_ids: Vec<NeuronId>,
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
    #[serde(default)]
    pub generation: u64,
    pub age_turns: u64,
    pub facing: FacingDirection,
    pub energy: f32,
    #[serde(default)]
    pub health: f32,
    #[serde(default)]
    pub max_health: f32,
    #[serde(default)]
    pub energy_prev: f32,
    #[serde(default)]
    pub dopamine: f32,
    #[serde(default)]
    pub damage_taken_last_turn: f32,
    #[serde(default)]
    pub is_gestating: bool,
    #[serde(default)]
    pub consumptions_count: u64,
    #[serde(default)]
    pub plant_consumptions_count: u64,
    #[serde(default)]
    pub prey_consumptions_count: u64,
    pub reproductions_count: u64,
    #[serde(default)]
    pub last_action_taken: ActionType,
    #[cfg(feature = "instrumentation")]
    #[serde(default, skip)]
    pub instrumentation: OrganismInstrumentationState,
    pub brain: BrainState,
    pub genome: OrganismGenome,
}

impl OrganismState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: OrganismId,
        species_id: SpeciesId,
        q: i32,
        r: i32,
        generation: u64,
        age_turns: u64,
        facing: FacingDirection,
        energy: f32,
        health: f32,
        max_health: f32,
        energy_prev: f32,
        dopamine: f32,
        damage_taken_last_turn: f32,
        is_gestating: bool,
        consumptions_count: u64,
        plant_consumptions_count: u64,
        prey_consumptions_count: u64,
        reproductions_count: u64,
        last_action_taken: ActionType,
        brain: BrainState,
        genome: OrganismGenome,
    ) -> Self {
        Self {
            id,
            species_id,
            q,
            r,
            generation,
            age_turns,
            facing,
            energy,
            health,
            max_health,
            energy_prev,
            dopamine,
            damage_taken_last_turn,
            is_gestating,
            consumptions_count,
            plant_consumptions_count,
            prey_consumptions_count,
            reproductions_count,
            last_action_taken,
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain,
            genome,
        }
    }
}

pub fn get_size(organism: &OrganismState) -> f32 {
    offspring_transfer_energy(organism.genome.gestation_ticks)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FoodState {
    pub id: FoodId,
    pub q: i32,
    pub r: i32,
    pub energy: f32,
    #[serde(default)]
    pub kind: FoodKind,
    pub visual: VisualProperties,
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
    pub terrain: Vec<TerrainCell>,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TerrainCell {
    pub q: i32,
    pub r: i32,
    pub terrain_type: TerrainType,
    pub visual: VisualProperties,
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
    pub reproduction_events: Vec<ReproductionEvent>,
    pub food_spawned: Vec<FoodState>,
    pub metrics: MetricsSnapshot,
}

pub fn organism_visual(color: RgbColor) -> VisualProperties {
    VisualProperties {
        r: color.r,
        g: color.g,
        b: color.b,
        opacity: ORGANISM_VISUAL_OPACITY,
    }
    .clamped()
}

pub fn food_visual(kind: FoodKind) -> VisualProperties {
    match kind {
        FoodKind::Plant => VisualProperties {
            r: 0.0,
            g: 1.0,
            b: 0.0,
            opacity: FOOD_VISUAL_OPACITY,
        },
        FoodKind::Corpse => VisualProperties {
            r: 0.95,
            g: 0.45,
            b: 0.10,
            opacity: FOOD_VISUAL_OPACITY,
        },
    }
}

pub fn terrain_visual(terrain_type: TerrainType) -> VisualProperties {
    match terrain_type {
        TerrainType::Spikes => VisualProperties {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            opacity: 0.0,
        },
        TerrainType::Mountain => VisualProperties {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            opacity: 1.0,
        },
    }
}
