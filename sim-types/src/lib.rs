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
pub const MAX_GESTATION_TICKS: u8 = 10;
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

    /// Stable per-action index used by dataset encodings and histograms:
    /// declaration order including `Idle` (`0..=6`).
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether this action is contingent on world state and can therefore
    /// fail. Single source of truth for both the sim's `ActionRecord`
    /// initialization and evaluation failure counting.
    pub const fn can_fail(self) -> bool {
        matches!(
            self,
            ActionType::Forward | ActionType::Eat | ActionType::Attack | ActionType::Reproduce
        )
    }

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
    pub shape: f32,
}

impl VisualProperties {
    pub fn clamped(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            opacity: self.opacity.clamp(0.0, 1.0),
            shape: self.shape.clamp(0.0, 1.0),
        }
    }
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
    Shape,
}

impl VisionChannel {
    pub const ALL: [VisionChannel; 4] = [
        VisionChannel::Red,
        VisionChannel::Green,
        VisionChannel::Blue,
        VisionChannel::Shape,
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
    EnergyDelta,
    LastActionForward,
    LastActionEat,
}

impl SensoryReceptor {
    /// Fixed relative ray offsets around facing direction.
    pub const VISION_RAY_OFFSETS: [i8; 3] = [-1, 0, 1];
    pub const VISION_CHANNEL_COUNT: u32 = VisionChannel::ALL.len() as u32;
    pub const SCALAR_NEURON_COUNT: u32 = 6;
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
            .chain([
                Self::ContactAhead,
                Self::Energy,
                Self::Health,
                Self::EnergyDelta,
                Self::LastActionForward,
                Self::LastActionEat,
            ])
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
        assert_eq!(receptors.len(), 18);
        assert_eq!(
            receptors,
            vec![
                SensoryReceptor::VisionRay {
                    ray_offset: -1,
                    channel: VisionChannel::Red,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: -1,
                    channel: VisionChannel::Green,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: -1,
                    channel: VisionChannel::Blue,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: -1,
                    channel: VisionChannel::Shape,
                },
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
                SensoryReceptor::VisionRay {
                    ray_offset: 0,
                    channel: VisionChannel::Shape,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: 1,
                    channel: VisionChannel::Red,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: 1,
                    channel: VisionChannel::Green,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: 1,
                    channel: VisionChannel::Blue,
                },
                SensoryReceptor::VisionRay {
                    ray_offset: 1,
                    channel: VisionChannel::Shape,
                },
                SensoryReceptor::ContactAhead,
                SensoryReceptor::Energy,
                SensoryReceptor::Health,
                SensoryReceptor::EnergyDelta,
                SensoryReceptor::LastActionForward,
                SensoryReceptor::LastActionEat,
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
pub struct TopologyGenes {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub vision_distance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecycleGenes {
    pub body_color: RgbColor,
    #[serde(default = "default_age_of_maturity")]
    pub age_of_maturity: u32,
    #[serde(default = "default_gestation_ticks")]
    pub gestation_ticks: u8,
    #[serde(default = "default_max_organism_age")]
    pub max_organism_age: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlasticityGenes {
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MutationRateGenes {
    #[serde(default)]
    pub age_of_maturity: f32,
    #[serde(default)]
    pub gestation_ticks: f32,
    #[serde(default)]
    pub max_organism_age: f32,
    #[serde(default)]
    pub vision_distance: f32,
    #[serde(default)]
    pub hebb_eta_gain: f32,
    #[serde(default)]
    pub juvenile_eta_scale: f32,
    #[serde(default)]
    pub inter_bias: f32,
    #[serde(default)]
    pub inter_update_rate: f32,
    #[serde(default)]
    pub eligibility_retention: f32,
    #[serde(default)]
    pub synapse_prune_threshold: f32,
    #[serde(default)]
    pub synapse_weight_perturbation: f32,
    #[serde(default)]
    pub add_synapse: f32,
    #[serde(default)]
    pub remove_synapse: f32,
    #[serde(default)]
    pub remove_neuron: f32,
    #[serde(default)]
    pub add_neuron_split_edge: f32,
    #[serde(default)]
    pub max_weight_delta_per_tick: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainTopology {
    pub inter_biases: Vec<f32>,
    pub inter_log_time_constants: Vec<f32>,
    #[serde(default)]
    pub action_biases: Vec<f32>,
    pub edges: Vec<SynapseGene>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
    pub topology: TopologyGenes,
    pub lifecycle: LifecycleGenes,
    pub plasticity: PlasticityGenes,
    #[serde(default)]
    pub mutation_rates: MutationRateGenes,
    pub brain: BrainTopology,
}

impl OrganismGenome {
    /// Canonical test-only fixture used across unit tests, benches, and integration tests.
    /// Callers mutate specific fields after construction as needed. Adding a new field on
    /// `OrganismGenome` means updating only this one builder.
    pub fn test_fixture() -> Self {
        let action_count = ActionType::ALL.len();
        Self {
            topology: TopologyGenes {
                num_neurons: 1,
                num_synapses: 0,
                vision_distance: 2,
            },
            lifecycle: LifecycleGenes {
                body_color: RgbColor::default(),
                age_of_maturity: 0,
                gestation_ticks: 2,
                max_organism_age: 500,
            },
            plasticity: PlasticityGenes {
                hebb_eta_gain: 0.0,
                juvenile_eta_scale: 0.5,
                eligibility_retention: 0.9,
                max_weight_delta_per_tick: 0.05,
                synapse_prune_threshold: 0.01,
            },
            mutation_rates: MutationRateGenes::default(),
            brain: BrainTopology {
                inter_biases: vec![0.0],
                inter_log_time_constants: vec![0.0],
                action_biases: vec![0.0; action_count],
                edges: Vec::new(),
            },
        }
    }
}

/// Heritable synapse gene: pure wiring + weight. Runtime plasticity state
/// (eligibility, pending coactivation) lives only on the expressed
/// [`SynapseEdge`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SynapseGene {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseEdge {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub weight: f32,
    #[serde(default)]
    pub eligibility: f32,
    #[serde(skip)]
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
    // Finite lifespan cap matching the mutation clamp in sim-core
    // (`MAX_MUTATED_MAX_ORGANISM_AGE`); seeds are no longer immortal.
    100_000
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
    pub activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensoryNeuronState {
    pub neuron: NeuronState,
    #[serde(flatten)]
    pub receptor: SensoryReceptor,
    pub synapses: Vec<SynapseEdge>,
    /// Cached index of the first action-targeting edge in `synapses` (which
    /// stay sorted by post neuron ID). Maintained by
    /// `refresh_action_synapse_starts_and_count` at birth and after pruning.
    #[serde(default, skip)]
    pub action_synapse_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InterNeuronState {
    pub neuron: NeuronState,
    #[serde(default)]
    pub state: f32,
    pub alpha: f32,
    pub synapses: Vec<SynapseEdge>,
    #[serde(default, skip)]
    pub action_synapse_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionNeuronState {
    pub neuron_id: NeuronId,
    pub logit: f32,
    pub action_type: ActionType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainState {
    pub sensory: Vec<SensoryNeuronState>,
    pub inter: Vec<InterNeuronState>,
    pub action: Vec<ActionNeuronState>,
    pub synapse_count: u32,
    /// Per-sensory-neuron EMA of activation used to center pending
    /// coactivations (covariance rule). Length tracks `sensory.len()`.
    #[serde(skip)]
    pub sensory_mean_activation: Vec<f32>,
    /// Per-inter-neuron EMA of activation. Length tracks `inter.len()`.
    #[serde(skip)]
    pub inter_mean_activation: Vec<f32>,
    /// Per-action-neuron EMA of the squashed action logit, used to center the
    /// covariance rule on inter→action edges. Length tracks `action.len()`.
    #[serde(skip)]
    pub action_mean_activation: Vec<f32>,
    /// True once the activation means have been bootstrapped to the live
    /// activations on the brain's first plasticity pass. Neurons are never
    /// added or removed after birth, so every mean initializes on the same
    /// tick and one flag covers the whole brain.
    #[serde(skip)]
    pub means_initialized: bool,
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
    /// Energy stash captured at sensing time so the EnergyDelta sensor reads a
    /// sensing-to-sensing delta (spanning eat/attack/action-cost changes),
    /// rolled forward at the start of every tick's sensing pass.
    #[serde(default)]
    pub energy_at_last_sensing: f32,
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
    #[serde(default)]
    pub base_metabolic_cost: f32,
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
            energy_at_last_sensing: energy,
            damage_taken_last_turn,
            is_gestating,
            consumptions_count,
            plant_consumptions_count,
            prey_consumptions_count,
            reproductions_count,
            last_action_taken,
            base_metabolic_cost: 0.0,
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain,
            genome,
        }
    }
}

pub fn get_size(organism: &OrganismState) -> f32 {
    offspring_transfer_energy(organism.genome.lifecycle.gestation_ticks)
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
    pub age_deaths_last_turn: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub organisms: Vec<OrganismState>,
    pub foods: Vec<FoodState>,
    pub terrain: Vec<TerrainCell>,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "id")]
pub enum Occupant {
    Organism(OrganismId),
    Food(FoodId),
    Wall,
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

pub const ORGANISM_SHAPE: f32 = 0.2;
pub const PLANT_SHAPE: f32 = 0.4;
pub const CORPSE_SHAPE: f32 = 0.6;
pub const SPIKE_SHAPE: f32 = 0.8;
pub const MOUNTAIN_SHAPE: f32 = 1.0;

pub const SPIKE_VISION_OPACITY: f32 = 0.25;

pub fn organism_visual(color: RgbColor) -> VisualProperties {
    VisualProperties {
        r: color.r,
        g: color.g,
        b: color.b,
        opacity: ORGANISM_VISUAL_OPACITY,
        shape: ORGANISM_SHAPE,
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
            shape: PLANT_SHAPE,
        },
        FoodKind::Corpse => VisualProperties {
            r: 0.95,
            g: 0.45,
            b: 0.10,
            opacity: FOOD_VISUAL_OPACITY,
            shape: CORPSE_SHAPE,
        },
    }
}

pub fn terrain_visual(terrain_type: TerrainType) -> VisualProperties {
    match terrain_type {
        TerrainType::Spikes => VisualProperties {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            opacity: SPIKE_VISION_OPACITY,
            shape: SPIKE_SHAPE,
        },
        TerrainType::Mountain => VisualProperties {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            opacity: 1.0,
            shape: MOUNTAIN_SHAPE,
        },
    }
}
