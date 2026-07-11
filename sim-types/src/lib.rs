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

/// Stable identity of a node in the heritable graph.
///
/// This is deliberately distinct from [`NeuronId`]. `GeneNodeId` survives
/// structural mutation and crossover, while `NeuronId` is a dense index in an
/// expressed runtime brain. Human-readable formats encode this as a decimal
/// string so structural hashes remain exact in JavaScript clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GeneNodeId(pub u64);

/// Stable historical identity of a heritable connection gene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnovationId(pub u64);

macro_rules! impl_stable_u64_id_serde {
    ($name:ident) => {
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                if serializer.is_human_readable() {
                    serializer.serialize_str(&self.0.to_string())
                } else {
                    serializer.serialize_u64(self.0)
                }
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                if !deserializer.is_human_readable() {
                    return u64::deserialize(deserializer).map(Self);
                }

                struct StableIdVisitor;

                impl<'de> serde::de::Visitor<'de> for StableIdVisitor {
                    type Value = u64;

                    fn expecting(
                        &self,
                        formatter: &mut std::fmt::Formatter<'_>,
                    ) -> std::fmt::Result {
                        formatter.write_str("a decimal u64 string or integer")
                    }

                    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                        Ok(value)
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        value.parse().map_err(E::custom)
                    }
                }

                deserializer.deserialize_any(StableIdVisitor).map(Self)
            }
        }
    };
}

impl_stable_u64_id_serde!(GeneNodeId);
impl_stable_u64_id_serde!(InnovationId);

// Gene-node roles occupy separate high-bit domains. The 62-bit payload keeps
// fixed interface IDs directly decodable while split-created hidden IDs use a
// large deterministic hash space.
const GENE_NODE_DOMAIN_SHIFT: u32 = 62;
const GENE_NODE_PAYLOAD_MASK: u64 = (1_u64 << GENE_NODE_DOMAIN_SHIFT) - 1;
const SENSOR_GENE_NODE_DOMAIN: u64 = 0;
const ACTION_GENE_NODE_DOMAIN: u64 = 1;
const SEED_HIDDEN_GENE_NODE_DOMAIN: u64 = 2;
const SPLIT_HIDDEN_GENE_NODE_DOMAIN: u64 = 3;
const CONNECTION_INNOVATION_DOMAIN: u64 = 0x434f_4e4e_5f49_4e4e;
const SPLIT_NODE_HASH_DOMAIN: u64 = 0x5350_4c49_545f_4e4f;

const fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub const fn sensory_gene_node_id(index: u32) -> GeneNodeId {
    GeneNodeId((SENSOR_GENE_NODE_DOMAIN << GENE_NODE_DOMAIN_SHIFT) | index as u64)
}

pub const fn action_gene_node_id(index: usize) -> GeneNodeId {
    GeneNodeId((ACTION_GENE_NODE_DOMAIN << GENE_NODE_DOMAIN_SHIFT) | index as u64)
}

pub const fn seed_hidden_gene_node_id(index: u32) -> GeneNodeId {
    GeneNodeId((SEED_HIDDEN_GENE_NODE_DOMAIN << GENE_NODE_DOMAIN_SHIFT) | index as u64)
}

pub const fn split_hidden_gene_node_id(parent_innovation: InnovationId) -> GeneNodeId {
    let payload = mix_u64(SPLIT_NODE_HASH_DOMAIN ^ parent_innovation.0) & GENE_NODE_PAYLOAD_MASK;
    GeneNodeId((SPLIT_HIDDEN_GENE_NODE_DOMAIN << GENE_NODE_DOMAIN_SHIFT) | payload)
}

pub const fn connection_innovation_id(
    pre_node_id: GeneNodeId,
    post_node_id: GeneNodeId,
) -> InnovationId {
    let with_pre = mix_u64(CONNECTION_INNOVATION_DOMAIN ^ pre_node_id.0);
    InnovationId(mix_u64(
        with_pre ^ post_node_id.0.wrapping_add(0x9e37_79b9_7f4a_7c15),
    ))
}

pub const fn gene_node_domain(id: GeneNodeId) -> u64 {
    id.0 >> GENE_NODE_DOMAIN_SHIFT
}

pub const fn sensory_gene_node_index(id: GeneNodeId) -> Option<u32> {
    if gene_node_domain(id) != SENSOR_GENE_NODE_DOMAIN {
        return None;
    }
    let payload = id.0 & GENE_NODE_PAYLOAD_MASK;
    if payload > u32::MAX as u64 {
        return None;
    }
    Some(payload as u32)
}

pub const fn action_gene_node_index(id: GeneNodeId) -> Option<usize> {
    if gene_node_domain(id) != ACTION_GENE_NODE_DOMAIN {
        return None;
    }
    let payload = id.0 & GENE_NODE_PAYLOAD_MASK;
    if payload > usize::MAX as u64 {
        return None;
    }
    Some(payload as usize)
}

pub const fn is_hidden_gene_node_id(id: GeneNodeId) -> bool {
    let domain = gene_node_domain(id);
    domain == SEED_HIDDEN_GENE_NODE_DOMAIN || domain == SPLIT_HIDDEN_GENE_NODE_DOMAIN
}

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
}
impl ActionType {
    pub const ALL: &'static [ActionType] = &[
        ActionType::TurnLeft,
        ActionType::TurnRight,
        ActionType::Forward,
        ActionType::Eat,
        ActionType::Attack,
    ];

    pub fn active(predation_enabled: bool) -> impl Iterator<Item = ActionType> {
        Self::ALL
            .iter()
            .copied()
            .filter(move |action| predation_enabled || *action != ActionType::Attack)
    }

    pub const fn is_enabled(self, predation_enabled: bool) -> bool {
        predation_enabled || !matches!(self, ActionType::Attack)
    }

    /// Stable per-action index used by dataset encodings and histograms:
    /// declaration order including `Idle` (`0..=5`).
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether this action is contingent on world state and can therefore
    /// fail. Single source of truth for both the sim's `ActionRecord`
    /// initialization and evaluation failure counting.
    pub const fn can_fail(self) -> bool {
        matches!(
            self,
            ActionType::Forward | ActionType::Eat | ActionType::Attack
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionRecord {
    pub organism_id: OrganismId,
    pub selected_action: ActionType,
    pub action_failed: bool,
    pub food_visible: [bool; SensoryReceptor::VISION_RAY_OFFSETS.len()],
    pub age_turns: u64,
    pub utilization: f32,
    pub consumptions_count: u64,
    /// Cumulative plant (foraging) consumptions, split out so the evaluation
    /// layer can score foraging and predation competence separately.
    pub plant_consumptions_count: u64,
    /// Cumulative prey/corpse (predation) consumptions.
    pub prey_consumptions_count: u64,
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TerrainType {
    Mountain,
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
    FoodRay { ray_offset: i8 },
    ContactAhead,
    Energy,
    OrganismRay { ray_offset: i8 },
    Health,
}

impl SensoryReceptor {
    /// Fixed relative ray offsets around facing direction.
    pub const VISION_RAY_OFFSETS: [i8; 3] = [-1, 0, 1];
    pub const BASELINE_NEURON_COUNT: u32 = 5;
    pub const TOTAL_NEURON_COUNT: u32 = 9;

    pub fn ordered() -> impl Iterator<Item = Self> {
        Self::VISION_RAY_OFFSETS
            .into_iter()
            .map(|ray_offset| Self::FoodRay { ray_offset })
            .chain([Self::ContactAhead, Self::Energy])
            .chain(
                Self::VISION_RAY_OFFSETS
                    .into_iter()
                    .map(|ray_offset| Self::OrganismRay { ray_offset }),
            )
            .chain([Self::Health])
    }

    pub fn active(predation_enabled: bool) -> impl Iterator<Item = Self> {
        Self::ordered().filter(move |receptor| {
            predation_enabled || !matches!(receptor, Self::OrganismRay { .. } | Self::Health)
        })
    }

    pub const fn is_predation_only(self) -> bool {
        matches!(self, Self::OrganismRay { .. } | Self::Health)
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
    use super::SensoryReceptor;

    #[test]
    fn sensory_receptors_have_five_input_baseline_and_optional_predation_inputs() {
        let receptors = SensoryReceptor::ordered().collect::<Vec<_>>();
        assert_eq!(receptors.len(), 9);
        assert_eq!(
            receptors,
            vec![
                SensoryReceptor::FoodRay { ray_offset: -1 },
                SensoryReceptor::FoodRay { ray_offset: 0 },
                SensoryReceptor::FoodRay { ray_offset: 1 },
                SensoryReceptor::ContactAhead,
                SensoryReceptor::Energy,
                SensoryReceptor::OrganismRay { ray_offset: -1 },
                SensoryReceptor::OrganismRay { ray_offset: 0 },
                SensoryReceptor::OrganismRay { ray_offset: 1 },
                SensoryReceptor::Health,
            ]
        );
        assert_eq!(SensoryReceptor::active(false).count(), 5);
        assert_eq!(SensoryReceptor::active(true).count(), 9);
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
    pub vision_distance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecycleGenes {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainTopology {
    pub hidden_nodes: Vec<HiddenNodeGene>,
    #[serde(default)]
    pub action_biases: Vec<f32>,
    pub edges: Vec<SynapseGene>,
}

/// Heritable hidden-node parameters keyed by a stable structural identity.
/// Runtime brains remap these IDs to dense [`NeuronId`] values at birth.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct HiddenNodeGene {
    pub id: GeneNodeId,
    pub bias: f32,
    pub log_time_constant: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
    pub topology: TopologyGenes,
    pub lifecycle: LifecycleGenes,
    pub plasticity: PlasticityGenes,
    pub brain: BrainTopology,
}

impl OrganismGenome {
    pub fn hidden_node_count(&self) -> usize {
        self.brain.hidden_nodes.len()
    }

    pub fn enabled_connection_count(&self) -> usize {
        self.brain.edges.iter().filter(|edge| edge.enabled).count()
    }

    pub fn encoded_connection_count(&self) -> usize {
        self.brain.edges.len()
    }

    /// Canonical test-only fixture used across unit tests, benches, and integration tests.
    /// Callers mutate specific fields after construction as needed. Adding a new field on
    /// `OrganismGenome` means updating only this one builder.
    pub fn test_fixture() -> Self {
        let action_count = ActionType::ALL.len();
        Self {
            topology: TopologyGenes { vision_distance: 2 },
            lifecycle: LifecycleGenes {
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
            brain: BrainTopology {
                hidden_nodes: vec![HiddenNodeGene {
                    id: seed_hidden_gene_node_id(0),
                    bias: 0.0,
                    log_time_constant: 0.0,
                }],
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
    pub innovation: InnovationId,
    pub pre_node_id: GeneNodeId,
    pub post_node_id: GeneNodeId,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseEdge {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub weight: f32,
    #[serde(default)]
    pub eligibility: f32,
    // `default` (not `skip`): normally consumed+zeroed the same tick it is set,
    // but a gestating organism's post-commit plasticity is skipped, so a nonzero
    // pending coactivation is frozen across its gestation ticks (see the
    // gestation note in turn/mod.rs). A world saved mid-gestation must persist it
    // or the reloaded run drops it into eligibility differently → divergence.
    #[serde(default)]
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
    //
    // `default` (not `skip`): persisted in the world blob so the post-load
    // edge split is correct without a rehydrate pass (the value is a pure
    // function of `synapses`, saved in a consistent between-tick state). Legacy
    // payloads that omit it default to 0; such a payload must be re-expressed.
    #[serde(default)]
    pub action_synapse_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InterNeuronState {
    pub neuron: NeuronState,
    #[serde(default)]
    pub state: f32,
    pub alpha: f32,
    pub synapses: Vec<SynapseEdge>,
    #[serde(default)]
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
    //
    // `default` (not `skip`): this is live plasticity state that carries across
    // ticks, so a world blob must persist it for a byte-identical reload. The
    // default keeps any legacy payload that omits it loadable (bootstrap will
    // re-seed). Means appear in the server wire too — additive, ignored by the
    // web client's brain normalizer.
    #[serde(default)]
    pub sensory_mean_activation: Vec<f32>,
    /// Per-inter-neuron EMA of activation. Length tracks `inter.len()`.
    #[serde(default)]
    pub inter_mean_activation: Vec<f32>,
    /// Per-action-neuron EMA of the squashed action logit, used to center the
    /// covariance rule on inter→action edges. Length tracks `action.len()`.
    #[serde(default)]
    pub action_mean_activation: Vec<f32>,
    /// True once the activation means have been bootstrapped to the live
    /// activations on the brain's first plasticity pass. Neurons are never
    /// added or removed after birth, so every mean initializes on the same
    /// tick and one flag covers the whole brain.
    #[serde(default)]
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
    /// Energy stash captured at sensing time for the post-action plasticity
    /// reward signal, rolled forward at the start of every tick's sensing pass.
    #[serde(default)]
    pub energy_at_last_sensing: f32,
    #[serde(default)]
    pub damage_taken_last_turn: f32,
    #[serde(default)]
    pub consumptions_count: u64,
    #[serde(default)]
    pub plant_consumptions_count: u64,
    #[serde(default)]
    pub prey_consumptions_count: u64,
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
        consumptions_count: u64,
        plant_consumptions_count: u64,
        prey_consumptions_count: u64,
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
            consumptions_count,
            plant_consumptions_count,
            prey_consumptions_count,
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
    /// Plant-food consumptions only; excludes corpses and direct predation.
    pub plant_consumptions_last_turn: u64,
    pub predations_last_turn: u64,
    pub total_consumptions: u64,
    /// Cumulative plant-food consumptions since the world was reset.
    pub total_plant_consumptions: u64,
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
    pub food_spawned: Vec<FoodState>,
    pub metrics: MetricsSnapshot,
}

pub const ORGANISM_SHAPE: f32 = 0.2;
pub const PLANT_SHAPE: f32 = 0.4;
pub const CORPSE_SHAPE: f32 = 0.6;
pub const MOUNTAIN_SHAPE: f32 = 1.0;

pub fn organism_visual() -> VisualProperties {
    VisualProperties {
        r: 0.15,
        g: 0.45,
        b: 0.95,
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
        TerrainType::Mountain => VisualProperties {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            opacity: 1.0,
            shape: MOUNTAIN_SHAPE,
        },
    }
}
