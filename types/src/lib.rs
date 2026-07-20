use serde::{Deserialize, Serialize};
use strum::VariantArray;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub plasticity_maturity_ticks: u32,
    pub initial_learning_rate: f32,
    pub juvenile_eta_scale: f32,
    pub eligibility_retention: f32,
    pub fast_weight_retention: f32,
    pub action_temperature_scale: f32,
    pub max_weight_delta_per_tick: f32,
    pub synapse_prune_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub num_organisms: u32,
    pub starting_energy: u32,
    pub attack_energy_transfer: u32,
    pub attack_attempt_cost: u32,
    pub action_temperature: f32,
    pub intent_parallel_threads: u32,
    pub terrain_noise_scale: f32,
    pub terrain_threshold: f32,
    pub runtime_plasticity_enabled: bool,
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
    pub force_random_actions: bool,
    pub seed_genome_config: SeedGenomeConfig,
}

impl WorldConfig {
    /// Purpose-built high-load fixture used by benchmarks, examples, and profiling.
    pub fn perf_fixture() -> Self {
        Self {
            world_width: 100,
            num_organisms: 2_000,
            starting_energy: 250,
            attack_energy_transfer: 40,
            attack_attempt_cost: 10,
            action_temperature: 0.5,
            intent_parallel_threads: 8,
            terrain_noise_scale: 0.02,
            terrain_threshold: 1.0,
            runtime_plasticity_enabled: true,
            leaky_neurons_enabled: false,
            predation_enabled: true,
            force_random_actions: false,
            seed_genome_config: SeedGenomeConfig {
                num_neurons: 20,
                num_synapses: 80,
                plasticity_maturity_ticks: 0,
                initial_learning_rate: 0.0,
                juvenile_eta_scale: 2.0,
                eligibility_retention: 0.9,
                fast_weight_retention: 1.0,
                action_temperature_scale: 1.0,
                max_weight_delta_per_tick: 0.05,
                synapse_prune_threshold: 0.01,
            },
        }
    }

    /// Purpose-built test fixture — small world with predation disabled for isolation.
    pub fn test_fixture() -> Self {
        Self {
            world_width: 10,
            num_organisms: 10,
            starting_energy: 250,
            attack_energy_transfer: 40,
            attack_attempt_cost: 10,
            action_temperature: 0.5,
            intent_parallel_threads: 8,
            terrain_noise_scale: 0.02,
            terrain_threshold: 1.0,
            runtime_plasticity_enabled: true,
            leaky_neurons_enabled: false,
            predation_enabled: false,
            force_random_actions: false,
            seed_genome_config: SeedGenomeConfig {
                num_neurons: 1,
                num_synapses: 0,
                plasticity_maturity_ticks: 0,
                initial_learning_rate: 0.0,
                juvenile_eta_scale: 0.5,
                eligibility_retention: 0.9,
                fast_weight_retention: 1.0,
                action_temperature_scale: 1.0,
                max_weight_delta_per_tick: 0.05,
                synapse_prune_threshold: 0.01,
            },
        }
    }
}

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

/// When a synapse reads its presynaptic activation.
///
/// Current-tick hidden connections form the instantaneous feed-forward DAG.
/// Previous-tick hidden connections read a frozen hidden-state snapshot;
/// previous-tick action-to-hidden connections read the selected-action
/// efference copy. Both may form cycles in the temporal graph without creating
/// algebraic cycles inside one brain evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SynapseTiming {
    CurrentTick = 0,
    PreviousTick = 1,
}

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

/// Stable fixed output used by the generic reward-prediction subsystem. It
/// shares the output-node domain but sits immediately after the motor alphabet.
pub const fn value_gene_node_id() -> GeneNodeId {
    action_gene_node_id(Symbol::COUNT)
}

pub const fn is_value_gene_node_id(id: GeneNodeId) -> bool {
    id.0 == value_gene_node_id().0
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
    timing: SynapseTiming,
) -> InnovationId {
    let with_pre = mix_u64(CONNECTION_INNOVATION_DOMAIN ^ pre_node_id.0);
    let with_post = mix_u64(with_pre ^ post_node_id.0.wrapping_add(0x9e37_79b9_7f4a_7c15));
    InnovationId(mix_u64(with_post ^ timing as u8 as u64))
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
pub const ORGANISM_VISUAL_OPACITY: f32 = 0.8;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Symbol {
    #[default]
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Space,
    End,
}

impl Symbol {
    pub const ALL: [Self; 28] = [
        Self::A,
        Self::B,
        Self::C,
        Self::D,
        Self::E,
        Self::F,
        Self::G,
        Self::H,
        Self::I,
        Self::J,
        Self::K,
        Self::L,
        Self::M,
        Self::N,
        Self::O,
        Self::P,
        Self::Q,
        Self::R,
        Self::S,
        Self::T,
        Self::U,
        Self::V,
        Self::W,
        Self::X,
        Self::Y,
        Self::Z,
        Self::Space,
        Self::End,
    ];
    pub const COUNT: usize = Self::ALL.len();

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn action_type(self) -> ActionType {
        match self {
            Self::A => ActionType::Idle,
            Self::B => ActionType::TurnLeft,
            Self::C => ActionType::TurnRight,
            Self::D => ActionType::Forward,
            Self::E
            | Self::F
            | Self::G
            | Self::H
            | Self::I
            | Self::J
            | Self::K
            | Self::L
            | Self::M
            | Self::N
            | Self::O
            | Self::P
            | Self::Q
            | Self::R
            | Self::S
            | Self::T
            | Self::U
            | Self::V
            | Self::W
            | Self::X
            | Self::Y
            | Self::Z
            | Self::Space => ActionType::Idle,
            Self::End => ActionType::Attack,
        }
    }

    pub const fn is_action_enabled(self, predation_enabled: bool) -> bool {
        matches!(self, Self::A | Self::B | Self::C | Self::D)
            || (predation_enabled && matches!(self, Self::End))
    }

    pub const fn action_neuron_id(self) -> NeuronId {
        NeuronId(ACTION_NEURON_ID_BASE + self as u32)
    }

    pub fn from_action_neuron_id(id: NeuronId) -> Option<Self> {
        Self::ALL
            .get(id.0.checked_sub(ACTION_NEURON_ID_BASE)? as usize)
            .copied()
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::C => "c",
            Self::D => "d",
            Self::E => "e",
            Self::F => "f",
            Self::G => "g",
            Self::H => "h",
            Self::I => "i",
            Self::J => "j",
            Self::K => "k",
            Self::L => "l",
            Self::M => "m",
            Self::N => "n",
            Self::O => "o",
            Self::P => "p",
            Self::Q => "q",
            Self::R => "r",
            Self::S => "s",
            Self::T => "t",
            Self::U => "u",
            Self::V => "v",
            Self::W => "w",
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
            Self::Space => "space",
            Self::End => "end",
        }
    }

    pub const fn from_ascii_char(value: char) -> Option<Self> {
        match value {
            'a' => Some(Self::A),
            'b' => Some(Self::B),
            'c' => Some(Self::C),
            'd' => Some(Self::D),
            'e' => Some(Self::E),
            'f' => Some(Self::F),
            'g' => Some(Self::G),
            'h' => Some(Self::H),
            'i' => Some(Self::I),
            'j' => Some(Self::J),
            'k' => Some(Self::K),
            'l' => Some(Self::L),
            'm' => Some(Self::M),
            'n' => Some(Self::N),
            'o' => Some(Self::O),
            'p' => Some(Self::P),
            'q' => Some(Self::Q),
            'r' => Some(Self::R),
            's' => Some(Self::S),
            't' => Some(Self::T),
            'u' => Some(Self::U),
            'v' => Some(Self::V),
            'w' => Some(Self::W),
            'x' => Some(Self::X),
            'y' => Some(Self::Y),
            'z' => Some(Self::Z),
            ' ' => Some(Self::Space),
            _ => None,
        }
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, VariantArray, Default)]
pub enum ActionType {
    #[default]
    Idle,
    TurnLeft,
    TurnRight,
    Forward,
    Attack,
}
impl ActionType {
    pub const ALL: &'static [ActionType] = &[
        ActionType::TurnLeft,
        ActionType::TurnRight,
        ActionType::Forward,
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
    /// declaration order including `Idle` (`0..=4`).
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable bit in an organism's per-tick motor-command mask. Idle has no
    /// command bit; the four explicit actions occupy bits 0 through 3 in the
    /// same order as [`Self::ALL`].
    pub const fn command_bit(self) -> u8 {
        match self {
            Self::Idle => 0,
            _ => 1 << (self as u8 - 1),
        }
    }

    /// Whether this action is contingent on world state and can therefore
    /// fail. Single source of truth for both the sim's `ActionRecord`
    /// initialization and evaluation failure counting.
    pub const fn can_fail(self) -> bool {
        matches!(self, ActionType::Forward | ActionType::Attack)
    }
}

#[cfg(feature = "instrumentation")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionRecord {
    pub organism_id: OrganismId,
    /// Compatibility/observer projection: the highest-logit emitted command,
    /// or Idle when no command was emitted.
    pub selected_action: ActionType,
    /// Bitset of every explicit command emitted this tick.
    pub selected_action_mask: u8,
    /// Selected contingent commands (Forward/Attack) that did not succeed.
    pub failed_action_mask: u8,
    pub action_failed: bool,
    pub age_turns: u64,
    pub utilization: f32,
    /// Cumulative successful attack energy transfers.
    pub successful_attacks_count: u64,
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "receptor_type")]
pub enum SensoryReceptor {
    Symbol { symbol: Symbol },
}

impl SensoryReceptor {
    pub const BASELINE_NEURON_COUNT: u32 = Symbol::COUNT as u32;
    pub const TOTAL_NEURON_COUNT: u32 = Symbol::COUNT as u32;

    pub fn ordered() -> impl Iterator<Item = Self> {
        Symbol::ALL
            .into_iter()
            .map(|symbol| Self::Symbol { symbol })
    }

    pub fn active(_predation_enabled: bool) -> impl Iterator<Item = Self> {
        Self::ordered()
    }

    pub const fn is_predation_only(self) -> bool {
        false
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
    use super::{SensoryReceptor, Symbol};

    #[test]
    fn sensory_receptors_match_the_symbol_alphabet() {
        let receptors = SensoryReceptor::ordered().collect::<Vec<_>>();
        assert_eq!(receptors.len(), Symbol::COUNT);
        assert_eq!(
            receptors,
            vec![
                SensoryReceptor::Symbol { symbol: Symbol::A },
                SensoryReceptor::Symbol { symbol: Symbol::B },
                SensoryReceptor::Symbol { symbol: Symbol::C },
                SensoryReceptor::Symbol { symbol: Symbol::D },
                SensoryReceptor::Symbol { symbol: Symbol::E },
                SensoryReceptor::Symbol { symbol: Symbol::F },
                SensoryReceptor::Symbol { symbol: Symbol::G },
                SensoryReceptor::Symbol { symbol: Symbol::H },
                SensoryReceptor::Symbol { symbol: Symbol::I },
                SensoryReceptor::Symbol { symbol: Symbol::J },
                SensoryReceptor::Symbol { symbol: Symbol::K },
                SensoryReceptor::Symbol { symbol: Symbol::L },
                SensoryReceptor::Symbol { symbol: Symbol::M },
                SensoryReceptor::Symbol { symbol: Symbol::N },
                SensoryReceptor::Symbol { symbol: Symbol::O },
                SensoryReceptor::Symbol { symbol: Symbol::P },
                SensoryReceptor::Symbol { symbol: Symbol::Q },
                SensoryReceptor::Symbol { symbol: Symbol::R },
                SensoryReceptor::Symbol { symbol: Symbol::S },
                SensoryReceptor::Symbol { symbol: Symbol::T },
                SensoryReceptor::Symbol { symbol: Symbol::U },
                SensoryReceptor::Symbol { symbol: Symbol::V },
                SensoryReceptor::Symbol { symbol: Symbol::W },
                SensoryReceptor::Symbol { symbol: Symbol::X },
                SensoryReceptor::Symbol { symbol: Symbol::Y },
                SensoryReceptor::Symbol { symbol: Symbol::Z },
                SensoryReceptor::Symbol {
                    symbol: Symbol::Space
                },
                SensoryReceptor::Symbol {
                    symbol: Symbol::End
                },
            ]
        );
        assert_eq!(SensoryReceptor::active(false).count(), Symbol::COUNT);
        assert_eq!(SensoryReceptor::active(true).count(), Symbol::COUNT);
    }
}

pub fn inter_neuron_id(index: u32) -> NeuronId {
    let mut dense_id = INTER_NEURON_ID_BASE
        .checked_add(index)
        .expect("inter-neuron index exceeds the u32 runtime ID space");
    if dense_id >= ACTION_NEURON_ID_BASE {
        dense_id = dense_id
            .checked_add(Symbol::COUNT as u32)
            .expect("inter-neuron index exceeds the u32 runtime ID space");
    }
    NeuronId(dense_id)
}

pub fn inter_neuron_index(id: NeuronId, num_neurons: u32) -> Option<u32> {
    let mut dense_id = id.0;
    let island_end = ACTION_NEURON_ID_BASE.checked_add(Symbol::COUNT as u32)?;
    if (ACTION_NEURON_ID_BASE..island_end).contains(&id.0) {
        return None;
    }
    if id.0 >= island_end {
        dense_id = dense_id.checked_sub(Symbol::COUNT as u32)?;
    }
    let idx = dense_id.checked_sub(INTER_NEURON_ID_BASE)?;
    (idx < num_neurons).then_some(idx)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecycleGenes {
    pub plasticity_maturity_ticks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlasticityGenes {
    #[serde(default)]
    pub initial_learning_rate: f32,
    #[serde(default = "default_juvenile_eta_scale")]
    pub juvenile_eta_scale: f32,
    #[serde(default = "default_eligibility_retention")]
    pub eligibility_retention: f32,
    /// Per-tick retention of the learned lifetime component of a synaptic
    /// weight. One makes learning persistent for the lifetime; smaller values
    /// forget learned displacement toward the inherited baseline.
    #[serde(default = "default_fast_weight_retention")]
    pub fast_weight_retention: f32,
    /// Heritable exploration/exploitation scale applied to an environment's
    /// reference action temperature during within-lifetime learning.
    #[serde(default = "default_action_temperature_scale")]
    pub action_temperature_scale: f32,
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
    /// Inherited starting bias for the generic reward-prediction output.
    #[serde(default)]
    pub value_bias: f32,
    pub edges: Vec<SynapseGene>,
}

/// Heritable hidden-node parameters keyed by a stable structural identity.
/// Runtime brains remap these IDs to dense [`NeuronId`] values at birth.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct HiddenNodeGene {
    pub id: GeneNodeId,
    pub bias: f32,
    pub log_time_constant: f32,
    /// Signed gain for the global reward-prediction-error broadcast. A zero
    /// receptor leaves this neuron insensitive to neuromodulatory feedback.
    #[serde(default)]
    pub neuromodulatory_receptor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrganismGenome {
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
        let action_count = Symbol::COUNT;
        Self {
            lifecycle: LifecycleGenes {
                plasticity_maturity_ticks: 0,
            },
            plasticity: PlasticityGenes {
                initial_learning_rate: 0.0,
                juvenile_eta_scale: 0.5,
                eligibility_retention: 0.9,
                fast_weight_retention: 1.0,
                action_temperature_scale: 1.0,
                max_weight_delta_per_tick: 0.05,
                synapse_prune_threshold: 0.01,
            },
            brain: BrainTopology {
                hidden_nodes: vec![HiddenNodeGene {
                    id: seed_hidden_gene_node_id(0),
                    bias: 0.0,
                    log_time_constant: 0.0,
                    neuromodulatory_receptor: 0.0,
                }],
                action_biases: vec![0.0; action_count],
                value_bias: 0.0,
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
    pub timing: SynapseTiming,
    pub weight: f32,
    pub plasticity_coefficient: f32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapseEdge {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub timing: SynapseTiming,
    /// Zero-based hidden-layer indices compiled at expression time for the
    /// recurrent hot path. `pre_inter_index` is populated for hidden-source
    /// recurrence; `pre_action_index` is populated for action efference-copy
    /// recurrence; `post_inter_index` is populated for both.
    pub pre_inter_index: Option<u32>,
    #[serde(default)]
    pub pre_action_index: Option<u32>,
    pub post_inter_index: Option<u32>,
    /// Stable inherited component copied from the expressed genome. Runtime
    /// learning changes `weight`; retention pulls that learned displacement
    /// back toward this baseline without altering the genome.
    #[serde(default)]
    pub inherited_weight: f32,
    pub weight: f32,
    pub plasticity_coefficient: f32,
    #[serde(default)]
    pub eligibility: f32,
    // `default` (not `skip`): normally consumed+zeroed the same tick it is set,
    // but a gestating organism's post-commit plasticity is skipped, so a nonzero
    // Persisted because it is live plasticity state carried between ticks.
    #[serde(default)]
    pub pending_coactivation: f32,
}

fn default_eligibility_retention() -> f32 {
    0.95
}

fn default_fast_weight_retention() -> f32 {
    1.0
}

fn default_action_temperature_scale() -> f32 {
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
    pub activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensoryNeuronState {
    pub neuron: NeuronState,
    #[serde(flatten)]
    pub receptor: SensoryReceptor,
    pub synapses: Vec<SynapseEdge>,
    /// Cached index of the first output-targeting edge in `synapses` (which
    /// keep inter targets first and output targets second, with numeric post-ID
    /// ordering inside each group). Maintained by
    /// `refresh_output_synapse_starts_and_count` at birth and after pruning.
    //
    // `default` (not `skip`): persisted in the world blob so the post-load
    // edge split is correct without a rehydrate pass (the value is a pure
    // function of `synapses`, saved in a consistent between-tick state). Legacy
    // payloads that omit it default to 0; such a payload must be re-expressed.
    #[serde(default)]
    pub output_synapse_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InterNeuronState {
    pub neuron: NeuronState,
    #[serde(default)]
    pub state: f32,
    pub alpha: f32,
    /// Evolvable sensitivity to the previous signed reward-prediction error.
    #[serde(default)]
    pub neuromodulatory_receptor: f32,
    pub synapses: Vec<SynapseEdge>,
    #[serde(default)]
    pub output_synapse_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionNeuronState {
    pub neuron_id: NeuronId,
    pub logit: f32,
    pub symbol: Symbol,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainState {
    pub sensory: Vec<SensoryNeuronState>,
    pub inter: Vec<InterNeuronState>,
    pub action: Vec<ActionNeuronState>,
    /// Dense hidden-to-hidden previous-tick connections. These are separate
    /// from the per-neuron outgoing arrays only in the compiled runtime; the
    /// genome retains one unified connection-gene collection.
    pub recurrent_synapses: Vec<SynapseEdge>,
    /// Previous-tick selected-action projections into hidden neurons. These
    /// are ordinary evolvable genome edges with action sources.
    #[serde(default)]
    pub action_feedback_synapses: Vec<SynapseEdge>,
    /// Frozen hidden activations read by every recurrent synapse on the next
    /// evaluation tick. Persisted as live world state for exact snapshot replay.
    pub previous_inter_activations: Vec<f32>,
    /// One-hot efference copy of the action actually selected last tick.
    #[serde(default)]
    pub previous_action_activations: [f32; Symbol::COUNT],
    /// Signed reward surprise broadcast on the next recurrent step.
    #[serde(default)]
    pub previous_prediction_error: f32,
    /// Runtime reward prediction bias and its decaying critic trace.
    #[serde(default)]
    pub value_bias: f32,
    /// Inherited baseline for the runtime reward-prediction bias.
    #[serde(default)]
    pub inherited_value_bias: f32,
    #[serde(default)]
    pub value_bias_eligibility: f32,
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
    pub energy: u32,
    /// Energy stash captured at sensing time for the post-action plasticity
    /// reward signal, rolled forward at the start of every tick's sensing pass.
    #[serde(default)]
    pub energy_at_last_sensing: u32,
    /// Signed predation energy flow committed during the preceding tick,
    /// excluding the universal one-energy lifetime drain.
    pub energy_flow_last_tick: i32,
    #[serde(default)]
    pub successful_attacks_count: u64,
    #[serde(default)]
    pub last_action_taken: ActionType,
    pub last_action_symbol: Symbol,
    /// Every explicit motor command emitted on the most recent tick. This is a
    /// bitset built from [`ActionType::command_bit`].
    #[serde(default)]
    pub last_action_mask: u8,
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
        energy: u32,
        successful_attacks_count: u64,
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
            energy_at_last_sensing: energy,
            energy_flow_last_tick: 0,
            successful_attacks_count,
            last_action_taken,
            last_action_symbol: Symbol::A,
            last_action_mask: last_action_taken.command_bit(),
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain,
            genome,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MetricsSnapshot {
    pub turns: u64,
    pub organisms: u32,
    pub synapse_ops_last_turn: u64,
    pub actions_applied_last_turn: u64,
    pub predations_last_turn: u64,
    pub starvations_last_turn: u64,
    pub age_deaths_last_turn: u64,
    /// Fail-closed physical energy accounting for the most recently completed
    /// tick. The row is all-zero before the first tick.
    #[serde(default)]
    pub energy_ledger_last_turn: EnergyLedgerRow,
}

/// Per-tick energy accounting over organism energy. All values are simulation
/// energy units. Successful attacks are internal transfers; every emitted
/// Attack command has a separate dissipative attempt cost, including misses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct EnergyLedgerRow {
    pub turn: u64,
    pub organism_energy_before: f64,
    pub organism_energy_after: f64,
    pub tick_drain_energy: f64,
    pub attack_transfer_energy: f64,
    pub attack_attempt_cost: f64,
    pub organism_residual: f64,
    pub total_residual: f64,
    pub residual_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub turn: u64,
    pub rng_seed: u64,
    pub config: WorldConfig,
    pub organisms: Vec<OrganismState>,
    pub terrain: Vec<TerrainCell>,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "id")]
pub enum Occupant {
    Organism(OrganismId),
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
    pub metrics: MetricsSnapshot,
}

pub const ORGANISM_SHAPE: f32 = 0.2;
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
