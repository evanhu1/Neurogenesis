//! The physics boundary. The substrate owns genome/brain/operators and the
//! `PopulationDriver`; the environment owns *only* physics. It reads bodies
//! through `BodyView` and requests changes through `EffectSink`, which the
//! driver applies in handle order (the snapshot-then-apply discipline promoted
//! from `sim-core`'s `apply_social_color_mortality`). The environment never
//! touches genome/brain/energy internals.

use crate::brain::BrainNet;
use crate::catalog::{ActionLayout, ObsLayout, SubstrateCatalog};
use crate::genome::Genome;
use crate::rng::Rng;
use serde::{Deserialize, Serialize};

/// Stable index into the population. Handles never move (the driver does not
/// compact in place); dead bodies are skipped via `Body::alive`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BodyHandle(pub u32);

/// Body parameters an environment derives once from morphology (e.g. hex `size`
/// from gestation), cached on the body so both sides read the same value and
/// nothing can diverge.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DerivedBodyParams {
    pub size: f32,
    pub max_health: f32,
    pub investment_energy: f32,
    pub metabolic_base: f32,
}

/// Immutable read of one body handed to the environment.
pub struct BodyView<'a> {
    pub handle: BodyHandle,
    pub id: u64,
    pub energy: f32,
    pub health: f32,
    pub age_turns: u64,
    pub is_gestating: bool,
    pub generation: u64,
    /// Energy stashed at the previous sensing pass — lets environments compute
    /// the within-tick energy-delta sensor without owning the stash.
    pub energy_at_last_sensing: f32,
    pub morphology: &'a [f32],
    pub derived: &'a DerivedBodyParams,
    /// Developed brain size, exposed so environments can price metabolism on
    /// neural complexity without reaching into the brain internals.
    pub brain_neurons: u32,
    pub brain_edges: u32,
}

/// The result of the brain's action selection for one body.
pub struct ActionOutput<'a> {
    pub logits: &'a [f32],
    /// Winning action-layout slot (`None` == implicit idle).
    pub selected: Option<usize>,
    pub confidence: f32,
    pub layout: &'a ActionLayout,
}

impl ActionOutput<'_> {
    /// Catalog actuator index of the selected action, if any.
    pub fn selected_actuator(&self) -> Option<usize> {
        self.selected
            .and_then(|slot| self.layout.actuator_indices.get(slot).copied())
    }
}

/// A driver-detected mating intent: the initiator wants to mate with `target`.
pub struct MateIntent {
    pub target: BodyHandle,
    pub confidence: f32,
}

/// Effects the environment requests; the driver applies them deterministically
/// in handle order.
#[derive(Default, Clone)]
pub struct EffectSink {
    pub energy_deltas: Vec<(BodyHandle, f32)>,
    pub health_deltas: Vec<(BodyHandle, f32)>,
    pub deaths: Vec<BodyHandle>,
}

impl EffectSink {
    pub fn clear(&mut self) {
        self.energy_deltas.clear();
        self.health_deltas.clear();
        self.deaths.clear();
    }
    pub fn add_energy(&mut self, handle: BodyHandle, delta: f32) {
        self.energy_deltas.push((handle, delta));
    }
    pub fn add_health(&mut self, handle: BodyHandle, delta: f32) {
        self.health_deltas.push((handle, delta));
    }
    pub fn kill(&mut self, handle: BodyHandle) {
        self.deaths.push(handle);
    }
}

/// The population as the environment sees it: handle-addressable, read-only.
pub trait PopulationRead {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn view(&self, handle: BodyHandle) -> Option<BodyView<'_>>;
    fn is_alive(&self, handle: BodyHandle) -> bool;
}

/// The environment/physics half of the engine.
pub trait Environment {
    /// Per-body decoded action intent (env-specific).
    type Intents;
    /// Placement token returned by `place_*` and consumed by `attach`.
    type SpawnSite;

    fn catalog(&self) -> &SubstrateCatalog;

    /// Compute derived body params from morphology — called ONCE at birth.
    fn derive_body_params(&self, morphology: &[f32]) -> DerivedBodyParams;

    /// Fill the observation vector for a body per its obs layout.
    fn observe(&self, view: &BodyView, layout: &ObsLayout, out: &mut [f32]);

    /// Per-tick metabolic energy cost for a body.
    fn metabolic_cost(&self, view: &BodyView) -> f32;

    /// Advance world physics not driven by actions (food, terrain, hazards,
    /// social fields). May request effects.
    fn step_world(&mut self, pop: &dyn PopulationRead, rng: &mut Rng, sink: &mut EffectSink);

    /// Decode a body's action output into an environment intent.
    fn decode_intents(&self, view: &BodyView, action: &ActionOutput) -> Self::Intents;

    /// Resolve all decoded action intents against the world. May request
    /// effects. Cross-body ordering is the environment's responsibility (it
    /// should be order-independent / id-keyed for determinism).
    fn resolve_actions(
        &mut self,
        intents: &[Self::Intents],
        pop: &dyn PopulationRead,
        rng: &mut Rng,
        sink: &mut EffectSink,
    );

    /// If the body selected its mate action and a valid partner is in range,
    /// return the mating intent (geometry only — the driver owns the protocol).
    fn mate_intent(&self, view: &BodyView, action: &ActionOutput) -> Option<MateIntent>;

    /// Choose a spawn site near the carrier for a birth; `None` == blocked.
    fn place_birth(&mut self, carrier: &BodyView, rng: &mut Rng) -> Option<Self::SpawnSite>;

    /// Choose a spawn site for a founder organism.
    fn place_founder(&mut self, rng: &mut Rng) -> Option<Self::SpawnSite>;

    /// Register a newly-created body (view carries its handle + morphology) at a
    /// spawn site.
    fn attach(&mut self, view: &BodyView, site: Self::SpawnSite);

    /// Notified after bodies die (drop corpses, free cells). May request
    /// effects (e.g. corpse food).
    fn on_deaths(&mut self, dead: &[BodyHandle], pop: &dyn PopulationRead, sink: &mut EffectSink);
}

/// Heritable + developed state of one organism, owned by the driver.
#[derive(Clone, Serialize, Deserialize)]
pub struct Body {
    pub id: u64,
    pub alive: bool,
    pub energy: f32,
    pub health: f32,
    pub age_turns: u64,
    pub generation: u64,
    pub is_gestating: bool,
    pub energy_at_last_sensing: f32,
    pub morphology: Vec<f32>,
    pub derived: DerivedBodyParams,
    pub genome: Genome,
    pub brain: BrainNet,
    pub obs_layout: ObsLayout,
    pub action_layout: ActionLayout,
    pub gestation: Option<Gestation>,
}

/// An in-progress pregnancy: the co-parent's genome snapshot + countdown.
#[derive(Clone, Serialize, Deserialize)]
pub struct Gestation {
    pub partner_genome: Genome,
    pub remaining: u8,
    pub investment: f32,
}

impl Body {
    pub fn view(&self, handle: BodyHandle) -> BodyView<'_> {
        BodyView {
            handle,
            id: self.id,
            energy: self.energy,
            health: self.health,
            age_turns: self.age_turns,
            is_gestating: self.is_gestating,
            generation: self.generation,
            energy_at_last_sensing: self.energy_at_last_sensing,
            morphology: &self.morphology,
            derived: &self.derived,
            brain_neurons: self.brain.neurons.len() as u32,
            brain_edges: self.brain.edges.len() as u32,
        }
    }
}
