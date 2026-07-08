//! `sim-toyenv` — the "Chemotaxis Ribbon", a second environment that shares no
//! hex assumption: a 1-D toroidal ribbon with a drifting scalar nutrient field.
//! Its only job is to prove the substrate is environment-agnostic — it reuses
//! `sim-substrate` **unchanged** (no substrate edits were made for it). Sensors
//! are gradient/concentration/energy/crowding (no rays, no RGB, no health);
//! actuators are move-left/right/eat/mate; morphology is metabolic-thrift +
//! gut-capacity (no body color, no vision distance). Mating uses the identical
//! shared protocol.

use rand::Rng as _;
use sim_substrate::catalog::{plane, ActuatorSpec, Coord, MorphologyParam, ObsLayout, SensorSpec};
use sim_substrate::environment::{
    ActionOutput, BodyHandle, BodyView, DerivedBodyParams, EffectSink, Environment, MateIntent,
    PopulationRead,
};
use sim_substrate::rng::Rng;
use sim_substrate::SubstrateCatalog;

const M_THRIFT: usize = 0;
const M_GUT: usize = 1;
const DIFFUSION: f32 = 0.10;
const REGROW_PER_STEP: f32 = 0.02;
const MAX_CONC: f32 = 5.0;

pub struct ChemotaxisRibbon {
    catalog: SubstrateCatalog,
    len: usize,
    conc: Vec<f32>,
    org_at: Vec<Option<BodyHandle>>,
    pos: Vec<Option<usize>>,
}

impl ChemotaxisRibbon {
    pub fn new(len: usize, seed: u64) -> Self {
        let catalog = SubstrateCatalog {
            sensors: vec![
                SensorSpec {
                    key: "gradient_sign".into(),
                    arity: 1,
                    coord: Coord::new(-0.6, 0.0, plane::EXTEROCEPTIVE),
                },
                SensorSpec {
                    key: "local_concentration".into(),
                    arity: 1,
                    coord: Coord::new(0.0, 0.0, plane::EXTEROCEPTIVE),
                },
                SensorSpec {
                    key: "energy".into(),
                    arity: 1,
                    coord: Coord::new(-0.3, 0.0, plane::INTEROCEPTIVE),
                },
                SensorSpec {
                    key: "crowding".into(),
                    arity: 1,
                    coord: Coord::new(0.6, 0.0, plane::EXTEROCEPTIVE),
                },
            ],
            actuators: vec![
                ActuatorSpec { key: "move_left".into(), coord: Coord::new(-0.8, 0.0, plane::ACTUATOR) },
                ActuatorSpec { key: "move_right".into(), coord: Coord::new(0.8, 0.0, plane::ACTUATOR) },
                ActuatorSpec { key: "eat".into(), coord: Coord::new(0.0, 0.0, plane::ACTUATOR) },
                ActuatorSpec { key: "mate".into(), coord: Coord::new(0.0, 0.5, plane::ACTUATOR) },
            ],
            morphology: vec![
                MorphologyParam { key: "metabolic_thrift".into(), min: 0.5, max: 2.0, default: 1.0 },
                MorphologyParam { key: "gut_capacity".into(), min: 1.0, max: 8.0, default: 3.0 },
            ],
        };
        // Seed a smooth-ish initial nutrient field.
        let conc: Vec<f32> = (0..len)
            .map(|i| {
                let h = sim_substrate::rng::mix_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
                (h >> 40) as f32 / ((1u64 << 24) as f32) * MAX_CONC
            })
            .collect();
        ChemotaxisRibbon {
            catalog,
            len,
            conc,
            org_at: vec![None; len],
            pos: Vec::new(),
        }
    }

    fn ensure(&mut self, h: BodyHandle) {
        let need = h.0 as usize + 1;
        if self.pos.len() < need {
            self.pos.resize(need, None);
        }
    }

    fn thrift(m: &[f32]) -> f32 {
        m.get(M_THRIFT).copied().unwrap_or(1.0).max(0.1)
    }
    fn gut(m: &[f32]) -> f32 {
        m.get(M_GUT).copied().unwrap_or(3.0).max(0.5)
    }
}

pub struct RibbonSite(pub usize);

#[derive(Clone, Copy)]
enum Kind {
    Idle,
    Left,
    Right,
    Eat,
}

pub struct RibbonIntent {
    handle: BodyHandle,
    kind: Kind,
    confidence: f32,
}

impl Environment for ChemotaxisRibbon {
    type Intents = RibbonIntent;
    type SpawnSite = RibbonSite;

    fn catalog(&self) -> &SubstrateCatalog {
        &self.catalog
    }

    fn derive_body_params(&self, morphology: &[f32]) -> DerivedBodyParams {
        let gut = Self::gut(morphology);
        let max_health = 20.0 + gut * 4.0;
        DerivedBodyParams {
            size: max_health,
            max_health,
            investment_energy: max_health * 0.3,
            metabolic_base: 0.0,
        }
    }

    fn observe(&self, view: &BodyView, layout: &ObsLayout, out: &mut [f32]) {
        let Some(cell) = self.pos.get(view.handle.0 as usize).copied().flatten() else {
            return;
        };
        let left = (cell + self.len - 1) % self.len;
        let right = (cell + 1) % self.len;
        let gradient = (self.conc[right] - self.conc[left]).signum();
        let local = (self.conc[cell] / MAX_CONC).clamp(0.0, 1.0);
        let energy =
            (view.energy / (view.energy.abs() + view.derived.max_health.max(1.0))).clamp(0.0, 1.0);
        let crowding = {
            let mut n = 0;
            if self.org_at[left].is_some() {
                n += 1;
            }
            if self.org_at[right].is_some() {
                n += 1;
            }
            n as f32 / 2.0
        };
        for (slot, &si) in layout.sensor_indices.iter().enumerate() {
            let off = layout.offsets[slot];
            let value = match self.catalog.sensors[si].key.as_str() {
                "gradient_sign" => gradient,
                "local_concentration" => local,
                "energy" => energy,
                "crowding" => crowding,
                _ => 0.0,
            };
            if let Some(o) = out.get_mut(off) {
                *o = value;
            }
        }
    }

    fn metabolic_cost(&self, view: &BodyView) -> f32 {
        let neurons = view.brain_neurons as f32;
        let thrift = Self::thrift(view.morphology);
        0.02 * (neurons + 4.0) / thrift
    }

    fn decode_intents(&self, view: &BodyView, action: &ActionOutput) -> RibbonIntent {
        let kind = match action.selected_actuator().map(|i| self.catalog.actuators[i].key.as_str()) {
            Some("move_left") => Kind::Left,
            Some("move_right") => Kind::Right,
            Some("eat") => Kind::Eat,
            _ => Kind::Idle,
        };
        RibbonIntent {
            handle: view.handle,
            kind,
            confidence: action.confidence,
        }
    }

    fn resolve_actions(
        &mut self,
        intents: &[RibbonIntent],
        pop: &dyn PopulationRead,
        _rng: &mut Rng,
        sink: &mut EffectSink,
    ) {
        let mut moves: Vec<(BodyHandle, usize, f32)> = Vec::new();
        for intent in intents {
            let Some(cell) = self.pos.get(intent.handle.0 as usize).copied().flatten() else {
                continue;
            };
            match intent.kind {
                Kind::Left => {
                    moves.push((intent.handle, (cell + self.len - 1) % self.len, intent.confidence))
                }
                Kind::Right => {
                    moves.push((intent.handle, (cell + 1) % self.len, intent.confidence))
                }
                Kind::Eat => {
                    let gut = pop
                        .view(intent.handle)
                        .map(|v| Self::gut(v.morphology))
                        .unwrap_or(3.0);
                    let take = self.conc[cell].min(gut);
                    self.conc[cell] -= take;
                    sink.add_energy(intent.handle, take * 6.0);
                }
                Kind::Idle => {}
            }
        }
        // Deterministic move resolution: (target cell, confidence desc, handle asc).
        moves.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.0 .0.cmp(&b.0 .0))
        });
        let mut claimed = vec![false; self.len];
        for (h, target, _c) in moves {
            if self.org_at[target].is_some() || claimed[target] {
                continue;
            }
            let Some(from) = self.pos[h.0 as usize] else {
                continue;
            };
            claimed[target] = true;
            self.org_at[from] = None;
            self.org_at[target] = Some(h);
            self.pos[h.0 as usize] = Some(target);
        }
    }

    fn mate_intent(&self, view: &BodyView, action: &ActionOutput) -> Option<MateIntent> {
        let is_mate = action
            .selected_actuator()
            .map(|i| self.catalog.actuators[i].key == "mate")
            .unwrap_or(false);
        if !is_mate {
            return None;
        }
        let cell = self.pos.get(view.handle.0 as usize).copied().flatten()?;
        let left = (cell + self.len - 1) % self.len;
        let right = (cell + 1) % self.len;
        self.org_at[left]
            .or(self.org_at[right])
            .map(|target| MateIntent {
                target,
                confidence: action.confidence,
            })
    }

    fn step_world(&mut self, _pop: &dyn PopulationRead, _rng: &mut Rng, _sink: &mut EffectSink) {
        // Diffuse + regrow the nutrient field (deterministic). Indexed loop:
        // each cell reads its wrapped neighbors, which iter_mut cannot express.
        let mut next = self.conc.clone();
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.len {
            let left = self.conc[(i + self.len - 1) % self.len];
            let right = self.conc[(i + 1) % self.len];
            next[i] = self.conc[i] + DIFFUSION * (0.5 * (left + right) - self.conc[i]);
            next[i] = (next[i] + REGROW_PER_STEP).min(MAX_CONC);
        }
        self.conc = next;
    }

    fn place_birth(&mut self, carrier: &BodyView, _rng: &mut Rng) -> Option<RibbonSite> {
        let cell = self.pos.get(carrier.handle.0 as usize).copied().flatten()?;
        for d in [self.len - 1, 1] {
            let c = (cell + d) % self.len;
            if self.org_at[c].is_none() {
                return Some(RibbonSite(c));
            }
        }
        None
    }

    fn place_founder(&mut self, rng: &mut Rng) -> Option<RibbonSite> {
        for _ in 0..32 {
            let c = rng.random_range(0..self.len);
            if self.org_at[c].is_none() {
                return Some(RibbonSite(c));
            }
        }
        None
    }

    fn attach(&mut self, view: &BodyView, site: RibbonSite) {
        self.ensure(view.handle);
        self.pos[view.handle.0 as usize] = Some(site.0);
        self.org_at[site.0] = Some(view.handle);
    }

    fn on_deaths(&mut self, dead: &[BodyHandle], pop: &dyn PopulationRead, _sink: &mut EffectSink) {
        for h in dead {
            let Some(cell) = self.pos.get(h.0 as usize).copied().flatten() else {
                continue;
            };
            self.org_at[cell] = None;
            self.pos[h.0 as usize] = None;
            // Corpse fertilizes the local nutrient field.
            if let Some(view) = pop.view(*h) {
                self.conc[cell] = (self.conc[cell] + view.energy.max(0.0) * 0.05).min(MAX_CONC);
            }
        }
    }
}
