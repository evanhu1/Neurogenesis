//! A self-contained proof that the substrate ticks a live population
//! deterministically. `RingLife` is a tiny in-example `Environment` (a ring of
//! cells with regrowing food) — enough to exercise sensing, actions, mating,
//! births, deaths, and plasticity without any hex assumptions. Two identical
//! runs must produce byte-identical populations.

use rand::Rng as _;
use sim_substrate::catalog::{plane, ActuatorSpec, Coord, MorphologyParam, SensorSpec};
use sim_substrate::environment::{
    ActionOutput, BodyHandle, BodyView, DerivedBodyParams, EffectSink, Environment, MateIntent,
    PopulationRead,
};
use sim_substrate::rng::Rng;
use sim_substrate::seed::seed_genome;
use sim_substrate::{DriverConfig, PopulationDriver, SubstrateCatalog};

const RING: usize = 64;

struct RingLife {
    catalog: SubstrateCatalog,
    food: Vec<f32>,
    /// handle -> cell (None once dead/removed).
    cell: Vec<Option<usize>>,
}

enum IntentKind {
    Idle,
    Eat,
    Move,
    Mate,
}

struct RingIntent {
    handle: BodyHandle,
    kind: IntentKind,
}

impl RingLife {
    fn new() -> Self {
        let catalog = SubstrateCatalog {
            sensors: vec![
                SensorSpec {
                    key: "intero.energy".into(),
                    arity: 1,
                    coord: Coord::new(0.0, 0.0, plane::INTEROCEPTIVE),
                },
                SensorSpec {
                    key: "extero.food".into(),
                    arity: 1,
                    coord: Coord::new(-0.5, 0.0, plane::EXTEROCEPTIVE),
                },
                SensorSpec {
                    key: "extero.crowding".into(),
                    arity: 1,
                    coord: Coord::new(0.5, 0.0, plane::EXTEROCEPTIVE),
                },
            ],
            actuators: vec![
                ActuatorSpec {
                    key: "eat".into(),
                    coord: Coord::new(-0.5, 0.0, plane::ACTUATOR),
                },
                ActuatorSpec {
                    key: "move".into(),
                    coord: Coord::new(0.0, 0.0, plane::ACTUATOR),
                },
                ActuatorSpec {
                    key: "mate".into(),
                    coord: Coord::new(0.5, 0.0, plane::ACTUATOR),
                },
            ],
            morphology: vec![MorphologyParam {
                key: "thrift".into(),
                min: 0.5,
                max: 2.0,
                default: 1.0,
            }],
        };
        RingLife {
            catalog,
            food: vec![10.0; RING],
            cell: Vec::new(),
        }
    }

    fn cell_of(&self, handle: BodyHandle) -> Option<usize> {
        self.cell.get(handle.0 as usize).copied().flatten()
    }

    fn occupancy(&self, cell: usize) -> usize {
        self.cell.iter().filter(|c| **c == Some(cell)).count()
    }

    fn ensure(&mut self, handle: BodyHandle) {
        let idx = handle.0 as usize;
        if idx >= self.cell.len() {
            self.cell.resize(idx + 1, None);
        }
    }
}

impl Environment for RingLife {
    type Intents = RingIntent;
    type SpawnSite = usize;

    fn catalog(&self) -> &SubstrateCatalog {
        &self.catalog
    }

    fn derive_body_params(&self, morphology: &[f32]) -> DerivedBodyParams {
        let thrift = morphology.first().copied().unwrap_or(1.0);
        DerivedBodyParams {
            size: 1.0,
            max_health: 10.0,
            investment_energy: 30.0,
            metabolic_base: 0.6 / thrift.max(0.1),
        }
    }

    fn observe(&self, view: &BodyView, layout: &sim_substrate::ObsLayout, out: &mut [f32]) {
        let cell = self.cell_of(view.handle);
        for (slot, &si) in layout.sensor_indices.iter().enumerate() {
            let off = layout.offsets[slot];
            let value = match self.catalog.sensors[si].key.as_str() {
                "intero.energy" => (view.energy / 100.0).clamp(0.0, 1.0),
                "extero.food" => cell.map(|c| (self.food[c] / 20.0).clamp(0.0, 1.0)).unwrap_or(0.0),
                "extero.crowding" => cell
                    .map(|c| (self.occupancy(c) as f32 / 5.0).clamp(0.0, 1.0))
                    .unwrap_or(0.0),
                _ => 0.0,
            };
            if let Some(slot_out) = out.get_mut(off) {
                *slot_out = value;
            }
        }
    }

    fn metabolic_cost(&self, view: &BodyView) -> f32 {
        view.derived.metabolic_base
    }

    fn step_world(&mut self, _pop: &dyn PopulationRead, _rng: &mut Rng, _sink: &mut EffectSink) {
        for f in &mut self.food {
            *f = (*f + 0.5).min(20.0);
        }
    }

    fn decode_intents(&self, view: &BodyView, action: &ActionOutput) -> RingIntent {
        let kind = match action.selected_actuator().map(|i| self.catalog.actuators[i].key.as_str()) {
            Some("eat") => IntentKind::Eat,
            Some("move") => IntentKind::Move,
            Some("mate") => IntentKind::Mate,
            _ => IntentKind::Idle,
        };
        RingIntent {
            handle: view.handle,
            kind,
        }
    }

    fn resolve_actions(
        &mut self,
        intents: &[RingIntent],
        _pop: &dyn PopulationRead,
        rng: &mut Rng,
        sink: &mut EffectSink,
    ) {
        for intent in intents {
            let Some(cell) = self.cell_of(intent.handle) else {
                continue;
            };
            match intent.kind {
                IntentKind::Eat => {
                    let take = self.food[cell].min(8.0);
                    self.food[cell] -= take;
                    sink.add_energy(intent.handle, take);
                }
                IntentKind::Move => {
                    let dir = if rng.random::<bool>() { 1 } else { RING - 1 };
                    let new_cell = (cell + dir) % RING;
                    if let Some(slot) = self.cell.get_mut(intent.handle.0 as usize) {
                        *slot = Some(new_cell);
                    }
                }
                IntentKind::Mate | IntentKind::Idle => {}
            }
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
        let cell = self.cell_of(view.handle)?;
        for (idx, c) in self.cell.iter().enumerate() {
            if *c == Some(cell) && idx as u32 != view.handle.0 {
                return Some(MateIntent {
                    target: BodyHandle(idx as u32),
                    confidence: action.confidence,
                });
            }
        }
        None
    }

    fn place_birth(&mut self, carrier: &BodyView, rng: &mut Rng) -> Option<usize> {
        let base = self.cell_of(carrier.handle).unwrap_or(0);
        let dir = if rng.random::<bool>() { 1 } else { RING - 1 };
        Some((base + dir) % RING)
    }

    fn place_founder(&mut self, rng: &mut Rng) -> Option<usize> {
        Some(rng.random_range(0..RING))
    }

    fn attach(&mut self, view: &BodyView, site: usize) {
        self.ensure(view.handle);
        self.cell[view.handle.0 as usize] = Some(site);
    }

    fn on_deaths(&mut self, dead: &[BodyHandle], _pop: &dyn PopulationRead, _sink: &mut EffectSink) {
        for h in dead {
            if let Some(cell) = self.cell_of(*h) {
                self.food[cell] = (self.food[cell] + 5.0).min(20.0);
            }
            if let Some(slot) = self.cell.get_mut(h.0 as usize) {
                *slot = None;
            }
        }
    }
}

fn run(seed: u64, ticks: u64) -> (usize, u64, u64) {
    let mut env = RingLife::new();
    let catalog = env.catalog().clone();
    let genome = seed_genome(&catalog);
    let baseline_rates = genome.header.mutation_rates;
    let mut driver = PopulationDriver::new(seed, baseline_rates, DriverConfig::default());
    driver.seed_population(&mut env, 24, &genome);

    for _ in 0..ticks {
        driver.tick(&mut env);
    }

    let mut fp: u64 = 0xABCD_1234;
    let mut alive = 0usize;
    for b in &driver.bodies {
        if !b.alive {
            continue;
        }
        alive += 1;
        let q = (b.energy * 100.0) as i64 as u64;
        fp = sim_substrate::rng::mix_u64(fp ^ b.id ^ q.rotate_left(21));
    }
    (alive, driver.total_ever(), fp)
}

fn main() {
    let ticks = 3000;
    let (alive, total_ids, fp) = run(7, ticks);
    let (alive2, total_ids2, fp2) = run(7, ticks);
    println!("run  A: alive={alive} total_ids={total_ids} fingerprint={fp:#018x}");
    println!("run  B: alive={alive2} total_ids={total_ids2} fingerprint={fp2:#018x}");
    assert_eq!(
        (alive, total_ids, fp),
        (alive2, total_ids2, fp2),
        "determinism violated: identical seed produced different populations"
    );
    assert!(alive > 0, "population went extinct — expected a living population");
    assert!(total_ids > 24, "no births occurred — mating/reproduction never fired");
    println!("OK: deterministic, {alive} alive after {ticks} ticks, {total_ids} organisms ever born");
}
