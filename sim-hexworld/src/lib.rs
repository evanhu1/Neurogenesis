//! `sim-hexworld` — the hex ecology as a `sim_substrate::Environment`. All
//! physics (grid, terrain, food, metabolism, predation, spikes, social-color,
//! raycast sensing, action resolution) lives here; the substrate supplies the
//! genome/brain/operators and drives the tick loop. Determinism-critical
//! formulas (Kleiber 0.75, predation size ratio, zero-sum social-color
//! sin-transfer, corpse retention) are ported from `sim-core`.

pub mod catalog;
pub mod grid;
pub mod sim;

pub use sim::{HexConfig, HexSim};

use grid::{cell_index, hex_neighbor, opposite_direction, rotate_left, rotate_right, ALL_FACINGS};
use rand::Rng as _;
use sim_substrate::catalog::ObsLayout;
use sim_substrate::environment::{
    ActionOutput, BodyHandle, BodyView, DerivedBodyParams, EffectSink, Environment, MateIntent,
    PopulationRead,
};
use sim_substrate::rng::Rng;
use sim_substrate::SubstrateCatalog;
use sim_types::{
    color_hue, food_visual, organism_visual, FacingDirection, FoodKind, RgbColor, VisualProperties,
};

const SPIKE_DAMAGE_FRACTION: f32 = 0.10;
const ATTACK_DAMAGE_FRACTION: f32 = 0.5;
const HEALTH_REGEN_FRACTION: f32 = 0.10;
const CORPSE_ENERGY_RETENTION: f32 = 0.80;
const SOCIAL_DAMAGE: f32 = 1.0;
const BODY_MASS_METABOLIC_EXPONENT: f32 = 0.75;
const PASSIVE_METABOLISM_COST_PER_UNIT: f32 = 0.03;
const BODY_MASS_METABOLIC_COST_COEFF: f32 = 0.01;
const FOOD_REGROW_INTERVAL: u64 = 2;
const PLANT_ENERGY: f32 = 60.0;

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum OccKind {
    Empty,
    Wall,
    Food,
    Org,
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum LastAct {
    Other,
    Forward,
    Eat,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct HexWorld {
    catalog: SubstrateCatalog,
    width: i32,
    turn: u64,
    // Per-cell physics.
    occ_kind: Vec<OccKind>,
    org_at: Vec<Option<BodyHandle>>,
    food_energy: Vec<f32>,
    food_kind: Vec<FoodKind>,
    spike: Vec<bool>,
    spike_visual: Vec<VisualProperties>,
    visual_map: Vec<VisualProperties>,
    fertility: Vec<f32>,
    // Per-handle organism state.
    pos: Vec<Option<(i32, i32)>>,
    facing: Vec<FacingDirection>,
    color: Vec<RgbColor>,
    last_act: Vec<LastAct>,
    // Predation kills suppress the corpse (energy already transferred).
    suppress_corpse: Vec<BodyHandle>,
}

impl HexWorld {
    pub fn new(width: u32, seed: u64) -> Self {
        let width = width as i32;
        let cells = (width * width) as usize;
        let mut world = HexWorld {
            catalog: catalog::build_catalog(),
            width,
            turn: 0,
            occ_kind: vec![OccKind::Empty; cells],
            org_at: vec![None; cells],
            food_energy: vec![0.0; cells],
            food_kind: vec![FoodKind::Plant; cells],
            spike: vec![false; cells],
            spike_visual: vec![VisualProperties::default(); cells],
            visual_map: vec![VisualProperties::default(); cells],
            fertility: vec![0.0; cells],
            pos: Vec::new(),
            facing: Vec::new(),
            color: Vec::new(),
            last_act: Vec::new(),
            suppress_corpse: Vec::new(),
        };
        world.generate_terrain(seed);
        world
    }

    /// Deterministic value-noise terrain: walls where noise is high, spikes in a
    /// mid band, and a fertility field for food regrowth.
    fn generate_terrain(&mut self, seed: u64) {
        let w = self.width;
        for r in 0..w {
            for q in 0..w {
                let idx = (r * w + q) as usize;
                let n = value_noise(q, r, seed);
                self.fertility[idx] = value_noise(q, r, seed ^ 0xF00D).clamp(0.0, 1.0);
                if n > 0.82 {
                    self.occ_kind[idx] = OccKind::Wall;
                    self.visual_map[idx] = VisualProperties {
                        r: 0.3,
                        g: 0.3,
                        b: 0.3,
                        opacity: 1.0,
                        shape: 1.0,
                    };
                } else if n > 0.74 {
                    self.spike[idx] = true;
                    self.spike_visual[idx] = VisualProperties {
                        r: 0.6,
                        g: 0.1,
                        b: 0.1,
                        opacity: 0.7,
                        shape: 0.9,
                    };
                }
            }
        }
    }

    #[inline]
    fn idx(&self, q: i32, r: i32) -> usize {
        cell_index(q, r, self.width)
    }

    // Cell-kind queries for the render snapshot (crate-internal).
    pub(crate) fn is_food_cell(&self, idx: usize) -> bool {
        self.occ_kind[idx] == OccKind::Food
    }
    pub(crate) fn is_wall_cell(&self, idx: usize) -> bool {
        self.occ_kind[idx] == OccKind::Wall
    }
    pub(crate) fn is_spike_cell(&self, idx: usize) -> bool {
        self.spike[idx]
    }

    fn ensure_handle(&mut self, h: BodyHandle) {
        let need = h.0 as usize + 1;
        if self.pos.len() < need {
            self.pos.resize(need, None);
            self.facing.resize(need, FacingDirection::East);
            self.color.resize(need, RgbColor { r: 0.5, g: 0.5, b: 0.5 });
            self.last_act.resize(need, LastAct::Other);
        }
    }

    fn set_org(&mut self, cell: usize, h: BodyHandle, color: RgbColor) {
        self.occ_kind[cell] = OccKind::Org;
        self.org_at[cell] = Some(h);
        self.visual_map[cell] = organism_visual(color);
    }

    fn clear_cell(&mut self, cell: usize) {
        self.occ_kind[cell] = OccKind::Empty;
        self.org_at[cell] = None;
        self.food_energy[cell] = 0.0;
        self.visual_map[cell] = VisualProperties::default();
    }

    fn set_food(&mut self, cell: usize, energy: f32, kind: FoodKind) {
        self.occ_kind[cell] = OccKind::Food;
        self.food_energy[cell] = energy;
        self.food_kind[cell] = kind;
        self.visual_map[cell] = food_visual(kind);
    }

    fn vision_distance(morphology: &[f32]) -> u32 {
        morphology
            .get(catalog::M_VISION)
            .copied()
            .unwrap_or(4.0)
            .round()
            .clamp(1.0, 10.0) as u32
    }

    fn body_color(morphology: &[f32]) -> RgbColor {
        RgbColor {
            r: morphology.get(catalog::M_BODY_R).copied().unwrap_or(0.5),
            g: morphology.get(catalog::M_BODY_G).copied().unwrap_or(0.5),
            b: morphology.get(catalog::M_BODY_B).copied().unwrap_or(0.5),
        }
        .clamped()
    }

    /// Raycast one ray, accumulating translucent visual contributions. Ported
    /// from `sim-core/src/brain/sensing.rs::scan_ray`.
    fn scan_ray(
        &self,
        origin: (i32, i32),
        ray_facing: FacingDirection,
        self_h: BodyHandle,
        max_dist: u32,
    ) -> (ColorSignal, bool) {
        let inv_max = 1.0 / max_dist.max(1) as f32;
        let (dq, dr) = grid::facing_delta(ray_facing);
        let (mut q, mut r) = origin;
        let mut color = ColorSignal::default();
        let mut food_visible = false;
        let mut remaining = 1.0f32;
        for distance in 1..=max_dist.max(1) {
            q = grid::wrap_coord(q + dq, self.width);
            r = grid::wrap_coord(r + dr, self.width);
            let idx = (r * self.width + q) as usize;
            let signal = (max_dist.max(1) - distance + 1) as f32 * inv_max;
            if self.spike[idx] {
                accumulate(&mut color, &mut remaining, self.spike_visual[idx], signal);
            }
            match self.occ_kind[idx] {
                OccKind::Org if self.org_at[idx] == Some(self_h) => {}
                OccKind::Food => {
                    food_visible |= remaining > 0.0;
                    accumulate(&mut color, &mut remaining, self.visual_map[idx], signal);
                }
                OccKind::Org | OccKind::Wall => {
                    accumulate(&mut color, &mut remaining, self.visual_map[idx], signal);
                }
                OccKind::Empty => {}
            }
            if remaining <= f32::EPSILON {
                break;
            }
        }
        (color.clamped(), food_visible)
    }
}

impl Environment for HexWorld {
    type Intents = HexIntent;
    type SpawnSite = SpawnSite;

    fn catalog(&self) -> &SubstrateCatalog {
        &self.catalog
    }

    fn derive_body_params(&self, morphology: &[f32]) -> DerivedBodyParams {
        let size = morphology.get(catalog::M_SIZE).copied().unwrap_or(200.0).max(1.0);
        DerivedBodyParams {
            size,
            max_health: size,
            investment_energy: size * 0.2,
            metabolic_base: 0.0,
        }
    }

    fn observe(&self, view: &BodyView, layout: &ObsLayout, out: &mut [f32]) {
        let Some(pos) = self.pos.get(view.handle.0 as usize).copied().flatten() else {
            return;
        };
        let facing = self.facing[view.handle.0 as usize];
        let max_health = view.derived.max_health.max(1.0);
        let vision = Self::vision_distance(view.morphology);

        let rays: [(ColorSignal, bool); 3] = std::array::from_fn(|i| {
            let rf = grid::rotate_by_steps(facing, catalog::RAY_OFFSETS[i]);
            self.scan_ray(pos, rf, view.handle, vision)
        });
        let contact = {
            let ahead = hex_neighbor(pos, facing, self.width);
            let ai = self.idx(ahead.0, ahead.1);
            f32::from(self.spike[ai] || self.occ_kind[ai] != OccKind::Empty)
        };
        let energy_signal = {
            let e = view.energy.max(0.0);
            e / (e + max_health)
        };
        let health_signal = (view.health / max_health).clamp(0.0, 1.0);
        let energy_delta = ((view.energy - view.energy_at_last_sensing) / max_health).tanh();
        let last_forward = f32::from(self.last_act[view.handle.0 as usize] == LastAct::Forward);
        let last_eat = f32::from(self.last_act[view.handle.0 as usize] == LastAct::Eat);

        for (slot, &si) in layout.sensor_indices.iter().enumerate() {
            let off = layout.offsets[slot];
            let key = self.catalog.sensors[si].key.as_str();
            let value = sensor_value(
                key, &rays, contact, energy_signal, health_signal, energy_delta, last_forward,
                last_eat,
            );
            if let Some(o) = out.get_mut(off) {
                *o = value;
            }
        }
    }

    fn metabolic_cost(&self, view: &BodyView) -> f32 {
        // Base cost = neural complexity + vision + Kleiber body-mass term, with
        // the homeostatic low-energy downregulation. Ported from
        // `sim-core/src/metabolism.rs`.
        let neurons = view.brain_neurons as f32;
        let vision = Self::vision_distance(view.morphology) as f32 / 3.0;
        let body_mass = BODY_MASS_METABOLIC_COST_COEFF
            * view.derived.max_health.max(0.0).powf(BODY_MASS_METABOLIC_EXPONENT);
        let base = neurons + vision + body_mass;
        const HOMEOSTATIC_THRESHOLD: f32 = 5.0;
        let factor = if view.energy >= HOMEOSTATIC_THRESHOLD {
            1.0
        } else {
            0.5 + 0.5 * (view.energy.max(0.0) / HOMEOSTATIC_THRESHOLD)
        };
        PASSIVE_METABOLISM_COST_PER_UNIT * base * factor
    }

    fn decode_intents(&self, view: &BodyView, action: &ActionOutput) -> HexIntent {
        let kind = match action.selected_actuator().map(|i| self.catalog.actuators[i].key.as_str()) {
            Some("turn_left") => IntentKind::TurnLeft,
            Some("turn_right") => IntentKind::TurnRight,
            Some("forward") => IntentKind::Forward,
            Some("eat") => IntentKind::Eat,
            Some("attack") => IntentKind::Attack,
            _ => IntentKind::Idle, // mate handled by mate_intent
        };
        HexIntent {
            handle: view.handle,
            kind,
            confidence: action.confidence,
        }
    }

    fn resolve_actions(
        &mut self,
        intents: &[HexIntent],
        pop: &dyn PopulationRead,
        rng: &mut Rng,
        sink: &mut EffectSink,
    ) {
        self.suppress_corpse.clear();
        // 1. Turns + record last action; gather move requests.
        let mut moves: Vec<(BodyHandle, (i32, i32), f32)> = Vec::new();
        for intent in intents {
            let h = intent.handle;
            let Some(pos) = self.pos.get(h.0 as usize).copied().flatten() else {
                continue;
            };
            self.last_act[h.0 as usize] = LastAct::Other;
            match intent.kind {
                IntentKind::TurnLeft => {
                    self.facing[h.0 as usize] = rotate_left(self.facing[h.0 as usize])
                }
                IntentKind::TurnRight => {
                    self.facing[h.0 as usize] = rotate_right(self.facing[h.0 as usize])
                }
                IntentKind::Forward => {
                    let target = hex_neighbor(pos, self.facing[h.0 as usize], self.width);
                    moves.push((h, target, intent.confidence));
                }
                IntentKind::Eat => {
                    let ahead = hex_neighbor(pos, self.facing[h.0 as usize], self.width);
                    let ci = self.idx(ahead.0, ahead.1);
                    if self.occ_kind[ci] == OccKind::Food {
                        let gain = self.food_energy[ci];
                        self.clear_cell(ci);
                        sink.add_energy(h, gain);
                        self.last_act[h.0 as usize] = LastAct::Eat;
                    }
                }
                IntentKind::Attack => {
                    let ahead = hex_neighbor(pos, self.facing[h.0 as usize], self.width);
                    let ci = self.idx(ahead.0, ahead.1);
                    if let Some(prey) = self.org_at[ci] {
                        self.resolve_attack(h, prey, pop, rng, sink);
                    }
                }
                IntentKind::Idle => {}
            }
        }
        // 2. Deterministic move resolution: (target cell, confidence desc, handle asc).
        moves.sort_by(|a, b| {
            self.idx(a.1 .0, a.1 .1)
                .cmp(&self.idx(b.1 .0, b.1 .1))
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.0 .0.cmp(&b.0 .0))
        });
        let mut claimed = vec![false; self.occ_kind.len()];
        for (h, target, _conf) in moves {
            let ti = self.idx(target.0, target.1);
            if self.occ_kind[ti] != OccKind::Empty || claimed[ti] {
                continue;
            }
            let Some(from) = self.pos[h.0 as usize] else {
                continue;
            };
            claimed[ti] = true;
            let fi = self.idx(from.0, from.1);
            self.clear_cell(fi);
            self.set_org(ti, h, self.color[h.0 as usize]);
            self.pos[h.0 as usize] = Some(target);
            self.last_act[h.0 as usize] = LastAct::Forward;
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
        let pos = self.pos.get(view.handle.0 as usize).copied().flatten()?;
        let facing = self.facing[view.handle.0 as usize];
        let ahead = hex_neighbor(pos, facing, self.width);
        let ci = self.idx(ahead.0, ahead.1);
        self.org_at[ci].map(|target| MateIntent {
            target,
            confidence: action.confidence,
        })
    }

    fn step_world(&mut self, pop: &dyn PopulationRead, rng: &mut Rng, sink: &mut EffectSink) {
        self.turn += 1;
        // Spikes + health regen over all living organisms.
        for h in 0..self.pos.len() {
            let handle = BodyHandle(h as u32);
            let Some(pos) = self.pos[h] else { continue };
            if !pop.is_alive(handle) {
                continue;
            }
            let Some(view) = pop.view(handle) else { continue };
            let ci = self.idx(pos.0, pos.1);
            if self.spike[ci] {
                let dmg = (view.derived.max_health * SPIKE_DAMAGE_FRACTION).min(view.health);
                if dmg >= view.health {
                    sink.kill(handle);
                } else {
                    sink.add_health(handle, -dmg);
                }
            }
            sink.add_health(handle, view.derived.max_health * HEALTH_REGEN_FRACTION);
        }
        // Zero-sum social-color energy transfer (ported, simplified).
        self.social_color(pop, sink);
        // Food regrowth on fertile empty cells.
        if self.turn.is_multiple_of(FOOD_REGROW_INTERVAL) {
            let cells = self.occ_kind.len();
            let spawn = (cells / 40).max(1);
            for _ in 0..spawn {
                let ci = rng.random_range(0..cells);
                if self.occ_kind[ci] == OccKind::Empty
                    && !self.spike[ci]
                    && rng.random::<f32>() < self.fertility[ci]
                {
                    self.set_food(ci, PLANT_ENERGY, FoodKind::Plant);
                }
            }
        }
    }

    fn place_birth(&mut self, carrier: &BodyView, rng: &mut Rng) -> Option<SpawnSite> {
        let pos = self.pos.get(carrier.handle.0 as usize).copied().flatten()?;
        let parent_facing = self.facing[carrier.handle.0 as usize];
        let mut candidates: Vec<FacingDirection> = vec![opposite_direction(parent_facing)];
        candidates.extend(ALL_FACINGS);
        for f in candidates {
            let cell = hex_neighbor(pos, f, self.width);
            let ci = self.idx(cell.0, cell.1);
            if self.occ_kind[ci] == OccKind::Empty && !self.spike[ci] {
                let _ = rng;
                return Some(SpawnSite {
                    q: cell.0,
                    r: cell.1,
                    facing: opposite_direction(parent_facing),
                });
            }
        }
        None
    }

    fn place_founder(&mut self, rng: &mut Rng) -> Option<SpawnSite> {
        for _ in 0..64 {
            let q = rng.random_range(0..self.width);
            let r = rng.random_range(0..self.width);
            let ci = self.idx(q, r);
            if self.occ_kind[ci] == OccKind::Empty && !self.spike[ci] {
                let facing = ALL_FACINGS[rng.random_range(0..6)];
                return Some(SpawnSite { q, r, facing });
            }
        }
        None
    }

    fn attach(&mut self, view: &BodyView, site: SpawnSite) {
        self.ensure_handle(view.handle);
        let color = Self::body_color(view.morphology);
        self.pos[view.handle.0 as usize] = Some((site.q, site.r));
        self.facing[view.handle.0 as usize] = site.facing;
        self.color[view.handle.0 as usize] = color;
        self.last_act[view.handle.0 as usize] = LastAct::Other;
        let ci = self.idx(site.q, site.r);
        self.set_org(ci, view.handle, color);
    }

    fn on_deaths(&mut self, dead: &[BodyHandle], pop: &dyn PopulationRead, _sink: &mut EffectSink) {
        for h in dead {
            let Some(pos) = self.pos.get(h.0 as usize).copied().flatten() else {
                continue;
            };
            let ci = self.idx(pos.0, pos.1);
            self.clear_cell(ci);
            self.pos[h.0 as usize] = None;
            if self.suppress_corpse.contains(h) {
                continue;
            }
            if let Some(view) = pop.view(*h) {
                let energy = (view.energy.max(0.0)) * CORPSE_ENERGY_RETENTION;
                if energy > 0.0 && self.occ_kind[ci] == OccKind::Empty {
                    self.set_food(ci, energy, FoodKind::Corpse);
                }
            }
        }
        self.suppress_corpse.clear();
    }
}

impl HexWorld {
    fn resolve_attack(
        &mut self,
        attacker: BodyHandle,
        prey: BodyHandle,
        pop: &dyn PopulationRead,
        rng: &mut Rng,
        sink: &mut EffectSink,
    ) {
        let (Some(att), Some(pv)) = (pop.view(attacker), pop.view(prey)) else {
            return;
        };
        if !pop.is_alive(prey) {
            return;
        }
        let predation_success = (att.derived.size / pv.derived.size.max(0.001)).clamp(0.0, 1.0);
        if rng.random::<f32>() >= predation_success {
            return;
        }
        let damage = att.derived.max_health * ATTACK_DAMAGE_FRACTION;
        if damage >= pv.health {
            // Lethal: attacker gets the prey's retained energy; corpse suppressed.
            let gained = pv.energy.max(0.0) * CORPSE_ENERGY_RETENTION;
            sink.add_energy(attacker, gained);
            sink.kill(prey);
            self.suppress_corpse.push(prey);
        } else {
            sink.add_health(prey, -damage);
        }
    }

    fn social_color(&self, pop: &dyn PopulationRead, sink: &mut EffectSink) {
        for h in 0..self.pos.len() {
            let handle = BodyHandle(h as u32);
            let Some(pos) = self.pos[h] else { continue };
            if !pop.is_alive(handle) {
                continue;
            }
            let self_hue = color_hue(self.color[h]);
            let mut flow = 0.0f32;
            for f in ALL_FACINGS {
                let n = hex_neighbor(pos, f, self.width);
                let ni = self.idx(n.0, n.1);
                if let Some(nh) = self.org_at[ni] {
                    let neighbor_hue = color_hue(self.color[nh.0 as usize]);
                    flow += (self_hue - neighbor_hue).sin();
                }
            }
            if flow != 0.0 {
                sink.add_energy(handle, SOCIAL_DAMAGE * flow);
            }
        }
    }
}

pub struct SpawnSite {
    pub q: i32,
    pub r: i32,
    pub facing: FacingDirection,
}

#[derive(Clone, Copy, PartialEq)]
enum IntentKind {
    Idle,
    TurnLeft,
    TurnRight,
    Forward,
    Eat,
    Attack,
}

pub struct HexIntent {
    handle: BodyHandle,
    kind: IntentKind,
    confidence: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ColorSignal {
    red: f32,
    green: f32,
    blue: f32,
    shape: f32,
}

impl ColorSignal {
    fn clamped(self) -> Self {
        ColorSignal {
            red: self.red.clamp(0.0, 1.0),
            green: self.green.clamp(0.0, 1.0),
            blue: self.blue.clamp(0.0, 1.0),
            shape: self.shape.clamp(0.0, 1.0),
        }
    }
    fn channel(self, ch: &str) -> f32 {
        match ch {
            "r" => self.red,
            "g" => self.green,
            "b" => self.blue,
            _ => self.shape,
        }
    }
}

fn accumulate(color: &mut ColorSignal, remaining: &mut f32, visual: VisualProperties, signal: f32) {
    if *remaining <= 0.0 {
        return;
    }
    let opacity = visual.opacity.clamp(0.0, 1.0);
    let contribution = opacity * signal * *remaining;
    color.red += visual.r * contribution;
    color.green += visual.g * contribution;
    color.blue += visual.b * contribution;
    color.shape += visual.shape * contribution;
    *remaining *= 1.0 - opacity;
}

#[allow(clippy::too_many_arguments)]
fn sensor_value(
    key: &str,
    rays: &[(ColorSignal, bool); 3],
    contact: f32,
    energy: f32,
    health: f32,
    energy_delta: f32,
    last_forward: f32,
    last_eat: f32,
) -> f32 {
    if let Some(rest) = key.strip_prefix("vision.") {
        let mut parts = rest.split('.');
        let off: i8 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let ch = parts.next().unwrap_or("r");
        let ray_idx = catalog::RAY_OFFSETS.iter().position(|o| *o == off).unwrap_or(1);
        return rays[ray_idx].0.channel(ch);
    }
    match key {
        "contact_ahead" => contact,
        "intero.energy" => energy,
        "intero.health" => health,
        "intero.energy_delta" => energy_delta,
        "intero.last_forward" => last_forward,
        "intero.last_eat" => last_eat,
        _ => 0.0,
    }
}

/// Cheap deterministic value noise in [0,1] from integer coords + seed.
fn value_noise(q: i32, r: i32, seed: u64) -> f32 {
    let h = sim_substrate::rng::mix_u64(
        seed ^ (q as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (r as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F),
    );
    (h >> 40) as f32 / ((1u64 << 24) - 1) as f32
}
