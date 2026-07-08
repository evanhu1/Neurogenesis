//! `HexSim` — the concrete, serializable hex world the tooling loads, ticks,
//! and saves (replacing `sim_core::Simulation`). It owns the substrate
//! `PopulationDriver` and the `HexWorld` physics. The world is **extinct** when
//! no organism is alive; ticking a live world past extinction is a no-op and
//! run loops stop there (there is no periodic injection — a dead world stays
//! dead).

use crate::HexWorld;
use serde::{Deserialize, Serialize};
use sim_substrate::seed::seed_genome;
use sim_substrate::{Body, DevelopConfig, DriverConfig, Environment, Genome, PopulationDriver};

/// World construction + run parameters (the hex-specific config).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HexConfig {
    pub world_width: u32,
    pub num_founders: usize,
    pub founder_energy: f32,
    pub global_mutation_rate_modifier: f32,
    pub meta_mutation_enabled: bool,
    pub action_temperature: f32,
}

impl Default for HexConfig {
    fn default() -> Self {
        HexConfig {
            world_width: 32,
            num_founders: 200,
            founder_energy: 400.0,
            global_mutation_rate_modifier: 1.0,
            meta_mutation_enabled: true,
            action_temperature: 1.0,
        }
    }
}

impl HexConfig {
    fn driver_config(&self) -> DriverConfig {
        DriverConfig {
            global_mutation_rate_modifier: self.global_mutation_rate_modifier,
            meta_mutation_enabled: self.meta_mutation_enabled,
            action_temperature: self.action_temperature,
            founder_energy: self.founder_energy,
            develop: DevelopConfig::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HexSim {
    pub driver: PopulationDriver,
    pub world: HexWorld,
    pub config: HexConfig,
    pub seed: u64,
    /// Turn at which the world went extinct (0 organisms), if it has.
    pub extinct_at: Option<u64>,
}

impl HexSim {
    /// A fresh world seeded from the primordial seed genome.
    pub fn new(config: HexConfig, seed: u64) -> Self {
        let genome = seed_genome(&crate::catalog::build_catalog());
        Self::new_from_pool(config, seed, std::slice::from_ref(&genome))
    }

    /// A fresh world whose founders are drawn cyclically from `pool` (e.g. a
    /// champion pool). Falls back to the seed genome when the pool is empty.
    pub fn new_from_pool(config: HexConfig, seed: u64, pool: &[Genome]) -> Self {
        let mut world = HexWorld::new(config.world_width, seed);
        let fallback = seed_genome(world.catalog());
        let effective_pool: Vec<Genome> = if pool.is_empty() {
            vec![fallback.clone()]
        } else {
            pool.to_vec()
        };
        let baseline = effective_pool[0].header.mutation_rates;
        let mut driver = PopulationDriver::new(seed, baseline, config.driver_config());
        driver.seed_population_from(&mut world, config.num_founders, &effective_pool);
        HexSim {
            driver,
            world,
            config,
            seed,
            extinct_at: None,
        }
    }

    /// Advance one turn. Returns `true` while the world is still alive; once
    /// extinct it records the turn and returns `false` (and further ticks are
    /// no-ops).
    pub fn tick(&mut self) -> bool {
        if self.extinct_at.is_some() {
            return false;
        }
        self.driver.tick(&mut self.world);
        if self.driver.alive_count() == 0 {
            self.extinct_at = Some(self.driver.turn);
            return false;
        }
        true
    }

    pub fn turn(&self) -> u64 {
        self.driver.turn
    }

    pub fn alive_count(&self) -> usize {
        self.driver.alive_count()
    }

    pub fn is_extinct(&self) -> bool {
        self.extinct_at.is_some() || self.driver.alive_count() == 0
    }

    /// All living bodies (dead ones are skipped).
    pub fn living_bodies(&self) -> impl Iterator<Item = &Body> {
        self.driver.bodies.iter().filter(|b| b.alive)
    }

    pub fn catalog(&self) -> &sim_substrate::SubstrateCatalog {
        self.world.catalog()
    }

    /// Find a living body by organism id, with its handle.
    pub fn body_by_id(&self, id: u64) -> Option<(sim_substrate::BodyHandle, &Body)> {
        self.driver
            .bodies
            .iter()
            .enumerate()
            .find(|(_, b)| b.id == id && b.alive)
            .map(|(i, b)| (sim_substrate::BodyHandle(i as u32), b))
    }

    /// Run the environment's sensing for one body, returning its observation
    /// vector — used by read commands (`decide`) to show what the brain sees.
    pub fn observe_body(&self, handle: sim_substrate::BodyHandle) -> Vec<f32> {
        let body = &self.driver.bodies[handle.0 as usize];
        let mut obs = vec![0.0; body.obs_layout.len];
        let view = body.view(handle);
        self.world.observe(&view, &body.obs_layout, &mut obs);
        obs
    }

    /// A lightweight render snapshot for the server/web client: living organism
    /// positions + colors, food, and terrain. Reads the world's private cell
    /// state (same-crate access).
    pub fn render_snapshot(&self) -> RenderSnapshot {
        let width = self.world.width;
        let mut organisms = Vec::new();
        for body in self.living_bodies() {
            if let Some(handle) = body_handle_of(&self.driver.bodies, body.id) {
                if let Some((q, r)) = self.world.pos.get(handle).copied().flatten() {
                    let c = self.world.color[handle];
                    organisms.push(RenderOrganism {
                        id: body.id,
                        q,
                        r,
                        energy: body.energy,
                        health: body.health,
                        color: [c.r, c.g, c.b],
                    });
                }
            }
        }
        let mut food = Vec::new();
        let mut walls = Vec::new();
        let mut spikes = Vec::new();
        for r in 0..width {
            for q in 0..width {
                let idx = (r * width + q) as usize;
                if self.world.is_food_cell(idx) {
                    food.push([q, r]);
                } else if self.world.is_wall_cell(idx) {
                    walls.push([q, r]);
                }
                if self.world.is_spike_cell(idx) {
                    spikes.push([q, r]);
                }
            }
        }
        RenderSnapshot {
            turn: self.turn(),
            width: width as u32,
            organisms,
            food,
            walls,
            spikes,
            extinct: self.is_extinct(),
        }
    }

    /// A normalized behavior descriptor for the QD archive: [brain complexity,
    /// generational depth, energy fraction]. Instantaneous proxies (no lifetime
    /// tracking) — enough to spread champions across niches.
    pub fn behavior_descriptor(&self, body: &Body) -> sim_substrate::BehaviorDescriptor {
        let brain_complexity = (body.brain.edges.len() as f32 / 64.0).clamp(0.0, 1.0);
        let depth = (body.generation as f32 / 50.0).clamp(0.0, 1.0);
        let energy_frac = (body.energy / (body.derived.max_health.max(1.0))).clamp(0.0, 1.0);
        sim_substrate::BehaviorDescriptor::new(vec![brain_complexity, depth, energy_frac])
    }

    /// Aggregate population statistics for the `state` command.
    pub fn population_stats(&self) -> PopulationStats {
        let mut alive = 0usize;
        let (mut energy, mut neurons, mut edges, mut generation) = (0.0f64, 0u64, 0u64, 0u64);
        let mut max_generation = 0u64;
        for b in self.living_bodies() {
            alive += 1;
            energy += b.energy as f64;
            neurons += b.brain.neurons.len() as u64;
            edges += b.brain.edges.len() as u64;
            generation += b.generation;
            max_generation = max_generation.max(b.generation);
        }
        let n = alive.max(1) as f64;
        PopulationStats {
            turn: self.turn(),
            alive,
            total_ever: self.driver.total_ever(),
            extinct_at: self.extinct_at,
            mean_energy: (energy / n) as f32,
            mean_neurons: (neurons as f64 / n) as f32,
            mean_edges: (edges as f64 / n) as f32,
            mean_generation: (generation as f64 / n) as f32,
            max_generation,
        }
    }
}

fn body_handle_of(bodies: &[Body], id: u64) -> Option<usize> {
    bodies.iter().position(|b| b.id == id && b.alive)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderOrganism {
    pub id: u64,
    pub q: i32,
    pub r: i32,
    pub energy: f32,
    pub health: f32,
    pub color: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderSnapshot {
    pub turn: u64,
    pub width: u32,
    pub organisms: Vec<RenderOrganism>,
    pub food: Vec<[i32; 2]>,
    pub walls: Vec<[i32; 2]>,
    pub spikes: Vec<[i32; 2]>,
    pub extinct: bool,
}

/// Aggregate population statistics surfaced by `sim-cli state`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PopulationStats {
    pub turn: u64,
    pub alive: usize,
    pub total_ever: u64,
    pub extinct_at: Option<u64>,
    pub mean_energy: f32,
    pub mean_neurons: f32,
    pub mean_edges: f32,
    pub mean_generation: f32,
    pub max_generation: u64,
}
