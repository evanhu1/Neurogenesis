use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sim_types::{
    FoodState, MetricsSnapshot, OccupancyCell, Occupant, OrganismId, OrganismState, TickDelta,
    WorldConfig, WorldSnapshot,
};
use thiserror::Error;

mod brain;
pub(crate) mod genome;
mod grid;
mod plasticity;
#[cfg(feature = "profiling")]
#[path = "../profiling/profiling.rs"]
pub mod profiling;
mod spawn;
mod turn;

pub use brain::derive_active_action_neuron_id;

#[cfg(test)]
mod tests;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
    #[error("invalid simulation state: {0}")]
    InvalidState(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Simulation {
    config: WorldConfig,
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    next_organism_id: u64,
    next_food_id: u64,
    organisms: Vec<OrganismState>,
    #[serde(default)]
    pending_actions: Vec<PendingActionState>,
    foods: Vec<FoodState>,
    occupancy: Vec<Option<Occupant>>,
    #[serde(default)]
    terrain_map: Vec<bool>,
    #[serde(default)]
    food_fertility: Vec<u16>,
    #[serde(default)]
    biomass: Vec<f32>,
    metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) enum PendingActionKind {
    #[default]
    None,
    Consume,
    Reproduce,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct PendingActionState {
    pub(crate) kind: PendingActionKind,
    pub(crate) turns_remaining: u8,
}

impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        validate_world_config(&config)?;
        genome::validate_seed_genome_config(&config.seed_genome_config)?;

        let capacity = grid::world_capacity(config.world_width);
        let mut sim = Self {
            config,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_organism_id: 0,
            next_food_id: 0,
            organisms: Vec::new(),
            pending_actions: Vec::new(),
            foods: Vec::new(),
            occupancy: vec![None; capacity],
            terrain_map: Vec::new(),
            food_fertility: Vec::new(),
            biomass: Vec::new(),
            metrics: MetricsSnapshot::default(),
        };

        sim.initialize_terrain();
        sim.spawn_initial_population();
        sim.initialize_food_ecology();
        sim.seed_initial_food_supply();
        sim.refresh_population_metrics();
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let mut organisms = self.organisms.clone();
        organisms.sort_by_key(|o| o.id);
        let mut foods = self.foods.clone();
        foods.sort_by_key(|food| food.id);

        let width = self.config.world_width as usize;
        let mut occupancy = Vec::with_capacity(self.occupancy.iter().flatten().count());
        for (idx, maybe_entity) in self.occupancy.iter().enumerate() {
            if let Some(occupant) = *maybe_entity {
                let q = (idx % width) as i32;
                let r = (idx / width) as i32;
                occupancy.push(OccupancyCell { q, r, occupant });
            }
        }
        occupancy.sort_by_key(|cell| (cell.q, cell.r));

        WorldSnapshot {
            turn: self.turn,
            rng_seed: self.seed,
            config: self.config.clone(),
            organisms,
            foods,
            occupancy,
            metrics: self.metrics.clone(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.seed = seed.unwrap_or(self.seed);
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.turn = 0;
        self.next_organism_id = 0;
        self.next_food_id = 0;
        self.organisms.clear();
        self.pending_actions.clear();
        self.foods.clear();
        self.occupancy.fill(None);
        self.terrain_map.clear();
        self.food_fertility.clear();
        self.biomass.clear();
        self.metrics = MetricsSnapshot::default();
        self.initialize_terrain();
        self.spawn_initial_population();
        self.initialize_food_ecology();
        self.seed_initial_food_supply();
        self.refresh_population_metrics();
    }

    pub fn step_n(&mut self, count: u32) -> Vec<TickDelta> {
        let mut deltas = Vec::with_capacity(count as usize);
        for _ in 0..count {
            deltas.push(self.tick());
        }
        deltas
    }

    pub fn advance_n(&mut self, count: u32) {
        for _ in 0..count {
            let _ = self.tick();
        }
    }

    pub fn focused_organism(&self, id: OrganismId) -> Option<OrganismState> {
        self.organisms
            .iter()
            .find(|organism| organism.id == id)
            .cloned()
    }

    pub fn metrics(&self) -> &MetricsSnapshot {
        &self.metrics
    }

    pub fn validate_state(&self) -> Result<(), SimError> {
        validate_world_config(&self.config)?;
        genome::validate_seed_genome_config(&self.config.seed_genome_config)?;

        let expected_capacity = grid::world_capacity(self.config.world_width);
        if self.occupancy.len() != expected_capacity {
            return Err(SimError::InvalidState(format!(
                "occupancy length {} does not match expected capacity {}",
                self.occupancy.len(),
                expected_capacity
            )));
        }
        if self.terrain_map.len() != expected_capacity {
            return Err(SimError::InvalidState(format!(
                "terrain_map length {} does not match expected capacity {}",
                self.terrain_map.len(),
                expected_capacity
            )));
        }
        if self.biomass.len() != expected_capacity {
            return Err(SimError::InvalidState(format!(
                "biomass length {} does not match expected capacity {}",
                self.biomass.len(),
                expected_capacity
            )));
        }

        if !self
            .organisms
            .windows(2)
            .all(|window| window[0].id < window[1].id)
        {
            return Err(SimError::InvalidState(
                "organisms must be sorted by ascending id".to_owned(),
            ));
        }

        if self.pending_actions.len() != self.organisms.len() {
            return Err(SimError::InvalidState(format!(
                "pending_actions length {} does not match organism count {}",
                self.pending_actions.len(),
                self.organisms.len()
            )));
        }

        if !self
            .foods
            .windows(2)
            .all(|window| window[0].id < window[1].id)
        {
            return Err(SimError::InvalidState(
                "foods must be sorted by ascending id".to_owned(),
            ));
        }

        let width = self.config.world_width as i32;
        let mut expected_occupancy = vec![None; expected_capacity];
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                expected_occupancy[idx] = Some(Occupant::Wall);
            }
        }

        for organism in &self.organisms {
            if organism.q < 0 || organism.r < 0 || organism.q >= width || organism.r >= width {
                return Err(SimError::InvalidState(format!(
                    "organism {:?} uses non-canonical coordinates ({}, {})",
                    organism.id, organism.q, organism.r
                )));
            }
            let idx = organism.r as usize * width as usize + organism.q as usize;
            if expected_occupancy[idx].is_some() {
                return Err(SimError::InvalidState(format!(
                    "duplicate occupancy at ({}, {})",
                    organism.q, organism.r
                )));
            }
            expected_occupancy[idx] = Some(Occupant::Organism(organism.id));
        }

        for food in &self.foods {
            if food.q < 0 || food.r < 0 || food.q >= width || food.r >= width {
                return Err(SimError::InvalidState(format!(
                    "food {:?} uses non-canonical coordinates ({}, {})",
                    food.id, food.q, food.r
                )));
            }
            let idx = food.r as usize * width as usize + food.q as usize;
            if expected_occupancy[idx].is_some() {
                return Err(SimError::InvalidState(format!(
                    "duplicate occupancy at ({}, {})",
                    food.q, food.r
                )));
            }
            expected_occupancy[idx] = Some(Occupant::Food(food.id));
        }

        if self.occupancy != expected_occupancy {
            return Err(SimError::InvalidState(
                "occupancy vector does not match organism/food positions".to_owned(),
            ));
        }

        let max_organism_id = self.organisms.iter().map(|o| o.id.0).max().unwrap_or(0);
        if !self.organisms.is_empty() && self.next_organism_id <= max_organism_id {
            return Err(SimError::InvalidState(format!(
                "next_organism_id {} must be greater than max organism id {}",
                self.next_organism_id, max_organism_id
            )));
        }

        let max_food_id = self.foods.iter().map(|f| f.id.0).max().unwrap_or(0);
        if !self.foods.is_empty() && self.next_food_id <= max_food_id {
            return Err(SimError::InvalidState(format!(
                "next_food_id {} must be greater than max food id {}",
                self.next_food_id, max_food_id
            )));
        }

        if self.metrics.organisms != self.organisms.len() as u32 {
            return Err(SimError::InvalidState(format!(
                "metrics.organisms {} does not match organism count {}",
                self.metrics.organisms,
                self.organisms.len()
            )));
        }

        Ok(())
    }
}

fn validate_world_config(config: &WorldConfig) -> Result<(), SimError> {
    if config.world_width == 0 {
        return Err(SimError::InvalidConfig(
            "world_width must be greater than zero".to_owned(),
        ));
    }
    if config.num_organisms == 0 {
        return Err(SimError::InvalidConfig(
            "num_organisms must be greater than zero".to_owned(),
        ));
    }
    if config.food_energy <= 0.0 {
        return Err(SimError::InvalidConfig(
            "food_energy must be greater than zero".to_owned(),
        ));
    }
    if config.move_action_energy_cost < 0.0 {
        return Err(SimError::InvalidConfig(
            "move_action_energy_cost must be >= 0".to_owned(),
        ));
    }
    if !config.action_temperature.is_finite() || config.action_temperature <= 0.0 {
        return Err(SimError::InvalidConfig(
            "action_temperature must be finite and greater than zero".to_owned(),
        ));
    }
    if !config.plant_growth_speed.is_finite() || config.plant_growth_speed <= 0.0 {
        return Err(SimError::InvalidConfig(
            "plant_growth_speed must be greater than zero".to_owned(),
        ));
    }
    if config.food_fertility_noise_scale <= 0.0 {
        return Err(SimError::InvalidConfig(
            "food_fertility_noise_scale must be greater than zero".to_owned(),
        ));
    }
    if config.food_fertility_exponent <= 0.0 {
        return Err(SimError::InvalidConfig(
            "food_fertility_exponent must be greater than zero".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.food_fertility_floor) {
        return Err(SimError::InvalidConfig(
            "food_fertility_floor must be in [0.0, 1.0]".to_owned(),
        ));
    }
    if config.terrain_noise_scale <= 0.0 {
        return Err(SimError::InvalidConfig(
            "terrain_noise_scale must be greater than zero".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.terrain_threshold) {
        return Err(SimError::InvalidConfig(
            "terrain_threshold must be in [0.0, 1.0]".to_owned(),
        ));
    }
    if !config.global_mutation_rate_modifier.is_finite()
        || config.global_mutation_rate_modifier < 0.0
    {
        return Err(SimError::InvalidConfig(
            "global_mutation_rate_modifier must be finite and >= 0".to_owned(),
        ));
    }
    Ok(())
}
