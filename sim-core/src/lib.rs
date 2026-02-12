use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sim_types::{
    FoodId, FoodState, MetricsSnapshot, OccupancyCell, Occupant, OrganismGenome, OrganismId,
    OrganismState, SpeciesId, TickDelta, WorldConfig, WorldSnapshot,
};
use std::collections::BTreeMap;
use thiserror::Error;

mod brain;
pub(crate) mod genome;
mod grid;
mod spawn;
mod turn;

pub use brain::derive_active_neuron_ids;

#[cfg(test)]
mod tests;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct Simulation {
    config: WorldConfig,
    species_registry: BTreeMap<SpeciesId, OrganismGenome>,
    next_species_id: u32,
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    next_organism_id: u64,
    next_food_id: u64,
    organisms: Vec<OrganismState>,
    foods: Vec<FoodState>,
    occupancy: Vec<Option<Occupant>>,
    metrics: MetricsSnapshot,
}

impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        validate_world_config(&config)?;
        genome::validate_seed_genome_config(&config.seed_genome_config)?;

        let capacity = grid::world_capacity(config.world_width);
        let mut sim = Self {
            config,
            species_registry: BTreeMap::new(),
            next_species_id: 0,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_organism_id: 0,
            next_food_id: 0,
            organisms: Vec::new(),
            foods: Vec::new(),
            occupancy: vec![None; capacity],
            metrics: MetricsSnapshot::default(),
        };

        sim.spawn_initial_population();
        sim.replenish_food_supply();
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
        let mut occupancy = Vec::with_capacity(self.organisms.len() + self.foods.len());
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
            species_registry: self.species_registry.clone(),
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
        self.species_registry.clear();
        self.next_species_id = 0;
        self.next_organism_id = 0;
        self.next_food_id = 0;
        self.organisms.clear();
        self.foods.clear();
        self.occupancy.fill(None);
        self.metrics = MetricsSnapshot::default();
        self.spawn_initial_population();
        self.replenish_food_supply();
        self.refresh_population_metrics();
    }

    pub fn step_n(&mut self, count: u32) -> Vec<TickDelta> {
        let mut deltas = Vec::with_capacity(count as usize);
        for _ in 0..count {
            deltas.push(self.tick());
        }
        deltas
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

    pub(crate) fn organism_index(&self, id: OrganismId) -> usize {
        self.organisms
            .binary_search_by_key(&id, |o| o.id)
            .expect("organism must exist")
    }

    pub(crate) fn food_index(&self, id: FoodId) -> usize {
        self.foods
            .binary_search_by_key(&id, |f| f.id)
            .expect("food must exist")
    }

    pub(crate) fn alloc_species_id(&mut self) -> SpeciesId {
        let id = SpeciesId(self.next_species_id);
        self.next_species_id = self.next_species_id.saturating_add(1);
        id
    }

    pub(crate) fn prune_extinct_species(&mut self) {
        let living: std::collections::HashSet<SpeciesId> =
            self.organisms.iter().map(|o| o.species_id).collect();
        self.species_registry.retain(|id, _| living.contains(id));
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
    if !(0.0..=1.0).contains(&config.center_spawn_min_fraction)
        || !(0.0..=1.0).contains(&config.center_spawn_max_fraction)
    {
        return Err(SimError::InvalidConfig(
            "center spawn fractions must be within [0, 1]".to_owned(),
        ));
    }
    if config.center_spawn_min_fraction >= config.center_spawn_max_fraction {
        return Err(SimError::InvalidConfig(
            "center_spawn_min_fraction must be less than center_spawn_max_fraction".to_owned(),
        ));
    }
    if config.starting_energy <= 0.0 {
        return Err(SimError::InvalidConfig(
            "starting_energy must be greater than zero".to_owned(),
        ));
    }
    if config.food_energy <= 0.0 {
        return Err(SimError::InvalidConfig(
            "food_energy must be greater than zero".to_owned(),
        ));
    }
    if config.reproduction_energy_cost < 0.0 {
        return Err(SimError::InvalidConfig(
            "reproduction_energy_cost must be >= 0".to_owned(),
        ));
    }
    if config.move_action_energy_cost < 0.0 {
        return Err(SimError::InvalidConfig(
            "move_action_energy_cost must be >= 0".to_owned(),
        ));
    }
    if config.turn_energy_cost < 0.0 {
        return Err(SimError::InvalidConfig(
            "turn_energy_cost must be >= 0".to_owned(),
        ));
    }
    if config.max_num_neurons == 0 {
        return Err(SimError::InvalidConfig(
            "max_num_neurons must be greater than zero".to_owned(),
        ));
    }
    if config.max_num_neurons > 256 {
        return Err(SimError::InvalidConfig(
            "max_num_neurons must be <= 256".to_owned(),
        ));
    }
    if config.max_num_neurons < config.seed_genome_config.num_neurons {
        return Err(SimError::InvalidConfig(
            "max_num_neurons must be >= seed_genome_config.num_neurons".to_owned(),
        ));
    }
    if config.food_coverage_divisor == 0 {
        return Err(SimError::InvalidConfig(
            "food_coverage_divisor must be greater than zero".to_owned(),
        ));
    }
    Ok(())
}
