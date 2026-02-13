use crate::spawn::FoodRegrowthEvent;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use sim_types::{
    FoodId, FoodState, MetricsSnapshot, OccupancyCell, Occupant, OrganismGenome, OrganismId,
    OrganismState, SpeciesId, TickDelta, WorldConfig, WorldSnapshot,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
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
    #[error("invalid simulation state: {0}")]
    InvalidState(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(default)]
    food_fertility: Vec<u16>,
    #[serde(default)]
    food_regrowth_generation: Vec<u32>,
    #[serde(default)]
    food_regrowth_queue: BTreeSet<FoodRegrowthEvent>,
    metrics: MetricsSnapshot,
    #[serde(skip, default = "default_intent_parallelism")]
    intent_parallelism: usize,
    #[serde(skip, default = "default_intent_parallel_min_organisms")]
    intent_parallel_min_organisms: usize,
}

fn default_intent_parallelism() -> usize {
    1
}

fn default_intent_parallel_min_organisms() -> usize {
    256
}

fn intent_pool_registry() -> &'static Mutex<HashMap<usize, Arc<ThreadPool>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn install_with_intent_pool<R, F>(threads: usize, work: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    let threads = threads.max(1);
    if threads == 1 || rayon::current_thread_index().is_some() {
        return work();
    }

    let pool = {
        let mut registry = intent_pool_registry()
            .lock()
            .expect("intent pool registry lock must not be poisoned");
        registry
            .entry(threads)
            .or_insert_with(|| {
                Arc::new(
                    ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .thread_name(|idx| format!("sim-intent-{idx}"))
                        .build()
                        .expect("intent thread pool must build"),
                )
            })
            .clone()
    };

    pool.install(work)
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
            food_fertility: Vec::new(),
            food_regrowth_generation: Vec::new(),
            food_regrowth_queue: BTreeSet::new(),
            metrics: MetricsSnapshot::default(),
            intent_parallelism: default_intent_parallelism(),
            intent_parallel_min_organisms: default_intent_parallel_min_organisms(),
        };

        sim.spawn_initial_population();
        sim.initialize_food_ecology();
        sim.seed_initial_food_supply();
        sim.bootstrap_food_regrowth_queue();
        sim.refresh_population_metrics();
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn set_intent_parallelism(&mut self, threads: usize) {
        self.intent_parallelism = threads.max(1);
    }

    pub fn set_intent_parallel_min_organisms(&mut self, min_organisms: usize) {
        self.intent_parallel_min_organisms = min_organisms.max(1);
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
        self.food_fertility.clear();
        self.food_regrowth_generation.clear();
        self.food_regrowth_queue.clear();
        self.metrics = MetricsSnapshot::default();
        self.spawn_initial_population();
        self.initialize_food_ecology();
        self.seed_initial_food_supply();
        self.bootstrap_food_regrowth_queue();
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

    pub(crate) fn should_parallelize_intents(&self, organism_count: usize) -> bool {
        self.intent_parallelism > 1 && organism_count >= self.intent_parallel_min_organisms
    }

    pub(crate) fn intent_parallelism(&self) -> usize {
        self.intent_parallelism
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

        if !self
            .organisms
            .windows(2)
            .all(|window| window[0].id < window[1].id)
        {
            return Err(SimError::InvalidState(
                "organisms must be sorted by ascending id".to_owned(),
            ));
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

        let max_species_id = self
            .species_registry
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0);
        if !self.species_registry.is_empty() && self.next_species_id <= max_species_id {
            return Err(SimError::InvalidState(format!(
                "next_species_id {} must be greater than max species id {}",
                self.next_species_id, max_species_id
            )));
        }

        let living_species: HashSet<SpeciesId> =
            self.organisms.iter().map(|o| o.species_id).collect();
        let known_species: HashSet<SpeciesId> = self.species_registry.keys().copied().collect();
        if !living_species.is_subset(&known_species) {
            return Err(SimError::InvalidState(
                "species_registry is missing one or more living species".to_owned(),
            ));
        }

        let mut computed_species_counts = BTreeMap::new();
        for organism in &self.organisms {
            *computed_species_counts
                .entry(organism.species_id)
                .or_insert(0_u32) += 1;
        }

        if self.metrics.organisms != self.organisms.len() as u32 {
            return Err(SimError::InvalidState(format!(
                "metrics.organisms {} does not match organism count {}",
                self.metrics.organisms,
                self.organisms.len()
            )));
        }

        if self.metrics.species_counts != computed_species_counts {
            return Err(SimError::InvalidState(
                "metrics.species_counts does not match organism population".to_owned(),
            ));
        }

        if self.metrics.total_species_created != self.next_species_id {
            return Err(SimError::InvalidState(format!(
                "metrics.total_species_created {} does not match next_species_id {}",
                self.metrics.total_species_created, self.next_species_id
            )));
        }

        Ok(())
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
    if !config.plant_growth_speed.is_finite() || config.plant_growth_speed <= 0.0 {
        return Err(SimError::InvalidConfig(
            "plant_growth_speed must be greater than zero".to_owned(),
        ));
    }
    if config.food_regrowth_min_cooldown_turns > config.food_regrowth_max_cooldown_turns {
        return Err(SimError::InvalidConfig(
            "food_regrowth_min_cooldown_turns must be <= food_regrowth_max_cooldown_turns"
                .to_owned(),
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
    Ok(())
}
