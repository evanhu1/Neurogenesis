use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sim_protocol::{
    FacingDirection, FoodId, FoodState, MetricsSnapshot, OccupancyCell, OrganismId, OrganismState,
    SpeciesConfig, SpeciesId, TickDelta, WorldConfig, WorldSnapshot,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use thiserror::Error;

mod brain;
mod grid;
mod spawn;
mod species;
mod turn;

pub use brain::derive_active_neuron_ids;

#[cfg(test)]
mod tests;

const SYNAPSE_STRENGTH_MAX: f32 = 8.0;
const DEFAULT_BIAS: f32 = 0.0;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct Simulation {
    config: WorldConfig,
    species_registry: BTreeMap<SpeciesId, SpeciesConfig>,
    next_species_id: u32,
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    next_organism_id: u64,
    next_food_id: u64,
    organisms: Vec<OrganismState>,
    foods: Vec<FoodState>,
    occupancy: Vec<Option<CellEntity>>,
    metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellEntity {
    Organism(OrganismId),
    Food(FoodId),
}

#[derive(Default)]
struct BrainEvaluation {
    actions: [bool; 4],
    action_activations: [f32; 4],
    synapse_ops: u64,
}

#[derive(Clone)]
struct ReproductionSpawn {
    species_id: SpeciesId,
    parent_facing: FacingDirection,
    q: i32,
    r: i32,
}

#[derive(Clone)]
enum SpawnRequestKind {
    Reproduction(ReproductionSpawn),
}

#[derive(Clone)]
struct SpawnRequest {
    kind: SpawnRequestKind,
}

impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        validate_world_config(&config)?;
        validate_species_config(&config.seed_species_config)?;

        let capacity = world_capacity(config.world_width);
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

        sim.initialize_species_registry_from_seed();
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
            if let Some(entity) = maybe_entity {
                let q = (idx % width) as i32;
                let r = (idx / width) as i32;
                let mut cell = OccupancyCell {
                    q,
                    r,
                    organism_ids: Vec::new(),
                    food_ids: Vec::new(),
                };
                match entity {
                    CellEntity::Organism(id) => cell.organism_ids.push(*id),
                    CellEntity::Food(id) => cell.food_ids.push(*id),
                }
                occupancy.push(cell);
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
        self.initialize_species_registry_from_seed();
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

    pub fn export_trace_jsonl(&mut self, turns: u32) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(
            serde_json::to_string(&self.snapshot())
                .expect("serialize initial snapshot for trace export"),
        );

        for _ in 0..turns {
            self.tick();
            lines.push(
                serde_json::to_string(&self.snapshot())
                    .expect("serialize turn snapshot for trace export"),
            );
        }
        lines
    }

    pub fn metrics(&self) -> &MetricsSnapshot {
        &self.metrics
    }
}

fn world_capacity(width: u32) -> usize {
    width as usize * width as usize
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
    Ok(())
}

fn validate_species_config(config: &SpeciesConfig) -> Result<(), SimError> {
    if !(0.0..=1.0).contains(&config.mutation_chance) {
        return Err(SimError::InvalidConfig(
            "mutation_chance must be within [0, 1]".to_owned(),
        ));
    }
    if config.max_num_neurons < config.num_neurons {
        return Err(SimError::InvalidConfig(
            "max_num_neurons must be >= num_neurons".to_owned(),
        ));
    }
    Ok(())
}

pub fn compare_snapshots(a: &WorldSnapshot, b: &WorldSnapshot) -> Ordering {
    let snapshot_a = serde_json::to_string(a).expect("serialize snapshot A");
    let snapshot_b = serde_json::to_string(b).expect("serialize snapshot B");
    snapshot_a.cmp(&snapshot_b)
}
