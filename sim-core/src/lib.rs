use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
#[cfg(feature = "instrumentation")]
use sim_types::ActionRecord;
use sim_types::{
    FoodState, MetricsSnapshot, OccupancyCell, Occupant, OrganismGenome, OrganismId, OrganismState,
    TerrainCell, TerrainType, TickDelta, VisualProperties, WorldConfig, WorldSnapshot,
};
use std::collections::BTreeMap;
use thiserror::Error;

#[path = "brain/actor_critic.rs"]
mod actor_critic;
mod brain;
pub(crate) mod genome;
#[path = "spawn/grid.rs"]
mod grid;
mod metabolism;
#[path = "brain/plasticity.rs"]
mod plasticity;
#[cfg(feature = "profiling")]
#[path = "../profiling/profiling.rs"]
pub mod profiling;
#[path = "brain/reward.rs"]
mod reward;
mod spawn;
#[path = "brain/topology.rs"]
mod topology;
mod turn;

pub(crate) use reward::{
    PendingActionKind, PendingActionState, RewardLedger, DEFAULT_REWARD_WEIGHTS,
    REWARD_WEIGHT_COUNT, REWARD_WEIGHT_MAX, REWARD_WEIGHT_MIN,
};

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
    #[serde(default)]
    champion_pool: Vec<OrganismGenome>,
    next_organism_id: u64,
    next_food_id: u64,
    organisms: Vec<OrganismState>,
    #[serde(default)]
    pending_actions: Vec<PendingActionState>,
    #[serde(skip)]
    reward_ledgers: Vec<RewardLedger>,
    foods: Vec<FoodState>,
    occupancy: Vec<Option<Occupant>>,
    #[serde(default)]
    terrain_map: Vec<bool>,
    #[serde(default)]
    spike_map: Vec<bool>,
    #[serde(default)]
    food_fertility: Vec<bool>,
    #[serde(default)]
    food_regrowth_due_turn: Vec<u64>,
    #[serde(default)]
    food_regrowth_schedule: BTreeMap<u64, Vec<usize>>,
    #[serde(skip)]
    visual_map: Vec<VisualProperties>,
    #[serde(skip)]
    visual_map_base: Vec<VisualProperties>,
    #[cfg(feature = "instrumentation")]
    #[serde(skip)]
    action_records: Vec<Option<ActionRecord>>,
    /// Per-simulation rayon pool. Using a dedicated pool avoids the shared-pool
    /// bottleneck when many seed simulations run concurrently in the evaluation
    /// harness — each sim's `par_iter_mut` work actually runs in parallel on
    /// its own workers instead of serializing through one global pool.
    #[serde(skip)]
    pub(crate) cached_thread_pool: std::sync::OnceLock<std::sync::Arc<rayon::ThreadPool>>,
    metrics: MetricsSnapshot,
}


impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        Self::new_with_champion_pool(config, seed, Vec::new())
    }

    pub fn new_with_champion_pool(
        config: WorldConfig,
        seed: u64,
        champion_pool: Vec<OrganismGenome>,
    ) -> Result<Self, SimError> {
        sim_config::validate_world_config(&config).map_err(SimError::InvalidConfig)?;

        let capacity = grid::world_capacity(config.world_width);
        let mut sim = Self {
            config,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            champion_pool,
            next_organism_id: 0,
            next_food_id: 0,
            organisms: Vec::new(),
            pending_actions: Vec::new(),
            reward_ledgers: Vec::new(),
            foods: Vec::new(),
            occupancy: vec![None; capacity],
            visual_map: Vec::new(),
            visual_map_base: Vec::new(),
            terrain_map: Vec::new(),
            spike_map: Vec::new(),
            food_fertility: Vec::new(),
            food_regrowth_due_turn: Vec::new(),
            food_regrowth_schedule: BTreeMap::new(),
            #[cfg(feature = "instrumentation")]
            action_records: Vec::new(),
            cached_thread_pool: std::sync::OnceLock::new(),
            metrics: MetricsSnapshot::default(),
        };

        sim.initialize_terrain();
        sim.build_visual_map_base();
        sim.spawn_initial_population();
        sim.initialize_food_ecology();
        sim.seed_initial_food_supply();
        sim.refresh_population_metrics();
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    fn build_visual_map_base(&mut self) {
        let capacity = self.config.world_width as usize * self.config.world_width as usize;
        self.visual_map_base = vec![VisualProperties::default(); capacity];
        for (idx, blocked) in self.terrain_map.iter().enumerate() {
            if *blocked {
                self.visual_map_base[idx] =
                    sim_types::terrain_visual(sim_types::TerrainType::Mountain);
            }
        }
        self.visual_map = self.visual_map_base.clone();
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let mut organisms = self.organisms.clone();
        organisms.sort_by_key(|o| o.id);
        let mut foods = self.foods.clone();
        foods.sort_by_key(|food| food.id);
        let mut terrain = Vec::new();

        let width = self.config.world_width as usize;
        let mut occupancy = Vec::with_capacity(self.occupancy.iter().flatten().count());
        for (idx, maybe_entity) in self.occupancy.iter().enumerate() {
            let q = (idx % width) as i32;
            let r = (idx / width) as i32;
            if self.terrain_map[idx] {
                terrain.push(TerrainCell {
                    q,
                    r,
                    terrain_type: TerrainType::Mountain,
                    visual: sim_types::terrain_visual(TerrainType::Mountain),
                });
            }
            if self.spike_map[idx] {
                terrain.push(TerrainCell {
                    q,
                    r,
                    terrain_type: TerrainType::Spikes,
                    visual: sim_types::terrain_visual(TerrainType::Spikes),
                });
            }
            if let Some(occupant) = *maybe_entity {
                occupancy.push(OccupancyCell { q, r, occupant });
            }
        }
        terrain.sort_by_key(|cell| (cell.q, cell.r));
        occupancy.sort_by_key(|cell| (cell.q, cell.r));

        WorldSnapshot {
            turn: self.turn,
            rng_seed: self.seed,
            config: self.config.clone(),
            organisms,
            foods,
            terrain,
            occupancy,
            metrics: self.metrics.clone(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.reset_with_champion_pool(seed, self.champion_pool.clone());
    }

    pub fn reset_with_champion_pool(
        &mut self,
        seed: Option<u64>,
        champion_pool: Vec<OrganismGenome>,
    ) {
        self.seed = seed.unwrap_or(self.seed);
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.turn = 0;
        self.champion_pool = champion_pool;
        self.next_organism_id = 0;
        self.next_food_id = 0;
        self.organisms.clear();
        self.pending_actions.clear();
        self.reward_ledgers.clear();
        self.foods.clear();
        self.occupancy.fill(None);
        self.visual_map.clear();
        self.visual_map_base.clear();
        self.terrain_map.clear();
        self.spike_map.clear();
        self.food_fertility.clear();
        self.food_regrowth_due_turn.clear();
        self.food_regrowth_schedule.clear();
        #[cfg(feature = "instrumentation")]
        self.action_records.clear();
        self.metrics = MetricsSnapshot::default();
        self.initialize_terrain();
        self.build_visual_map_base();
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

    pub fn organisms(&self) -> &[OrganismState] {
        &self.organisms
    }

    pub fn metrics(&self) -> &MetricsSnapshot {
        &self.metrics
    }

    #[cfg(feature = "instrumentation")]
    pub fn action_records(&self) -> &[Option<ActionRecord>] {
        &self.action_records
    }

    #[cfg(feature = "instrumentation")]
    pub fn clear_action_records(&mut self) {
        self.action_records.clear();
    }

    #[cfg(feature = "instrumentation")]
    pub(crate) fn mark_action_succeeded(&mut self, organism_idx: usize) {
        if let Some(Some(record)) = self.action_records.get_mut(organism_idx) {
            record.action_failed = false;
        }
    }

    pub fn validate_state(&self) -> Result<(), SimError> {
        sim_config::validate_world_config(&self.config).map_err(SimError::InvalidConfig)?;

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
        if self.spike_map.len() != expected_capacity {
            return Err(SimError::InvalidState(format!(
                "spike_map length {} does not match expected capacity {}",
                self.spike_map.len(),
                expected_capacity
            )));
        }
        if self.food_fertility.len() != expected_capacity {
            return Err(SimError::InvalidState(format!(
                "food_fertility length {} does not match expected capacity {}",
                self.food_fertility.len(),
                expected_capacity
            )));
        }
        if self.food_regrowth_due_turn.len() != expected_capacity {
            return Err(SimError::InvalidState(format!(
                "food_regrowth_due_turn length {} does not match expected capacity {}",
                self.food_regrowth_due_turn.len(),
                expected_capacity
            )));
        }
        for (due_turn, cell_indices) in &self.food_regrowth_schedule {
            for &cell_idx in cell_indices {
                if cell_idx >= expected_capacity {
                    return Err(SimError::InvalidState(format!(
                        "food_regrowth_schedule contains out-of-bounds cell index {}",
                        cell_idx
                    )));
                }
                if self.food_regrowth_due_turn[cell_idx] != *due_turn {
                    return Err(SimError::InvalidState(format!(
                        "food regrowth schedule mismatch at cell {}",
                        cell_idx
                    )));
                }
            }
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
            if blocked && self.spike_map[idx] {
                return Err(SimError::InvalidState(format!(
                    "cell index {} cannot be both wall terrain and spikes",
                    idx
                )));
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
