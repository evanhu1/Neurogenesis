use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;
use thiserror::Error;
#[cfg(feature = "instrumentation")]
use types::ActionRecord;
use types::{
    FoodState, MetricsSnapshot, Occupant, OrganismGenome, OrganismId, OrganismState, TerrainCell,
    TerrainType, VisualProperties, WorldConfig, WorldSnapshot,
};

pub(crate) const PLANT_COLOR_JITTER: f32 = 0.15;

pub(crate) fn jitter_visual_rgb_uniform(
    base: VisualProperties,
    rng: &mut ChaCha8Rng,
    strength: f32,
) -> VisualProperties {
    use rand::Rng;
    VisualProperties {
        r: base.r + rng.random_range(-strength..=strength),
        g: base.g + rng.random_range(-strength..=strength),
        b: base.b + rng.random_range(-strength..=strength),
        opacity: base.opacity,
        shape: base.shape,
    }
    .clamped()
}

mod grid;
#[cfg(feature = "profiling")]
#[path = "../profiling/profiling.rs"]
pub mod profiling;
mod sensing;
mod spawn;
mod turn;

pub use turn::{AttackEvent, AttackOutcome};

#[cfg(test)]
mod tests;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
    #[error("invalid simulation state: {0}")]
    InvalidState(String),
    #[error("world serialization failed: {0}")]
    Serialization(String),
}

fn validate_runtime_config(config: &WorldConfig) -> Result<(), SimError> {
    config::validate_world_config(config).map_err(SimError::InvalidConfig)?;
    if config.seed_genome_config.num_neurons > brain::genome::MAX_INTER_NEURONS {
        return Err(SimError::InvalidConfig(format!(
            "seed_genome_config.num_neurons must be <= {} to fit the runtime neuron-ID space",
            brain::genome::MAX_INTER_NEURONS
        )));
    }
    Ok(())
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Simulation {
    config: WorldConfig,
    /// Experiment provenance only. `true` means the CLI's non-canonical
    /// `--scale WIDTH,POP` shortcut constructed this world. Persisting the bit
    /// prevents later one-shot reads from accidentally presenting a scaled
    /// run as canonical.
    #[serde(default)]
    experiment_scaled: bool,
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    champion_pool: Vec<OrganismGenome>,
    /// Evaluation-only diagnostic: when set, attacks cannot affect organisms
    /// sourced from the same champion-pool entry (`species_id % pool_count`).
    /// This is deliberately not world configuration or organism-observable
    /// state; it isolates friendly fire as a predation failure mechanism.
    #[serde(default)]
    cross_pool_predation_pool_count: Option<usize>,
    #[serde(default)]
    #[serde(skip)]
    pub(crate) attack_events_last_turn: Vec<turn::AttackEvent>,
    next_organism_id: u64,
    next_food_id: u64,
    organisms: Vec<OrganismState>,
    foods: Vec<FoodState>,
    occupancy: Vec<Option<Occupant>>,
    terrain_map: Vec<bool>,
    food_tiles: Vec<bool>,
    food_regrowth_due_turn: Vec<u64>,
    food_regrowth_schedule: BTreeMap<u64, Vec<usize>>,
    /// Wire-format terrain cells, sorted by (q, r). Terrain is immutable after
    /// world generation, so this is built once per reset and cloned per
    /// snapshot instead of being rebuilt and re-sorted on every call.
    terrain_cells: Vec<TerrainCell>,
    // Instrumentation-only and not behavior-affecting. Persist the most recent
    // records so one-shot CLI `decide`/`inspect` reads remain useful after a
    // mutating command saves and exits.
    #[cfg(feature = "instrumentation")]
    #[serde(default)]
    action_records: Vec<Option<ActionRecord>>,
    /// Per-simulation rayon pool. Using a dedicated pool avoids the shared-pool
    /// bottleneck when many seed simulations run concurrently in the evaluation
    /// harness — each sim's `par_iter_mut` work actually runs in parallel on
    /// its own workers instead of serializing through one global pool.
    // Not serializable and not behavior-affecting (parallelism only); a loaded
    // sim lazily rebuilds its own pool.
    #[serde(skip)]
    pub(crate) cached_thread_pool: std::sync::OnceLock<std::sync::Arc<rayon::ThreadPool>>,
    /// Reusable per-tick scratch buffers (see `turn::TurnScratch`). Transient:
    /// always cleared + resized before use, so contents never affect behavior.
    #[serde(skip)]
    turn_scratch: turn::TurnScratch,
    metrics: MetricsSnapshot,
}

impl Clone for Simulation {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            experiment_scaled: self.experiment_scaled,
            turn: self.turn,
            seed: self.seed,
            rng: self.rng.clone(),
            champion_pool: self.champion_pool.clone(),
            cross_pool_predation_pool_count: self.cross_pool_predation_pool_count,
            attack_events_last_turn: Vec::new(),
            next_organism_id: self.next_organism_id,
            next_food_id: self.next_food_id,
            organisms: self.organisms.clone(),
            foods: self.foods.clone(),
            occupancy: self.occupancy.clone(),
            terrain_map: self.terrain_map.clone(),
            food_tiles: self.food_tiles.clone(),
            food_regrowth_due_turn: self.food_regrowth_due_turn.clone(),
            food_regrowth_schedule: self.food_regrowth_schedule.clone(),
            terrain_cells: self.terrain_cells.clone(),
            #[cfg(feature = "instrumentation")]
            action_records: self.action_records.clone(),
            // Deliberately NOT cloned: each simulation gets a dedicated rayon
            // pool (see the field's doc comment), so a clone starts empty and
            // lazily builds its own pool instead of sharing the original's.
            cached_thread_pool: std::sync::OnceLock::new(),
            // Scratch buffers are transient and cleared before every use, so a
            // clone starts with fresh (empty) ones.
            turn_scratch: turn::TurnScratch::default(),
            metrics: self.metrics.clone(),
        }
    }
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
        validate_runtime_config(&config)?;

        let capacity = grid::world_capacity(config.world_width);
        let mut sim = Self {
            config,
            experiment_scaled: false,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            champion_pool: Vec::new(),
            cross_pool_predation_pool_count: None,
            attack_events_last_turn: Vec::new(),
            next_organism_id: 0,
            next_food_id: 0,
            organisms: Vec::new(),
            foods: Vec::new(),
            occupancy: vec![None; capacity],
            terrain_cells: Vec::new(),
            terrain_map: Vec::new(),
            food_tiles: Vec::new(),
            food_regrowth_due_turn: Vec::new(),
            food_regrowth_schedule: BTreeMap::new(),
            #[cfg(feature = "instrumentation")]
            action_records: Vec::new(),
            cached_thread_pool: std::sync::OnceLock::new(),
            turn_scratch: turn::TurnScratch::default(),
            metrics: MetricsSnapshot::default(),
        };

        sim.reset_with_champion_pool(None, champion_pool);
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn food_tile_count(&self) -> usize {
        self.food_tiles
            .iter()
            .filter(|&&is_food_tile| is_food_tile)
            .count()
    }

    pub fn habitable_cell_count(&self) -> usize {
        self.terrain_map.iter().filter(|&&blocked| !blocked).count()
    }

    pub fn experiment_scaled(&self) -> bool {
        self.experiment_scaled
    }

    pub fn set_experiment_scaled(&mut self, scaled: bool) {
        self.experiment_scaled = scaled;
    }

    pub fn set_cross_pool_predation_pool_count(&mut self, pool_count: Option<usize>) {
        self.cross_pool_predation_pool_count = pool_count.filter(|&count| count > 0);
    }

    pub fn attack_events_last_turn(&self) -> &[turn::AttackEvent] {
        &self.attack_events_last_turn
    }

    fn build_terrain_cells(&mut self) {
        let width = self.config.world_width as usize;
        let capacity = width * width;
        self.terrain_cells.clear();
        for idx in 0..capacity {
            let q = (idx % width) as i32;
            let r = (idx / width) as i32;
            if self.terrain_map[idx] {
                self.terrain_cells.push(TerrainCell {
                    q,
                    r,
                    terrain_type: TerrainType::Mountain,
                    visual: types::terrain_visual(TerrainType::Mountain),
                });
            }
        }
        self.terrain_cells.sort_by_key(|cell| (cell.q, cell.r));
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        // Sorted-by-ascending-id is a maintained invariant (monotonic ID
        // allocation + push/retain); validate_state rejects unsorted vectors.
        debug_assert!(self
            .organisms
            .windows(2)
            .all(|window| window[0].id < window[1].id));
        debug_assert!(self
            .foods
            .windows(2)
            .all(|window| window[0].id < window[1].id));
        let organisms = self.organisms.clone();
        let foods = self.foods.clone();
        // Terrain is immutable after world generation; clone the cached
        // sorted cell list instead of rebuilding it per snapshot.
        let terrain = self.terrain_cells.clone();

        WorldSnapshot {
            turn: self.turn,
            rng_seed: self.seed,
            config: self.config.clone(),
            organisms,
            foods,
            terrain,
            metrics: self.metrics.clone(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        let pool = std::mem::take(&mut self.champion_pool);
        self.reset_with_champion_pool(seed, pool);
    }

    pub fn reset_with_champion_pool(
        &mut self,
        seed: Option<u64>,
        mut champion_pool: Vec<OrganismGenome>,
    ) {
        self.seed = seed.unwrap_or(self.seed);
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.turn = 0;
        // Champion-pool genomes come from disk (champion_pool.json, evaluation
        // snapshots) and may be malformed; sanitize at intake so expression
        // never sees e.g. num_neurons above MAX_INTER_NEURONS or misaligned
        // brain vectors. Sanitization uses a dedicated RNG stream derived from
        // the seed — never the world-generation stream — so a malformed pool
        // entry that draws during repair cannot shift terrain or spawn layout
        // for the same config + seed. Well-formed genomes pass through
        // unchanged either way.
        const CHAMPION_SANITIZE_RNG_MIX: u64 = 0xC4A5_3C5D_2BBF_3A91;
        let mut sanitize_rng = ChaCha8Rng::seed_from_u64(self.seed ^ CHAMPION_SANITIZE_RNG_MIX);
        for genome in &mut champion_pool {
            brain::genome::align_genome_vectors(genome, &mut sanitize_rng);
            brain::genome::restrict_predation_genes(genome, self.config.predation_enabled);
        }
        self.champion_pool = champion_pool;
        self.next_organism_id = 0;
        self.next_food_id = 0;
        self.organisms.clear();
        self.foods.clear();
        self.occupancy.fill(None);
        // terrain_map/food_tiles/food_regrowth_due_turn/food_regrowth_schedule are
        // wholesale reassigned by initialize_terrain/build_terrain_cells/
        // initialize_food_ecology below, so they need no clearing here.
        #[cfg(feature = "instrumentation")]
        self.action_records.clear();
        self.metrics = MetricsSnapshot::default();
        self.initialize_terrain();
        self.build_terrain_cells();
        self.spawn_initial_population();
        self.initialize_food_ecology();
        self.seed_initial_food_supply();
        self.refresh_population_metrics();
    }

    pub fn advance_n(&mut self, count: u32) {
        for _ in 0..count {
            let _ = self.tick();
        }
    }

    /// Serialize the full world to a writer as CBOR (self-describing binary).
    /// Round-trips deterministically: `save` then [`Simulation::load`] yields a
    /// world that advances byte-identically to the original (the RNG state and
    /// all id counters are persisted exactly). Transient/parallelism fields
    /// (`turn_scratch`, `cached_thread_pool`) are not written and start empty
    /// on load. Instrumentation action records are observational only but are
    /// persisted so post-save CLI inspection describes the last executed tick.
    pub fn save<W: std::io::Write>(&self, writer: W) -> Result<(), SimError> {
        ciborium::into_writer(self, writer).map_err(|e| SimError::Serialization(e.to_string()))
    }

    /// Deserialize a world previously written by [`Simulation::save`]. Runs
    /// [`Simulation::validate_state`] as an integrity gate before returning, so
    /// a corrupt or incompatible blob fails loudly rather than ticking into
    /// undefined behavior.
    pub fn load<R: std::io::Read>(reader: R) -> Result<Self, SimError> {
        let sim: Self =
            ciborium::from_reader(reader).map_err(|e| SimError::Serialization(e.to_string()))?;
        sim.validate_state()?;
        Ok(sim)
    }

    pub fn focused_organism(&self, id: OrganismId) -> Option<OrganismState> {
        // Organisms are maintained sorted ascending by id (see validate_state),
        // so a binary search replaces the previous linear scan.
        self.organisms
            .binary_search_by_key(&id, |organism| organism.id)
            .ok()
            .map(|idx| self.organisms[idx].clone())
    }

    pub fn organisms(&self) -> &[OrganismState] {
        &self.organisms
    }

    pub fn foods(&self) -> &[FoodState] {
        &self.foods
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

    /// Patch the current tick's record with the post-commit cumulative
    /// consumption counts so a consumption is visible in the same tick's
    /// record (the record is built during the intent phase, before commit).
    /// All three counters are patched together so `plant + prey == total`
    /// holds on the record, not just on the organism.
    #[cfg(feature = "instrumentation")]
    pub(crate) fn record_consumption_counts(
        &mut self,
        organism_idx: usize,
        total: u64,
        plant: u64,
        prey: u64,
    ) {
        if let Some(Some(record)) = self.action_records.get_mut(organism_idx) {
            record.consumptions_count = total;
            record.plant_consumptions_count = plant;
            record.prey_consumptions_count = prey;
        }
    }

    pub fn validate_state(&self) -> Result<(), SimError> {
        validate_runtime_config(&self.config)?;

        let expected_capacity = grid::world_capacity(self.config.world_width);
        for (name, len) in [
            ("occupancy", self.occupancy.len()),
            ("terrain_map", self.terrain_map.len()),
            ("food_tiles", self.food_tiles.len()),
            ("food_regrowth_due_turn", self.food_regrowth_due_turn.len()),
        ] {
            if len != expected_capacity {
                return Err(SimError::InvalidState(format!(
                    "{} length {} does not match expected capacity {}",
                    name, len, expected_capacity
                )));
            }
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
        // Reverse direction: every cell with a due turn set must appear in the
        // schedule. A cell with `food_regrowth_due_turn` set but missing from
        // `food_regrowth_schedule` can never regrow food again, because
        // schedule_food_regrowth early-returns on an already-set due turn.
        // Combined with the per-entry check above, equal counts imply a
        // one-to-one correspondence.
        let scheduled_cell_count: usize = self.food_regrowth_schedule.values().map(Vec::len).sum();
        let due_cell_count = self
            .food_regrowth_due_turn
            .iter()
            .filter(|&&due_turn| due_turn != spawn::NO_REGROWTH_SCHEDULED)
            .count();
        if due_cell_count != scheduled_cell_count {
            return Err(SimError::InvalidState(format!(
                "{} cells have a food regrowth due turn set but food_regrowth_schedule contains {} cell entries",
                due_cell_count, scheduled_cell_count
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
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                expected_occupancy[idx] = Some(Occupant::Wall);
            }
            if blocked && self.food_tiles[idx] {
                return Err(SimError::InvalidState(format!(
                    "wall cell index {idx} cannot be a food tile"
                )));
            }
        }

        for organism in &self.organisms {
            // Plasticity indexes these per-neuron buffers directly (no length-safe
            // zip) and `bootstrap_means` only re-seeds when `means_initialized`
            // is false — so a loaded blob whose buffers don't match the layer
            // sizes would panic out-of-bounds in release. Reject it here instead.
            let brain = &organism.brain;
            if brain.sensory_mean_activation.len() != brain.sensory.len()
                || brain.inter_mean_activation.len() != brain.inter.len()
                || brain.action_mean_activation.len() != brain.action.len()
            {
                return Err(SimError::InvalidState(format!(
                    "organism {:?} brain mean-activation buffers do not match layer sizes",
                    organism.id
                )));
            }
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
