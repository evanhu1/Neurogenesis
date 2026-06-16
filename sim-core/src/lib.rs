use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[cfg(feature = "instrumentation")]
use sim_types::ActionRecord;
use sim_types::{
    FoodState, MetricsSnapshot, Occupant, OrganismGenome, OrganismId, OrganismState, TerrainCell,
    TerrainType, VisualProperties, WorldConfig, WorldSnapshot,
};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::spawn::world::{hash_2d, hash_to_unit_interval};

const TERRAIN_COLOR_JITTER: f32 = 0.15;
pub(crate) const PLANT_COLOR_JITTER: f32 = 0.15;

pub(crate) fn jitter_visual_rgb(
    base: VisualProperties,
    x: i64,
    y: i64,
    seed: u64,
    strength: f32,
) -> VisualProperties {
    let r_jitter = unit_signed_hash(x, y, seed) * strength;
    let g_jitter = unit_signed_hash(x, y, seed.wrapping_add(0x1)) * strength;
    let b_jitter = unit_signed_hash(x, y, seed.wrapping_add(0x2)) * strength;
    VisualProperties {
        r: base.r + r_jitter,
        g: base.g + g_jitter,
        b: base.b + b_jitter,
        opacity: base.opacity,
        shape: base.shape,
    }
    .clamped()
}

fn unit_signed_hash(x: i64, y: i64, seed: u64) -> f32 {
    // Bit-identical to the previous inline conversion: same `>> 11` shift and
    // 2^-53 scale, performed in f64 before the f32 cast.
    (hash_to_unit_interval(hash_2d(x, y, seed)) as f32) * 2.0 - 1.0
}

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

mod brain;
pub(crate) mod genome;
mod grid;
mod metabolism;
#[path = "brain/pending_action.rs"]
mod pending_action;
#[path = "brain/plasticity.rs"]
mod plasticity;
#[cfg(feature = "profiling")]
#[path = "../profiling/profiling.rs"]
pub mod profiling;
mod spawn;
#[path = "brain/topology.rs"]
mod topology;
mod turn;

pub(crate) use pending_action::{PendingActionKind, PendingActionState};

#[cfg(test)]
mod tests;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
    #[error("invalid simulation state: {0}")]
    InvalidState(String),
}

#[derive(Debug)]
pub struct Simulation {
    config: WorldConfig,
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    champion_pool: Vec<OrganismGenome>,
    next_organism_id: u64,
    next_food_id: u64,
    organisms: Vec<OrganismState>,
    pending_actions: Vec<PendingActionState>,
    foods: Vec<FoodState>,
    occupancy: Vec<Option<Occupant>>,
    terrain_map: Vec<bool>,
    spike_map: Vec<bool>,
    food_fertility: Vec<bool>,
    food_regrowth_due_turn: Vec<u64>,
    food_regrowth_schedule: BTreeMap<u64, Vec<usize>>,
    visual_map: Vec<VisualProperties>,
    visual_map_base: Vec<VisualProperties>,
    spike_visual_map: Vec<VisualProperties>,
    /// Wire-format terrain cells, sorted by (q, r). Terrain is immutable after
    /// world generation, so this is built once per reset and cloned per
    /// snapshot instead of being rebuilt and re-sorted on every call.
    terrain_cells: Vec<TerrainCell>,
    #[cfg(feature = "instrumentation")]
    action_records: Vec<Option<ActionRecord>>,
    /// Per-simulation rayon pool. Using a dedicated pool avoids the shared-pool
    /// bottleneck when many seed simulations run concurrently in the evaluation
    /// harness — each sim's `par_iter_mut` work actually runs in parallel on
    /// its own workers instead of serializing through one global pool.
    pub(crate) cached_thread_pool: std::sync::OnceLock<std::sync::Arc<rayon::ThreadPool>>,
    /// Reusable per-tick scratch buffers (see `turn::TurnScratch`). Transient:
    /// always cleared + resized before use, so contents never affect behavior.
    turn_scratch: turn::TurnScratch,
    metrics: MetricsSnapshot,
}

impl Clone for Simulation {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            turn: self.turn,
            seed: self.seed,
            rng: self.rng.clone(),
            champion_pool: self.champion_pool.clone(),
            next_organism_id: self.next_organism_id,
            next_food_id: self.next_food_id,
            organisms: self.organisms.clone(),
            pending_actions: self.pending_actions.clone(),
            foods: self.foods.clone(),
            occupancy: self.occupancy.clone(),
            terrain_map: self.terrain_map.clone(),
            spike_map: self.spike_map.clone(),
            food_fertility: self.food_fertility.clone(),
            food_regrowth_due_turn: self.food_regrowth_due_turn.clone(),
            food_regrowth_schedule: self.food_regrowth_schedule.clone(),
            visual_map: self.visual_map.clone(),
            visual_map_base: self.visual_map_base.clone(),
            spike_visual_map: self.spike_visual_map.clone(),
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
        sim_config::validate_world_config(&config).map_err(SimError::InvalidConfig)?;

        let capacity = grid::world_capacity(config.world_width);
        let mut sim = Self {
            config,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            champion_pool: Vec::new(),
            next_organism_id: 0,
            next_food_id: 0,
            organisms: Vec::new(),
            pending_actions: Vec::new(),
            foods: Vec::new(),
            occupancy: vec![None; capacity],
            visual_map: Vec::new(),
            visual_map_base: Vec::new(),
            spike_visual_map: Vec::new(),
            terrain_cells: Vec::new(),
            terrain_map: Vec::new(),
            spike_map: Vec::new(),
            food_fertility: Vec::new(),
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

    fn build_visual_map_base(&mut self) {
        let width = self.config.world_width as usize;
        let capacity = width * width;
        self.visual_map_base = vec![VisualProperties::default(); capacity];
        for (idx, blocked) in self.terrain_map.iter().enumerate() {
            if *blocked {
                self.visual_map_base[idx] =
                    sim_types::terrain_visual(sim_types::TerrainType::Mountain);
            }
        }
        self.visual_map = self.visual_map_base.clone();

        const SPIKE_COLOR_JITTER_MIX: u64 = 0xA1B2_C3D4_E5F6_0789;
        let spike_jitter_seed = self.seed ^ SPIKE_COLOR_JITTER_MIX;
        let base_spike = sim_types::terrain_visual(sim_types::TerrainType::Spikes);
        self.spike_visual_map = vec![VisualProperties::default(); capacity];
        for (idx, &is_spike) in self.spike_map.iter().enumerate() {
            if !is_spike {
                continue;
            }
            let q = (idx % width) as i64;
            let r = (idx / width) as i64;
            self.spike_visual_map[idx] =
                jitter_visual_rgb(base_spike, q, r, spike_jitter_seed, TERRAIN_COLOR_JITTER);
        }

        // Terrain is immutable after world generation, so build the sorted
        // wire-format cell list once here instead of on every snapshot. Spike
        // cells carry the per-cell jittered visual organisms actually sense
        // (spike_visual_map), not the flat base terrain color.
        self.terrain_cells.clear();
        for idx in 0..capacity {
            let q = (idx % width) as i32;
            let r = (idx / width) as i32;
            if self.terrain_map[idx] {
                self.terrain_cells.push(TerrainCell {
                    q,
                    r,
                    terrain_type: TerrainType::Mountain,
                    visual: self.visual_map_base[idx],
                });
            }
            if self.spike_map[idx] {
                self.terrain_cells.push(TerrainCell {
                    q,
                    r,
                    terrain_type: TerrainType::Spikes,
                    visual: self.spike_visual_map[idx],
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
            genome::align_genome_vectors(genome, &mut sanitize_rng);
        }
        self.champion_pool = champion_pool;
        self.next_organism_id = 0;
        self.next_food_id = 0;
        self.organisms.clear();
        self.pending_actions.clear();
        self.foods.clear();
        self.occupancy.fill(None);
        // visual_map/visual_map_base/spike_visual_map/terrain_map/spike_map/
        // food_fertility/food_regrowth_due_turn/food_regrowth_schedule are
        // wholesale reassigned by initialize_terrain/build_visual_map_base/
        // initialize_food_ecology below, so they need no clearing here.
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

    pub fn advance_n(&mut self, count: u32) {
        for _ in 0..count {
            let _ = self.tick();
        }
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
        sim_config::validate_world_config(&self.config).map_err(SimError::InvalidConfig)?;

        let expected_capacity = grid::world_capacity(self.config.world_width);
        for (name, len) in [
            ("occupancy", self.occupancy.len()),
            ("terrain_map", self.terrain_map.len()),
            ("spike_map", self.spike_map.len()),
            ("food_fertility", self.food_fertility.len()),
            ("food_regrowth_due_turn", self.food_regrowth_due_turn.len()),
            ("visual_map", self.visual_map.len()),
            ("visual_map_base", self.visual_map_base.len()),
            ("spike_visual_map", self.spike_visual_map.len()),
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
