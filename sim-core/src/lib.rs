use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sim_protocol::{
    FacingDirection, MetricsSnapshot, OccupancyCell, OrganismId, OrganismState, TickDelta,
    WorldConfig, WorldSnapshot,
};
use std::cmp::Ordering;
use thiserror::Error;

mod brain;
mod grid;
mod spawn;
mod turn;

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
    turn: u64,
    seed: u64,
    rng: ChaCha8Rng,
    next_organism_id: u64,
    organisms: Vec<OrganismState>,
    occupancy: Vec<Option<OrganismId>>,
    metrics: MetricsSnapshot,
}

#[derive(Default)]
struct BrainEvaluation {
    actions: [bool; 3],
    synapse_ops: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SpawnRequestKind {
    StarvationReplacement,
    Reproduction { parent: OrganismId },
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SpawnRequest {
    kind: SpawnRequestKind,
}

impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        validate_config(&config)?;

        let capacity = world_capacity(config.world_width);
        let mut sim = Self {
            config,
            turn: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_organism_id: 0,
            organisms: Vec::new(),
            occupancy: vec![None; capacity],
            metrics: MetricsSnapshot::default(),
        };

        sim.spawn_initial_population();
        sim.metrics.organisms = sim.organisms.len() as u32;
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let mut organisms = self.organisms.clone();
        organisms.sort_by_key(|o| o.id);

        let width = self.config.world_width as usize;
        let mut occupancy = Vec::with_capacity(self.organisms.len());
        for (idx, maybe_id) in self.occupancy.iter().enumerate() {
            if let Some(id) = maybe_id {
                let q = (idx % width) as i32;
                let r = (idx / width) as i32;
                occupancy.push(OccupancyCell {
                    q,
                    r,
                    organism_ids: vec![*id],
                });
            }
        }
        occupancy.sort_by_key(|cell| (cell.q, cell.r));

        WorldSnapshot {
            turn: self.turn,
            rng_seed: self.seed,
            config: self.config.clone(),
            organisms,
            occupancy,
            metrics: self.metrics.clone(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.seed = seed.unwrap_or(self.seed);
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.turn = 0;
        self.next_organism_id = 0;
        self.organisms.clear();
        self.occupancy.fill(None);
        self.metrics = MetricsSnapshot::default();
        self.spawn_initial_population();
        self.metrics.organisms = self.organisms.len() as u32;
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

    fn debug_assert_consistent_state(&self) {
        if cfg!(debug_assertions) {
            debug_assert_eq!(
                self.organisms.len(),
                self.occupancy.iter().flatten().count(),
                "occupancy vector count should match organism count",
            );
            for organism in &self.organisms {
                let idx = self
                    .cell_index(organism.q, organism.r)
                    .expect("organism position must remain in bounds");
                debug_assert_eq!(
                    self.occupancy[idx],
                    Some(organism.id),
                    "occupancy must point at organism occupying that cell",
                );
            }
        }
    }

    fn random_bias(&mut self) -> f32 {
        self.rng.random_range(-1.0..1.0)
    }

    fn random_facing(&mut self) -> FacingDirection {
        FacingDirection::ALL[self.rng.random_range(0..FacingDirection::ALL.len())]
    }

    fn add_organism(&mut self, organism: OrganismState) -> bool {
        let Some(cell_idx) = self.cell_index(organism.q, organism.r) else {
            return false;
        };
        if self.occupancy[cell_idx].is_some() {
            return false;
        }

        self.occupancy[cell_idx] = Some(organism.id);
        self.organisms.push(organism);
        true
    }

    fn rebuild_occupancy(&mut self) {
        self.occupancy.fill(None);
        for organism in &self.organisms {
            let idx = self
                .cell_index(organism.q, organism.r)
                .expect("organism must remain in bounds");
            debug_assert!(self.occupancy[idx].is_none());
            self.occupancy[idx] = Some(organism.id);
        }
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn occupant_at(&self, q: i32, r: i32) -> Option<OrganismId> {
        let idx = self.cell_index(q, r)?;
        self.occupancy[idx]
    }

    fn in_bounds(&self, q: i32, r: i32) -> bool {
        let width = self.config.world_width as i32;
        q >= 0 && r >= 0 && q < width && r < width
    }

    fn cell_index(&self, q: i32, r: i32) -> Option<usize> {
        if !self.in_bounds(q, r) {
            return None;
        }
        let width = self.config.world_width as usize;
        Some(r as usize * width + q as usize)
    }

    fn target_population(&self) -> usize {
        (self.config.num_organisms as usize).min(world_capacity(self.config.world_width))
    }

    fn empty_positions(&self) -> Vec<(i32, i32)> {
        let width = self.config.world_width as i32;
        let mut positions = Vec::new();
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_none() {
                    positions.push((q, r));
                }
            }
        }
        positions
    }
}

fn world_capacity(width: u32) -> usize {
    width as usize * width as usize
}

fn validate_config(config: &WorldConfig) -> Result<(), SimError> {
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
    if config.turns_to_starve == 0 {
        return Err(SimError::InvalidConfig(
            "turns_to_starve must be >= 1".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.mutation_chance) {
        return Err(SimError::InvalidConfig(
            "mutation_chance must be within [0, 1]".to_owned(),
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
    Ok(())
}

pub fn compare_snapshots(a: &WorldSnapshot, b: &WorldSnapshot) -> Ordering {
    let snapshot_a = serde_json::to_string(a).expect("serialize snapshot A");
    let snapshot_b = serde_json::to_string(b).expect("serialize snapshot B");
    snapshot_a.cmp(&snapshot_b)
}
