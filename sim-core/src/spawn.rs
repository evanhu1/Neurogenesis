use crate::brain::express_genome;
use crate::genome::{generate_seed_genome, mutate_genome};
use crate::grid::{opposite_direction, world_capacity};
use crate::Simulation;
use rand::seq::SliceRandom;
use rand::Rng;
use sim_types::{
    ActionType, FacingDirection, FoodId, FoodKind, FoodState, Occupant, OrganismGenome, OrganismId,
    OrganismState, SpeciesId,
};

mod food;
mod organisms;
mod world;

const NO_REGROWTH_SCHEDULED: u64 = u64::MAX;

#[derive(Clone)]
pub(crate) struct ReproductionSpawn {
    pub(crate) parent_genome: OrganismGenome,
    pub(crate) parent_generation: u64,
    pub(crate) parent_species_id: SpeciesId,
    pub(crate) parent_facing: FacingDirection,
    pub(crate) offspring_starting_energy: f32,
    pub(crate) q: i32,
    pub(crate) r: i32,
}

#[derive(Clone)]
pub(crate) struct PeriodicInjectionSpawn {
    pub(crate) q: i32,
    pub(crate) r: i32,
}

#[derive(Clone)]
pub(crate) enum SpawnRequestKind {
    Reproduction(ReproductionSpawn),
    PeriodicInjection(PeriodicInjectionSpawn),
}

#[derive(Clone)]
pub(crate) struct SpawnRequest {
    pub(crate) kind: SpawnRequestKind,
}

#[derive(Clone, Copy)]
struct OrganismSpawnParams {
    species_id: SpeciesId,
    q: i32,
    r: i32,
    generation: u64,
    facing: FacingDirection,
    starting_energy_override: Option<f32>,
}

impl SpawnRequest {
    fn target_position(&self) -> (i32, i32) {
        self.kind.target_position()
    }
}

impl SpawnRequestKind {
    fn target_position(&self) -> (i32, i32) {
        match self {
            Self::Reproduction(spawn) => (spawn.q, spawn.r),
            Self::PeriodicInjection(spawn) => (spawn.q, spawn.r),
        }
    }
}

fn founder_species_id(id: OrganismId) -> SpeciesId {
    SpeciesId(id.0)
}
