use crate::brain::express_genome;
use crate::genome::{
    generate_seed_genome, restrict_predation_genes, MAX_MUTATED_VISION_DISTANCE,
    MIN_MUTATED_VISION_DISTANCE,
};
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
pub(crate) mod world;

pub(crate) use food::CORPSE_ENERGY_RETENTION;

pub(crate) const NO_REGROWTH_SCHEDULED: u64 = u64::MAX;

pub(crate) struct ReproductionSpawn {
    /// The parent genome, cloned at conception. Reproduction is clonal.
    pub(crate) offspring_genome: OrganismGenome,
    pub(crate) offspring_generation: u64,
    pub(crate) parent_species_id: SpeciesId,
    pub(crate) parent_facing: FacingDirection,
    pub(crate) offspring_starting_energy: f32,
    pub(crate) q: i32,
    pub(crate) r: i32,
}

pub(crate) enum SpawnRequest {
    Reproduction(Box<ReproductionSpawn>),
}

struct OrganismSpawnParams {
    /// Inherited species ID; `None` marks a founder, whose species ID is
    /// derived from its own organism ID inside `build_organism`.
    species_id: Option<SpeciesId>,
    q: i32,
    r: i32,
    generation: u64,
    facing: FacingDirection,
    starting_energy_override: Option<f32>,
}
