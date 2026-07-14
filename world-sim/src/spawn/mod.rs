use crate::grid::world_capacity;
use crate::Simulation;
use brain::express_genome;
use brain::genome::{generate_seed_genome, restrict_predation_genes};
use rand::seq::SliceRandom;
use rand::Rng;
use types::{
    ActionType, FacingDirection, FoodId, FoodState, Occupant, OrganismGenome, OrganismId,
    OrganismState, SpeciesId,
};

mod food;
mod organisms;
pub(crate) mod world;

pub(crate) const NO_REGROWTH_SCHEDULED: u64 = u64::MAX;

struct OrganismSpawnParams {
    /// Inherited species ID; `None` marks a founder, whose species ID is
    /// derived from its own organism ID inside `build_organism`.
    species_id: Option<SpeciesId>,
    q: i32,
    r: i32,
    generation: u64,
    facing: FacingDirection,
    starting_energy_override: Option<u32>,
}
