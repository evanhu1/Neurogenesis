use crate::Simulation;
use brain::express_genome;
use brain::genome::{generate_seed_genome, restrict_action_genes};
use rand::seq::SliceRandom;
use rand::Rng;
use types::{
    ActionType, FacingDirection, Occupant, OrganismGenome, OrganismId, OrganismState, SpeciesId,
};

mod organisms;
pub(crate) mod world;

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
