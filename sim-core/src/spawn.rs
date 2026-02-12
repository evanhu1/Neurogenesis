use crate::brain::express_genome;
use crate::genome::{generate_seed_genome, genome_distance, mutate_genome};
use crate::grid::{opposite_direction, world_capacity};
use crate::Simulation;
use rand::seq::SliceRandom;
use rand::Rng;
use sim_types::{
    FacingDirection, FoodId, FoodState, Occupant, OrganismGenome, OrganismId, OrganismState,
    SpeciesId,
};

#[derive(Clone)]
pub(crate) struct ReproductionSpawn {
    pub(crate) parent_genome: OrganismGenome,
    pub(crate) parent_species_id: SpeciesId,
    pub(crate) parent_facing: FacingDirection,
    pub(crate) q: i32,
    pub(crate) r: i32,
}

#[derive(Clone)]
pub(crate) enum SpawnRequestKind {
    Reproduction(ReproductionSpawn),
}

#[derive(Clone)]
pub(crate) struct SpawnRequest {
    pub(crate) kind: SpawnRequestKind,
}

impl Simulation {
    pub(crate) fn resolve_spawn_requests(&mut self, queue: &[SpawnRequest]) -> Vec<OrganismState> {
        let mut spawned = Vec::new();
        for request in queue {
            let organism = match &request.kind {
                SpawnRequestKind::Reproduction(reproduction) => {
                    let mut child_genome = reproduction.parent_genome.clone();
                    mutate_genome(
                        &mut child_genome,
                        self.config.max_num_neurons,
                        &mut self.rng,
                    );

                    let threshold = self.config.speciation_threshold;
                    let child_species_id = {
                        let within_lineage = self
                            .species_registry
                            .get(&reproduction.parent_species_id)
                            .map(|root| genome_distance(&child_genome, root) <= threshold)
                            .unwrap_or(false);
                        if within_lineage {
                            reproduction.parent_species_id
                        } else {
                            let id = self.alloc_species_id();
                            self.species_registry.insert(id, child_genome.clone());
                            id
                        }
                    };

                    let brain = express_genome(&child_genome);
                    OrganismState {
                        id: self.alloc_organism_id(),
                        species_id: child_species_id,
                        q: reproduction.q,
                        r: reproduction.r,
                        age_turns: 0,
                        facing: opposite_direction(reproduction.parent_facing),
                        energy: self.config.starting_energy,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        brain,
                        genome: child_genome,
                    }
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
        }

        spawned
    }

    pub(crate) fn spawn_initial_population(&mut self) {
        let seed_config = self.config.seed_genome_config.clone();

        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);

        for _ in 0..self.target_population() {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let genome =
                generate_seed_genome(&seed_config, self.config.max_num_neurons, &mut self.rng);
            let brain = express_genome(&genome);

            // Seed genomes are independently random — each gets its own species.
            // Species assignment via genome distance only matters for mutation-derived offspring.
            let species_id = self.alloc_species_id();
            self.species_registry.insert(species_id, genome.clone());

            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                species_id,
                q,
                r,
                age_turns: 0,
                facing,
                energy: self.config.starting_energy,
                consumptions_count: 0,
                reproductions_count: 0,
                brain,
                genome,
            };
            let added = self.add_organism(organism);
            debug_assert!(added);
        }
    }

    fn random_facing(&mut self) -> FacingDirection {
        FacingDirection::ALL[self.rng.random_range(0..FacingDirection::ALL.len())]
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn alloc_food_id(&mut self) -> FoodId {
        let id = FoodId(self.next_food_id);
        self.next_food_id += 1;
        id
    }

    fn target_population(&self) -> usize {
        (self.config.num_organisms as usize).min(world_capacity(self.config.world_width))
    }

    fn target_food_count(&self) -> usize {
        world_capacity(self.config.world_width) / self.config.food_coverage_divisor as usize
    }

    pub(crate) fn replenish_food_supply(&mut self) -> Vec<FoodState> {
        let target = self.target_food_count();
        if self.foods.len() >= target {
            return Vec::new();
        }

        let need = target - self.foods.len();
        let capacity = world_capacity(self.config.world_width);
        let width = self.config.world_width as usize;
        let max_attempts = need * 16;
        let mut spawned = Vec::with_capacity(need);
        let mut attempts = 0;

        while spawned.len() < need && attempts < max_attempts {
            let cell_idx = self.rng.random_range(0..capacity);
            if self.occupancy[cell_idx].is_some() {
                attempts += 1;
                continue;
            }
            let q = (cell_idx % width) as i32;
            let r = (cell_idx / width) as i32;
            let food = FoodState {
                id: self.alloc_food_id(),
                q,
                r,
                energy: self.config.food_energy,
            };
            self.occupancy[cell_idx] = Some(Occupant::Food(food.id));
            self.foods.push(food.clone());
            spawned.push(food);
        }

        spawned
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
