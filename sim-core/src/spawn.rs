use crate::brain::reset_brain_runtime_state;
use crate::grid::{opposite_direction, world_capacity};
use crate::Simulation;
use rand::seq::SliceRandom;
use rand::Rng;
use sim_protocol::{FacingDirection, FoodId, FoodState, OrganismId, OrganismState, SpeciesId};

const FOOD_COVERAGE_DIVISOR: usize = 10;

#[derive(Clone)]
pub(crate) struct ReproductionSpawn {
    pub(crate) species_id: SpeciesId,
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
                    let Some(species_config) =
                        self.species_config(reproduction.species_id).cloned()
                    else {
                        continue;
                    };

                    let mut brain = self.generate_brain(&species_config);
                    reset_brain_runtime_state(&mut brain);
                    OrganismState {
                        id: self.alloc_organism_id(),
                        species_id: reproduction.species_id,
                        q: reproduction.q,
                        r: reproduction.r,
                        age_turns: 0,
                        facing: opposite_direction(reproduction.parent_facing),
                        energy: self.config.starting_energy,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        brain,
                    }
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
        }

        self.organisms.sort_by_key(|organism| organism.id);
        spawned
    }

    pub(crate) fn spawn_initial_population(&mut self) {
        let Some((&seed_species_id, seed_species_config)) = self.species_registry.first_key_value()
        else {
            return;
        };
        let seed_species_config = seed_species_config.clone();

        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);

        for _ in 0..self.target_population() {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let brain = self.generate_brain(&seed_species_config);
            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                species_id: seed_species_id,
                q,
                r,
                age_turns: 0,
                facing,
                energy: self.config.starting_energy,
                consumptions_count: 0,
                reproductions_count: 0,
                brain,
            };
            let added = self.add_organism(organism);
            debug_assert!(added);
        }

        self.organisms.sort_by_key(|organism| organism.id);
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
        world_capacity(self.config.world_width) / FOOD_COVERAGE_DIVISOR
    }

    pub(crate) fn replenish_food_supply(&mut self) -> Vec<FoodState> {
        let target = self.target_food_count();
        if self.foods.len() >= target {
            return Vec::new();
        }

        let mut empty_positions = self.empty_positions();
        empty_positions.shuffle(&mut self.rng);
        let spawn_count = (target - self.foods.len()).min(empty_positions.len());

        let mut spawned = Vec::with_capacity(spawn_count);
        for _ in 0..spawn_count {
            let (q, r) = empty_positions
                .pop()
                .expect("spawn count must never exceed empty positions");
            let food = FoodState {
                id: self.alloc_food_id(),
                q,
                r,
                energy: self.config.food_energy,
            };
            let added = self.add_food(food.clone());
            debug_assert!(added);
            if added {
                spawned.push(food);
            }
        }

        self.foods.sort_by_key(|food| food.id);
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
