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

const DEFAULT_TERRAIN_THRESHOLD: f64 = 0.86;
const MAX_BIOMASS_PER_TILE: f32 = 1.0;
const BIOMASS_SPAWN_THRESHOLD: f32 = 1.0;
const BLOCKED_BIOMASS_DECAY_PER_TICK: f32 = 0.0;

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
                        self.config.global_mutation_rate_modifier,
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

                    let brain = express_genome(&child_genome, &mut self.rng);
                    OrganismState {
                        id: self.alloc_organism_id(),
                        species_id: child_species_id,
                        q: reproduction.q,
                        r: reproduction.r,
                        age_turns: 0,
                        facing: opposite_direction(reproduction.parent_facing),
                        energy: child_genome.starting_energy,
                        energy_prev: child_genome.starting_energy,
                        dopamine: 0.0,
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
        let initial_population = self.target_population().min(open_positions.len());

        for _ in 0..initial_population {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let genome = generate_seed_genome(&seed_config, &mut self.rng);
            let brain = express_genome(&genome, &mut self.rng);

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
                energy: genome.starting_energy,
                energy_prev: genome.starting_energy,
                dopamine: 0.0,
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
        let max_population = self.config.num_organisms as usize;
        let available_cells = if self.terrain_map.is_empty() {
            world_capacity(self.config.world_width)
        } else {
            self.terrain_map.iter().filter(|blocked| !**blocked).count()
        };
        max_population.min(available_cells)
    }

    pub(crate) fn initialize_terrain(&mut self) {
        let width = self.config.world_width;
        let terrain_seed = self.seed ^ 0xA5A5_A5A5_u64;
        self.terrain_map = if (self.config.terrain_threshold as f64 - DEFAULT_TERRAIN_THRESHOLD)
            .abs()
            > f64::EPSILON
        {
            build_terrain_map_with_threshold(
                width,
                width,
                self.config.terrain_noise_scale as f64,
                terrain_seed,
                self.config.terrain_threshold as f64,
            )
        } else {
            build_terrain_map(
                width,
                width,
                self.config.terrain_noise_scale as f64,
                terrain_seed,
            )
        };
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                self.occupancy[idx] = Some(Occupant::Wall);
            }
        }
    }

    pub(crate) fn initialize_food_ecology(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        self.food_fertility = build_fertility_map(self.config.world_width, self.seed, &self.config);
        debug_assert_eq!(self.food_fertility.len(), capacity);
        self.biomass = vec![0.0; capacity];
    }

    pub(crate) fn seed_initial_food_supply(&mut self) {
        self.ensure_food_ecology_state();
        let mut spawned_any = false;
        for cell_idx in 0..self.food_fertility.len() {
            if matches!(self.occupancy[cell_idx], Some(Occupant::Wall)) {
                self.biomass[cell_idx] = 0.0;
                continue;
            }

            let fertility = self.fertility_value(cell_idx);
            // Warm-start plant mass so the world is not "bare" at turn zero.
            let initial_fill =
                BIOMASS_SPAWN_THRESHOLD * (0.5 * self.rng.random::<f32>() + 0.5 * fertility);
            self.biomass[cell_idx] = initial_fill.min(MAX_BIOMASS_PER_TILE);

            if self.occupancy[cell_idx].is_none() && self.rng.random::<f32>() <= fertility {
                if self.spawn_food_at_cell(cell_idx).is_some() {
                    spawned_any = true;
                    self.biomass[cell_idx] =
                        (self.biomass[cell_idx] - BIOMASS_SPAWN_THRESHOLD).max(0.0);
                }
            }
        }

        // Ensure at least one food source exists at startup when an empty tile is available.
        if !spawned_any {
            let mut best_tile = None;
            let mut best_fertility = f32::MIN;
            for cell_idx in 0..self.food_fertility.len() {
                if self.occupancy[cell_idx].is_some() {
                    continue;
                }
                let fertility = self.fertility_value(cell_idx);
                if fertility > best_fertility {
                    best_fertility = fertility;
                    best_tile = Some(cell_idx);
                }
            }

            if let Some(cell_idx) = best_tile {
                if self.spawn_food_at_cell(cell_idx).is_some() {
                    self.biomass[cell_idx] =
                        (self.biomass[cell_idx] - BIOMASS_SPAWN_THRESHOLD).max(0.0);
                }
            }
        }
    }

    pub(crate) fn replenish_food_supply(&mut self) -> Vec<FoodState> {
        self.ensure_food_ecology_state();
        let mut spawned = Vec::new();

        debug_assert!(BIOMASS_SPAWN_THRESHOLD > 0.0);
        debug_assert!(MAX_BIOMASS_PER_TILE >= BIOMASS_SPAWN_THRESHOLD);

        for cell_idx in 0..self.food_fertility.len() {
            if self.occupancy[cell_idx].is_none() {
                let fertility = self.fertility_value(cell_idx);
                let grown = (self.biomass[cell_idx] + fertility * self.config.plant_growth_speed)
                    .min(MAX_BIOMASS_PER_TILE);
                self.biomass[cell_idx] = grown;

                // Cap to one spawned food per tile per tick by occupancy.
                if grown >= BIOMASS_SPAWN_THRESHOLD {
                    if let Some(food) = self.spawn_food_at_cell(cell_idx) {
                        spawned.push(food);
                        self.biomass[cell_idx] =
                            (self.biomass[cell_idx] - BIOMASS_SPAWN_THRESHOLD).max(0.0);
                    }
                }
            } else if BLOCKED_BIOMASS_DECAY_PER_TICK > 0.0 {
                self.biomass[cell_idx] =
                    (self.biomass[cell_idx] - BLOCKED_BIOMASS_DECAY_PER_TICK).max(0.0);
            }
        }

        spawned
    }

    fn ensure_food_ecology_state(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        let valid_fertility = self.food_fertility.len() == capacity;
        let valid_biomass = self.biomass.len() == capacity;
        if valid_fertility && valid_biomass {
            return;
        }

        self.initialize_food_ecology();
    }

    fn spawn_food_at_cell(&mut self, cell_idx: usize) -> Option<FoodState> {
        if self.occupancy[cell_idx].is_some() {
            return None;
        }
        let width = self.config.world_width as usize;
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
        Some(food)
    }

    fn fertility_value(&self, tile_idx: usize) -> f32 {
        self.food_fertility[tile_idx] as f32 / u16::MAX as f32
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

fn build_fertility_map(world_width: u32, seed: u64, config: &sim_types::WorldConfig) -> Vec<u16> {
    let width = world_width as usize;
    let mut fertility = Vec::with_capacity(width * width);
    for r in 0..width {
        for q in 0..width {
            let x = q as f64 * config.food_fertility_noise_scale as f64;
            let y = r as f64 * config.food_fertility_noise_scale as f64;
            let value = fractal_perlin_2d(x, y, seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0) as f32;
            let shifted = config.food_fertility_floor
                + (1.0 - config.food_fertility_floor)
                    * normalized.powf(config.food_fertility_exponent);
            let encoded = (shifted.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
            fertility.push(encoded);
        }
    }
    fertility
}

pub(crate) fn build_terrain_map(width: u32, height: u32, scale: f64, seed: u64) -> Vec<bool> {
    build_terrain_map_with_threshold(width, height, scale, seed, DEFAULT_TERRAIN_THRESHOLD)
}

fn build_terrain_map_with_threshold(
    width: u32,
    height: u32,
    scale: f64,
    seed: u64,
    terrain_threshold: f64,
) -> Vec<bool> {
    let width = width as usize;
    let height = height as usize;
    let mut blocked = Vec::with_capacity(width * height);
    for r in 0..height {
        for q in 0..width {
            let x = q as f64 * scale;
            let y = r as f64 * scale;
            let value = fractal_perlin_2d(x, y, seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
            blocked.push(normalized > terrain_threshold);
        }
    }
    blocked
}

fn fractal_perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    const OCTAVES: usize = 1;
    let mut amplitude = 1.0_f64;
    let mut frequency = 1.0_f64;
    let mut total = 0.0_f64;
    let mut weight = 0.0_f64;

    for octave in 0..OCTAVES {
        let octave_seed =
            seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(octave as u64 + 1));
        total += amplitude * perlin_2d(x * frequency, y * frequency, octave_seed);
        weight += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    if weight == 0.0 {
        0.0
    } else {
        total / weight
    }
}

fn perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let n00 = grad(hash_2d(x0, y0, seed), dx, dy);
    let n10 = grad(hash_2d(x1, y0, seed), dx - 1.0, dy);
    let n01 = grad(hash_2d(x0, y1, seed), dx, dy - 1.0);
    let n11 = grad(hash_2d(x1, y1, seed), dx - 1.0, dy - 1.0);

    let u = fade(dx);
    let v = fade(dy);
    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);
    lerp(nx0, nx1, v)
}

fn hash_2d(x: i64, y: i64, seed: u64) -> u64 {
    let mut z = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z
}

fn grad(hash: u64, x: f64, y: f64) -> f64 {
    match (hash & 7) as u8 {
        0 => x + y,
        1 => x - y,
        2 => -x + y,
        3 => -x - y,
        4 => x,
        5 => -x,
        6 => y,
        _ => -y,
    }
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
