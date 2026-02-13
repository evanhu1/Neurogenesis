use crate::brain::express_genome;
use crate::genome::{
    generate_seed_genome, genome_distance, mutate_genome, prune_disconnected_inter_neurons,
};
use crate::grid::{opposite_direction, world_capacity};
use crate::Simulation;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sim_types::{
    FacingDirection, FoodId, FoodState, Occupant, OrganismGenome, OrganismId, OrganismState,
    SpeciesId,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FoodRegrowthEvent {
    pub(crate) due_turn: u64,
    #[serde(default)]
    pub(crate) tie_break: u64,
    pub(crate) tile_idx: usize,
    pub(crate) generation: u32,
}

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
                    prune_disconnected_inter_neurons(&mut child_genome);

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

    pub(crate) fn initialize_food_ecology(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        self.food_fertility = build_fertility_map(self.config.world_width, self.seed, &self.config);
        debug_assert_eq!(self.food_fertility.len(), capacity);
        self.food_regrowth_generation = vec![0; capacity];
        self.food_regrowth_queue.clear();
    }

    pub(crate) fn seed_initial_food_supply(&mut self) {
        self.ensure_food_ecology_state();
        for cell_idx in 0..self.food_fertility.len() {
            if self.occupancy[cell_idx].is_some() {
                continue;
            }
            if self.accept_fertility_sample(cell_idx) {
                let _ = self.spawn_food_at_cell(cell_idx);
            }
        }
    }

    pub(crate) fn bootstrap_food_regrowth_queue(&mut self) {
        self.ensure_food_ecology_state();
        for idx in 0..self.food_fertility.len() {
            if matches!(self.occupancy[idx], Some(Occupant::Food(_))) {
                continue;
            }
            let delay = self.regrowth_delay_for_tile(idx);
            self.schedule_food_regrowth_with_delay(idx, delay);
        }
    }

    pub(crate) fn mark_food_consumed(&mut self, tile_idx: usize) {
        self.ensure_food_ecology_state();
        let delay = self.regrowth_delay_for_tile(tile_idx);
        self.schedule_food_regrowth_with_delay(tile_idx, delay);
    }

    pub(crate) fn replenish_food_supply(&mut self) -> Vec<FoodState> {
        self.ensure_food_ecology_state();
        let mut spawned = Vec::new();

        loop {
            let Some(next_event) = self.food_regrowth_queue.first().copied() else {
                break;
            };
            if next_event.due_turn > self.turn {
                break;
            }
            let event = self
                .food_regrowth_queue
                .pop_first()
                .expect("first event must still be present");
            if self.food_regrowth_generation[event.tile_idx] != event.generation {
                continue;
            }
            if self.occupancy[event.tile_idx].is_some() {
                self.schedule_food_regrowth_with_delay(
                    event.tile_idx,
                    u64::from(self.config.food_regrowth_retry_cooldown_turns.max(1)),
                );
                continue;
            }
            if let Some(food) = self.spawn_food_at_cell(event.tile_idx) {
                spawned.push(food);
            }
        }

        spawned
    }

    fn ensure_food_ecology_state(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        let valid_fertility = self.food_fertility.len() == capacity;
        let valid_generations = self.food_regrowth_generation.len() == capacity;
        if valid_fertility && valid_generations {
            return;
        }

        self.initialize_food_ecology();
        for idx in 0..capacity {
            if matches!(self.occupancy[idx], Some(Occupant::Food(_))) {
                continue;
            }
            let delay = self.regrowth_delay_for_tile(idx);
            self.schedule_food_regrowth_with_delay(idx, delay);
        }
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

    fn schedule_food_regrowth_with_delay(&mut self, tile_idx: usize, delay: u64) {
        let generation = self.food_regrowth_generation[tile_idx].saturating_add(1);
        self.food_regrowth_generation[tile_idx] = generation;
        self.food_regrowth_queue.insert(FoodRegrowthEvent {
            due_turn: self.turn.saturating_add(delay),
            tie_break: hash_2d(tile_idx as i64, generation as i64, self.seed),
            tile_idx,
            generation,
        });
    }

    fn regrowth_delay_for_tile(&mut self, tile_idx: usize) -> u64 {
        let min_turns = self.config.food_regrowth_min_cooldown_turns;
        let max_turns = self.config.food_regrowth_max_cooldown_turns;
        let span = max_turns.saturating_sub(min_turns);
        let fertility = self.fertility_value(tile_idx);
        let cooldown = min_turns + (((1.0 - fertility) * span as f32).round() as u32).min(span);
        let jitter = if self.config.food_regrowth_jitter_turns == 0 {
            0
        } else {
            self.rng
                .random_range(0..=self.config.food_regrowth_jitter_turns)
        };
        let base_delay = cooldown.saturating_add(jitter);
        (f64::from(base_delay) / f64::from(self.config.plant_growth_speed)).ceil() as u64
    }

    fn accept_fertility_sample(&mut self, tile_idx: usize) -> bool {
        let chance = self.fertility_value(tile_idx);
        self.rng.random::<f32>() <= chance
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

fn fractal_perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    const OCTAVES: usize = 4;
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
