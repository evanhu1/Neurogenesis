use crate::brain::express_genome;
use crate::genome::{generate_seed_genome, mutate_genome};
use crate::grid::{opposite_direction, world_capacity};
use crate::Simulation;
use rand::seq::SliceRandom;
use rand::Rng;
use sim_types::{
    FacingDirection, FoodId, FoodState, Occupant, OrganismGenome, OrganismId, OrganismState,
    SpeciesId,
};

const DEFAULT_TERRAIN_THRESHOLD: f64 = 0.86;
const NO_REGROWTH_SCHEDULED: u64 = u64::MAX;
const FOOD_FERTILITY_SEED_MIX: u64 = 0x6A09_E667_F3BC_C909;
const FOOD_FERTILITY_NOISE_SCALE: f64 = 0.012;
const FOOD_FERTILITY_THRESHOLD: f64 = 0.6;

#[derive(Clone)]
pub(crate) struct ReproductionSpawn {
    pub(crate) parent_genome: OrganismGenome,
    pub(crate) parent_generation: u64,
    pub(crate) parent_species_id: SpeciesId,
    pub(crate) parent_facing: FacingDirection,
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
                        self.config.meta_mutation_enabled,
                        &mut self.rng,
                    );
                    let generation = reproduction.parent_generation.saturating_add(1);
                    let id = self.alloc_organism_id();
                    let brain = express_genome(&child_genome, &mut self.rng);
                    OrganismState {
                        id,
                        species_id: reproduction.parent_species_id,
                        q: reproduction.q,
                        r: reproduction.r,
                        generation,
                        age_turns: 0,
                        facing: opposite_direction(reproduction.parent_facing),
                        energy: child_genome.starting_energy,
                        energy_prev: child_genome.starting_energy,
                        dopamine: 0.0,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        last_action_taken: sim_types::ActionType::Idle,
                        brain,
                        genome: child_genome,
                    }
                }
                SpawnRequestKind::PeriodicInjection(injection) => {
                    let genome =
                        generate_seed_genome(&self.config.seed_genome_config, &mut self.rng);
                    let id = self.alloc_organism_id();
                    let brain = express_genome(&genome, &mut self.rng);
                    OrganismState {
                        id,
                        species_id: founder_species_id(id),
                        q: injection.q,
                        r: injection.r,
                        generation: 0,
                        age_turns: 0,
                        facing: self.random_facing(),
                        energy: genome.starting_energy,
                        energy_prev: genome.starting_energy,
                        dopamine: 0.0,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        last_action_taken: sim_types::ActionType::Idle,
                        brain,
                        genome,
                    }
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
        }

        spawned
    }

    pub(crate) fn enqueue_periodic_injections(&mut self, queue: &mut Vec<SpawnRequest>) {
        let interval = self.config.periodic_injection_interval_turns as u64;
        let count = self.config.periodic_injection_count as usize;
        if interval == 0 || count == 0 {
            return;
        }

        let next_turn = self.turn.saturating_add(1);
        if next_turn % interval != 0 {
            return;
        }

        let width = self.config.world_width as usize;
        let mut reserved = vec![false; self.occupancy.len()];
        for request in queue.iter() {
            let (q, r) = request.target_position();
            let idx = r as usize * width + q as usize;
            reserved[idx] = true;
        }

        let mut open_positions = Vec::new();
        for (idx, occupant) in self.occupancy.iter().enumerate() {
            if occupant.is_none() && !reserved[idx] {
                let q = (idx % width) as i32;
                let r = (idx / width) as i32;
                open_positions.push((q, r));
            }
        }
        open_positions.shuffle(&mut self.rng);

        for (q, r) in open_positions.into_iter().take(count) {
            queue.push(SpawnRequest {
                kind: SpawnRequestKind::PeriodicInjection(PeriodicInjectionSpawn { q, r }),
            });
        }
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

            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                species_id: founder_species_id(id),
                q,
                r,
                generation: 0,
                age_turns: 0,
                facing,
                energy: genome.starting_energy,
                energy_prev: genome.starting_energy,
                dopamine: 0.0,
                consumptions_count: 0,
                reproductions_count: 0,
                last_action_taken: sim_types::ActionType::Idle,
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
        self.food_fertility = build_fertility_map(self.config.world_width, self.seed);
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                self.food_fertility[idx] = false;
            }
        }
        debug_assert_eq!(self.food_fertility.len(), capacity);
        self.food_regrowth_due_turn = vec![NO_REGROWTH_SCHEDULED; capacity];
        self.food_regrowth_schedule.clear();
    }

    pub(crate) fn seed_initial_food_supply(&mut self) {
        self.ensure_food_ecology_state();
        for cell_idx in 0..self.food_fertility.len() {
            if !self.food_fertility[cell_idx] {
                continue;
            }
            if self.occupancy[cell_idx].is_none() {
                let _ = self.spawn_food_at_cell(cell_idx);
            } else if matches!(self.occupancy[cell_idx], Some(Occupant::Organism(_))) {
                self.schedule_food_regrowth(cell_idx);
            }
        }
    }

    pub(crate) fn replenish_food_supply(&mut self) -> Vec<FoodState> {
        self.ensure_food_ecology_state();
        let mut spawned = Vec::new();
        let due_turns: Vec<u64> = self
            .food_regrowth_schedule
            .range(..=self.turn)
            .map(|(&due_turn, _)| due_turn)
            .collect();

        for due_turn in due_turns {
            let Some(cell_indices) = self.food_regrowth_schedule.remove(&due_turn) else {
                continue;
            };
            for cell_idx in cell_indices {
                if cell_idx >= self.food_regrowth_due_turn.len() {
                    continue;
                }
                if self.food_regrowth_due_turn[cell_idx] != due_turn {
                    continue;
                }
                self.food_regrowth_due_turn[cell_idx] = NO_REGROWTH_SCHEDULED;
                if !self.food_fertility[cell_idx] {
                    continue;
                }
                if self.occupancy[cell_idx].is_none() {
                    if let Some(food) = self.spawn_food_at_cell(cell_idx) {
                        spawned.push(food);
                    }
                } else if matches!(self.occupancy[cell_idx], Some(Occupant::Organism(_))) {
                    self.defer_food_regrowth(cell_idx);
                }
            }
        }

        spawned
    }

    pub(crate) fn schedule_food_regrowth(&mut self, cell_idx: usize) {
        self.ensure_food_ecology_state();
        if cell_idx >= self.food_fertility.len()
            || !self.food_fertility[cell_idx]
            || self.food_regrowth_due_turn[cell_idx] != NO_REGROWTH_SCHEDULED
        {
            return;
        }
        let due_turn = self.turn.saturating_add(self.regrowth_delay_turns());
        self.schedule_food_regrowth_for_turn(cell_idx, due_turn);
    }

    fn ensure_food_ecology_state(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        let valid_fertility = self.food_fertility.len() == capacity;
        let valid_regrowth_due = self.food_regrowth_due_turn.len() == capacity;
        if valid_fertility && valid_regrowth_due {
            return;
        }

        self.initialize_food_ecology();
    }

    fn regrowth_delay_turns(&mut self) -> u64 {
        let interval = i64::from(self.config.food_regrowth_interval);
        let jitter = i64::from(self.config.food_regrowth_jitter);
        if jitter == 0 {
            return interval.max(1) as u64;
        }
        let offset = self.rng.random_range(-jitter..=jitter);
        (interval + offset).max(1) as u64
    }

    fn defer_food_regrowth(&mut self, cell_idx: usize) {
        let due_turn = self.turn.saturating_add(1);
        self.schedule_food_regrowth_for_turn(cell_idx, due_turn);
    }

    fn schedule_food_regrowth_for_turn(&mut self, cell_idx: usize, due_turn: u64) {
        if cell_idx >= self.food_regrowth_due_turn.len()
            || !self.food_fertility[cell_idx]
            || self.food_regrowth_due_turn[cell_idx] != NO_REGROWTH_SCHEDULED
        {
            return;
        }
        self.food_regrowth_due_turn[cell_idx] = due_turn;
        self.food_regrowth_schedule
            .entry(due_turn)
            .or_default()
            .push(cell_idx);
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

fn founder_species_id(id: OrganismId) -> SpeciesId {
    SpeciesId(id.0)
}

fn build_fertility_map(world_width: u32, seed: u64) -> Vec<bool> {
    let width = world_width as usize;
    let fertility_seed = seed ^ FOOD_FERTILITY_SEED_MIX;
    let mut fertility = Vec::with_capacity(width * width);
    for r in 0..width {
        for q in 0..width {
            let x = q as f64 * FOOD_FERTILITY_NOISE_SCALE;
            let y = r as f64 * FOOD_FERTILITY_NOISE_SCALE;
            let value = fractal_perlin_2d(x, y, fertility_seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
            fertility.push(normalized >= FOOD_FERTILITY_THRESHOLD);
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
