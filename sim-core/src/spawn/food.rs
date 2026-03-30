use super::world::{fractal_perlin_2d, hash_2d};
use super::*;
use sim_config::food_ecology_policy;

impl Simulation {
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
                } else if self.occupancy[cell_idx].is_some() {
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

    pub(crate) fn spawn_corpse_at_cell(
        &mut self,
        cell_idx: usize,
        energy: f32,
    ) -> Option<FoodState> {
        self.spawn_food_with_kind_at_cell(cell_idx, energy, FoodKind::Corpse)
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
        self.spawn_food_with_kind_at_cell(cell_idx, self.config.food_energy, FoodKind::Plant)
    }

    fn spawn_food_with_kind_at_cell(
        &mut self,
        cell_idx: usize,
        energy: f32,
        kind: FoodKind,
    ) -> Option<FoodState> {
        if self.occupancy[cell_idx].is_some() || energy <= 0.0 {
            return None;
        }
        let width = self.config.world_width as usize;
        let q = (cell_idx % width) as i32;
        let r = (cell_idx / width) as i32;
        let food = FoodState {
            id: self.alloc_food_id(),
            q,
            r,
            energy,
            kind,
        };
        self.occupancy[cell_idx] = Some(Occupant::Food(food.id));
        self.foods.push(food.clone());
        Some(food)
    }

    fn alloc_food_id(&mut self) -> FoodId {
        let id = FoodId(self.next_food_id);
        self.next_food_id += 1;
        id
    }
}

fn build_fertility_map(world_width: u32, seed: u64) -> Vec<bool> {
    let policy = food_ecology_policy();
    let width = world_width as usize;
    let fertility_seed = seed ^ policy.fertility_seed_mix;
    let jitter_seed = fertility_seed ^ policy.fertility_jitter_seed_mix;
    let mut fertility = Vec::with_capacity(width * width);
    for r in 0..width {
        for q in 0..width {
            let x = q as f64 * policy.fertility_noise_scale;
            let y = r as f64 * policy.fertility_noise_scale;
            let value = fractal_perlin_2d(x, y, fertility_seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
            let jitter = cell_jitter(q as i64, r as i64, jitter_seed);
            let jittered = (normalized * jitter).clamp(0.0, 1.0);
            fertility.push(jittered >= policy.fertility_threshold);
        }
    }
    fertility
}

fn cell_jitter(x: i64, y: i64, seed: u64) -> f64 {
    const MAX_U53: f64 = ((1_u64 << 53) - 1) as f64;
    let sample = (hash_2d(x, y, seed) >> 11) as f64 / MAX_U53;
    0.5 + sample
}
