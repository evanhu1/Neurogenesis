use super::world::{hash_2d, noise_2d};
use super::*;
use sim_config::food_ecology_policy;

impl Simulation {
    pub(crate) fn initialize_food_ecology(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        self.food_fertility = build_fertility_map(
            self.config.world_width,
            self.seed,
            self.config.food_fertility_threshold,
            self.config.food_fertility_jitter_strength,
        );
        debug_assert_eq!(self.food_fertility.len(), capacity);
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                self.food_fertility[idx] = false;
            }
        }
        self.food_regrowth_due_turn = vec![NO_REGROWTH_SCHEDULED; capacity];
        self.food_regrowth_schedule.clear();
    }

    pub(crate) fn seed_initial_food_supply(&mut self) {
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
        let mut spawned = Vec::new();
        // Due-but-blocked cells (occupied target) are collected here and
        // re-inserted once as a single batch at `turn + 1`, recycling a
        // drained schedule Vec, instead of a per-cell
        // `entry().or_default().push()` plus Vec alloc/dealloc every turn.
        // Cells are kept in processing order so plant-spawn RNG draws (color
        // jitter) replay identically to the previous per-cell deferral.
        let mut blocked: Vec<usize> = Vec::new();
        while let Some(entry) = self.food_regrowth_schedule.first_entry() {
            if *entry.key() > self.turn {
                break;
            }
            let (due_turn, mut cell_indices) = entry.remove_entry();
            cell_indices.retain(|&cell_idx| {
                // Invariant: a cell has at most one outstanding schedule entry
                // (inserts require an unscheduled slot), so the slot always
                // matches its entry's due turn.
                debug_assert_eq!(self.food_regrowth_due_turn[cell_idx], due_turn);
                // Invariant: regrowth is only scheduled for fertile cells and
                // fertility never decays after initialize_food_ecology.
                debug_assert!(self.food_fertility[cell_idx]);
                if self.occupancy[cell_idx].is_none() {
                    self.food_regrowth_due_turn[cell_idx] = NO_REGROWTH_SCHEDULED;
                    if let Some(food) = self.spawn_food_at_cell(cell_idx) {
                        spawned.push(food);
                    }
                    false
                } else {
                    true
                }
            });
            if !cell_indices.is_empty() {
                if blocked.is_empty() {
                    // Recycle the drained entry's allocation for the batch.
                    blocked = cell_indices;
                } else {
                    blocked.append(&mut cell_indices);
                }
            }
        }

        if !blocked.is_empty() {
            let due_turn = self.turn.saturating_add(1);
            for &cell_idx in &blocked {
                self.food_regrowth_due_turn[cell_idx] = due_turn;
            }
            match self.food_regrowth_schedule.entry(due_turn) {
                std::collections::btree_map::Entry::Occupied(mut occupied) => {
                    occupied.get_mut().append(&mut blocked);
                }
                std::collections::btree_map::Entry::Vacant(vacant) => {
                    vacant.insert(blocked);
                }
            }
        }

        spawned
    }

    pub(crate) fn schedule_food_regrowth(&mut self, cell_idx: usize) {
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

    fn regrowth_delay_turns(&mut self) -> u64 {
        let interval = i64::from(self.config.food_regrowth_interval);
        let jitter = i64::from(self.config.food_regrowth_jitter);
        if jitter == 0 {
            return interval.max(1) as u64;
        }
        let offset = self.rng.random_range(-jitter..=jitter);
        (interval + offset).max(1) as u64
    }

    fn schedule_food_regrowth_for_turn(&mut self, cell_idx: usize, due_turn: u64) {
        // The sole caller (schedule_food_regrowth) checks fertility and an
        // unscheduled slot before calling.
        debug_assert!(self.food_fertility[cell_idx]);
        debug_assert_eq!(self.food_regrowth_due_turn[cell_idx], NO_REGROWTH_SCHEDULED);
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
        let mut visual = sim_types::food_visual(kind);
        if kind == FoodKind::Plant {
            visual =
                crate::jitter_visual_rgb_uniform(visual, &mut self.rng, crate::PLANT_COLOR_JITTER);
        }
        let food = FoodState {
            id: self.alloc_food_id(),
            q,
            r,
            energy,
            kind,
            visual,
        };
        self.occupancy[cell_idx] = Some(Occupant::Food(food.id));
        self.visual_map[cell_idx] = food.visual;
        self.foods.push(food.clone());
        Some(food)
    }

    fn alloc_food_id(&mut self) -> FoodId {
        let id = FoodId(self.next_food_id);
        self.next_food_id += 1;
        id
    }
}

fn build_fertility_map(
    world_width: u32,
    seed: u64,
    fertility_threshold: f32,
    fertility_jitter_strength: f32,
) -> Vec<bool> {
    let policy = food_ecology_policy();
    let width = world_width as usize;
    let fertility_seed = seed ^ policy.fertility_seed_mix;
    let jitter_seed = fertility_seed ^ policy.fertility_jitter_seed_mix;
    let mut fertility = Vec::with_capacity(width * width);
    for r in 0..width {
        for q in 0..width {
            let x = q as f64 * policy.fertility_noise_scale;
            let y = r as f64 * policy.fertility_noise_scale;
            let value = noise_2d(x, y, fertility_seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
            let jitter = cell_jitter(q as i64, r as i64, jitter_seed, fertility_jitter_strength);
            let jittered = (normalized * jitter).clamp(0.0, 1.0);
            fertility.push(jittered >= f64::from(fertility_threshold));
        }
    }
    fertility
}

fn cell_jitter(x: i64, y: i64, seed: u64, strength: f32) -> f64 {
    const MAX_U53: f64 = ((1_u64 << 53) - 1) as f64;
    let sample = (hash_2d(x, y, seed) >> 11) as f64 / MAX_U53;
    let base_jitter = 0.5 + sample;
    let extra_hole_punch = sample.powf(f64::from((strength - 1.0).max(0.0)));
    base_jitter * extra_hole_punch
}
