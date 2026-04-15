use super::*;
use rand::Rng;

struct CommitPhaseContext<'a> {
    sim: &'a mut Simulation,
    intents: &'a [OrganismIntent],
    resolutions: &'a [MoveResolution],
    world_width_usize: usize,
    removed_food: Vec<bool>,
    dead_organisms: Vec<bool>,
    contingent_succeeded: Vec<bool>,
    result: CommitResult,
}

impl Simulation {
    pub(super) fn commit_phase(
        &mut self,
        world_width: i32,
        intents: &[OrganismIntent],
        resolutions: &[MoveResolution],
        gestation_started_this_tick: &mut Vec<bool>,
    ) -> CommitResult {
        CommitPhaseContext::new(self, world_width, intents, resolutions)
            .run(gestation_started_this_tick)
    }
}

impl<'a> CommitPhaseContext<'a> {
    fn new(
        sim: &'a mut Simulation,
        world_width: i32,
        intents: &'a [OrganismIntent],
        resolutions: &'a [MoveResolution],
    ) -> Self {
        let org_count = sim.organisms.len();
        let food_count = sim.foods.len();
        Self {
            sim,
            intents,
            resolutions,
            world_width_usize: world_width as usize,
            removed_food: vec![false; food_count],
            dead_organisms: vec![false; org_count],
            contingent_succeeded: vec![false; org_count],
            result: CommitResult::default(),
        }
    }

    fn run(mut self, gestation_started_this_tick: &mut Vec<bool>) -> CommitResult {
        self.apply_facing_and_action_costs();
        self.apply_moves();
        self.apply_spike_hazards();
        self.resolve_interactions();
        self.mark_wasted_contingent_actions();
        self.finalize(gestation_started_this_tick);
        self.result
    }

    fn mark_wasted_contingent_actions(&mut self) {
        if !self.sim.config.wasted_action_reward_enabled {
            return;
        }
        for (idx, intent) in self.intents.iter().enumerate() {
            if self.dead_organisms[idx] {
                continue;
            }
            let attempted_contingent = intent.wants_eat || intent.wants_attack;
            if attempted_contingent && !self.contingent_succeeded[idx] {
                self.sim.organisms[idx].contingent_action_wasted_last_turn = true;
            }
        }
    }

    fn apply_facing_and_action_costs(&mut self) {
        let action_energy_cost = self.sim.config.move_action_energy_cost;
        for (idx, intent) in self.intents.iter().enumerate() {
            debug_assert_eq!(intent.idx, idx);
            let organism = &mut self.sim.organisms[idx];
            if organism.facing != intent.facing_after_actions {
                self.result.facing_updates.push(OrganismFacing {
                    id: organism.id,
                    facing: intent.facing_after_actions,
                });
            }
            organism.facing = intent.facing_after_actions;
            let action_count = u64::from(intent.action_cost_count);
            self.result.actions_applied += action_count;
            if action_count > 0 {
                let energy_cost = action_energy_cost * intent.action_cost_count as f32;
                organism.energy -= energy_cost;
            }
        }
    }

    fn apply_moves(&mut self) {
        for resolution in self.resolutions {
            let from_idx =
                resolution.from.1 as usize * self.world_width_usize + resolution.from.0 as usize;
            let to_idx =
                resolution.to.1 as usize * self.world_width_usize + resolution.to.0 as usize;
            debug_assert_eq!(
                self.sim.occupancy[from_idx],
                Some(Occupant::Organism(resolution.actor_id))
            );
            debug_assert_eq!(self.sim.occupancy[to_idx], None);

            self.sim.occupancy[from_idx] = None;
            self.sim.occupancy[to_idx] = Some(Occupant::Organism(resolution.actor_id));
            if !self.sim.visual_map.is_empty() {
                self.sim.visual_map[to_idx] = self.sim.visual_map[from_idx];
                self.sim.visual_map[from_idx] = self.sim.visual_map_base[from_idx];
            }
            let organism = &mut self.sim.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
            self.contingent_succeeded[resolution.actor_idx] = true;
            #[cfg(feature = "instrumentation")]
            self.sim.mark_action_succeeded(resolution.actor_idx);
        }
    }

    fn apply_spike_hazards(&mut self) {
        for idx in 0..self.sim.organisms.len() {
            if self.dead_organisms[idx] {
                continue;
            }
            let (q, r, max_health, current_health, organism_id, corpse_energy) = {
                let organism = &self.sim.organisms[idx];
                (
                    organism.q,
                    organism.r,
                    organism.max_health.max(1.0),
                    organism.health.max(0.0),
                    organism.id,
                    organism.energy.max(0.0),
                )
            };
            let cell_idx = r as usize * self.world_width_usize + q as usize;
            if !self.sim.spike_map[cell_idx] {
                continue;
            }

            let spike_damage = (max_health * SPIKE_DAMAGE_FRACTION).min(current_health);
            let organism = &mut self.sim.organisms[idx];
            organism.health = (organism.health - spike_damage).max(0.0);
            organism.damage_taken_last_turn += spike_damage;

            if organism.health <= 0.0 {
                self.mark_organism_dead(idx, organism_id, cell_idx, (q, r), corpse_energy);
            }
        }
    }

    fn resolve_interactions(&mut self) {
        for (idx, intent) in self.intents.iter().enumerate() {
            if (!intent.wants_eat && !intent.wants_attack) || self.dead_organisms[idx] {
                continue;
            }
            let Some((target_q, target_r)) = intent.interaction_target else {
                continue;
            };
            let target_idx = target_r as usize * self.world_width_usize + target_q as usize;

            match self.sim.occupancy[target_idx] {
                Some(Occupant::Food(food_id)) if intent.wants_eat => {
                    self.consume_food(idx, target_idx, food_id);
                }
                Some(Occupant::Organism(prey_id)) if intent.wants_attack => {
                    self.resolve_attack_damage(idx, prey_id, target_idx);
                }
                None | Some(Occupant::Wall) => {}
                Some(Occupant::Food(_)) | Some(Occupant::Organism(_)) => {}
            }
        }
    }

    fn consume_food(&mut self, predator_idx: usize, target_idx: usize, food_id: sim_types::FoodId) {
        let Some(food_idx) = food_index_by_id(&self.sim.foods, food_id) else {
            return;
        };
        if food_idx >= self.removed_food.len() || self.removed_food[food_idx] {
            return;
        }

        let food = self.sim.foods[food_idx].clone();
        self.removed_food[food_idx] = true;
        self.sim.occupancy[target_idx] = None;
        if !self.sim.visual_map.is_empty() {
            self.sim.visual_map[target_idx] = self.sim.visual_map_base[target_idx];
        }
        if food.kind == FoodKind::Plant {
            self.sim.schedule_food_regrowth(target_idx);
        }

        let gained_energy = food.energy.max(0.0) * food_consumption_energy_fraction(food.kind);
        let predator = &mut self.sim.organisms[predator_idx];
        predator.energy += gained_energy;
        predator.consumptions_count = predator.consumptions_count.saturating_add(1);
        self.contingent_succeeded[predator_idx] = true;
        match food.kind {
            FoodKind::Plant => {
                predator.plant_consumptions_count =
                    predator.plant_consumptions_count.saturating_add(1);
            }
            FoodKind::Corpse => {
                predator.prey_consumptions_count =
                    predator.prey_consumptions_count.saturating_add(1);
            }
        }
        #[cfg(feature = "instrumentation")]
        self.sim.mark_action_succeeded(predator_idx);
        self.result.consumptions += 1;
        self.result.removed_positions.push(RemovedEntityPosition {
            entity_id: EntityId::Food(food_id),
            q: food.q,
            r: food.r,
        });
    }

    fn resolve_attack_damage(
        &mut self,
        predator_idx: usize,
        prey_id: OrganismId,
        target_idx: usize,
    ) {
        let Some(prey_idx) = organism_index_by_id(&self.sim.organisms, prey_id) else {
            return;
        };
        if predator_idx == prey_idx || self.dead_organisms[prey_idx] {
            return;
        }

        let predator_size = sim_types::get_size(&self.sim.organisms[predator_idx]).max(1.0);
        let prey_size = sim_types::get_size(&self.sim.organisms[prey_idx]).max(1.0);
        let predation_success = (predator_size / prey_size).clamp(0.0, 1.0);
        if self.sim.rng.random::<f32>() >= predation_success {
            return;
        }

        let prey = &mut self.sim.organisms[prey_idx];
        let damage = (prey.max_health * ATTACK_DAMAGE_FRACTION).min(prey.health.max(0.0));
        prey.health = (prey.health - damage).max(0.0);
        prey.damage_taken_last_turn += damage;
        let killed = prey.health <= 0.0;
        let prey_q = prey.q;
        let prey_r = prey.r;
        let prey_energy = prey.energy.max(0.0);

        self.contingent_succeeded[predator_idx] = true;
        #[cfg(feature = "instrumentation")]
        self.sim.mark_action_succeeded(predator_idx);

        if killed {
            self.result.predations += 1;
            self.mark_organism_dead(prey_idx, prey_id, target_idx, (prey_q, prey_r), prey_energy);
        }
    }

    fn mark_organism_dead(
        &mut self,
        organism_idx: usize,
        organism_id: OrganismId,
        cell_idx: usize,
        position: (i32, i32),
        corpse_energy: f32,
    ) {
        if self.dead_organisms[organism_idx] {
            return;
        }

        self.dead_organisms[organism_idx] = true;
        self.result.removed_positions.push(RemovedEntityPosition {
            entity_id: EntityId::Organism(organism_id),
            q: position.0,
            r: position.1,
        });
        self.sim.occupancy[cell_idx] = None;
        if !self.sim.visual_map.is_empty() {
            self.sim.visual_map[cell_idx] = self.sim.visual_map_base[cell_idx];
        }
        if let Some(corpse) = self.sim.spawn_corpse_at_cell(cell_idx, corpse_energy) {
            self.result.food_spawned.push(corpse);
        }
    }

    fn finalize(&mut self, gestation_started_this_tick: &mut Vec<bool>) {
        self.sim
            .compact_organism_state(&self.dead_organisms, Some(gestation_started_this_tick));

        let food_count = self.removed_food.len();
        let mut write = 0_usize;
        for read in 0..self.sim.foods.len() {
            if read >= food_count || !self.removed_food[read] {
                if write != read {
                    self.sim.foods.swap(write, read);
                }
                write += 1;
            }
        }
        self.sim.foods.truncate(write);

        self.result
            .food_spawned
            .extend(self.sim.replenish_food_supply());
        self.result.moves = self
            .resolutions
            .iter()
            .map(|resolution| OrganismMove {
                id: resolution.actor_id,
                from: resolution.from,
                to: resolution.to,
            })
            .collect();
    }
}
