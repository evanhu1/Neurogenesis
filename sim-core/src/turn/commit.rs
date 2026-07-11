use super::*;

struct CommitPhaseContext<'a> {
    sim: &'a mut Simulation,
    intents: &'a [OrganismIntent],
    resolutions: &'a [MoveResolution],
    world_width_usize: usize,
    removed_food: Vec<bool>,
    dead_organisms: Vec<bool>,
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
        let mut removed_food = std::mem::take(&mut sim.turn_scratch.removed_food);
        removed_food.clear();
        removed_food.resize(food_count, false);
        let mut dead_organisms = std::mem::take(&mut sim.turn_scratch.dead_organisms);
        dead_organisms.clear();
        dead_organisms.resize(org_count, false);
        Self {
            sim,
            intents,
            resolutions,
            world_width_usize: world_width as usize,
            removed_food,
            dead_organisms,
            result: CommitResult::default(),
        }
    }

    fn run(mut self, gestation_started_this_tick: &mut Vec<bool>) -> CommitResult {
        self.apply_facing_and_action_costs();
        self.apply_moves();
        self.resolve_interactions();
        self.finalize(gestation_started_this_tick);
        self.result
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
            if intent.took_action {
                self.result.actions_applied += 1;
                organism.energy -= action_energy_cost;
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
            let organism = &mut self.sim.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
            #[cfg(feature = "instrumentation")]
            self.sim.mark_action_succeeded(resolution.actor_idx);
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
                Some(Occupant::Organism(prey_id))
                    if self.sim.config.predation_enabled && intent.wants_attack =>
                {
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
        if food_idx >= self.removed_food.len() {
            // Corpses spawned earlier in this commit by attack kills live past
            // the buffer sized at commit start; they are
            // edible, so grow lazily instead of conflating "spawned this tick"
            // with "already consumed".
            self.removed_food.resize(self.sim.foods.len(), false);
        }
        // A food can never be consumed twice in one tick: it occupies exactly
        // one cell (occupancy<->foods bijection) and the first consumption
        // clears that cell's occupancy below, so no later intent can resolve
        // the same food id.
        debug_assert!(!self.removed_food[food_idx]);

        let food = self.sim.foods[food_idx].clone();
        self.removed_food[food_idx] = true;
        self.sim.occupancy[target_idx] = None;
        if food.kind == FoodKind::Plant {
            self.sim.schedule_food_regrowth(target_idx);
        }

        let gained_energy = food.energy.max(0.0);
        let predator = &mut self.sim.organisms[predator_idx];
        predator.energy += gained_energy;
        predator.consumptions_count = predator.consumptions_count.saturating_add(1);
        match food.kind {
            FoodKind::Plant => {
                predator.plant_consumptions_count =
                    predator.plant_consumptions_count.saturating_add(1);
                self.result.plant_consumptions += 1;
            }
            FoodKind::Corpse => {
                predator.prey_consumptions_count =
                    predator.prey_consumptions_count.saturating_add(1);
            }
        }
        #[cfg(feature = "instrumentation")]
        {
            let predator = &self.sim.organisms[predator_idx];
            let total = predator.consumptions_count;
            let plant = predator.plant_consumptions_count;
            let prey = predator.prey_consumptions_count;
            self.sim
                .record_consumption_counts(predator_idx, total, plant, prey);
            self.sim.mark_action_succeeded(predator_idx);
        }
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
        if let Some(pool_count) = self.sim.cross_pool_predation_pool_count {
            let predator_pool = self.sim.organisms[predator_idx].species_id.0 as usize % pool_count;
            let prey_pool = self.sim.organisms[prey_idx].species_id.0 as usize % pool_count;
            if predator_pool == prey_pool {
                return;
            }
        }

        // Larger attackers land hits on smaller prey reliably; punching up is
        // proportionally unlikely. The roll is a deterministic hash (see
        // `deterministic_predation_sample`); a failed roll is a wasted
        // contingent action. A non-lethal hit yields no energy — the predator
        // is only fed by a kill (see the `killed` branch below), where it
        // consumes the prey directly. Energy is conserved: the meal is the
        // prey's own stash times CORPSE_ENERGY_RETENTION, identical to what
        // eating the prey's corpse would have transferred, so predation is an
        // energy-recycling path, not a source.
        let predator = &self.sim.organisms[predator_idx];
        let predator_id = predator.id;
        let predator_size = sim_types::get_size(predator).max(1.0);
        // Damage scales with the attacker's own size, so bigger predators hit
        // harder regardless of how tough the prey is.
        let predator_max_health = predator.max_health.max(0.0);
        let prey_size = sim_types::get_size(&self.sim.organisms[prey_idx]).max(1.0);
        let predation_success = (predator_size / prey_size).clamp(0.0, 1.0);
        let sample =
            deterministic_predation_sample(self.sim.seed, self.sim.turn, predator_id, prey_id);
        if sample >= predation_success {
            return;
        }

        let prey = &mut self.sim.organisms[prey_idx];
        let damage = (predator_max_health * ATTACK_DAMAGE_FRACTION).min(prey.health.max(0.0));
        prey.health = (prey.health - damage).max(0.0);
        prey.damage_taken_last_turn += damage;
        let killed = prey.health <= 0.0;
        let prey_q = prey.q;
        let prey_r = prey.r;
        let prey_energy = prey.energy.max(0.0);

        #[cfg(feature = "instrumentation")]
        self.sim.mark_action_succeeded(predator_idx);

        if killed {
            self.result.predations += 1;

            // The kill is the meal: the attacker directly consumes the prey.
            // Energy gained matches eating the prey's corpse (prey energy times
            // CORPSE_ENERGY_RETENTION), so total energy is conserved and no
            // corpse is left behind. This turns the existing predation behavior
            // into prey_consumption directly, establishing a predator niche.
            let gained_energy = prey_energy * CORPSE_ENERGY_RETENTION;
            let predator = &mut self.sim.organisms[predator_idx];
            predator.energy += gained_energy;
            predator.consumptions_count = predator.consumptions_count.saturating_add(1);
            predator.prey_consumptions_count = predator.prey_consumptions_count.saturating_add(1);

            self.result.consumptions += 1;
            #[cfg(feature = "instrumentation")]
            {
                let predator = &self.sim.organisms[predator_idx];
                let total = predator.consumptions_count;
                let plant = predator.plant_consumptions_count;
                let prey = predator.prey_consumptions_count;
                self.sim
                    .record_consumption_counts(predator_idx, total, plant, prey);
            }

            // Suppress the corpse: the prey was eaten, not left behind. Other
            // starvation and age deaths still drop a corpse.
            self.kill_organism(
                prey_idx,
                prey_id,
                target_idx,
                (prey_q, prey_r),
                prey_energy,
                false,
            );
        }
    }

    /// Commit-phase death bookkeeping. `spawn_corpse` is false for a predation
    /// kill, where the attacker has already
    /// consumed the prey's energy directly (see `resolve_attack_damage`), so no
    /// corpse is left behind. Corpse spawning draws no RNG, so suppressing it on
    /// the predation path does not perturb the deterministic RNG stream.
    fn kill_organism(
        &mut self,
        organism_idx: usize,
        organism_id: OrganismId,
        cell_idx: usize,
        position: (i32, i32),
        corpse_energy: f32,
        spawn_corpse: bool,
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
        if spawn_corpse {
            if let Some(corpse) = self.sim.spawn_corpse_at_cell(cell_idx, corpse_energy) {
                self.result.food_spawned.push(corpse);
            }
        }
    }

    fn finalize(&mut self, gestation_started_this_tick: &mut Vec<bool>) {
        self.sim
            .compact_organism_state(&self.dead_organisms, Some(gestation_started_this_tick));

        // Mirror compact_organism_state's early-out: skip the per-food scan
        // entirely on ticks with no consumptions.
        if self.removed_food.iter().any(|&removed| removed) {
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
        }

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

        // Return the scratch buffers to the simulation for reuse next tick.
        self.sim.turn_scratch.removed_food = std::mem::take(&mut self.removed_food);
        self.sim.turn_scratch.dead_organisms = std::mem::take(&mut self.dead_organisms);
    }
}
