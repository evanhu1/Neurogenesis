use super::*;
use sim_types::ReproductionFailureCause;

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
        self.apply_spike_hazards();
        self.apply_social_color_mortality();
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
            self.sim.visual_map[to_idx] = self.sim.visual_map[from_idx];
            self.sim.visual_map[from_idx] = self.sim.visual_map_base[from_idx];
            let organism = &mut self.sim.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
            // Dir3: moving INTO a spike cell is a self-harming move — the
            // organism still enters (and takes spike damage in
            // `apply_spike_hazards`), but the Forward is NOT marked successful,
            // so action_effectiveness penalizes blundering into a hazard and
            // rewards routing around it.
            #[cfg(feature = "instrumentation")]
            if !self.sim.spike_map[to_idx] {
                self.sim.mark_action_succeeded(resolution.actor_idx);
            }
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

    /// Social color-cyclic adjacency mortality ("social hue pressure").
    ///
    /// Each living organism takes pure damage (health loss — never energy
    /// gain/transfer) from its ≤6 hex-adjacent organisms whose body-color hue
    /// *dominates* it on the color wheel:
    ///
    /// ```text
    /// damage(o) = SOCIAL_DAMAGE * Σ_{n ∈ neighbors(o)} max(0, sin(hue_n - hue_o))
    /// ```
    ///
    /// `sin(hue_n - hue_o) > 0` ⟺ n's hue leads o's by 0..180° ⟺ n dominates o.
    /// This is antisymmetric (if n hurts o, o does not hurt n by the same
    /// pairing) — an intransitive rock-paper-scissors on the color wheel with no
    /// dominant hue — and frequency-dependent (the damage depends on the colors
    /// of whoever is around you), so the population hue can keep winding instead
    /// of converging. Death drops a corpse, exactly like spike/starvation/age
    /// deaths (reuses `mark_organism_dead`).
    ///
    /// Determinism: damage is a pure function of persisted, *pre-damage* state
    /// (post-move occupancy + each organism's body color). Neither positions nor
    /// body colors are mutated by this phase, and the per-organism damage never
    /// reads any organism's health, so the computation is order-independent. We
    /// still compute every organism's damage into a snapshot buffer first, then
    /// apply in index order, so the result can never depend on iteration order.
    /// No RNG is drawn. `SOCIAL_DAMAGE == 0.0` ⇒ every damage is 0 ⇒ no health
    /// changes and no deaths ⇒ byte-identical to baseline.
    fn apply_social_color_mortality(&mut self) {
        if SOCIAL_DAMAGE == 0.0 {
            return;
        }
        let org_count = self.sim.organisms.len();
        let world_width = self.world_width_usize as i32;

        // Snapshot: per-organism social damage, computed from pre-damage state.
        let mut social_damage = std::mem::take(&mut self.sim.turn_scratch.social_damage);
        social_damage.clear();
        social_damage.resize(org_count, 0.0_f32);

        for idx in 0..org_count {
            if self.dead_organisms[idx] {
                continue;
            }
            let organism = &self.sim.organisms[idx];
            let self_hue = sim_types::color_hue(organism.genome.lifecycle.body_color);
            let (q, r) = (organism.q, organism.r);

            let mut accum = 0.0_f32;
            for &facing in FacingDirection::ALL {
                let (nq, nr) = hex_neighbor((q, r), facing, world_width);
                let cell_idx = nr as usize * self.world_width_usize + nq as usize;
                let Some(Occupant::Organism(neighbor_id)) = self.sim.occupancy[cell_idx] else {
                    continue;
                };
                let Some(neighbor_idx) = organism_index_by_id(&self.sim.organisms, neighbor_id)
                else {
                    continue;
                };
                // A neighbor that already died earlier this commit (spike) still
                // occupies its cell here only if its corpse hasn't replaced it;
                // `kill_organism` clears occupancy on death, so a live occupant
                // is genuinely alive. Guard anyway for robustness.
                if self.dead_organisms[neighbor_idx] {
                    continue;
                }
                let neighbor_hue = sim_types::color_hue(
                    self.sim.organisms[neighbor_idx].genome.lifecycle.body_color,
                );
                // FULL antisymmetric sin (not max(0)): flow TO self is positive
                // when self DOMINATES the neighbor on the hue wheel. The pair's
                // two contributions cancel (sin(a-b) = -sin(b-a)), so the energy
                // transfer is ZERO-SUM (conservative — not "ease").
                accum += (self_hue - neighbor_hue).sin();
            }
            social_damage[idx] = SOCIAL_DAMAGE * accum;
        }

        // Apply as a zero-sum energy TRANSFER in index order: a dominated
        // organism (surrounded by leading hues) bleeds energy and starves via
        // the normal lifecycle; a dominant one gains and out-reproduces. Net
        // flow over all organisms is ~0 (antisymmetric pairs cancel); the
        // energy>=0 clamp only ever DESTROYS energy, never creates it, so this
        // is conservative / ease-safe. Zero-sum preserves the antisymmetric
        // structure a sustained intransitive cycle needs (pure damage broke it).
        for idx in 0..org_count {
            if self.dead_organisms[idx] {
                continue;
            }
            let flow = social_damage[idx];
            if flow == 0.0 {
                continue;
            }
            let organism = &mut self.sim.organisms[idx];
            organism.energy = (organism.energy + flow).max(0.0);
        }

        self.sim.turn_scratch.social_damage = social_damage;
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
        if food_idx >= self.removed_food.len() {
            // Corpses spawned earlier in this commit (spike hazards / attack
            // kills) live past the buffer sized at commit start; they are
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
        self.sim.visual_map[target_idx] = self.sim.visual_map_base[target_idx];
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
            // death paths (starvation / age / spike) still drop a corpse.
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

    fn mark_organism_dead(
        &mut self,
        organism_idx: usize,
        organism_id: OrganismId,
        cell_idx: usize,
        position: (i32, i32),
        corpse_energy: f32,
    ) {
        self.kill_organism(organism_idx, organism_id, cell_idx, position, corpse_energy, true);
    }

    /// Death bookkeeping shared by every death path. `spawn_corpse` is `true`
    /// for starvation / old-age / spike deaths (the body is left to be eaten);
    /// it is `false` only for a predation kill, where the attacker has already
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
        self.sim.visual_map[cell_idx] = self.sim.visual_map_base[cell_idx];
        if spawn_corpse {
            if let Some(corpse) = self.sim.spawn_corpse_at_cell(cell_idx, corpse_energy) {
                self.result.food_spawned.push(corpse);
            }
        }
    }

    fn finalize(&mut self, gestation_started_this_tick: &mut Vec<bool>) {
        // Gestating parents killed during commit (spike hazards or predation)
        // must still report a failed reproduction before their pending action
        // is compacted away, mirroring the starvation/old-age path in
        // lifecycle_phase.
        for (idx, dead) in self.dead_organisms.iter().enumerate() {
            if !dead || self.sim.pending_actions[idx].kind != PendingActionKind::Reproduce {
                continue;
            }
            let organism = &self.sim.organisms[idx];
            self.result.reproduction_events.push(ReproductionEvent {
                parent_id: organism.id,
                parent_species_id: organism.species_id,
                parent_age_turns: organism.age_turns,
                parent_generation: organism.generation,
                investment_energy: self.sim.pending_actions[idx].reproduction_energy(),
                parent_energy_after_event: organism.energy,
                child_id: None,
                failure_cause: Some(ReproductionFailureCause::ParentDied),
            });
        }

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
