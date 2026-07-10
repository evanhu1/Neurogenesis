use super::*;
use crate::metabolism::organism_passive_metabolic_energy_cost;
use sim_types::{ReproductionEvent, ReproductionFailureCause};

impl Simulation {
    pub(super) fn lifecycle_phase(
        &mut self,
    ) -> (
        u64,
        u64,
        Vec<RemovedEntityPosition>,
        Vec<ReproductionEvent>,
        Vec<FoodState>,
    ) {
        let world_width = self.config.world_width as usize;
        let organism_count = self.organisms.len();
        // Recycled scratch buffer (see `TurnScratch`); the commit phase only
        // takes it later, and re-clears it there, so contents cannot leak.
        // Resized lazily on the first death; no-death ticks dominate.
        let mut dead = std::mem::take(&mut self.turn_scratch.dead_organisms);
        dead.clear();
        let mut starvation_count = 0_u64;
        let mut age_death_count = 0_u64;
        let mut removed_positions = Vec::new();
        let mut reproduction_events = Vec::new();
        // (cell_idx, remaining energy) of old-age deaths; spawned as corpses
        // after the loop (the loop holds a mutable borrow of self.organisms).
        // Allocates only on ticks with age deaths.
        let mut age_corpse_spawns: Vec<(usize, f32)> = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let passive_metabolic_energy_cost =
                organism_passive_metabolic_energy_cost(&self.config, organism);
            organism.energy -= passive_metabolic_energy_cost;
            let starved = organism.energy <= 0.0;
            if starved
                || organism.age_turns >= u64::from(organism.genome.lifecycle.max_organism_age)
            {
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
                if starved {
                    starvation_count += 1;
                } else {
                    age_death_count += 1;
                    // Old-age deaths recycle the organism's remaining energy
                    // as a corpse on the freed cell, mirroring the
                    // commit-phase death path (mark_organism_dead ->
                    // spawn_corpse_at_cell). Starvation deaths (energy <= 0)
                    // leave nothing worth recycling.
                    age_corpse_spawns.push((cell_idx, organism.energy.max(0.0)));
                }
                if self.pending_actions[idx].kind == PendingActionKind::Reproduce {
                    let pending_reproduction = self.pending_reproductions[idx]
                        .as_ref()
                        .expect("Reproduce pending action must carry reproduction metadata");
                    reproduction_events.push(pending_reproduction.event(
                        organism,
                        self.pending_actions[idx].reproduction_energy(),
                        None,
                        Some(ReproductionFailureCause::ParentDied),
                    ));
                }
                if dead.is_empty() {
                    dead.resize(organism_count, false);
                }
                dead[idx] = true;
                self.occupancy[cell_idx] = None;
                removed_positions.push(RemovedEntityPosition {
                    entity_id: EntityId::Organism(organism.id),
                    q: organism.q,
                    r: organism.r,
                });
            }
        }

        if removed_positions.is_empty() {
            self.turn_scratch.dead_organisms = dead;
            return (0, 0, removed_positions, reproduction_events, Vec::new());
        }

        // Spawn old-age corpses on the cells freed above, in organism
        // iteration order so food-ID allocation stays deterministic (no RNG
        // is drawn for corpse visuals). These corpses occupy their cells
        // before the intent phase, so they are sensible/edible this tick,
        // exactly like commit-phase corpses.
        let mut food_spawned = Vec::new();
        for &(cell_idx, energy) in &age_corpse_spawns {
            if let Some(corpse) = self.spawn_corpse_at_cell(cell_idx, energy) {
                food_spawned.push(corpse);
            }
        }

        self.compact_organism_state(&dead, None);
        self.turn_scratch.dead_organisms = dead;

        (
            starvation_count,
            age_death_count,
            removed_positions,
            reproduction_events,
            food_spawned,
        )
    }
}
