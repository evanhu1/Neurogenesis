use super::*;
use crate::metabolism::organism_passive_metabolic_energy_cost;

impl Simulation {
    pub(super) fn lifecycle_phase(
        &mut self,
    ) -> (u64, u64, Vec<RemovedEntityPosition>, Vec<FoodState>) {
        let world_width = self.config.world_width as usize;
        let organism_count = self.organisms.len();
        // Recycled scratch buffer (see `TurnScratch`); the commit phase only
        // takes it later, and re-clears it there, so contents cannot leak.
        // Resized lazily on the first death; no-death ticks dominate.
        let mut dead = std::mem::take(&mut self.turn_scratch.dead_organisms);
        dead.clear();
        let mut starvation_count = 0_u64;
        let mut removed_positions = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let passive_metabolic_energy_cost =
                organism_passive_metabolic_energy_cost(&self.config, organism);
            organism.energy -= passive_metabolic_energy_cost;
            // Survival is purely energy-based: an organism lives as long as it
            // has energy (no age cap). Reproduction no longer exists, so there
            // is no gestation-death path either. Starvation leaves no corpse
            // (energy <= 0, nothing to recycle).
            if organism.energy <= 0.0 {
                starvation_count += 1;
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
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
            return (0, 0, removed_positions, Vec::new());
        }

        self.compact_organism_state(&dead, None);
        self.turn_scratch.dead_organisms = dead;

        (starvation_count, 0, removed_positions, Vec::new())
    }
}
