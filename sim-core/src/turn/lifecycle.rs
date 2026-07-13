use super::*;
use crate::metabolism::organism_passive_metabolic_energy_cost;

impl Simulation {
    pub(super) fn lifecycle_phase(
        &mut self,
    ) -> (
        u64,
        u64,
        Vec<RemovedEntityPosition>,
        Vec<FoodState>,
        LifecycleEnergyFlow,
    ) {
        let world_width = self.config.world_width as usize;
        let organism_count = self.organisms.len();
        // Recycled scratch buffer (see `TurnScratch`); the commit phase only
        // takes it later, and re-clears it there, so contents cannot leak.
        // Resized lazily on the first death; no-death ticks dominate.
        let mut dead = std::mem::take(&mut self.turn_scratch.dead_organisms);
        dead.clear();
        let mut starvation_count = 0_u64;
        let mut removed_positions = Vec::new();
        let mut energy_flow = LifecycleEnergyFlow::default();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let passive_metabolic_energy_cost =
                organism_passive_metabolic_energy_cost(&self.config, organism);
            assert!(
                passive_metabolic_energy_cost.is_finite(),
                "energy ledger: nonfinite passive metabolism for organism {:?} at turn {}",
                organism.id,
                self.turn.saturating_add(1)
            );
            let energy_before_metabolism = organism.energy;
            organism.energy -= passive_metabolic_energy_cost;
            assert!(
                organism.energy.is_finite(),
                "energy ledger: passive metabolism produced nonfinite energy for organism {:?} at turn {}",
                organism.id,
                self.turn.saturating_add(1)
            );
            energy_flow.passive_metabolism_energy +=
                f64::from(energy_before_metabolism) - f64::from(organism.energy);
            // Survival is purely energy-based: an organism lives as long as it
            // has energy (no age cap). Reproduction no longer exists, so there
            // is no gestation-death path either. Starvation leaves no corpse
            // (energy <= 0, nothing to recycle).
            if organism.energy <= 0.0 {
                starvation_count += 1;
                let removed_energy = f64::from(organism.energy);
                energy_flow.unrecycled_energy_removed += removed_energy;
                // Removing signed energy changes the represented total by its
                // negation. Starvation normally discards a small energy debt,
                // so this adjustment is positive and visible rather than
                // silently manufacturing energy.
                energy_flow.removal_adjustment -= removed_energy;
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
            return (0, 0, removed_positions, Vec::new(), energy_flow);
        }

        self.compact_organism_state(&dead, None);
        self.turn_scratch.dead_organisms = dead;

        (
            starvation_count,
            0,
            removed_positions,
            Vec::new(),
            energy_flow,
        )
    }
}
