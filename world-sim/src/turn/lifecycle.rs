use super::*;

impl Simulation {
    /// Charge the single universal lifetime cost after every survivor has
    /// sensed, acted, resolved interactions, aged, and applied plasticity.
    /// Organisms at one energy complete this tick and then die at exactly zero.
    pub(super) fn drain_energy_at_end_of_tick(
        &mut self,
    ) -> (u64, Vec<RemovedEntityPosition>, LifecycleEnergyFlow) {
        let world_width = self.config.world_width as usize;
        let organism_count = self.organisms.len();
        let mut dead = std::mem::take(&mut self.turn_scratch.dead_organisms);
        dead.clear();
        dead.resize(organism_count, false);
        let mut starvation_count = 0_u64;
        let mut removed_positions = Vec::new();

        let mut drained = 0_u64;
        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            if organism.energy > 0 {
                organism.energy -= 1;
                drained = drained.saturating_add(1);
            }
            if organism.energy == 0 {
                starvation_count = starvation_count.saturating_add(1);
                dead[idx] = true;
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
                self.occupancy[cell_idx] = None;
                removed_positions.push(RemovedEntityPosition {
                    entity_id: EntityId::Organism(organism.id),
                    q: organism.q,
                    r: organism.r,
                });
            }
        }

        let energy_flow = LifecycleEnergyFlow {
            tick_drain_energy: drained as f64,
        };
        if starvation_count > 0 {
            self.compact_organism_state(&dead);
        }
        self.turn_scratch.dead_organisms = dead;
        (starvation_count, removed_positions, energy_flow)
    }
}
