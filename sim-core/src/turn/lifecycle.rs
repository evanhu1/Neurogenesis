use super::*;

impl Simulation {
    pub(super) fn lifecycle_phase(&mut self) -> (u64, Vec<RemovedEntityPosition>) {
        let max_age = self.config.max_organism_age as u64;
        let world_width = self.config.world_width as usize;
        let mut dead = vec![false; self.organisms.len()];
        let mut starved_positions = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let passive_metabolic_energy_cost =
                organism_passive_metabolic_energy_cost(&self.config, organism);
            organism.energy -= passive_metabolic_energy_cost;
            organism.health = (organism.health + organism_health_regeneration(organism))
                .min(organism.max_health.max(1.0));
            if organism.energy <= 0.0 || organism.age_turns >= max_age {
                dead[idx] = true;
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
                self.occupancy[cell_idx] = None;
                starved_positions.push(RemovedEntityPosition {
                    entity_id: EntityId::Organism(organism.id),
                    q: organism.q,
                    r: organism.r,
                });
            }
        }

        let starvation_count = starved_positions.len() as u64;
        if starvation_count == 0 {
            return (0, starved_positions);
        }

        self.compact_organism_state(&dead, None);

        (starvation_count, starved_positions)
    }
}
