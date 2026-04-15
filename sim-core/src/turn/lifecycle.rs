use super::*;
use crate::metabolism::organism_passive_metabolic_energy_cost;
use sim_types::{ReproductionEvent, ReproductionFailureCause};

impl Simulation {
    pub(super) fn lifecycle_phase(
        &mut self,
    ) -> (u64, Vec<RemovedEntityPosition>, Vec<ReproductionEvent>) {
        let world_width = self.config.world_width as usize;
        let mut dead = vec![false; self.organisms.len()];
        let mut starved_positions = Vec::new();
        let mut reproduction_events = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let passive_metabolic_energy_cost =
                organism_passive_metabolic_energy_cost(&self.config, organism);
            organism.energy -= passive_metabolic_energy_cost;
            if organism.energy <= 0.0
                || organism.age_turns >= u64::from(organism.genome.lifecycle.max_organism_age)
            {
                if self.pending_actions[idx].kind == PendingActionKind::Reproduce {
                    reproduction_events.push(ReproductionEvent {
                        parent_id: organism.id,
                        parent_species_id: organism.species_id,
                        parent_age_turns: organism.age_turns,
                        parent_generation: organism.generation,
                        investment_energy: self.pending_actions[idx].reproduction_energy(),
                        parent_energy_after_event: organism.energy,
                        child_id: None,
                        failure_cause: Some(ReproductionFailureCause::ParentDied),
                    });
                }
                dead[idx] = true;
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
                self.occupancy[cell_idx] = None;
                if !self.visual_map.is_empty() {
                    self.visual_map[cell_idx] = self.visual_map_base[cell_idx];
                }
                starved_positions.push(RemovedEntityPosition {
                    entity_id: EntityId::Organism(organism.id),
                    q: organism.q,
                    r: organism.r,
                });
            }
        }

        let starvation_count = starved_positions.len() as u64;
        if starvation_count == 0 {
            return (0, starved_positions, reproduction_events);
        }

        self.compact_organism_state(&dead, None);

        (starvation_count, starved_positions, reproduction_events)
    }
}
