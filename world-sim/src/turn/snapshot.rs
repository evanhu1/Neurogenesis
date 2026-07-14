use super::*;

impl Simulation {
    pub(super) fn clear_turn_transient_state(&mut self) {
        self.attack_events_last_turn.clear();
    }

    pub(super) fn compact_organism_state(&mut self, removed: &[bool]) {
        debug_assert_eq!(removed.len(), self.organisms.len());

        if !removed.iter().any(|removed_flag| *removed_flag) {
            return;
        }

        let mut write = 0_usize;
        for (read, removed_flag) in removed.iter().enumerate() {
            if !*removed_flag {
                if write != read {
                    self.organisms.swap(write, read);
                }
                write += 1;
            }
        }
        self.organisms.truncate(write);
    }

    pub(super) fn increment_age_for_survivors(&mut self) {
        for organism in &mut self.organisms {
            organism.age_turns = organism.age_turns.saturating_add(1);
        }
    }

    pub(crate) fn refresh_population_metrics(&mut self) {
        self.metrics.organisms = self.organisms.len() as u32;
    }
}
