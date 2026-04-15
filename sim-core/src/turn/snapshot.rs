use super::*;

impl Simulation {
    pub(super) fn reconcile_pending_actions(&mut self) {
        if self.pending_actions.len() != self.organisms.len() {
            self.pending_actions
                .resize(self.organisms.len(), PendingActionState::default());
        }
    }

    pub(super) fn reconcile_reward_ledgers(&mut self) {
        if self.reward_ledgers.len() != self.organisms.len() {
            self.reward_ledgers
                .resize(self.organisms.len(), crate::RewardLedger::default());
        }
    }

    pub(super) fn clear_turn_transient_state(&mut self) {
        for organism in self.organisms.iter_mut() {
            organism.damage_taken_last_turn = 0.0;
            organism.contingent_action_wasted_last_turn = false;
        }
    }

    pub(super) fn compact_organism_state(
        &mut self,
        removed: &[bool],
        mut gestation_started_this_tick: Option<&mut Vec<bool>>,
    ) {
        debug_assert_eq!(removed.len(), self.organisms.len());
        debug_assert_eq!(self.pending_actions.len(), self.organisms.len());
        debug_assert_eq!(self.reward_ledgers.len(), self.organisms.len());
        if let Some(skip) = gestation_started_this_tick.as_ref() {
            debug_assert_eq!(skip.len(), self.organisms.len());
        }

        if !removed.iter().any(|removed_flag| *removed_flag) {
            return;
        }

        let mut write = 0_usize;
        for read in 0..self.organisms.len() {
            if !removed[read] {
                if write != read {
                    self.organisms.swap(write, read);
                    self.pending_actions.swap(write, read);
                    self.reward_ledgers.swap(write, read);
                    if let Some(ref mut skip) = gestation_started_this_tick {
                        skip.swap(write, read);
                    }
                }
                write += 1;
            }
        }
        self.organisms.truncate(write);
        self.pending_actions.truncate(write);
        self.reward_ledgers.truncate(write);
        if let Some(skip) = gestation_started_this_tick {
            skip.truncate(write);
        }
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
