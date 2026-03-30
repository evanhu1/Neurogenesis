use super::*;

impl Simulation {
    pub(super) fn build_turn_snapshot(&self) -> TurnSnapshot {
        let len = self.organisms.len();
        let mut organism_ids = Vec::with_capacity(len);
        let mut organism_states = Vec::with_capacity(len);

        for organism in &self.organisms {
            organism_ids.push(organism.id);
            organism_states.push(SnapshotOrganismState {
                q: organism.q,
                r: organism.r,
                facing: organism.facing,
            });
        }

        TurnSnapshot {
            world_width: self.config.world_width as i32,
            organism_count: len,
            organism_ids,
            organism_states,
        }
    }

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

    pub(super) fn clear_reward_ledgers(&mut self) {
        for ledger in &mut self.reward_ledgers {
            ledger.clear();
        }
    }

    pub(super) fn clear_damage_state(&mut self) {
        for organism in &mut self.organisms {
            organism.damage_taken_last_turn = 0.0;
        }
    }

    pub(super) fn compact_organism_state(
        &mut self,
        removed: &[bool],
        skip_pending_action_decrement: Option<&mut Vec<bool>>,
    ) {
        debug_assert_eq!(removed.len(), self.organisms.len());
        debug_assert_eq!(self.pending_actions.len(), self.organisms.len());
        debug_assert_eq!(self.reward_ledgers.len(), self.organisms.len());
        if let Some(skip) = skip_pending_action_decrement.as_ref() {
            debug_assert_eq!(skip.len(), self.organisms.len());
        }

        if !removed.iter().any(|removed_flag| *removed_flag) {
            return;
        }

        let mut survivors = Vec::with_capacity(self.organisms.len());
        let mut survivor_pending_actions = Vec::with_capacity(self.pending_actions.len());
        let mut survivor_reward_ledgers = Vec::with_capacity(self.reward_ledgers.len());
        let mut survivor_skip = skip_pending_action_decrement
            .as_ref()
            .map(|skip| Vec::with_capacity(skip.len()));

        for (idx, ((organism, pending_action), reward_ledger)) in self
            .organisms
            .drain(..)
            .zip(self.pending_actions.drain(..))
            .zip(self.reward_ledgers.drain(..))
            .enumerate()
        {
            if removed[idx] {
                continue;
            }
            survivors.push(organism);
            survivor_pending_actions.push(pending_action);
            survivor_reward_ledgers.push(reward_ledger);
            if let Some(skip) = skip_pending_action_decrement.as_ref() {
                survivor_skip
                    .as_mut()
                    .expect("skip sidecar capacity should be initialized")
                    .push(skip[idx]);
            }
        }

        self.organisms = survivors;
        self.pending_actions = survivor_pending_actions;
        self.reward_ledgers = survivor_reward_ledgers;
        if let Some(skip) = skip_pending_action_decrement {
            *skip = survivor_skip.unwrap_or_default();
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
