use super::*;

const MOVE_UNKNOWN: u8 = 0;
const MOVE_VISITING: u8 = 1;
const MOVE_SUCCEEDS: u8 = 2;
const MOVE_BLOCKED: u8 = 3;
const NO_INDEX: usize = usize::MAX;

impl Simulation {
    pub(super) fn resolve_moves(&mut self, intents: &[OrganismIntent]) -> Vec<MoveResolution> {
        let w = self.config.world_width as usize;
        let mut candidates = std::mem::take(&mut self.turn_scratch.move_candidates);
        candidates.clear();

        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * w + target.0 as usize;
            let target_can_vacate =
                matches!(self.occupancy[cell_idx], None | Some(Occupant::Organism(_)));
            if if self.config.compositional_actions_enabled {
                !target_can_vacate
            } else {
                self.occupancy[cell_idx].is_some()
            } {
                continue;
            }
            candidates.push((
                cell_idx,
                MoveCandidate {
                    actor_idx: intent.idx,
                    actor_id: intent.id,
                    from: intent.from,
                    target,
                    confidence: intent.move_confidence,
                },
            ));
        }

        // One deterministic winner per target cell.
        candidates.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| {
                b.1.confidence
                    .total_cmp(&a.1.confidence)
                    .then_with(|| a.1.actor_id.cmp(&b.1.actor_id))
            })
        });
        candidates.dedup_by_key(|candidate| candidate.0);

        let mut resolutions = std::mem::take(&mut self.turn_scratch.move_resolutions);
        resolutions.clear();

        if !self.config.compositional_actions_enabled {
            resolutions.extend(candidates.iter().map(|(_, winner)| MoveResolution {
                actor_idx: winner.actor_idx,
                actor_id: winner.actor_id,
                from: winner.from,
                to: winner.target,
            }));
        } else {
            // A winner may enter an occupied cell exactly when the occupant is
            // itself a successful winner moving out. Resolve the resulting
            // dependency graph against the same snapshot. A back-edge is a
            // legal simultaneous cycle (including a two-organism swap).
            let mut actor_by_cell = std::mem::take(&mut self.turn_scratch.move_actor_by_cell);
            actor_by_cell.clear();
            actor_by_cell.resize(w * w, NO_INDEX);
            for intent in intents {
                let cell_idx = intent.from.1 as usize * w + intent.from.0 as usize;
                actor_by_cell[cell_idx] = intent.idx;
            }

            let mut winner_by_actor = std::mem::take(&mut self.turn_scratch.move_winner_by_actor);
            winner_by_actor.clear();
            winner_by_actor.resize(intents.len(), NO_INDEX);
            for (candidate_idx, (_, winner)) in candidates.iter().enumerate() {
                winner_by_actor[winner.actor_idx] = candidate_idx;
            }

            let mut status = std::mem::take(&mut self.turn_scratch.move_dependency_status);
            status.clear();
            status.resize(candidates.len(), MOVE_UNKNOWN);
            for candidate_idx in 0..candidates.len() {
                if move_dependency_succeeds(
                    candidate_idx,
                    &candidates,
                    &self.occupancy,
                    &actor_by_cell,
                    &winner_by_actor,
                    &mut status,
                ) {
                    let winner = candidates[candidate_idx].1;
                    resolutions.push(MoveResolution {
                        actor_idx: winner.actor_idx,
                        actor_id: winner.actor_id,
                        from: winner.from,
                        to: winner.target,
                    });
                }
            }

            self.turn_scratch.move_actor_by_cell = actor_by_cell;
            self.turn_scratch.move_winner_by_actor = winner_by_actor;
            self.turn_scratch.move_dependency_status = status;
        }

        self.turn_scratch.move_candidates = candidates;
        resolutions
    }
}

fn move_dependency_succeeds(
    candidate_idx: usize,
    candidates: &[(usize, MoveCandidate)],
    occupancy: &[Option<Occupant>],
    actor_by_cell: &[usize],
    winner_by_actor: &[usize],
    status: &mut [u8],
) -> bool {
    match status[candidate_idx] {
        MOVE_SUCCEEDS => return true,
        MOVE_BLOCKED => return false,
        // Re-entering a visiting node closes a dependency cycle in which every
        // target is uniquely won, so all members can vacate simultaneously.
        MOVE_VISITING => return true,
        MOVE_UNKNOWN => {}
        _ => unreachable!(),
    }
    status[candidate_idx] = MOVE_VISITING;
    let target_cell_idx = candidates[candidate_idx].0;
    let succeeds = match occupancy[target_cell_idx] {
        None => true,
        Some(Occupant::Organism(_)) => {
            let occupant_actor = actor_by_cell[target_cell_idx];
            if occupant_actor == NO_INDEX {
                false
            } else {
                let occupant_winner = winner_by_actor[occupant_actor];
                occupant_winner != NO_INDEX
                    && move_dependency_succeeds(
                        occupant_winner,
                        candidates,
                        occupancy,
                        actor_by_cell,
                        winner_by_actor,
                        status,
                    )
            }
        }
        Some(Occupant::Wall) => false,
    };
    status[candidate_idx] = if succeeds {
        MOVE_SUCCEEDS
    } else {
        MOVE_BLOCKED
    };
    succeeds
}
