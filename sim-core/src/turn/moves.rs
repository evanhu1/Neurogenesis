use super::*;

impl Simulation {
    pub(super) fn resolve_moves(
        &self,
        snapshot: &TurnSnapshot,
        occupancy: &[Option<Occupant>],
        intents: &[OrganismIntent],
    ) -> Vec<MoveResolution> {
        let w = snapshot.world_width as usize;
        let mut best_by_cell: HashMap<usize, MoveCandidate> =
            HashMap::with_capacity(intents.len().saturating_div(2).max(1));

        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * w + target.0 as usize;
            if occupancy[cell_idx].is_some() {
                continue;
            }
            let candidate = MoveCandidate {
                actor_idx: intent.idx,
                actor_id: intent.id,
                from: intent.from,
                target,
                confidence: intent.move_confidence,
            };
            match best_by_cell.get_mut(&cell_idx) {
                Some(current)
                    if compare_move_candidates(&candidate, current) == Ordering::Greater =>
                {
                    *current = candidate;
                }
                Some(_) => {}
                None => {
                    best_by_cell.insert(cell_idx, candidate);
                }
            }
        }

        let mut winners: Vec<MoveCandidate> = best_by_cell.into_values().collect();
        winners.sort_by_key(|winner| winner.actor_idx);

        winners
            .into_iter()
            .map(|winner| MoveResolution {
                actor_idx: winner.actor_idx,
                actor_id: winner.actor_id,
                from: winner.from,
                to: winner.target,
            })
            .collect()
    }
}

pub(super) fn compare_move_candidates(a: &MoveCandidate, b: &MoveCandidate) -> Ordering {
    a.confidence
        .total_cmp(&b.confidence)
        .then_with(|| b.actor_id.cmp(&a.actor_id))
}
