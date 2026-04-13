use super::*;

impl Simulation {
    pub(super) fn resolve_moves(
        &self,
        world_width: i32,
        occupancy: &[Option<Occupant>],
        intents: &[OrganismIntent],
    ) -> Vec<MoveResolution> {
        let w = world_width as usize;
        let mut candidates: Vec<(usize, MoveCandidate)> = Vec::new();

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

        // Sort by cell index, then best candidate first within each cell.
        // dedup_by_key keeps the first (best) candidate per cell.
        candidates.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| {
                b.1.confidence
                    .total_cmp(&a.1.confidence)
                    .then_with(|| a.1.actor_id.cmp(&b.1.actor_id))
            })
        });
        candidates.dedup_by_key(|c| c.0);

        // Sort winners by actor_idx for deterministic commit ordering.
        candidates.sort_unstable_by_key(|c| c.1.actor_idx);

        candidates
            .into_iter()
            .map(|(_, winner)| MoveResolution {
                actor_idx: winner.actor_idx,
                actor_id: winner.actor_id,
                from: winner.from,
                to: winner.target,
            })
            .collect()
    }
}
