use super::*;

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
            if self.occupancy[cell_idx].is_some() {
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

        // Winners stay in cell-index order — already fully deterministic, so
        // no re-sort is needed. Commit order is also irrelevant: target cells
        // are pairwise distinct (dedup above), from-cells are pairwise
        // distinct (one candidate per actor, one actor per cell), and no
        // winner's from can equal another winner's target because every
        // target was unoccupied at the snapshot above while every from was
        // occupied by its actor. Each applied move therefore touches only
        // cells and organism state no other move reads or writes.
        let mut resolutions = std::mem::take(&mut self.turn_scratch.move_resolutions);
        resolutions.clear();
        resolutions.extend(candidates.iter().map(|(_, winner)| MoveResolution {
            actor_idx: winner.actor_idx,
            actor_id: winner.actor_id,
            from: winner.from,
            to: winner.target,
        }));

        // Return the scratch buffer to the simulation for reuse next tick.
        self.turn_scratch.move_candidates = candidates;
        resolutions
    }
}
