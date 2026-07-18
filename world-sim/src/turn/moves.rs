use super::*;

impl Simulation {
    pub(super) fn resolve_moves(&mut self, intents: &[OrganismIntent]) -> Vec<MoveResolution> {
        let width = self.config.world_width as usize;
        let mut candidates = std::mem::take(&mut self.turn_scratch.move_candidates);
        candidates.clear();
        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * width + target.0 as usize;
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
        resolutions.extend(candidates.iter().map(|(_, winner)| MoveResolution {
            actor_idx: winner.actor_idx,
            actor_id: winner.actor_id,
            from: winner.from,
            to: winner.target,
        }));
        self.turn_scratch.move_candidates = candidates;
        resolutions
    }
}
