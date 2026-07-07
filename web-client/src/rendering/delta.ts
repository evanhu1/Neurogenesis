//! In-memory world-state transitions for the rendering module. `applyTickDelta`
//! folds a per-tick delta into the previous snapshot to produce the next
//! animated frame; `findOrganism` is the linear lookup the renderer and focus
//! logic share. Both moved here from the wire-normalization layer (`protocol.ts`)
//! because they are presentation state, not wire decoding.

import type {
  ApiMetricsSnapshot,
  MetricsSnapshot,
  TickDelta,
  WorldOrganismState,
  WorldSnapshot,
} from '../types';

/// Re-derive the per-frame metrics that depend on the live organism set:
/// `species_counts` (recounted from organisms) and the monotonic
/// `total_species_created` high-water mark. Mirrors the derivation the wire
/// normalizer applies to full snapshots so animated frames stay consistent.
function recomputeMetrics(
  metrics: ApiMetricsSnapshot,
  organisms: WorldOrganismState[],
  previousTotalSpeciesCreated: number,
): MetricsSnapshot {
  const species_counts: Record<string, number> = {};
  for (const organism of organisms) {
    const key = String(organism.species_id);
    species_counts[key] = (species_counts[key] ?? 0) + 1;
  }
  return {
    ...metrics,
    species_counts,
    total_species_created: Math.max(previousTotalSpeciesCreated, Object.keys(species_counts).length),
  };
}

export function applyTickDelta(snapshot: WorldSnapshot, delta: TickDelta): WorldSnapshot {
  const movements = new Map<number, [number, number]>();
  for (const move of delta.moves) {
    movements.set(move.id, move.to);
  }
  const facings = new Map<number, WorldOrganismState['facing']>();
  for (const update of delta.facing_updates) {
    facings.set(update.id, update.facing);
  }

  const removedOrganisms = new Set<number>();
  const removedFoods = new Set<number>();
  for (const entry of delta.removed_positions) {
    if (entry.entity_id.entity_type === 'Organism') {
      removedOrganisms.add(entry.entity_id.id);
    } else {
      removedFoods.add(entry.entity_id.id);
    }
  }

  const organisms = snapshot.organisms
    .filter((organism) => !removedOrganisms.has(organism.id))
    .map((organism) => {
      const next = movements.get(organism.id);
      return {
        ...organism,
        q: next ? next[0] : organism.q,
        r: next ? next[1] : organism.r,
        facing: facings.get(organism.id) ?? organism.facing,
        age_turns: organism.age_turns + 1,
      };
    })
    .concat(delta.spawned);

  const foods =
    removedFoods.size === 0 && delta.food_spawned.length === 0
      ? snapshot.foods
      : snapshot.foods
          .filter((food) => !removedFoods.has(food.id))
          .concat(delta.food_spawned);

  return {
    ...snapshot,
    turn: delta.turn,
    metrics: recomputeMetrics(delta.metrics, organisms, snapshot.metrics.total_species_created),
    organisms,
    foods,
  };
}

export function findOrganism(snapshot: WorldSnapshot, organismId: number) {
  return snapshot.organisms.find((item) => item.id === organismId) ?? null;
}
