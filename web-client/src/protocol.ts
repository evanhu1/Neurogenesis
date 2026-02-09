import type { OrganismId, OrganismState, TickDelta, WorldSnapshot } from './types';

export function unwrapId(id: OrganismId | number | { 0: number }) {
  if (typeof id === 'number') return id;
  return id[0];
}

export function applyTickDelta(snapshot: WorldSnapshot, delta: TickDelta): WorldSnapshot {
  const movements = new Map<number, [number, number]>();
  for (const move of delta.moves) {
    movements.set(unwrapId(move.id), move.to);
  }

  const organisms = snapshot.organisms.map((organism: OrganismState) => {
    const next = movements.get(unwrapId(organism.id));
    if (!next) return organism;
    return { ...organism, q: next[0], r: next[1] };
  });

  return {
    ...snapshot,
    turn: delta.turn,
    metrics: delta.metrics,
    organisms,
  };
}

export function findBrain(snapshot: WorldSnapshot, organismId: number) {
  const organism = snapshot.organisms.find((item) => unwrapId(item.id) === organismId);
  return organism?.brain ?? null;
}
