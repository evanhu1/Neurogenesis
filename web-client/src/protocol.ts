import type { FoodId, OrganismId, OrganismState, TickDelta, WorldSnapshot } from './types';

export function unwrapId(id: OrganismId | FoodId | number | { 0: number }) {
  if (typeof id === 'number') return id;
  return id[0];
}

export function applyTickDelta(snapshot: WorldSnapshot, delta: TickDelta): WorldSnapshot {
  const movements = new Map<number, [number, number]>();
  for (const move of delta.moves) {
    movements.set(unwrapId(move.id), move.to);
  }
  const removed = new Set(delta.removed_positions.map((entry) => unwrapId(entry.id)));
  const removedFoods = new Set(
    (Array.isArray(delta.food_removed_positions) ? delta.food_removed_positions : []).map(
      (entry) => unwrapId(entry.id),
    ),
  );

  const organisms = snapshot.organisms
    .filter((organism: OrganismState) => !removed.has(unwrapId(organism.id)))
    .map((organism: OrganismState) => {
      const next = movements.get(unwrapId(organism.id));
      if (!next) {
        return { ...organism, age_turns: organism.age_turns + 1 };
      }
      return { ...organism, q: next[0], r: next[1], age_turns: organism.age_turns + 1 };
    })
    .concat(delta.spawned)
    .sort((a, b) => unwrapId(a.id) - unwrapId(b.id));
  const foods = (Array.isArray(snapshot.foods) ? snapshot.foods : [])
    .filter((food) => !removedFoods.has(unwrapId(food.id)))
    .concat(Array.isArray(delta.food_spawned) ? delta.food_spawned : [])
    .sort((a, b) => unwrapId(a.id) - unwrapId(b.id));

  return {
    ...snapshot,
    turn: delta.turn,
    metrics: delta.metrics,
    organisms,
    foods,
  };
}

export function findOrganism(snapshot: WorldSnapshot, organismId: number) {
  return snapshot.organisms.find((item) => unwrapId(item.id) === organismId) ?? null;
}
