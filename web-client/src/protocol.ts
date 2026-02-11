import type {
  FoodId,
  OrganismId,
  RemovedEntityPosition,
  TickDelta,
  WorldOrganismState,
  WorldSnapshot,
} from './types';

export function unwrapId(id: OrganismId | FoodId | number | { 0: number }) {
  if (typeof id === 'number') return id;
  return id[0];
}

export function applyTickDelta(snapshot: WorldSnapshot, delta: TickDelta): WorldSnapshot {
  const movements = new Map<number, [number, number]>();
  const deltaMoves = Array.isArray(delta.moves) ? delta.moves : [];
  for (const move of deltaMoves) {
    movements.set(unwrapId(move.id), move.to as [number, number]);
  }

  const removedOrganisms = new Set<number>();
  const removedFoods = new Set<number>();
  const removedPositions = Array.isArray(delta.removed_positions) ? delta.removed_positions : [];
  for (const entry of removedPositions) {
    if (entry.entity_id.entity_type === 'Organism') {
      removedOrganisms.add(unwrapId(entry.entity_id.id));
    } else {
      removedFoods.add(unwrapId(entry.entity_id.id));
    }
  }

  const spawnedOrganisms = Array.isArray(delta.spawned) ? delta.spawned : [];
  const foodSpawned = Array.isArray(delta.food_spawned) ? delta.food_spawned : [];

  const organisms = snapshot.organisms
    .filter((organism: WorldOrganismState) => !removedOrganisms.has(unwrapId(organism.id)))
    .map((organism: WorldOrganismState) => {
      const next = movements.get(unwrapId(organism.id));
      if (!next) {
        return { ...organism, age_turns: organism.age_turns + 1 };
      }
      return { ...organism, q: next[0], r: next[1], age_turns: organism.age_turns + 1 };
    })
    .concat(spawnedOrganisms);
  const foods = (Array.isArray(snapshot.foods) ? snapshot.foods : [])
    .filter((food) => !removedFoods.has(unwrapId(food.id)))
    .concat(foodSpawned);

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
