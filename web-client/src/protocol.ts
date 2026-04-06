import type {
  ApiChampionPoolEntry,
  ApiChampionPoolResponse,
  ApiCreateSessionResponse,
  ApiEntityId,
  ApiFocusBrainData,
  ApiFoodState,
  ApiLiveMetricsData,
  ApiMetricsSnapshot,
  ApiOccupancyCell,
  ApiTerrainCell,
  ApiOccupant,
  ApiOrganismGenome,
  ApiOrganismState,
  ApiScalarId,
  ApiSynapseEdge,
  ApiTickDelta,
  ApiWorldOrganismState,
  ApiWorldSnapshot,
  BrainState,
  ChampionPoolResponse,
  CreateSessionResponse,
  EntityId,
  FoodId,
  FoodState,
  FocusBrainData,
  LiveMetricsData,
  MetricsSnapshot,
  NeuronId,
  NeuronState,
  OccupancyCell,
  TerrainCell,
  Occupant,
  OrganismGenome,
  OrganismId,
  OrganismState,
  RemovedEntityPosition,
  SynapseEdge,
  TickDelta,
  WorldOrganismState,
  WorldSnapshot,
} from './types';

export function unwrapId(id: ApiScalarId | number): number {
  if (typeof id === 'number') return id;
  return id[0];
}

function normalizeSynapseEdge(edge: ApiSynapseEdge): SynapseEdge {
  return {
    pre_neuron_id: unwrapId(edge.pre_neuron_id),
    post_neuron_id: unwrapId(edge.post_neuron_id),
    weight: edge.weight,
  };
}

function normalizeNeuronState(neuron: {
  neuron_id: ApiScalarId;
  neuron_type: NeuronState['neuron_type'];
  bias: number;
  x: number;
  y: number;
  activation: number;
  parent_ids: ApiScalarId[];
}): NeuronState {
  return {
    neuron_id: unwrapId(neuron.neuron_id),
    neuron_type: neuron.neuron_type,
    bias: neuron.bias,
    x: neuron.x,
    y: neuron.y,
    activation: neuron.activation,
    parent_ids: neuron.parent_ids.map(unwrapId),
  };
}

function normalizeOrganismGenome(genome: ApiOrganismGenome): OrganismGenome {
  return {
    ...genome,
    edges: genome.edges.map(normalizeSynapseEdge),
  };
}

function normalizeChampionPoolEntry(entry: ApiChampionPoolEntry) {
  return {
    ...entry,
    genome: normalizeOrganismGenome(entry.genome),
  };
}

function normalizeBrainState(brain: ApiOrganismState['brain']): BrainState {
  return {
    synapse_count: brain.synapse_count,
    sensory: brain.sensory.map((sensory) => {
      const normalized = {
        ...sensory,
        neuron: normalizeNeuronState(sensory.neuron),
        synapses: sensory.synapses.map(normalizeSynapseEdge),
      };
      return normalized;
    }),
    inter: brain.inter.map((inter) => ({
      neuron: normalizeNeuronState(inter.neuron),
      alpha: inter.alpha,
      synapses: inter.synapses.map(normalizeSynapseEdge),
    })),
    action: brain.action.map((action) => ({
      neuron_id: unwrapId(action.neuron_id),
      x: action.x,
      y: action.y,
      logit: action.logit,
      parent_ids: action.parent_ids.map(unwrapId),
      action_type: action.action_type,
    })),
  };
}

function normalizeFoodState(food: ApiFoodState): FoodState {
  return {
    id: unwrapId(food.id),
    q: food.q,
    r: food.r,
    energy: food.energy,
    kind: food.kind,
  };
}

function normalizeWorldOrganismState(organism: ApiWorldOrganismState): WorldOrganismState {
  return {
    id: unwrapId(organism.id),
    species_id: unwrapId(organism.species_id),
    q: organism.q,
    r: organism.r,
    generation: organism.generation,
    age_turns: organism.age_turns,
    facing: organism.facing,
    energy: organism.energy,
    health: organism.health,
    max_health: organism.max_health,
    damage_taken_last_turn: organism.damage_taken_last_turn,
    consumptions_count: organism.consumptions_count,
    reproductions_count: organism.reproductions_count,
  };
}

function normalizeOrganismState(organism: ApiOrganismState): OrganismState {
  return {
    id: unwrapId(organism.id),
    species_id: unwrapId(organism.species_id),
    q: organism.q,
    r: organism.r,
    generation: organism.generation,
    age_turns: organism.age_turns,
    facing: organism.facing,
    energy: organism.energy,
    health: organism.health,
    max_health: organism.max_health,
    energy_prev: organism.energy_prev,
    dopamine: organism.dopamine,
    damage_taken_last_turn: organism.damage_taken_last_turn,
    consumptions_count: organism.consumptions_count,
    reproductions_count: organism.reproductions_count,
    last_action_taken: organism.last_action_taken,
    brain: normalizeBrainState(organism.brain),
    genome: normalizeOrganismGenome(organism.genome),
  };
}

function normalizeOccupant(occupant: ApiOccupant): Occupant {
  if (occupant.type === 'Wall') {
    return occupant;
  }
  if (occupant.type === 'Organism') {
    return {
      type: 'Organism',
      id: unwrapId(occupant.id) as OrganismId,
    };
  }
  return {
    type: 'Food',
    id: unwrapId(occupant.id) as FoodId,
  };
}

function normalizeOccupancyCell(cell: ApiOccupancyCell): OccupancyCell {
  return {
    q: cell.q,
    r: cell.r,
    occupant: normalizeOccupant(cell.occupant),
  };
}

function normalizeTerrainCell(cell: ApiTerrainCell): TerrainCell {
  return {
    q: cell.q,
    r: cell.r,
    terrain_type: cell.terrain_type,
  };
}

function normalizeEntityId(entityId: ApiEntityId): EntityId {
  if (entityId.entity_type === 'Organism') {
    return {
      entity_type: 'Organism',
      id: unwrapId(entityId.id) as OrganismId,
    };
  }
  return {
    entity_type: 'Food',
    id: unwrapId(entityId.id) as FoodId,
  };
}

function computeSpeciesCounts(organisms: WorldOrganismState[]): Record<string, number> {
  const speciesCounts: Record<string, number> = {};
  for (const organism of organisms) {
    const key = String(organism.species_id);
    speciesCounts[key] = (speciesCounts[key] ?? 0) + 1;
  }
  return speciesCounts;
}

function normalizeMetrics(
  metrics: ApiMetricsSnapshot,
  organisms: WorldOrganismState[],
  previousTotalSpeciesCreated = 0,
): MetricsSnapshot {
  const species_counts = computeSpeciesCounts(organisms);
  const speciesAlive = Object.keys(species_counts).length;
  return {
    ...metrics,
    species_counts,
    total_species_created: Math.max(previousTotalSpeciesCreated, speciesAlive),
  };
}

function normalizeRemovedEntityPosition(entry: {
  entity_id: ApiEntityId;
  q: number;
  r: number;
}): RemovedEntityPosition {
  return {
    entity_id: normalizeEntityId(entry.entity_id),
    q: entry.q,
    r: entry.r,
  };
}

export function normalizeChampionPoolResponse(
  response: ApiChampionPoolResponse,
): ChampionPoolResponse {
  return {
    entries: response.entries.map(normalizeChampionPoolEntry),
  };
}

export function normalizeWorldSnapshot(
  snapshot: ApiWorldSnapshot,
  previousTotalSpeciesCreated = 0,
): WorldSnapshot {
  const organisms = snapshot.organisms.map(normalizeWorldOrganismState);
  return {
    turn: snapshot.turn,
    rng_seed: snapshot.rng_seed,
    config: snapshot.config,
    organisms,
    foods: snapshot.foods.map(normalizeFoodState),
    terrain: snapshot.terrain.map(normalizeTerrainCell),
    occupancy: snapshot.occupancy.map(normalizeOccupancyCell),
    metrics: normalizeMetrics(snapshot.metrics, organisms, previousTotalSpeciesCreated),
  };
}

export function normalizeTickDelta(delta: ApiTickDelta): TickDelta {
  return {
    turn: delta.turn,
    moves: delta.moves.map((move) => ({
      id: unwrapId(move.id),
      from: move.from,
      to: move.to,
    })),
    facing_updates: delta.facing_updates.map((update) => ({
      id: unwrapId(update.id),
      facing: update.facing,
    })),
    removed_positions: delta.removed_positions.map(normalizeRemovedEntityPosition),
    spawned: delta.spawned.map(normalizeWorldOrganismState),
    reproduction_events: delta.reproduction_events.map((event) => ({
      parent_id: unwrapId(event.parent_id),
      parent_species_id: unwrapId(event.parent_species_id),
      parent_age_turns: event.parent_age_turns,
      parent_generation: event.parent_generation,
      investment_energy: event.investment_energy,
      parent_energy_after_event: event.parent_energy_after_event,
      child_id: event.child_id == null ? null : unwrapId(event.child_id),
      failure_cause: event.failure_cause,
    })),
    food_spawned: delta.food_spawned.map(normalizeFoodState),
    metrics: {
      ...delta.metrics,
      total_species_created: 0,
      species_counts: {},
    },
  };
}

export function normalizeFocusBrainData(data: ApiFocusBrainData): FocusBrainData {
  return {
    turn: data.turn,
    organism: normalizeOrganismState(data.organism),
    active_action_neuron_id:
      data.active_action_neuron_id == null ? null : (unwrapId(data.active_action_neuron_id) as NeuronId),
  };
}

export function normalizeCreateSessionResponse(response: ApiCreateSessionResponse): CreateSessionResponse {
  return {
    metadata: response.metadata,
    snapshot: normalizeWorldSnapshot(response.snapshot),
  };
}

export function normalizeLiveMetricsData(
  data: ApiLiveMetricsData,
  previousTotalSpeciesCreated = 0,
): LiveMetricsData {
  const species_counts = { ...data.species_counts };
  return {
    turn: data.turn,
    metrics: {
      ...data.metrics,
      species_counts,
      total_species_created: Math.max(
        previousTotalSpeciesCreated,
        Object.keys(species_counts).length,
      ),
    },
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
      const facing = facings.get(organism.id) ?? organism.facing;
      const next = movements.get(organism.id);
      if (!next) {
        return { ...organism, facing, age_turns: organism.age_turns + 1 };
      }
      return {
        ...organism,
        q: next[0],
        r: next[1],
        facing,
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
    metrics: normalizeMetrics(
      delta.metrics,
      organisms,
      snapshot.metrics.total_species_created,
    ),
    organisms,
    foods,
    terrain: snapshot.terrain,
  };
}

export function findOrganism(snapshot: WorldSnapshot, organismId: number) {
  return snapshot.organisms.find((item) => item.id === organismId) ?? null;
}
