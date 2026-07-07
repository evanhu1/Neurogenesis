import type {
  ApiChampionPoolResponse,
  ApiEntityId,
  ApiFoodState,
  ApiMetricsSnapshot,
  ApiOrganismDetail,
  ApiOrganismGenome,
  ApiOrganismState,
  ApiScalarId,
  ApiTickDelta,
  ApiWorldOrganismState,
  ApiWorldSnapshot,
  BrainState,
  ChampionPoolResponse,
  EntityId,
  FoodState,
  MetricsSnapshot,
  NeuronState,
  OrganismDetail,
  OrganismGenome,
  OrganismState,
  TickDelta,
  WorldOrganismState,
  WorldSnapshot,
} from './types';

export function unwrapId(id: ApiScalarId): number {
  return typeof id === 'number' ? id : id[0];
}

const unwrapNullableId = (id: ApiScalarId | null) => (id == null ? null : unwrapId(id));

function normalizeNeuronState(neuron: ApiOrganismState['brain']['sensory'][number]['neuron']): NeuronState {
  return {
    ...neuron,
    neuron_id: unwrapId(neuron.neuron_id),
  };
}

function normalizeOrganismGenome(genome: ApiOrganismGenome): OrganismGenome {
  return {
    ...genome,
    brain: {
      ...genome.brain,
      edges: genome.brain.edges.map((edge) => ({
        ...edge,
        pre_neuron_id: unwrapId(edge.pre_neuron_id),
        post_neuron_id: unwrapId(edge.post_neuron_id),
      })),
    },
  };
}

function normalizeBrainState(brain: ApiOrganismState['brain']): BrainState {
  const normalizeSynapses = (synapses: ApiOrganismState['brain']['sensory'][number]['synapses']) =>
    synapses.map((synapse) => ({
      ...synapse,
      pre_neuron_id: unwrapId(synapse.pre_neuron_id),
      post_neuron_id: unwrapId(synapse.post_neuron_id),
    }));

  return {
    synapse_count: brain.synapse_count,
    sensory: brain.sensory.map((sensory) => ({
      ...sensory,
      neuron: normalizeNeuronState(sensory.neuron),
      synapses: normalizeSynapses(sensory.synapses),
    })),
    inter: brain.inter.map((inter) => ({
      ...inter,
      neuron: normalizeNeuronState(inter.neuron),
      synapses: normalizeSynapses(inter.synapses),
    })),
    action: brain.action.map((action) => ({
      ...action,
      neuron_id: unwrapId(action.neuron_id),
    })),
  };
}

function normalizeFoodState(food: ApiFoodState): FoodState {
  return { ...food, id: unwrapId(food.id) };
}

function normalizeWorldOrganismState(organism: ApiWorldOrganismState): WorldOrganismState {
  return {
    ...organism,
    id: unwrapId(organism.id),
    species_id: unwrapId(organism.species_id),
  };
}

function normalizeOrganismState(organism: ApiOrganismState): OrganismState {
  return {
    ...organism,
    id: unwrapId(organism.id),
    species_id: unwrapId(organism.species_id),
    brain: normalizeBrainState(organism.brain),
    genome: normalizeOrganismGenome(organism.genome),
  };
}

function normalizeEntityId(entityId: ApiEntityId): EntityId {
  return { ...entityId, id: unwrapId(entityId.id) };
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
  return {
    ...metrics,
    species_counts,
    total_species_created: Math.max(
      previousTotalSpeciesCreated,
      Object.keys(species_counts).length,
    ),
  };
}

export function normalizeChampionPoolResponse(
  response: ApiChampionPoolResponse,
): ChampionPoolResponse {
  return {
    entries: response.entries.map((entry) => ({
      ...entry,
      genome: normalizeOrganismGenome(entry.genome),
    })),
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
    terrain: snapshot.terrain,
    metrics: normalizeMetrics(snapshot.metrics, organisms, previousTotalSpeciesCreated),
  };
}

export function normalizeTickDelta(delta: ApiTickDelta): TickDelta {
  return {
    turn: delta.turn,
    moves: delta.moves.map((move) => ({ ...move, id: unwrapId(move.id) })),
    facing_updates: delta.facing_updates.map((update) => ({
      ...update,
      id: unwrapId(update.id),
    })),
    removed_positions: delta.removed_positions.map((entry) => ({
      ...entry,
      entity_id: normalizeEntityId(entry.entity_id),
    })),
    spawned: delta.spawned.map(normalizeWorldOrganismState),
    food_spawned: delta.food_spawned.map(normalizeFoodState),
    metrics: delta.metrics,
  };
}

export function normalizeOrganismDetail(data: ApiOrganismDetail): OrganismDetail {
  return {
    turn: data.turn,
    organism: normalizeOrganismState(data.organism),
    active_action_neuron_id: unwrapNullableId(data.active_action_neuron_id),
  };
}

