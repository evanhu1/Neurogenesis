import type {
  ApiEntityId,
  ApiMetricsSnapshot,
  ApiOrganismDetail,
  ApiOrganismGenome,
  ApiOrganismState,
  ApiScalarId,
  ApiTickDelta,
  ApiWorldOrganismState,
  ApiWorldSnapshot,
  BrainState,
  EntityId,
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
      hidden_nodes: genome.brain.hidden_nodes.map((node) => ({ ...node })),
      edges: genome.brain.edges.map((edge) => ({ ...edge })),
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
    recurrent_synapses: normalizeSynapses(brain.recurrent_synapses),
    action_feedback_synapses: normalizeSynapses(brain.action_feedback_synapses),
    previous_inter_activations: [...brain.previous_inter_activations],
    previous_action_activations: [...brain.previous_action_activations],
    previous_prediction_error: brain.previous_prediction_error,
    value_bias: brain.value_bias,
    inherited_value_bias: brain.inherited_value_bias,
    value_bias_eligibility: brain.value_bias_eligibility,
    sensory_mean_activation: [...brain.sensory_mean_activation],
    inter_mean_activation: [...brain.inter_mean_activation],
    action_mean_activation: [...brain.action_mean_activation],
    means_initialized: brain.means_initialized,
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
