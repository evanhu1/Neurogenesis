import defaultConfigToml from '../../config/default.toml?raw';

export type OrganismId = { 0: number } | number;

export type Envelope<T> = {
  protocol_version: number;
  payload: T;
};

export type WorldConfig = {
  world_width: number;
  steps_per_second: number;
  num_organisms: number;
  num_neurons: number;
  max_num_neurons: number;
  num_synapses: number;
  turns_to_starve: number;
  mutation_chance: number;
  mutation_magnitude: number;
  center_spawn_min_fraction: number;
  center_spawn_max_fraction: number;
};

export type SynapseEdge = { post_neuron_id: number | { 0: number }; weight: number };

export type NeuronState = {
  neuron_id: number | { 0: number };
  neuron_type: 'Sensory' | 'Inter' | 'Action' | string;
  bias: number;
  activation: number;
  parent_ids: Array<number | { 0: number }>;
};

export type SensoryNeuronState = {
  neuron: NeuronState;
  receptor_type: string;
  synapses: SynapseEdge[];
};

export type InterNeuronState = {
  neuron: NeuronState;
  synapses: SynapseEdge[];
};

export type ActionNeuronState = {
  neuron: NeuronState;
  action_type: string;
  is_active: boolean;
};

export type BrainState = {
  sensory: SensoryNeuronState[];
  inter: InterNeuronState[];
  action: ActionNeuronState[];
  synapse_count: number;
};

export type FacingDirection =
  | 'East'
  | 'NorthEast'
  | 'NorthWest'
  | 'West'
  | 'SouthWest'
  | 'SouthEast';

export type OrganismState = {
  id: OrganismId;
  q: number;
  r: number;
  facing: FacingDirection;
  turns_since_last_meal: number;
  meals_eaten: number;
  brain: BrainState;
};

export type MetricsSnapshot = {
  turns: number;
  organisms: number;
  synapse_ops_last_turn: number;
  actions_applied_last_turn: number;
  meals_last_turn: number;
  starvations_last_turn: number;
  births_last_turn: number;
};

export type WorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: OrganismState[];
  occupancy: Array<{ q: number; r: number; organism_ids: OrganismId[] }>;
  metrics: MetricsSnapshot;
};

export type SessionMetadata = {
  id: string;
  created_at_unix_ms: number;
  config: WorldConfig;
};

export type CreateSessionResponse = {
  metadata: SessionMetadata;
  snapshot: WorldSnapshot;
};

export type TickDelta = {
  turn: number;
  moves: Array<{ id: OrganismId; from: [number, number]; to: [number, number] }>;
  removed: OrganismId[];
  spawned: OrganismState[];
  metrics: MetricsSnapshot;
};

export type ServerEvent = {
  type: 'StateSnapshot' | 'TickDelta' | 'FocusBrain' | 'Metrics' | 'Error';
  data: unknown;
};

function parseRequiredNumber(map: Record<string, number>, key: string): number {
  const value = map[key];
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`default config is missing numeric key: ${key}`);
  }
  return value;
}

function parseDefaultConfigToml(tomlText: string): WorldConfig {
  const topLevel: Record<string, number> = {};

  for (const rawLine of tomlText.split('\n')) {
    const line = rawLine.split('#')[0].trim();
    if (!line) continue;
    if (line.startsWith('[') && line.endsWith(']')) continue;

    const eqIdx = line.indexOf('=');
    if (eqIdx === -1) continue;

    const key = line.slice(0, eqIdx).trim();
    const valueRaw = line.slice(eqIdx + 1).trim();
    const value = Number(valueRaw);
    if (Number.isNaN(value)) {
      continue;
    }

    topLevel[key] = value;
  }

  return {
    world_width: parseRequiredNumber(topLevel, 'world_width'),
    steps_per_second: parseRequiredNumber(topLevel, 'steps_per_second'),
    num_organisms: parseRequiredNumber(topLevel, 'num_organisms'),
    num_neurons: parseRequiredNumber(topLevel, 'num_neurons'),
    max_num_neurons: parseRequiredNumber(topLevel, 'max_num_neurons'),
    num_synapses: parseRequiredNumber(topLevel, 'num_synapses'),
    turns_to_starve: parseRequiredNumber(topLevel, 'turns_to_starve'),
    mutation_chance: parseRequiredNumber(topLevel, 'mutation_chance'),
    mutation_magnitude: parseRequiredNumber(topLevel, 'mutation_magnitude'),
    center_spawn_min_fraction: parseRequiredNumber(topLevel, 'center_spawn_min_fraction'),
    center_spawn_max_fraction: parseRequiredNumber(topLevel, 'center_spawn_max_fraction'),
  };
}

export const DEFAULT_CONFIG: WorldConfig = parseDefaultConfigToml(defaultConfigToml);
