import defaultConfigToml from '../../config/default.toml?raw';

export type OrganismId = { 0: number } | number;
export type SpeciesId = { 0: number } | number;

export type Envelope<T> = {
  protocol_version: number;
  payload: T;
};

export type WorldConfig = {
  world_width: number;
  steps_per_second: number;
  num_organisms: number;
  center_spawn_min_fraction: number;
  center_spawn_max_fraction: number;
  starting_energy: number;
  reproduction_energy_cost: number;
  move_action_energy_cost: number;
  seed_species_config: SpeciesConfig;
};

export type SpeciesConfig = {
  num_neurons: number;
  max_num_neurons: number;
  num_synapses: number;
  mutation_chance: number;
};

export type SynapseEdge = { post_neuron_id: number | { 0: number }; weight: number };

export type NeuronState = {
  neuron_id: number | { 0: number };
  neuron_type: 'Sensory' | 'Inter' | 'Action' | string;
  bias: number;
  activation: number;
  parent_ids: Array<number | { 0: number }>;
};

export type FocusBrainData = {
  organism: OrganismState;
  active_neuron_ids: number[];
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
  species_id: SpeciesId;
  q: number;
  r: number;
  age_turns: number;
  facing: FacingDirection;
  energy: number;
  consumptions_count: number;
  reproductions_count: number;
  brain: BrainState;
};

export type MetricsSnapshot = {
  turns: number;
  organisms: number;
  synapse_ops_last_turn: number;
  actions_applied_last_turn: number;
  consumptions_last_turn: number;
  reproductions_last_turn: number;
  starvations_last_turn: number;
  species_counts: Record<string, number>;
};

export type WorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  species_registry: Record<string, SpeciesConfig>;
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
  removed_positions: Array<{ id: OrganismId; q: number; r: number }>;
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
  const worldLevel: Record<string, number> = {};
  const seedSpeciesLevel: Record<string, number> = {};
  let section = '';

  for (const rawLine of tomlText.split('\n')) {
    const line = rawLine.split('#')[0].trim();
    if (!line) continue;
    if (line.startsWith('[') && line.endsWith(']')) {
      section = line.slice(1, -1).trim();
      continue;
    }

    const eqIdx = line.indexOf('=');
    if (eqIdx === -1) continue;

    const key = line.slice(0, eqIdx).trim();
    const valueRaw = line.slice(eqIdx + 1).trim();
    const value = Number(valueRaw);
    if (Number.isNaN(value)) {
      continue;
    }

    if (section === 'seed_species_config') {
      seedSpeciesLevel[key] = value;
    } else {
      worldLevel[key] = value;
    }
  }
  const speciesSource = Object.keys(seedSpeciesLevel).length > 0 ? seedSpeciesLevel : worldLevel;

  return {
    world_width: parseRequiredNumber(worldLevel, 'world_width'),
    steps_per_second: parseRequiredNumber(worldLevel, 'steps_per_second'),
    num_organisms: parseRequiredNumber(worldLevel, 'num_organisms'),
    center_spawn_min_fraction: parseRequiredNumber(worldLevel, 'center_spawn_min_fraction'),
    center_spawn_max_fraction: parseRequiredNumber(worldLevel, 'center_spawn_max_fraction'),
    starting_energy: parseRequiredNumber(worldLevel, 'starting_energy'),
    reproduction_energy_cost: parseRequiredNumber(worldLevel, 'reproduction_energy_cost'),
    move_action_energy_cost: parseRequiredNumber(worldLevel, 'move_action_energy_cost'),
    seed_species_config: {
      num_neurons: parseRequiredNumber(speciesSource, 'num_neurons'),
      max_num_neurons: parseRequiredNumber(speciesSource, 'max_num_neurons'),
      num_synapses: parseRequiredNumber(speciesSource, 'num_synapses'),
      mutation_chance: parseRequiredNumber(speciesSource, 'mutation_chance'),
    },
  };
}

export const DEFAULT_CONFIG: WorldConfig = parseDefaultConfigToml(defaultConfigToml);
