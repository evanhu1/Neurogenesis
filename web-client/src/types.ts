import defaultConfigToml from '../../config/default.toml?raw';

export type OrganismId = { 0: number } | number;
export type SpeciesId = { 0: number } | number;
export type FoodId = { 0: number } | number;

export type SeedGenomeConfig = {
  num_neurons: number;
  max_num_neurons: number;
  num_synapses: number;
  mutation_rate: number;
  vision_distance: number;
};

export type WorldConfig = {
  world_width: number;
  steps_per_second: number;
  num_organisms: number;
  center_spawn_min_fraction: number;
  center_spawn_max_fraction: number;
  starting_energy: number;
  food_energy: number;
  reproduction_energy_cost: number;
  move_action_energy_cost: number;
  turn_energy_cost: number;
  food_coverage_divisor: number;
  max_organism_age: number;
  speciation_threshold: number;
  seed_genome_config: SeedGenomeConfig;
};

export type OrganismGenome = {
  num_neurons: number;
  max_num_neurons: number;
  vision_distance: number;
  mutation_rate: number;
  inter_biases: number[];
  edges: SynapseEdge[];
};

export type SynapseEdge = {
  pre_neuron_id: number | { 0: number };
  post_neuron_id: number | { 0: number };
  weight: number;
};

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
  look_target?: string;
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
  genome: OrganismGenome;
};

export type WorldOrganismState = {
  id: OrganismId;
  species_id: SpeciesId;
  q: number;
  r: number;
  age_turns: number;
  facing: FacingDirection;
  energy: number;
  consumptions_count: number;
  reproductions_count: number;
};

export type FoodState = {
  id: FoodId;
  q: number;
  r: number;
  energy: number;
};

export type MetricsSnapshot = {
  turns: number;
  organisms: number;
  synapse_ops_last_turn: number;
  actions_applied_last_turn: number;
  consumptions_last_turn: number;
  predations_last_turn: number;
  total_consumptions: number;
  reproductions_last_turn: number;
  starvations_last_turn: number;
  total_species_created: number;
  species_counts: Record<string, number>;
};

export type WorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: WorldOrganismState[];
  foods: FoodState[];
  metrics: MetricsSnapshot;
  species_registry?: Record<string, OrganismGenome>;
  occupancy?: Array<{ q: number; r: number; occupant: { type: 'Organism' | 'Food'; id: OrganismId | FoodId } }>;
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

export type EntityId =
  | { entity_type: 'Organism'; id: OrganismId }
  | { entity_type: 'Food'; id: FoodId };

export type RemovedEntityPosition = {
  entity_id: EntityId;
  q: number;
  r: number;
};

export type TickDelta = {
  turn: number;
  moves: Array<{ id: OrganismId; from: [number, number]; to: [number, number] }>;
  facing_updates: Array<{ id: OrganismId; facing: FacingDirection }>;
  removed_positions: RemovedEntityPosition[];
  spawned: WorldOrganismState[];
  food_spawned: FoodState[];
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

function parseNumberWithDefault(map: Record<string, number>, key: string, fallback: number): number {
  const value = map[key];
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return fallback;
  }
  return value;
}

function parseDefaultConfigToml(tomlText: string): WorldConfig {
  const worldLevel: Record<string, number> = {};
  const seedGenomeLevel: Record<string, number> = {};
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

    if (section === 'seed_genome_config') {
      seedGenomeLevel[key] = value;
    } else {
      worldLevel[key] = value;
    }
  }
  const genomeSource = Object.keys(seedGenomeLevel).length > 0 ? seedGenomeLevel : worldLevel;

  return {
    world_width: parseRequiredNumber(worldLevel, 'world_width'),
    steps_per_second: parseRequiredNumber(worldLevel, 'steps_per_second'),
    num_organisms: parseRequiredNumber(worldLevel, 'num_organisms'),
    center_spawn_min_fraction: parseRequiredNumber(worldLevel, 'center_spawn_min_fraction'),
    center_spawn_max_fraction: parseRequiredNumber(worldLevel, 'center_spawn_max_fraction'),
    starting_energy: parseRequiredNumber(worldLevel, 'starting_energy'),
    food_energy: parseRequiredNumber(worldLevel, 'food_energy'),
    reproduction_energy_cost: parseRequiredNumber(worldLevel, 'reproduction_energy_cost'),
    move_action_energy_cost: parseRequiredNumber(worldLevel, 'move_action_energy_cost'),
    turn_energy_cost: parseRequiredNumber(worldLevel, 'turn_energy_cost'),
    food_coverage_divisor: parseRequiredNumber(worldLevel, 'food_coverage_divisor'),
    max_organism_age: parseRequiredNumber(worldLevel, 'max_organism_age'),
    speciation_threshold: parseNumberWithDefault(worldLevel, 'speciation_threshold', 50.0),
    seed_genome_config: {
      num_neurons: parseRequiredNumber(genomeSource, 'num_neurons'),
      max_num_neurons: parseRequiredNumber(genomeSource, 'max_num_neurons'),
      num_synapses: parseRequiredNumber(genomeSource, 'num_synapses'),
      mutation_rate: parseRequiredNumber(genomeSource, 'mutation_rate'),
      vision_distance: parseNumberWithDefault(genomeSource, 'vision_distance', 2),
    },
  };
}

export const DEFAULT_CONFIG: WorldConfig = parseDefaultConfigToml(defaultConfigToml);
