export type OrganismId = { 0: number } | number;
export type SpeciesId = { 0: number } | number;
export type FoodId = { 0: number } | number;

export type SeedGenomeConfig = {
  num_neurons: number;
  num_synapses: number;
  spatial_prior_sigma: number;
  vision_distance: number;
  starting_energy: number;
  age_of_maturity: number;
  hebb_eta_gain: number;
  eligibility_retention: number;
  synapse_prune_threshold: number;
  mutation_rate_age_of_maturity: number;
  mutation_rate_vision_distance: number;
  mutation_rate_inter_bias: number;
  mutation_rate_inter_update_rate: number;
  mutation_rate_action_bias: number;
  mutation_rate_eligibility_retention: number;
  mutation_rate_synapse_prune_threshold: number;
  mutation_rate_neuron_location: number;
  mutation_rate_synapse_weight_perturbation: number;
  mutation_rate_add_neuron_split_edge: number;
};

export type WorldConfig = {
  world_width: number;
  steps_per_second: number;
  num_organisms: number;
  periodic_injection_interval_turns: number;
  periodic_injection_count: number;
  food_energy: number;
  move_action_energy_cost: number;
  action_temperature: number;
  neuron_metabolism_cost: number;
  food_regrowth_interval: number;
  food_regrowth_jitter: number;
  food_fertility_noise_scale: number;
  food_fertility_threshold: number;
  terrain_noise_scale: number;
  terrain_threshold: number;
  max_organism_age: number;
  speciation_threshold: number;
  global_mutation_rate_modifier: number;
  seed_genome_config: SeedGenomeConfig;
};

export type OrganismGenome = {
  num_neurons: number;
  num_synapses: number;
  spatial_prior_sigma: number;
  vision_distance: number;
  starting_energy: number;
  age_of_maturity: number;
  hebb_eta_gain: number;
  eligibility_retention: number;
  synapse_prune_threshold: number;
  mutation_rate_age_of_maturity: number;
  mutation_rate_vision_distance: number;
  mutation_rate_inter_bias: number;
  mutation_rate_inter_update_rate: number;
  mutation_rate_action_bias: number;
  mutation_rate_eligibility_retention: number;
  mutation_rate_synapse_prune_threshold: number;
  mutation_rate_neuron_location: number;
  mutation_rate_synapse_weight_perturbation: number;
  mutation_rate_add_neuron_split_edge: number;
  inter_biases: number[];
  inter_log_time_constants: number[];
  action_biases: number[];
  sensory_locations: BrainLocation[];
  inter_locations: BrainLocation[];
  action_locations: BrainLocation[];
};

export type BrainLocation = {
  x: number;
  y: number;
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
  x: number;
  y: number;
  activation: number;
  parent_ids: Array<number | { 0: number }>;
};

export type FocusBrainData = {
  organism: OrganismState;
  active_action_neuron_id: number | { 0: number } | null;
};

export type SensoryNeuronState = {
  neuron: NeuronState;
  receptor_type: string;
  ray_offset?: number;
  look_target?: string;
  synapses: SynapseEdge[];
};

export type InterNeuronState = {
  neuron: NeuronState;
  alpha: number;
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
  generation: number;
  age_turns: number;
  facing: FacingDirection;
  energy: number;
  energy_prev: number;
  dopamine: number;
  consumptions_count: number;
  reproductions_count: number;
  last_action_taken: string;
  brain: BrainState;
  genome: OrganismGenome;
};

export type WorldOrganismState = {
  id: OrganismId;
  species_id: SpeciesId;
  q: number;
  r: number;
  generation: number;
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
  occupancy?: Array<{
    q: number;
    r: number;
    occupant:
      | { type: 'Organism'; id: OrganismId }
      | { type: 'Food'; id: FoodId }
      | { type: 'Wall' };
  }>;
};

export type SessionMetadata = {
  id: string;
  created_at_unix_ms: number;
  config: WorldConfig;
  running: boolean;
  ticks_per_second: number;
};

export type CreateSessionResponse = {
  metadata: SessionMetadata;
  snapshot: WorldSnapshot;
};

export type BatchRunStatus = 'Running' | 'Completed' | 'Failed';

export type BatchAggregateStats = {
  total_organisms_alive: number;
  total_species_alive: number;
  mean_organisms_alive: number;
  mean_species_alive: number;
  min_organisms_alive: number;
  max_organisms_alive: number;
  min_species_alive: number;
  max_species_alive: number;
};

export type ArchivedWorldSource =
  | {
      type: 'BatchRun';
      data: {
        run_id: string;
        world_index: number;
        universe_seed: number;
        world_seed: number;
        ticks_simulated: number;
      };
    }
  | {
      type: 'Session';
      data: {
        session_id: string;
      };
    };

export type ArchivedWorldSummary = {
  world_id: string;
  created_at_unix_ms: number;
  turn: number;
  organisms_alive: number;
  species_alive: number;
  source: ArchivedWorldSource;
};

export type CreateBatchRunResponse = {
  run_id: string;
};

export type BatchRunStatusResponse = {
  run_id: string;
  created_at_unix_ms: number;
  status: BatchRunStatus;
  total_worlds: number;
  completed_worlds: number;
  aggregate: BatchAggregateStats | null;
  worlds: ArchivedWorldSummary[];
  error: string | null;
};

export type ListArchivedWorldsResponse = {
  worlds: ArchivedWorldSummary[];
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

export type StepProgressData = {
  requested_count: number;
  completed_count: number;
};

export type ServerEvent = {
  type: 'StateSnapshot' | 'TickDelta' | 'StepProgress' | 'FocusBrain' | 'Metrics' | 'Error';
  data: unknown;
};
