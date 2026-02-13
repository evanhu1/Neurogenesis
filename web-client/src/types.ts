export type OrganismId = { 0: number } | number;
export type SpeciesId = { 0: number } | number;
export type FoodId = { 0: number } | number;

export type SeedGenomeConfig = {
  num_neurons: number;
  num_synapses: number;
  vision_distance: number;
  age_of_maturity: number;
  hebb_eta_baseline: number;
  hebb_eta_gain: number;
  eligibility_decay_lambda: number;
  synapse_prune_threshold: number;
  mutation_rate_age_of_maturity: number;
  mutation_rate_vision_distance: number;
  mutation_rate_add_edge: number;
  mutation_rate_remove_edge: number;
  mutation_rate_split_edge: number;
  mutation_rate_inter_bias: number;
  mutation_rate_inter_update_rate: number;
  mutation_rate_action_bias: number;
  mutation_rate_eligibility_decay_lambda: number;
  mutation_rate_synapse_prune_threshold: number;
};

export type WorldConfig = {
  world_width: number;
  steps_per_second: number;
  num_organisms: number;
  starting_energy: number;
  food_energy: number;
  reproduction_energy_cost: number;
  move_action_energy_cost: number;
  turn_energy_cost: number;
  plant_target_coverage: number;
  food_regrowth_min_cooldown_turns: number;
  food_regrowth_max_cooldown_turns: number;
  food_regrowth_jitter_turns: number;
  food_regrowth_retry_cooldown_turns: number;
  food_fertility_noise_scale: number;
  food_fertility_exponent: number;
  food_fertility_floor: number;
  max_organism_age: number;
  max_num_neurons: number;
  speciation_threshold: number;
  seed_genome_config: SeedGenomeConfig;
};

export type OrganismGenome = {
  num_neurons: number;
  vision_distance: number;
  age_of_maturity: number;
  hebb_eta_baseline: number;
  hebb_eta_gain: number;
  eligibility_decay_lambda: number;
  synapse_prune_threshold: number;
  mutation_rate_age_of_maturity: number;
  mutation_rate_vision_distance: number;
  mutation_rate_add_edge: number;
  mutation_rate_remove_edge: number;
  mutation_rate_split_edge: number;
  mutation_rate_inter_bias: number;
  mutation_rate_inter_update_rate: number;
  mutation_rate_action_bias: number;
  mutation_rate_eligibility_decay_lambda: number;
  mutation_rate_synapse_prune_threshold: number;
  inter_biases: number[];
  inter_log_taus: number[];
  interneuron_types: InterNeuronType[];
  action_biases: number[];
  edges: SynapseEdge[];
};

export type InterNeuronType = 'Excitatory' | 'Inhibitory';

export type SynapseEdge = {
  pre_neuron_id: number | { 0: number };
  post_neuron_id: number | { 0: number };
  weight: number;
  eligibility: number;
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
  active_neuron_ids: Array<number | { 0: number }>;
};

export type SensoryNeuronState = {
  neuron: NeuronState;
  receptor_type: string;
  look_target?: string;
  synapses: SynapseEdge[];
};

export type InterNeuronState = {
  neuron: NeuronState;
  interneuron_type: InterNeuronType;
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
