export type ApiScalarId = number | { 0: number };

export type ApiOrganismId = ApiScalarId;
export type ApiSpeciesId = ApiScalarId;
export type ApiFoodId = ApiScalarId;
export type ApiNeuronId = ApiScalarId;

export type OrganismId = number;
export type SpeciesId = number;
export type FoodId = number;
export type NeuronId = number;

export type ActionType = 'Idle' | 'TurnLeft' | 'TurnRight' | 'Forward' | 'Consume' | 'Reproduce';

export type NeuronType = 'Sensory' | 'Inter' | 'Action';

export type FacingDirection =
  | 'East'
  | 'NorthEast'
  | 'NorthWest'
  | 'West'
  | 'SouthWest'
  | 'SouthEast';

export type EntityType = 'Food' | 'Organism' | 'Wall';

type GenomeCoreParams = {
  num_neurons: number;
  num_synapses: number;
  spatial_prior_sigma: number;
  vision_distance: number;
  starting_energy: number;
  age_of_maturity: number;
  hebb_eta_gain: number;
  eligibility_retention: number;
  synapse_prune_threshold: number;
};

type GenomeMutationRateParams = {
  mutation_rate_age_of_maturity: number;
  mutation_rate_vision_distance: number;
  mutation_rate_inter_bias: number;
  mutation_rate_inter_update_rate: number;
  mutation_rate_action_bias: number;
  mutation_rate_eligibility_retention: number;
  mutation_rate_synapse_prune_threshold: number;
  mutation_rate_neuron_location: number;
  mutation_rate_synapse_weight_perturbation: number;
  mutation_rate_add_synapse: number;
  mutation_rate_remove_synapse: number;
  mutation_rate_add_neuron_split_edge: number;
};

export type GenomeHyperparams = GenomeCoreParams & GenomeMutationRateParams;

export type SeedGenomeConfig = GenomeHyperparams;
export type ApiSeedGenomeConfig = GenomeHyperparams;

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
  terrain_noise_scale: number;
  terrain_threshold: number;
  max_organism_age: number;
  speciation_threshold: number;
  global_mutation_rate_modifier: number;
  meta_mutation_enabled: boolean;
  seed_genome_config: SeedGenomeConfig;
};

export type BrainLocation = {
  x: number;
  y: number;
};

export type ApiSynapseEdge = {
  pre_neuron_id: ApiNeuronId;
  post_neuron_id: ApiNeuronId;
  weight: number;
};

export type SynapseEdge = {
  pre_neuron_id: NeuronId;
  post_neuron_id: NeuronId;
  weight: number;
};

type OrganismGenomeVectors<TEdge> = {
  inter_biases: number[];
  inter_log_time_constants: number[];
  action_biases: number[];
  sensory_locations: BrainLocation[];
  inter_locations: BrainLocation[];
  action_locations: BrainLocation[];
  edges: TEdge[];
};

export type ApiOrganismGenome = GenomeHyperparams & OrganismGenomeVectors<ApiSynapseEdge>;

export type OrganismGenome = GenomeHyperparams & OrganismGenomeVectors<SynapseEdge>;

export type ApiNeuronState = {
  neuron_id: ApiNeuronId;
  neuron_type: NeuronType;
  bias: number;
  x: number;
  y: number;
  activation: number;
  parent_ids: ApiNeuronId[];
};

export type NeuronState = {
  neuron_id: NeuronId;
  neuron_type: NeuronType;
  bias: number;
  x: number;
  y: number;
  activation: number;
  parent_ids: NeuronId[];
};

type LookRayReceptor = {
  receptor_type: 'LookRay';
  ray_offset: number;
  look_target: EntityType;
};

type EnergyReceptor = {
  receptor_type: 'Energy';
};

export type SensoryReceptor = LookRayReceptor | EnergyReceptor;

export type ApiSensoryNeuronState = {
  neuron: ApiNeuronState;
  synapses: ApiSynapseEdge[];
} & SensoryReceptor;

export type SensoryNeuronState = {
  neuron: NeuronState;
  synapses: SynapseEdge[];
} & SensoryReceptor;

export type ApiInterNeuronState = {
  neuron: ApiNeuronState;
  alpha: number;
  synapses: ApiSynapseEdge[];
};

export type InterNeuronState = {
  neuron: NeuronState;
  alpha: number;
  synapses: SynapseEdge[];
};

export type ApiActionNeuronState = {
  neuron: ApiNeuronState;
  action_type: ActionType;
};

export type ActionNeuronState = {
  neuron: NeuronState;
  action_type: ActionType;
};

export type ApiBrainState = {
  sensory: ApiSensoryNeuronState[];
  inter: ApiInterNeuronState[];
  action: ApiActionNeuronState[];
  synapse_count: number;
};

export type BrainState = {
  sensory: SensoryNeuronState[];
  inter: InterNeuronState[];
  action: ActionNeuronState[];
  synapse_count: number;
};

export type ApiOrganismState = {
  id: ApiOrganismId;
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
  last_action_taken: ActionType;
  brain: ApiBrainState;
  genome: ApiOrganismGenome;
};

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
  last_action_taken: ActionType;
  brain: BrainState;
  genome: OrganismGenome;
};

export type ApiWorldOrganismState = {
  id: ApiOrganismId;
  q: number;
  r: number;
  generation: number;
  age_turns: number;
  facing: FacingDirection;
  energy: number;
  consumptions_count: number;
  reproductions_count: number;
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

export type ApiFoodState = {
  id: ApiFoodId;
  q: number;
  r: number;
  energy: number;
};

export type FoodState = {
  id: FoodId;
  q: number;
  r: number;
  energy: number;
};

export type ApiMetricsSnapshot = {
  turns: number;
  organisms: number;
  synapse_ops_last_turn: number;
  actions_applied_last_turn: number;
  consumptions_last_turn: number;
  predations_last_turn: number;
  total_consumptions: number;
  reproductions_last_turn: number;
  starvations_last_turn: number;
};

export type MetricsSnapshot = ApiMetricsSnapshot & {
  total_species_created: number;
  species_counts: Record<string, number>;
};

export type ApiOccupant =
  | { type: 'Organism'; id: ApiOrganismId }
  | { type: 'Food'; id: ApiFoodId }
  | { type: 'Wall' };

export type Occupant =
  | { type: 'Organism'; id: OrganismId }
  | { type: 'Food'; id: FoodId }
  | { type: 'Wall' };

export type ApiOccupancyCell = {
  q: number;
  r: number;
  occupant: ApiOccupant;
};

export type OccupancyCell = {
  q: number;
  r: number;
  occupant: Occupant;
};

export type ApiWorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: ApiWorldOrganismState[];
  foods: ApiFoodState[];
  occupancy: ApiOccupancyCell[];
  metrics: ApiMetricsSnapshot;
};

export type WorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: WorldOrganismState[];
  foods: FoodState[];
  occupancy: OccupancyCell[];
  metrics: MetricsSnapshot;
};

export type ApiSessionMetadata = {
  id: string;
  created_at_unix_ms: number;
  config: WorldConfig;
  running: boolean;
  ticks_per_second: number;
};

export type SessionMetadata = ApiSessionMetadata;

export type ApiCreateSessionResponse = {
  metadata: ApiSessionMetadata;
  snapshot: ApiWorldSnapshot;
};

export type CreateSessionResponse = {
  metadata: SessionMetadata;
  snapshot: WorldSnapshot;
};

export type BatchRunStatus = 'Running' | 'Completed' | 'Failed';

export type ApiBatchAggregateStats = {
  total_organisms_alive: number;
  mean_organisms_alive: number;
  min_organisms_alive: number;
  max_organisms_alive: number;
};

export type BatchAggregateStats = ApiBatchAggregateStats & {
  total_species_alive: number;
  mean_species_alive: number;
  min_species_alive: number;
  max_species_alive: number;
};

export type ApiArchivedWorldSource =
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

export type ArchivedWorldSource = ApiArchivedWorldSource;

export type ApiArchivedWorldSummary = {
  world_id: string;
  created_at_unix_ms: number;
  turn: number;
  organisms_alive: number;
  source: ApiArchivedWorldSource;
};

export type ArchivedWorldSummary = ApiArchivedWorldSummary & {
  species_alive: number;
};

export type ApiCreateBatchRunResponse = {
  run_id: string;
};

export type CreateBatchRunResponse = ApiCreateBatchRunResponse;

export type ApiBatchRunStatusResponse = {
  run_id: string;
  created_at_unix_ms: number;
  status: BatchRunStatus;
  total_worlds: number;
  completed_worlds: number;
  aggregate: ApiBatchAggregateStats | null;
  worlds: ApiArchivedWorldSummary[];
  error: string | null;
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

export type ApiListArchivedWorldsResponse = {
  worlds: ApiArchivedWorldSummary[];
};

export type ListArchivedWorldsResponse = {
  worlds: ArchivedWorldSummary[];
};

export type ApiEntityId =
  | { entity_type: 'Organism'; id: ApiOrganismId }
  | { entity_type: 'Food'; id: ApiFoodId };

export type EntityId =
  | { entity_type: 'Organism'; id: OrganismId }
  | { entity_type: 'Food'; id: FoodId };

export type ApiRemovedEntityPosition = {
  entity_id: ApiEntityId;
  q: number;
  r: number;
};

export type RemovedEntityPosition = {
  entity_id: EntityId;
  q: number;
  r: number;
};

export type ApiOrganismMove = { id: ApiOrganismId; from: [number, number]; to: [number, number] };
export type ApiOrganismFacing = { id: ApiOrganismId; facing: FacingDirection };

export type OrganismMove = { id: OrganismId; from: [number, number]; to: [number, number] };
export type OrganismFacing = { id: OrganismId; facing: FacingDirection };

export type ApiTickDelta = {
  turn: number;
  moves: ApiOrganismMove[];
  facing_updates: ApiOrganismFacing[];
  removed_positions: ApiRemovedEntityPosition[];
  spawned: ApiWorldOrganismState[];
  food_spawned: ApiFoodState[];
  metrics: ApiMetricsSnapshot;
};

export type TickDelta = {
  turn: number;
  moves: OrganismMove[];
  facing_updates: OrganismFacing[];
  removed_positions: RemovedEntityPosition[];
  spawned: WorldOrganismState[];
  food_spawned: FoodState[];
  metrics: MetricsSnapshot;
};

export type ApiStepProgressData = {
  requested_count: number;
  completed_count: number;
};

export type StepProgressData = ApiStepProgressData;

export type ApiFocusBrainData = {
  organism: ApiOrganismState;
  active_action_neuron_id: ApiNeuronId | null;
};

export type FocusBrainData = {
  organism: OrganismState;
  active_action_neuron_id: NeuronId | null;
};

export type ApiErrorData = {
  code: string;
  message: string;
};

export type ApiServerEvent =
  | { type: 'StateSnapshot'; data: ApiWorldSnapshot }
  | { type: 'TickDelta'; data: ApiTickDelta }
  | { type: 'StepProgress'; data: ApiStepProgressData }
  | { type: 'FocusBrain'; data: ApiFocusBrainData }
  | { type: 'Metrics'; data: ApiMetricsSnapshot }
  | { type: 'Error'; data: ApiErrorData };

export type ServerEvent = ApiServerEvent;
