// Wire (`Api*`) types mirror the Rust schema exactly (sim-types + sim-server/protocol).
// UI types are identical except scalar ids are unwrapped to plain numbers; both views
// are generated from shared `...Of<Id>` shapes so they cannot drift apart.

/** Scalar ids may arrive as plain numbers or single-field tuple objects. */
export type ApiScalarId = number | { 0: number };

export type OrganismId = number;
export type SpeciesId = number;
export type FoodId = number;
export type NeuronId = number;
export type StableGeneId = string;

export type VisualProperties = {
  r: number;
  g: number;
  b: number;
  opacity: number;
  shape: number;
};

export type ActionType =
  | 'Idle'
  | 'TurnLeft'
  | 'TurnRight'
  | 'Forward'
  | 'Eat'
  | 'Attack';

export type FoodKind = 'Plant' | 'Corpse';
export type TerrainType = 'Mountain';

export type NeuronType = 'Sensory' | 'Inter' | 'Action';

export type FacingDirection =
  | 'East'
  | 'NorthEast'
  | 'NorthWest'
  | 'West'
  | 'SouthWest'
  | 'SouthEast';

export type TopologyGenes = {
  vision_distance: number;
};

export type LifecycleGenes = {
  age_of_maturity: number;
  gestation_ticks: number;
  max_organism_age: number;
};

export type PlasticityGenes = {
  hebb_eta_gain: number;
  juvenile_eta_scale: number;
  eligibility_retention: number;
  max_weight_delta_per_tick: number;
  synapse_prune_threshold: number;
};

// SeedGenomeConfig wire format stays flat (Rust SeedGenomeConfig is not nested).
export type SeedGenomeConfig = {
  num_neurons: number;
  num_synapses: number;
  vision_distance: number;
  age_of_maturity: number;
  gestation_ticks: number;
  max_organism_age: number;
  hebb_eta_gain: number;
  juvenile_eta_scale: number;
  eligibility_retention: number;
  max_weight_delta_per_tick: number;
  synapse_prune_threshold: number;
};

export type WorldConfig = {
  world_width: number;
  num_organisms: number;
  food_energy: number;
  passive_metabolism_cost_per_unit: number;
  body_mass_metabolic_cost_coeff: number;
  move_action_energy_cost: number;
  action_temperature: number;
  intent_parallel_threads: number;
  food_regrowth_interval: number;
  food_regrowth_jitter: number;
  food_tile_fraction: number;
  terrain_noise_scale: number;
  terrain_threshold: number;
  runtime_plasticity_enabled: boolean;
  leaky_neurons_enabled: boolean;
  predation_enabled: boolean;
  force_random_actions: boolean;
  seed_genome_config: SeedGenomeConfig;
};

type SynapseEdgeOf<Id> = {
  pre_neuron_id: Id;
  post_neuron_id: Id;
  weight: number;
  eligibility: number;
};

export type ApiSynapseEdge = SynapseEdgeOf<ApiScalarId>;
export type SynapseEdge = SynapseEdgeOf<NeuronId>;

export type HiddenNodeGene = {
  id: StableGeneId;
  bias: number;
  log_time_constant: number;
};

// Heritable connection identity is stable across structural mutation. Runtime
// brains separately use dense numeric NeuronId values.
export type SynapseGene = {
  innovation: StableGeneId;
  pre_node_id: StableGeneId;
  post_node_id: StableGeneId;
  weight: number;
  enabled: boolean;
};

export type ApiSynapseGene = SynapseGene;

export type BrainTopologyGenes = {
  hidden_nodes: HiddenNodeGene[];
  action_biases: number[];
  edges: SynapseGene[];
};

export type OrganismGenome = {
  topology: TopologyGenes;
  lifecycle: LifecycleGenes;
  plasticity: PlasticityGenes;
  brain: BrainTopologyGenes;
};

export type ApiOrganismGenome = OrganismGenome;

type NeuronStateOf<Id> = {
  neuron_id: Id;
  neuron_type: NeuronType;
  bias: number;
  activation: number;
};

export type ApiNeuronState = NeuronStateOf<ApiScalarId>;
export type NeuronState = NeuronStateOf<NeuronId>;

export type SensoryReceptor =
  | { receptor_type: 'FoodRay'; ray_offset: number }
  | { receptor_type: 'ContactAhead' }
  | { receptor_type: 'Energy' }
  | { receptor_type: 'OrganismRay'; ray_offset: number }
  | { receptor_type: 'Health' };

type SensoryNeuronStateOf<Id> = {
  neuron: NeuronStateOf<Id>;
  synapses: SynapseEdgeOf<Id>[];
} & SensoryReceptor;

export type ApiSensoryNeuronState = SensoryNeuronStateOf<ApiScalarId>;
export type SensoryNeuronState = SensoryNeuronStateOf<NeuronId>;

type InterNeuronStateOf<Id> = {
  neuron: NeuronStateOf<Id>;
  alpha: number;
  synapses: SynapseEdgeOf<Id>[];
};

export type ApiInterNeuronState = InterNeuronStateOf<ApiScalarId>;
export type InterNeuronState = InterNeuronStateOf<NeuronId>;

type ActionNeuronStateOf<Id> = {
  neuron_id: Id;
  logit: number;
  action_type: ActionType;
};

export type ApiActionNeuronState = ActionNeuronStateOf<ApiScalarId>;
export type ActionNeuronState = ActionNeuronStateOf<NeuronId>;

type BrainStateOf<Id> = {
  sensory: SensoryNeuronStateOf<Id>[];
  inter: InterNeuronStateOf<Id>[];
  action: ActionNeuronStateOf<Id>[];
  synapse_count: number;
};

export type ApiBrainState = BrainStateOf<ApiScalarId>;
export type BrainState = BrainStateOf<NeuronId>;

type OrganismStateOf<Id> = {
  id: Id;
  species_id: Id;
  q: number;
  r: number;
  generation: number;
  age_turns: number;
  facing: FacingDirection;
  energy: number;
  health: number;
  max_health: number;
  energy_at_last_sensing: number;
  damage_taken_last_turn: number;
  consumptions_count: number;
  plant_consumptions_count: number;
  prey_consumptions_count: number;
  last_action_taken: ActionType;
  base_metabolic_cost: number;
  brain: BrainStateOf<Id>;
  genome: OrganismGenome;
};

export type ApiOrganismState = OrganismStateOf<ApiScalarId>;
export type OrganismState = OrganismStateOf<OrganismId>;

type WorldOrganismStateOf<Id> = {
  id: Id;
  species_id: Id;
  q: number;
  r: number;
  generation: number;
  age_turns: number;
  facing: FacingDirection;
  energy: number;
  health: number;
  max_health: number;
  damage_taken_last_turn: number;
  consumptions_count: number;
  plant_consumptions_count: number;
  prey_consumptions_count: number;
  visual: VisualProperties;
};

export type ApiWorldOrganismState = WorldOrganismStateOf<ApiScalarId>;
export type WorldOrganismState = WorldOrganismStateOf<OrganismId>;

type FoodStateOf<Id> = {
  id: Id;
  q: number;
  r: number;
  energy: number;
  kind: FoodKind;
  visual: VisualProperties;
};

export type ApiFoodState = FoodStateOf<ApiScalarId>;
export type FoodState = FoodStateOf<FoodId>;

export type EnergyLedgerRow = {
  turn: number;
  organism_energy_before: number;
  organism_energy_after: number;
  food_energy_before: number;
  food_energy_after: number;
  plant_spawn_energy: number;
  passive_metabolism_energy: number;
  action_cost_energy: number;
  food_consumption_debit: number;
  food_consumption_credit: number;
  predation_prey_energy_removed: number;
  predation_energy_credit: number;
  predation_retention_loss: number;
  corpse_source_energy_removed: number;
  corpse_spawn_energy: number;
  corpse_retention_loss: number;
  unrecycled_energy_removed: number;
  removal_adjustment: number;
  organism_residual: number;
  food_residual: number;
  total_residual: number;
  transfer_residual: number;
  residual_tolerance: number;
};

export type ApiMetricsSnapshot = {
  turns: number;
  organisms: number;
  synapse_ops_last_turn: number;
  actions_applied_last_turn: number;
  consumptions_last_turn: number;
  plant_consumptions_last_turn: number;
  predations_last_turn: number;
  total_consumptions: number;
  total_plant_consumptions: number;
  starvations_last_turn: number;
  age_deaths_last_turn: number;
  energy_ledger_last_turn: EnergyLedgerRow;
};

export type MetricsSnapshot = ApiMetricsSnapshot & {
  total_species_created: number;
  species_counts: Record<string, number>;
};

export type TerrainCell = {
  q: number;
  r: number;
  terrain_type: TerrainType;
  visual: VisualProperties;
};

export type ApiTerrainCell = TerrainCell;

export type ApiWorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: ApiWorldOrganismState[];
  foods: ApiFoodState[];
  terrain: ApiTerrainCell[];
  metrics: ApiMetricsSnapshot;
};

export type WorldSnapshot = {
  turn: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: WorldOrganismState[];
  foods: FoodState[];
  terrain: TerrainCell[];
  metrics: MetricsSnapshot;
};

// Result of a mutating world command (create/step/run-to): the world's name +
// its fresh render snapshot.
export type ApiWorldResponse = {
  name: string;
  snapshot: ApiWorldSnapshot;
};

export type WorldResponse = {
  name: string;
  snapshot: WorldSnapshot;
};

export type ChampionPoolEntry = {
  genome: OrganismGenome;
  source_turn: number;
  source_created_at_unix_ms: number;
  generation: number;
  age_turns: number;
  consumptions_count: number;
  energy: number;
};

export type ApiChampionPoolEntry = ChampionPoolEntry;

export type ApiChampionPoolResponse = {
  entries: ApiChampionPoolEntry[];
};

export type ChampionPoolResponse = {
  entries: ChampionPoolEntry[];
};

type EntityIdOf<Id> =
  | { entity_type: 'Organism'; id: Id }
  | { entity_type: 'Food'; id: Id };

export type ApiEntityId = EntityIdOf<ApiScalarId>;
export type EntityId = EntityIdOf<number>;

type RemovedEntityPositionOf<Id> = {
  entity_id: EntityIdOf<Id>;
  q: number;
  r: number;
};

export type ApiRemovedEntityPosition = RemovedEntityPositionOf<ApiScalarId>;
export type RemovedEntityPosition = RemovedEntityPositionOf<number>;

type OrganismMoveOf<Id> = { id: Id; from: [number, number]; to: [number, number] };
type OrganismFacingOf<Id> = { id: Id; facing: FacingDirection };

export type ApiOrganismMove = OrganismMoveOf<ApiScalarId>;
export type ApiOrganismFacing = OrganismFacingOf<ApiScalarId>;
export type OrganismMove = OrganismMoveOf<OrganismId>;
export type OrganismFacing = OrganismFacingOf<OrganismId>;

export type ApiTickDelta = {
  turn: number;
  moves: ApiOrganismMove[];
  facing_updates: ApiOrganismFacing[];
  removed_positions: ApiRemovedEntityPosition[];
  spawned: ApiWorldOrganismState[];
  food_spawned: ApiFoodState[];
  metrics: ApiMetricsSnapshot;
};

// UI tick delta: metrics stay raw until applyTickDelta derives
// species counts from the updated organism list.
export type TickDelta = {
  turn: number;
  moves: OrganismMove[];
  facing_updates: OrganismFacing[];
  removed_positions: RemovedEntityPosition[];
  spawned: WorldOrganismState[];
  food_spawned: FoodState[];
  metrics: ApiMetricsSnapshot;
};

// Overview tiles source this from the renderer's current snapshot.
export type LiveMetricsData = {
  turn: number;
  metrics: MetricsSnapshot;
};

// `/worlds/{name}/organism/{id}`: the full detail of one organism for the
// inspector's brain visualization (was FocusBrainData in the session model).
type OrganismDetailOf<Id> = {
  turn: number;
  organism: OrganismStateOf<Id>;
  active_action_neuron_id: Id | null;
};

export type ApiOrganismDetail = OrganismDetailOf<ApiScalarId>;
export type OrganismDetail = OrganismDetailOf<number>;

export type ApiErrorData = {
  code: string;
  message: string;
};

// Frames pushed over the `/worlds/{name}/stream` WebSocket.
export type ApiStreamFrame =
  | { type: 'StateSnapshot'; data: ApiWorldSnapshot }
  | { type: 'TickDelta'; data: ApiTickDelta };

// ---------------------------------------------------------------------------
// Research reads (CLI-parity JSON reads surfaced in the cockpit). These payloads
// are plain data with no scalar-id newtypes, so the wire and UI types are one
// and the same — no normalization needed.
// ---------------------------------------------------------------------------

export type StatsSummary = {
  n: number;
  min: number;
  p50: number;
  mean: number;
  p90: number;
  max: number;
};

export type PillarIntervalMetric = {
  tick: number;
  action_effectiveness: number | null;
  plant_consumption_rate: number | null;
  prey_consumption_rate: number | null;
  mi_sa: number | null;
  learning_slope: number | null;
  pop: number;
};

export type PillarsView = {
  window_start_tick: number;
  window_end_tick: number;
  intervals: number;
  partial: boolean;
  scaled: boolean;
  plant_consumption_rate: number | null;
  prey_consumption_rate: number | null;
  action_effectiveness: number | null;
  mi_sa: number | null;
  learning_slope: number | null;
  granular: {
    report_every: number;
    window_start_tick: number;
    window_end_tick: number;
    intervals: PillarIntervalMetric[];
  };
};

export type EcoTrajectory = {
  ticks: number;
  population_series: number[];
  food_series: number[];
  births_per_tick: number;
  deaths_per_tick: number;
  deaths_by_cause: {
    total: number;
    starvation: number;
    age: number;
    predation: number;
    other: number;
  };
  consumptions_per_tick: number;
  predations_per_tick: number;
  carrying_capacity_est: number;
};

export type EcoView = {
  turn: number;
  population: number;
  descendants: number;
  food: { plants: number; corpses: number; total_energy: number };
  trajectory: EcoTrajectory | null;
  note?: string;
};

export type LineageView = {
  population: number;
  generation: { stats: StatsSummary | null; histogram: number[] };
  lineages: { distinct: number; top: { species_id: number; count: number; pct: number }[] };
  note?: string;
};

export type GenomeGeneStat = { group: string; stats: StatsSummary | null };
export type GenomeView = {
  population: number;
  genes: Record<string, GenomeGeneStat>;
  drift_note?: string;
};

export type TimeseriesData = Record<string, (number | null)[]>;

export type FindRow = Record<string, number | boolean>;
export type FindResult = { matched: number; shown: number; rows: FindRow[] };
