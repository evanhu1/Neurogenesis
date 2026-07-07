// Wire (`Api*`) types mirror the Rust schema exactly (sim-types + sim-server/protocol).
// UI types are identical except scalar ids are unwrapped to plain numbers; both views
// are generated from shared `...Of<Id>` shapes so they cannot drift apart.

/** Scalar ids may arrive as plain numbers or single-field tuple objects. */
export type ApiScalarId = number | { 0: number };

export type OrganismId = number;
export type SpeciesId = number;
export type FoodId = number;
export type NeuronId = number;

export type RgbColor = {
  r: number;
  g: number;
  b: number;
};

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
  | 'Attack'
  | 'Reproduce';

export type FoodKind = 'Plant' | 'Corpse';
export type TerrainType = 'Spikes' | 'Mountain';
export type VisionChannel = 'Red' | 'Green' | 'Blue' | 'Shape';

export type NeuronType = 'Sensory' | 'Inter' | 'Action';

export type FacingDirection =
  | 'East'
  | 'NorthEast'
  | 'NorthWest'
  | 'West'
  | 'SouthWest'
  | 'SouthEast';

export type TopologyGenes = {
  num_neurons: number;
  num_synapses: number;
  vision_distance: number;
};

export type LifecycleGenes = {
  body_color: RgbColor;
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

export type MutationRateGenes = {
  age_of_maturity: number;
  gestation_ticks: number;
  max_organism_age: number;
  vision_distance: number;
  hebb_eta_gain: number;
  juvenile_eta_scale: number;
  inter_bias: number;
  inter_update_rate: number;
  eligibility_retention: number;
  synapse_prune_threshold: number;
  synapse_weight_perturbation: number;
  add_synapse: number;
  remove_synapse: number;
  remove_neuron: number;
  add_neuron_split_edge: number;
  max_weight_delta_per_tick: number;
};

// SeedGenomeConfig wire format stays flat (Rust SeedGenomeConfig is not nested).
export type SeedGenomeConfig = {
  num_neurons: number;
  num_synapses: number;
  vision_distance: number;
  body_color: RgbColor;
  age_of_maturity: number;
  gestation_ticks: number;
  max_organism_age: number;
  hebb_eta_gain: number;
  juvenile_eta_scale: number;
  eligibility_retention: number;
  max_weight_delta_per_tick: number;
  synapse_prune_threshold: number;
  mutation_rate_age_of_maturity: number;
  mutation_rate_gestation_ticks: number;
  mutation_rate_max_organism_age: number;
  mutation_rate_vision_distance: number;
  mutation_rate_hebb_eta_gain: number;
  mutation_rate_juvenile_eta_scale: number;
  mutation_rate_inter_bias: number;
  mutation_rate_inter_update_rate: number;
  mutation_rate_eligibility_retention: number;
  mutation_rate_synapse_prune_threshold: number;
  mutation_rate_synapse_weight_perturbation: number;
  mutation_rate_add_synapse: number;
  mutation_rate_remove_synapse: number;
  mutation_rate_remove_neuron: number;
  mutation_rate_add_neuron_split_edge: number;
  mutation_rate_max_weight_delta_per_tick: number;
};

export type WorldConfig = {
  world_width: number;
  num_organisms: number;
  periodic_injection_interval_turns: number;
  periodic_injection_count: number;
  food_energy: number;
  passive_metabolism_cost_per_unit: number;
  body_mass_metabolic_cost_coeff: number;
  move_action_energy_cost: number;
  action_temperature: number;
  food_regrowth_interval: number;
  food_regrowth_jitter: number;
  terrain_noise_scale: number;
  terrain_threshold: number;
  spike_density: number;
  global_mutation_rate_modifier: number;
  meta_mutation_enabled: boolean;
  runtime_plasticity_enabled: boolean;
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

// Heritable synapse gene: wiring + weight only (no runtime plasticity state).
type SynapseGeneOf<Id> = {
  pre_neuron_id: Id;
  post_neuron_id: Id;
  weight: number;
};

export type ApiSynapseGene = SynapseGeneOf<ApiScalarId>;
export type SynapseGene = SynapseGeneOf<NeuronId>;

type BrainTopologyGenesOf<Id> = {
  inter_biases: number[];
  inter_log_time_constants: number[];
  action_biases: number[];
  edges: SynapseGeneOf<Id>[];
};

type OrganismGenomeOf<Id> = {
  topology: TopologyGenes;
  lifecycle: LifecycleGenes;
  plasticity: PlasticityGenes;
  mutation_rates: MutationRateGenes;
  brain: BrainTopologyGenesOf<Id>;
};

export type ApiOrganismGenome = OrganismGenomeOf<ApiScalarId>;
export type OrganismGenome = OrganismGenomeOf<NeuronId>;

type NeuronStateOf<Id> = {
  neuron_id: Id;
  neuron_type: NeuronType;
  bias: number;
  activation: number;
};

export type ApiNeuronState = NeuronStateOf<ApiScalarId>;
export type NeuronState = NeuronStateOf<NeuronId>;

export type SensoryReceptor =
  | { receptor_type: 'VisionRay'; ray_offset: number; channel: VisionChannel }
  | { receptor_type: 'ContactAhead' }
  | { receptor_type: 'Energy' }
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
  is_gestating: boolean;
  consumptions_count: number;
  plant_consumptions_count: number;
  prey_consumptions_count: number;
  reproductions_count: number;
  last_action_taken: ActionType;
  base_metabolic_cost: number;
  brain: BrainStateOf<Id>;
  genome: OrganismGenomeOf<Id>;
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
  is_gestating: boolean;
  consumptions_count: number;
  plant_consumptions_count: number;
  prey_consumptions_count: number;
  reproductions_count: number;
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
  age_deaths_last_turn: number;
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

type ChampionPoolEntryOf<Id> = {
  genome: OrganismGenomeOf<Id>;
  source_turn: number;
  source_created_at_unix_ms: number;
  generation: number;
  age_turns: number;
  reproductions_count: number;
  consumptions_count: number;
  energy: number;
};

export type ApiChampionPoolEntry = ChampionPoolEntryOf<ApiScalarId>;
export type ChampionPoolEntry = ChampionPoolEntryOf<NeuronId>;

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

export type ReproductionFailureCause = 'BlockedBirth' | 'ParentDied';

export type ApiReproductionEvent = {
  parent_id: ApiScalarId;
  parent_species_id: ApiScalarId;
  parent_age_turns: number;
  parent_generation: number;
  investment_energy: number;
  parent_energy_after_event: number;
  child_id: ApiScalarId | null;
  failure_cause: ReproductionFailureCause | null;
};

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
  reproduction_events: ApiReproductionEvent[];
  food_spawned: ApiFoodState[];
  metrics: ApiMetricsSnapshot;
};

// UI tick delta: reproduction_events are never read by the client, so
// normalization drops them; metrics stay raw until applyTickDelta derives
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
export type GenomeMutationRate = { mean: number; state: 'hot' | 'cold' };
export type GenomeView = {
  population: number;
  genes: Record<string, GenomeGeneStat>;
  mutation_rates: Record<string, GenomeMutationRate>;
  drift_note?: string;
};

export type TimeseriesData = Record<string, (number | null)[]>;

export type FindRow = Record<string, number | boolean>;
export type FindResult = { matched: number; shown: number; rows: FindRow[] };
