use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use sim_core::{derive_active_action_neuron_id, SimError, Simulation};
use sim_server::{
    load_default_world_config,
    protocol::{
        ApiError, ChampionPoolEntry, ChampionPoolResponse, ClientCommand, CountRequest,
        CreateSessionRequest, CreateSessionResponse, FocusBrainData, FocusRequest, LiveMetricsData,
        ServerEvent, SessionMetadata, StepProgressData, StreamMode, WorldSnapshotView,
    },
};
use sim_types::{
    inter_neuron_id, inter_neuron_index, ActionType, BrainLocation, NeuronId, OrganismGenome,
    OrganismState, RgbColor, SensoryReceptor, SpeciesId, SynapseEdge, WorldSnapshot,
};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    sessions: Arc<RwLock<HashMap<Uuid, Arc<Session>>>>,
    champion_pool: Arc<ChampionPoolStore>,
}

struct Session {
    metadata: SessionMetadata,
    simulation: Mutex<Simulation>,
    events: broadcast::Sender<ServerEvent>,
    runtime: Mutex<RuntimeState>,
}

#[derive(Default)]
struct RuntimeState {
    running: bool,
    ticks_per_second: u32,
    stream_mode: StreamMode,
    runner: Option<JoinHandle<()>>,
}

const CHAMPION_POOL_SCHEMA_VERSION: u32 = 3;
const CHAMPION_POOL_MAX_GENOMES: usize = 32;
const CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD: usize = 32;
const DEFAULT_PERSISTED_BRAIN_LOCATION: BrainLocation = BrainLocation { x: 5.0, y: 5.0 };

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChampionPoolFile {
    schema_version: u32,
    updated_at_unix_ms: u128,
    entries: Vec<PersistedChampionGenomeRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ChampionGenomeRecord {
    genome: OrganismGenome,
    source_turn: u64,
    source_created_at_unix_ms: u128,
    generation: u64,
    age_turns: u64,
    reproductions_count: u64,
    consumptions_count: u64,
    energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PersistedChampionGenomeRecord {
    genome: PersistedOrganismGenome,
    source_turn: u64,
    source_created_at_unix_ms: u128,
    generation: u64,
    age_turns: u64,
    reproductions_count: u64,
    consumptions_count: u64,
    energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PersistedOrganismGenome {
    num_neurons: u32,
    num_synapses: u32,
    spatial_prior_sigma: f32,
    vision_distance: u32,
    body_color: RgbColor,
    max_health: f32,
    age_of_maturity: u32,
    gestation_ticks: u8,
    max_organism_age: u32,
    plasticity_start_age: u32,
    hebb_eta_gain: f32,
    juvenile_eta_scale: f32,
    eligibility_retention: f32,
    max_weight_delta_per_tick: f32,
    synapse_prune_threshold: f32,
    mutation_rate_age_of_maturity: f32,
    mutation_rate_gestation_ticks: f32,
    mutation_rate_max_organism_age: f32,
    mutation_rate_vision_distance: f32,
    mutation_rate_max_health: f32,
    mutation_rate_inter_bias: f32,
    mutation_rate_inter_update_rate: f32,
    mutation_rate_eligibility_retention: f32,
    mutation_rate_synapse_prune_threshold: f32,
    mutation_rate_neuron_location: f32,
    mutation_rate_synapse_weight_perturbation: f32,
    mutation_rate_add_synapse: f32,
    mutation_rate_remove_synapse: f32,
    mutation_rate_remove_neuron: f32,
    mutation_rate_add_neuron_split_edge: f32,
    inter_biases: Vec<f32>,
    inter_log_time_constants: Vec<f32>,
    sensory_locations: Vec<PersistedSensoryLocation>,
    inter_locations: Vec<BrainLocation>,
    action_locations: Vec<PersistedActionLocation>,
    edges: Vec<PersistedSynapseEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PersistedSensoryLocation {
    receptor: SensoryReceptor,
    location: BrainLocation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PersistedActionLocation {
    action_type: ActionType,
    location: BrainLocation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PersistedSynapseEdge {
    pre_neuron: PersistedNeuronRef,
    post_neuron: PersistedNeuronRef,
    weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "neuron_type", rename_all = "snake_case")]
enum PersistedNeuronRef {
    Sensory { receptor: SensoryReceptor },
    Inter { index: u32 },
    Action { action_type: ActionType },
}

struct ChampionPoolStore {
    pool_path: PathBuf,
    entries: StdRwLock<Vec<ChampionGenomeRecord>>,
}

const STEP_PROGRESS_TARGET_BATCHES: u32 = 48;
const STEP_PROGRESS_MIN_BATCH_SIZE: u32 = 32;
const STEP_PROGRESS_MAX_BATCH_SIZE: u32 = 2_048;
const STEP_PROGRESS_TARGET_UPDATES: u32 = 64;
const UNBOUNDED_TICKS_PER_SECOND: u32 = 0;
const METRICS_ONLY_STREAM_INTERVAL_TICKS: u32 = 100;

fn step_batch_size(total_count: u32) -> u32 {
    let target = (total_count / STEP_PROGRESS_TARGET_BATCHES).max(1);
    target
        .clamp(STEP_PROGRESS_MIN_BATCH_SIZE, STEP_PROGRESS_MAX_BATCH_SIZE)
        .min(total_count.max(1))
}

fn step_progress_stride(total_count: u32) -> u32 {
    total_count.saturating_add(STEP_PROGRESS_TARGET_UPDATES.saturating_sub(1))
        / STEP_PROGRESS_TARGET_UPDATES.max(1)
}

fn now_unix_ms() -> Result<u128, AppError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| AppError::Internal(e.to_string()))
        .map(|duration| duration.as_millis())
}

fn champion_pool_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("champion_pool.json")
}

fn load_runtime_default_world_config() -> Result<sim_types::WorldConfig, AppError> {
    load_default_world_config().map_err(|err| {
        AppError::Internal(format!(
            "failed to load {}: {err}",
            sim_server::default_world_config_path().display()
        ))
    })
}

fn build_species_counts(organisms: &[OrganismState]) -> BTreeMap<String, u32> {
    let mut counts = BTreeMap::<SpeciesId, u32>::new();
    for organism in organisms {
        *counts.entry(organism.species_id).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(species_id, count)| (species_id.0.to_string(), count))
        .collect()
}

fn build_live_metrics_data(simulation: &Simulation) -> LiveMetricsData {
    LiveMetricsData {
        turn: simulation.turn(),
        metrics: simulation.metrics().clone(),
        species_counts: build_species_counts(simulation.organisms()),
    }
}

fn compare_champion_records(
    left: &ChampionGenomeRecord,
    right: &ChampionGenomeRecord,
) -> std::cmp::Ordering {
    right
        .generation
        .cmp(&left.generation)
        .then_with(|| right.reproductions_count.cmp(&left.reproductions_count))
        .then_with(|| right.consumptions_count.cmp(&left.consumptions_count))
        .then_with(|| right.energy.total_cmp(&left.energy))
        .then_with(|| right.age_turns.cmp(&left.age_turns))
        .then_with(|| {
            right
                .source_created_at_unix_ms
                .cmp(&left.source_created_at_unix_ms)
        })
}

fn champion_record_from_organism(
    source_created_at_unix_ms: u128,
    source_turn: u64,
    organism: &OrganismState,
) -> ChampionGenomeRecord {
    ChampionGenomeRecord {
        genome: organism.genome.clone(),
        source_turn,
        source_created_at_unix_ms,
        generation: organism.generation,
        age_turns: organism.age_turns,
        reproductions_count: organism.reproductions_count,
        consumptions_count: organism.consumptions_count,
        energy: organism.energy,
    }
}

fn select_champion_candidates(
    source_created_at_unix_ms: u128,
    source_turn: u64,
    organisms: &[OrganismState],
) -> Vec<ChampionGenomeRecord> {
    let mut candidates: Vec<ChampionGenomeRecord> = organisms
        .iter()
        .map(|organism| {
            champion_record_from_organism(source_created_at_unix_ms, source_turn, organism)
        })
        .collect();
    candidates.sort_by(compare_champion_records);

    let mut unique =
        Vec::with_capacity(CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD.min(candidates.len()));
    for candidate in candidates {
        if unique
            .iter()
            .any(|existing: &ChampionGenomeRecord| existing.genome == candidate.genome)
        {
            continue;
        }
        unique.push(candidate);
        if unique.len() >= CHAMPION_POOL_MAX_CANDIDATES_PER_WORLD {
            break;
        }
    }

    unique
}

fn merge_champion_entries(
    existing: &[ChampionGenomeRecord],
    candidates: &[ChampionGenomeRecord],
) -> Vec<ChampionGenomeRecord> {
    let mut merged = Vec::with_capacity(existing.len() + candidates.len());
    merged.extend(existing.iter().cloned());
    merged.extend(candidates.iter().cloned());
    merged.sort_by(compare_champion_records);

    let mut unique = Vec::with_capacity(CHAMPION_POOL_MAX_GENOMES.min(merged.len()));
    for entry in merged {
        if unique
            .iter()
            .any(|existing_entry: &ChampionGenomeRecord| existing_entry.genome == entry.genome)
        {
            continue;
        }
        unique.push(entry);
        if unique.len() >= CHAMPION_POOL_MAX_GENOMES {
            break;
        }
    }

    unique
}

fn encode_champion_record(
    entry: &ChampionGenomeRecord,
) -> Result<PersistedChampionGenomeRecord, String> {
    Ok(PersistedChampionGenomeRecord {
        genome: encode_persisted_genome(&entry.genome)?,
        source_turn: entry.source_turn,
        source_created_at_unix_ms: entry.source_created_at_unix_ms,
        generation: entry.generation,
        age_turns: entry.age_turns,
        reproductions_count: entry.reproductions_count,
        consumptions_count: entry.consumptions_count,
        energy: entry.energy,
    })
}

fn decode_champion_record(entry: PersistedChampionGenomeRecord) -> ChampionGenomeRecord {
    ChampionGenomeRecord {
        genome: decode_persisted_genome(entry.genome),
        source_turn: entry.source_turn,
        source_created_at_unix_ms: entry.source_created_at_unix_ms,
        generation: entry.generation,
        age_turns: entry.age_turns,
        reproductions_count: entry.reproductions_count,
        consumptions_count: entry.consumptions_count,
        energy: entry.energy,
    }
}

fn encode_persisted_genome(genome: &OrganismGenome) -> Result<PersistedOrganismGenome, String> {
    Ok(PersistedOrganismGenome {
        num_neurons: genome.num_neurons,
        num_synapses: genome.num_synapses,
        spatial_prior_sigma: genome.spatial_prior_sigma,
        vision_distance: genome.vision_distance,
        body_color: genome.body_color,
        max_health: genome.max_health,
        age_of_maturity: genome.age_of_maturity,
        gestation_ticks: genome.gestation_ticks,
        max_organism_age: genome.max_organism_age,
        plasticity_start_age: genome.plasticity_start_age,
        hebb_eta_gain: genome.hebb_eta_gain,
        juvenile_eta_scale: genome.juvenile_eta_scale,
        eligibility_retention: genome.eligibility_retention,
        max_weight_delta_per_tick: genome.max_weight_delta_per_tick,
        synapse_prune_threshold: genome.synapse_prune_threshold,
        mutation_rate_age_of_maturity: genome.mutation_rate_age_of_maturity,
        mutation_rate_gestation_ticks: genome.mutation_rate_gestation_ticks,
        mutation_rate_max_organism_age: genome.mutation_rate_max_organism_age,
        mutation_rate_vision_distance: genome.mutation_rate_vision_distance,
        mutation_rate_max_health: genome.mutation_rate_max_health,
        mutation_rate_inter_bias: genome.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: genome.mutation_rate_inter_update_rate,
        mutation_rate_eligibility_retention: genome.mutation_rate_eligibility_retention,
        mutation_rate_synapse_prune_threshold: genome.mutation_rate_synapse_prune_threshold,
        mutation_rate_neuron_location: genome.mutation_rate_neuron_location,
        mutation_rate_synapse_weight_perturbation: genome.mutation_rate_synapse_weight_perturbation,
        mutation_rate_add_synapse: genome.mutation_rate_add_synapse,
        mutation_rate_remove_synapse: genome.mutation_rate_remove_synapse,
        mutation_rate_remove_neuron: genome.mutation_rate_remove_neuron,
        mutation_rate_add_neuron_split_edge: genome.mutation_rate_add_neuron_split_edge,
        inter_biases: genome.inter_biases.clone(),
        inter_log_time_constants: genome.inter_log_time_constants.clone(),
        sensory_locations: genome
            .sensory_locations
            .iter()
            .enumerate()
            .filter_map(|(idx, location)| {
                SensoryReceptor::from_neuron_id(NeuronId(idx as u32)).map(|receptor| {
                    PersistedSensoryLocation {
                        receptor,
                        location: *location,
                    }
                })
            })
            .collect(),
        inter_locations: genome.inter_locations.clone(),
        action_locations: ActionType::ALL
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, action_type)| {
                genome
                    .action_locations
                    .get(idx)
                    .copied()
                    .map(|location| PersistedActionLocation {
                        action_type,
                        location,
                    })
            })
            .collect(),
        edges: genome
            .edges
            .iter()
            .map(|edge| encode_persisted_edge(edge, genome.num_neurons))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn decode_persisted_genome(genome: PersistedOrganismGenome) -> OrganismGenome {
    let sensory_count = SensoryReceptor::ordered().count();
    let action_count = ActionType::ALL.len();
    let mut sensory_locations = vec![DEFAULT_PERSISTED_BRAIN_LOCATION; sensory_count];
    let mut action_locations = vec![DEFAULT_PERSISTED_BRAIN_LOCATION; action_count];

    for sensory_location in genome.sensory_locations {
        if let Some(idx) = sensory_location.receptor.current_index() {
            sensory_locations[idx] = sensory_location.location;
        }
    }

    for action_location in genome.action_locations {
        if let Some(idx) = action_type_index(action_location.action_type) {
            action_locations[idx] = action_location.location;
        }
    }

    let mut edges = genome
        .edges
        .into_iter()
        .filter_map(|edge| decode_persisted_edge(edge, genome.num_neurons))
        .collect::<Vec<_>>();
    edges.sort_unstable_by(|left, right| {
        left.pre_neuron_id
            .cmp(&right.pre_neuron_id)
            .then_with(|| left.post_neuron_id.cmp(&right.post_neuron_id))
            .then_with(|| left.weight.total_cmp(&right.weight))
    });
    edges.dedup_by(|left, right| {
        left.pre_neuron_id == right.pre_neuron_id && left.post_neuron_id == right.post_neuron_id
    });

    OrganismGenome {
        num_neurons: genome.num_neurons,
        num_synapses: edges.len() as u32,
        spatial_prior_sigma: genome.spatial_prior_sigma,
        vision_distance: genome.vision_distance,
        body_color: genome.body_color,
        max_health: genome.max_health,
        age_of_maturity: genome.age_of_maturity,
        gestation_ticks: genome.gestation_ticks,
        max_organism_age: genome.max_organism_age,
        plasticity_start_age: genome.plasticity_start_age,
        hebb_eta_gain: genome.hebb_eta_gain,
        juvenile_eta_scale: genome.juvenile_eta_scale,
        eligibility_retention: genome.eligibility_retention,
        max_weight_delta_per_tick: genome.max_weight_delta_per_tick,
        synapse_prune_threshold: genome.synapse_prune_threshold,
        mutation_rate_age_of_maturity: genome.mutation_rate_age_of_maturity,
        mutation_rate_gestation_ticks: genome.mutation_rate_gestation_ticks,
        mutation_rate_max_organism_age: genome.mutation_rate_max_organism_age,
        mutation_rate_vision_distance: genome.mutation_rate_vision_distance,
        mutation_rate_max_health: genome.mutation_rate_max_health,
        mutation_rate_inter_bias: genome.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: genome.mutation_rate_inter_update_rate,
        mutation_rate_eligibility_retention: genome.mutation_rate_eligibility_retention,
        mutation_rate_synapse_prune_threshold: genome.mutation_rate_synapse_prune_threshold,
        mutation_rate_neuron_location: genome.mutation_rate_neuron_location,
        mutation_rate_synapse_weight_perturbation: genome.mutation_rate_synapse_weight_perturbation,
        mutation_rate_add_synapse: genome.mutation_rate_add_synapse,
        mutation_rate_remove_synapse: genome.mutation_rate_remove_synapse,
        mutation_rate_remove_neuron: genome.mutation_rate_remove_neuron,
        mutation_rate_add_neuron_split_edge: genome.mutation_rate_add_neuron_split_edge,
        inter_biases: genome.inter_biases,
        inter_log_time_constants: genome.inter_log_time_constants,
        sensory_locations,
        inter_locations: genome.inter_locations,
        action_locations,
        edges,
    }
}

fn encode_persisted_edge(
    edge: &SynapseEdge,
    num_neurons: u32,
) -> Result<PersistedSynapseEdge, String> {
    Ok(PersistedSynapseEdge {
        pre_neuron: encode_neuron_ref(edge.pre_neuron_id, num_neurons)?,
        post_neuron: encode_neuron_ref(edge.post_neuron_id, num_neurons)?,
        weight: edge.weight,
    })
}

fn decode_persisted_edge(edge: PersistedSynapseEdge, num_neurons: u32) -> Option<SynapseEdge> {
    Some(SynapseEdge {
        pre_neuron_id: decode_neuron_ref(edge.pre_neuron, num_neurons)?,
        post_neuron_id: decode_neuron_ref(edge.post_neuron, num_neurons)?,
        weight: edge.weight,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    })
}

fn encode_neuron_ref(neuron_id: NeuronId, num_neurons: u32) -> Result<PersistedNeuronRef, String> {
    if let Some(receptor) = SensoryReceptor::from_neuron_id(neuron_id) {
        return Ok(PersistedNeuronRef::Sensory { receptor });
    }
    if let Some(index) = inter_neuron_index(neuron_id, num_neurons) {
        return Ok(PersistedNeuronRef::Inter { index });
    }
    if let Some(action_type) = ActionType::from_neuron_id(neuron_id) {
        return Ok(PersistedNeuronRef::Action { action_type });
    }

    Err(format!("unsupported neuron id {}", neuron_id.0))
}

fn decode_neuron_ref(neuron_ref: PersistedNeuronRef, num_neurons: u32) -> Option<NeuronId> {
    match neuron_ref {
        PersistedNeuronRef::Sensory { receptor } => receptor.neuron_id(),
        PersistedNeuronRef::Inter { index } => {
            (index < num_neurons).then(|| inter_neuron_id(index))
        }
        PersistedNeuronRef::Action { action_type } => action_type.neuron_id(),
    }
}

fn action_type_index(action_type: ActionType) -> Option<usize> {
    ActionType::ALL
        .iter()
        .position(|candidate| *candidate == action_type)
}

impl ChampionPoolStore {
    fn bootstrap(pool_path: PathBuf) -> Result<Self, AppError> {
        if let Some(parent) = pool_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                AppError::Internal(format!(
                    "failed to create champion pool directory {}: {err}",
                    parent.display()
                ))
            })?;
        }

        let entries = match fs::read(&pool_path) {
            Ok(bytes) => match serde_json::from_slice::<ChampionPoolFile>(&bytes) {
                Ok(file) if file.schema_version == CHAMPION_POOL_SCHEMA_VERSION => file
                    .entries
                    .into_iter()
                    .map(decode_champion_record)
                    .collect(),
                Ok(file) => {
                    warn!(
                        "ignoring champion pool {} with schema version {} (expected {})",
                        pool_path.display(),
                        file.schema_version,
                        CHAMPION_POOL_SCHEMA_VERSION
                    );
                    Vec::new()
                }
                Err(err) => {
                    warn!(
                        "failed to parse champion pool {}: {err}",
                        pool_path.display()
                    );
                    Vec::new()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(err) => {
                return Err(AppError::Internal(format!(
                    "failed to read champion pool {}: {err}",
                    pool_path.display()
                )))
            }
        };

        Ok(Self {
            pool_path,
            entries: StdRwLock::new(entries),
        })
    }

    fn snapshot_genomes(&self) -> Result<Vec<OrganismGenome>, AppError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        Ok(entries.iter().map(|entry| entry.genome.clone()).collect())
    }

    fn snapshot_entries(&self) -> Result<Vec<ChampionGenomeRecord>, AppError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        Ok(entries.clone())
    }

    fn clear(&self) -> Result<(), AppError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        self.persist_entries(&[])?;
        entries.clear();
        Ok(())
    }

    fn delete_entry(&self, index: usize) -> Result<Vec<ChampionGenomeRecord>, AppError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        if index >= entries.len() {
            return Err(AppError::NotFound(format!(
                "champion pool entry {index} not found"
            )));
        }
        entries.remove(index);
        self.persist_entries(entries.as_slice())?;
        Ok(entries.clone())
    }

    fn update_from_simulation(&self, simulation: &Simulation) -> Result<(), AppError> {
        let source_created_at_unix_ms = now_unix_ms()?;
        let candidates = select_champion_candidates(
            source_created_at_unix_ms,
            simulation.turn(),
            simulation.organisms(),
        );
        if candidates.is_empty() {
            return Ok(());
        }

        let mut entries = self
            .entries
            .write()
            .map_err(|_| AppError::Internal("failed to lock champion pool".to_owned()))?;
        let merged = merge_champion_entries(entries.as_slice(), &candidates);
        self.persist_entries(&merged)?;
        *entries = merged;
        Ok(())
    }

    fn persist_entries(&self, entries: &[ChampionGenomeRecord]) -> Result<(), AppError> {
        let file = ChampionPoolFile {
            schema_version: CHAMPION_POOL_SCHEMA_VERSION,
            updated_at_unix_ms: now_unix_ms()?,
            entries: entries
                .iter()
                .map(encode_champion_record)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| {
                    AppError::Internal(format!(
                        "failed to encode champion pool entries {}: {err}",
                        self.pool_path.display()
                    ))
                })?,
        };
        let encoded = serde_json::to_vec_pretty(&file).map_err(|err| {
            AppError::Internal(format!(
                "failed to encode champion pool {}: {err}",
                self.pool_path.display()
            ))
        })?;

        let temp_path = self.pool_path.with_extension("json.tmp");
        fs::write(&temp_path, encoded).map_err(|err| {
            AppError::Internal(format!(
                "failed to write champion pool temp file {}: {err}",
                temp_path.display()
            ))
        })?;
        fs::rename(&temp_path, &self.pool_path).map_err(|err| {
            AppError::Internal(format!(
                "failed to finalize champion pool {}: {err}",
                self.pool_path.display()
            ))
        })?;
        Ok(())
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct StepResponse {
    snapshot: WorldSnapshot,
}

impl From<ChampionGenomeRecord> for ChampionPoolEntry {
    fn from(entry: ChampionGenomeRecord) -> Self {
        Self {
            genome: entry.genome,
            source_turn: entry.source_turn,
            source_created_at_unix_ms: entry.source_created_at_unix_ms,
            generation: entry.generation,
            age_turns: entry.age_turns,
            reproductions_count: entry.reproductions_count,
            consumptions_count: entry.consumptions_count,
            energy: entry.energy,
        }
    }
}

#[derive(Debug)]
enum AppError {
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::NotFound(message)
            | AppError::BadRequest(message)
            | AppError::Internal(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AppError {}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal", msg),
        };

        let error = ApiError {
            code: code.to_owned(),
            message,
        };

        (status, Json(error)).into_response()
    }
}

impl From<SimError> for AppError {
    fn from(value: SimError) -> Self {
        AppError::BadRequest(value.to_string())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "sim_server=info,axum=info,tower_http=info".to_owned()),
        )
        .init();

    let app = build_app(new_state()?);

    let addr = std::env::var("SIM_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("sim-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn new_state() -> Result<AppState, AppError> {
    let champion_pool = Arc::new(ChampionPoolStore::bootstrap(champion_pool_path())?);
    Ok(AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        champion_pool,
    })
}

fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route(
            "/v1/champion-pool",
            get(get_champion_pool).delete(clear_champion_pool),
        )
        .route(
            "/v1/champion-pool/{index}",
            delete(delete_champion_pool_entry),
        )
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/{id}", get(get_session_metadata))
        .route("/v1/sessions/{id}/state", get(get_state))
        .route("/v1/sessions/{id}/step", post(step_session))
        .route("/v1/sessions/{id}/champions", post(save_session_champions))
        .route("/v1/sessions/{id}/focus", post(set_focus))
        .route("/v1/sessions/{id}/stream", get(stream_session))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn get_champion_pool(
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    let entries = state.champion_pool.snapshot_entries()?;
    Ok(Json(ChampionPoolResponse {
        entries: entries.into_iter().map(ChampionPoolEntry::from).collect(),
    }))
}

async fn clear_champion_pool(
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    state.champion_pool.clear()?;
    Ok(Json(ChampionPoolResponse {
        entries: Vec::new(),
    }))
}

async fn delete_champion_pool_entry(
    Path(index): Path<usize>,
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    let entries = state.champion_pool.delete_entry(index)?;
    Ok(Json(ChampionPoolResponse {
        entries: entries.into_iter().map(ChampionPoolEntry::from).collect(),
    }))
}

async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    let config = load_runtime_default_world_config()?;
    let champion_pool = state.champion_pool.snapshot_genomes()?;
    let simulation = Simulation::new_with_champion_pool(config, req.seed, champion_pool)?;
    let response = create_runtime_session(&state, simulation).await?;
    Ok(Json(response))
}

async fn create_runtime_session(
    state: &AppState,
    simulation: Simulation,
) -> Result<CreateSessionResponse, AppError> {
    let now_ms = now_unix_ms()?;
    let id = Uuid::new_v4();
    let snapshot = simulation.snapshot();
    let ticks_per_second = UNBOUNDED_TICKS_PER_SECOND;
    let metadata = SessionMetadata {
        id,
        created_at_unix_ms: now_ms,
        config: simulation.config().clone(),
        running: false,
        ticks_per_second,
        stream_mode: StreamMode::Full,
    };

    let (events_tx, _events_rx) = broadcast::channel(1024);
    let session = Arc::new(Session {
        metadata: metadata.clone(),
        simulation: Mutex::new(simulation),
        events: events_tx,
        runtime: Mutex::new(RuntimeState {
            running: false,
            ticks_per_second,
            stream_mode: StreamMode::Full,
            runner: None,
        }),
    });

    let mut sessions = state.sessions.write().await;
    sessions.insert(id, session);

    Ok(CreateSessionResponse {
        metadata,
        snapshot: snapshot.into(),
    })
}

async fn get_session_metadata(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<SessionMetadata>, AppError> {
    let session = get_session(&state, id).await?;
    let mut metadata = session.metadata.clone();
    let runtime = session.runtime.lock().await;
    metadata.running = runtime.running;
    metadata.ticks_per_second = runtime.ticks_per_second;
    metadata.stream_mode = runtime.stream_mode;
    Ok(Json(metadata))
}

async fn get_state(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let session = get_session(&state, id).await?;
    let sim = session.simulation.lock().await;
    Ok(Json(sim.snapshot().into()))
}

async fn step_session(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<CountRequest>,
) -> Result<Json<StepResponse>, AppError> {
    let session = get_session(&state, id).await?;
    let mut sim = session.simulation.lock().await;
    sim.advance_n(req.count.max(1));
    let snapshot = sim.snapshot();
    drop(sim);

    let _ = session
        .events
        .send(ServerEvent::StateSnapshot(snapshot.clone().into()));

    Ok(Json(StepResponse { snapshot }))
}

async fn save_session_champions(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<ChampionPoolResponse>, AppError> {
    let session = get_session(&state, id).await?;
    let simulation = {
        let sim = session.simulation.lock().await;
        sim.clone()
    };
    let champion_pool_store = state.champion_pool.clone();
    let entries = tokio::task::spawn_blocking(move || {
        if let Err(err) = champion_pool_store.update_from_simulation(&simulation) {
            warn!("failed to update champion pool from session {id}: {err}");
        }
        champion_pool_store.snapshot_entries()
    })
    .await
    .map_err(|err| {
        AppError::Internal(format!("session champion-save worker join error: {err}"))
    })??;

    Ok(Json(ChampionPoolResponse {
        entries: entries.into_iter().map(ChampionPoolEntry::from).collect(),
    }))
}

async fn set_focus(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<FocusRequest>,
) -> Result<Json<WorldSnapshotView>, AppError> {
    let session = get_session(&state, id).await?;
    let sim = session.simulation.lock().await;
    if let Some(org) = sim.focused_organism(req.organism_id) {
        let active = derive_active_action_neuron_id(&org);
        let _ = session.events.send(ServerEvent::FocusBrain(FocusBrainData {
            turn: sim.turn(),
            organism: org,
            active_action_neuron_id: active,
        }));
    }
    let snapshot = sim.snapshot();
    drop(sim);

    Ok(Json(snapshot.into()))
}

async fn stream_session(
    ws: WebSocketUpgrade,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    let session = get_session(&state, id).await?;
    Ok(ws
        .on_upgrade(move |socket| socket_loop(socket, session))
        .into_response())
}

async fn socket_loop(socket: WebSocket, session: Arc<Session>) {
    let mut rx = session.events.subscribe();
    let (mut sender, mut receiver) = socket.split();

    {
        let sim = session.simulation.lock().await;
        let runtime = session.runtime.lock().await;
        let event = match runtime.stream_mode {
            StreamMode::Full => ServerEvent::StateSnapshot(sim.snapshot().into()),
            StreamMode::MetricsOnly => ServerEvent::Metrics(build_live_metrics_data(&sim)),
        };
        if let Ok(text) = serde_json::to_string(&event) {
            if sender.send(Message::Text(text.into())).await.is_err() {
                return;
            }
        }
    }

    let session_for_send = session.clone();
    let send_task = tokio::spawn(async move {
        loop {
            let event = match rx.recv().await {
                Ok(event) => event,
                Err(RecvError::Lagged(skipped)) => {
                    error!("ws receiver lagged by {skipped} events; sending state snapshot");
                    let refresh_event = {
                        let sim = session_for_send.simulation.lock().await;
                        let runtime = session_for_send.runtime.lock().await;
                        match runtime.stream_mode {
                            StreamMode::Full => ServerEvent::StateSnapshot(sim.snapshot().into()),
                            StreamMode::MetricsOnly => {
                                ServerEvent::Metrics(build_live_metrics_data(&sim))
                            }
                        }
                    };
                    if send_ws_event(&mut sender, &refresh_event).await.is_err() {
                        break;
                    }
                    continue;
                }
                Err(RecvError::Closed) => break,
            };

            if send_ws_event(&mut sender, &event).await.is_err() {
                break;
            }
        }
    });

    while let Some(message) = receiver.next().await {
        match message {
            Ok(Message::Text(text)) => {
                let command = match serde_json::from_str::<ClientCommand>(&text) {
                    Ok(cmd) => cmd,
                    Err(err) => {
                        let _ = session.events.send(ServerEvent::Error(ApiError {
                            code: "bad_command".to_owned(),
                            message: format!("failed to parse command: {err}"),
                        }));
                        continue;
                    }
                };

                if let Err(err) = handle_command(command, session.clone()).await {
                    let _ = session.events.send(ServerEvent::Error(ApiError {
                        code: "command_error".to_owned(),
                        message: err,
                    }));
                }
            }
            Ok(Message::Binary(_)) => {}
            Ok(Message::Ping(_)) => {}
            Ok(Message::Pong(_)) => {}
            Ok(Message::Close(_)) => break,
            Err(err) => {
                error!("ws receive error: {err}");
                break;
            }
        }
    }

    send_task.abort();
}

async fn handle_command(command: ClientCommand, session: Arc<Session>) -> Result<(), String> {
    match command {
        ClientCommand::Start {
            ticks_per_second,
            stream_mode,
        } => {
            session_start(session, ticks_per_second, stream_mode).await;
            Ok(())
        }
        ClientCommand::Pause => {
            session_pause(&session).await;
            Ok(())
        }
        ClientCommand::Step { count } => {
            let requested_count = count.max(1);
            let _ = session
                .events
                .send(ServerEvent::StepProgress(StepProgressData {
                    requested_count,
                    completed_count: 0,
                }));

            let batch_size = step_batch_size(requested_count);
            let progress_stride = step_progress_stride(requested_count).max(1);
            let mut sim = session.simulation.lock().await;
            let mut completed_count = 0;
            let mut next_progress_emit = progress_stride;
            while completed_count < requested_count {
                let remaining = requested_count.saturating_sub(completed_count);
                let batch_count = remaining.min(batch_size);
                sim.advance_n(batch_count);
                completed_count = completed_count.saturating_add(batch_count);

                let is_final_batch = completed_count == requested_count;
                if is_final_batch || completed_count >= next_progress_emit {
                    next_progress_emit = completed_count.saturating_add(progress_stride);
                    let _ = session
                        .events
                        .send(ServerEvent::StepProgress(StepProgressData {
                            requested_count,
                            completed_count,
                        }));
                    tokio::task::yield_now().await;
                }
            }
            let snapshot = sim.snapshot();
            drop(sim);
            let _ = session
                .events
                .send(ServerEvent::StateSnapshot(snapshot.into()));
            Ok(())
        }
        ClientCommand::SetFocus { organism_id } => {
            let sim = session.simulation.lock().await;
            if let Some(organism) = sim.focused_organism(organism_id) {
                let active = derive_active_action_neuron_id(&organism);
                let _ = session.events.send(ServerEvent::FocusBrain(FocusBrainData {
                    turn: sim.turn(),
                    organism,
                    active_action_neuron_id: active,
                }));
            }
            Ok(())
        }
    }
}

async fn session_start(session: Arc<Session>, ticks_per_second: u32, stream_mode: StreamMode) {
    let mut runtime = session.runtime.lock().await;
    runtime.ticks_per_second = ticks_per_second;
    runtime.stream_mode = stream_mode;

    if runtime.running {
        return;
    }

    runtime.running = true;
    let session_for_task = session.clone();
    runtime.runner = Some(tokio::spawn(async move {
        loop {
            let (tps, stream_mode) = {
                let rt = session_for_task.runtime.lock().await;
                if !rt.running {
                    break;
                }
                (rt.ticks_per_second, rt.stream_mode)
            };

            match stream_mode {
                StreamMode::Full => {
                    let delta = {
                        let mut sim = session_for_task.simulation.lock().await;
                        sim.step_n(1).into_iter().next()
                    };

                    if let Some(delta) = delta {
                        let _ = session_for_task
                            .events
                            .send(ServerEvent::TickDelta(delta.into()));
                    }
                }
                StreamMode::MetricsOnly => {
                    let metrics = {
                        let mut sim = session_for_task.simulation.lock().await;
                        sim.advance_n(METRICS_ONLY_STREAM_INTERVAL_TICKS);
                        build_live_metrics_data(&sim)
                    };
                    let _ = session_for_task.events.send(ServerEvent::Metrics(metrics));
                }
            }

            if tps > 0 {
                tokio::time::sleep(Duration::from_millis((1000_u64 / tps as u64).max(1))).await;
            } else {
                tokio::task::yield_now().await;
            }
        }

        let mut rt = session_for_task.runtime.lock().await;
        rt.running = false;
        rt.runner = None;
    }));
}

async fn send_ws_event(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    event: &ServerEvent,
) -> Result<(), ()> {
    match serde_json::to_string(event) {
        Ok(text) => sender
            .send(Message::Text(text.into()))
            .await
            .map_err(|_| ()),
        Err(err) => {
            error!("failed to serialize server event: {err}");
            Ok(())
        }
    }
}

async fn session_pause(session: &Arc<Session>) {
    let mut runtime = session.runtime.lock().await;
    runtime.running = false;
    if let Some(handle) = runtime.runner.take() {
        handle.abort();
    }
}

async fn get_session(state: &AppState, id: Uuid) -> Result<Arc<Session>, AppError> {
    let sessions = state.sessions.read().await;
    sessions
        .get(&id)
        .cloned()
        .ok_or_else(|| AppError::NotFound(format!("session {id} not found")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{ActionType, BrainState, FacingDirection, OrganismId, SpeciesId};

    fn make_record(
        generation: u64,
        reproductions: u64,
        consumptions: u64,
        energy: f32,
    ) -> ChampionGenomeRecord {
        let mut genome = OrganismGenome {
            num_neurons: 1,
            num_synapses: 0,
            spatial_prior_sigma: 3.5,
            vision_distance: 2,
            body_color: RgbColor {
                r: 0.3,
                g: 0.6,
                b: 0.9,
            },
            max_health: energy.max(1.0),
            age_of_maturity: 0,
            gestation_ticks: 2,
            max_organism_age: u32::MAX,
            plasticity_start_age: 0,
            hebb_eta_gain: 0.0,
            juvenile_eta_scale: 0.5,
            eligibility_retention: 0.9,
            max_weight_delta_per_tick: 0.05,
            synapse_prune_threshold: 0.01,
            mutation_rate_age_of_maturity: 0.0,
            mutation_rate_gestation_ticks: 0.0,
            mutation_rate_max_organism_age: 0.0,
            mutation_rate_vision_distance: 0.0,
            mutation_rate_max_health: 0.0,
            mutation_rate_inter_bias: 0.0,
            mutation_rate_inter_update_rate: 0.0,
            mutation_rate_eligibility_retention: 0.0,
            mutation_rate_synapse_prune_threshold: 0.0,
            mutation_rate_neuron_location: 0.0,
            mutation_rate_synapse_weight_perturbation: 0.0,
            mutation_rate_add_synapse: 0.0,
            mutation_rate_remove_synapse: 0.0,
            mutation_rate_remove_neuron: 0.0,
            mutation_rate_add_neuron_split_edge: 0.0,
            inter_biases: vec![0.0],
            inter_log_time_constants: vec![0.0],
            sensory_locations: vec![
                sim_types::BrainLocation { x: 0.0, y: 0.0 };
                sim_types::SensoryReceptor::ordered().count()
            ],
            inter_locations: vec![sim_types::BrainLocation { x: 1.0, y: 1.0 }],
            action_locations: vec![
                sim_types::BrainLocation { x: 2.0, y: 2.0 };
                ActionType::ALL.len()
            ],
            edges: Vec::new(),
        };
        genome.num_neurons = generation as u32 + 1;
        genome.vision_distance = generation as u32 + 2;
        genome.inter_biases = vec![generation as f32];
        genome.inter_log_time_constants = vec![generation as f32 * 0.1];
        genome.inter_locations = vec![sim_types::BrainLocation {
            x: generation as f32,
            y: generation as f32,
        }];

        ChampionGenomeRecord {
            genome,
            source_turn: generation * 10,
            source_created_at_unix_ms: generation as u128,
            generation,
            age_turns: generation * 2,
            reproductions_count: reproductions,
            consumptions_count: consumptions,
            energy,
        }
    }

    #[test]
    fn merge_champion_entries_prefers_higher_ranked_unique_genomes() {
        let weak = make_record(3, 1, 1, 10.0);
        let strong = make_record(8, 5, 9, 80.0);
        let duplicate_strong = strong.clone();

        let merged = merge_champion_entries(&[weak.clone()], &[duplicate_strong, strong.clone()]);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], strong);
        assert_eq!(merged[1], weak);
    }

    #[test]
    fn select_champion_candidates_deduplicates_identical_genomes() {
        let strong = make_record(10, 4, 2, 40.0);
        let weaker_duplicate = ChampionGenomeRecord {
            energy: 5.0,
            generation: 2,
            ..strong.clone()
        };
        let empty_brain = BrainState {
            sensory: Vec::new(),
            inter: Vec::new(),
            action: Vec::new(),
            synapse_count: 0,
        };

        let organisms = vec![
            OrganismState::new(
                OrganismId(1),
                SpeciesId(1),
                0,
                0,
                strong.generation,
                strong.age_turns,
                FacingDirection::East,
                strong.energy,
                strong.energy.max(1.0),
                strong.energy.max(1.0),
                strong.energy,
                0.0,
                0.0,
                false,
                strong.consumptions_count,
                0,
                0,
                strong.reproductions_count,
                ActionType::Idle,
                empty_brain.clone(),
                strong.genome.clone(),
            ),
            OrganismState::new(
                OrganismId(2),
                SpeciesId(2),
                1,
                0,
                weaker_duplicate.generation,
                weaker_duplicate.age_turns,
                FacingDirection::East,
                weaker_duplicate.energy,
                weaker_duplicate.energy.max(1.0),
                weaker_duplicate.energy.max(1.0),
                weaker_duplicate.energy,
                0.0,
                0.0,
                false,
                weaker_duplicate.consumptions_count,
                0,
                0,
                weaker_duplicate.reproductions_count,
                ActionType::Idle,
                empty_brain,
                weaker_duplicate.genome,
            ),
        ];

        let candidates = select_champion_candidates(1, 100, &organisms);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].generation, strong.generation);
        assert_eq!(candidates[0].energy, strong.energy);
    }

    #[test]
    fn champion_persistence_round_trip_preserves_brain_layout_and_edges() {
        let food_receptor = SensoryReceptor::VisionRay {
            ray_offset: 0,
            channel: sim_types::VisionChannel::Green,
        };
        let blue_receptor = SensoryReceptor::VisionRay {
            ray_offset: 0,
            channel: sim_types::VisionChannel::Blue,
        };
        let forward_action = ActionType::Forward;
        let attack_action = ActionType::Attack;

        let mut record = make_record(4, 2, 3, 25.0);
        record.genome.num_neurons = 2;
        record.genome.inter_biases = vec![0.1, -0.2];
        record.genome.inter_log_time_constants = vec![0.0, 0.3];
        record.genome.inter_locations = vec![
            BrainLocation { x: 1.0, y: 2.0 },
            BrainLocation { x: 3.0, y: 4.0 },
        ];
        record.genome.sensory_locations =
            vec![BrainLocation { x: 0.0, y: 0.0 }; SensoryReceptor::ordered().count()];
        record.genome.sensory_locations[food_receptor.current_index().unwrap()] =
            BrainLocation { x: 7.0, y: 8.0 };
        record.genome.sensory_locations[blue_receptor.current_index().unwrap()] =
            BrainLocation { x: 9.0, y: 1.5 };
        record.genome.action_locations[action_type_index(forward_action).unwrap()] =
            BrainLocation { x: 6.0, y: 6.5 };
        record.genome.action_locations[action_type_index(attack_action).unwrap()] =
            BrainLocation { x: 8.0, y: 3.5 };
        record.genome.edges = vec![
            SynapseEdge {
                pre_neuron_id: food_receptor.neuron_id().unwrap(),
                post_neuron_id: inter_neuron_id(0),
                weight: 0.75,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
            SynapseEdge {
                pre_neuron_id: blue_receptor.neuron_id().unwrap(),
                post_neuron_id: attack_action.neuron_id().unwrap(),
                weight: -0.5,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
            SynapseEdge {
                pre_neuron_id: inter_neuron_id(0),
                post_neuron_id: forward_action.neuron_id().unwrap(),
                weight: 1.25,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
        ];
        record.genome.num_synapses = record.genome.edges.len() as u32;

        let persisted = encode_champion_record(&record).expect("record should encode");
        let decoded = decode_champion_record(persisted);

        assert_eq!(decoded, record);
    }

    #[test]
    fn champion_persistence_json_uses_symbolic_neuron_refs() {
        let food_receptor = SensoryReceptor::VisionRay {
            ray_offset: 0,
            channel: sim_types::VisionChannel::Green,
        };
        let mut record = make_record(2, 1, 1, 10.0);
        record.genome.edges = vec![SynapseEdge {
            pre_neuron_id: food_receptor.neuron_id().unwrap(),
            post_neuron_id: ActionType::Forward.neuron_id().unwrap(),
            weight: 0.9,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        }];
        record.genome.num_synapses = 1;

        let persisted = encode_champion_record(&record).expect("record should encode");
        let json = serde_json::to_value(&persisted).expect("persisted record should serialize");

        assert_eq!(
            json["genome"]["edges"][0]["pre_neuron"]["neuron_type"],
            "sensory"
        );
        assert_eq!(
            json["genome"]["edges"][0]["pre_neuron"]["receptor"]["receptor_type"],
            "VisionRay"
        );
        assert_eq!(
            json["genome"]["edges"][0]["post_neuron"]["neuron_type"],
            "action"
        );
        assert_eq!(
            json["genome"]["edges"][0]["post_neuron"]["action_type"],
            "Forward"
        );
        assert!(json["genome"]["edges"][0]["pre_neuron_id"].is_null());
        assert!(json["genome"]["edges"][0]["post_neuron_id"].is_null());
    }
}
