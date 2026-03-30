use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const DEFAULT_WORLD_CONFIG_REL_PATH: &str = "config.toml";

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct PopulationDefaults {
    pub periodic_injection_interval_turns: u32,
    pub periodic_injection_count: u32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct LifecycleDefaults {
    pub reproduction_investment_energy: f32,
    pub action_temperature: f32,
    pub intent_parallel_threads: u32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct FoodRegrowthDefaults {
    pub interval: u32,
    pub jitter: u32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct TerrainDefaults {
    pub terrain_noise_scale: f32,
    pub terrain_threshold: f32,
    pub spike_density: f32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct EvolutionDefaults {
    pub global_mutation_rate_modifier: f32,
    pub meta_mutation_enabled: bool,
    pub runtime_plasticity_enabled: bool,
    pub force_random_actions: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct WorldConfigDefaults {
    pub population: PopulationDefaults,
    pub lifecycle: LifecycleDefaults,
    pub food_regrowth: FoodRegrowthDefaults,
    pub terrain: TerrainDefaults,
    pub evolution: EvolutionDefaults,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct SeedGenomeConfigDefaults {
    pub starting_energy: f32,
    pub plasticity_start_age: u32,
    pub juvenile_eta_scale: f32,
    pub max_weight_delta_per_tick: f32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct DerivedWorldPolicy {
    pub max_organism_age_per_world_width: u32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct TerrainGenerationPolicy {
    pub terrain_seed_mix: u64,
    pub default_threshold: f64,
    pub spike_seed_mix: u64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct FoodEcologyPolicy {
    pub fertility_seed_mix: u64,
    pub fertility_jitter_seed_mix: u64,
    pub fertility_noise_scale: f64,
    pub fertility_threshold: f64,
}

pub fn world_config_defaults() -> WorldConfigDefaults {
    WorldConfigDefaults {
        population: PopulationDefaults {
            periodic_injection_interval_turns: 100,
            periodic_injection_count: 100,
        },
        lifecycle: LifecycleDefaults {
            reproduction_investment_energy: 500.0,
            action_temperature: 0.5,
            intent_parallel_threads: 8,
        },
        food_regrowth: FoodRegrowthDefaults {
            interval: 10,
            jitter: 2,
        },
        terrain: TerrainDefaults {
            terrain_noise_scale: 0.02,
            terrain_threshold: 0.86,
            spike_density: 0.10,
        },
        evolution: EvolutionDefaults {
            global_mutation_rate_modifier: 1.0,
            meta_mutation_enabled: true,
            runtime_plasticity_enabled: true,
            force_random_actions: false,
        },
    }
}

pub fn seed_genome_config_defaults() -> SeedGenomeConfigDefaults {
    SeedGenomeConfigDefaults {
        starting_energy: 1.0,
        plasticity_start_age: 0,
        juvenile_eta_scale: 0.5,
        max_weight_delta_per_tick: 0.05,
    }
}

pub fn derived_world_policy() -> DerivedWorldPolicy {
    DerivedWorldPolicy {
        max_organism_age_per_world_width: 5,
    }
}

pub fn terrain_generation_policy() -> TerrainGenerationPolicy {
    TerrainGenerationPolicy {
        terrain_seed_mix: 0xA5A5_A5A5_u64,
        default_threshold: 0.86,
        spike_seed_mix: 0xCBBB_9D5D_C105_9ED8,
    }
}

pub fn food_ecology_policy() -> FoodEcologyPolicy {
    FoodEcologyPolicy {
        fertility_seed_mix: 0x6A09_E667_F3BC_C909,
        fertility_jitter_seed_mix: 0x510E_527F_9B05_688C,
        fertility_noise_scale: 0.012,
        fertility_threshold: 0.6,
    }
}

pub fn world_config_reference_markdown() -> String {
    let world = world_config_defaults();
    let genome = seed_genome_config_defaults();
    let derived = derived_world_policy();
    let terrain_policy = terrain_generation_policy();
    let food_policy = food_ecology_policy();

    let mut out = String::new();
    out.push_str("# Config Reference\n\n");
    out.push_str("Generated from `sim-config/src/config.rs`.\n\n");

    push_doc_section(
        &mut out,
        "World Config",
        "Population",
        &vec![
            (
                "world_width",
                "required".to_owned(),
                "Toroidal world width used for both axial dimensions.",
            ),
            (
                "num_organisms",
                "required".to_owned(),
                "Target initial population before terrain limits are applied.",
            ),
            (
                "periodic_injection_interval_turns",
                world
                    .population
                    .periodic_injection_interval_turns
                    .to_string(),
                "Cadence for seed-genome injections when omitted from TOML.",
            ),
            (
                "periodic_injection_count",
                world.population.periodic_injection_count.to_string(),
                "How many injection attempts run at each periodic-injection turn.",
            ),
            (
                "max_organism_age",
                format!(
                    "derived: world_width * {}",
                    derived.max_organism_age_per_world_width
                ),
                "Inserted during TOML normalization when omitted.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "World Config",
        "Lifecycle And Actions",
        &vec![
            (
                "food_energy",
                "required".to_owned(),
                "Energy stored in each plant food item.",
            ),
            (
                "move_action_energy_cost",
                "required".to_owned(),
                "Flat energy cost applied to non-idle actions.",
            ),
            (
                "reproduction_investment_energy",
                world.lifecycle.reproduction_investment_energy.to_string(),
                "Immediate parent energy investment when reproduction starts.",
            ),
            (
                "action_temperature",
                world.lifecycle.action_temperature.to_string(),
                "Softmax temperature for action sampling.",
            ),
            (
                "intent_parallel_threads",
                world.lifecycle.intent_parallel_threads.to_string(),
                "Worker-count default for intent evaluation.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "World Config",
        "Food Ecology",
        &vec![
            (
                "food_regrowth_interval",
                world.food_regrowth.interval.to_string(),
                "Base plant regrowth delay in turns.",
            ),
            (
                "food_regrowth_jitter",
                world.food_regrowth.jitter.to_string(),
                "Uniform +/- jitter applied to regrowth delay.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "World Config",
        "Terrain",
        &vec![
            (
                "terrain_noise_scale",
                world.terrain.terrain_noise_scale.to_string(),
                "Perlin input scale for terrain walls.",
            ),
            (
                "terrain_threshold",
                world.terrain.terrain_threshold.to_string(),
                "Wall cutoff used by default terrain generation.",
            ),
            (
                "spike_density",
                world.terrain.spike_density.to_string(),
                "Fraction of non-wall cells assigned spikes by deterministic per-cell scatter.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "World Config",
        "Evolution Features",
        &vec![
            (
                "global_mutation_rate_modifier",
                world.evolution.global_mutation_rate_modifier.to_string(),
                "Global multiplier applied to mutation-rate genes.",
            ),
            (
                "meta_mutation_enabled",
                bool_label(world.evolution.meta_mutation_enabled).to_owned(),
                "Enables mutation of the mutation-rate genes themselves.",
            ),
            (
                "runtime_plasticity_enabled",
                bool_label(world.evolution.runtime_plasticity_enabled).to_owned(),
                "Master toggle for mature runtime plasticity updates.",
            ),
            (
                "force_random_actions",
                bool_label(world.evolution.force_random_actions).to_owned(),
                "Validation/debug override for random policy execution.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "Seed Genome Config",
        "Core",
        &vec![
            (
                "num_neurons",
                "required".to_owned(),
                "Inter-neuron count for seed genomes.",
            ),
            (
                "num_synapses",
                "required".to_owned(),
                "Requested synapse count before sanitization.",
            ),
            (
                "spatial_prior_sigma",
                "required".to_owned(),
                "Spatial connectivity prior width.",
            ),
            (
                "vision_distance",
                "required".to_owned(),
                "Maximum look distance for look-ray sensors.",
            ),
            (
                "starting_energy",
                genome.starting_energy.to_string(),
                "Default starting energy when omitted in seed-genome TOML.",
            ),
            (
                "age_of_maturity",
                "required".to_owned(),
                "Age threshold for adulthood.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "Seed Genome Config",
        "Plasticity",
        &vec![
            (
                "plasticity_start_age",
                genome.plasticity_start_age.to_string(),
                "Default age when runtime plasticity may begin.",
            ),
            (
                "hebb_eta_gain",
                "required".to_owned(),
                "Base Hebbian learning-rate gain.",
            ),
            (
                "juvenile_eta_scale",
                genome.juvenile_eta_scale.to_string(),
                "Scaling factor for juvenile plasticity when enabled.",
            ),
            (
                "eligibility_retention",
                "required".to_owned(),
                "Eligibility trace retention factor.",
            ),
            (
                "max_weight_delta_per_tick",
                genome.max_weight_delta_per_tick.to_string(),
                "Clamp for per-tick synapse updates.",
            ),
            (
                "synapse_prune_threshold",
                "required".to_owned(),
                "Pruning threshold for mature synapses.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "Seed Genome Config",
        "Mutation Rates",
        &vec![
            (
                "mutation_rate_age_of_maturity",
                "required".to_owned(),
                "Mutation rate for maturity age.",
            ),
            (
                "mutation_rate_vision_distance",
                "required".to_owned(),
                "Mutation rate for vision distance.",
            ),
            (
                "mutation_rate_inter_bias",
                "required".to_owned(),
                "Mutation rate for inter biases.",
            ),
            (
                "mutation_rate_inter_update_rate",
                "required".to_owned(),
                "Mutation rate for inter-neuron time constants.",
            ),
            (
                "mutation_rate_eligibility_retention",
                "required".to_owned(),
                "Mutation rate for eligibility retention.",
            ),
            (
                "mutation_rate_synapse_prune_threshold",
                "required".to_owned(),
                "Mutation rate for prune thresholds.",
            ),
            (
                "mutation_rate_neuron_location",
                "required".to_owned(),
                "Mutation rate for sensory/inter/action locations.",
            ),
            (
                "mutation_rate_synapse_weight_perturbation",
                "required".to_owned(),
                "Mutation rate for synapse weight perturbation/replacement.",
            ),
            (
                "mutation_rate_add_synapse",
                "required".to_owned(),
                "Mutation rate for adding synapses.",
            ),
            (
                "mutation_rate_remove_synapse",
                "required".to_owned(),
                "Mutation rate for removing synapses.",
            ),
            (
                "mutation_rate_remove_neuron",
                "required".to_owned(),
                "Mutation rate for removing inter neurons.",
            ),
            (
                "mutation_rate_add_neuron_split_edge",
                "required".to_owned(),
                "Mutation rate for splitting an edge with a new neuron.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "Hidden Policies",
        "Terrain Generation",
        &vec![
            (
                "terrain_seed_mix",
                format!("0x{:X}", terrain_policy.terrain_seed_mix),
                "Seed xor used to derive terrain noise from the run seed.",
            ),
            (
                "default_threshold",
                terrain_policy.default_threshold.to_string(),
                "Canonical terrain-wall threshold used by `build_terrain_map`.",
            ),
            (
                "spike_seed_mix",
                format!("0x{:X}", terrain_policy.spike_seed_mix),
                "Seed xor used to derive spike scatter hashes from the run seed.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "Hidden Policies",
        "Food Ecology",
        &vec![
            (
                "fertility_seed_mix",
                format!("0x{:X}", food_policy.fertility_seed_mix),
                "Seed xor used to derive plant fertility noise.",
            ),
            (
                "fertility_jitter_seed_mix",
                format!("0x{:X}", food_policy.fertility_jitter_seed_mix),
                "Secondary seed xor used for fertility jitter.",
            ),
            (
                "fertility_noise_scale",
                food_policy.fertility_noise_scale.to_string(),
                "Perlin input scale for fertility sampling.",
            ),
            (
                "fertility_threshold",
                food_policy.fertility_threshold.to_string(),
                "Binary fertility cutoff after jitter.",
            ),
        ],
    );

    out
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub spatial_prior_sigma: f32,
    pub vision_distance: u32,
    #[serde(default = "default_starting_energy")]
    pub starting_energy: f32,
    pub age_of_maturity: u32,
    #[serde(default = "default_plasticity_start_age")]
    pub plasticity_start_age: u32,
    pub hebb_eta_gain: f32,
    #[serde(default = "default_juvenile_eta_scale")]
    pub juvenile_eta_scale: f32,
    pub eligibility_retention: f32,
    #[serde(default = "default_max_weight_delta_per_tick")]
    pub max_weight_delta_per_tick: f32,
    pub synapse_prune_threshold: f32,
    pub mutation_rate_age_of_maturity: f32,
    pub mutation_rate_vision_distance: f32,
    pub mutation_rate_inter_bias: f32,
    pub mutation_rate_inter_update_rate: f32,
    pub mutation_rate_eligibility_retention: f32,
    pub mutation_rate_synapse_prune_threshold: f32,
    pub mutation_rate_neuron_location: f32,
    #[serde(default)]
    pub mutation_rate_synapse_weight_perturbation: f32,
    #[serde(default)]
    pub mutation_rate_add_synapse: f32,
    #[serde(default)]
    pub mutation_rate_remove_synapse: f32,
    #[serde(default)]
    pub mutation_rate_remove_neuron: f32,
    #[serde(default)]
    pub mutation_rate_add_neuron_split_edge: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    pub periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    pub periodic_injection_count: u32,
    pub food_energy: f32,
    pub move_action_energy_cost: f32,
    #[serde(default = "default_reproduction_investment_energy")]
    pub reproduction_investment_energy: f32,
    #[serde(default = "default_action_temperature")]
    pub action_temperature: f32,
    #[serde(default = "default_intent_parallel_threads")]
    pub intent_parallel_threads: u32,
    #[serde(default = "default_food_regrowth_interval")]
    pub food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    pub food_regrowth_jitter: u32,
    #[serde(default = "default_terrain_noise_scale")]
    pub terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    pub terrain_threshold: f32,
    #[serde(default = "default_spike_density")]
    pub spike_density: f32,
    pub max_organism_age: u32,
    #[serde(default = "default_global_mutation_rate_modifier")]
    pub global_mutation_rate_modifier: f32,
    #[serde(default = "default_meta_mutation_enabled")]
    pub meta_mutation_enabled: bool,
    #[serde(default = "default_runtime_plasticity_enabled")]
    pub runtime_plasticity_enabled: bool,
    #[serde(default = "default_force_random_actions")]
    pub force_random_actions: bool,
    pub seed_genome_config: SeedGenomeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldConfigDeserialize {
    world_width: u32,
    num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    periodic_injection_count: u32,
    food_energy: f32,
    move_action_energy_cost: f32,
    #[serde(default = "default_reproduction_investment_energy")]
    reproduction_investment_energy: f32,
    #[serde(default = "default_action_temperature")]
    action_temperature: f32,
    #[serde(default = "default_intent_parallel_threads")]
    intent_parallel_threads: u32,
    #[serde(default = "default_food_regrowth_interval")]
    food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    food_regrowth_jitter: u32,
    #[serde(default = "default_terrain_noise_scale")]
    terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    terrain_threshold: f32,
    #[serde(default = "default_spike_density")]
    spike_density: f32,
    max_organism_age: u32,
    #[serde(default = "default_global_mutation_rate_modifier")]
    global_mutation_rate_modifier: f32,
    #[serde(default = "default_meta_mutation_enabled")]
    meta_mutation_enabled: bool,
    #[serde(default = "default_runtime_plasticity_enabled")]
    runtime_plasticity_enabled: bool,
    #[serde(default = "default_force_random_actions")]
    force_random_actions: bool,
    seed_genome_config: SeedGenomeConfig,
}

impl<'de> Deserialize<'de> for WorldConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = WorldConfigDeserialize::deserialize(deserializer)?;
        Ok(Self {
            world_width: raw.world_width,
            num_organisms: raw.num_organisms,
            periodic_injection_interval_turns: raw.periodic_injection_interval_turns,
            periodic_injection_count: raw.periodic_injection_count,
            food_energy: raw.food_energy,
            move_action_energy_cost: raw.move_action_energy_cost,
            reproduction_investment_energy: raw.reproduction_investment_energy,
            action_temperature: raw.action_temperature,
            intent_parallel_threads: raw.intent_parallel_threads,
            food_regrowth_interval: raw.food_regrowth_interval,
            food_regrowth_jitter: raw.food_regrowth_jitter,
            terrain_noise_scale: raw.terrain_noise_scale,
            terrain_threshold: raw.terrain_threshold,
            spike_density: raw.spike_density,
            max_organism_age: raw.max_organism_age,
            global_mutation_rate_modifier: raw.global_mutation_rate_modifier,
            meta_mutation_enabled: raw.meta_mutation_enabled,
            runtime_plasticity_enabled: raw.runtime_plasticity_enabled,
            force_random_actions: raw.force_random_actions,
            seed_genome_config: raw.seed_genome_config,
        })
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        default_world_config()
    }
}

pub fn world_config_from_toml_str(raw: &str) -> Result<WorldConfig, toml::de::Error> {
    let mut value: toml::Value = toml::from_str(raw)?;
    normalize_world_config_toml(&mut value);
    value.try_into()
}

pub fn default_world_config() -> WorldConfig {
    world_config_from_toml_str(include_str!("../config.toml"))
        .expect("default world config TOML must deserialize")
}

pub fn default_world_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_WORLD_CONFIG_REL_PATH)
}

pub fn load_default_world_config() -> Result<WorldConfig> {
    load_world_config_from_path(&default_world_config_path())
}

pub fn load_world_config_from_path(path: &Path) -> Result<WorldConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read world config from {}", path.display()))?;
    world_config_from_toml_str(&raw)
        .context("world config TOML failed schema deserialization")
        .with_context(|| format!("failed to parse world config from {}", path.display()))
}

pub fn validate_world_config(config: &WorldConfig) -> Result<(), String> {
    if config.world_width == 0 {
        return Err("world_width must be greater than zero".to_owned());
    }
    if config.num_organisms == 0 {
        return Err("num_organisms must be greater than zero".to_owned());
    }
    if config.food_energy <= 0.0 {
        return Err("food_energy must be greater than zero".to_owned());
    }
    if config.move_action_energy_cost < 0.0 {
        return Err("move_action_energy_cost must be >= 0".to_owned());
    }
    if config.reproduction_investment_energy < 0.0 {
        return Err("reproduction_investment_energy must be >= 0".to_owned());
    }
    if !config.action_temperature.is_finite() || config.action_temperature <= 0.0 {
        return Err("action_temperature must be finite and greater than zero".to_owned());
    }
    if config.intent_parallel_threads == 0 {
        return Err("intent_parallel_threads must be greater than zero".to_owned());
    }
    if config.food_regrowth_interval == 0 {
        return Err("food_regrowth_interval must be greater than zero".to_owned());
    }
    if config.terrain_noise_scale <= 0.0 {
        return Err("terrain_noise_scale must be greater than zero".to_owned());
    }
    if !(0.0..=1.0).contains(&config.terrain_threshold) {
        return Err("terrain_threshold must be in [0.0, 1.0]".to_owned());
    }
    if !(0.0..=1.0).contains(&config.spike_density) {
        return Err("spike_density must be in [0.0, 1.0]".to_owned());
    }
    if !config.global_mutation_rate_modifier.is_finite()
        || config.global_mutation_rate_modifier < 0.0
    {
        return Err("global_mutation_rate_modifier must be finite and >= 0".to_owned());
    }
    if !config.seed_genome_config.juvenile_eta_scale.is_finite()
        || config.seed_genome_config.juvenile_eta_scale < 0.0
    {
        return Err("seed_genome_config.juvenile_eta_scale must be finite and >= 0".to_owned());
    }
    if !config
        .seed_genome_config
        .max_weight_delta_per_tick
        .is_finite()
        || config.seed_genome_config.max_weight_delta_per_tick < 0.0
    {
        return Err(
            "seed_genome_config.max_weight_delta_per_tick must be finite and >= 0".to_owned(),
        );
    }
    Ok(())
}

fn normalize_world_config_toml(value: &mut toml::Value) {
    let Some(table) = value.as_table_mut() else {
        return;
    };

    let world_width = table
        .get("world_width")
        .and_then(toml::Value::as_integer)
        .and_then(|v| u32::try_from(v).ok());
    let legacy_starting_energy = table
        .get("starting_energy")
        .and_then(|value| match value {
            toml::Value::Float(v) => Some(*v),
            toml::Value::Integer(v) => Some(*v as f64),
            _ => None,
        })
        .or_else(|| world_width.map(|w| w as f64));

    if let Some(seed_genome_table) = table
        .entry("seed_genome_config")
        .or_insert_with(|| toml::Value::Table(Default::default()))
        .as_table_mut()
    {
        if let Some(legacy_starting_energy) = legacy_starting_energy {
            seed_genome_table
                .entry("starting_energy")
                .or_insert_with(|| toml::Value::Float(legacy_starting_energy));
        }
    }

    if let Some(world_width) = world_width {
        let derived_max_age = derived_max_organism_age(world_width);
        table
            .entry("max_organism_age")
            .or_insert_with(|| toml::Value::Integer(i64::from(derived_max_age)));
        table
            .entry("periodic_injection_interval_turns")
            .or_insert_with(|| toml::Value::Integer(i64::from(derived_max_age)));
    }
}

fn default_starting_energy() -> f32 {
    seed_genome_config_defaults().starting_energy
}

fn default_reproduction_investment_energy() -> f32 {
    world_config_defaults()
        .lifecycle
        .reproduction_investment_energy
}

fn default_juvenile_eta_scale() -> f32 {
    seed_genome_config_defaults().juvenile_eta_scale
}

fn default_max_weight_delta_per_tick() -> f32 {
    seed_genome_config_defaults().max_weight_delta_per_tick
}

fn default_food_regrowth_interval() -> u32 {
    world_config_defaults().food_regrowth.interval
}

fn default_food_regrowth_jitter() -> u32 {
    world_config_defaults().food_regrowth.jitter
}

fn derived_max_organism_age(world_width: u32) -> u32 {
    world_width.saturating_mul(derived_world_policy().max_organism_age_per_world_width)
}

fn default_periodic_injection_interval_turns() -> u32 {
    world_config_defaults()
        .population
        .periodic_injection_interval_turns
}

fn default_periodic_injection_count() -> u32 {
    world_config_defaults().population.periodic_injection_count
}

fn default_terrain_noise_scale() -> f32 {
    world_config_defaults().terrain.terrain_noise_scale
}

fn default_terrain_threshold() -> f32 {
    world_config_defaults().terrain.terrain_threshold
}

fn default_spike_density() -> f32 {
    world_config_defaults().terrain.spike_density
}

fn default_action_temperature() -> f32 {
    world_config_defaults().lifecycle.action_temperature
}

fn default_intent_parallel_threads() -> u32 {
    world_config_defaults().lifecycle.intent_parallel_threads
}

fn default_global_mutation_rate_modifier() -> f32 {
    world_config_defaults()
        .evolution
        .global_mutation_rate_modifier
}

fn default_meta_mutation_enabled() -> bool {
    world_config_defaults().evolution.meta_mutation_enabled
}

fn default_runtime_plasticity_enabled() -> bool {
    world_config_defaults().evolution.runtime_plasticity_enabled
}

fn default_force_random_actions() -> bool {
    world_config_defaults().evolution.force_random_actions
}

fn default_plasticity_start_age() -> u32 {
    seed_genome_config_defaults().plasticity_start_age
}

fn bool_label(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn push_doc_section(out: &mut String, major: &str, minor: &str, rows: &[(&str, String, &str)]) {
    let _ = write!(out, "## {major}: {minor}\n\n");
    out.push_str("| Key | Default / Derivation | Notes |\n");
    out.push_str("| --- | --- | --- |\n");
    for (key, default, notes) in rows {
        let _ = writeln!(out, "| `{key}` | `{default}` | {notes} |");
    }
    out.push('\n');
}
