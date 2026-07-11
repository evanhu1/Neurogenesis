use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const DEFAULT_WORLD_CONFIG_REL_PATH: &str = "config.toml";
const DEFAULT_SEED_GENOME_CONFIG_REL_PATH: &str = "seed_genome.toml";

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct LifecycleDefaults {
    pub passive_metabolism_cost_per_unit: f32,
    pub body_mass_metabolic_cost_coeff: f32,
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
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct FlagsDefaults {
    pub runtime_plasticity_enabled: bool,
    pub leaky_neurons_enabled: bool,
    pub predation_enabled: bool,
    pub force_random_actions: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct WorldConfigDefaults {
    pub lifecycle: LifecycleDefaults,
    pub food_regrowth: FoodRegrowthDefaults,
    pub terrain: TerrainDefaults,
    pub flags: FlagsDefaults,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct SeedGenomeConfigDefaults {
    pub max_organism_age: u32,
    pub gestation_ticks: u8,
    pub juvenile_eta_scale: f32,
    pub max_weight_delta_per_tick: f32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct TerrainGenerationPolicy {
    pub terrain_seed_mix: u64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct FoodEcologyPolicy {
    pub food_tile_seed_mix: u64,
}

pub fn world_config_defaults() -> WorldConfigDefaults {
    WorldConfigDefaults {
        lifecycle: LifecycleDefaults {
            passive_metabolism_cost_per_unit: 0.005,
            body_mass_metabolic_cost_coeff: 1.0,
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
        },
        flags: FlagsDefaults {
            runtime_plasticity_enabled: false,
            leaky_neurons_enabled: false,
            predation_enabled: false,
            force_random_actions: false,
        },
    }
}

pub fn seed_genome_config_defaults() -> SeedGenomeConfigDefaults {
    SeedGenomeConfigDefaults {
        // Finite lifespan cap matching the mutation clamp in sim-core
        // (`MAX_MUTATED_MAX_ORGANISM_AGE` = 100_000); seeds are no longer
        // immortal.
        max_organism_age: 100_000,
        gestation_ticks: 2,
        juvenile_eta_scale: 2.0,
        max_weight_delta_per_tick: 0.05,
    }
}

pub fn terrain_generation_policy() -> TerrainGenerationPolicy {
    TerrainGenerationPolicy {
        terrain_seed_mix: 0xA5A5_A5A5_u64,
    }
}

pub fn food_ecology_policy() -> FoodEcologyPolicy {
    FoodEcologyPolicy {
        food_tile_seed_mix: 0x6A09_E667_F3BC_C909,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub vision_distance: u32,
    pub age_of_maturity: u32,
    #[serde(default = "default_gestation_ticks")]
    pub gestation_ticks: u8,
    #[serde(default = "default_max_organism_age")]
    pub max_organism_age: u32,
    pub hebb_eta_gain: f32,
    #[serde(default = "default_juvenile_eta_scale")]
    pub juvenile_eta_scale: f32,
    pub eligibility_retention: f32,
    #[serde(default = "default_max_weight_delta_per_tick")]
    pub max_weight_delta_per_tick: f32,
    pub synapse_prune_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub num_organisms: u32,
    pub food_energy: f32,
    #[serde(default = "default_passive_metabolism_cost_per_unit")]
    pub passive_metabolism_cost_per_unit: f32,
    #[serde(default = "default_body_mass_metabolic_cost_coeff")]
    pub body_mass_metabolic_cost_coeff: f32,
    pub move_action_energy_cost: f32,
    #[serde(default = "default_action_temperature")]
    pub action_temperature: f32,
    #[serde(default = "default_intent_parallel_threads")]
    pub intent_parallel_threads: u32,
    #[serde(default = "default_food_regrowth_interval")]
    pub food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    pub food_regrowth_jitter: u32,
    #[serde(default = "default_food_tile_fraction")]
    pub food_tile_fraction: f32,
    #[serde(default = "default_terrain_noise_scale")]
    pub terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    pub terrain_threshold: f32,
    #[serde(default = "default_runtime_plasticity_enabled")]
    pub runtime_plasticity_enabled: bool,
    #[serde(default = "default_leaky_neurons_enabled")]
    pub leaky_neurons_enabled: bool,
    #[serde(default = "default_predation_enabled")]
    pub predation_enabled: bool,
    #[serde(default = "default_force_random_actions")]
    pub force_random_actions: bool,
    pub seed_genome_config: SeedGenomeConfig,
}

impl WorldConfig {
    /// Canonical performance fixture used by benchmarks, examples, and profiling.
    /// Callers may override `intent_parallel_threads` after construction.
    pub fn perf_fixture() -> Self {
        Self {
            world_width: 100,
            num_organisms: 2_000,
            food_energy: 10.0,
            passive_metabolism_cost_per_unit: 0.005,
            body_mass_metabolic_cost_coeff: 1.0,
            move_action_energy_cost: 0.0,
            action_temperature: 0.5,
            intent_parallel_threads: 8,
            food_regrowth_interval: 10,
            food_regrowth_jitter: 2,
            food_tile_fraction: 0.4,
            terrain_noise_scale: 0.02,
            terrain_threshold: 1.0,
            runtime_plasticity_enabled: true,
            leaky_neurons_enabled: false,
            predation_enabled: false,
            force_random_actions: false,
            seed_genome_config: SeedGenomeConfig {
                num_neurons: 20,
                num_synapses: 80,
                vision_distance: 10,
                age_of_maturity: 0,
                gestation_ticks: 2,
                max_organism_age: 100_000,
                hebb_eta_gain: 0.0,
                juvenile_eta_scale: 2.0,
                eligibility_retention: 0.9,
                max_weight_delta_per_tick: 0.05,
                synapse_prune_threshold: 0.01,
            },
        }
    }

    /// Canonical test fixture used by unit tests — small world, all mutation rates zeroed.
    pub fn test_fixture() -> Self {
        Self {
            world_width: 10,
            num_organisms: 10,
            food_energy: 10.0,
            passive_metabolism_cost_per_unit: 0.005,
            body_mass_metabolic_cost_coeff: 1.0,
            move_action_energy_cost: 1.0,
            action_temperature: 0.5,
            intent_parallel_threads: 8,
            food_regrowth_interval: 10,
            food_regrowth_jitter: 2,
            food_tile_fraction: 0.4,
            terrain_noise_scale: 0.02,
            terrain_threshold: 1.0,
            runtime_plasticity_enabled: true,
            leaky_neurons_enabled: false,
            predation_enabled: false,
            force_random_actions: false,
            seed_genome_config: SeedGenomeConfig {
                num_neurons: 1,
                num_synapses: 0,
                vision_distance: 2,
                age_of_maturity: 0,
                gestation_ticks: 2,
                max_organism_age: 500,
                hebb_eta_gain: 0.0,
                juvenile_eta_scale: 0.5,
                eligibility_retention: 0.9,
                max_weight_delta_per_tick: 0.05,
                synapse_prune_threshold: 0.01,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct WorldConfigToml {
    world: WorldGeometryToml,
    population: WorldPopulationToml,
    lifecycle: WorldLifecycleToml,
    food: WorldFoodToml,
    terrain: WorldTerrainToml,
    flags: WorldFlagsToml,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldGeometryToml {
    world_width: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldPopulationToml {
    num_organisms: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldLifecycleToml {
    #[serde(default = "default_passive_metabolism_cost_per_unit")]
    passive_metabolism_cost_per_unit: f32,
    #[serde(default = "default_body_mass_metabolic_cost_coeff")]
    body_mass_metabolic_cost_coeff: f32,
    move_action_energy_cost: f32,
    #[serde(default = "default_action_temperature")]
    action_temperature: f32,
    #[serde(default = "default_intent_parallel_threads")]
    intent_parallel_threads: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldFoodToml {
    food_energy: f32,
    #[serde(default = "default_food_regrowth_interval")]
    food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    food_regrowth_jitter: u32,
    #[serde(default = "default_food_tile_fraction")]
    food_tile_fraction: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldTerrainToml {
    #[serde(default = "default_terrain_noise_scale")]
    terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    terrain_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldFlagsToml {
    #[serde(default = "default_runtime_plasticity_enabled")]
    runtime_plasticity_enabled: bool,
    #[serde(default = "default_leaky_neurons_enabled")]
    leaky_neurons_enabled: bool,
    #[serde(default = "default_predation_enabled")]
    predation_enabled: bool,
    #[serde(default = "default_force_random_actions")]
    force_random_actions: bool,
}

impl WorldConfigToml {
    fn into_runtime(self, seed_genome_config: SeedGenomeConfig) -> WorldConfig {
        WorldConfig {
            world_width: self.world.world_width,
            num_organisms: self.population.num_organisms,
            food_energy: self.food.food_energy,
            passive_metabolism_cost_per_unit: self.lifecycle.passive_metabolism_cost_per_unit,
            body_mass_metabolic_cost_coeff: self.lifecycle.body_mass_metabolic_cost_coeff,
            move_action_energy_cost: self.lifecycle.move_action_energy_cost,
            action_temperature: self.lifecycle.action_temperature,
            intent_parallel_threads: self.lifecycle.intent_parallel_threads,
            food_regrowth_interval: self.food.food_regrowth_interval,
            food_regrowth_jitter: self.food.food_regrowth_jitter,
            food_tile_fraction: self.food.food_tile_fraction,
            terrain_noise_scale: self.terrain.terrain_noise_scale,
            terrain_threshold: self.terrain.terrain_threshold,
            runtime_plasticity_enabled: self.flags.runtime_plasticity_enabled,
            leaky_neurons_enabled: self.flags.leaky_neurons_enabled,
            predation_enabled: self.flags.predation_enabled,
            force_random_actions: self.flags.force_random_actions,
            seed_genome_config,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        default_world_config()
    }
}

pub fn world_config_from_toml_parts(
    world_raw: &str,
    seed_genome_raw: &str,
) -> Result<WorldConfig, toml::de::Error> {
    let world_config: WorldConfigToml = toml::from_str(world_raw)?;
    let seed_genome_config: SeedGenomeConfig = toml::from_str(seed_genome_raw)?;
    Ok(world_config.into_runtime(seed_genome_config))
}

pub fn default_world_config() -> WorldConfig {
    world_config_from_toml_parts(
        include_str!("../config.toml"),
        include_str!("../seed_genome.toml"),
    )
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
    let seed_genome_path = path.with_file_name(DEFAULT_SEED_GENOME_CONFIG_REL_PATH);
    let seed_genome_raw = std::fs::read_to_string(&seed_genome_path).with_context(|| {
        format!(
            "failed to read seed genome config from {}",
            seed_genome_path.display()
        )
    })?;
    world_config_from_toml_parts(&raw, &seed_genome_raw)
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
    if !config.passive_metabolism_cost_per_unit.is_finite()
        || config.passive_metabolism_cost_per_unit < 0.0
    {
        return Err("passive_metabolism_cost_per_unit must be finite and >= 0".to_owned());
    }
    if !config.body_mass_metabolic_cost_coeff.is_finite()
        || config.body_mass_metabolic_cost_coeff < 0.0
    {
        return Err("body_mass_metabolic_cost_coeff must be finite and >= 0".to_owned());
    }
    if config.move_action_energy_cost < 0.0 {
        return Err("move_action_energy_cost must be >= 0".to_owned());
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
    if !(0.0..=1.0).contains(&config.food_tile_fraction) {
        return Err("food_tile_fraction must be in [0.0, 1.0]".to_owned());
    }
    if config.terrain_noise_scale <= 0.0 {
        return Err("terrain_noise_scale must be greater than zero".to_owned());
    }
    if !(0.0..=1.0).contains(&config.terrain_threshold) {
        return Err("terrain_threshold must be in [0.0, 1.0]".to_owned());
    }
    if !config.seed_genome_config.juvenile_eta_scale.is_finite()
        || config.seed_genome_config.juvenile_eta_scale < 0.0
    {
        return Err("seed_genome_config.juvenile_eta_scale must be finite and >= 0".to_owned());
    }
    if config.seed_genome_config.gestation_ticks > 4 {
        return Err("seed_genome_config.gestation_ticks must be in [0, 4]".to_owned());
    }
    if !(1..=10).contains(&config.seed_genome_config.vision_distance) {
        return Err("seed_genome_config.vision_distance must be in [1, 10]".to_owned());
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

fn default_gestation_ticks() -> u8 {
    seed_genome_config_defaults().gestation_ticks
}

fn default_juvenile_eta_scale() -> f32 {
    seed_genome_config_defaults().juvenile_eta_scale
}

fn default_max_weight_delta_per_tick() -> f32 {
    seed_genome_config_defaults().max_weight_delta_per_tick
}

fn default_max_organism_age() -> u32 {
    seed_genome_config_defaults().max_organism_age
}

fn default_food_regrowth_interval() -> u32 {
    world_config_defaults().food_regrowth.interval
}

fn default_food_regrowth_jitter() -> u32 {
    world_config_defaults().food_regrowth.jitter
}

fn default_food_tile_fraction() -> f32 {
    0.4
}

fn default_passive_metabolism_cost_per_unit() -> f32 {
    world_config_defaults()
        .lifecycle
        .passive_metabolism_cost_per_unit
}

fn default_body_mass_metabolic_cost_coeff() -> f32 {
    world_config_defaults()
        .lifecycle
        .body_mass_metabolic_cost_coeff
}

fn default_terrain_noise_scale() -> f32 {
    world_config_defaults().terrain.terrain_noise_scale
}

fn default_terrain_threshold() -> f32 {
    world_config_defaults().terrain.terrain_threshold
}

fn default_action_temperature() -> f32 {
    world_config_defaults().lifecycle.action_temperature
}

fn default_intent_parallel_threads() -> u32 {
    world_config_defaults().lifecycle.intent_parallel_threads
}

fn default_runtime_plasticity_enabled() -> bool {
    world_config_defaults().flags.runtime_plasticity_enabled
}

fn default_leaky_neurons_enabled() -> bool {
    world_config_defaults().flags.leaky_neurons_enabled
}

fn default_predation_enabled() -> bool {
    world_config_defaults().flags.predation_enabled
}

fn default_force_random_actions() -> bool {
    world_config_defaults().flags.force_random_actions
}
