use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
pub use types::{SeedGenomeConfig, WorldConfig};

/// Repository-relative path to the one canonical world/evolution environment.
pub const CANONICAL_WORLD_CONFIG_PATH: &str = "config/world.toml";
const DEFAULT_SEED_GENOME_CONFIG_REL_PATH: &str = "seed_genome.toml";

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct SeedGenomeConfigDefaults {
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

pub fn seed_genome_config_defaults() -> SeedGenomeConfigDefaults {
    SeedGenomeConfigDefaults {
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldConfigToml {
    world: WorldGeometryToml,
    population: WorldPopulationToml,
    lifecycle: WorldLifecycleToml,
    food: WorldFoodToml,
    terrain: WorldTerrainToml,
    flags: WorldFlagsToml,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldGeometryToml {
    world_width: u32,
    vision_range: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldPopulationToml {
    num_organisms: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldLifecycleToml {
    starting_energy: u32,
    attack_energy_transfer: u32,
    attack_attempt_cost: u32,
    action_temperature: f32,
    intent_parallel_threads: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldFoodToml {
    food_energy: u32,
    food_regrowth_interval: u32,
    food_tile_fraction: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldTerrainToml {
    terrain_noise_scale: f32,
    terrain_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldFlagsToml {
    runtime_plasticity_enabled: bool,
    leaky_neurons_enabled: bool,
    predation_enabled: bool,
    force_random_actions: bool,
}

impl WorldConfigToml {
    fn into_runtime(self, seed_genome_config: SeedGenomeConfig) -> WorldConfig {
        WorldConfig {
            world_width: self.world.world_width,
            vision_range: self.world.vision_range,
            num_organisms: self.population.num_organisms,
            starting_energy: self.lifecycle.starting_energy,
            attack_energy_transfer: self.lifecycle.attack_energy_transfer,
            attack_attempt_cost: self.lifecycle.attack_attempt_cost,
            food_energy: self.food.food_energy,
            action_temperature: self.lifecycle.action_temperature,
            intent_parallel_threads: self.lifecycle.intent_parallel_threads,
            food_regrowth_interval: self.food.food_regrowth_interval,
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
        include_str!("../world.toml"),
        include_str!("../seed_genome.toml"),
    )
    .expect("default world config TOML must deserialize")
}

pub fn default_world_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("config must live in the workspace root")
        .join(CANONICAL_WORLD_CONFIG_PATH)
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
    if config.vision_range == 0 {
        return Err("vision_range must be greater than zero".to_owned());
    }
    if config.num_organisms == 0 {
        return Err("num_organisms must be greater than zero".to_owned());
    }
    if config.starting_energy == 0 {
        return Err("starting_energy must be greater than zero".to_owned());
    }
    if config.attack_energy_transfer == 0 {
        return Err("attack_energy_transfer must be greater than zero".to_owned());
    }
    if config.attack_attempt_cost == 0 {
        return Err("attack_attempt_cost must be greater than zero".to_owned());
    }
    if config.attack_energy_transfer <= config.attack_attempt_cost {
        return Err("attack_energy_transfer must be greater than attack_attempt_cost".to_owned());
    }
    if config.food_energy == 0 {
        return Err("food_energy must be greater than zero".to_owned());
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
