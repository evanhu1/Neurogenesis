use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const DEFAULT_WORLD_CONFIG_REL_PATH: &str = "default.toml";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeedGenomeConfig {
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub spatial_prior_sigma: f32,
    pub vision_distance: u32,
    #[serde(default = "default_starting_energy")]
    pub starting_energy: f32,
    pub age_of_maturity: u32,
    pub hebb_eta_gain: f32,
    pub eligibility_retention: f32,
    pub synapse_prune_threshold: f32,
    pub mutation_rate_age_of_maturity: f32,
    pub mutation_rate_vision_distance: f32,
    pub mutation_rate_inter_bias: f32,
    pub mutation_rate_inter_update_rate: f32,
    pub mutation_rate_action_bias: f32,
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
    pub mutation_rate_add_neuron_split_edge: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub steps_per_second: u32,
    pub num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    pub periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    pub periodic_injection_count: u32,
    pub food_energy: f32,
    pub move_action_energy_cost: f32,
    #[serde(default = "default_action_temperature")]
    pub action_temperature: f32,
    #[serde(default = "default_food_regrowth_interval")]
    pub food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    pub food_regrowth_jitter: u32,
    #[serde(default = "default_terrain_noise_scale")]
    pub terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    pub terrain_threshold: f32,
    pub max_organism_age: u32,
    #[serde(default = "default_global_mutation_rate_modifier")]
    pub global_mutation_rate_modifier: f32,
    #[serde(default = "default_runtime_plasticity_enabled")]
    pub runtime_plasticity_enabled: bool,
    pub seed_genome_config: SeedGenomeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldConfigDeserialize {
    world_width: u32,
    steps_per_second: u32,
    num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    periodic_injection_count: u32,
    food_energy: f32,
    move_action_energy_cost: f32,
    #[serde(default = "default_action_temperature")]
    action_temperature: f32,
    #[serde(default = "default_food_regrowth_interval")]
    food_regrowth_interval: u32,
    #[serde(default = "default_food_regrowth_jitter")]
    food_regrowth_jitter: u32,
    #[serde(default = "default_terrain_noise_scale")]
    terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    terrain_threshold: f32,
    max_organism_age: u32,
    #[serde(default = "default_global_mutation_rate_modifier")]
    global_mutation_rate_modifier: f32,
    #[serde(default = "default_runtime_plasticity_enabled")]
    runtime_plasticity_enabled: bool,
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
            steps_per_second: raw.steps_per_second,
            num_organisms: raw.num_organisms,
            periodic_injection_interval_turns: raw.periodic_injection_interval_turns,
            periodic_injection_count: raw.periodic_injection_count,
            food_energy: raw.food_energy,
            move_action_energy_cost: raw.move_action_energy_cost,
            action_temperature: raw.action_temperature,
            food_regrowth_interval: raw.food_regrowth_interval,
            food_regrowth_jitter: raw.food_regrowth_jitter,
            terrain_noise_scale: raw.terrain_noise_scale,
            terrain_threshold: raw.terrain_threshold,
            max_organism_age: raw.max_organism_age,
            global_mutation_rate_modifier: raw.global_mutation_rate_modifier,
            runtime_plasticity_enabled: raw.runtime_plasticity_enabled,
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
    world_config_from_toml_str(include_str!("../default.toml"))
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
    if !config.action_temperature.is_finite() || config.action_temperature <= 0.0 {
        return Err("action_temperature must be finite and greater than zero".to_owned());
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
    if !config.global_mutation_rate_modifier.is_finite()
        || config.global_mutation_rate_modifier < 0.0
    {
        return Err("global_mutation_rate_modifier must be finite and >= 0".to_owned());
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
        table
            .entry("max_organism_age")
            .or_insert_with(|| toml::Value::Integer(i64::from(world_width.saturating_mul(10))));
    }
}

fn default_starting_energy() -> f32 {
    1.0
}

fn default_food_regrowth_interval() -> u32 {
    10
}

fn default_food_regrowth_jitter() -> u32 {
    2
}

fn default_periodic_injection_interval_turns() -> u32 {
    100
}

fn default_periodic_injection_count() -> u32 {
    100
}

fn default_terrain_noise_scale() -> f32 {
    0.02
}

fn default_terrain_threshold() -> f32 {
    0.86
}

fn default_action_temperature() -> f32 {
    0.5
}

fn default_global_mutation_rate_modifier() -> f32 {
    1.0
}

fn default_runtime_plasticity_enabled() -> bool {
    true
}
