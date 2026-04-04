use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const DEFAULT_WORLD_CONFIG_REL_PATH: &str = "config.toml";
const DEFAULT_SEED_GENOME_CONFIG_REL_PATH: &str = "seed_genome.toml";

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct PopulationDefaults {
    pub periodic_injection_interval_turns: u32,
    pub periodic_injection_count: u32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct LifecycleDefaults {
    pub passive_metabolism_cost_per_unit: f32,
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
    pub max_health: f32,
    pub max_organism_age: u32,
    pub plasticity_start_age: u32,
    pub juvenile_eta_scale: f32,
    pub max_weight_delta_per_tick: f32,
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
            passive_metabolism_cost_per_unit: 0.005,
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
        max_health: 1.0,
        max_organism_age: u32::MAX,
        plasticity_start_age: 0,
        juvenile_eta_scale: 0.5,
        max_weight_delta_per_tick: 0.05,
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
                "Cadence for seed-genome injections.",
            ),
            (
                "periodic_injection_count",
                world.population.periodic_injection_count.to_string(),
                "How many injection attempts run at each periodic-injection turn.",
            ),
        ],
    );
    push_doc_section(
        &mut out,
        "World Config",
        "Lifecycle And Actions",
        &vec![
            (
                "passive_metabolism_cost_per_unit",
                world.lifecycle.passive_metabolism_cost_per_unit.to_string(),
                "Per-complexity-unit passive energy drain applied each tick.",
            ),
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
                "max_health",
                genome.max_health.to_string(),
                "Default maximum health when omitted in seed-genome TOML.",
            ),
            (
                "age_of_maturity",
                "required".to_owned(),
                "Age threshold for adulthood.",
            ),
            (
                "max_organism_age",
                genome.max_organism_age.to_string(),
                "Default lifespan cap when omitted in seed-genome TOML.",
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
                "mutation_rate_max_organism_age",
                "required".to_owned(),
                "Mutation rate for organism lifespan.",
            ),
            (
                "mutation_rate_vision_distance",
                "required".to_owned(),
                "Mutation rate for vision distance.",
            ),
            (
                "mutation_rate_max_health",
                "required".to_owned(),
                "Mutation rate for maximum health.",
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
    #[serde(default = "default_max_health")]
    pub max_health: f32,
    #[serde(default = "default_max_organism_age")]
    pub max_organism_age: u32,
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
    pub mutation_rate_max_organism_age: f32,
    pub mutation_rate_vision_distance: f32,
    pub mutation_rate_max_health: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldConfig {
    pub world_width: u32,
    pub num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    pub periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    pub periodic_injection_count: u32,
    pub food_energy: f32,
    #[serde(default = "default_passive_metabolism_cost_per_unit")]
    pub passive_metabolism_cost_per_unit: f32,
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
struct SeedGenomeConfigToml {
    topology: SeedGenomeTopologyToml,
    lifecycle: SeedGenomeLifecycleToml,
    plasticity: SeedGenomePlasticityToml,
    mutation_rates: SeedGenomeMutationRatesToml,
}

#[derive(Debug, Clone, Deserialize)]
struct SeedGenomeTopologyToml {
    num_neurons: u32,
    num_synapses: u32,
    spatial_prior_sigma: f32,
    vision_distance: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct SeedGenomeLifecycleToml {
    #[serde(default = "default_starting_energy")]
    starting_energy: f32,
    #[serde(default = "default_max_health")]
    max_health: f32,
    age_of_maturity: u32,
    #[serde(default = "default_max_organism_age")]
    max_organism_age: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct SeedGenomePlasticityToml {
    #[serde(default = "default_plasticity_start_age")]
    plasticity_start_age: u32,
    hebb_eta_gain: f32,
    #[serde(default = "default_juvenile_eta_scale")]
    juvenile_eta_scale: f32,
    eligibility_retention: f32,
    #[serde(default = "default_max_weight_delta_per_tick")]
    max_weight_delta_per_tick: f32,
    synapse_prune_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct SeedGenomeMutationRatesToml {
    mutation_rate_age_of_maturity: f32,
    mutation_rate_max_organism_age: f32,
    mutation_rate_vision_distance: f32,
    mutation_rate_max_health: f32,
    mutation_rate_inter_bias: f32,
    mutation_rate_inter_update_rate: f32,
    mutation_rate_eligibility_retention: f32,
    mutation_rate_synapse_prune_threshold: f32,
    mutation_rate_neuron_location: f32,
    #[serde(default)]
    mutation_rate_synapse_weight_perturbation: f32,
    #[serde(default)]
    mutation_rate_add_synapse: f32,
    #[serde(default)]
    mutation_rate_remove_synapse: f32,
    #[serde(default)]
    mutation_rate_remove_neuron: f32,
    #[serde(default)]
    mutation_rate_add_neuron_split_edge: f32,
}

impl From<SeedGenomeConfigToml> for SeedGenomeConfig {
    fn from(raw: SeedGenomeConfigToml) -> Self {
        Self {
            num_neurons: raw.topology.num_neurons,
            num_synapses: raw.topology.num_synapses,
            spatial_prior_sigma: raw.topology.spatial_prior_sigma,
            vision_distance: raw.topology.vision_distance,
            starting_energy: raw.lifecycle.starting_energy,
            max_health: raw.lifecycle.max_health,
            age_of_maturity: raw.lifecycle.age_of_maturity,
            max_organism_age: raw.lifecycle.max_organism_age,
            plasticity_start_age: raw.plasticity.plasticity_start_age,
            hebb_eta_gain: raw.plasticity.hebb_eta_gain,
            juvenile_eta_scale: raw.plasticity.juvenile_eta_scale,
            eligibility_retention: raw.plasticity.eligibility_retention,
            max_weight_delta_per_tick: raw.plasticity.max_weight_delta_per_tick,
            synapse_prune_threshold: raw.plasticity.synapse_prune_threshold,
            mutation_rate_age_of_maturity: raw
                .mutation_rates
                .mutation_rate_age_of_maturity,
            mutation_rate_max_organism_age: raw
                .mutation_rates
                .mutation_rate_max_organism_age,
            mutation_rate_vision_distance: raw
                .mutation_rates
                .mutation_rate_vision_distance,
            mutation_rate_max_health: raw.mutation_rates.mutation_rate_max_health,
            mutation_rate_inter_bias: raw.mutation_rates.mutation_rate_inter_bias,
            mutation_rate_inter_update_rate: raw
                .mutation_rates
                .mutation_rate_inter_update_rate,
            mutation_rate_eligibility_retention: raw
                .mutation_rates
                .mutation_rate_eligibility_retention,
            mutation_rate_synapse_prune_threshold: raw
                .mutation_rates
                .mutation_rate_synapse_prune_threshold,
            mutation_rate_neuron_location: raw
                .mutation_rates
                .mutation_rate_neuron_location,
            mutation_rate_synapse_weight_perturbation: raw
                .mutation_rates
                .mutation_rate_synapse_weight_perturbation,
            mutation_rate_add_synapse: raw.mutation_rates.mutation_rate_add_synapse,
            mutation_rate_remove_synapse: raw
                .mutation_rates
                .mutation_rate_remove_synapse,
            mutation_rate_remove_neuron: raw.mutation_rates.mutation_rate_remove_neuron,
            mutation_rate_add_neuron_split_edge: raw
                .mutation_rates
                .mutation_rate_add_neuron_split_edge,
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
    evolution: WorldEvolutionToml,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldGeometryToml {
    world_width: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldPopulationToml {
    num_organisms: u32,
    #[serde(default = "default_periodic_injection_interval_turns")]
    periodic_injection_interval_turns: u32,
    #[serde(default = "default_periodic_injection_count")]
    periodic_injection_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldLifecycleToml {
    #[serde(default = "default_passive_metabolism_cost_per_unit")]
    passive_metabolism_cost_per_unit: f32,
    move_action_energy_cost: f32,
    #[serde(default = "default_reproduction_investment_energy")]
    reproduction_investment_energy: f32,
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
}

#[derive(Debug, Clone, Deserialize)]
struct WorldTerrainToml {
    #[serde(default = "default_terrain_noise_scale")]
    terrain_noise_scale: f32,
    #[serde(default = "default_terrain_threshold")]
    terrain_threshold: f32,
    #[serde(default = "default_spike_density")]
    spike_density: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct WorldEvolutionToml {
    #[serde(default = "default_global_mutation_rate_modifier")]
    global_mutation_rate_modifier: f32,
    #[serde(default = "default_meta_mutation_enabled")]
    meta_mutation_enabled: bool,
    #[serde(default = "default_runtime_plasticity_enabled")]
    runtime_plasticity_enabled: bool,
    #[serde(default = "default_force_random_actions")]
    force_random_actions: bool,
}

impl WorldConfigToml {
    fn into_runtime(self, seed_genome_config: SeedGenomeConfig) -> WorldConfig {
        WorldConfig {
            world_width: self.world.world_width,
            num_organisms: self.population.num_organisms,
            periodic_injection_interval_turns: self.population.periodic_injection_interval_turns,
            periodic_injection_count: self.population.periodic_injection_count,
            food_energy: self.food.food_energy,
            passive_metabolism_cost_per_unit: self.lifecycle.passive_metabolism_cost_per_unit,
            move_action_energy_cost: self.lifecycle.move_action_energy_cost,
            reproduction_investment_energy: self.lifecycle.reproduction_investment_energy,
            action_temperature: self.lifecycle.action_temperature,
            intent_parallel_threads: self.lifecycle.intent_parallel_threads,
            food_regrowth_interval: self.food.food_regrowth_interval,
            food_regrowth_jitter: self.food.food_regrowth_jitter,
            terrain_noise_scale: self.terrain.terrain_noise_scale,
            terrain_threshold: self.terrain.terrain_threshold,
            spike_density: self.terrain.spike_density,
            global_mutation_rate_modifier: self.evolution.global_mutation_rate_modifier,
            meta_mutation_enabled: self.evolution.meta_mutation_enabled,
            runtime_plasticity_enabled: self.evolution.runtime_plasticity_enabled,
            force_random_actions: self.evolution.force_random_actions,
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
    let seed_genome_config: SeedGenomeConfigToml = toml::from_str(seed_genome_raw)?;
    Ok(world_config.into_runtime(seed_genome_config.into()))
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

fn default_max_health() -> f32 {
    seed_genome_config_defaults().max_health
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

fn default_periodic_injection_interval_turns() -> u32 {
    world_config_defaults()
        .population
        .periodic_injection_interval_turns
}

fn default_periodic_injection_count() -> u32 {
    world_config_defaults().population.periodic_injection_count
}

fn default_passive_metabolism_cost_per_unit() -> f32 {
    world_config_defaults().lifecycle.passive_metabolism_cost_per_unit
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
