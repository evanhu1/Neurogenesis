# Config Reference

Generated from `sim-config/src/config.rs`.

## World Config: Population

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `world_width` | `required` | Toroidal world width used for both axial dimensions. |
| `num_organisms` | `required` | Target initial population before terrain limits are applied. |
| `periodic_injection_interval_turns` | `100` | Cadence for seed-genome injections. |
| `periodic_injection_count` | `100` | How many injection attempts run at each periodic-injection turn. |

## World Config: Lifecycle And Actions

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `passive_metabolism_cost_per_unit` | `0.005` | Per-complexity-unit passive energy drain applied each tick. |
| `food_energy` | `required` | Energy stored in each plant food item. |
| `move_action_energy_cost` | `required` | Flat energy cost applied to non-idle actions. |
| `reproduction_investment_energy` | `500` | Immediate parent energy investment when reproduction starts. |
| `action_temperature` | `0.5` | Softmax temperature for action sampling. |
| `intent_parallel_threads` | `8` | Worker-count default for intent evaluation. |

## World Config: Food Ecology

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `food_regrowth_interval` | `10` | Base plant regrowth delay in turns. |
| `food_regrowth_jitter` | `2` | Uniform +/- jitter applied to regrowth delay. |

## World Config: Terrain

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `terrain_noise_scale` | `0.02` | Perlin input scale for terrain walls. |
| `terrain_threshold` | `0.86` | Wall cutoff used by default terrain generation. |
| `spike_density` | `0.1` | Fraction of non-wall cells assigned spikes by deterministic per-cell scatter. |

## World Config: Evolution Features

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `global_mutation_rate_modifier` | `1` | Global multiplier applied to mutation-rate genes. |
| `meta_mutation_enabled` | `true` | Enables mutation of the mutation-rate genes themselves. |
| `runtime_plasticity_enabled` | `true` | Master toggle for mature runtime plasticity updates. |
| `force_random_actions` | `false` | Validation/debug override for random policy execution. |

## Seed Genome Config: Core

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `num_neurons` | `required` | Inter-neuron count for seed genomes. |
| `num_synapses` | `required` | Requested synapse count before sanitization. |
| `spatial_prior_sigma` | `required` | Spatial connectivity prior width. |
| `vision_distance` | `required` | Maximum look distance for look-ray sensors. |
| `starting_energy` | `1` | Default starting energy when omitted in seed-genome TOML. |
| `max_health` | `1` | Default maximum health when omitted in seed-genome TOML. |
| `age_of_maturity` | `required` | Age threshold for adulthood. |
| `max_organism_age` | `4294967295` | Default lifespan cap when omitted in seed-genome TOML. |

## Seed Genome Config: Plasticity

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `plasticity_start_age` | `0` | Default age when runtime plasticity may begin. |
| `hebb_eta_gain` | `required` | Base Hebbian learning-rate gain. |
| `juvenile_eta_scale` | `0.5` | Scaling factor for juvenile plasticity when enabled. |
| `eligibility_retention` | `required` | Eligibility trace retention factor. |
| `max_weight_delta_per_tick` | `0.05` | Clamp for per-tick synapse updates. |
| `synapse_prune_threshold` | `required` | Pruning threshold for mature synapses. |

## Seed Genome Config: Mutation Rates

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `mutation_rate_age_of_maturity` | `required` | Mutation rate for maturity age. |
| `mutation_rate_max_organism_age` | `required` | Mutation rate for organism lifespan. |
| `mutation_rate_vision_distance` | `required` | Mutation rate for vision distance. |
| `mutation_rate_max_health` | `required` | Mutation rate for maximum health. |
| `mutation_rate_inter_bias` | `required` | Mutation rate for inter biases. |
| `mutation_rate_inter_update_rate` | `required` | Mutation rate for inter-neuron time constants. |
| `mutation_rate_eligibility_retention` | `required` | Mutation rate for eligibility retention. |
| `mutation_rate_synapse_prune_threshold` | `required` | Mutation rate for prune thresholds. |
| `mutation_rate_neuron_location` | `required` | Mutation rate for sensory/inter/action locations. |
| `mutation_rate_synapse_weight_perturbation` | `required` | Mutation rate for synapse weight perturbation/replacement. |
| `mutation_rate_add_synapse` | `required` | Mutation rate for adding synapses. |
| `mutation_rate_remove_synapse` | `required` | Mutation rate for removing synapses. |
| `mutation_rate_remove_neuron` | `required` | Mutation rate for removing inter neurons. |
| `mutation_rate_add_neuron_split_edge` | `required` | Mutation rate for splitting an edge with a new neuron. |

## Hidden Policies: Terrain Generation

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `terrain_seed_mix` | `0xA5A5A5A5` | Seed xor used to derive terrain noise from the run seed. |
| `default_threshold` | `0.86` | Canonical terrain-wall threshold used by `build_terrain_map`. |
| `spike_seed_mix` | `0xCBBB9D5DC1059ED8` | Seed xor used to derive spike scatter hashes from the run seed. |

## Hidden Policies: Food Ecology

| Key | Default / Derivation | Notes |
| --- | --- | --- |
| `fertility_seed_mix` | `0x6A09E667F3BCC909` | Seed xor used to derive plant fertility noise. |
| `fertility_jitter_seed_mix` | `0x510E527F9B05688C` | Secondary seed xor used for fertility jitter. |
| `fertility_noise_scale` | `0.012` | Perlin input scale for fertility sampling. |
| `fertility_threshold` | `0.6` | Binary fertility cutoff after jitter. |

