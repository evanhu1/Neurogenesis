# Systems & Mechanisms

A master list of the major systems across the **brain**, **evolution**, and
**ecology**, accurate to the current engine. Two systems were deliberately
removed and are **not** present anywhere below: the actor-critic / TD-learning /
dopamine / genetic-reward system, and the 2D cartesian "spatial brain" (neuron
coordinates, distance-biased wiring, and per-synapse wiring-length metabolism).

File pointers cite the owning module + function/constant; line numbers drift, so
the symbol names are the durable reference.

---

## The Brain

A 3-layer feedforward+recurrent network — **sensory → inter → action** — built
once at birth from the genome, then tuned over the organism's life by
unsupervised Hebbian plasticity. `sim-core/src/brain/`.

- **Structure & expression** (`brain/expression.rs::express_genome`). The default
  interface has 5 sensory neurons (stable IDs `0..5`); enabling
  `predation_enabled` adds 4 predation-only receptors (stable IDs `5..9`).
  `num_neurons` evolvable inter neurons use IDs `1000+`, and action neurons use
  IDs `2000+`. Inter→inter self-edges are allowed
  and act as gated memory (the presynaptic inter's *previous-tick* activation is
  read). Neurons are never added/removed after birth; only synapses are pruned.

- **Sensory system** (`brain/sensing.rs::encode_sensory_inputs`). The default
  five receptors are three `FoodRay`s (offsets −1/0/+1 relative to facing),
  `ContactAhead`, and `Energy` (`e/(e+max_health)`). With
  `predation_enabled=true`, three `OrganismRay`s and `Health`
  (`h/max_health`) are added.

- **Vision raycasting** (`sensing.rs::scan_ray`). Each ray marches toroidally up
  to `vision_distance`, stops at the nearest occupied cell, and reports food or
  an organism with linear distance falloff `(max−d+1)/max`. Walls and the first
  visible nonmatching entity occlude anything behind them. Organisms have no
  color phenotype and brains receive no RGB input.

- **Forward evaluation** (`brain/evaluation.rs::evaluate_brain`). Sensory
  activations fan into inter inputs. By default (`leaky_neurons_enabled=false`),
  each inter neuron uses the instantaneous NEAT-style activation
  `activation=fast_tanh(input)`. With the flag enabled it instead uses
  `state=(1−α)·state+α·input`, `activation=fast_tanh(state)`. Then inter
  activations fan into other inter neurons and into action logits; action biases
  are added last. `fast_tanh` (`brain/mod.rs`) is a Padé approximation clamped to
  ±1.

- **Inter-neuron dynamics / time constants** (`genome/scalar.rs::inter_alpha_from_log_time_constant`).
  `τ = exp(log_τ).clamp(0.1, 10.0)`, `α = 1 − exp(−1/τ)`. Low τ → fast,
  reactive neuron; high τ → slow, integrating memory. `log_τ` is per-neuron and
  evolvable; default ≈ −1.204 (τ ≈ 0.3).

- **Action selection** (`evaluation.rs::sample_action_from_logits`). Softmax over
  5 active action logits by default, or all 6 when `predation_enabled=true`,
  **plus an implicit Idle option** (`EXPLICIT_IDLE_LOGIT_BIAS = −0.01`). Attack
  has zero sampling probability while the flag is off. Health damage,
  regeneration, organism vision, the Health receptor, and predation resolution
  are part of the same flag treatment; the retained health fields are inert
  storage in the baseline. Sampling uses `action_temperature` (clamped ≥
  `1e-6`) and a deterministic per-organism uniform draw. `Idle` is not a neuron
  — it is the always-present "do nothing" outcome.

- **Plasticity — Hebbian covariance rule** (`brain/plasticity.rs`). No reward, no
  gating. Per edge: `pending = (pre − pre_mean)·(post − post_mean)` (centered
  covariance; the post term at the motor boundary is `fast_tanh(action_logit)`).
  Eligibility is a decaying sum `elig = eligibility_retention·elig + pending`,
  and the weight moves by `Δw = clamp(η·elig − 0.001·w, ±max_weight_delta_per_tick)`
  then `constrain_weight`. Activation means are EMAs with `ACTIVATION_MEAN_ALPHA
  = 0.05` (~20-tick window).

- **Critical period** (`plasticity.rs::learning_rate_scale`). Before
  `age_of_maturity`, the effective learning rate is scaled by
  `juvenile_eta_scale` (evolvable, default 0.5); mature organisms learn at full
  `hebb_eta_gain`.

- **Synapse pruning** (`plasticity.rs::prune_low_weight_synapses`). Every
  `SYNAPSE_PRUNE_INTERVAL_TICKS = 10` ticks **and only after maturity**, drop
  edges unless `|w| ≥ synapse_prune_threshold` or `|elig| ≥ 2·threshold`.

- **Gestation lock** (`brain/pending_action.rs`). Choosing `Reproduce` puts the
  organism into `PendingActionKind::Reproduce` for `gestation_ticks` turns,
  during which it neither acts nor learns.

---

## Evolution

Open-ended neuroevolution with no explicit fitness function. `sim-core/src/genome/`.

- **Genome structure** (`sim-types/src/lib.rs::OrganismGenome`). Four gene groups:
  - *Topology:* `num_neurons`, `num_synapses`, `vision_distance`.
  - *Lifecycle:* `age_of_maturity`, `gestation_ticks` (0–10),
    `max_organism_age`.
  - *Plasticity:* `hebb_eta_gain` [0,0.2], `juvenile_eta_scale` [0,4],
    `eligibility_retention` [0,1] (def 0.95), `max_weight_delta_per_tick`
    [0.005,0.5] (def 0.05), `synapse_prune_threshold` [0,1].
  - *Brain topology:* `inter_biases[]`, `inter_log_time_constants[]`,
    `action_biases[6]`, `edges[]` (sorted, unique `(pre,post)`).
  - *Mutation rates:* 16 per-operator rates (see meta-mutation).

- **Seed genome** (`genome/seed.rs::generate_seed_genome`). Fresh founders start
  with the configured `num_neurons` (baseline **0** inter neurons), random biases
  and time constants, **0 edges**, then `reconcile_synapse_count` adds
  `num_synapses` uniformly-random synapses. Sensors wire straight to actions.

- **Structural mutations** (`genome/topology.rs`).
  - *Add synapse* — pick a uniformly-random unconnected `(pre,post)` pair
    (`synapse_creation.rs::add_synapse_genes`, priority `−ln(U)` per candidate,
    smallest wins); weight log-normal, 80% excitatory.
  - *Remove synapse* — delete a random edge.
  - *Remove neuron* — drop a random inter neuron, its bias/τ, and incident edges;
    remap higher IDs down by one (keeps edges sorted).
  - *Split edge into neuron* — pick an edge weighted by `|w|`, insert a new inter
    neuron whose bias/τ average the endpoints (perturbed by
    `NEW_NEURON_PERTURBATION_SCALE = 0.5`), rewire `pre→new` (w=1) and
    `new→post` (old w).

- **Weight, bias & scalar mutations** (`genome/scalar.rs`, `genome/mod.rs`). Weight
  perturbation is multiplicative log-normal with a 10% full-replacement chance;
  inter/action biases and `log_τ` perturb a per-neuron fraction (`*_PERTURB_NEURON_RATE
  = 0.8`); lifecycle/plasticity scalars use clamped-Gaussian or log-normal
  (`LARGE_UNBOUNDED_LOG_STDDEV = 0.1`) steps.

- **Meta-mutation** (`genome/mutation_rates.rs`). The 16 mutation-rate genes are
  themselves evolvable. 1–3 are mutated per offspring in **logit space** with a
  pull toward the seed baseline (`META_MUTATION_BASELINE_PULL = 0.15`), an
  exploration tail (10% chance, wider σ), and an exploration floor. **Zero is
  absorbing** — a rate inherited as exactly 0 hard-disables that operator for the
  lineage.

- **Reproduction & inheritance** (`turn/reproduction.rs`, `spawn/organisms.rs`).
  Requires intent + `energy ≥ transfer` + `age ≥ age_of_maturity` + a free target
  cell. Parent pays `offspring_transfer_energy(gestation_ticks) = 100 +
  100·gestation_ticks`, gestates, then the offspring spawns behind the parent
  (opposite facing, `generation+1`) from a **`mutate_genome`'d clone**. Blocked
  births are recorded, not retried indefinitely.

- **Champion pool** (`sim-server/src/main.rs`). Up to 32 unique best genomes
  persisted to `champion_pool.json`, ranked by generation → reproductions →
  consumptions → energy → age → recency. New worlds seed organisms from the pool;
  progress compounds across sessions.

- **Periodic injection** (`spawn/organisms.rs::enqueue_periodic_injections`). Every
  `periodic_injection_interval_turns`, drop `periodic_injection_count` fresh seed
  genomes via rejection sampling — keeps diversity flowing and guards against
  extinction.

- **Selection** — purely ecological. No fitness target, no species registry, no
  speciation bookkeeping. What survives and out-reproduces in the world is what
  propagates.

- **Determinism & sanitization** (`genome/mod.rs::mutate_genome`,
  `genome/sanitization.rs`). Each operator is gated by exactly one RNG draw in a
  fixed order (new gates are appended so existing draw prefixes stay stable);
  `align_genome_vectors` clamps lengths/values, drops invalid or non-finite
  edges, and keeps edges sorted-unique on every intake path.

---

## Ecology

A closed-energy ecosystem on a hex world where lossy digestion is the only sink.
`sim-core/src/turn/`, `sim-core/src/spawn/`, `sim-core/src/metabolism.rs`.

- **Tick pipeline** (`turn/mod.rs::Simulation::tick` — canonical phase order):
  1. **Lifecycle** — charge passive metabolism; starvation & old-age deaths.
  2. **Intents** — evaluate every brain in parallel, select actions.
  3. **Reproduction** — decrement gestation, queue births.
  4. **Move resolution** — deterministic conflict resolution.
  5. **Commit** — apply moves, eating, predation, corpse spawning.
  6. **Age** — increment ages.
  7. **Spawn** — resolve queued offspring + periodic injections.
  8. **Plasticity** — runtime weight updates (skips newborns/gestating).
  9. **Metrics** — tally actions, consumptions, predations, deaths.

- **World & grid** (`grid.rs`). Toroidal `world_width × world_width` hex grid in
  axial `(q,r)`, row-major indexing, **one entity per cell** (organism, food, or
  wall) via an occupancy vector.

- **Terrain** (`spawn/world.rs`). Impassable walls/mountains come from
  single-octave Perlin noise above `terrain_threshold`. There are no damaging
  hazard tiles.

- **Food / plant ecology** (`spawn/food.rs`). Exactly `food_tile_fraction` of
  non-wall cells are selected by a deterministic, seed-keyed random ranking as
  persistent plant tiles; there is no spatial noise field or fertility jitter.
  Regrowth is **event-driven**: eating a plant schedules a refill at
  `turn + regrowth_interval ± jitter` (a `BTreeMap` queue), retried if the cell
  is occupied. Plants carry `food_energy`; corpses carry the dead organism's
  remaining energy.

- **Energy economy** (`turn/mod.rs`, `spawn/food.rs`). Energy is created by plant
  regrowth (each plant holds `config.food_energy`) and drained by metabolism;
  **eating transfers a food item's energy in full** (no digestion multiplier).
  Corpses are discounted at creation, not at the bite: a corpse stores
  `CORPSE_ENERGY_RETENTION = 0.80` of the dead organism's leftover energy
  (`spawn_corpse_at_cell`). No hard population cap — thermodynamics regulates
  standing population.

- **Metabolism** (`metabolism.rs`). Per-tick passive cost
  `= passive_metabolism_cost_per_unit · (inter + sensory neuron counts +
  vision_distance/3 + coeff · max_health^0.75)`. Kleiber `^0.75` body scaling;
  **no per-synapse cost** (removed) — every *neuron* a lineage keeps must pay for
  itself.

- **Movement & intent resolution** (`turn/moves.rs`). Move candidates targeting a
  free cell are sorted by `(target cell, confidence desc, organism_id asc)` and
  deduped per cell, so the highest-confidence organism wins and ties break by ID
  — fully deterministic and parallel-safe. Turning rotates facing; Forward steps
  one hex.

- **Predation** (`turn/commit.rs`). On `Attack` into an occupied cell, success
  probability `= clamp(predator_size / prey_size, 0, 1)` resolved by a
  deterministic hash; on success the prey loses `ATTACK_DAMAGE_FRACTION = 0.5` of
  the **attacker's** max-health, so bigger predators hit harder regardless of how
  tough the prey is. No direct energy transfer — a kill leaves a **corpse** that
  must be eaten, so death feeds the food web.

- **Consumption** (`turn/commit.rs`). `Eat` into a food cell removes it, credits
  the lossy energy fraction, and (for plants) schedules regrowth. `Idle`
  regenerates `HEALTH_REGEN_FRACTION = 0.10` of max-health.

- **Health & death** (`turn/lifecycle.rs`, `turn/commit.rs`). Deaths come from
  starvation (`energy ≤ 0`), old age (`age ≥ max_organism_age`), or predation
  damage. Old-age and damage deaths leave a corpse;
  starvation does not.

- **Determinism** (`turn/mod.rs`). Every stochastic choice (action sampling,
  predation rolls) is a stateless hash of `(seed, turn, organism IDs)`, so the
  parallel passes can't reorder history: fixed config + seed reproduces the world
  tick-for-tick on a given build.

---

## A unifying quantity: "size"

`sim_types::get_size(organism) = offspring_transfer_energy(gestation_ticks) = 100
+ 100·gestation_ticks` is a single evolvable body-size dial that simultaneously
sets an organism's **max-health/energy budget**, its **predation size-ratio**
(attacker vs. prey) and the **damage it deals when attacking** (`0.5 ·
max_health`), the **energy it invests per offspring**, and its **Kleiber
metabolic cost**. Evolving `gestation_ticks` therefore trades reproductive
investment, combat weight, durability, and upkeep all at once.
