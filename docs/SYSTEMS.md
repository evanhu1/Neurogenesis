# Systems & Mechanisms

A master list of the major systems across the **genome**, **brain**, **evolution**,
and **ecology**, accurate to the current engine after the substrate redesign.

The defining property of the current design is the **substrate/environment
split**: the genome, developmental map, brain runtime, operators, and tick loop
live in `sim-substrate` and know nothing about any world. A world is an
`Environment` implementation (`sim-hexworld`, `sim-toyenv`) that supplies physics
only, reading bodies through an immutable `BodyView` and requesting changes
through an `EffectSink`.

File pointers cite the owning module + type/function; line numbers drift, so the
symbol names are the durable reference.

---

## The Genome (indirect / generative encoding)

`sim-substrate/src/{cppn,genome}.rs`. The genome is **not** a list of synapses; it
is a compact program that *develops* into the phenotype.

- **`Genome`** = `{ cppn: CppnGenome, header: HeaderGenes }`.
- **`CppnGenome`** (`cppn.rs`) — a NEAT-style graph: `CppnNodeGene`s (id, kind,
  activation ∈ {Tanh, Sigmoid, Gaussian, Sin, Abs, Linear}, bias) and
  `CppnConnGene`s (innovation, from, to, weight, enabled). **Fixed I/O**: 11
  inputs (two 3-D substrate coordinates `x1,y1,z1 / x2,y2,z2`, their deltas,
  euclidean distance, a bias lane) and 6 outputs — `W` (weight), `LEO`
  (link-expression), `NBIAS` (neuron bias), `LTC` (log time constant), `AFF`
  (affordance), `PLR` (per-edge plasticity-rate scale).
- **Structural-hash identity** (`rng.rs::hash_conn`/`hash_node`). Connection and
  split-node ids are domain-separated 64-bit hashes of their structure, **not** a
  global innovation counter — so crossover aligns homologous genes
  deterministically under parallel, continuous-birth reproduction.
- **`HeaderGenes`** (`genome.rs`) — direct scalars the CPPN does not paint:
  global plasticity (`hebb_eta_gain` [0,0.2], `juvenile_eta_scale` [0,4],
  `eligibility_retention` [0,1], `max_weight_delta_per_tick` [0.005,0.5],
  `synapse_prune_threshold`), lifecycle (`age_of_maturity`, `gestation_ticks`
  0–10, `max_organism_age`), develop params (AFF/LEO thresholds, weight scale,
  quadtree depth, variance threshold, `max_neurons`/`max_edges`), a normalized
  `morphology` vector aligned to the environment's morphology schema, and the
  15-gene `MutationRates` block.
- **Flat form** — `bincode::serialize(&genome)`; canonicalized (conns
  innovation-sorted, nodes by id) so it is bit-stable and cacheable.

---

## Development: genotype → phenotype

`sim-substrate/src/develop.rs::develop(&Genome, &SubstrateCatalog, &DevelopConfig)`.
Pure and RNG-free (so it is identical on every thread and cacheable by CPPN
content hash). ES-HyperNEAT pipeline:

1. **Compile** the CPPN (topological sort).
2. **Interface selection** — for each catalog sensor/actuator, self-probe the CPPN
   (`x1==x2`) and read `AFF`; it expresses iff `AFF > aff_threshold`. A **viability
   floor** forces the top-`AFF` candidate if none cross, so no organism is
   degenerate. Expressed sensors/actuators define the observation- and
   action-vector layouts (`ObsLayout`/`ActionLayout`).
3. **Hidden-neuron discovery** — an adaptive quadtree over the hidden plane
   subdivides where CPPN-output variance is high (bounded per birth by
   `quadtree_depth`/`max_neurons`, unbounded across generations); high-variance
   leaf centers become hidden neurons.
4. **Connections** — for each candidate `(src,tgt)` pair, query the CPPN; add an
   edge iff `LEO > leo_threshold`, with weight `clamp(W · weight_scale)` and a
   per-edge `plasticity_scale` from `PLR` (hybrid adaptive-HyperNEAT).
5. Neuron bias/α from `NBIAS`/`LTC`; **prune dangling** hidden neurons; enforce
   hard neuron/edge rails; assemble a `BrainNet`.

`Phenotype = { brain: BrainNet, obs_layout, action_layout, morphology_values }`.

---

## The Brain

`sim-substrate/src/brain.rs`. A developed network of input / hidden / output
neurons, tuned over the organism's life by unsupervised Hebbian plasticity. The
neuron math is ported verbatim from the pre-redesign engine.

- **Forward pass** (`BrainNet::step`). Inputs take the observation vector; hidden
  neurons are **leaky integrators** reading their *previous-tick* activation:
  `state = (1−α)·state + α·input`, `activation = fast_tanh(state)`; outputs then
  read the freshly-updated hidden + input activations to form action logits.
- **Time constants** (`inter_alpha_from_log_time_constant`). `τ = exp(log_τ)
  .clamp(0.1,10)`, `α = 1 − exp(−1/τ)`. Painted per hidden neuron by the CPPN's
  `LTC` output.
- **Action selection** (`sample_action`). Softmax over the expressed action logits
  **plus an implicit Idle** (`EXPLICIT_IDLE_LOGIT_BIAS = −0.01`), temperature
  clamped ≥ `1e-6`, categorically sampled with a deterministic per-organism draw.
  The embodiment layer samples so the `(seed,turn,id)` hash stays out of the
  substrate.
- **Plasticity — Hebbian covariance rule** (`plasticity.rs`). No reward. Per edge:
  `pending = (pre − pre_mean)·(post − post_mean)` (the output post term is
  `fast_tanh(logit)`); eligibility is a decaying sum
  `elig = eligibility_retention·elig + pending`; the weight moves by
  `Δw = clamp(plasticity_scale · m · η · elig − 0.001·w, ±max_weight_delta_per_tick)`
  then `constrain_weight`. `m` is a bounded within-tick energy-delta neuromodulator
  (`NEUROMOD_GAIN = 0.04`, band `[0.85,1.15]`), and `plasticity_scale` is the
  CPPN's per-edge `PLR`. Activation means are EMAs (`ACTIVATION_MEAN_ALPHA = 0.05`).
- **Critical period & pruning**. Before `age_of_maturity` the learning rate is
  scaled by `juvenile_eta_scale`; every 10 ticks after maturity, edges below
  `synapse_prune_threshold` (by weight or eligibility) are dropped.
- **Non-Lamarckian**. Learned weights are runtime-only and **discarded at
  reproduction** — the genome breeds, not the trained brain.

---

## Evolution

`sim-substrate/src/operators.rs`. Open-ended neuroevolution with no explicit
fitness function.

- **`mutate`** — a frozen, append-only gate sequence (each operator gated by one
  RNG draw so existing draw prefixes stay stable): header scalar perturbations,
  CPPN structural ops (perturb/replace weights, add-connection, add-node =
  split an edge, toggle-enable, mutate-activation, perturb-bias), then
  meta-mutation of the rate block.
- **Meta-mutation** — the 15 `MutationRates` genes are evolvable, mutated in
  **logit space** with a pull toward the seed baseline
  (`META_MUTATION_BASELINE_PULL = 0.15`) and an exploration tail. **Zero is
  absorbing** — a rate inherited as 0 hard-disables that operator for the lineage.
- **`crossover`** — NEAT innovation-aligned: walk both connection lists in
  innovation-sorted order; matching genes coin-flip from either parent,
  disjoint/excess from the fitter parent; header crossed per-scalar. One
  deterministic offspring.
- **`reproduce`** — sexual: `crossover(a,b) → mutate`.
- **Champion archive** (`qd.rs::QdArchive`, used by `sim-server`/`sim-evaluation`).
  A **Quality-Diversity MAP-Elites** grid keyed by a normalized behavior
  descriptor, keeping the highest-quality elite per niche. `coverage` (occupied
  cells) and `qd_score` (summed quality) are the open-ended progress signal.
- **Selection** — purely ecological. No fitness target, no generations counter, no
  species registry. What survives and out-reproduces is what propagates.

---

## The tick loop

`sim-substrate/src/driver.rs::PopulationDriver::tick` — the canonical phase order.
The driver owns the `Vec<Body>`, the RNG streams, brain eval, plasticity, and all
cross-organism bookkeeping; the environment supplies physics.

1. **Metabolism + lifecycle** — charge `env.metabolic_cost`; starvation / zero-
   health / old-age deaths; `env.on_deaths` (corpses).
2. **Sensing + action** — `env.observe` fills each obs vector; `brain.step`
   produces logits; `sample_action` picks an action with a `(seed,turn,id)` hash.
3. **Mating** — gather `Mate` intents (`env.mate_intent`), pair deterministically
   by `(target, confidence desc, id asc)`, set gestation.
4. **Action resolution** — `env.decode_intents` + `env.resolve_actions`; effects
   applied in handle order.
5. **World step** — `env.step_world` (food, hazards, social field).
6. **Gestation + births** — countdown gestation; on completion,
   `reproduce(parent, partner)` and place via `env.place_birth`.
7. **Age + plasticity** — increment ages; run `plasticity_step` for eligible bodies.

**Determinism**: every RNG draw is seeded from `(seed, turn, phase)` or hashed
from `(seed, turn, id)`, and every cross-organism decision is handle-ordered
serial code, so parallelism cannot reorder history — and save→load→advance is
byte-identical (`HexSim` bincode round-trip).

**No periodic injection**: the driver only seeds founders. A world that reaches
zero living organisms is **extinct** — `HexSim::tick()` records `extinct_at` and
becomes a no-op, and all run loops stop there.

---

## The hex ecology (`sim-hexworld`)

A closed-energy ecosystem on a hex world, implemented as an `Environment`.
`sim-hexworld/src/{lib,grid,catalog,sim}.rs`.

- **World & grid** (`grid.rs`, `lib.rs`). Toroidal `width × width` hex grid in
  axial `(q,r)`, one entity per cell (organism, food, wall) via per-cell arrays.
  Value-noise terrain places **walls** (high noise) and **spikes** (a mid band),
  plus a per-cell fertility field.
- **Catalog** (`catalog.rs`) — the substrate interface the world exposes: **18
  sensors** (12 vision = 3 ray offsets × 4 channels R/G/B/Shape, `ContactAhead`,
  and interoceptive `Energy`/`Health`/`EnergyDelta`/`LastActionForward`/
  `LastActionEat`), **6 actuators** (TurnLeft, TurnRight, Forward, Eat, Attack,
  **Mate**; Idle is implicit), and a **morphology** schema (body_color RGB,
  vision_distance, and a directly-evolvable `size` dial).
- **Vision raycasting** (`lib.rs::scan_ray`, ported). Each ray marches hex cells
  up to `vision_distance`, opacity-blending occupant/terrain color with linear
  distance falloff and accumulating remaining visibility until exhausted.
- **Metabolism** (`metabolic_cost`, ported). Per-tick cost scales with neuron
  count + vision range + Kleiber `max_health^0.75` body mass, with a homeostatic
  low-energy downregulation.
- **Movement** — Forward requests are resolved deterministically by
  `(target cell, confidence desc, handle asc)`, one winner per cell.
- **Predation** — `Attack` into an occupied cell succeeds with probability
  `clamp(attacker_size / prey_size, 0, 1)` (a deterministic roll); a lethal hit
  transfers the prey's retained energy to the attacker and suppresses the corpse
  (`CORPSE_ENERGY_RETENTION = 0.80`), so energy is conserved.
- **Food** — event-driven plant regrowth on fertile empty cells; corpses (dead
  organisms) drop food carrying 80% of leftover energy. `Eat` consumes the cell
  ahead.
- **Social-color field** — a zero-sum, hue-keyed energy transfer between hex
  neighbors (`Σ sin(hue_self − hue_neighbor)`), antisymmetric so it conserves
  energy.
- **`HexSim`** (`sim.rs`) — the concrete, serializable world (`{ driver, world,
  config, extinct_at }`) that the CLI/server/evaluation load, tick, and save. It
  exposes `population_stats`, `render_snapshot`, `behavior_descriptor` (for the QD
  archive), and per-organism reads.

---

## The second environment (`sim-toyenv`)

The "Chemotaxis Ribbon" — a 1-D toroidal ribbon with a drifting scalar nutrient
field, sensors `{gradient_sign, local_concentration, energy, crowding}`, actuators
`{move_left, move_right, eat, mate}`, morphology `{metabolic_thrift, gut_capacity}`
(no color, no vision, no health). It runs on the **unchanged** `sim-substrate` and
exists to prove the substrate carries no hex/embodiment assumptions.

---

## A unifying quantity: "size"

In the hex world, `size` is a directly-evolvable morphology dial (decoupled from
gestation, which the substrate header owns) that simultaneously sets an organism's
**max-health/energy budget**, its **predation size-ratio** (attacker vs. prey) and
the **damage it deals** (`0.5 · max_health`), the **energy it invests per
offspring**, and its **Kleiber metabolic cost**. Evolving `size` trades combat
weight, durability, reproductive investment, and upkeep all at once.
