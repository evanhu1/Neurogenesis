# Neurogenesis

Neurogenesis is a Rust-based neuroevolution artificial-life simulation.

Its overarching goal is to simulate the real world open-ended evolution of
cognition, from basic sensory-motor reactions in early bilaterians 600+ million
years ago to complex intelligence.

It adheres to an emergent design philosophy that combines biological realism
(i.e. modeling biological evolution along with the most simple and general
principles underlying life) with extreme computational efficiency (i.e.
massively compressing the timeline).

Inspirations:

- The starting place of bilaterian organisms/brains and a
  foraging/navigation-centric environment are directly inspired by Max Bennett’s
  book A Brief History of Intelligence, which examines the phylogenetic history
  of human intelligence starting at the very beginning of cellular life.
- The brain’s design is a mix of ideas from biological and artificial neural
  networks.
- The design of the environment and ecological dynamics is heavily inspired by
  real world evolutionary biology and ecology

The main challenge is creating an evolutionary curriculum that scales
environment complexity alongside cognitive evolution while avoiding convergence
to degenerate niches along the way.

## High level design

- Brain:
  - Each agent's brain is a neural network, without self connections, consisting
    of 3 types of neurons: sensory, inter, and action.
  - Inter Neurons are modeled as leaky integrators with time constant, bias term
    and tanh activation.
  - Synapses are simple floating weights, initialized with a log-normal
    distribution
  - Softmax of action logits are computed each tick to determine the single
    action output of that tick (one action per tick)
  - Hebbian plasticity is implemented with EMA coactivation eligibility traces
    per synapse multiplied by a global neuromodulatory signal hand-computed as
    energy delta between previous tick and current one.
  - Synapse weights decay by a small factor each tick, and are pruned if they
    drop below a threshold (e.g. 0.05) any time after the maturation period
- Environment:
  - Plants grow in patches according to a Perlin noise fertility map. These
    patches migrate over time to introduce seasonal non-stationarity. Plant
    growth is implemented via biomass accumulation per cell, with rate governed
    by the fertility value.
  - Mountains also are placed in patches according a different Perlin noise map
    to introduce path finding complexity
  - The world is a 2D hexagon grid and toroidal meaning edges wrap around
- Algorithms:
  - World is simulated in discrete time intervals called ticks. Each tick
    corresponds to one action taken by all organisms, roughly corresponding to 1
    second
  - At the highest level of simulation, multiple worlds are simulated in
    parallel on different CPU cores. These worlds are configured with small
    variations in fundamental constants, simulating universe fine-tuning for
    life
- Evolution:
  - Organisms asexually reproduce with mutation rates per gene of <5% on average
  - Mutation rates themselves are genetically encoded and can be mutated (meta
    mutation)
  - Mutations on continuous genes are Gaussian N(0, 1) perturbs
  - Occasionally a batch of random new organisms are injected into the
    simulation
  - Speciation bookkeeping happens by comparing genomic distance. A lineage
    starts with a founder and diverges when an ancestor has a genome of
    sufficient distance from the founder’s genome
  - During reproduction the child inherits the genome and starting brain
    topology. The rules of inheritance are designed to optimize for the Baldwin
    effect—in which the organism’s ability to learn within its lifetime is
    strongly selected for.
- Ecology:
  - 10% of energy is transferred between trophic levels
  - Metabolic energy cost is linearly proportional to neuron count in the brain.
    Actions also cost energy.
  - Reproduction becomes possible after the organism’s maturation period, and
    transfers a fixed amount of energy from parent to child, along with a
    smaller lost overhead energy

## Layout

- `sim-types/` — shared domain types used across all Rust crates.
- `sim-core/` — deterministic simulation engine:
  - `lib.rs` — `Simulation` struct, config validation.
  - `turn.rs` — turn pipeline, categorical action intents, movement/bite
    resolution.
  - `brain.rs` — neural network evaluation and genome expression.
  - `genome.rs` — genome generation, mutation, species assignment.
  - `spawn.rs` — initial population spawn and reproduction placement.
  - `grid.rs` — hex-grid geometry and occupancy helpers.
- `sim-server/` — Axum HTTP + WebSocket server. Server-only API types live in
  `src/protocol.rs`.
- `web-client/` — React + TailwindCSS + Vite canvas UI.
- `config/default.toml` — baseline simulation configuration.

## Quickstart

1. `cargo check --workspace`
2. `cargo test --workspace`
3. Start server: `cargo run -p sim-server`
4. In another shell: `cd web-client && npm install && npm run dev`
5. Open `http://127.0.0.1:5173`

## World Model

Toroidal axial hex grid, `(q, r)` coordinates with wraparound modulo
`world_width` on both axes. Capacity is `world_width * world_width`. Occupancy
is a dense `Vec<Option<Occupant>>` indexed by `r * world_width + q`, where
`Occupant` is `Organism(OrganismId)`, `Food(FoodId)`, or `Wall`. At most one
entity per cell.

Terrain walls are generated at world creation from Perlin noise and occupy cells
permanently. Walls block movement, vision rays, food placement, and spawn
placement.

Six hex directions: `East (q+1,r)`, `NorthEast (q+1,r-1)`, `NorthWest (q,r-1)`,
`West (q-1,r)`, `SouthWest (q-1,r+1)`, `SouthEast (q,r+1)`.

## Config

All hyperparameters are configured in `config/default.toml`. Parameters
configure either the World, or set default values for the first Species Genome.
World terrain is controlled by `terrain_noise_scale` and `terrain_threshold`.

## Turn Pipeline

Phases execute in this order each tick:

1. **Lifecycle** — deduct `neuron_metabolism_cost * (enabled interneuron count)`
   from all organisms. Remove any with `energy <= 0` or
   `age_turns >= max_organism_age`. Rebuild occupancy.
2. **Snapshot** — freeze occupancy and organism state. Stable ordering by ID.
3. **Intent** — evaluate each brain against the frozen snapshot. Select exactly
   one action per organism (categorical argmax over action neurons), then derive
   intent fields (`facing_after_actions`, `wants_move`, `move_target`,
   `wants_reproduce`, `move_confidence`, `action_cost_count`). Any non-`Idle`
   action pays one `move_action_energy_cost`. Apply runtime plasticity after
   evaluation.
4. **Reproduction** — organisms with `Reproduce` active,
   `age_turns >=
   age_of_maturity`, and sufficient energy queue a spawn
   request at the hex behind them (opposite facing). Cell must be unoccupied.
   Conflicts resolved by ID order. Energy deducted on success.
5. **Move resolution** — resolve all move intents simultaneously. Contenders for
   the same cell: highest confidence wins, ties broken by lower ID. Empty target
   = move in. Food target = consume and move in. Occupied organism target =
   passive bite (drain `min(prey_energy, food_energy * 2.0)` and transfer to
   predator) without displacement. Walls block movement. Vacated target
   (occupant also moving) = move in. Cycles resolve naturally.
6. **Commit** — atomically apply facing, moves, energy costs, consumption kills,
   energy transfers. Rebuild occupancy. Process due food regrowth events.
7. **Age** — increment `age_turns` for all survivors.
8. **Spawn** — process spawn queue in order. Offspring get mutated genome,
   opposite facing, and genome `starting_energy`. Species assigned by genome
   distance.
9. **Metrics & delta** — prune extinct species, increment turn counter, emit
   `TickDelta`.

Fixed config + seed + command sequence produces identical results. Organism ID
ordering is the universal tie-breaker.

## Brain

Sensory receptors are multi-ray:

- `LookRay { ray_offset, look_target }` with `ray_offset in [-2,-1,0,1,2]` and
  `look_target in {Food, Organism}`.
- `Energy`.

This yields 11 sensory neurons total (10 look-ray + 1 energy). Ray scans are
cached per tick and return the closest visible target signal
`(max_dist - dist + 1) / max_dist`; walls and occupied cells occlude farther
targets along a ray.

Neuron IDs: sensory `0..11`, inter `1000..1000+n`, action `2000..2008`.

Evaluation order: sensory→inter, inter→inter (previous tick activations), then
sensory→action and inter→action. Inter uses per-neuron leaky integration:
`h_i(t) = (1 - alpha_i) * h_i(t-1) + alpha_i * tanh(z_i(t))`, where
`z_i(t) = b_i + sensory_to_inter_i + inter_to_inter_i(h(t-1))` and `alpha_i` is
derived from a log-time-constant parameterization. Actions are scored and
resolved by single-choice categorical selection (argmax).

Actions: `Idle`, `TurnLeft`, `TurnRight`, `Forward`, `TurnLeftForward`,
`TurnRightForward`, `Consume`, `Reproduce`.

Interneurons carry a polarity (`Excitatory` or `Inhibitory`). Sensory neurons
are always excitatory. Dale's law is enforced for generated/mutated synapses:
all outgoing weights from a neuron share the sign implied by source type. Neuron
type distribution is biased towards an 80:20 excitatory:inhibitory ratio to
match biological brains.

### Runtime Plasticity

After brain evaluation each tick, runtime plasticity is applied:

1. **Eligibility trace update** — for every synapse:
   `e = eligibility_retention * e + pre * post`.
2. **Reward signal** —
   `dopamine = tanh((energy - energy_prev + passive_metabolism_baseline) / 10.0)`,
   then `energy_prev` is updated to current energy.
3. **Mature-only weight update** — once `age_turns >= age_of_maturity`:
   `w += eta * dopamine * e - 0.001 * w`, where `eta = max(0, hebb_eta_gain)`,
   with Dale-sign and clamp constraints preserved.
4. **Synapse pruning** — once `age_turns >= age_of_maturity`, every 10 ticks
   synapses with `|weight| < threshold` and `|eligibility| < 2 * threshold` are
   removed. `synapse_prune_threshold` is a mutable genome parameter.

## Genome & Mutation

`OrganismGenome`: `num_neurons`, `vision_distance`, `age_of_maturity`,
`hebb_eta_gain`, `eligibility_retention`, `synapse_prune_threshold`,
`inter_biases` (vec), `inter_log_time_constants` (vec), `interneuron_types`
(vec), `action_biases` (vec), `edges` (sorted `SynapseEdge` list with per-edge
`eligibility` trace), and explicit per-operator mutation-rate genes.

Mutation applies to offspring only. Each operator is gated by its own mutation
rate gene, and mutation-rate genes self-adapt every mutation step using
`time_constant = 1 / sqrt(2 * sqrt(n))`, where `n` is the number of
mutation-rate genes.

Implemented operators:

- `age_of_maturity` +/-1 `[0, 10000]`.
- `vision_distance` +/-1 `[1, 32]`.
- Add edge: random non-duplicate (20 retries), weight magnitude sampled from a
  log-normal distribution and signed by source neuron polarity.
- Remove edge: random swap-remove.
- Split-edge-with-neuron: sample one existing edge uniformly, remove it, add a
  new interneuron, and insert `(old_pre -> new_inter)` +
  `(new_inter -> old_post)`.
- Inter bias perturbation: Gaussian additive (stddev `0.15`), clamped
  `[-1.0, 1.0]`.
- Inter log-time-constant perturbation: Gaussian additive (stddev `0.05`).
- Action bias perturbation: Gaussian additive (stddev `0.15`), clamped
  `[-1.0, 1.0]`.
- `eligibility_retention` perturbation: Gaussian additive (stddev `0.05`),
  clamped `[0.0, 1.0]`.
- `synapse_prune_threshold` perturbation: Gaussian additive (stddev `0.02`),
  clamped `[0.0, 1.0]`.

Weight mutation is not an evolutionary operator; weights are initialised at
birth and modified only by Hebbian learning during the organism's lifetime.

## Food & Fertility

Food placement uses a Perlin-noise fertility map (4-octave fractal noise)
computed at world creation. Each cell has a fertility value in `[floor, 1.0]`
controlled by `food_fertility_noise_scale`, `food_fertility_exponent`, and
`food_fertility_floor`. Initial seeding accepts cells probabilistically based on
fertility.

Food regrowth is event-driven via a `BTreeSet<FoodRegrowthEvent>` priority queue
sorted by due turn. When food is consumed, a regrowth event is scheduled with
cooldown `min + (1 - fertility) * (max - min)` plus random jitter, divided by
`plant_growth_speed`. Occupied cells retry with exponential backoff. A
generation counter invalidates stale events.

## Species

Organisms carry a `SpeciesId`. The simulation maintains a `species_registry`
mapping `SpeciesId -> OrganismGenome`. Initialization seeds one species from
`seed_genome_config`. Species assignment uses `genome_distance()` — a merge-join
over sorted edge lists plus trait/bias L1 distance. Offspring whose distance
exceeds `speciation_threshold` from all existing founders create a new species.
Extinct species are pruned each turn.

## Server API

REST: create / get metadata / get state / step / reset / focus / stream.

WS commands: `Start { ticks_per_second }`, `Pause`, `Step { count }`,
`SetFocus { organism_id }`.

`TickDelta`: `turn`, `moves`, `removed_positions` (unified
`RemovedEntityPosition[]` with `entity_id: EntityId` tagged `Organism` or
`Food`), `spawned`, `food_spawned`, `metrics`. `WorldSnapshot.occupancy` now
contains tagged occupancy cells for `Organism`, `Food`, and `Wall`. Clients
apply deltas incrementally; periodic full `WorldSnapshot`s serve as sync points.

## Performance

`cargo bench -p sim-core --bench turn_throughput`

Performance regression guard tests (ignored by default):

`cargo test -p sim-core --release performance_regression -- --ignored --nocapture`

Optional CI budget override:

`SIM_CORE_TICK_BUDGET_NS_PER_TURN=130000 make perf-test`
