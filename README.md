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
  - Each organism has a 3-layer neural circuit with sensory, inter, and action
    neurons.
  - Inter neurons are leaky integrators (`alpha` from log time constants) with
    tanh nonlinearity.
  - Synapse weights are initialized from a log-normal magnitude distribution,
    then updated by runtime plasticity and genome mutation operators.
  - Action policy is categorical sampling from softmax logits with configurable
    `action_temperature`; sampling is deterministic for fixed seed + turn +
    organism ID.
  - Runtime plasticity uses coactivation traces, dopamine from baseline-corrected
    energy delta, weight decay, and maturity-gated pruning.
- Environment:
  - Toroidal axial hex grid with static terrain walls generated from Perlin
    noise.
  - Food ecology uses a binary fertility map from Perlin noise; regrowth is
    event-driven per cell.
  - Occupancy is single-entity per cell (`Organism`, `Food`, or `Wall`).
- Algorithms:
  - Simulation advances in deterministic tick phases with stable tie-breaking.
  - Batch world runs are supported on the server and execute in parallel workers.
- Evolution:
  - Reproduction is asexual with per-operator mutation-rate genes and optional
    meta-mutation.
  - NEAT-style topology mutators are implemented: add synapse, remove synapse,
    and split-edge/add-neuron.
  - Spatial priors bias new synapse creation based on neuron positions.
  - The server maintains a persistent champion genome pool and bootstraps new
    sessions/batch worlds from it when available, instead of always starting
    from the primitive seed genome.
  - Periodic random injections add fresh seed-genome organisms.
- Ecology:
  - `Eat` consumes food only. Plants return `20%` of stored energy; corpses
    return `80%`.
  - `Attack` damages organisms only when split attack actions are enabled and
    spawns corpse food on lethal hits.
  - Passive metabolism scales with inter-neuron count, sensory count, synapse
    count, and vision distance.
  - Reproduction requires maturity and energy investment, then completes after a
    lock delay.

## Layout

- `sim-types/` — shared domain types used across all Rust crates.
- `sim-config/` — world/seed-genome configuration crate, TOML loader,
  validation, owned defaults/policies, and generated config reference.
- `sim-core/` — deterministic simulation engine:
  - `lib.rs` — `Simulation` struct, config validation.
  - `turn.rs` + `turn/` — tick orchestration split into lifecycle, snapshot,
    intents, move resolution, reproduction, and commit modules.
  - `brain.rs` + `brain/` — genome expression, sensing, and evaluation helpers.
  - `plasticity.rs` — runtime eligibility/coactivation and Hebbian updates.
  - `genome.rs` + `genome/` — seed generation, mutation-rate accessors, scalar
    mutation, topology mutation, spatial prior logic, and sanitization.
  - `spawn.rs` + `spawn/` — organism spawning, terrain generation, and food
    ecology/regrowth.
  - `topology.rs` — shared neuron/synapse topology helpers and invariants.
  - `grid.rs` — hex-grid geometry and occupancy helpers.
- `sim-validation/` — headless validation harness split into CLI,
  orchestration, aggregation, comparison, output, and report modules.
- `sim-server/` — Axum HTTP + WebSocket server. Server-only API types live in
  `src/protocol.rs`.
- `web-client/` — React + TailwindCSS + Vite canvas UI.
- `sim-config/config.toml` — baseline simulation configuration.
- `sim-config/CONFIG_REFERENCE.md` — generated config reference derived from the
  config source of truth.

## Quickstart

1. `cargo check --workspace`
2. `cargo test --workspace`
3. Start server: `cargo run -p sim-server`
4. In another shell: `cd web-client && npm install && npm run dev`
5. Open `http://127.0.0.1:5173`

## Validation Harness

Use `sim-validation` to benchmark the evolution loop and inspect whether
behavioral adaptation is emerging.

- Cargo command (release):
  - `cargo run -p sim-validation --release --`
- Make command:
  - `make validate`
- Baseline/random-action control:
  - `cargo run -p sim-validation --release -- --baseline`
- Faster smoke run:
  - `cargo run -p sim-validation --release -- --ticks 1000 --report-every 250`

By default the harness runs a fixed 10-seed benchmark suite. You can override it
with `--seed` and a comma-separated list such as `--seed 42,123,7`.

Each run writes artifacts under `artifacts/validation/...` including
`timeseries.csv`, `summary.json`, and `report.html`.

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

Runtime hyperparameters live in `sim-config/config.toml`. Config parsing and
validation live in `sim-config`, which now also owns explicit defaults/policy
objects for world config, derived-world policy, terrain generation, and food
ecology. The generated reference at `sim-config/CONFIG_REFERENCE.md` is the
documentation source of truth.

`intent_parallel_threads` controls the per-simulation Rayon worker count used by
intent/brain evaluation and post-commit runtime plasticity. This is
machine-specific; tune it to your CPU core budget.

## Turn Pipeline

Phases execute in this order each tick:

1. **Lifecycle** — deduct passive metabolism from each organism:
   `(food_energy / 10000) * (num_neurons + sensory_count + synapse_count + vision_distance / 3)`.
   Remove any with `energy <= 0` or `age_turns >= max_organism_age`.
2. **Snapshot** — freeze occupancy and organism state. Stable ordering by ID.
3. **Intent** — evaluate each brain against the frozen snapshot. Select exactly
   one action per organism (softmax categorical sample with deterministic per-org
   RNG), then derive facing/move/consume/reproduce intents. Any non-`Idle`
   action pays one `move_action_energy_cost`. Runtime plasticity coactivations
   are accumulated here.
4. **Reproduction trigger** — `Reproduce` starts a lock (`2` turns) if mature and
   energy is sufficient. Parent pays `reproduction_investment_energy`
   immediately.
5. **Move resolution** — resolve all move intents simultaneously. Contenders for
   the same empty cell: highest confidence wins, ties broken by lower ID.
6. **Commit** — apply facing/moves/action costs, then resolve interactions.
   `Eat` only consumes food and `Attack` only damages organisms. Plant food
   grants `20%` of stored energy, corpse food grants `80%`, lethal attacks
   spawn corpse food, and spike tiles damage organisms after movement. Due food
   regrowth is processed here.
7. **Age** — increment `age_turns` for all survivors.
8. **Spawn** — complete reproduction locks into spawn requests (if back cell is
   free), apply genome mutation, and inject periodic seed-genome spawns.
   Post-commit runtime plasticity updates run after spawning.
9. **Metrics & delta** — increment turn counters/metrics and emit `TickDelta`.

Fixed config + seed + command sequence produces identical results. Organism ID
ordering is the universal tie-breaker.

## Brain

Sensory receptors are:

- `LookRay { ray_offset, look_target }` with `ray_offset = 0` and
  `look_target in {Food, Organism, Wall, Spikes}`.
- `ContactAhead`
- `Damage`
- `Energy`.

This yields 7 sensory neurons total (4 look-ray + contact + damage + energy).
Ray scans are cached per tick and return the closest visible target signal
`(max_dist - dist + 1) / max_dist`; walls and occupied cells occlude farther
targets along a ray, and spikes are sensed even on otherwise empty tiles.

Neuron IDs: sensory `0..SENSORY_COUNT-1`, inter `1000..1000+n`, action
`2000..2000+ACTION_COUNT-1` for the non-idle action set.

Evaluation order: sensory→inter, inter→inter (previous tick activations), then
sensory→action and inter→action. Inter uses per-neuron leaky integration:
`h_i(t) = (1 - alpha_i) * h_i(t-1) + alpha_i * tanh(z_i(t))`, where
`z_i(t) = b_i + sensory_to_inter_i + inter_to_inter_i(h(t-1))` and `alpha_i` is
derived from a log-time-constant parameterization. Actions are scored with
logits and selected by temperature-scaled categorical sampling.

Action neurons map to `TurnLeft`, `TurnRight`, `Forward`, `Eat`, `Attack`, and
`Reproduce`. `Idle` is handled separately by the policy logic.

### Runtime Plasticity

After brain evaluation, pending coactivations are accumulated. After commit and
spawn, runtime plasticity folds those traces into eligibilities and optionally
updates weights:

1. **Pending coactivation** — for every synapse, store instantaneous `pre * post`
   during intent evaluation.
2. **Reward signal** — dopamine is derived from the reward ledger signal,
   scaled by `20.0`, passed through `tanh`, and multiplied by the optional
   reward-signal override used by the validation harness.
3. **Weight update** — mature organisms use `hebb_eta_gain`; juveniles also
   learn before maturity, scaled by `juvenile_eta_scale`. Per-tick updates are
   clamped by `max_weight_delta_per_tick`.
4. **Eligibility fold-in** — each tick:
   `e = eligibility_retention * e + (1 - eligibility_retention) * pending_coactivation`.
5. **Synapse pruning** — mature organisms prune every 10 ticks when both weight
   and eligibility fall below the configured thresholds.

## Genome & Mutation

`OrganismGenome`: `num_neurons`, `num_synapses`, `spatial_prior_sigma`,
`vision_distance`, `starting_energy`, `age_of_maturity`,
`plasticity_start_age`, `hebb_eta_gain`, `juvenile_eta_scale`,
`eligibility_retention`, `max_weight_delta_per_tick`,
`synapse_prune_threshold`, per-operator mutation-rate genes, `inter_biases`,
`inter_log_time_constants`, `sensory_locations`, `inter_locations`,
`action_locations`, and sorted `edges` (`SynapseEdge`) with runtime
`eligibility` + `pending_coactivation`.

Mutation applies to offspring only. Each operator is gated by its own mutation
rate gene. With `meta_mutation_enabled = true`, mutation-rate genes are
self-adapted in latent/logit space (shared + per-gene Gaussian perturbation,
clamped to `[1e-4, 0.5]`).

Implemented operators:

- `age_of_maturity` +/-1 `[0, 10000]`.
- `vision_distance` +/-1 `[1, 32]`.
- Inter bias perturbation: Gaussian additive (stddev `0.15`), clamped
  `[-1.0, 1.0]`.
- Inter log-time-constant perturbation: Gaussian additive (stddev `0.05`),
  clamped to valid range.
- `eligibility_retention` perturbation: Gaussian additive (stddev `0.05`),
  clamped `[0.0, 1.0]`.
- `synapse_prune_threshold` perturbation: Gaussian additive (stddev `0.02`),
  clamped `[0.0, 1.0]`.
- Neuron-location perturbation for sensory/inter/action coordinates.
- Synapse-weight mutation: edge-wise perturbation/replacement.
- Add edge: sampled with spatial prior and weighted-without-replacement
  selection.
- Remove edge: random swap-remove.
- Split-edge-with-neuron: sample one existing edge (weighted by `|weight|`),
  remove it, add a new interneuron, and insert `(old_pre -> new_inter)` +
  `(new_inter -> old_post)`.

## Food & Fertility

Food fertility is binary and generated once per world from Perlin noise using
the hidden food-ecology policy owned by `sim-config` and documented in
`sim-config/CONFIG_REFERENCE.md`:

- `FOOD_FERTILITY_NOISE_SCALE = 0.012`
- `FOOD_FERTILITY_THRESHOLD = 0.6`

Terrain-wall cells are always infertile. Initial seeding fills every fertile
empty cell with food.

Regrowth is event-driven with per-cell due-turn tracking and a
`BTreeMap<u64, Vec<cell_idx>>` schedule. Delay is
`food_regrowth_interval +/- food_regrowth_jitter` (min 1). If a due cell is
occupied by an organism, regrowth is deferred by one turn.

## Lineage

Core simulation no longer performs speciation clustering. Organisms carry
`generation` and offspring increment it by one. (The web client may derive
UI-only grouping fields from generation.)

## Server API

REST:

- Batch runs: create/status (`/v1/world-runs`, `/v1/world-runs/{id}`)
- Archived worlds: list/delete/instantiate session (`/v1/worlds*`)
- Sessions: create/get state/step/reset/archive/focus/stream (`/v1/sessions*`)

Runtime sessions and batch runs created from config use a persisted champion
genome pool at `sim-server/champion_pool.json` when it exists. Archived worlds
still resume exact saved simulation state.

WS commands: `Start { ticks_per_second }` (`0` means unbounded/headless speed),
`Pause`, `Step { count }`, `SetFocus { organism_id }`.

`TickDelta`: `turn`, `moves`, `facing_updates`, `removed_positions` (unified
`RemovedEntityPosition[]` with `entity_id: EntityId` tagged `Organism` or
`Food`), `spawned`, `food_spawned`, `metrics`. `WorldSnapshot.occupancy` now
contains tagged occupancy cells for `Organism`, `Food`, and `Wall`. Clients
apply deltas incrementally; periodic full `WorldSnapshot`s serve as sync points.

`WorldSnapshotView` and `TickDeltaView.spawned` use compact
`WorldOrganismState` payloads. Full organism brain/genome state is sent via
`FocusBrain`.

## Web API Normalization

`web-client/src/types.ts` separates wire-facing `Api*` types from normalized
UI types. `web-client/src/protocol.ts` unwraps scalar IDs, normalizes snapshot
and delta payloads, and derives UI-only fields at the HTTP/WS boundary.

## Performance

`cargo bench -p sim-core --bench turn_throughput`

Performance regression guard tests (ignored by default):

`cargo test -p sim-core --release performance_regression -- --ignored --nocapture`

Optional CI budget override:

`SIM_CORE_TICK_BUDGET_NS_PER_TURN=130000 make perf-test`
