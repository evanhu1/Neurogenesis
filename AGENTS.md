# Repository Guidelines

## Project Structure

- `sim-types/`: shared domain types used across all Rust crates.
- `sim-config/`: world + seed-genome config crate (`config.toml`,
  `seed_genome.toml`, typed loader/validation, owned defaults/policies,
  inline-commented TOML baselines).
- `sim-core/`: deterministic simulation engine:
  - `lib.rs`: `Simulation` struct, module declarations, config validation.
  - `metabolism.rs`: spatial-distance-weighted passive metabolism cost.
  - `turn/`: tick orchestration — `mod.rs` (tick pipeline), `lifecycle.rs`,
    `snapshot.rs`, `intents.rs`, `moves.rs`, `reproduction.rs`, `commit.rs`.
  - `brain/`: neural network — `mod.rs`, `evaluation.rs`, `expression.rs`,
    `sensing.rs` (receptor encoding), `plasticity.rs` (covariance/actor-trace
    updates and pruning), `reward.rs` (reward ledger + dopamine derivation),
    `actor_critic.rs` (critic value function), `topology.rs` (shared
    neuron/synapse helpers).
  - `genome/`: genome operations — `mod.rs` (core mutation loop),
    `mutation_rates.rs` (macro-generated rate accessors), `scalar.rs`,
    `topology.rs` (add-neuron/synapse mutations), `spatial_prior.rs`,
    `sanitization.rs`, `seed.rs`.
  - `spawn/`: organism spawning, world generation, food ecology — `mod.rs`,
    `organisms.rs`, `world.rs`, `food.rs`, `grid.rs` (hex-grid geometry and
    occupancy helpers).
- `sim-evaluation/`: headless evaluation harness built on a dataset-centric
  ETL pipeline (simulation emits raw facts to partitioned Parquet; all metrics
  and reports are derived post-hoc from that dataset):
  - `main.rs` + `cli.rs`: `Run` (default) and `Analyze <run>` subcommands.
  - `orchestration.rs`: multi-threaded per-seed harness, drives `Ledger` +
    dataset writer, then invokes the analysis layer.
  - `ledger.rs`: per-tick action/population bookkeeping used while the sim
    runs, producing the dataset rows.
  - `dataset/`: partitioned Parquet dataset — `schema.rs` (row types:
    `TickSummaryRow`, `PopulationSnapshotRow`, `ActionCountRow`,
    `OrganismLifetimeRow`, `ReproductionEventRow`, `GenomeSnapshotIndexRow`),
    `writer.rs`, `reader.rs`, `manifest.rs`.
  - `analysis/`: re-derives metrics from the dataset — `intervals.rs`
    (`IntervalMetrics` timeseries), `pillars.rs` (foraging / intelligence /
    competition scores), `demographics.rs` (reproduction + survival
    analytics), `averaging.rs` (cross-seed folding), `cli.rs` (`analyze_run`
    entrypoint with identifier resolution).
  - `report.rs` + `output.rs`: write `report.html`, `timeseries.csv`,
    `summary.json` artifacts at both the run root and per-seed directories.
  - `comparison.rs`: A/B treatment-vs-control with bootstrapped CIs.
  - `types.rs`: shared data structures (`IntervalMetrics`, `PillarScores`,
    `DemographicAnalytics`, `SeedEvaluationSummary`, `EvaluationSummary`).
- `sim-server/`: Axum HTTP + WebSocket server (`src/main.rs`), with server-only
  API types in `src/protocol.rs`.
- `web-client/`: React + TailwindCSS + Vite canvas UI (`src/`).
- `sim-config/config.toml`: baseline world simulation configuration.
- `sim-config/seed_genome.toml`: baseline seed-genome configuration.
- `sim-evaluation/config.toml`: stable evaluation world configuration.
- `sim-evaluation/seed_genome.toml`: stable evaluation seed-genome
  configuration. Keep both evaluation files in sync with the `sim-config`
  baseline when config-schema or baseline tuning changes affect evaluation
  assumptions.

## Build & Test

- `cargo check --workspace`: fast compile check.
- `cargo test --workspace`: run all Rust tests.
- `make fmt`: format Rust code.
- `make lint`: clippy with warnings as errors.
- `cargo run -p sim-evaluation --release --`: run the default 8-seed
  evolution-loop evaluation harness and generate
  `report.html`/`timeseries.csv`/`summary.json`.
- `cargo run -p sim-evaluation --release -- --seed <u64>[,<u64>...]`: override
  the default seed suite.
- `make evaluate ARGS="--seed <u64>[,<u64>...]"`: same harness via Makefile
  wrapper.
- `cargo run -p sim-evaluation --release -- analyze <run>`: re-derive
  `summary.json`/`timeseries.csv`/`report.html` from a persisted dataset
  without re-running the sim. `<run>` accepts a path (run root or seed dir),
  a timestamp prefix resolved under `artifacts/evaluation/`, or `latest`.
- `cargo run -p sim-server`: start backend on `127.0.0.1:8080`. Flags:
  - `--champion-pool-path <path>`: override the default `champion_pool.json`
    location used for read-back and persistence of saved champions.
  - `--seed-genome-snapshot <path.bin>`: load a single `OrganismGenome` from
    a bincode-encoded evaluation snapshot
    (`artifacts/evaluation/.../seed_<seed>/genomes/tNNNNNN.bin`) and use it
    as a one-entry in-memory champion pool; every initial organism spawns
    with this genome, and champion-save endpoints no-op (no disk writes) for
    the session.
- `cd web-client && npm run dev`: run frontend (Vite).
- `cd web-client && npm run build && npm run typecheck`: production build + TS.

## Coding Style

- Rust: `rustfmt`, `snake_case` functions, `PascalCase` types,
  `SCREAMING_SNAKE_CASE` constants.
- Rust performance is a top priority: avoid unnecessary allocations/copies,
  prefer zero-cost abstractions.
- TypeScript: strict mode, `camelCase` vars/functions, `PascalCase` types.
- Use Tailwind for frontend styling.
- Keep Rust and web data models synchronized using this split:
  - `web-client/src/types.ts` `Api*` types must match Rust wire schema
    (`sim-types/src/lib.rs` + `sim-server/src/protocol.rs`) exactly.
  - UI-only types in `web-client/src/types.ts` may extend/derive fields, but
    must be produced by normalization in `web-client/src/protocol.ts`.
  - Do not read raw API payloads directly in feature code; normalize at the
    HTTP/WS boundary first.
  - When changing wire fields, update Rust structs, TS `Api*` types, and
    normalizers in the same change.
- DO NOT CARE ABOUT backwards compatibility

## Testing

- Primary runner: `cargo test --workspace`.
- For evolution-loop benchmarking and regression checks, run `sim-evaluation`
  (prefer release mode) and compare outputs across seeds/configs.

## Frontend Browser Testing

- For frontend changes, verify behavior with `agent-browser` against the local app.
- Start the backend with `cargo run -p sim-server` and the frontend with
  `cd web-client && npm run dev`.
- If needed, install once with `npm install -g agent-browser && agent-browser install`.
- Typical flow: `agent-browser open http://127.0.0.1:5173`, then use
  `snapshot`, `click`, `fill`, `get text`, and `screenshot` to exercise the UI.
- Stop any background `sim-server` or Vite processes when finished.

## Architecture Context

### World

Toroidal axial hex grid `(q, r)` with wraparound modulo `world_width` on both
axes. Occupancy is a dense `Vec<Option<Occupant>>`
(`Organism(OrganismId)`, `Food(FoodId)`, or `Wall`), indexed by
`r * world_width + q`. At most one entity per cell. Terrain walls are generated
from Perlin noise at world init (`terrain_noise_scale`, `terrain_threshold`) and
block spawn, movement, and ray vision.

Food fertility is binary (boolean per cell), generated from Perlin noise using
the hidden food-ecology policy owned by `sim-config`. The current built-in
policy uses `FOOD_FERTILITY_NOISE_SCALE = 0.012`, and the default TOML
override is `food_fertility_threshold = 0.6`.
Terrain-wall cells are infertile.

### Turn Pipeline (execution order)

See `sim-core/src/turn/mod.rs::Simulation::tick` for the canonical sequence.

1. **Reconcile** — clear pending actions, roll the reward ledgers forward,
   reset transient per-tick state.
2. **Lifecycle** — deduct passive metabolism (see Metabolism below), remove
   starved/aged organisms, emit any lifecycle-triggered reproduction events.
3. **Snapshot** — capture `TurnSnapshot` (world width + organism count) for
   phases that need a stable view.
4. **Intents** — parallel brain evaluation. For each organism, sample one
   action from softmax logits over the 6 action neurons (deterministic sample
   stream keyed on `(seed, turn, organism_id)`); derive facing/move/eat/
   attack/reproduce intent. `Idle` is the fallback when no action neuron
   fires. Any non-`Idle` action pays `move_action_energy_cost`. When runtime
   plasticity is enabled, per-edge pending coactivations are computed here.
5. **Reproduction triggers** — `Reproduce` intents and lifecycle triggers
   queue gestation locks if the parent is mature and energy is sufficient;
   parent energy is debited immediately by the offspring transfer energy.
6. **Move resolution** — simultaneous movement into empty targets only;
   highest confidence wins, ties broken by lower ID. Occupied cells and walls
   block.
7. **Commit** — apply facing + moves + action costs; resolve forward-cell
   interactions. `Eat` only consumes food, `Attack` only damages organisms.
   Plant food grants 20% of stored energy, corpse food grants 80%, lethal
   attacks spawn corpse food, spike tiles damage organisms after movement.
   Process due food regrowth. Queue reproduction completions whose gestation
   finishes this tick.
8. **Age** — increment `age_turns` on survivors.
9. **Spawn** — turn spawn requests into new organisms (if cells are free),
   mutate genomes for offspring, process periodic seed-genome injections,
   then apply post-commit runtime weight updates.
10. **Consistency check** — debug assertions on occupancy/organism state
    (no-op in release).
11. **Metrics & delta** — advance `turn`, refresh counters, emit `TickDelta`.

### Brain

Sensory receptors — **15 neurons** in fixed order
(`SensoryReceptor::ordered()` in `sim-types/src/lib.rs`):

- **9 vision rays**: 3 ray offsets `[-1, 0, 1]` around the current facing ×
  3 RGB channels. Rays are cast to `vision_distance` cells, stop on the first
  blocker (walls, organisms, terrain), and accumulate per-channel color
  weighted by `(max_dist - dist + 1) / max_dist`.
- **6 scalar receptors** (in order): `ContactAhead`, `Energy`, `Health`,
  `EnergyDelta`, `LastActionForward`, `LastActionEat`.

Neuron ID ranges (`sim-core/src/brain/topology.rs`):

- Sensory: `0..16`
- Inter: `1000..1000 + num_neurons`
- Action: `2000..2005` — `TurnLeft`, `TurnRight`, `Forward`, `Eat`, `Attack`,
  `Reproduce`. `Idle` is a 7th `ActionType` but has no action neuron; it is
  the fallback when the sampling step does not commit to any of the six.

Policy is single-choice categorical sampling per tick from temperature-scaled
softmax logits.

Runtime plasticity (`sim-core/src/brain/plasticity.rs`):

- Pending coactivation:
  - Inter edges: mean-subtracted covariance,
    `pending = (pre - pre_mean) * (post - post_mean)` with `pre_mean`/
    `post_mean` tracked via EMA.
  - Action edges: `pending = pre * (a_i - p_i)`, where `a_i ∈ {0, 1}` marks
    the chosen action and `p_i` is its softmax probability.
- Eligibility fold: `e = retention * e + (1 - retention) * pending` with
  `retention = eligibility_retention` per-genome.
- Dopamine signal comes from the `reward.rs` ledger routed through an
  actor-critic baseline (`actor_critic.rs`), then `tanh`-squashed.
- Weight update: `Δw = η * dopamine * e - decay * w`, clamped to
  `max_weight_delta_per_tick`. Juveniles learn too, scaled by
  `juvenile_eta_scale`.
- Synapse pruning is maturity-gated (every 10 ticks), removing edges below
  a weight+eligibility threshold.

### Genome

`OrganismGenome` (`sim-types/src/lib.rs`) is composed of five sub-structs plus
a reward-weight vector:

- **`topology: TopologyGenes`** — `num_neurons`, `num_synapses`,
  `spatial_prior_sigma`, `vision_distance`.
- **`lifecycle: LifecycleGenes`** — `body_color`, `max_health`,
  `age_of_maturity`, `gestation_ticks`, `max_organism_age`.
- **`plasticity: PlasticityGenes`** — `hebb_eta_gain`, `juvenile_eta_scale`,
  `eligibility_retention`, `max_weight_delta_per_tick`,
  `synapse_prune_threshold`.
- **`mutation_rates: MutationRateGenes`** — per-operator rates for every
  mutation (age, vision, bias, inter update rate, retention, prune threshold,
  neuron locations, synapse weight perturb, add/remove synapse, add/remove
  neuron, etc.).
- **`brain: BrainTopology`** — `inter_biases`, `inter_log_time_constants`,
  `action_biases`, neuron-location vectors (`sensory_locations`,
  `inter_locations`, `action_locations`), and sorted `edges` (with runtime
  eligibility/pending-coactivation traces).
- **`reward_weights: Vec<f32>`** — per-channel coefficients applied to the
  reward ledger before the dopamine squash. Index order matches the
  `RewardLedger` fields in `sim-core/src/brain/reward.rs`.

Implemented mutation operators include: age-of-maturity / gestation /
max-age / vision / max-health scalar mutations, bias and update-rate
perturbation, retention / prune-threshold perturbation, neuron-location jitter,
synapse weight perturb/replace, add synapse, remove synapse, split-edge
add-neuron, remove neuron, plus optional meta-mutation of mutation-rate genes.

No species registry / speciation in sim-core.

### Metabolism

Passive per-tick cost is
`config.passive_metabolism_cost_per_unit * base_metabolic_cost`, where
`base_metabolic_cost` (`sim-core/src/metabolism.rs`) is:

```
base = inter_neuron_count
     + sensory_neuron_count
     + Σ synapse_spatial_cost
     + vision_distance / 3
synapse_spatial_cost = 0.25 + dist²(pre, post) / 9.0
```

Synapse cost is spatial: tightly clustered wiring pays the
`0.25`-per-edge floor; long-range links scale quadratically with latent-space
distance, making sprawling circuits expensive. The cost is cached on
`OrganismState::base_metabolic_cost` and refreshed whenever synapses change.

### Food Ecology

Food regrowth is event-driven with per-cell due-turn bookkeeping
(`food_regrowth_due_turn` + `food_regrowth_schedule`). Delay is
`food_regrowth_interval +/- food_regrowth_jitter` (floor 1 turn). Due cells
occupied by organisms are deferred by 1 turn.

### Delta Model

`TickDelta` has unified `removed_positions: Vec<RemovedEntityPosition>` (each
with `entity_id: EntityId` tagged `Organism` or `Food`). No separate food
removal field. `facing_updates` and `food_spawned` are included. Clients
partition removals by entity type. Full `WorldSnapshot` occupancy cells include
`Wall`.

### Determinism

Fixed config + seed = identical results. All tie-breaking uses organism ID
ordering. Action sampling and predation sampling are deterministic hashes of
`(seed, turn, organism IDs)`. Any change to turn logic must preserve determinism.

### Evaluation Dataset Pipeline

`sim-evaluation` is dataset-centric: the simulation emits raw facts and all
derived metrics are re-computed on demand from the persisted dataset.

Per-seed output directory layout (`artifacts/evaluation/<timestamp>_seeds_*/seed_<n>/`):

- `manifest.json` — schema version, seed, total ticks, `report_every`,
  `world_config` snapshot, created-at timestamp.
- `tick_summary/`, `action_counts/`, `organism_lifetimes/`,
  `population_snapshots/`, `reproduction_events/`, `genome_snapshots/`,
  `genomes/` — partitioned Parquet tables (one row type each, see
  `sim-evaluation/src/dataset/schema.rs`).
- `summary.json`, `timeseries.csv`, `report.html` — derived artifacts, always
  reproducible from the raw tables.

Run root (`artifacts/evaluation/<timestamp>_seeds_*/`) additionally holds the
cross-seed aggregate `summary.json` / `timeseries.csv` / `report.html`.

The analysis layer (`sim-evaluation/src/analysis/`) derives three products
from the dataset:

- `IntervalMetrics` timeseries (one row per `report_every` window,
  `intervals.rs`): population, births/deaths, mean neuron/synapse counts,
  attack/failure rates, `p_fwd_food`, `MI(S;A)`, idle fraction, inter-neuron
  utilization, generation time (mean `parent_age_turns` over successful
  reproductions), action histogram.
- `PillarScores` (`pillars.rs`) over a configurable scoring window (default
  last 10%): three niche-agnostic competence axes — **Foraging**
  (`p_fwd_food`), **Intelligence** (action effectiveness, MI, anti-idle,
  utilization), **Competition** (attack success, attack attempts). There is
  deliberately no aggregate score; each pillar stands on its own.
- `DemographicAnalytics` (`demographics.rs`): successful / blocked /
  parent-died reproductions, mean parent energy after birth, mean age at
  first successful reproduction, mean successful-birth interval, survival
  thresholds (age 30, maturity).

Re-derive artifacts without re-running the simulation:
`cargo run -p sim-evaluation --release -- analyze <run>` where `<run>` is a
path (run root or seed dir), a timestamp prefix resolved under
`artifacts/evaluation/`, or the literal `latest`. Run-root targets re-analyze
every seed and rebuild the aggregate.
