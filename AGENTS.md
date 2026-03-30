# Repository Guidelines

## Project Structure

- `sim-types/`: shared domain types used across all Rust crates.
- `sim-config/`: world + seed-genome config crate (`config.toml`, typed
  loader/validation, owned defaults/policies, generated config reference).
- `sim-core/`: deterministic simulation engine:
  - `lib.rs`: `Simulation` struct, config validation.
  - `turn.rs` + `turn/`: tick orchestration, lifecycle, snapshot, intents,
    movement, reproduction, and commit phases.
  - `brain.rs` + `brain/`: neural network evaluation, genome expression, and
    sensing helpers.
  - `plasticity.rs`: runtime eligibility/coactivation and Hebbian updates.
  - `genome.rs` + `genome/`: seed generation, mutation-rate accessors, scalar
    mutation, topology mutation, spatial prior, and sanitization.
  - `spawn.rs` + `spawn/`: organism spawning, world generation, and food
    ecology/regrowth.
  - `topology.rs`: shared neuron/synapse topology helpers and invariants.
  - `grid.rs`: hex-grid geometry and occupancy helpers.
- `sim-evaluation/`: headless evaluation harness split into CLI,
  orchestration, aggregation, comparison, output, and reporting modules.
- `sim-server/`: Axum HTTP + WebSocket server (`src/main.rs`), with server-only
  API types in `src/protocol.rs`.
- `web-client/`: React + TailwindCSS + Vite canvas UI (`src/`).
- `sim-config/config.toml`: baseline simulation configuration.
- `sim-config/CONFIG_REFERENCE.md`: generated config reference derived from
  `sim-config/src/config.rs`.
- `sim-evaluation/config.toml`: stable evaluation benchmark configuration.
  Keep this in sync with `sim-config/config.toml` when config-schema or baseline
  tuning changes affect evaluation assumptions.

## Build & Test

- `cargo check --workspace`: fast compile check.
- `cargo test --workspace`: run all Rust tests.
- `cargo run -p sim-config --bin generate_config_docs`: regenerate
  `sim-config/CONFIG_REFERENCE.md`.
- `make fmt`: format Rust code.
- `make lint`: clippy with warnings as errors.
- `cargo run -p sim-evaluation --release --`: run the default 10-seed
  evolution-loop evaluation harness and generate
  `report.html`/`timeseries.csv`/`summary.json`.
- `cargo run -p sim-evaluation --release -- --seed <u64>[,<u64>...]`: override
  the default seed suite.
- `make evaluate ARGS="--seed <u64>[,<u64>...]"`: same harness via Makefile
  wrapper.
- `cargo run -p sim-server`: start backend on `127.0.0.1:8080`.
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
- Add deterministic tests for simulation behavior changes (fixed seeds, snapshot
  assertions).
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
the hidden food-ecology policy owned by `sim-config` and documented in
`sim-config/CONFIG_REFERENCE.md`:
`FOOD_FERTILITY_NOISE_SCALE = 0.012`, `FOOD_FERTILITY_THRESHOLD = 0.6`.
Terrain-wall cells are infertile.

### Turn Pipeline (execution order)

1. **Lifecycle** — deduct passive metabolism
   `(food_energy / 10000) * (num_neurons + sensory_count + synapse_count + vision_distance / 3)`,
   remove starved/aged organisms.
2. **Snapshot** — freeze occupancy + organism state, stable ID ordering.
3. **Intent** — evaluate brains, sample one action per organism from
   temperature-scaled softmax logits (deterministic sample stream), derive
   facing/move/eat/attack/reproduce intent. Any non-`Idle` action pays one
   `move_action_energy_cost`. When runtime plasticity is enabled, compute
   per-edge pending coactivations (`pre * post`).
4. **Reproduction trigger** — `Reproduce` starts a 2-turn lock if mature and
   energy is sufficient; parent energy is debited immediately by
   `reproduction_investment_energy`.
5. **Move resolution** — simultaneous movement into empty targets only; highest
   confidence wins, ties broken by lower ID. Occupied cells and walls block.
6. **Commit** — apply facing + moves + action costs; resolve forward-cell
   interactions. `Eat` only consumes food and `Attack` only damages organisms.
   Plant food grants `20%` of stored energy, corpse food grants `80%`, lethal
   attacks spawn corpse food, and spike tiles damage organisms after movement.
   Process due food regrowth here.
7. **Age** — increment `age_turns`.
8. **Spawn** — complete reproduction locks into spawn requests (if cells are
   free), mutate genomes for offspring, and process periodic seed-genome
   injections. Post-commit runtime weight updates run after spawning.
9. **Metrics & delta** — update counters and emit `TickDelta`.

### Brain

Sensory receptors are:
`LookRay { ray_offset = 0, target in {Food, Organism, Wall, Spikes} }`
plus `ContactAhead`, `Damage`, and `Energy` (7 sensory neurons total). Ray
scans are cached per tick and stop on first blocking entity (including walls),
with signal
`(max_dist - dist + 1) / max_dist`.

Inter neurons are `1000..1000+n`. Action neurons are `2000..2005` and map to
the non-idle action set: `TurnLeft`, `TurnRight`, `Forward`, `Eat`, `Attack`,
`Reproduce`. `Idle` is handled separately by the policy logic. Policy is
single-choice categorical sampling per tick from softmax logits.

Runtime plasticity:
- Pending coactivation accumulation: `pending = pre * post` during intent eval.
- Reward signal: dopamine is derived from the reward ledger signal, scaled by
  `20.0`, passed through `tanh`, and multiplied by the optional reward-signal
  override used by the evaluation harness.
- Weight update: mature organisms use `hebb_eta_gain`; juveniles also learn
  before maturity, scaled by `juvenile_eta_scale`. Per-tick updates are
  clamped by `max_weight_delta_per_tick`.
- Eligibility fold: `e = retention * e + (1 - retention) * pending`.
- Synapse pruning is maturity-gated (every 10 ticks) using weight+eligibility
  thresholds.

### Genome

`OrganismGenome` has `num_neurons`, `num_synapses`, `spatial_prior_sigma`,
`vision_distance`, `starting_energy`, `age_of_maturity`,
`plasticity_start_age`, `hebb_eta_gain`, `juvenile_eta_scale`,
`eligibility_retention`, `max_weight_delta_per_tick`,
`synapse_prune_threshold`, `inter_biases`, `inter_log_time_constants`,
neuron-location vectors (`sensory_locations`, `inter_locations`,
`action_locations`), sorted `edges` (with runtime traces), and per-operator
mutation-rate genes.

Implemented mutation operators include: age/maturity and vision step mutation,
bias/time-constant/retention/prune-threshold perturbation, neuron-location
perturbation, synapse weight perturb/replace, add synapse, remove synapse,
split-edge add-neuron, plus optional meta-mutation of mutation-rate genes.

No species registry/speciation in sim-core.

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
