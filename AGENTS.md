# Repository Guidelines

## Project Structure

- `sim-types/`: shared domain types used across all Rust crates.
- `sim-core/`: deterministic simulation engine:
  - `lib.rs`: `Simulation` struct, config validation.
  - `turn.rs`: turn pipeline, move/consume/starvation resolution.
  - `brain.rs`: neural network evaluation and genome expression.
  - `genome.rs`: genome generation, mutation, species assignment.
  - `spawn.rs`: initial population spawn and reproduction placement.
  - `grid.rs`: hex-grid geometry and occupancy helpers.
- `sim-server/`: Axum HTTP + WebSocket server (`src/main.rs`), with server-only
  API types in `src/protocol.rs`.
- `web-client/`: React + TailwindCSS + Vite canvas UI (`src/`).
- `config/default.toml`: baseline simulation configuration.

## Build & Test

- `cargo check --workspace`: fast compile check.
- `cargo test --workspace`: run all Rust tests.
- `make fmt`: format Rust code.
- `make lint`: clippy with warnings as errors.
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
- Keep field names aligned between Rust structs and `web-client/src/types.ts`.
- DO NOT CARE ABOUT backwards compatibility

## Testing

- Primary runner: `cargo test --workspace`.
- Add deterministic tests for simulation behavior changes (fixed seeds, snapshot
  assertions).

## Architecture Context

### World

Toroidal axial hex grid `(q, r)` with wraparound modulo `world_width` on both
axes. Occupancy is a dense `Vec<Option<Occupant>>`
(`Organism(OrganismId)`, `Food(FoodId)`, or `Wall`), indexed by
`r * world_width + q`. At most one entity per cell. Terrain walls are generated
from Perlin noise at world init (`terrain_noise_scale`, `terrain_threshold`) and
block spawn, movement, and ray vision.

### Turn Pipeline (execution order)

1. **Lifecycle** — deduct
   `neuron_metabolism_cost * (enabled interneuron count)`, remove dead/old
   organisms.
2. **Snapshot** — freeze occupancy + organism state, stable ID ordering.
3. **Intent** — evaluate brains, select one categorical action per organism
   (argmax over action neurons), then derive facing/move/reproduce intent from
   that single action. Apply runtime plasticity (eligibility traces +
   maturity-gated 3-factor updates). Any non-`Idle` action pays one
   `move_action_energy_cost`.
4. **Reproduction** — queue spawn at hex behind (opposite facing). Requires
   `age_turns >= age_of_maturity` and sufficient energy.
5. **Move resolution** — simultaneous resolution; highest confidence wins, ties
   broken by lower ID. Moving into food consumes it. Moving into an occupied
   organism cell applies passive bite (`drain = min(prey_energy, food_energy *
   2.0)`) instead of displacement. Walls block movement.
6. **Commit** — apply moves, kills, energy transfers. Replenish food.
7. **Age** — increment `age_turns`.
8. **Spawn** — process queue, mutate genome, assign species.
9. **Metrics & delta** — prune extinct species, emit `TickDelta`.

### Brain

Sensory receptors are multi-ray:
`LookRay { ray_offset in [-2,-1,0,1,2], target in {Food, Organism} }` plus
`Energy` (11 sensory neurons total). Ray scans are cached per tick and stop on
first blocking entity (including walls), with signal
`(max_dist - dist + 1) / max_dist`.

Inter neurons are `1000..1000+n`. Action neurons are `2000..2008` mapped to:
`Idle`, `TurnLeft`, `TurnRight`, `Forward`, `TurnLeftForward`,
`TurnRightForward`, `Consume`, `Reproduce`. Policy is single-choice categorical
per tick (highest activation).

Runtime plasticity:
- Eligibility: `e = eligibility_retention * e + pre * post`.
- Reward signal:
  `dopamine = tanh((energy - energy_prev + passive_metabolism_baseline) / 10.0)`.
- Mature-only update: `w += eta * dopamine * e - 0.001 * w`,
  where `eta = max(0, hebb_eta_gain)`.
- Sign/clamp constraints preserve excitatory/inhibitory polarity.
- Synapse pruning is maturity-gated.

### Genome

`OrganismGenome` has `num_neurons`, `vision_distance`, `age_of_maturity`,
`hebb_eta_gain`, `eligibility_retention`,
`synapse_prune_threshold`, `inter_biases`, `inter_log_time_constants`,
`interneuron_types`, `action_biases`, `edges` (sorted, with per-edge
`eligibility` trace), and per-operator mutation-rate genes. No weight mutation;
weights are set at birth and modified by Hebbian learning. Species assigned by
L1 genome distance; exceeding `speciation_threshold` creates a new species.

### Delta Model

`TickDelta` has unified `removed_positions: Vec<RemovedEntityPosition>` (each
with `entity_id: EntityId` tagged `Organism` or `Food`). No separate food
removal field. `food_spawned` for new food. Clients partition removals by entity
type. Full `WorldSnapshot` occupancy cells now include `Wall`.

### Determinism

Fixed config + seed = identical results. All tie-breaking uses organism ID
ordering. Any change to turn logic must preserve determinism.
