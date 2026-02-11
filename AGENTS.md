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

## Testing

- Primary runner: `cargo test --workspace`.
- Add deterministic tests for simulation behavior changes (fixed seeds,
  snapshot assertions).

## Architecture Context

### World

Bounded axial hex grid `(q, r)`, `0 <= q,r < world_width`. Occupancy is a
dense `Vec<Option<Occupant>>` (`Organism(OrganismId)` or `Food(FoodId)`),
indexed by `r * world_width + q`. At most one entity per cell.

### Turn Pipeline (execution order)

1. **Lifecycle** — deduct `turn_energy_cost`, remove dead/old organisms.
2. **Snapshot** — freeze occupancy + organism state, stable ID ordering.
3. **Intent** — evaluate brains, produce per-organism intents.
4. **Move resolution** — simultaneous resolution; highest confidence wins,
   ties broken by lower ID. Supports consumption and cycles.
5. **Commit** — apply moves, kills, energy transfers. Replenish food.
6. **Reproduction** — queue spawn at hex behind (opposite facing).
7. **Age** — increment `age_turns`.
8. **Spawn** — process queue, mutate genome, assign species.
9. **Metrics & delta** — prune extinct species, emit `TickDelta`.

### Brain

4 sensory neurons (Look Food/Organism/OutOfBounds + Energy), inter neurons
at IDs `1000..1000+n`, action neurons at `2000..2004`. Evaluation:
sensory→inter, inter→inter (prev tick), then →action. Inter=tanh,
action=sigmoid, fires at `> 0.5`.

### Genome

`OrganismGenome` has `num_neurons`, `max_num_neurons`, `vision_distance`,
`mutation_rate`, `inter_biases`, `edges` (sorted). Mutations are each
independently gated by `mutation_rate`. Species assigned by L1 genome
distance; exceeding `speciation_threshold` creates a new species.

### Delta Model

`TickDelta` has unified `removed_positions: Vec<RemovedEntityPosition>`
(each with `entity_id: EntityId` tagged `Organism` or `Food`). No separate
food removal field. `food_spawned` for new food. Clients partition removals
by entity type.

### Determinism

Fixed config + seed = identical results. All tie-breaking uses organism ID
ordering. Any change to turn logic must preserve determinism.
