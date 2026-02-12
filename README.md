# NeuroGenesis

Neuromorphic brains grown from scratch with simulated evolution in Rust.

## Layout

- `sim-types/` — shared domain types used across all Rust crates.
- `sim-core/` — deterministic simulation engine:
  - `lib.rs` — `Simulation` struct, config validation.
  - `turn.rs` — turn pipeline, move/consume/starvation resolution.
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

Bounded axial hex grid, `(q, r)` coordinates, `0 <= q,r < world_width`. Capacity
is `world_width * world_width`. Occupancy is a dense `Vec<Option<Occupant>>`
indexed by `r * world_width + q`, where `Occupant` is `Organism(OrganismId)` or
`Food(FoodId)`. At most one entity per cell.

Six hex directions: `East (q+1,r)`, `NorthEast (q+1,r-1)`, `NorthWest (q,r-1)`,
`West (q-1,r)`, `SouthWest (q-1,r+1)`, `SouthEast (q,r+1)`. `Turn` sign rotates
one step (`<0` = left, `>0` = right), with a deadzone around `0` meaning no
rotation. `MoveForward` uses post-turn facing.

## Config

`WorldConfig` fields: `world_width`, `steps_per_second`, `num_organisms`,
`center_spawn_min_fraction`, `center_spawn_max_fraction`, `starting_energy`,
`food_energy`, `reproduction_energy_cost`, `move_action_energy_cost`,
`turn_energy_cost`, `food_coverage_divisor`, `max_organism_age`,
`speciation_threshold`, `seed_genome_config`.

`SeedGenomeConfig` fields: `num_neurons`, `max_num_neurons`, `num_synapses`,
`mutation_rate`, `vision_distance`.

Defaults are in `config/default.toml`.

## Turn Pipeline

Phases execute in this order each tick:

1. **Lifecycle** — deduct `turn_energy_cost` from all organisms. Remove any with
   `energy <= 0` or `age_turns >= max_organism_age`. Rebuild occupancy.
2. **Snapshot** — freeze occupancy and organism state. Stable ordering by ID.
3. **Intent** — evaluate each brain against the frozen snapshot. Produce per-
   organism intent: `facing_after_turn`, `wants_move`, `move_target`,
   `wants_reproduce`, `move_confidence`.
4. **Move resolution** — resolve all move intents simultaneously. Contenders for
   the same cell: highest confidence wins, ties broken by lower ID. Empty target
   = move in. Stationary occupant = consume and replace. Food target = consume.
   Vacated target (occupant also moving) = move in. Cycles resolve naturally.
5. **Commit** — atomically apply facing, moves, energy costs, consumption kills,
   energy transfers. Rebuild occupancy. Replenish food to
   `capacity / food_coverage_divisor`.
6. **Reproduction** — organisms with `Reproduce` active and sufficient energy
   queue a spawn request at the hex behind them (opposite facing). Cell must be
   in-bounds and unoccupied. Conflicts resolved by ID order. Energy deducted on
   success.
7. **Age** — increment `age_turns` for all survivors.
8. **Spawn** — process spawn queue in order. Offspring get mutated genome,
   opposite facing, `starting_energy`. Species assigned by genome distance.
9. **Metrics & delta** — prune extinct species, increment turn counter, emit
   `TickDelta`.

Fixed config + seed + command sequence produces identical results. Organism ID
ordering is the universal tie-breaker.

## Brain

Four sensory neurons:

- `Look(Food)`, `Look(Organism)`, `Look(OutOfBounds)` — scan forward up to
  `vision_distance` hexes. Signal = `(max_dist - dist + 1) / max_dist` for the
  closest matching entity (with occlusion), or `0.0` if none found.
- `Energy` — `(energy / 100).clamp(0, 1)`.

Neuron IDs: sensory `0..4`, inter `1000..1000+n`, action `2000..2003`.

Evaluation order: sensory→inter, inter→inter (previous tick activations), then
sensory→action and inter→action. Inter uses per-neuron leaky integration:
`h_i(t) = (1 - alpha_i) * h_i(t-1) + alpha_i * tanh(z_i(t))`, where
`z_i(t) = b_i + sensory_to_inter_i + inter_to_inter_i(h(t-1))`.
Actions use `sigmoid` except `Turn`, which uses `tanh`. Discrete action firing
threshold remains `> 0.5` for non-turn actions.

Actions: `MoveForward`, `Turn`, `Reproduce`.

## Genome & Mutation

`OrganismGenome`: `num_neurons`, `max_num_neurons`, `vision_distance`,
`mutation_rate`, `inter_biases` (vec), `inter_update_rates` (vec),
`action_biases` (vec), `edges` (sorted `SynapseEdge` list).

Mutation applies to offspring only, each type independently gated by
`mutation_rate`:

- `num_neurons` +/-1 `[0, max_num_neurons]` — add inits bias + 1-2 edges, remove
  prunes incident edges.
- `max_num_neurons` +/-1 `[num_neurons, 256]`.
- `vision_distance` +/-1 `[1, 32]`.
- `mutation_rate` +/-0.01 `[0.0, 1.0]`.
- Weight perturbation (gated at `2 * rate`): Gaussian (stddev 0.3), clamped
  `[-3.0, 3.0]`.
- Add edge: random non-duplicate (20 retries).
- Remove edge: random swap-remove.
- Bias perturbation: Gaussian (stddev 0.3), clamped `[-1.0, 1.0]`.
- Inter update-rate perturbation: Gaussian (stddev 0.05), clamped `[0.03, 1.0]`.
- Action bias perturbation: Gaussian (stddev 0.3), clamped `[-1.0, 1.0]`.

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
`Food`), `spawned`, `food_spawned`, `metrics`. Clients apply deltas
incrementally; periodic full `WorldSnapshot`s serve as sync points.

## Performance

`cargo bench -p sim-core --bench turn_throughput`

Performance regression guard tests (ignored by default):

`cargo test -p sim-core --release performance_regression -- --ignored --nocapture`

Optional CI budget override:

`SIM_CORE_TICK_BUDGET_NS_PER_TURN=130000 make perf-test`

## To Do

- [ ] Cull useless neurons and synapses, neuronal pruning (also simulating
      exuberant synaptogenesis)
- [ ] Use Hebbian/SDTP to guide synaptogenesis (neurons that fire together wire
      together, neurons that fire out of sync lose their link). Synaptogenesis
      should not be random
- [ ] Implement temporal credit assignment
- [ ] Experiment with local gradient descent in the brain.
- [ ] Create multiple concentric evolution loops. Innermost is the organism
      loop. Evolve worlds, with a world DNA substrate that sets the "laws of
      physics" and is mutated. The fitness of a world is the max fitness
      achieved by life in that world.
