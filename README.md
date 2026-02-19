# Neurogenesis

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

Toroidal axial hex grid, `(q, r)` coordinates with wraparound modulo
`world_width` on both axes. Capacity is `world_width * world_width`. Occupancy
is a dense `Vec<Option<Occupant>>` indexed by `r * world_width + q`, where
`Occupant` is `Organism(OrganismId)` or `Food(FoodId)`. At most one entity per
cell.

Six hex directions: `East (q+1,r)`, `NorthEast (q+1,r-1)`, `NorthWest (q,r-1)`,
`West (q-1,r)`, `SouthWest (q-1,r+1)`, `SouthEast (q,r+1)`. `Turn` sign rotates
one step (`<0` = left, `>0` = right), with a deadzone around `0` meaning no
rotation. `MoveForward` uses post-turn facing.

## Config

All hyperparameters are configured in `config/default.toml`. Parameters
configure either the World, or set default values for the first Species Genome.

## Turn Pipeline

Phases execute in this order each tick:

1. **Lifecycle** — deduct `neuron_metabolism_cost * (enabled interneuron count)` from all organisms. Remove any with
   `energy <= 0` or `age_turns >= max_organism_age`. Rebuild occupancy.
2. **Snapshot** — freeze occupancy and organism state. Stable ordering by ID.
3. **Intent** — evaluate each brain against the frozen snapshot. Produce per-
   organism intent: `facing_after_turn`, `wants_move`, `move_target`,
   `wants_reproduce`, `move_confidence`. Apply runtime plasticity (Hebbian
   learning with eligibility traces and synapse pruning) after evaluation.
4. **Reproduction** — organisms with `Reproduce` active,
   `age_turns >=
   age_of_maturity`, and sufficient energy queue a spawn
   request at the hex behind them (opposite facing). Cell must be unoccupied.
   Conflicts resolved by ID order. Energy deducted on success.
5. **Move resolution** — resolve all move intents simultaneously. Contenders for
   the same cell: highest confidence wins, ties broken by lower ID. Empty target
   = move in. Stationary occupant = consume and replace. Food target = consume.
   Vacated target (occupant also moving) = move in. Cycles resolve naturally.
6. **Commit** — atomically apply facing, moves, energy costs, consumption kills,
   energy transfers. Rebuild occupancy. Process due food regrowth events.
7. **Age** — increment `age_turns` for all survivors.
8. **Spawn** — process spawn queue in order. Offspring get mutated genome,
   opposite facing, and genome `starting_energy`. Species assigned by genome distance.
9. **Metrics & delta** — prune extinct species, increment turn counter, emit
   `TickDelta`.

Fixed config + seed + command sequence produces identical results. Organism ID
ordering is the universal tie-breaker.

## Brain

Three sensory neurons:

- `Look(Food)`, `Look(Organism)` — scan forward up to `vision_distance` hexes on
  a toroidal (wraparound) hex grid. Signal =
  `(max_dist - dist + 1) /
  max_dist` for the closest matching entity (with
  occlusion), or `0.0` if none found.
- `Energy` — `ln(1 + energy) / ln(101)` with negative energy clamped to `0`.

Neuron IDs: sensory `0..3`, inter `1000..1000+n`, action `2000..2005`.

Evaluation order: sensory→inter, inter→inter (previous tick activations), then
sensory→action and inter→action. Inter uses per-neuron leaky integration:
`h_i(t) = (1 - alpha_i) * h_i(t-1) + alpha_i * tanh(z_i(t))`, where
`z_i(t) = b_i + sensory_to_inter_i + inter_to_inter_i(h(t-1))` and `alpha_i` is
derived from a log-tau parameterisation. Actions use `sigmoid` except `Turn`,
which uses `tanh`. Discrete action firing threshold remains `> 0.5` for non-turn
actions.

Actions: `MoveForward`, `Turn`, `Consume`, `Reproduce`, `Dopamine`.

Interneurons carry a polarity (`Excitatory` or `Inhibitory`). Sensory neurons
are always excitatory. Dale's law is enforced for generated/mutated synapses:
all outgoing weights from a neuron share the sign implied by source type. Neuron
type distribution is biased towards an 80:20 excitatory:inhibitory ratio to
match biological brains.

### Dopamine Neuromodulation

The `Dopamine` action neuron produces a signal in `[-1, 1]` that modulates
online Hebbian learning. The effective learning rate each tick is
`eta = hebb_eta_gain * dopamine_signal`.

### Hebbian Learning (Oja's Rule)

After brain evaluation each tick, runtime plasticity is applied:

1. **Eligibility trace update** — for every synapse:
   `e(t) = (1 - lambda) * e(t-1) + pre * post`, where `lambda` is the genome's
   `eligibility_decay_lambda` and `pre * post` represents Hebbian coactivation
   "fire-together".
2. **Weight update** — Oja's rule: `w += eta * post * (pre - post * w)` with
   Dale's-law sign preservation.
3. **Synapse pruning** — once `age_turns >= age_of_maturity`, every 10 ticks
   synapses with `|weight| < threshold` and `|eligibility| < 2 * threshold` are
   removed. `synapse_prune_threshold` is a mutable genome parameter.

## Genome & Mutation

`OrganismGenome`: `num_neurons`, `vision_distance`, `age_of_maturity`,
`hebb_eta_baseline`, `hebb_eta_gain`, `eligibility_decay_lambda`,
`synapse_prune_threshold`, `inter_biases` (vec), `inter_log_taus` (vec),
`interneuron_types` (vec), `action_biases` (vec), `edges` (sorted `SynapseEdge`
list with per-edge `eligibility` trace), and explicit per-operator mutation-rate
genes.

Mutation applies to offspring only. Each operator is gated by its own mutation
rate gene, and mutation-rate genes self-adapt every mutation step using
`tau = 1 / sqrt(2 * sqrt(n))`, where `n` is the number of mutation-rate genes.

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
- Inter log-tau perturbation: Gaussian additive (stddev `0.05`).
- Action bias perturbation: Gaussian additive (stddev `0.15`), clamped
  `[-1.0, 1.0]`.
- `eligibility_decay_lambda` perturbation: Gaussian additive (stddev `0.05`),
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
`Food`), `spawned`, `food_spawned`, `metrics`. Clients apply deltas
incrementally; periodic full `WorldSnapshot`s serve as sync points.

## Performance

`cargo bench -p sim-core --bench turn_throughput`

Performance regression guard tests (ignored by default):

`cargo test -p sim-core --release performance_regression -- --ignored --nocapture`

Optional CI budget override:

`SIM_CORE_TICK_BUDGET_NS_PER_TURN=130000 make perf-test`

## To Do

- [x] Cull useless neurons and synapses, neuronal pruning (synapse pruning at
      maturity based on weight + eligibility thresholds)
- [x] Use Hebbian/SDTP to guide synaptogenesis (Oja's rule with dopamine-
      modulated learning rate and eligibility traces)
- [x] Implement temporal credit assignment (eligibility traces with configurable
      decay lambda)
- [ ] Experiment with local gradient descent in the brain.
- [ ] Create multiple concentric evolution loops. Innermost is the organism
      loop. Evolve worlds, with a world DNA substrate that sets the "laws of
      physics" and is mutated. The fitness of a world is the max fitness
      achieved by life in that world.
