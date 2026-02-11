# NeuroGenesis Spec (Current Implementation)

This document reflects the implemented behavior across:

- `sim-core` (authoritative turn runner)
- `sim-protocol` (wire/data model)
- `sim-server` (runtime surface)
- `web-client` (delta/state application)

## 1. World Model

- The world is a bounded axial hex grid using `(q, r)` coordinates.
- Bounds: `0 <= q < world_width`, `0 <= r < world_width`.
- Capacity: `world_width * world_width`.
- Occupancy is a dense O(1) table: `Vec<Option<OrganismId>>`, indexed by
  `r * world_width + q`.
- Invariant after every turn: at most one organism per cell and occupancy table
  matches organism positions.

## 2. Config (`WorldConfig` + `SpeciesConfig`)

World-level fields:

- `world_width`
- `steps_per_second`
- `num_organisms`
- `center_spawn_min_fraction`
- `center_spawn_max_fraction`
- `starting_energy`
- `reproduction_energy_cost`
- `move_action_energy_cost`
- `seed_species_config`

Species-level fields:

- `num_neurons`
- `max_num_neurons`
- `num_synapses`
- `mutation_chance`

Runtime species model:

- Organisms reference species by stable `species_id` (`u32`).
- `Simulation` keeps a species registry (`species_id -> SpeciesConfig`).
- Registry is exposed on every world snapshot as `species_registry`.
- Current limitation: world initialization seeds exactly one species at
  `species_id = 0` from `seed_species_config`.

## 3. Turn Runner Pipeline

Each turn is executed centrally in phases.

### 3.1 Snapshot Phase

Build immutable start-of-turn state:

- occupancy snapshot
- stable organism ordering by `OrganismId`
- organism pose/facing snapshot
- per-organism move confidence from hidden-state signal

Move confidence is derived from the `MoveForward` action activation at turn
start (fallback `0.0` when missing).

### 3.2 Intent Phase

For each living organism in stable ID order:

- evaluate brain against the same frozen snapshot occupancy
- compute `facing_after_turn` from turn actions
- compute `wants_move`
- compute in-bounds `move_target` if `wants_move`
- compute `wants_reproduce`
- attach deterministic confidence from snapshot phase

No world mutation happens here.

### 3.3 Move Resolution Phase (Global)

All movement intents are resolved simultaneously.

Rules:

- contenders for the same target are resolved by:
  - highest confidence first
  - deterministic tie-break by lower `OrganismId`
- one winner per target cell
- if target is empty: winner moves in
- if target occupant is also moving this turn: winner moves in (vacated-target)
- if target occupant is not moving: winner performs consume-and-replace

Cycles (swap and longer cycles) are naturally supported because winners are
resolved globally and moves commit atomically.

### 3.4 Commit Phase (Atomic)

In one commit pass:

- apply facing updates for all organisms
- apply resolved movement results
- apply successful movement energy cost (`move_action_energy_cost`)
- apply consumption kills
- transfer consumed organism energy to the consumer
- increment consumer consumptions (`consumptions_count += 1`)

Move cost is charged only for successful committed moves.

### 3.5 Reproduction Action Phase

Still inside the same runner turn:

- organism must have `Reproduce` action active
- organism must have enough energy (`>= reproduction_energy_cost`)
- spawn cell must be the hex opposite current facing and must be unblocked
- same-cell reproduction conflicts are resolved deterministically by organism ID
  order (first reservation wins)
- on success:
  - reproduction spawn request is queued
  - `reproduction_energy_cost` is deducted immediately
  - organism `reproductions_count` increments
  - world metric `reproductions_last_turn` increments
- on failure (insufficient energy / blocked cell / out-of-bounds): no side
  effects

### 3.6 Lifecycle Phase (Energy Decay + Starvation)

Still inside the same runner turn:

- each surviving organism loses `1.0` energy
- organisms with `energy <= 0.0` die of starvation
- starvation only removes organisms from the world (no replacement spawn)

### 3.7 Spawn Resolution Phase

Spawn queue is processed deterministically in enqueue order:

- reproduction requests only
- spawn is skipped if placement is invalid or no empty cells remain

Spawn kinds:

- reproduction spawn: parent-derived offspring with opposite-facing child,
  mutation applied, inheriting parent species DNA

All spawned organisms start with `starting_energy`.

### 3.8 Metrics + Delta Phase

After turn commit/reproduction/lifecycle/spawn:

- `turn += 1`
- metrics are finalized:
  - `synapse_ops_last_turn`
  - `actions_applied_last_turn` (successful moves + successful reproductions)
  - `consumptions_last_turn`
  - `reproductions_last_turn`
  - `starvations_last_turn`
- turn delta is emitted from committed results

## 4. Brain Evaluation Model

- Sensory receptors: currently `Look` only.
- `Look` = `1.0` when faced neighbor is occupied by another organism, otherwise
  `0.0`; out-of-bounds returns `0.0`.
- Inter activations use tanh updates.
- Action activations use sigmoid updates.
- `NeuronState.is_active` is server-authored.
- Sensory/inter neuron active when activation `> 0.0`.
- Action neuron active when activation `> 0.5`.
- `synapse_ops_last_turn` counts traversed edge contributions during intent
  evaluation.

Action set:

- `MoveForward`
- `TurnLeft`
- `TurnRight`
- `Reproduce`

## 5. Movement/Facing Semantics

Facing directions:

- `East`, `NorthEast`, `NorthWest`, `West`, `SouthWest`, `SouthEast`

Neighbor offsets:

- `East -> (q+1, r)`
- `NorthEast -> (q+1, r-1)`
- `NorthWest -> (q, r-1)`
- `West -> (q-1, r)`
- `SouthWest -> (q-1, r+1)`
- `SouthEast -> (q, r+1)`

Turn behavior:

- exactly one of `TurnLeft`/`TurnRight` active => rotate one step
- both active or both inactive => no rotation

`MoveForward` uses `facing_after_turn` when producing move target.

## 6. Mutation

Applied to reproduction offspring only:

- skipped when RNG exceeds `mutation_chance`
- one or more small-step trait mutations are applied:
  - `num_neurons` mutates by `+1` or `-1` within bounds
  - `max_num_neurons` mutates by `+1` or `-1` within bounds
  - `num_synapses` mutates by `+1` or `-1` within bounds
  - `mutation_chance` mutates by a small bounded delta
- at least one trait mutation is always applied per successful speciation event
- additional trait mutations may be added probabilistically from
  `mutation_chance`, with a fixed internal cap for runtime stability

Weights remain clamped to `[-8.0, 8.0]`.

## 7. Protocol/Delta Model

`TickDelta` contains:

- `turn`
- `moves` (`OrganismMove[]`)
- `removed_positions` (`{ id, q, r }[]`)
- `spawned` (`OrganismState[]`)
- `metrics`

This allows clients to apply move/consume/starve/spawn effects incrementally
without waiting for full snapshots.

`WorldSnapshot` also contains:

- `species_registry`: runtime species DNA map (`species_id -> SpeciesConfig`)
- organism `species_id` in every `OrganismState`

## 8. Runtime Surfaces

### Server

- REST: create/get state/step/reset/focus/stream
- WS commands:
  - `Start { ticks_per_second }`
  - `Pause`
  - `Step { count }`
  - `SetFocus { organism_id }`

### Web Client

- Applies tick deltas by:
  - removing IDs from `removed_positions`
  - applying committed `moves`
  - appending `spawned`
  - sorting organisms by ID
- Still accepts periodic full snapshots as source-of-truth sync.

## 9. Determinism

For fixed config/seed and command sequence:

- snapshot/intent/move/spawn ordering is deterministic
- move conflict tie-breaks are deterministic (`OrganismId`)
- spawn queue ordering is deterministic
- reproduction-cell reservation ordering is deterministic
- RNG usage order is deterministic

Covered by deterministic seed tests, targeted complex-turn determinism tests,
and golden snapshot regression.
