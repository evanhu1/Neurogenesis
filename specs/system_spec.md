# NeuroGenesis Spec (Current Implementation)

This document matches the currently implemented behavior in:

- `sim-core` (authoritative simulation behavior)
- `sim-protocol` (serialized data model)
- `sim-server` and `sim-cli` (execution/runtime surfaces)
- `web-client` (UI types and controls aligned to protocol)

## 1. World Model

- Geometry uses a hex grid with axial coordinates `(q, r)`.
- World bounds are `0 <= q < world_width` and `0 <= r < world_width`.
- Total addressable cells: `world_width * world_width`.
- Occupancy is stored in `sim-core` as a dense vector: `Vec<Option<OrganismId>>`
  indexed by `r * world_width + q`.
- This provides O(1) read/write occupancy checks and updates.

## 2. Config (`WorldConfig`)

Fields:

- `world_width` (default `20`)
- `steps_per_second` (default `5`)
- `num_organisms` (default `200`)
- `num_neurons` (default `2`)
- `max_num_neurons` (default `20`)
- `num_synapses` (default `4`)
- `turns_to_starve` (default `20`)
- `mutation_chance` (default `0.04`)
- `mutation_magnitude` (default `1.0`)
- `center_spawn_min_fraction` (default `0.25`)
- `center_spawn_max_fraction` (default `0.75`)

Validation:

- `world_width > 0`
- `num_organisms > 0`
- `turns_to_starve >= 1`
- `mutation_chance in [0,1]`
- center spawn fractions in `[0,1]`
- `center_spawn_min_fraction < center_spawn_max_fraction`

## 3. Organism State

Each organism stores:

- `id`
- `q`, `r`
- `facing` (one of 6 directions)
- `turns_since_last_meal`
- `meals_eaten`
- `brain`

Facing directions:

- `East`, `NorthEast`, `NorthWest`, `West`, `SouthWest`, `SouthEast`

Neighbor mapping used by movement/look:

- `East -> (q+1, r)`
- `NorthEast -> (q+1, r-1)`
- `NorthWest -> (q, r-1)`
- `West -> (q-1, r)`
- `SouthWest -> (q-1, r+1)`
- `SouthEast -> (q, r+1)`

## 4. Brain Model (Simplified)

### 4.1 Neurons

`NeuronState` fields:

- `neuron_id`
- `neuron_type` (`Sensory` / `Inter` / `Action`)
- `bias`
- `activation`
- `parent_ids`

### 4.2 Neuron sets and IDs

- Sensory neurons are fixed by `SensoryReceptorType::ALL`:
  - `Look` id `0`
- Inter neurons start at `1000 + i`
- Action neurons are fixed by `ActionType::ALL`:
  - `MoveForward` id `2000`
  - `TurnLeft` id `2001`
  - `TurnRight` id `2002`

### 4.3 Synapses

- Outgoing synapses exist on sensory and inter neurons only.
- Postsynaptic targets can be inter or action neurons.
- Autapses and duplicate `(pre, post)` edges are rejected.
- Weights are clamped to `[-8.0, 8.0]`.
- Post neuron `parent_ids` are maintained sorted/unique.

### 4.4 Inference per turn

Given organism position/facing and occupancy:

1. Reset all action `is_active = false`.
2. Compute sensory activation:
   - `Look` returns `1.0` if the directly faced hex is occupied by another
     organism, else `0.0`.
   - Out-of-bounds look returns `0.0`.
3. Compute inter activations:
   - `inter_input = bias + (sensory->inter contributions) + (previous inter activations -> inter contributions)`
   - `inter.activation = tanh(inter_input)`
4. Compute action activations:
   - `action_input = bias + (sensory->action contributions) + (current inter activations -> action contributions)`
   - `action.activation = tanh(action_input)`
   - `action.is_active = action.activation > 0.0`

`synapse_ops_last_turn` counts each traversed edge contribution during this
evaluation.

## 5. Actions

Actions are evaluated every turn for each organism:

- Turning:
  - If exactly one of `TurnLeft`/`TurnRight` is active, rotate one side.
  - If both are active or both inactive, no rotation.
- `MoveForward` then attempts movement to the neighbor in current facing.

Move outcomes:

- Out of bounds: blocked.
- Empty destination: organism moves.
- Occupied by another organism: eater/prey interaction (see evolution rules).

## 6. Evolution and Survival Rules (Turn-Based)

Per organism turn:

1. Increment `turns_since_last_meal`.
2. If `turns_since_last_meal >= turns_to_starve`:
   - organism dies immediately,
   - a replacement random organism is spawned in center region (fallback: any
     empty cell),
   - brain is newly generated random brain,
   - starvation/birth metrics updated.
3. Otherwise evaluate brain/actions.

Eating rule:

- If `MoveForward` targets an occupied neighbor:
  - prey is removed,
  - predator moves into prey cell,
  - predator `turns_since_last_meal = 0`, `meals_eaten += 1`,
  - predator reproduces: one offspring cloned from predator brain with mutation,
  - offspring spawn uses center region random empty cell (fallback: any empty
    cell),
  - meal/birth metrics updated.

## 7. Spawn Region

Center region uses half-open bounds derived from fractions:

- `min = floor(world_width * center_spawn_min_fraction)`
- `max = floor(world_width * center_spawn_max_fraction)`
- allowed cells satisfy `min <= q < max` and `min <= r < max`

If no empty center cell exists, spawn falls back to any empty world cell. If no
empty cell exists, spawn is skipped.

## 8. Mutation

Mutation runs only on offspring clones (not on random replacements) and only if
random check passes:

- If `rng > mutation_chance`: no mutation.
- Else perform `round(max(1.0, mutation_magnitude))` mutation operations.

Each operation:

- 50% topology mutation:
  - add inter neuron (if below `max_num_neurons`) and optionally a random
    synapse,
  - remove random inter neuron,
  - mutate random inter/action bias by `[-1,1]` (clamped to `[-8,8]`).
- 50% synapse mutation:
  - add random valid synapse,
  - remove random outgoing synapse,
  - perturb random outgoing synapse weight by `[-magnitude, magnitude]`, clamped
    to `[-8,8]`.

## 9. Metrics and Snapshot Fields

`WorldSnapshot`:

- `turn`
- `rng_seed`
- `config`
- `organisms`
- `occupancy` (`q`,`r`,`organism_ids`)
- `metrics`

`MetricsSnapshot`:

- `turns`
- `organisms`
- `synapse_ops_last_turn`
- `actions_applied_last_turn`
- `meals_last_turn`
- `starvations_last_turn`
- `births_last_turn`

## 10. Runtime Surfaces

### Server

- REST endpoints: create/get state/step/reset/focus/stream.
- Removed epoch/scatter/survivor-process APIs.
- WS commands:
  - `Start { ticks_per_second }`
  - `Pause`
  - `Step { count }`
  - `SetFocus { organism_id }`

### CLI

- `run`, `step`, `benchmark`, `export`, `replay` are turn-based.
- `--epochs` has been replaced by `--turns`.

## 11. Determinism

- RNG is `ChaCha8Rng` seeded by `seed`.
- Same config + seed + command sequence yields identical snapshots.
- Covered by deterministic and golden tests in `sim-core`.
