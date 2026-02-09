# Turn Runner Spec

## Goal

Define a centralized, deterministic turn engine that resolves movement, eating,
starvation, and spawning globally for the whole population each turn.

## Why

Immediate per-organism mutation creates order bias and race-condition artifacts
(e.g., moving into a cell whose occupant may move out later in the same turn). A
global resolver removes these artifacts and opens cleaner optimization paths.

## Turn Pipeline

### 1. Snapshot Phase

Build an immutable start-of-turn view:

- occupancy map
- organism pose (`q`, `r`, `facing`)
- hunger/lifecycle state
- stable `OrganismId` ordering

No world mutation occurs in this phase.

### 2. Intent Phase

For every living organism (in stable ID order):

- Evaluate brain from the same frozen snapshot.
- Produce an intent:
  - `facing_after_turn`
  - `wants_move`
  - `move_target` (if in bounds)

No world mutation occurs in this phase.

### 3. Move Resolution Phase

Resolve all movement intents simultaneously with deterministic tie-breaking: the
move with the highest neuron hidden state weight wins (most "confidence").
you'll have to track this information in the snapshot phase.

Rules:

- Empty target:
  - one winner may enter.
- Occupied target where occupant successfully moves out this turn:
  - winner may enter (no eat event).
- Occupied target where occupant remains:
  - one winner may perform eat-and-replace (attacker enters occupant cell,
    occupant dies).
- Multiple contenders for one target:
  - deterministic single winner, others fail.
- Cycles (A->B, B->A or longer):
  - allowed when all involved moves are compatible under the same global
    resolution rules.

### 4. Commit Phase (Movement + Eating)

Apply all resolved changes atomically:

- update facing for all organisms
- apply moves
- apply kills from successful eats
- update eater stats (`turns_since_last_meal = 0`, `meals_eaten += 1`)

Collect spawn requests for each successful eat (offspring).

### 5. Lifecycle Phase (Starvation)

Run starvation as part of the same centralized turn runner:

- For non-eaters, increment `turns_since_last_meal`.
- Organisms reaching starvation threshold die this turn.
- For each starvation death, enqueue one replacement spawn request.

### 6. Spawn Resolution Phase

Resolve all spawn requests in deterministic order:

- if no empty cells remain, skip spawn (globally num organisms > hex tiles)
- spawn new organisms randomly in a gaussian distribution from the center of the
  map, only on empty hexes

Spawn types:

- Starvation replacement: random new organism.
- Reproduction spawn: offspring cloned from eater with mutation.

### 7. Metrics + Delta Phase

After commit is complete, compute turn outputs:

- `turn += 1`
- metrics (`actions_applied_last_turn`, `meals_last_turn`,
  `starvations_last_turn`, `births_last_turn`, synapse ops)
- movement delta list

## Determinism Requirements

- All conflict resolution and lifecycle ordering must be deterministic for fixed
  seed and inputs.
- Tie-breakers must be explicit and stable (`OrganismId` ordering).
- RNG usage for spawn placement/mutation must occur in fixed order.

## Invariants

After each turn:

- At most one organism per cell.
- Occupancy map and organism list are fully consistent.
- All organism positions are in bounds.
- Population remains bounded by world capacity.

## Optimization Opportunities (out of scope)

A centralized runner enables:

- batched intent computation
- reduced occupancy churn (single commit pass)
- eventual parallel brain evaluation from immutable snapshot
- clearer profiling boundaries by phase

## Test Cases to Enforce

- Move into cell vacated this turn succeeds.
- Two organisms swapping positions in one turn resolves deterministically.
- Multi-attacker same target resolves to one deterministic winner.
- Attacker vs escaping prey behavior is deterministic and documented.
- Starvation + replacement spawn occurs in the same turn pipeline.
- No-overlap occupancy invariant holds after mixed movement/eat/starve/spawn
  scenarios.
