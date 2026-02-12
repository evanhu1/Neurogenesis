# Neurogenesis Spec

This file is the authoritative implementation-aligned specification for the
simulation engine, including turn-runner behavior.

## World

- Grid: bounded axial hex coordinates `(q, r)` with `0 <= q,r < world_width`.
- Occupancy: dense `Vec<Option<Occupant>>` indexed by `r * world_width + q`.
- One entity per cell (`Organism(OrganismId)` or `Food(FoodId)`).

## Turn Runner

Each tick executes in strict order:

1. Lifecycle: subtract `turn_energy_cost`, remove starved/aged organisms.
2. Snapshot: freeze occupancy + organisms for intent evaluation.
3. Intent: evaluate each organism brain.
4. Move resolution: simultaneous resolution, winner is highest confidence,
   tie-break by lower `OrganismId`.
5. Commit: apply moves/facing/consumption/energy transfers and food replenish.
6. Reproduction queue: eligible organisms queue spawn at hex behind them.
7. Age: increment `age_turns`.
8. Spawn: process queue in deterministic order, mutate offspring genomes, assign
   species.
9. Metrics & delta: prune extinct species and emit `TickDelta`.

Determinism: fixed config + RNG seed + command sequence produces identical
results.

## Brain Model

- Neuron IDs:
  - Sensory: `0..4` (`Look(Food)`, `Look(Organism)`, `Look(OutOfBounds)`,
    `Energy`)
  - Inter: `1000..1000+n`
  - Action: `2000..2003` (`MoveForward`, `Turn`, `Reproduce`)
- Interneurons include `interneuron_type`: `Excitatory` or `Inhibitory`.
- Evaluation:
  - sensory -> inter
  - inter(t-1) -> inter
  - sensory/inter -> action
- Inter update: `h_i(t) = (1-alpha_i) * h_i(t-1) + alpha_i * tanh(input_i(t))`,
  `alpha_i in [0.1, 1.0]`.
- Action outputs:
  - `Turn`: `tanh`
  - Others: `sigmoid`, fire at `> 0.5`

Runtime invariant assertions in genome expression (`debug_assert!`):

- Sensory outgoing synapse weights are strictly positive.
- Interneuron outgoing synapse weights match `interneuron_type` sign.

## Genome Model

`OrganismGenome` fields:

- Scalar traits: `num_neurons`, `vision_distance`
- Gene vectors:
  - `inter_biases`
  - `inter_update_rates`
  - `interneuron_types`
  - `action_biases`
- Graph genes: sorted `edges` (`SynapseEdge`)
- Per-operator mutation-rate genes:
  - `mutation_rate_vision_distance`
  - `mutation_rate_weight`
  - `mutation_rate_add_edge`
  - `mutation_rate_remove_edge`
  - `mutation_rate_split_edge`
  - `mutation_rate_inter_bias`
  - `mutation_rate_inter_update_rate`
  - `mutation_rate_action_bias`

## Synapse Sign Rules

- Sensory neurons are always excitatory (`+` outgoing weights).
- Interneuron outgoing sign is fixed by `interneuron_type`:
  - `Excitatory` => positive weights
  - `Inhibitory` => negative weights
- Dale's law is enforced in generation and mutation paths.

## Initialization

- New seed genomes:
  - Interneuron types sampled with prior: `80%` excitatory, `20%` inhibitory.
  - Synapse magnitudes sampled from log-normal distribution.
  - Synapse sign assigned by source neuron type.
- New neurons added by split-edge mutation use the same `80/20` type prior.

## Mutation

Mutation is applied to offspring genomes only.

### Self-adaptive mutation rates

- Mutation-rate genes mutate every mutation step using:
  `tau = 1 / sqrt(2 * sqrt(n))` where `n = number of mutation-rate genes`.
- Rate update form: `rate <- clamp(rate * exp(tau * N(0,1)), 0, 1)`.

### Operator gating

Each operator is independently gated by its own mutation-rate gene:

- `vision_distance` step mutation (`[1, 32]`)
- edge weight Gaussian perturbation (then sign projection)
- add edge
- remove edge
- split edge
- inter bias Gaussian perturbation
- inter update-rate Gaussian perturbation
- action bias Gaussian perturbation

### Structural mutations

- Old add-neuron operator removed.
- Old remove-neuron operator removed.
- Split-edge operator:
  - Uniformly sample an existing edge.
  - Remove sampled edge.
  - Create new interneuron with next available inter ID.
  - Insert `(old_pre -> new_inter)` and `(new_inter -> old_post)`.
  - Preserve edge uniqueness and sorted-edge invariant.

Continuous perturbations use Gaussian noise; synapse creation uses log-normal
magnitudes.

`max_num_neurons` is a fixed world-level cap (`WorldConfig.max_num_neurons`) and
is not a mutable genome gene.

## Config Validation

`WorldConfig` validates:

- `max_num_neurons in [1, 256]`.
- `max_num_neurons >= seed_genome_config.num_neurons`.

`SeedGenomeConfig` validates:

- Every mutation-rate field in `[0, 1]`.
- `vision_distance in [1, 32]`.

## Genome Distance

`genome_distance` is L1-style distance over:

- scalar traits,
- all mutation-rate genes,
- interneuron-type mismatches,
- bias/update/action vectors,
- sorted edge merge-join with edge-weight deltas.
