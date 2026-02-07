# NeuroGenesis v1 Behavior Contract

## Scope
This contract defines canonical behavior for `sim-core`. All other components consume the behavior and do not redefine it.

## Default Config
- `columns=20`
- `rows=20`
- `steps_per_epoch=20`
- `steps_per_second=5`
- `num_organisms=500`
- `num_neurons=2`
- `max_num_neurons=20`
- `num_synapses=4`
- `vision_depth=3`
- `action_potential_length=1`
- `mutation_chance=0.04`
- `mutation_magnitude=1.0`
- `unfit_kill_probability=0.95`
- `offspring_fill_ratio=0.2`
- `survival_rule=CenterBandX(0.25, 0.75)`

## Tick Order
1. Reset all `ActionNeuron.is_active=false`.
2. Decay all neuron potentials toward resting potential.
3. Sum all potentials.
- Sensory neurons pull receptor values (`LookLeft/Right/Up/Down/X/Y`).
- Inter/action neurons add incoming current and clear incoming current.
4. Evaluate action potentials for sensory/inter/action neurons.
- If threshold reached and no action potential in progress, set timer to `0`.
- Fire when `action_potential_time >= action_potential_length`.
5. Resolve action outcomes from active action neurons.
6. Apply movement in action enum order (`Up`, `Down`, `Left`, `Right`) with bounds checks.

## Epoch Order
1. Scatter organisms until each fails survival rule.
2. Execute `steps_per_epoch` ticks.
3. Evaluate survivors by `survival_rule`.
4. Kill unfit with probability `unfit_kill_probability`.
5. Refill to `num_organisms`:
- Clone survivors for `offspring_fill_ratio * deficit`.
- Fill remaining deficit with new random organisms.
6. Increment epoch counter and reset `tick_in_epoch=0`.

## Mutation Model
Mutation applies per offspring with probability `mutation_chance`.
Number of mutation operations per offspring is `round(max(1, mutation_magnitude))`.

Topology mutations:
- Insert interneuron if `< max_num_neurons`.
- Delete random interneuron.
- Substitute random interneuron properties.

Synapse mutations:
- Add synapse (no duplicate and no autapse).
- Remove random synapse from a random output neuron.
- Perturb random synapse weight by value in `[-mutation_magnitude, mutation_magnitude]`.

## Invariants
- Population size after `process_survivors` equals `num_organisms`.
- Organism coordinates always in bounds.
- No synapse references removed neurons.
- Same config + seed => deterministic snapshots.

## Pseudocode
```text
tick(world):
  reset_action_flags()
  decay_all()
  sum_all()
  fire_all()
  resolve_actions()
  apply_moves()

run_epoch(world):
  scatter()
  repeat steps_per_epoch: tick()
  process_survivors()
  epoch += 1
  tick_in_epoch = 0
```
