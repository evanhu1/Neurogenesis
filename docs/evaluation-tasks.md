# Task-library boundary

`task-library` contains deterministic symbolic environments. A task defines
only domain semantics:

- its serializable configuration and private state;
- legal actions and optional symbolic observations;
- deterministic instance construction from `(panel_seed, instance_index)`;
- rewards, atomic success events, task-relative correctness, trial boundaries,
  trial outcomes, and termination.

It must not import `brain` or `evolution`, inspect a genome, install a neural
representation, invoke a learning rule, assign reproductive tickets, select
parents, mutate offspring, or choose training/audit panel sizes.

## Runtime boundary

```text
task-library::SymbolicTask
  observation -> [evolution::TaskEcology -> brain] -> action
  transition  <- [evolution::TaskEcology]
                         |
                  success events
                         v
             finite reproductive tickets
                         |
             asexual tournament search
```

`evolution::TaskEcology<T>` is the only adapter. It owns genome expression,
sensory encoding, action sampling, learning, frozen probes, panel construction,
agent-state policy at semantic trial boundaries, controls, metrics, and the
conversion from task success events to reproductive tickets.

`evolution::run_resource_ecology` sees only the generic
`ResourceEcologyTask` contract. It owns equal-panel evaluation, finite
reproduction, exact elite retention, tournament parent selection, asexual
mutation, audits, and artifacts. Reproduction never occurs inside evaluation.

## Included tasks

- `reaction`: observe `a`-`d` or `end`; emit the same symbol. Each correct
  reaction is one atomic success event.
- `memory`: receive no observation; discover a hidden `a`-`h` sequence over
  repeated rewarded learning attempts, then solve it in a frozen greedy probe.
  Learning attempts emit no success events. Each correct final-probe position
  is one symmetric atomic success event, while the trial outcome records exact
  sequence success.
- `next-token`: teacher-force the complete fixed English snippet from a boundary
  token and predict the next character at every prefix position. Four complete
  supervised passes retain learned weights while resetting recurrent dynamics
  at pass boundaries. A final dynamics reset begins a plasticity-frozen greedy
  probe; each correct probe prediction is a success event.
- `continual`: receive no observation during one uninterrupted lifetime. The
  rewarded `a`-`h` action switches to a different action every deterministic
  32--96 ticks. Every correct action is one atomic success event; neural
  dynamics and learned weights are never reset by the task.
- `renewable`: receive no observation; discover the current hidden `a`-`h`
  target. Correct actions emit success events; consuming the configured stock
  deterministically renews the target.

## Adding a task

1. Add a module under `task-library/src/` implementing `SymbolicTask`.
2. Keep all task state private and deterministic from the supplied seed and
   index.
3. Define success events as facts of the environment, independent of any
   optimizer.
4. Register only CLI parsing and construction in `cli/src/ecology.rs`.
5. Use `cli ecology <task> plan` before running it.

No task-specific evaluator, optimizer, brain helper, or mutation path should be
added.
