# Modular NEAT evaluation tasks

The `evolution` crate separates the generational NEAT algorithm from the task
that assigns fitness to one genome.

```text
EvaluationTask::evaluate(genome)
              |
              v
absolute fitness -> rank/speciate/select/breed -> next population
```

`evolution::run_neat` owns population initialization, compatibility speciation,
fitness sharing, offspring allocation, elitism, crossover, and mutation. It
does not know about symbol streams, worlds, agents, or task-specific metrics.

## Task contract

A task implements `evolution::EvaluationTask` and supplies:

- a serializable task configuration;
- a serializable per-genome evaluation result;
- validation of the resolved task contract;
- deterministic evaluation of one `OrganismGenome`;
- a finite, nonnegative scalar fitness used by NEAT;
- an optional normalized value used only for reporting;
- an optional validation evaluation for each generation winner;
- a task-owned validation cadence;
- an optional sealed evaluation called once for the final winner;
- optional founder, sensor, and action-output constraints owned by the task;
- a deterministic work-report hook, zero by default, for task-specific brain
  synapse-operation telemetry.

Validation evaluation is never passed to ranking, selection, fitness sharing,
or breeding. Tasks may make it sparse with `validation_due`; tasks that do not
need it return `None` using the trait default.

The trait is generic rather than dynamically dispatched. Rust monomorphizes
the evaluation call, preserving a direct hot path while allowing each task to
persist its own strongly typed configuration and result details.

## Adding a task

1. Add a module under `evolution/src/tasks/` and export it from
   `evolution/src/tasks/mod.rs`.
2. Define task configuration and evaluation result types with `Clone`,
   `Serialize`, and `Deserialize`.
3. Implement `EvaluationTask`. Fixed configuration plus seed must produce the
   same evaluation.
4. Add a concrete CLI route that parses the task configuration, invokes
   `run_neat(&task, neat_config, ...)`, and renders the task-specific result.
5. Keep task logic out of `evolution/src/lib.rs`; changing tasks must not alter
   speciation, selection, crossover, or mutation.

## Reference training task

`evolution::tasks::symbol_copy::SymbolCopyTask` is the reference implementation.
It owns the deterministic training/holdout corpora, sensory presentation,
action decoding, exact-match fitness, and task-specific result schema. The
generic NEAT loop sees only its returned training fitness and optional holdout
evaluation.

## Hidden-string adaptation experiment

`evolution::tasks::hidden_string::HiddenStringTask` is the first learning task.
It supplies no sensory signal, evaluates a fresh learned-weight lifetime for
each hidden target, and applies immediate reward to runtime hidden-to-action
weights. It uses a deterministic recurrent founder, exposes `a`–`h` but not
`end` to structural mutation, and selects on greedy longest-correct-prefix
fitness at the final probe. A case receives `k/4`, where `k` is the number of
consecutive correct argmax outputs before its first error; later correct
positions receive no credit. Hard exact-string rate remains the normalized
competence metric and threshold gate, while unordered character accuracy is
diagnostic. Training,
development, and sealed target cohorts are disjoint and composition-matched.
Training uses one final probe; development runs
primary-only on a sparse cadence; shuffled-reward and reset-weights controls run
only during the once-per-run sealed evaluation. Use `cli hidden-string`.

The generic runner also owns standalone generation-boundary checkpoint state,
resume, normalized-fitness threshold events, historical frozen champions,
deterministic work totals, and explicit completion/early-stop metadata.
Filesystem and signal handling remain CLI responsibilities. Hidden-string
frozen-genome reevaluation remains task-specific and does not widen the generic
evaluation boundary. See [`neat-run-lifecycle.md`](neat-run-lifecycle.md).
