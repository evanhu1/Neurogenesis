# Basic reaction task

Status: saturated
Slug: 2026-07-17-basic-reaction-task
Date: 2026-07-17

## Hypothesis

Generational NEAT can evolve a controller that exactly copies a serial input
drawn from the original four-symbol reaction benchmark.

## Task

- Alphabet: `a` through `d`, plus `end`.
- Baseline corpus: 32 deterministic pseudorandom training streams and 32
  separately seeded holdout streams. Every stream contains 16 shuffled body
  symbols plus `end`, with `a` through `d` all represented.
- One brain evaluation per input symbol.
- Emission: deterministic maximum action logit, with alphabet-order tie break.
- Fitness: one point for every emitted symbol equal to its input symbol.
- Holdout score is measured for each training winner but never used by
  selection or breeding.
- Brain state persists within a stream and resets between independent streams.
- Each genome is evaluated alone. No ecology, world simulation, opponents,
  matchups, relative score, or survival objective participates in selection.

## Retained mechanisms

The existing NEAT outer loop remains: historical markings, speciation,
fitness-sharing offspring allocation, elitism, crossover, parameter mutation,
add-connection mutation, and add-node mutation. The existing brain retains its
current-step feed-forward DAG and previous-step recurrent hidden connections.

## First run

```bash
cargo build -p cli --release
./target/release/cli basic-reaction plan --population 64 --generations 100
./target/release/cli basic-reaction --seed 17 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/basic-reaction-baseline
```

Success is a final winner with training fitness `544/544` and unseen holdout
fitness `544/544`, with exact emitted outputs in both corpora.

## Conclusion

The seed-17 run met the full gate at generation 24: `544/544` on both the
training corpus and the separately seeded unseen holdout corpus. The parrot
benchmark is saturated as a research target; it remains a reference evaluator
and regression workload.
