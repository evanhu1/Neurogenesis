# Pure symbol-copy task

Status: proposed
Slug: 2026-07-17-symbol-copy-task
Date: 2026-07-17

## Hypothesis

Generational NEAT can evolve a recurrent controller whose nine-symbol action
interface exactly copies a serial input presented through the identical
nine-symbol sensory interface.

## Task

- Alphabet: `a` through `h`, plus `end`.
- Baseline corpus: 32 deterministic pseudorandom training streams and 32
  separately seeded holdout streams. Every stream contains 16 shuffled body
  symbols plus `end`, with `a` through `h` all represented.
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
./target/release/cli plan --population 64 --generations 100
./target/release/cli --seed 17 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/symbol-copy-baseline
```

Success is a final winner with training fitness `544/544` and unseen holdout
fitness `544/544`, with exact emitted outputs in both corpora.
