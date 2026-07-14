# Archived substrate-redesign audit

Candidate: commit `7889242` (`Redesign: environment-agnostic, indirectly-encoded evolutionary substrate`), audited in a detached worktree at `/Users/evanhu/code/NeuroGenesis-substrate-audit`.

## What it implements

- An environment-agnostic CPPN/ES-HyperNEAT-like genome and developmental map.
- Evolvable per-edge plasticity scales, morphology headers, sexual reproduction,
  deterministic hex and toy environments, and a `QdArchive` type.
- A simplified `sim-evaluation` that constructs an archive only from the final
  living population.

## Verification

`cargo check --workspace` and `cargo test --workspace` pass. The workspace has
no unit tests; Cargo reports zero tests in every crate. The build emits example
binary-name collision warnings for the three `headless` examples.

Reproducible screen:

```sh
cd /Users/evanhu/code/NeuroGenesis-substrate-audit
cargo run -p sim-evaluation --release -- \
  --seeds 1,2,3,4 --ticks 5000 --width 32 --founders 200
target/release/sim-evaluation \
  --seeds 1,2,3,4 --ticks 20000 --width 32 --founders 200
```

| Horizon | Mean alive | Mean final-state QD coverage | Per-seed coverage |
|---:|---:|---:|---|
| 5,000 | 193.25 | 31.5 | 39, 22, 37, 28 |
| 20,000 | 188.5 | 21.5 | 23, 10, 33, 20 |

At 20,000 ticks, mean connections are seed-dependent and can inflate sharply
(seed 2: 119.24 mean edges) while the evaluator reports no competence or
behavioral-trace measure.

## Adversarial verdict

Blocked. The archive descriptor is only `[brain complexity, generation depth,
energy fraction]`; it aliases structural bloat and energy state to behavior.
The archive is not used by selection and is rebuilt from the terminal living
population, so its coverage can fall (and did fall by 31.7% from the 5k to 20k
screen). `qd_score` grows mechanically with generation because generation is
also added to the per-cell quality. This is metric gaming plus bounded terminal
diversity, not adaptive tail novelty. The representation work may be reusable,
but it is not an open-ended algorithm or demonstration.
