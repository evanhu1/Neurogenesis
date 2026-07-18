# Hidden-string evaluator optimization and calibration

Status: completed implementation and calibration

## Question

Can the hidden-string research loop remove routine measurement overhead and use
parallel genome evaluation without changing deterministic evolutionary
semantics? Can a sampled target panel preserve the full evaluator's rankings?

## Implementation

- Parallelism moved to the independent population-genome boundary. A local
  Rayon pool is explicitly sized by `NeatConfig.evaluation_workers` and the CLI
  `--workers` flag. Brain trajectories remain sequential.
- Training now performs one frozen probe, at attempt 32, and fitness is only the
  final exact-string rate. The logarithmically sampled pseudo-AUC was removed.
- Development evaluates the generation winner every 25 generations and on the
  final generation. It runs the primary treatment only.
- Sealed evaluation runs once after final-winner selection. Its primary probe
  schedule is `[0,8,16,32]`; shuffled-reward and reset-weights controls run only
  the final probe. The zero-attempt primary probe is the static baseline, so a
  routine plasticity-off replay was removed.
- Target panels are fixed, disjoint, hash-shuffled collections of additive
  eight-symbol orbits. Every position has exact symbol marginals and panels
  share a `3:16:13` ratio of two/three/four-distinct-symbol orbits. This removes
  the old modular cohort equation and its cross-position constraint.
- Target panels and the expressed inherited brain are precomputed and reused.
  Exploration samples are keyed by target identity rather than panel index.

## Sampling calibration

Source population artifact:

`artifacts/research/runs/diagnostics/hidden-string-v1-calibration/neat-hidden-string-1784353263189-40332.json.zst`

The calibration reevaluated complete population checkpoints at generations 0,
50, and 99 under three fixed panel seeds. Each candidate was compared with a
1,024-target, two-rollout reference. A candidate had to pass every comparison:

- Spearman rank correlation at least 0.95;
- at least 6/8 overlap in the top eight;
- reference winner ranked in the candidate top three.

Candidates `128x1`, `256x1`, `256x2`, `512x1`, `512x2`, `768x2`, and
`1024x1` all failed. The strongest reduced candidate, `768x2`, had worst-case
Spearman 0.636, top-eight overlap 2/8, and reference-winner rank 15. The
`1024x1` candidate still fell to Spearman 0.587, overlap 3/8, and winner rank 5.
Only the full `1024x2` contract passed.

Decision: retain 1,024 targets and two training rollouts. Sampling was
calibrated and rejected; reducing it would materially change exact-string
selection.

Full output:

`artifacts/research/runs/diagnostics/hidden-string-v2-calibration/sampling.json`

## Horizon diagnostic

A frozen generation-99 winner was evaluated at 8, 16, 32, 64, and 128
attempts on 1,024 targets and two rollouts. Character accuracy rose from 24.5%
to 38.9%; 32 attempts reached 92.6% of the 128-attempt accuracy. Exact-string
rate was non-monotonic and peaked at 32 attempts (0.342%), versus 0.146% at 128.

Decision: retain 32 as the explicit adaptation budget. The exact-string curve
does not justify extending or shortening it.

Full output:

`artifacts/research/runs/diagnostics/hidden-string-v2-calibration/horizon.json`

## Determinism and throughput

Matched population-16, three-generation runs at seed 919 used one and four
evaluation workers. After removing only the persisted worker-count field, both
artifacts had the identical semantic SHA-256:

`25dbd8a34c07c5e21c4ef29a674f780bcc252a7f35bec5b99c199f905964ed5d`

Wall time fell from 0.93 seconds to 0.32 seconds, a 2.9x speedup. Generated
artifacts are under:

`artifacts/research/runs/diagnostics/hidden-string-v2-determinism/`

## Decision

Adopt deterministic population-level parallelism, one training probe, sparse
primary-only development, and final-only sealed controls. Retain the calibrated
1,024-target/two-rollout training panel and 32-attempt horizon.
