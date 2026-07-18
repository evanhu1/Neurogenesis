# Hidden-string greedy-prefix v4, 100-generation diagnostic

Status: completed; signs of life observed, competence gate not reached

## Question

Does greedy longest-correct-prefix fitness produce early, seed-consistent
evolutionary progress before committing to the preregistered 500-generation
replication?

## Method

Seeds 211, 307, and 401 ran concurrently for 100 generations with population
64, four evaluator workers per seed, and the v4 hidden-string contract. Fitness
was the mean final greedy prefix score: `0/4` through `4/4`, stopping credit at
the first incorrect argmax output. Hard exact rate remained the competence
metric. Each seed performed 6,405 total genome evaluations including sparse
development and final sealed evaluation.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-greedy-prefix-v4-100g-check/`

## Results

| Seed | Train prefix gen 0 | Train prefix gen 99 | Sealed prefix | Sealed hard exact | Sealed character accuracy | Reset prefix | Shuffled prefix | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 211 | 0.0992 | 0.1819 | 0.1890 | 0.54% | 38.82% | 0.0344 | 0.0144 | 37.2 s |
| 307 | 0.1033 | 0.2367 | 0.2402 | 1.66% | 42.13% | 0.0359 | 0.0111 | 39.3 s |
| 401 | 0.0964 | 0.2476 | 0.2516 | 1.12% | 43.47% | 0.0515 | 0.0115 | 38.0 s |

The uniform-random expected prefix score is approximately 0.0357 and its exact
rate is `1/4096`, or 0.0244%. All three sealed treatments substantially exceed
both baselines. Treatment prefix score is 4.9x to 6.7x its reset-weights
control, and every shuffled-reward control remains near 0.01 prefix score with
zero exact strings.

All three training prefix trajectories ended at their historical maximum on
generation 99. Landmark training prefix scores were:

| Seed | Gen 0 | Gen 24 | Gen 49 | Gen 74 | Gen 99 |
|---:|---:|---:|---:|---:|---:|
| 211 | 0.0992 | 0.1691 | 0.1721 | 0.1776 | 0.1819 |
| 307 | 0.1033 | 0.1698 | 0.1959 | 0.1959 | 0.2367 |
| 401 | 0.0964 | 0.1786 | 0.1814 | 0.1846 | 0.2476 |

No seed crossed the 20% hard-exact threshold, which is expected for a short
diagnostic and means the competence gate did not pass.

## Interpretation

V4 has the requested signs of life. Ordered partial credit produces a strong,
generalizing signal in all three seeds; development and sealed prefix scores
closely track training, causal controls remain low, and none of the trajectories
had plateaued by generation 100. Unlike v3's early behavior, the same seed did
not remain at zero hard exact: every v4 seed produced measurable sealed exact
strings.

This is not yet evidence that prefix fitness solves robust discovery. The score
can favor an early-position curriculum and sealed exact rates remain only
0.54%–1.66%. The preregistered 500-generation run is justified, with prefix-by-
position and hard exact remaining the decisive diagnostics.

## Decision

Advance v4 to the matched 500-generation replication. Do not unlock the
sequential continual-learning task unless at least two of three seeds pass the
unchanged sealed competence and persistence gate.
