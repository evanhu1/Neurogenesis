# Hidden-string soft-exact v3, 500 generations

Status: completed; robust-discovery hypothesis rejected

## Question

Does replacing hard exact-string selection with a smooth complete-string
probability make NEAT a more efficient and robust discoverer of the
within-lifetime learner?

## Method

The preregistered v3 task used an eight-symbol alphabet, length-four hidden
targets, 32 reward-guided attempts, and no sensory input. For each final frozen
probe case, selection fitness was the product of the four softmax probabilities
assigned to the correct symbols. Fitness was the mean of this complete-string
score across 1,024 training targets and two rollout seeds. Hard greedy exact
rate remained the normalized competence score and was not included in fitness.
There was no character-accuracy or AUC term.

Seeds 211, 307, and 401 ran concurrently for 500 generations with population
64 and four evaluator workers per seed. Development ran every 25 generations;
sealed primary, shuffled-reward, and reset-weights assays ran once for the
terminal winner. Population checkpoints were written every 10 generations.

Artifacts:

- `artifacts/research/runs/completed/2026-07-18-hidden-string-soft-exact-v3-500g/`
- seed 211 run: `neat-hidden-string-run-1784360168735-91038/`
- seed 307 run: `neat-hidden-string-run-1784360168736-91039/`
- seed 401 run: `neat-hidden-string-run-1784360168736-91040/`

## Results

| Seed | Train hard exact | Development hard exact | Sealed hard exact | Sealed soft exact | Sealed character accuracy | Shuffled hard exact | Reset hard exact | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 211 | 52.20% | 57.81% | 50.00% | 0.21108 | 83.92% | 0.00% | 0.00% | pass |
| 307 | 0.39% | 0.00% | 1.22% | 0.00833 | 49.57% | 0.00% | 0.15% | fail |
| 401 | 11.23% | 12.11% | 7.52% | 0.03477 | 60.84% | 0.00% | 0.00% | fail |

All causal controls passed their 2% hard-exact ceiling, but only seed 211
passed the 20% sealed competence and 15-point adaptation-gain gates. The
required two-of-three robustness rule therefore failed.

Seed 211 crossed 20% training hard exact at generation 238 after 15,296
population genome evaluations, then crossed 50% at generation 469 after
30,080 evaluations. Seeds 307 and 401 never crossed 20%. All three terminal
winners were still historical soft-fitness maxima at generation 499, so the
soft objective had not plateaued by the imposed 500-generation horizon.

### Comparison at generation 499

The previous hard-exact batch used the same three evolutionary seeds. Its
generation-499 logs provide the closest direct comparison, although its older
runner and evaluator instrumentation make elapsed time a system-level rather
than selection-only comparison.

| Seed | Previous train hard exact | V3 train hard exact | Change | Previous elapsed | V3 total wall | Per-seed speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 211 | 24.61% | 52.20% | +27.59 pp | 1,370.1 s | 298.8 s | 4.58x |
| 307 | 39.45% | 0.39% | -39.06 pp | 1,957.0 s | 213.9 s | 9.15x |
| 401 | 3.13% | 11.23% | +8.11 pp | 1,272.1 s | 284.1 s | 4.48x |

Concurrent batch wall time fell from the previous slowest process's 1,957
seconds to 299 seconds, a 6.55x improvement. Each v3 run performed 32,021 total
genome evaluations. Synapse operations were 255.31 billion, 107.06 billion,
and 211.74 billion for seeds 211, 307, and 401 respectively.

Every run completed normally and wrote 50 standalone population checkpoints,
an independent terminal champion, historical champions, threshold telemetry,
and final development/sealed results.

## Interpretation

The scoring correction worked mechanically. Early genomes received a changing
selection signal while hard exact rate was still zero, and seeds 211 and 401
made substantially more hard-exact progress than their previous counterparts.
The efficient evaluator and population parallelism also reduced batch wall
time by more than sixfold.

It did not make evolutionary discovery robust. Seed 307 moved character
accuracy to 49.57% and increased soft exact score while producing almost no
greedy exact strings. The identity of the failed seed changed rather than the
failure mode disappearing, and the median training hard-exact rate fell from
24.61% in the previous batch to 11.23% in v3. Mean training hard exact was
roughly unchanged because seed 211's large gain offset seed 307's collapse.

The remaining mismatch is now clearer: selection optimizes expected exact
probability under a temperature-one softmax, while competence is greedy exact
emission. That surrogate is smooth and sequence-level, but it can reward
distributed target probability without forcing every correct symbol across
the argmax boundary. The next scoring experiment should retain dense sequence
credit while tightening it toward the greedy endpoint, for example with a
lower-temperature correct-vs-best-wrong margin surrogate or a scheduled hybrid
that increasingly weights hard exact after discovery. It should be tested as a
matched scoring ablation before spending another long evolutionary budget.

## Decision

Retain the lifecycle, checkpoint, telemetry, and parallel evaluator changes.
Retain soft sequence diagnostics. Reject temperature-one soft exact alone as a
robust solution to evolutionary discovery, and do not unlock the sequential
continual-learning task from this result.
