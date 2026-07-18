# Current objective: robust within-lifetime hidden-string adaptation

## Objective

Establish that a recurrent NEAT brain can learn an unknown four-symbol target
during inference from reward alone, without a sensory stream that reveals the
target and without evolution encoding each answer directly.

The symbol-copy evaluator remains a reference training task behind the modular
`EvaluationTask` boundary. It is not the current research target.

## Current evidence

The seed-17 hidden-string v0 experiment is a positive existence proof. On a
sealed 64-target cohort, treatment accuracy rose from 25.0% before learning to
77.9% after 32 attempts. Plasticity-off, permuted-reward, and reset-each-attempt
controls stayed below their preregistered ceilings. A matched fixed-topology arm
reached only 45.3%, while the treatment grew from one hidden node and four
enabled connections to five and 22.

See
[`archive/experiments/2026-07-17-hidden-string-adaptation-v0.md`](archive/experiments/2026-07-17-hidden-string-adaptation-v0.md).

The independent replication did not establish robustness. All three fresh
seeds adapted by at least 33.4 points and retained chance plasticity-off
behavior, but only seed 7 passed the complete gate. Seeds 27 and 47 ended at
58.4% and 71.9% sealed accuracy; seed 47 also exceeded the reset-control
ceiling. See
[`archive/experiments/2026-07-17-hidden-string-adaptation-replication.md`](archive/experiments/2026-07-17-hidden-string-adaptation-replication.md).

The v3 soft exact-string experiment also failed robustness. At 500 generations,
sealed hard exact rates were 50.00%, 1.22%, and 7.52% for seeds 211, 307, and
401; only seed 211 passed the complete competence/persistence gate. All causal
controls passed. The parallel evaluator reduced concurrent batch wall time by
6.55x, and the smooth score exposed progress before hard exact hits appeared,
but it changed which seed failed rather than eliminating seed-dependent
basins. See
[`archive/experiments/2026-07-18-hidden-string-soft-exact-v3-500g.md`](archive/experiments/2026-07-18-hidden-string-soft-exact-v3-500g.md).

## Next gate

The v0 diagnosis found a weak, repeat-heavy basin: its four-symbol cohorts had
unequal repeat composition, its score rewarded character accuracy, and the
founder topology varied with evolutionary seed. The incomplete v1 run removed
those shortcuts but embedded exhaustive measurement and controls in every
generation.

Hidden-string v3 retains the strict eight-symbol hard exact-string competence
gate and deterministic founder, but corrects v2's sparse selection signal. A
uniform policy produces only 0.5 expected hard-exact hits across the 2,048
training cases. Evolution now selects on mean final complete-string
probability: the product of the four correct-symbol probabilities. There is no
character or AUC fitness term; hard greedy exact rate remains the normalized
competence score.

The disjoint panels retain exact position balance and matched repeat
composition. Rank-fidelity calibration rejected every reduced training
contract through `1024x1`, so training remains `1024x2`. Population evaluation
is deterministic across worker counts; training has one probe, development
runs primary-only every 25 generations, and causal controls run only in the
final sealed evaluation. Runs now checkpoint incrementally, stop gracefully,
resume deterministically, persist champions independently, and report
evaluations-to-threshold, synapse operations, and wall time.

The v3 gate failed, so sequential targets remain locked. Hidden-string v4 now
uses greedy longest-correct-prefix fitness: `0/4` through `4/4`, stopping credit
at the first wrong argmax output. This removes soft probability from selection
while retaining ordered partial credit. Its 100-generation diagnostic showed
consistent early life: sealed prefix scores were 0.1890, 0.2402, and 0.2516,
sealed exact rates were 0.54%–1.66%, all controls stayed low, and every seed
ended at its best prefix score. The next gate is a matched three-seed,
500-generation v4 replication under the unchanged hard-exact competence and
causal-control rule. See the [v4 diagnostic](archive/experiments/2026-07-18-hidden-string-greedy-prefix-v4-100g-diagnostic.md),
[v4 proposal](proposed/2026-07-18-hidden-string-greedy-prefix-v4.md),
[v3 result](archive/experiments/2026-07-18-hidden-string-soft-exact-v3-500g.md),
and [optimization calibration](archive/experiments/2026-07-17-hidden-string-evaluator-optimization.md).
