# 2026-07-17-hidden-string-exact-string-v1: balanced eight-symbol adaptation

Status: superseded before completion by the calibrated v2 contract

The three 1,000-generation processes stopped without result artifacts after
generations 714, 554, and 838. Their progress JSONL is diagnostic only. The
modular cohort equation, per-generation controls, seven-probe pseudo-AUC, and
single-core evaluator are replaced by
[`hidden-string-exact-string-v2`](2026-07-17-hidden-string-exact-string-v2.md).

## Question

Can NEAT evolve a recurrent brain whose within-lifetime reward-driven updates
learn a hidden four-symbol string when success is measured only by reproducing
the complete string?

## Hypothesis

Immediate signed reward plus evolved recurrent state will create a learning
system that improves exact-string reproduction on unseen targets. The v1
contract removes v0's majority/repeated-symbol route: the body alphabet is
`a`–`h`, cohorts have identical repeat composition, the founder is fixed, and
fitness never contains a per-character term.

The mechanism is falsified if a treatment can reach the exact-string gate with
plasticity disabled, permuted reward, or weights reset between attempts.

## Contract

- Code revision: working tree containing `hidden_string_adaptation_v1`.
- Canonical config: `config/seed_genome.toml`; the task replaces its topology
  with one deterministic recurrent hidden unit and eight hidden-to-action edges.
- Treatment: immediate action reward, inherited learning-rate gene, and
  persistent runtime hidden-to-action weights.
- Controls: plasticity off; cyclically permuted reward; reset runtime weights
  after each attempt.
- Evolutionary seeds: 211, 307, 401.
- Target cohorts: four deterministic 1,024-string partitions of all 4-symbol
  strings over `a`–`h`, assigned by `(a + b + c + 2d) mod 4`; train=0,
  development=1, sealed=2.
- Rollout seeds: task defaults, two per target/cohort.
- Population: 64.
- Generations: 1,000.
- Episode horizon: 32 learning attempts; frozen probes after 0, 1, 2, 4, 8,
  16, and 32 attempts.
- Cases per genome: 2,048 training target/rollout cases; four emitted symbols
  per attempt.
- Evaluation workers: one process per evolutionary seed.
- Artifact directory:
  `artifacts/research/runs/active/2026-07-17-hidden-string-exact-string-v1-1000g/`

No parameter override is permitted. The diagnostic smoke seed 101 is excluded
from the evidence set.

## Measurements

The primary endpoint is sealed final exact-string rate. Selection fitness is
`0.9 * final_exact_string_rate + 0.1 * exact_string_learning_auc` on the
training cohort. Per-character accuracy is diagnostic only.

For each cohort, report the exact-string probe curve, final exact-string gain,
and the three control final exact-string rates. The integrity checks are: no
sensor-to-brain edges in the founder, no `end` output edges in the task founder
or task structural mutation, equal 1/2/3/4-distinct-symbol target composition
in every cohort, and sealed evaluation only after final winner selection.

## Decision rule

The experiment passes only if at least two of the three seeds satisfy all of:

1. sealed final exact-string rate >= 0.20;
2. sealed exact-string gain >= 0.15 from the zero-attempt frozen probe;
3. each sealed control final exact-string rate <= 0.02.

Otherwise the task is a negative or inconclusive result. A high character
accuracy without the exact-string threshold is not competence.

## Commands and provenance

The exact plan and commands will be saved beside the outputs. For each seed:

```bash
./target/release/cli hidden-string plan --seed SEED --population 64 --generations 1000
./target/release/cli hidden-string --seed SEED --population 64 --generations 1000 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-exact-string-v1-1000g
```

## Result

Not run.

## Interpretation and next decision

Not run.
