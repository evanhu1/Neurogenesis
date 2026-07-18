# Hidden-string adaptation 200-generation budget diagnostic

Status: completed; insufficient budget is a partial explanation (2/3 passed)
Slug: `2026-07-17-hidden-string-adaptation-200g-diagnostic`
Date: 2026-07-17

## Question

Did the fresh-seed replication fail because 100 generations was too short for
structural search, or do the weak seeds remain below the frozen competence and
persistence gate with twice the evolutionary budget?

## Hypothesis

Seed 27 added its second hidden node only near generation 99 and was still
improving, so extending the unchanged run to 200 generations may rescue it.
Seed 47 plateaued by generation 80, making its rescue less likely. If both
failed seeds pass at generation 199 while seed 7 retains its pass, the
100-generation budget explanation is supported.

## Frozen contract

Use the exact hidden-string v0 treatment and the same already-observed seeds
`7`, `27`, and `47`. Change only `generations` from 100 to 200:

- population 64 and canonical seed genome;
- default NEAT structural and parameter mutation;
- zero sensory input, four-symbol targets, 32 rewarded attempts;
- unchanged target cohorts, rollout seeds, reward, probes, learning rule,
  fitness, and three within-genome controls.

These are deterministic reruns from generation zero, not checkpoint resumes.
Generation-99 training/development metrics and winner structure must exactly
match the corresponding 100-generation artifacts. A mismatch invalidates the
comparison.

## Measurements

For each seed, compare generation 99 and generation 199 training/development
accuracy, network structure, and learning rate. At generation 199 report sealed
accuracy, adaptation gain, exact-string rate, probe curve, and all controls.

## Decision rule

Apply the unchanged complete gate to every generation-199 sealed result:

1. final accuracy at least 0.75;
2. adaptation gain at least 0.30;
3. plasticity-off accuracy at most 0.35;
4. permuted-reward accuracy at most 0.35;
5. reset-each-attempt accuracy at most 0.40.

- Support the insufficient-budget explanation only if seeds 27 and 47 are both
  rescued and seed 7 retains a complete pass.
- Classify it as a partial explanation if exactly one failed seed is rescued.
- Reject it if neither failed seed is rescued.

Even a 3/3 diagnostic pass does not establish robustness because these seeds
were selected after observing their 100-generation results. It would require a
fresh-seed confirmation at 200 generations.

## Compute contract and commands

```bash
cargo build -p cli --release

./target/release/cli hidden-string plan \
  --seed 7 --population 64 --generations 200
./target/release/cli hidden-string plan \
  --seed 27 --population 64 --generations 200
./target/release/cli hidden-string plan \
  --seed 47 --population 64 --generations 200

./target/release/cli hidden-string \
  --seed 7 --population 64 --generations 200 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-200g-diagnostic/seed-7
./target/release/cli hidden-string \
  --seed 27 --population 64 --generations 200 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-200g-diagnostic/seed-27
./target/release/cli hidden-string \
  --seed 47 --population 64 --generations 200 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-200g-diagnostic/seed-47
```

Each seed evaluates 12,800 genomes and 209,715,200 rewarded training decisions.
The diagnostic totals 38,400 genome evaluations and 629,145,600 rewarded
training decisions, plus probes and development evaluations.

## Result

All generation-99 training evaluations, development evaluations, fitnesses,
winner genomes, learning rates, and winner structures exactly matched the
corresponding 100-generation artifacts. The deterministic-prefix integrity gate
passed for every seed.

| Seed | Accuracy at 100g | Accuracy at 200g | Change | Reset at 200g | Exact strings at 200g | 200g gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 0.7871 | 0.8789 | +0.0918 | 0.3203 | 0.5859 | pass |
| 27 | 0.5840 | 0.6641 | +0.0801 | 0.3086 | 0.1797 | fail: accuracy |
| 47 | 0.7188 | 0.8652 | +0.1465 | 0.3086 | 0.6016 | pass |

Seed 47 was fully rescued: it crossed the competence threshold and its reset
control fell from 0.4277 to 0.3086. Seed 7 retained its pass and improved. Seed
27 improved but remained 0.0859 below the 0.75 accuracy gate. Pass count rose
from one of three at 100 generations to two of three at 200 generations.

At 200 generations, final accuracy had minimum 0.6641, median 0.8652, mean
0.8027, and maximum 0.8789. Every seed passed adaptation gain, plasticity-off,
wrong-reward, and reset-control thresholds; seed 27's final accuracy was the
only remaining gate failure.

Seed 27 rose quickly from 0.5762 training accuracy at generation 99 to 0.7031 by
generation 160, then remained flat through generation 199. Its final structure
retained two hidden nodes and grew only from 8 to 11 enabled connections. Seed
47 grew from 3 hidden nodes/15 connections at generation 99 to 4/19 at the
finish; seed 7 retained 3 hidden nodes while growing from 13 to 21 connections.

The three concurrent runs took approximately 28.5, 24.4, and 31.4 seconds.
Plans, progress, typed results, generation-99 equality records, aggregate and
paired analyses, source status, commands, and verified checksums are stored
under
`artifacts/research/runs/completed/2026-07-17-hidden-string-adaptation-200g-diagnostic/`.

`cargo fmt --check`, `git diff --check`, and `cargo test --workspace` passed;
all 24 existing tests passed and no tests were added.

## Interpretation and next decision

The preregistered outcome is a partial budget explanation. Doubling the budget
rescued seed 47 and materially strengthened seed 7, proving that the original
100-generation cutoff was premature for some populations. It did not rescue
seed 27, so generation count alone does not remove the search sensitivity.

This remains strong evidence that the learning mechanism can work: two seeds
now exceed 86% sealed accuracy with all causal controls passing, and even seed
27 improves by 41.4 points within a target lifetime. It is not evidence for a
reliable baseline because the observed seeds were reused and one stayed trapped
below competence.

Do not advance to sequential targets yet. The next experiment should target
the structural-search bottleneck rather than add more generations blindly. A
matched multi-seed comparison of the default founder against a founder with
enough initial hidden capacity for four temporal positions would test whether
NEAT is being asked to discover both a usable clock and a learning rule in the
same sparse search. Any promoted treatment must then pass on fresh seeds.
