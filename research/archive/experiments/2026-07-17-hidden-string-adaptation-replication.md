# Hidden-string adaptation independent replication

Status: completed; robust-performance gate rejected (1/3 seeds passed)
Slug: `2026-07-17-hidden-string-adaptation-replication`
Date: 2026-07-17

## Question

Does the hidden-string v0 treatment reproduce across three fresh evolutionary
seeds without changing the task, learning rule, evaluation cohorts, compute
budget, or success thresholds after observing the seed-17 discovery run?

## Hypothesis

The reward-driven adaptation mechanism is not specific to the discovery seed.
Normal NEAT complexification should repeatedly evolve recurrent position
representations and a useful learning rate, producing sealed within-lifetime
adaptation while the plasticity-off, incorrect-reward, and weight-reset controls
remain below their preregistered ceilings.

## Frozen contract

This replication uses the treatment contract from
[`2026-07-17-hidden-string-adaptation-v0.md`](../archive/experiments/2026-07-17-hidden-string-adaptation-v0.md)
without modification:

- zero sensory input and four `a`/`b`/`c`/`d` outputs per attempt;
- disjoint 64-target training, development, and sealed cohorts;
- two fixed rollout streams per target;
- 32 rewarded attempts with probes after `0, 1, 2, 4, 8, 16, 32` attempts;
- immediate reward `+1` correct and `-1/3` incorrect;
- runtime hidden-to-action learning with evolved `hebb_eta_gain` and no
  eligibility trace;
- fitness `0.9 * final_accuracy + 0.1 * learning_auc` on training only;
- population 64, 100 generations, canonical seed genome, default NEAT
  structural and parameter mutation.

The discovery seed 17 is excluded from this replication. Fresh seeds are
`7`, `27`, and `47`, chosen as a fixed arithmetic sequence before execution.
No seed may be replaced after observing its result.

## Measurements

For every seed: sealed pre-learning accuracy, final accuracy, adaptation gain,
exact-string rate, learning AUC, full probe curve, three control final
accuracies, evolved learning rate, hidden nodes, and enabled connections.

Across seeds: minimum, median, mean, and range of final accuracy and adaptation
gain; number of seeds passing each individual gate and the complete v0 gate.
Development results remain diagnostic. Each sealed cohort is evaluated only
once for its final evolutionary winner.

## Decision rule

Robust adaptation passes only if all three fresh seeds independently satisfy
the unchanged v0 gate:

1. sealed final accuracy at least 0.75;
2. sealed adaptation gain at least 0.30;
3. plasticity-off final accuracy at most 0.35;
4. permuted-reward final accuracy at most 0.35;
5. reset-each-attempt final accuracy at most 0.40.

If two of three pass, classify the result as seed-sensitive and do not advance
to continual learning without diagnosing the failed run. If fewer than two
pass, reject robustness. Exact-string rate is reported but is not an added gate.

## Compute contract and commands

```bash
cargo build -p cli --release

./target/release/cli hidden-string plan \
  --seed 7 --population 64 --generations 100
./target/release/cli hidden-string plan \
  --seed 27 --population 64 --generations 100
./target/release/cli hidden-string plan \
  --seed 47 --population 64 --generations 100

./target/release/cli hidden-string \
  --seed 7 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-replication/seed-7
./target/release/cli hidden-string \
  --seed 27 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-replication/seed-27
./target/release/cli hidden-string \
  --seed 47 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-replication/seed-47
```

Each seed evaluates 6,400 genomes and 104,857,600 rewarded training action
decisions, plus frozen probes and development evaluation. The full replication
therefore contains 19,200 genome evaluations and 314,572,800 rewarded training
decisions.

## Result

All three fresh seeds completed under the frozen contract.

| Seed | Final accuracy | Gain | Exact strings | Plasticity off | Wrong reward | Reset each attempt | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 0.7871 | +0.5371 | 0.2031 | 0.2500 | 0.0664 | 0.2793 | pass |
| 27 | 0.5840 | +0.3340 | 0.1016 | 0.2500 | 0.1406 | 0.3008 | fail: accuracy |
| 47 | 0.7188 | +0.4688 | 0.1641 | 0.2500 | 0.0938 | 0.4277 | fail: accuracy, reset |

Only one of three seeds passed the complete gate. Under the preregistered rule,
fewer than two passes rejects robust adaptation.

Across seeds, sealed final accuracy had minimum 0.5840, median 0.7188, mean
0.6966, and maximum 0.7871. Adaptation gain had minimum 0.3340, median 0.4688,
mean 0.4466, and maximum 0.5371. All three gains exceeded the +0.30 threshold;
all plasticity-off controls stayed at chance and all incorrect-reward controls
stayed below 0.15.

The final networks differed materially: seed 7 evolved 3 hidden nodes and 13
enabled connections; seed 27 evolved 2 and 8; seed 47 evolved 3 and 15. Their
learning rates were 0.3206, 0.4693, and 0.2950 respectively.

Seed 27 continued improving through generation 99 and added its second hidden
node only near the end. Seed 47 reached its final training/development level by
generation 80 and then plateaued. These trajectories make the fixed
100-generation search budget a plausible contributor, but not an established
explanation.

The runs took approximately 13.5, 11.2, and 14.5 seconds concurrently. Plans,
progress JSONL, typed compressed results, aggregate analysis, source status,
commands, and verified SHA-256 checksums are stored under
`artifacts/research/runs/completed/2026-07-17-hidden-string-adaptation-replication/`.

`cargo fmt --check`, `git diff --check`, and `cargo test --workspace` passed; all
24 existing tests passed and no tests were added.

## Interpretation and next decision

Reward-dependent within-lifetime adaptation replicated qualitatively in all
three fresh seeds: every frozen probe trajectory rose substantially, and the
plasticity-off and wrong-reward controls ruled out an inherited answer or
reward-independent drift. What did not replicate was the preregistered level of
competence and persistence. Only seed 7 crossed 0.75 accuracy, and seed 47's
reset control showed that much of its performance could be reacquired within a
single attempt rather than accumulated across the lifetime.

Therefore the project should not yet advance to sequential hidden targets or
claim robust continual learning. The next narrow diagnostic should rerun the
same three seeds for 200 generations with every other contract field frozen.
That distinguishes a too-short structural-search budget—especially for seed
27—from a stable seed-sensitive limitation. Any promotion to a 200-generation
baseline would still need a subsequent fresh-seed confirmation.
# Architecture-audit note

Invalidated as canonical-brain competence on 2026-07-18 because this evaluator
used a task-rewritten recurrent/readout founder.
