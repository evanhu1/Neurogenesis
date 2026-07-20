# Hidden-string seed-27 500-generation diagnostic

Status: completed; seed 27 passed exactly at the competence threshold
Slug: `2026-07-17-hidden-string-seed27-500g-diagnostic`
Date: 2026-07-17

## Question

Can the remaining failed seed 27 eventually cross the unchanged hidden-string
competence gate when its evolutionary budget is extended from 200 to 500
generations?

## Hypothesis

Seed 27 may be trapped by slow structural search rather than an absolute
learning limitation. If additional topology appears after its generation-160
plateau, another 300 generations may rescue sealed accuracy. Continued
plateauing would reject generation budget as a sufficient fix for this seed.

## Frozen contract

Rerun seed 27 deterministically from generation zero with population 64 and the
unchanged hidden-string v0 treatment. Change only `generations` from 200 to 500.
Generation-199 fitness, evaluations, winner genome, learning rate, and structure
must exactly match the archived 200-generation artifact.

This seed was selected because it failed at 100 and 200 generations. The run is
therefore diagnostic and cannot establish robustness.

## Decision rule

Apply the unchanged complete sealed gate at generation 499:

1. final accuracy at least 0.75;
2. adaptation gain at least 0.30;
3. plasticity-off accuracy at most 0.35;
4. permuted-reward accuracy at most 0.35;
5. reset-each-attempt accuracy at most 0.40.

A complete pass shows that additional budget can rescue seed 27. A failure
rejects budget alone as sufficient in the tested 500-generation range.

## Compute contract and command

```bash
./target/release/cli hidden-string plan \
  --seed 27 --population 64 --generations 500

./target/release/cli hidden-string \
  --seed 27 --population 64 --generations 500 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-seed27-500g-diagnostic
```

The run evaluates 32,000 genomes and 524,288,000 rewarded training decisions,
plus frozen probes and development evaluations.

## Result

The generation-199 fitness, training and development evaluations, winner
genome, learning rate, and structure exactly matched the archived
200-generation result. The deterministic-prefix integrity gate passed.

| Measurement | 100g | 200g | 500g | Gate |
|---|---:|---:|---:|---:|
| Sealed final accuracy | 0.5840 | 0.6641 | 0.7500 | >= 0.75 |
| Adaptation gain | +0.3340 | +0.4141 | +0.5000 | >= 0.30 |
| Exact-string rate | 0.1016 | 0.1797 | 0.2031 | report only |
| Plasticity off | 0.2500 | 0.2500 | 0.2500 | <= 0.35 |
| Wrong reward | 0.1406 | 0.0898 | 0.1191 | <= 0.35 |
| Reset each attempt | 0.3008 | 0.3086 | 0.3145 | <= 0.40 |

Seed 27 passed the complete gate at generation 499, landing exactly on the
0.75 competence threshold. Its sealed probe accuracy after
`0, 1, 2, 4, 8, 16, 32` attempts was
`0.2500, 0.3086, 0.3379, 0.4434, 0.5957, 0.7070, 0.7500`.

Training accuracy crossed 0.75 for the first time at generation 336. It then
spent long intervals near that level. The final winner had 6 hidden nodes, 17
enabled connections, and learning rate 0.6415, compared with 2 hidden nodes,
11 connections, and learning rate 0.3479 at generation 199. Winner topology was
not monotonic: some earlier two-node winners also crossed 0.75 training
accuracy, so this run does not isolate topology growth as the cause.

The release run completed in approximately 66.3 seconds. Plans, progress,
typed results, the generation-199 equality record, comparison analysis, source
status, commands, and verified checksums are stored under
`artifacts/research/runs/completed/2026-07-17-hidden-string-seed27-500g-diagnostic/`.

`cargo fmt --check`, `git diff --check`, and `cargo test --workspace` passed;
all 24 existing tests passed and no tests were added.

## Interpretation and next decision

Additional evolutionary budget was sufficient to rescue the specifically
selected seed within 500 generations. Across the observed seeds, the learning
mechanism can therefore reach the v0 gate without changing the task or learning
rule. The 100- and 200-generation cutoffs understated its eventual capability.

The result is marginal rather than comfortable: sealed accuracy equals the
threshold exactly, the seed was chosen after two failures, and the run gives no
fresh-seed success probability. It also cannot distinguish more time for weight
and learning-rate search from more time for topology search.

Do not treat this as robust continual learning or begin sequential targets yet.
The clean confirmation is a preregistered 500-generation suite on fresh seeds.
A richer-founder comparison remains useful as an efficiency experiment, but is
no longer required to show that seed 27 can eventually solve v0.
# Architecture-audit note

Invalidated as canonical-brain competence on 2026-07-18 because this evaluator
used a task-rewritten recurrent/readout founder.
