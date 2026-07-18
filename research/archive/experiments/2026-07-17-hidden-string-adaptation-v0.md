# Hidden-string within-lifetime adaptation v0

Status: completed; v0 gate passed as a one-seed existence proof
Slug: `2026-07-17-hidden-string-adaptation-v0`
Date: 2026-07-17

## Question

Can NEAT evolve a brain and learning rate that discover a previously unseen,
four-symbol target during one lifetime using only immediate reward, without any
sensory input that identifies the target?

This is an adaptation assay, not yet a claim of general continual learning.

## Hypothesis

Evolution can shape recurrent position representations and an inherited online
learning rate so that reward-modulated runtime changes to hidden-to-action
weights bind four temporal positions to their correct actions. Consequently,
accuracy should rise substantially within 32 attempts on unseen target strings.

The mechanism is falsified if sealed-target accuracy does not rise over the
pre-learning probe, or if the apparent gain survives with plasticity disabled,
incorrectly permuted reward, or inherited weights restored every attempt.

## Task contract

- Target: four symbols drawn from `a`, `b`, `c`, `d`; `end` is not scored.
- Sensory neurons: always zero, and sensory-origin connections are removed from
  the founder and excluded from structural mutation.
- Output: four recurrent brain steps per attempt, one action per target position.
- Feedback: immediate `+1` for the correct symbol and `-1/3` otherwise.
- Learning: reward-modulated hidden-to-action runtime weight update with no
  eligibility trace (`lambda = 0`); the evolved `hebb_eta_gain` is the learning
  rate. Learned weights persist for the target lifetime. Recurrent activations
  reset between attempts.
- Lifetime: 32 rewarded attempts from fresh inherited runtime weights.
- Frozen deterministic probes: before learning and after attempts
  `1, 2, 4, 8, 16, 32`. Probes do not update weights.
- Target split: three disjoint, position-balanced 64-string cohorts selected by
  `(a + 3b + c + 3d) mod 4`: cohort 0 training, cohort 1 development, cohort 2
  sealed final evaluation.
- Replication: two independent deterministic rollout streams per target and
  cohort.
- Evolutionary fitness: `0.9 * final_accuracy + 0.1 * learning_auc` on the
  training cohort only. Development and sealed results never participate in
  ranking, selection, or breeding.

## Arms and controls

Both arms use evolutionary seed 17, population 64, 100 generations, the
canonical `config/seed_genome.toml`, and otherwise default NEAT parameters.

- Treatment: normal NEAT parameter and structural mutation.
- Fixed-topology arm: identical, except `add_connection_probability=0` and
  `add_node_probability=0`. Weight, neuron, and learning-rate evolution remain
  enabled.

Every development and sealed evaluation also executes these within-genome
controls:

- plasticity off;
- reward for a fixed incorrect permutation (`a->b->c->d->a`);
- restore inherited runtime weights at the start of every attempt.

Chance symbol accuracy is 0.25. A non-learning policy can exploit cohort
statistics only through inherited behavior; the plasticity-off control measures
that directly.

## Measurements

Primary endpoint: sealed-cohort final symbol accuracy after 32 attempts.

Secondary endpoints: sealed pre-learning accuracy, adaptation gain, learning
AUC across post-attempt probes, exact four-symbol string rate, probe trajectory,
the three control final accuracies, evolved learning rate, hidden-node count,
and enabled-connection count. Development metrics are diagnostic; the decision
uses the sealed cohort once, after evolution ends.

## Decision rule

Call v0 successful only if the treatment's sealed result satisfies all of:

1. final symbol accuracy is at least 0.75;
2. adaptation gain over the pre-learning probe is at least 0.30;
3. plasticity-off final accuracy is at most 0.35;
4. permuted-reward final accuracy is at most 0.35;
5. reset-each-attempt final accuracy is at most 0.40.

The fixed-topology arm is a mechanistic comparison, not an additional success
gate. With one evolutionary seed, a pass establishes a working existence proof,
not robustness; a later multi-seed replication would be required.

## Compute contract and commands

```bash
cargo build -p cli --release

./target/release/cli hidden-string plan \
  --seed 17 --population 64 --generations 100

./target/release/cli hidden-string plan \
  --seed 17 --population 64 --generations 100 \
  --param add_connection_probability=0 \
  --param add_node_probability=0

./target/release/cli hidden-string \
  --seed 17 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-v0/treatment

./target/release/cli hidden-string \
  --seed 17 --population 64 --generations 100 \
  --param add_connection_probability=0 \
  --param add_node_probability=0 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-adaptation-v0/fixed-topology
```

Each arm evaluates 6,400 genomes, 64 training targets per genome, two rollout
replicates per target, 32 attempts per lifetime, and four action decisions per
attempt: 104,857,600 rewarded training decisions per arm, plus frozen probes
and development evaluation of each generation winner.

## Result

Both preregistered arms completed. The treatment passed all five sealed-cohort
gates:

| Measurement | Treatment | Fixed topology | Treatment gate |
|---|---:|---:|---:|
| Pre-learning symbol accuracy | 0.2500 | 0.2500 | — |
| Final symbol accuracy | 0.7793 | 0.4531 | >= 0.75 |
| Adaptation gain | +0.5293 | +0.2031 | >= 0.30 |
| Final exact-string rate | 0.3984 | 0.0000 | — |
| Learning AUC | 0.5732 | 0.3398 | — |
| Plasticity-off final accuracy | 0.2500 | 0.2500 | <= 0.35 |
| Permuted-reward final accuracy | 0.0859 | 0.1797 | <= 0.35 |
| Reset-each-attempt final accuracy | 0.3594 | 0.2539 | <= 0.40 |

Treatment sealed probe accuracy after `0, 1, 2, 4, 8, 16, 32` attempts was
`0.2500, 0.3457, 0.4355, 0.5410, 0.6211, 0.7168, 0.7793`. Exact-string rate
rose from 0 to 0.3984. The final treatment genome evolved learning rate
0.6437, five hidden nodes, and 22 enabled connections. The fixed-topology
winner retained one hidden node and four enabled connections, with learning
rate 0.2831.

The treatment exceeded the fixed-topology arm by 0.3262 final symbol accuracy
and 0.3984 exact-string rate. Development results were consistent with sealed
results: treatment final accuracy 0.7656 versus 0.4707 fixed-topology.

Both arms used seed 17, population 64, and 100 generations exactly as
preregistered. The treatment took approximately 11.9 seconds and the
fixed-topology arm approximately 9.5 seconds in release mode. The persisted
artifact includes plans, JSONL progress, compressed typed results, analysis,
commands, source status, and SHA-256 checksums under
`artifacts/research/runs/completed/2026-07-17-hidden-string-adaptation-v0/`.

Validation passed with `cargo check --workspace`, warning-as-error workspace
Clippy, and `cargo test --workspace` (24 tests passed). No new tests were added.

## Interpretation and next decision

This is a positive v0 result: a zero-input brain learned a newly instantiated
four-symbol target during inference from immediate reward, and the effect
failed in the intended causal controls. The monotonic frozen probe trajectory
shows adaptation across the lifetime rather than an inherited final policy.

The structural arm comparison supports the proposed mechanism. Topology growth
created enough recurrent position representation and action connectivity to
reach the success threshold; parameter evolution on the founder topology
learned modestly but plateaued far below it.

This does not yet establish robust continual learning. It is one evolutionary
seed, targets reset between independent lifetimes, feedback is dense and
immediate, and the learned state need not survive task switches. The next
experiment should replicate the treatment across at least three evolutionary
seeds before adding sequential targets with retained weights and interference
measurements.
