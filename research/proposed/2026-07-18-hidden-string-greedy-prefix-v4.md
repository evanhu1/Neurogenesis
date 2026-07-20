# 2026-07-18-hidden-string-greedy-prefix-v4: ordered partial credit

Status: completed and rejected after the 500-generation replication. See the
[result](../archive/experiments/2026-07-18-hidden-string-greedy-prefix-v4-500g.md).

## Question

Can greedy longest-correct-prefix fitness provide dense enough evolutionary
credit without the probability-versus-argmax mismatch observed in v3?

## Selection score

The final frozen probe decodes every position with greedy argmax. For one
four-symbol target, let `k` be the number of consecutive correct outputs from
position zero before the first incorrect output. The case score is:

```text
greedy_prefix_score = k / 4
```

Examples:

- first symbol wrong: `0/4`;
- first correct, second wrong: `1/4`;
- first two correct, third wrong: `2/4`;
- first three correct, fourth wrong: `3/4`;
- all four correct: `4/4`.

A correct symbol after the first error receives no credit. Genome fitness is
the mean case score across the training targets and rollout seeds. There is no
unordered character-accuracy, soft-probability, AUC, or earlier-probe term in
fitness. Hard exact-string rate remains the normalized competence metric and
threshold gate.

This is a stepped, ordered signal rather than a continuous logit-level signal.
Its intended benefit is direct alignment with greedy sequence emission. Its
known risk is a curriculum basin that perfects earlier positions before later
ones; the complete-string gate and per-position diagnostics must detect that.

## Contract

- Task: `hidden_string_adaptation_v4`.
- Alphabet: `a` through `h`; target length four; no sensory input.
- Attempts: 32 with immediate signed per-character reward during adaptation.
- Training: 1,024 fixed targets x two rollout seeds and one final frozen probe.
- Development: 256 disjoint targets x one rollout, primary-only every 25
  generations and terminal, with probes `[0,8,16,32]`.
- Sealed: 1,024 disjoint targets x two rollouts, once for the terminal winner;
  shuffled-reward and reset-weights controls receive only the final probe.
- Population: 64; generations: 500; four evaluator workers per seed.
- Evolutionary seeds: 211, 307, and 401 for a matched comparison with v3.

## Decision rule

At least two of three seeds must satisfy all of:

1. sealed hard exact-string rate at least 0.20;
2. sealed hard exact-string gain at least 0.15 from the zero-attempt probe;
3. shuffled-reward hard exact-string rate at most 0.02;
4. reset-weights hard exact-string rate at most 0.02.

Efficiency is measured by evaluations to hard-exact thresholds, synapse
operations, and wall time. Prefix fitness itself is not a substitute for the
hard-exact gate.

## Commands

```bash
./target/release/cli hidden-string plan \
  --seed SEED --population 64 --generations 500 --workers 4

./target/release/cli hidden-string \
  --seed SEED --population 64 --generations 500 --workers 4 \
  --out-dir artifacts/research/runs/active/2026-07-18-hidden-string-greedy-prefix-v4-500g
```

## Result

The full replication has not run. A matched 100-generation diagnostic produced
sealed prefix scores of 0.1890, 0.2402, and 0.2516, with every seed still at its
historical maximum on generation 99 and controls remaining low. This justifies
the 500-generation replication but does not satisfy the hard-exact gate.
