# 2026-07-18-hidden-string-soft-exact-v3: smooth exact-string selection

Status: completed; robust-discovery hypothesis rejected. See the
[experiment record](../archive/experiments/2026-07-18-hidden-string-soft-exact-v3-500g.md).

## Question

Can a smooth sequence-level surrogate make NEAT a more efficient and robust
discoverer of within-lifetime hidden-string learning while preserving hard
exact emission as the competence criterion?

## Motivation

The v2 selection score was the mean hard exact-string rate at the final frozen
probe. Each target/rollout case therefore contributed either zero or one, with
no distinction between zero, one, two, or three wrong output positions. With
eight symbols and a length-four target, a uniform policy succeeds with
probability `1 / 8^4 = 1 / 4096`. The 2,048 training cases contain only 0.5
expected random exact successes. This makes early mutations fitness-neutral
even when they substantially improve the probability of the correct string.

Doubling the alphabet from four to eight made this signal 16 times sparser. It
reduced simple character-level shortcuts, but it also plausibly contributed to
slow, seed-fragile discovery. It does not by itself explain a plateau after
hard exact rate is already high enough to yield hundreds of successful cases.

## Selection score

At the final reward-free probe, let `p_i(x_i)` be the eight-action softmax
probability assigned to the correct symbol at position `i`. One target/rollout
case receives the soft exact-string score

```text
soft_exact(x) = product(i = 1..4, p_i(x_i))
```

Genome fitness is the arithmetic mean of `soft_exact(x)` across all training
target/rollout cases. This is the policy's probability of sampling the entire
correct string under the frozen probe distribution. It changes continuously
with output margins and remains multiplicative across positions, so one weak
position suppresses the complete-string score.

There is no character-accuracy term, learning-curve AUC, or earlier-probe term
in fitness. Hard greedy exact-string rate remains separately reported as
normalized fitness and is used for competence thresholds. Character accuracy
remains diagnostic only.

The learned runtime weights still depend on sampled actions and rewards, so the
end-to-end evolutionary landscape is only piecewise smooth. This correction
removes the final argmax/exact-count discontinuity; it does not make the whole
learning trajectory differentiable.

## Contract

- Task: `hidden_string_adaptation_v3`.
- Alphabet: `a` through `h`; target length four; no sensory input.
- Attempts: 32 with immediate signed per-character reward during adaptation.
- Training: 1,024 fixed targets x two rollout seeds and one frozen probe after
  attempt 32.
- Development: 256 disjoint targets x one rollout, primary-only every 25
  generations and on the terminal generation, with probes `[0,8,16,32]`.
- Sealed: 1,024 disjoint targets x two rollouts, once for the terminal winner;
  shuffled-reward and reset-weights controls receive only the final probe.
- Panels remain disjoint, hash-shuffled, repeat-pattern matched, and exactly
  symbol-balanced at every position.
- Population: 64; generations: 500; evaluation workers: four per seed when
  running three seeds concurrently.
- Evolutionary seeds: 211, 307, and 401 for direct continuity with the v2
  diagnostic series.

## Decision rule

The competence/persistence gate is unchanged. At least two of three seeds must
satisfy all of:

1. sealed hard exact-string rate at least 0.20;
2. sealed hard exact-string gain at least 0.15 from the zero-attempt probe;
3. shuffled-reward hard exact-string rate at most 0.02;
4. reset-weights hard exact-string rate at most 0.02.

Evolutionary efficiency is assessed separately using first-generation and
cumulative genome evaluations to hard exact thresholds `0.2`, `0.5`, `0.8`,
and `0.9`, plus synapse operations and wall time. The principal comparison is
whether discovery is earlier and less seed-dependent than the hard-exact v2
diagnostics; it is not enough for soft fitness alone to improve.

## Commands

```bash
./target/release/cli hidden-string plan \
  --seed SEED --population 64 --generations 500 --workers 4

./target/release/cli hidden-string \
  --seed SEED --population 64 --generations 500 --workers 4 \
  --out-dir artifacts/research/runs/completed/2026-07-18-hidden-string-soft-exact-v3-500g
```

## Result

Only seed 211 passed the complete sealed gate: sealed hard exact rates were
50.00%, 1.22%, and 7.52% for seeds 211, 307, and 401. The smoother score and
parallel evaluator reduced concurrent batch wall time by 6.55x, but changed
which seed failed rather than producing robust discovery. Full results are in
the [experiment record](../archive/experiments/2026-07-18-hidden-string-soft-exact-v3-500g.md).
