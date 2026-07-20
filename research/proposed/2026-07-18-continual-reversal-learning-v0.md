# Continual reversal learning v0

Status: implemented, initial diagnostic run
Slug: 2026-07-18-continual-reversal-learning-v0
Date: 2026-07-18

## Hypothesis

The canonical recurrent/plastic brain can evolve a general within-lifetime
strategy that repeatedly discovers, exploits, detects, and relearns an
evaluator-randomized rewarded action without episodic state resets.

## Contract

- Alphabet/actions: `a` through `h`; there are no sensory inputs and `end` is
  disabled.
- One lifetime starts from the inherited brain. Recurrent dynamics and runtime
  weights persist for all 512 ticks; there are no attempts or internal reset
  events.
- One hidden action is rewarded at a time. Regimes last a uniformly sampled
  32--96 ticks, then switch to a different hidden action.
- Panels contain groups of eight rotated lifetimes. They are exactly action
  balanced, and all rotations in a group use identical action-sampling draws.
  Target identity can therefore affect behavior only through reward.
- Correct actions receive `+1`; incorrect actions receive `-1/7`. The ordinary
  immediate hidden-to-action plasticity rule is the only reward pathway.
- Training/development/sealed panels contain 64/64/256 independently generated
  lifetimes. Development runs every 25 generations and at the end; sealed runs
  once for the final winner.
- Fitness is mean correct-action rate, equivalently one minus normalized
  dynamic regret. Report a 32-tick lifetime curve, a 32-tick post-reversal
  curve, first-correct latency, and latency to a four-correct-action streak.
- Sealed controls disable plasticity or reset recurrent dynamics after every
  tick while preserving runtime weights.

## Success rule

Do not claim continual adaptation from a single score. A confirmation requires
fresh evolutionary seeds and a fresh sealed panel with high cumulative accuracy,
shorter recovery latency than the plasticity-off control, and a reported
continuous-versus-reset-dynamics comparison. A reset-equivalent result is a
valid result: it means the present task is solved by synaptic adaptation rather
than recurrent continuity.

## Commands

```bash
cargo build -p cli --release
CLI=./target/release/cli

$CLI continual-reversal plan --seed 101 --population 64 --generations 100
$CLI continual-reversal --seed 101 --population 64 --generations 100 --workers 4 \
  --out-dir artifacts/research/runs/active/2026-07-18-continual-reversal-learning-v0
```

The basic reaction and basic memory tasks are saturated references, not
competitors for this objective.

## Initial diagnostic

Seed 101, population 64, 100 generations, four workers, and the default
64/64/256 lifetime panels completed in 11.2 seconds. The final winner reached
25.97% development and 24.65% sealed accuracy, above the 12.50% exact-balanced
chance and plasticity-off baselines. The sealed reset-dynamics-each-tick control
reached 23.52%, a 1.13-point deficit from continuous dynamics. This is evidence
that the evaluator and its causal controls work; it is not yet a replication or
a claim of robust continual adaptation.

Artifact:
`artifacts/research/runs/active/2026-07-18-continual-reversal-learning-v0/neat-continual-reversal-1784437639905-54206.json.zst`.

The dynamics-reset-each-tick control described above is historical. It was
removed from the live evaluator after v1 because it erased recurrent activity,
previous-action state, and prediction-error state together rather than
isolating one mechanism.
