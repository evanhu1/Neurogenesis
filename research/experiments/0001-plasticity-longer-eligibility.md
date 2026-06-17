---
type: Experiment
title: Longer eligibility trace (credit-assignment window)
description: Raise eligibility_retention so coactivations contribute to weight change over a longer window.
iteration: 1
coordinator: plasticity-genome
agent: longer-eligibility
surface_area: plasticity-genome
base_ref: 70b7700
git_ref: autoresearch/exp-0001-plasticity-longer-eligibility
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: null, prey_consumption_rate: null, action_effectiveness: null, mi_sa: null, learning_slope: 0.000592 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { action_effectiveness: -0.0427, mi_sa: 0.0289, learning_slope: 0.001281 }
tags: [plasticity, eligibility, rejected]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
A longer eligibility trace gives coactivations more time to be reinforced,
strengthening consolidation and lifting learning_slope.

# Change
`eligibility_retention` default raised (seed genome, both copies).

# Result
**Crossed the learning_slope target: +0.000592** (seed 2026 went positive),
mi_sa +0.0289. BUT action_effectiveness −0.0427 (fails HOLD) and prey collapsed
~91% (inherent). Coordinator-reported (per-seed not separately re-validated; full
metrics frontmatter partially null — git_ref is the durable evidence).

# Learnings
A pure-Hebbian gene tweak CAN push slope positive, but at the cost of action
quality — consistent with the cohort pattern that learning gains trade against
action_effectiveness in this regime. Not promotable, but informative: the slope is
movable.

# Concerns
action_effectiveness regression; only coordinator-level numbers captured.

# Reproduce
```
git show c71075f ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show c71075f`
