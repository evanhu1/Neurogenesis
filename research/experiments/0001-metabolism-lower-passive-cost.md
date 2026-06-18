---
type: Experiment
title: Lower flat passive metabolism cost (0.005→0.004)
description: Blunt uniform cut of passive_metabolism_cost_per_unit.
iteration: 1
coordinator: metabolism-lifecycle
agent: lower-passive-cost
surface_area: metabolism-lifecycle
base_ref: 70b7700
git_ref: autoresearch/exp-0001-metabolism-lower-passive-cost
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0662, prey_consumption_rate: 0.00233, action_effectiveness: 0.5087, mi_sa: 0.1084, learning_slope: -0.000308 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0063, prey_consumption_rate: -0.0194, action_effectiveness: -0.0479, mi_sa: 0.0129, learning_slope: 0.000381 }
tags: [metabolism, death-spiral, dead-end]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
A uniform cut to passive metabolic cost reduces death pressure → less starvation
→ better slope.

# Change
`passive_metabolism_cost_per_unit` 0.005→0.004 (config default, both copies).

# Result
learning_slope +0.000381, all 4 seeds survive. BUT the **worst action_effectiveness
regression of the cohort (−0.0479, in EVERY seed)** — a uniform cost cut removes
selective pressure indiscriminately. Dead end vs the structural variants.

# Learnings
The structural changes (homeostatic, brain-discount) achieve the same
learning/survival win while sparing action quality; a blunt flat cut does not.
→ [[dead-ends/flat-metabolism-cut-removes-action-selective-pressure]].

# Concerns
AE regression across all seeds (real, not composition).

# Reproduce
```
git show 60cacf8 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 60cacf8`
