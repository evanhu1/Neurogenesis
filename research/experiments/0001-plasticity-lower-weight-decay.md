---
type: Experiment
title: Lower plastic weight decay (consolidation)
description: PLASTIC_WEIGHT_DECAY 0.001→0.0002 so learned weights persist.
iteration: 1
coordinator: plasticity-genome
agent: lower-weight-decay
surface_area: plasticity-genome
base_ref: 70b7700
git_ref: autoresearch/exp-0001-plasticity-lower-weight-decay
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0673, prey_consumption_rate: 0.00185, action_effectiveness: 0.5170, mi_sa: 0.0977, learning_slope: -0.000562 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0074, prey_consumption_rate: -0.0199, action_effectiveness: -0.0396, mi_sa: 0.0022, learning_slope: 0.000127 }
tags: [plasticity, decay, dead-end]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
Weaker passive decay lets useful learned weights persist instead of washing out →
better consolidation, higher slope.

# Change
`PLASTIC_WEIGHT_DECAY` 0.001→0.0002 (5× lower) in `plasticity.rs`.

# Result
Slope +0.000127 (marginal), plant +0.0074. BUT action_effectiveness −0.0396 and
prey collapse. Mechanism: weaker decay lets stale/incidental correlations persist
→ noisier action selection; the rarer predation policy fails to consolidate
cleanly. Dead end.

# Learnings
Persistence without selectivity hurts action quality. → contributes to
[[findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime]].

# Concerns
AE regression consistent across seeds.

# Reproduce
```
git show 9312189 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 9312189`
