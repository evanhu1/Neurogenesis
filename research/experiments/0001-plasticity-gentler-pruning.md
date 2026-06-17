---
type: Experiment
title: Gentler synapse pruning
description: Prune interval 10→50 and eligibility-multiplier 2.0→6.0 so weak-but-active synapses survive.
iteration: 1
coordinator: plasticity-genome
agent: gentler-pruning
surface_area: plasticity-genome
base_ref: 70b7700
git_ref: autoresearch/exp-0001-plasticity-gentler-pruning
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0676, prey_consumption_rate: 0.00168, action_effectiveness: 0.5106, mi_sa: 0.0877, learning_slope: -0.000433 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0077, prey_consumption_rate: -0.0200, action_effectiveness: -0.0460, mi_sa: -0.0078, learning_slope: 0.000256 }
tags: [plasticity, pruning, dead-end]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
Aggressive pruning destroys consolidating structure; keeping weak-but-active
synapses longer should improve learning.

# Change
`SYNAPSE_PRUNE_INTERVAL_TICKS` 10→50, `PRUNE_ELIGIBILITY_MULTIPLIER` 2.0→6.0.

# Result
Slope +0.000256, plant +0.0077, BUT action_effectiveness −0.046 AND mi_sa −0.0078
(the only iteration-1 experiment regressing BOTH HOLD pillars). Keeping more
transient/noisy edges dilutes action selectivity. Dead end.

# Learnings
Pruning is doing useful work for action quality; loosening it trades AE/mi_sa for
a small slope gain. → [[findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime]].

# Concerns
Regresses both HOLD pillars.

# Reproduce
```
git show 88881db ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 88881db`
