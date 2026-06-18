---
type: Experiment
title: Corpse energy retention 0.80→0.95
description: Make scavenging more rewarding by retaining more of a dead organism's energy in its corpse.
iteration: 2
coordinator: predation
agent: corpse-energy-retention
surface_area: predation-mechanics
base_ref: a90244a
git_ref: autoresearch/exp-0002-predation-corpse-energy-retention
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0644, prey_consumption_rate: 0.0017, action_effectiveness: 0.5316, mi_sa: 0.1300, learning_slope: -0.000583 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: -0.0046, prey_consumption_rate: -0.0001, action_effectiveness: -0.0331, mi_sa: -0.0107, learning_slope: -0.000160 }
tags: [predation, corpse, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Corpses are under-rewarding (80% retention) so scavenging doesn't pay; raising to
95% makes corpse-eating worthwhile and lifts prey_consumption.

# Change
`CORPSE_ENERGY_RETENTION` 0.80→0.95 in `sim-core/src/spawn/food.rs`.

# Result
Dead-end. A real but **transient early-sim scavenging boost** (seed-7 @50k:
≥3-corpse eaters 18→72, predation deaths +38%) that **fully decays by the
460k–500k scoring window** — at carrying capacity, standing corpses are sparse
(~50) and prey_consumption sits at baseline (~0.0017). No durable scavenger niche;
corpse-eating stays a minor opportunistic supplement to plant foraging.

# Learnings
Confirms [[findings/predation-needs-an-energy-conserving-kill-reward-not-richer-corpses]]:
making corpses *richer* doesn't create predators at equilibrium — the binding
constraint is that organisms don't evolve corpse-SEEKING behavior, not corpse
energy content.

# Reproduce
`git checkout 1c9e610; cargo build -p sim-cli --release`; per-seed `new`+`run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 1c9e610`
