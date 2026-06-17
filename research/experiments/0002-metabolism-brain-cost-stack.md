---
type: Experiment
title: Sub-linear (sqrt) neural+vision cost on the homeostatic base
description: Discount brain-size metabolic cost so bigger brains aren't a starvation risk; refutes the iter1 hope that the homeostatic base would let it hold action_effectiveness.
iteration: 2
coordinator: metabolism
agent: brain-cost-stack
surface_area: metabolism-lifecycle
base_ref: a90244a
git_ref: autoresearch/exp-0002-metabolism-brain-cost-stack
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0630, prey_consumption_rate: 0.0019, action_effectiveness: 0.4974, mi_sa: 0.1107, learning_slope: -0.000240 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: -0.0060, prey_consumption_rate: 0.0001, action_effectiveness: -0.0673, mi_sa: -0.0300, learning_slope: 0.000183 }
tags: [metabolism, brain-cost, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
sqrt-discounting neural+vision metabolic cost removes the starvation penalty for
bigger brains; stacked on the homeostatic champion it should lift learning_slope
AND (vs iter1's −0.023 aeff) this time HOLD action_effectiveness.

# Change
`organism_base_metabolic_cost`: `sqrt(inter+sensory+vision/3)` replaces the linear
sum (homeostatic factor + body-mass^0.75 kept). `metabolism.rs` only.

# Result
Dead-end — **hypothesis refuted.** Brains balloon as designed (synapses p50 13→26–59;
seed-2026 neurons p50 60) but action_effectiveness drops on **every** seed
(−0.067 mean). learning_slope improves on 3/4 (mean +0.000183) but stays negative
everywhere and regresses on seed 123; never near +0.0005.

# Learnings
The gentle homeostatic economy **tolerates rather than prunes** the bigger brains,
so they're noisier policies → lower action_effectiveness. Direct evidence for
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]: cheaper
brains without selection pressure ⇒ bloated, less-competent brains.

# Reproduce
`git checkout ed433a4; cargo build -p sim-cli --release`; per-seed `new`+`run-to 500000`+`pillars`.

# Citations
[1] diff: `git show ed433a4`
