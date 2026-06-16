---
type: Experiment
title: <short title>
description: <one-line summary>
iteration: <int>
coordinator: <surface-area>
agent: <id>
surface_area: <lever-family>
base_ref: <sha it forked from>
git_ref: autoresearch/exp-<iter4>-<coordinator>-<id>
status: built | build-failed | evaluated | promoted | rejected
determinism: ok | broken | not-checked
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0, prey_consumption_rate: 0, action_effectiveness: 0, mi_sa: 0, learning_slope: 0 }
baseline_metrics: { plant_consumption_rate: 0, prey_consumption_rate: 0, action_effectiveness: 0, mi_sa: 0, learning_slope: 0 }
delta: { plant_consumption_rate: 0, prey_consumption_rate: 0, action_effectiveness: 0, mi_sa: 0, learning_slope: 0 }
tags: []
timestamp: <ISO 8601>
---

# Hypothesis
<what change, why it should move which metric, via what mechanism>

# Change
<what code moved (files, the idea); keep it to one surface area>

# Result
<cross-seed mean ± spread table; delta vs base_ref; eval-time note>

# Learnings
<what actually moved which metric and the likely mechanism>

# Concerns
<confounds, instability, determinism, eval-time cost, suspected artifacts>

# Citations
[1] [sweep result](/../artifacts/runs/sweep-<...>.json)
[2] [diff](<git_ref>)
