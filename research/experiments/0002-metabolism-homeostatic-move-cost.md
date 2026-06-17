---
type: Experiment
title: Homeostatic move cost (energy-dependent movement) — INTERRUPTED
description: Cheaper movement when energy is low; trending dead-end on action_effectiveness, not persisted (interrupted mid-confirm).
iteration: 2
coordinator: metabolism
agent: homeostatic-move-cost
surface_area: metabolism-lifecycle
base_ref: a90244a
git_ref: null
status: rejected
determinism: ok
seeds: [42, 123]
metrics: { plant_consumption_rate: 0.0648, prey_consumption_rate: 0.0020, action_effectiveness: 0.4506, mi_sa: 0.1209, learning_slope: -0.000350 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { note: "partial n=2 (seeds 42,123 only); action_effectiveness regressed (42: 0.515, 123: 0.387)" }
tags: [metabolism, move-cost, dead-end, partial, not-persisted]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Extending the homeostatic discount to `move_action_energy_cost` (cheaper movement
when energy is low) gives starving organisms affordable foraging → fewer
starvation deaths → higher learning_slope.

# Change
Energy-dependent discount on the move-cost charge site (mirroring the passive
homeostatic ramp). **NOT persisted** — the agent was still on its cross-seed
confirm (seeds 7/2026 pending) when the planner killed the runaway P2 explosion
sim with a blanket `pkill`, and the worktree was removed before a branch was cut.

# Result
Trending dead-end on the available seeds: **action_effectiveness regressed
(seed 42 0.515, seed 123 0.387 vs champion 0.528/0.631)**; learning_slope
inconsistent. The agent's own read: cheaper movement keeps low-energy organisms
alive but emitting *failing* actions → action_effectiveness dilutes — i.e.
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] again.

# Learnings
Same survival-softening↔action-quality trade as the other metabolism levers. Not
worth re-running. (Process note: blanket `pkill` of a runaway can clip a sibling's
unpersisted confirm — agents should persist the branch BEFORE the confirm, or the
planner should target only the runaway PID.)

# Reproduce
Change not persisted; would need re-implementation. Low priority (dead-end).
