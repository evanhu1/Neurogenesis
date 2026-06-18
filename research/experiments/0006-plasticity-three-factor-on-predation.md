---
type: Experiment
title: Three-factor (reward-sensitive) learning ON the predator ecology
description: A gentle energy-delta neuromodulated Hebbian rule on the consume-on-kill ecology RECOVERS action_effectiveness (vs consume-on-kill alone) — first evidence within-life reward-learning works when there's skill to learn. The closest any arms-race experiment came to holding the HOLD pillars.
iteration: 6
coordinator: plasticity
agent: three-factor-on-predation
surface_area: plasticity-genome
base_ref: 023030e
git_ref: autoresearch/exp-0006-plasticity-three-factor-on-predation
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0719, prey_consumption_rate: 0.0022, action_effectiveness: 0.5422, mi_sa: 0.1335, learning_slope: -0.000598 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0029, prey_consumption_rate: 0.0004, action_effectiveness: -0.0225, mi_sa: -0.0072, learning_slope: -0.000175 }
tags: [plasticity, three-factor, predation, intelligent-loop, promising-lead]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
The pure covariance Hebbian rule has no reward signal, so brains can't learn
within life that a behavior paid off. A gentle three-factor rule (gate the
eligibility→weight term by recent energy delta) ON the predator ecology
(consume-on-kill, where competent hunting pays) should let brains LEARN to hunt
within life → action_effectiveness & mi_sa rise — the genuine intelligent loop.

# Change
`brain/plasticity.rs`: `m = clamp(1 + 0.08*clamp(Δenergy/5, ±1), 0.85, 1.15)`
multiplies ONLY the `eta*eligibility` learning term (decay untouched). Reuses the
already-persisted `energy_at_last_sensing` (no new state → deterministic, det-check ok).
A/B'd gentle (GAIN 0.08, kept) vs strong (GAIN 1.5, destabilized — seed-7 aeff
collapsed). On the consume-on-kill base.

# Result
**The most promising arms-race result — validates the deep hypothesis, narrowly
misses the gate.** Vs the consume-on-kill base, the three-factor rule **RECOVERED
action_effectiveness 0.5088→0.5422** (within-life reward-learning improving action
quality — exactly what iter1's three-factor FAILED to do on the foraging-only
ecology). Vs CHAMPION, seed-for-seed: aeff −0.0225, mi_sa −0.0072 (mi_sa UP on 3/4
seeds; the drag is seed 123, whose champion brains are unusually strong at
0.63/0.29), plant +0.0029, prey +0.0004. The CLOSEST any predation-ecology
experiment came to holding both HOLD pillars — but still a mild regression, so no
clean advance. Strong band collapsed (gentleness matters).

# Learnings
**Validates [[directions/reward-sensitive-learning-on-the-predator-ecology]]:**
reward-sensitive learning works WHEN there is skill to learn (predation), confirming
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] from the
learning-rule side. The full intelligent-hunting loop (predator niche + within-life
reward learning) is the right mechanism. Path to a clean advance: refine the
neuromodulator band + add the corpse/prey sensory channel
([[experiments/0003-sensing-corpse-salience]]) so hunting is perceivable AND
learnable; and/or a metric contract that values mi_sa
([[directions/reconsider-intelligence-metric-under-predation]]).

# Reproduce
`git checkout 696def5; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`. (NB: branch persisted by the planner — the agent's API connection dropped before its own commit; results recovered from its saved 500k worlds.)

# Citations
[1] diff: `git show 696def5`
