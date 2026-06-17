---
type: Experiment
title: Homeostatic ramp tuning (threshold/floor sweep)
description: Deeper/earlier passive-cost downregulation to push learning_slope; the champion's 5.0/0.5 ramp is already near the knee.
iteration: 2
coordinator: metabolism
agent: homeostatic-ramp-tune
surface_area: metabolism-lifecycle
base_ref: a90244a
git_ref: autoresearch/exp-0002-metabolism-homeostatic-ramp-tune
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5147, mi_sa: 0.1000, learning_slope: -0.000337 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { action_effectiveness: -0.0500, mi_sa: -0.0407, learning_slope: 0.000086 }
tags: [metabolism, homeostatic, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
A deeper/earlier downregulation (lower floor, higher threshold) gives starving
organisms more recovery time and lifts learning_slope.

# Change
Swept threshold∈{5,8,12}×floor∈{0.5,0.35,0.25} in `metabolism.rs` (A/B at seed-7
250k); committed the best mean-slope cell **12.0/0.35**.

# Result
Dead-end. Best-mean-slope cell gives slope −0.000337 (+0.000086) but the gain is
**carried entirely by a seed-7 outlier (+0.000101); 3/4 seeds are worse**, and it
breaks both HOLD pillars (aeff −0.050, mi_sa −0.041). The 250k screen ranked cells
differently from the 500k window — slope **noise**. `eco`: starvation stays
**57–72% of deaths** even with the ramp.

# Learnings
The champion's 5.0/0.5 ramp is at a sensible knee — further softening trades the
HOLD pillars without robustly buying learning_slope. Two durable notes:
**learning_slope is seed-noise-dominated at n=4** (small deltas untrustworthy →
the `sufficient_n`/per-seed-consistency guard matters), and softening passive cost
is a case of [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]].
Metabolism-lifecycle is drying up as a learning_slope lever.

# Reproduce
`git checkout 4f5d0c0; cargo build -p sim-cli --release`; per-seed `new`+`run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 4f5d0c0`
