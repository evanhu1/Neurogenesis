---
type: Experiment
title: Live-prey vision channel (matched perception) on the predator+learning base
description: The correctly-matched perception primitive (live-prey channel) ALSO dilutes brain topology and regresses intelligence — worse than the no-new-neurons iter6 base. Adding sensory channels costs more than the hunting signal repays.
iteration: 8
coordinator: sensing
agent: live-prey-perception
surface_area: brain-topology
base_ref: 696def5
git_ref: autoresearch/exp-0008-sensing-live-prey
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0729, prey_consumption_rate: 0.0023, action_effectiveness: 0.4991, mi_sa: 0.0838, learning_slope: -0.000128 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0039, prey_consumption_rate: 0.0005, action_effectiveness: -0.0656, mi_sa: -0.0569, learning_slope: 0.000295 }
tags: [sensing, brain-topology, perception-dilution, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
iter7 failed because the corpse channel mismatched the consume-on-kill reward. A
live-prey (organism) vision channel — matched to the reward (attack live prey) —
should be retained and let the three-factor rule consolidate a real hunting policy
→ action_eff & mi_sa reach champion (clean advance).

# Change
consume-on-kill + three-factor base + new `VisionChannel::Organism` firing for live
organisms (distinct from food). `sensing.rs` + wire types; topology auto-scaled.

# Result
**FAIL — worse than the base.** Even matched to the reward, the live-prey channel
regressed BOTH intelligence pillars below the iter6 base (0.5422/0.1335) AND
champion (0.5647/0.1407): aeff 0.4991, mi_sa 0.0838 (seeds 7 & 2026 collapsed to
0.43/0.46 aeff). plant +0.004, prey +0.0005, slope improved — but the HOLD pillars
fall. Determinism ok.

# Learnings
Decisive: **adding a sensory channel dilutes brain topology more than the hunting
signal repays — even when the perception is matched to the reward and within-life
learning is present** ([[findings/perception-augmentation-dilutes-topology-the-best-arms-race-substrate-is-iter6]]).
Both perception experiments (corpse iter7, live-prey iter8) regress intelligence;
the no-new-neurons three-factor (iter6) is the best arms-race substrate. The
mechanism space is exhausted — the binding constraint is the metric contract
([[directions/reconsider-intelligence-metric-under-predation]]).

# Reproduce
`git checkout 4ed69f0; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 4ed69f0`
