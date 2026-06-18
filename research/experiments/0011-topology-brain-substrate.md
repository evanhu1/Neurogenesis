---
type: Experiment
title: Brain substrate increase (more synapses / neurons / vision range) — dilutes
description: More initial brain substrate slows convergence to an effective policy; the minimal brain has the highest action_effectiveness & mi_sa. Capability without a strong-enough reward dilutes — confirmed for topology too.
iteration: 11
coordinator: topology
agent: brain-substrate
surface_area: brain-topology
base_ref: 1dab610
git_ref: autoresearch/exp-0011-topology-brain-substrate
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0730, prey_consumption_rate: 0.00175, action_effectiveness: 0.5037, mi_sa: 0.0995, learning_slope: -0.000404 }
baseline_metrics: { plant_consumption_rate: 0.0786, prey_consumption_rate: 0.00235, action_effectiveness: 0.5435, mi_sa: 0.1952, learning_slope: -0.000487 }
delta: { plant_consumption_rate: -0.0056, prey_consumption_rate: -0.0006, action_effectiveness: -0.0398, mi_sa: -0.0957, learning_slope: 0.000083 }
tags: [brain-topology, dilution, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Give brains more substrate (synapses / inter neurons / vision range) so they can
evolve richer, higher-mi_sa policies and broaden the seed-7-heavy intelligence gain.

# Change
Swept seed-genome `num_synapses` {10,20,40} × `vision_distance` {5,7} ×
`num_neurons` {0,2,4}; best candidate n2s20. (Both seed_genome copies.)

# Result
Dead-end — DILUTES. The **minimal baseline (10/5/0) had the highest
action_effectiveness at both screen seeds and the highest mi_sa at seed 7**. With
only 4 action neurons, extra sensory→action synapses + inter neurons add noise
channels that within-life Hebbian learning + selection must overcome inside the
window → slower convergence → lower competence. Best candidate (n2s20) regresses
the champion on all pillars (aeff 0.5037, mi_sa 0.0995). vision_distance>5 also
costs metabolism (v7 worst). No explosion/collapse (homeostatic kept pops alive).

# Learnings
**Capability without a strong-enough reward dilutes competence** — now confirmed
for brain topology, matching the sensory-channel result
([[findings/perception-augmentation-dilutes-topology-the-best-arms-race-substrate-is-iter6]])
and the central law ([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]).
The minimal brain + the tuned three-factor rule (iter9) is the optimum on the
current architecture. With iter10, this is the **2nd consecutive dry iteration —
the frontier has plateaued**.

# Reproduce
`git checkout 8f4c001; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 8f4c001`
