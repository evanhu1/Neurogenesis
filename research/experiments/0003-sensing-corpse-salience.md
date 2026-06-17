---
type: Experiment
title: Corpse sensory salience (VisionChannel::Corpse)
description: A distinct sensory channel for corpses makes scavenging learnable — brains wire Corpse→Eat — but alone it's a weak/negative enabler (extra inputs dilute foraging without a reward).
iteration: 3
coordinator: sensing
agent: corpse-sensory-salience
surface_area: brain-topology
base_ref: a1d33b7
git_ref: autoresearch/exp-0003-sensing-corpse-salience
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0693, prey_consumption_rate: 0.0021, action_effectiveness: 0.5292, mi_sa: 0.0904, learning_slope: -0.000080 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0003, prey_consumption_rate: 0.0003, action_effectiveness: -0.0355, mi_sa: -0.0503, learning_slope: 0.000343 }
tags: [sensing, brain-topology, enabler]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Brains can't learn to seek corpses if they can't perceive them. Baseline vision
wiring is dominated by the plant (Green) channel; corpses share Red with spikes.
A distinct corpse channel makes scavenging learnable.

# Change
`sim-core/src/brain/sensing.rs` (+ wire types): new `VisionChannel::Corpse` fed by
`VisualProperties.corpse=1.0` only for `FoodKind::Corpse`; rides the existing per-ray
visual blending; topology auto-scales (sensory 18→21, 3 corpse receptors/ray);
seed-genome synapse sampling auto-includes the new presynaptic IDs.

# Result
**Mechanistically validated enabler**: after the change, evolved brains DO wire
the corpse channel — at 50k `Corpse→Eat` (w=1.49) / `Corpse→Attack` (w=1.18); at
500k, 59 corpse-channel synapses across top organisms with `Corpse→Eat` up to
w=1.37 — a learned scavenge reflex that did not exist before. BUT **alone it's
weak/negative**: prey gains noise-level (3/4 seeds up ~0.001–0.003), and
action_effectiveness −0.036 / mi_sa −0.050 (3 extra sensory inputs enlarge the
expression space and dilute the strongly-evolved foraging circuitry with no reward
to pay off scavenging). learning_slope improved (notably +0.0007 on seed 7).

# Learnings
The enabler works (corpse-seeking is now learnable) but needs a reward to pay off —
i.e. **combine with a predation/scavenge reward** (the planner gate-tests
salience + consume-on-kill). Adding capability without reward dilutes competence —
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]].

# Reproduce
`git checkout autoresearch/exp-0003-sensing-corpse-salience; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show autoresearch/exp-0003-sensing-corpse-salience`
