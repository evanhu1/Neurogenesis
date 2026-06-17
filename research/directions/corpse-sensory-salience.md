---
type: Direction
title: Corpse/prey sensory salience — let brains perceive what to hunt
description: Organisms don't evolve corpse-seeking partly because corpses/prey may be weakly represented in the sensory input; make them salient so hunting/scavenging is learnable.
priority: medium
status: open
surface_area: brain-topology
tags: [predation, sensing, brain, enabler]
timestamp: 2026-06-17T00:00:00Z
---

# Direction

[[experiments/0002-predation-attack-viability]] and
[[experiments/0002-predation-corpse-energy-retention]] showed organisms don't seek
corpses even when corpses are abundant/rich — they can't or don't *perceive* the
opportunity. Check `sim-core/src/brain/sensing.rs` (and the visual map): how are
corpses (`FoodKind::Corpse`) and prey organisms encoded in sensory input vs plant
food? If corpses share/dilute the plant-food channel or are low-salience, brains
can't learn a distinct hunt/scavenge policy. Give corpses/prey a distinct,
salient sensory channel so corpse-seeking is *learnable*. This is an **enabler**
for [[directions/redistributive-kill-reward]] (a reward only helps if the behavior
it rewards is perceivable/learnable). Surface area: brain sensing/topology — keep
disjoint from the predation-mechanics (reward) change so each is attributable.
