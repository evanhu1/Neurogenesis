---
type: Mechanism
title: Predation (corpse-eating) is inversely coupled to population health
description: prey_consumption_rate reflects a stressed, corpse-rich ecosystem; any intervention that improves survival/learning yields a healthier population that scavenges fewer corpses.
confidence: high
supported_by:
  - experiments/0001-metabolism-homeostatic-metabolism
  - experiments/0001-metabolism-lower-passive-cost
  - experiments/0001-metabolism-brain-cost-discount
  - experiments/0001-plasticity-lower-weight-decay
  - experiments/0001-plasticity-gentler-pruning
tags: [predation, corpse, ecology, trade-off]
timestamp: 2026-06-16T00:00:00Z
---

# Mechanism

`prey_consumption_rate` counts **eating a `FoodKind::Corpse`** (not attack kills).
**Starvation deaths leave no corpse** (energy ≤ 0); only age / predation / spike
deaths spawn corpses (×0.80 energy retention). Therefore corpse supply — and
prey_consumption — is high precisely when the population is **stressed and dying**.

Empirically, **all 8 experiments that improved survival/learning collapsed
prey_consumption ~10× (0.0217 → ~0.002) on EVERY seed**, including the
always-surviving 7/42/123 (verified seed-for-seed against authoritative baseline
worlds; the baseline 0.0217 was re-confirmed stable and tight across seeds). The
collapse is identical across mechanistically-distinct changes (metabolism AND
plasticity), none of which touched corpse/attack code — so it is **not a
treatment-specific or composition artifact**; it is the healthier population
scavenging fewer corpses.

**Consequence (the key strategic law):** the predation target
(`prey_consumption ≥ 0.025`, i.e. MORE corpse-eating) is in **fundamental tension**
with the foraging/learning/survival targets (a well-fed, non-starving population).
It **cannot be co-satisfied by reducing death pressure** — that lowers it.
Reaching the predation target requires making predation *energetically attractive*
as a deliberate strategy (active hunting / corpse-reward mechanics), i.e. the
corpse/predation-mechanics surface area, NOT the energy economy.
See [[directions/predation-needs-energetic-attractiveness]].

# Citations
Per-seed prey deltas embedded in the supporting experiments; baseline re-confirmed
from saved 500k worlds (iteration-0 diagnostic in [[best-program]]).
