---
type: Finding
title: Predation needs an energy-conserving kill reward — not richer or more-numerous corpses
description: Corpse-richness and kill-lethality don't move prey_consumption (no scavenging niche forms at equilibrium); a direct kill reward does — but must conserve energy or the ecology explodes.
confidence: high
status: active
supported_by:
  - experiments/0002-predation-corpse-energy-retention
  - experiments/0002-predation-attack-viability
  - experiments/0002-predation-kill-reward
tags: [predation, corpse, prey]
timestamp: 2026-06-17T00:00:00Z
---

# Finding

Three predation experiments triangulate the prey_consumption wall:

1. **Richer corpses** (retention 0.80→0.95) → only a transient early-sim
   scavenging boost; decays to baseline by the scored window. No durable niche.
2. **More lethal attacks** (damage 0.5→0.8) → predation rises to 17–31% of deaths
   but prey_consumption stays flat (~0.0022). **More kills ≠ more eating.**
3. **Direct kill reward** (attacker gains prey energy) → prey_consumption hits
   0.036 (> target) — *the lever works* — but as built it was *additive* (minted
   free energy), exploding the population.

Conclusion: the binding constraint is that organisms **don't evolve corpse-seeking
behavior at carrying capacity**; making corpses richer/more-numerous can't fix
that. Rewarding the *attack itself* does create predators — but the reward must be
**redistributive (energy-conserving)**: move energy out of the corpse into the
attacker, not add new energy. See [[directions/redistributive-kill-reward]]. A
complementary enabler is giving brains a corpse/prey **sensory cue** so they *can*
learn to hunt/scavenge ([[directions/corpse-sensory-salience]]).
