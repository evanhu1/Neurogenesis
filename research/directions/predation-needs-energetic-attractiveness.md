---
type: Direction
title: Make predation energetically attractive (corpse/attack mechanics) — the only path to the prey target
description: prey_consumption can't be won by reducing death pressure; it needs active hunting to pay off. Engine-level (iteration 2).
priority: high
status: open
surface_area: corpse-predation-mechanics
tags: [predation, corpse, engine, iter2]
timestamp: 2026-06-16T00:00:00Z
---

# Direction

[[mechanisms/predation-inversely-coupled-to-population-health]] shows the prey
target is unreachable by improving survival (that *lowers* corpse scavenging). The
only path is to make **predation a deliberate, rewarding strategy** rather than
desperation scavenging. Candidate engine changes (corpse/attack surface area):

- Increase the energy payoff of attack-kills / corpse consumption (raise
  `CORPSE_ENERGY_RETENTION`, or give attackers a direct energy transfer on kill).
- Make corpses more persistent / findable (slower decay) so hunting pays off.
- Reduce attack cost or raise attack success so predation is a viable niche.
- Add a sensory cue for prey/corpses so brains can learn to hunt.

This is engine code (a slower loop) and is the reserved **iteration-2** surface
area. It is the dominant remaining gap: prey is the only target where the
iteration-1 champion regressed (to ~0.002, target 0.025).
