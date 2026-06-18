---
type: Direction
title: Amplify the predation dynamic so it registers in the metrics
description: Energy-conserving kill rewards make a real but minority predator niche (prey ~0.004). Push predation to a larger share of the ecology so prey_consumption_rate can reach the target — without re-introducing the energy-creation explosion.
priority: high
status: open
surface_area: predation-mechanics
tags: [predation, arms-race, ecology]
timestamp: 2026-06-17T00:00:00Z
---

# Direction

[[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]]:
the predator niche is real but a minority (~25% of the population), so
`prey_consumption_rate` (normalized by total actions) tops out ~5–10× below 0.025.
To make the arms race *register* AND deepen the intelligence pressure, make
predation a larger share of the energy economy — candidates (each
energy-conserving; screen population stability on seed 7 BEFORE any 500k confirm):

- **Tilt the plant/predation balance**: modestly reduce plant availability *and*
  add the kill reward, so hunting is competitive with foraging (but watch the
  selection-pressure law — scarcity must reward *skill*, not just starve).
- **Stack the corpse sensory channel** ([[experiments/0003-sensing-corpse-salience]])
  with the reward so hunting is *efficient* (the planner's combo gate tests this).
- **Higher transfer fraction with a per-kill cooldown / handling time** so each
  kill yields more but predators can't trivially farm — keeps the niche bounded
  while raising its energy share.
- Consider whether prey should be *easier to catch when abundant* (density-
  dependent predation) so a predator niche scales with prey supply.

Pair with [[directions/reconsider-intelligence-metric-under-predation]] — the goal
is open-ended hunting intelligence, which `mi_sa` captures better than
`action_effectiveness` under predation.
