---
type: Direction
title: Tune the homeostatic metabolism ramp (threshold / floor)
description: The champion used threshold=5.0, floor=0.5×; sweep for more learning_slope headroom without destabilizing population.
priority: medium
status: open
surface_area: metabolism-lifecycle
tags: [metabolism, champion, tuning]
timestamp: 2026-06-16T00:00:00Z
---

# Direction

The champion [[experiments/0001-metabolism-homeostatic-metabolism]] hard-codes
HOMEOSTATIC_THRESHOLD=5.0 and a 0.5× floor. These were a first guess. Sweep
threshold ∈ {3,5,8,12} × floor ∈ {0.3,0.5,0.7} (cheap — one structural constant,
no other pillar touched) to find the point that maximizes learning_slope while
keeping populations stable (watch the explosion trap). Likely additional headroom
in the same mechanism. Consider promoting the constants to config/genome so they
can co-evolve.
