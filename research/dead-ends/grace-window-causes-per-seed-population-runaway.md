---
type: DeadEnd
title: Starvation grace window (3-tick) as-is
description: Lifts slope and mi_sa but lets a single seed's population explode and action_effectiveness collapse.
reason: A multi-tick grace window can let one seed over-populate (seed 42 → pop 2543, action_eff → 0.277), dragging cross-seed action_effectiveness down and risking eval-time blowup.
ruled_out_by:
  - experiments/0001-metabolism-soft-death-grace
tags: [metabolism, lifecycle, dead-end, instability]
timestamp: 2026-06-16T00:00:00Z
---

# Dead end

[[experiments/0001-metabolism-soft-death-grace]] gave a strong mi_sa gain but
introduced **per-seed population instability** (seed 42 exploded to 2543 with
action_effectiveness collapsing to 0.277). Grace windows risk runaway populations
on individual seeds. The homeostatic champion dominates it on stability. Ruled out
**as-is**; a tighter window (≤2 ticks) or a health penalty *during* grace could be
revisited, but only if homeostatic tuning ([[directions/tune-homeostatic-ramp]])
is exhausted first.
