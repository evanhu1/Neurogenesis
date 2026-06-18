---
type: DeadEnd
title: Flat passive-metabolism cut (uniform 0.005→0.004)
description: Same learning/survival win as the structural variants but regresses action_effectiveness in every seed.
reason: A uniform cost cut removes selective pressure indiscriminately; structural variants get the same learning/survival gain while sparing action quality.
ruled_out_by:
  - experiments/0001-metabolism-lower-passive-cost
tags: [metabolism, dead-end]
timestamp: 2026-06-16T00:00:00Z
---

# Dead end

[[experiments/0001-metabolism-lower-passive-cost]] lifted learning_slope and saved
seed 2026, but regressed action_effectiveness −0.048 in EVERY seed (the worst of
the metabolism cohort). The structural energy-economy changes (homeostatic,
brain-cost-discount) achieve the same learning/survival gains while holding action
quality, so a blunt flat cut is dominated. Do not re-explore flat
`passive_metabolism_cost_per_unit` reductions as a champion lever.
