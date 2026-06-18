---
type: Finding
title: The lower-fertility-threshold foraging win does NOT hold on the homeostatic champion
description: Carried iter1 foraging lever raises plant modestly but breaks both HOLD pillars on the softer champion base — population grows and selection pressure drops.
confidence: high
status: active
supported_by:
  - experiments/0001-food-lower-fertility-threshold
tags: [food-ecology, foraging, gate, superseded]
timestamp: 2026-06-17T00:00:00Z
---

# Finding

`lower-fertility-threshold` (food_fertility_threshold 0.6→0.45) looked like a clean
foraging win in iteration 1 (n=3, on the harsher pre-homeostatic base: plant
+0.0144, HOLD pillars held). **Planner-gated onto the homeostatic champion
(a90244a), it FAILS**: seed-for-seed (n=4 common survivors) plant +0.0036 but
**action_effectiveness −0.0645 (seed 42 → 0.378) and mi_sa −0.097**, population
+~50%.

The homeostatic base is already softer; adding more food tips the ecology past the
slack threshold where selection for competent foraging collapses — a direct case
of [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]. This
closes the iteration-1 [[directions/food-findability-with-hold-pillar-guard]]
avenue *as a stack on the homeostatic champion*: more standing food de-pressures
brains. Foraging toward 0.10 must come from rewarding foraging *skill*, not adding
abundance. (Gate evidence: candidate per-seed plant 0.0721/0.0738/0.0702/0.0742;
aeff 0.5195/0.3777/0.5551/0.5488; vs champion aeff 0.5575/0.5276/0.6311/0.5428.)
