---
type: Finding
title: The prey_consumption_rate ≥ 0.025 target is structurally unreachable in a STABLE ecology
description: prey_rate = kills / total-actions; hitting 0.025 needs ~25 kills/tick on ~1,400 organisms (~2% of the population killed every tick) — no stable predator-prey system sustains that. Only iter2's population explosion ever reached it.
confidence: high
status: active
supported_by:
  - experiments/0004-predation-reliable
  - experiments/0003-predation-consume-on-kill
  - experiments/0002-predation-kill-reward
tags: [predation, metric, evaluation, ceiling, human-decision]
timestamp: 2026-06-17T00:00:00Z
---

# Finding

`prey_consumption_rate` is `prey_consumptions / total_actions` over the death
cohort. At the canonical stable population (~1,200–1,900), reaching **0.025** would
require on the order of **~25 corpse-eating/kill events per tick** — roughly 2% of
the entire population consumed *every tick*. No stable predator–prey ecology
sustains that turnover (the prey supply collapses first). Across four predation
experiments:

- consume-on-kill (strong niche): prey ~0.0026.
- reliable predation (8% hunters, 35% of deaths): prey ~0.0021.
- redistributive: prey ~0.0042.
- The ONLY run to exceed 0.025 (additive kill-reward, 0.036) did so via a
  **population explosion** — an *unstable* ecology, immediately disqualified.

So `prey ≥ 0.025` is effectively unreachable by any stable, energy-conserving
mechanism — it is a **metric-calibration problem, not a mechanism problem.** This
should be surfaced to the human who owns the eval contract
([[directions/reconsider-intelligence-metric-under-predation]]): either recalibrate
the predation target to an achievable stable value (e.g. ~0.004–0.006, which a
healthy predator niche reaches), or redefine prey_consumption to a per-capita /
encounter-normalized measure. Until then, the predation axis cannot advance the
champion regardless of mechanism, and predation work should be judged by
*goal-signals* (predator-niche size, emergent hunting behavior, mi_sa) rather than
this rate. Related: [[mechanisms/predation-is-encounter-limited]].
