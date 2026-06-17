---
type: Experiment
title: Reliable predation (consume-on-kill + success floor + higher damage)
description: More reliable attacks doubled the predator niche but prey still ~10× short — the prey-rate target is structurally unreachable in a stable ecology.
iteration: 4
coordinator: predation
agent: predation-reliable
surface_area: predation-mechanics
base_ref: 023030e
git_ref: autoresearch/exp-0004-predation-reliable
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0700, prey_consumption_rate: 0.0021, action_effectiveness: 0.4931, mi_sa: 0.1003, learning_slope: -0.000527 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0010, prey_consumption_rate: 0.0003, action_effectiveness: -0.0716, mi_sa: -0.0404, learning_slope: -0.000104 }
tags: [predation, reliability, structural-ceiling, screen-further]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
On consume-on-kill, make attacks more reliable so a larger fraction of the
population hunts successfully → prey↑, hunting effectiveness selected.

# Change
`resolve_attack_damage`/mod.rs: `PREDATION_SUCCESS_FLOOR=0.40`
(`predation_success = 0.40 + 0.60*size_ratio`) + `ATTACK_DAMAGE_FRACTION` 0.5→0.75.
Screened 5 strengths; one-shot (dmg=1.0) bred FEWER specialists + crushed mi_sa (rejected).

# Result
Best predator niche yet — dedicated hunters <0.2%→~8% of population, predation
share of deaths 17%→~35%; prey beat champion on 3/4 seeds; mi_sa/learning_slope
held NEAR champion. No prey crash. **But prey still ~0.002 — ~10× below the 0.025
target**, and action_effectiveness dipped to 0.49. **Structural ceiling:** prey_rate
= kills/total-actions; 0.025 would need ~25 kills/tick on ~1,400 organisms (≈2% of
the population killed every tick) — unsustainable in any stable ecology.

# Learnings
→ [[findings/prey-consumption-target-is-structurally-unreachable-in-a-stable-ecology]].
Reliability grows the niche but can't break the metric ceiling. The remaining lever
is encounter amplification ([[mechanisms/predation-is-encounter-limited]],
[[directions/amplify-the-predation-dynamic]]). Pre-existing test
`lethal_attack_spawns_corpse_food_without_feeding_attacker` fails identically on the
consume-on-kill base (not caused by this change).

# Reproduce
`git checkout 7360d46; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 7360d46`
