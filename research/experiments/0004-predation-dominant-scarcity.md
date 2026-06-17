---
type: Experiment
title: Predation-dominant via plant scarcity (consume-on-kill + scarcer plant)
description: Reduce plant so organisms must hunt — backfires: predation is encounter-limited, scarcity lowers density and SUPPRESSES predation.
iteration: 4
coordinator: predation
agent: predation-dominant-scarcity
surface_area: predation-x-food
base_ref: 023030e
git_ref: autoresearch/exp-0004-predation-dominant-scarcity
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0694, prey_consumption_rate: 0.0018, action_effectiveness: 0.5111, mi_sa: 0.1190, learning_slope: -0.000287 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { action_effectiveness: -0.0536, mi_sa: -0.0217, prey_consumption_rate: 0.0000 }
tags: [predation, food, encounter-limited, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
On consume-on-kill, reduce plant availability so foraging can't sustain →
organisms must hunt → predation dominates → prey↑ and hunting skill selected.

# Change
consume-on-kill base + raise `food_fertility_threshold` (swept 0.65–0.85) /
`food_regrowth_interval`=400 (scarcer plants), both config copies. Chose 0.65 (mildest viable).

# Result
**Hypothesis falsified — clean null.** Predation is **ENCOUNTER-LIMITED**: a kill
fires only when a predator is co-located with prey, so predation is
*density*-driven, not hunger-driven. Scarcity lowers carrying capacity → lower
density → fewer encounters → **LESS predation** (seed-7 predation share fell
monotonically 17.1%→3.7% as threshold rose 0.60→0.85; predations/tick collapsed
10×). prey stayed ~0.0018; action_effectiveness did NOT recover (0.51); populations
fell below champion on every seed.

# Learnings
→ [[mechanisms/predation-is-encounter-limited]]. Scarcity and predation are
anti-correlated through density; no regime satisfies both. To grow predation you
need a **density-preserving or encounter-AMPLIFYING** lever (attack reach,
prey-seeking pursuit), not scarcity.

# Reproduce
`git checkout 514dd82; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 514dd82`
