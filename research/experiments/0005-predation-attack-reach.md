---
type: Experiment
title: Attack reach (encounter amplification, R=2)
description: Extending the attack radius grows the predator niche (encounters↑) but DESTROYS intelligence — free reach removes the hunting skill.
iteration: 5
coordinator: predation
agent: attack-reach
surface_area: predation-mechanics
base_ref: 023030e
git_ref: autoresearch/exp-0005-predation-attack-reach
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0773, prey_consumption_rate: 0.0053, action_effectiveness: 0.3109, mi_sa: 0.0320, learning_slope: 0.000208 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0083, prey_consumption_rate: 0.0035, action_effectiveness: -0.2538, mi_sa: -0.1087, learning_slope: 0.000631 }
tags: [predation, encounter, de-skilling, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Predation is encounter-limited; extending the attack to prey within radius R=2
amplifies encounters → a richer arms race that rewards spatial positioning →
mi_sa↑, action_effectiveness held.

# Change
consume-on-kill base + attack reach R=2 (strike a deterministically-chosen prey
within 2 hexes, not just the adjacent cell). Swept R∈{1,2,3}. `turn/` attack path.

# Result
**Hypothesis falsified.** The niche DID grow (predation ~33% of deaths, prey 0.0053
— highest of any experiment; learning_slope even went positive on seed 7) — but
**both intelligence pillars COLLAPSED**: action_effectiveness 0.51→0.31, mi_sa
0.17→0.032 (~81%). R=3 was near-collapse. Why: (1) reach lets attacks land WITHOUT
aiming/positioning → **the hunting skill is removed** → attacks fire freely, most
fail the size-ratio roll, flooding the action stream with failures; (2) higher
predation churns lineages before competence accumulates (high-churn short-lived,
injection-propped, unstable).

# Learnings
Encounter amplification by a **free buff** *de-skills* hunting → intelligence
collapses — a direct case of
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] and the
de-skilling note now in [[mechanisms/predation-is-encounter-limited]]. The only
encounter amplification that would RAISE intelligence is **skilled pursuit** (brain
navigates to perceived prey, attacks adjacent) — which must be LEARNED, pointing to
a reward-sensitive plasticity rule on the predator ecology
([[directions/reward-sensitive-learning-on-the-predator-ecology]]).

# Reproduce
`git checkout cb0c793; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show cb0c793`
