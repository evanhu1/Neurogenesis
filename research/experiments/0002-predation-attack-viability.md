---
type: Experiment
title: Attack viability (ATTACK_DAMAGE_FRACTION 0.5→0.8)
description: Make predation attacks more lethal so a predator niche can establish.
iteration: 2
coordinator: predation
agent: attack-viability
surface_area: predation-mechanics
base_ref: a90244a
git_ref: autoresearch/exp-0002-predation-attack-viability
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0678, prey_consumption_rate: 0.0022, action_effectiveness: 0.5527, mi_sa: 0.0831, learning_slope: -0.000525 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: -0.0012, prey_consumption_rate: 0.0004, action_effectiveness: -0.0120, mi_sa: -0.0576, learning_slope: -0.000102 }
tags: [predation, attack, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Attacks rarely kill, so predation never establishes; more-lethal attacks let a
predator/prey dynamic form.

# Change
`ATTACK_DAMAGE_FRACTION` 0.5→0.8 in `sim-core/src/turn/mod.rs`.

# Result
Dead-end. Predation's share of deaths rose from ~12% to **17–31%** across seeds,
but **prey_consumption stayed flat (~0.0022)** — far below the 0.025 target. A
clean seed-7 A/B vs an unmodified control made it conclusive (prey 0.0019→0.0022;
aeff 0.5575→0.5195). action_effectiveness regressed on 2/4 seeds.

# Learnings
Decisive: **the bottleneck is corpse-EATING behavior, not kill production.** More
kills just recycle — organisms don't seek/eat the corpses. → motivates making the
ATTACK itself directly rewarding ([[experiments/0002-predation-kill-reward]]) and
[[findings/predation-needs-an-energy-conserving-kill-reward-not-richer-corpses]].

# Reproduce
`git checkout 55a3137; cargo build -p sim-cli --release`; per-seed `new`+`run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 55a3137`
