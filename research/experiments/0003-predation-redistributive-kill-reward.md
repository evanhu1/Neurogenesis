---
type: Experiment
title: Redistributive (energy-conserving) kill reward
description: On a kill, attacker gains frac×prey energy and the corpse keeps the remainder — induces a real predator niche with NO explosion, but prey stays far below target and intelligence regresses.
iteration: 3
coordinator: predation
agent: redistributive-kill-reward
surface_area: predation-mechanics
base_ref: a1d33b7
git_ref: autoresearch/exp-0003-predation-redistributive-kill-reward
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0691, prey_consumption_rate: 0.0042, action_effectiveness: 0.5085, mi_sa: 0.1035, learning_slope: -0.000425 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0001, prey_consumption_rate: 0.0024, action_effectiveness: -0.0562, mi_sa: -0.0372, learning_slope: -0.000002 }
tags: [predation, kill-reward, arms-race, breakthrough]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
The iter2 additive kill-reward proved the lever but minted free energy (explosion).
An energy-conserving version (attacker takes from the corpse) should induce a
predator niche WITHOUT the runaway.

# Change
`resolve_attack_damage` (commit.rs): on a kill, attacker gains `frac*prey_energy`;
corpse spawns with `(1-frac)*prey_energy` (then the usual 0.80 decay on the
residual). `KILL_ENERGY_TRANSFER_FRACTION` const (mod.rs); A/B'd frac∈{0.25,0.5,0.75},
chose 0.5. Total post-kill energy = `prey*(0.8+0.2*frac) ≤ prey` — strictly conserved.

# Result
Energy conservation works perfectly — **NO explosion at any fraction** (pops
1124–1753, ≤ champion). **A genuine predator niche FORMED on every seed**:
predations 26–44% of deaths, ~25% of the population with ≥1 kill, top hunters
57–100 kills with *evolved contextual brains* (ContactAhead→Eat w=1.5, positive
Attack logit). prey_consumption rose ~2–2.6× (to 0.0042 mean). **But fails the
bar:** prey still **~5× below the 0.025 target** (hunting minority, normalized by
total foraging actions), and action_effectiveness −0.056 / mi_sa −0.037 on the
seed-for-seed comparison (predation kills organisms younger → lower death-cohort
competence). learning_slope mixed (positive on seed 2026).

# Learnings
The arms race is **inducible and bounded** — emergent hunting brains, a real
open-ended-evolution signature. The metric, not the mechanism, is the wall:
`prey_consumption_rate = prey_consumptions/total_actions` barely moves for a
hunting *minority*. See [[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]].
One existing test (`lethal_attack_spawns_corpse_food_without_feeding_attacker`)
now fails by design (left for the human maintainer per AGENTS.md).

# Reproduce
`git checkout 504144b; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 504144b`
