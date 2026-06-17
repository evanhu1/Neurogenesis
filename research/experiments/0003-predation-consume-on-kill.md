---
type: Experiment
title: Consume-on-kill (the kill is the meal, no corpse)
description: A predation kill directly feeds the attacker (prey×0.80, no corpse) — strongest predator niche; raises mi_sa where predation is strongest, but prey still short and action_effectiveness slips.
iteration: 3
coordinator: predation
agent: consume-on-kill
surface_area: predation-mechanics
base_ref: a1d33b7
git_ref: autoresearch/exp-0003-predation-consume-on-kill
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0792, prey_consumption_rate: 0.0026, action_effectiveness: 0.5088, mi_sa: 0.1728, learning_slope: -0.000556 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0102, prey_consumption_rate: 0.0008, action_effectiveness: -0.0559, mi_sa: 0.0321, learning_slope: -0.000133 }
tags: [predation, consume-on-kill, arms-race, mi_sa]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
Organisms already attack (iter2); the kill only leaves a rarely-eaten corpse.
Make the kill *itself* the meal → existing attack behavior converts directly to
prey_consumption (energy-conserving, no corpse to find).

# Change
`resolve_attack_damage` (commit.rs): on a kill, `predator.energy += prey.energy*0.80`
(reuse CORPSE_ENERGY_RETENTION), `prey_consumptions_count++`, and spawn **no corpse**
for that kill (other death paths still drop corpses). `kill_organism` gained a
`spawn_corpse` flag. Energy conserved (= what eating the corpse would transfer).

# Result
**Strongest predator niche of the cohort** — predation 43–48% of deaths on seeds
42/123/2026; pure-predator phenotypes (plant=0, prey=10–16). NO explosion (pops
*lower* than champion on 3/4 — predation thins the population). Notable:
**mi_sa ROSE where the niche is strongest (123: 0.184, 2026: 0.360 vs 0.141)** and
**plant rose** (mean 0.079) — predation selects for information-rich behavior on
those lineages. BUT prey_consumption mean only ~0.0026 (fell on seed 7 to 0.0009 —
suppressing the kill-corpse removed a corpse source low-predation seeds fed on);
action_effectiveness slipped everywhere (−0.056). Far short of the 0.025 target.

# Learnings
Confirms the predator niche is inducible and can *raise* the mi_sa intelligence
metric — but `action_effectiveness` (death-cohort competence) regresses under
predation, and `prey_consumption_rate` resists the target (hunting minority /
removed corpse source). →
[[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]].
Test `lethal_attack_spawns_corpse_food_without_feeding_attacker` fails by design.

# Reproduce
`git checkout 023030e; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 023030e`
