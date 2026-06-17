---
type: Experiment
title: Lower food energy (the conflicting lever)
description: Lower food_energy — raises eat-rate but starves (per the metabolism/food_energy mechanism).
iteration: 1
coordinator: food-ecology
agent: lower-food-energy
surface_area: food-ecology
base_ref: 70b7700
git_ref: autoresearch/exp-0001-food-lower-food-energy
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: null, prey_consumption_rate: null, action_effectiveness: null, mi_sa: null, learning_slope: null }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: {}
tags: [food-ecology, food-energy, dead-end]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
Per `plant_rate ≈ metabolism / food_energy`, lowering food_energy raises the eat
rate (foraging). Included as the known conflicting lever for comparison.

# Change
`food_energy` lowered (config default, both copies).

# Result
Raised plant the most of any food experiment (seed 7 plant 0.0872 / AE 0.4431;
seed 42 plant 0.0943 / AE 0.5530) — but **craters action_effectiveness** (seed 7
to 0.443 vs 0.566 baseline). Confirms the long-standing mechanism: food_energy is
the dominant foraging lever and it conflicts with the others (starvation). Dead
end as a champion lever. → [[mechanisms/plant-rate-tracks-metabolism-over-food-energy]].

# Learnings
Reaching plant 0.10 via food_energy alone is possible but pays an unacceptable
action_effectiveness/learning cost. Foraging must come from findability +
homeostatic survival, not energy cuts.

# Concerns
HOLD-pillar collapse; partial planner data (coordinator-level).

# Reproduce
```
git show 31edac9 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 31edac9`
