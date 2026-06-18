---
type: Experiment
title: Lower fertility threshold (more fertile cells)
description: Lower food_fertility_threshold so more cells can grow plants (denser food map).
iteration: 1
coordinator: food-ecology
agent: lower-fertility-threshold
surface_area: food-ecology
base_ref: 70b7700
git_ref: autoresearch/exp-0001-food-lower-fertility-threshold
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: null, prey_consumption_rate: null, action_effectiveness: null, mi_sa: null, learning_slope: null }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: {}
tags: [food-ecology, findability, rejected, partial-data]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
More fertile cells → denser plant distribution → more encounters (findability).

# Change
`food_fertility_threshold` lowered (config default + FoodEcologyPolicy, both copies).

# Result
Rescued seed 2026 and behaved "in line with the other findability levers" per the
coordinator (modest plant gain, action_effectiveness depressed like the others).
Full per-seed metrics not separately captured by the planner before the food
coordinator was stopped for budget — **the branch is the durable evidence; re-run
to regenerate.** Not promotable (same HOLD-pillar tension as the cohort).

# Learnings
Same family as faster-regrowth / wider-fertile-regions; findability raises foraging
modestly at an action_effectiveness cost. → [[directions/food-findability-with-hold-pillar-guard]].

# Concerns
Partial planner data (coordinator-level only).

# Reproduce
```
git show ce08871 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show ce08871`
