---
type: Experiment
title: Wider fertile regions (fertility noise scale)
description: Adjust fertility noise so fertile regions are larger/denser (findability).
iteration: 1
coordinator: food-ecology
agent: wider-fertile-regions
surface_area: food-ecology
base_ref: 70b7700
git_ref: autoresearch/exp-0001-food-wider-fertile-regions
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
Larger contiguous fertile regions → easier-to-find food (findability).

# Change
`FoodEcologyPolicy` fertility noise/jitter adjusted (sim-config, both copies).

# Result
Seed 7: plant 0.0666 / AE 0.4936 / mi_sa 0.1266 / slope −0.000264 / pop 1736
(healthy, seed 2026 also rescued). In line with the findability cohort: modest
plant gain, AE depressed. Full per-seed not captured before the food coordinator
was stopped for budget — **branch is the durable evidence.** Not promotable.

# Learnings
Same family as faster-regrowth / lower-fertility-threshold. →
[[directions/food-findability-with-hold-pillar-guard]].

# Concerns
Partial planner data; AE depression.

# Reproduce
```
git show 533a271 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 533a271`
