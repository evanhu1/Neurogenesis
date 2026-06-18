---
type: Experiment
title: Starvation grace window (3-tick soft death)
description: Allow ~3 ticks at energy≤0 before death so one bad tick isn't instantly fatal.
iteration: 1
coordinator: metabolism-lifecycle
agent: soft-death-grace
surface_area: metabolism-lifecycle
base_ref: 70b7700
git_ref: autoresearch/exp-0001-metabolism-soft-death-grace
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0624, prey_consumption_rate: 0.00164, action_effectiveness: 0.5240, mi_sa: 0.1506, learning_slope: -0.000320 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0025, prey_consumption_rate: -0.0201, action_effectiveness: -0.0326, mi_sa: 0.0551, learning_slope: 0.000369 }
tags: [metabolism, death-spiral, rejected, instability]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
A short grace window at energy≤0 (with the organism still able to act) lets a
single bad tick be recovered, breaking the spiral.

# Change
`sim-core/src/turn/lifecycle.rs`: 3-tick starvation grace before marking a
starved organism dead.

# Result
learning_slope +0.000369 and the biggest mi_sa gain of the cohort (+0.0551), all
4 seeds survive. BUT action_effectiveness −0.0326, driven by a **seed-42
population explosion to 2543** with AE collapsing to 0.277 — grace windows let a
single seed over-populate and dilute action quality. Fails HOLD + instability.

# Learnings
Grace windows risk per-seed runaway. homeostatic-metabolism dominates it on
stability. If revisited: tighter window (2 ticks) or a health penalty during
grace. → [[dead-ends/grace-window-causes-per-seed-population-runaway]].

# Concerns
Seed-42 instability (pop 2543). Could worsen at longer horizons.

# Reproduce
```
git show 88215a3 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 88215a3`
