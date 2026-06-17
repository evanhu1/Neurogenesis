---
type: Experiment
title: Sub-linear (sqrt) neural+vision metabolic cost
description: Discount brain-size & vision metabolic cost via sqrt scaling so bigger brains aren't a starvation risk.
iteration: 1
coordinator: metabolism-lifecycle
agent: brain-cost-discount
surface_area: metabolism-lifecycle
base_ref: 70b7700
git_ref: autoresearch/exp-0001-metabolism-brain-cost-discount
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0668, prey_consumption_rate: 0.00258, action_effectiveness: 0.5336, mi_sa: 0.0909, learning_slope: -0.000238 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0069, prey_consumption_rate: -0.0191, action_effectiveness: -0.0230, mi_sa: -0.0046, learning_slope: 0.000451 }
tags: [metabolism, death-spiral, rejected]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
Punishing brain/vision metabolic cost makes investing in a bigger brain a
starvation risk; discounting it (sqrt scaling of the neuron+vision cost terms in
`metabolism.rs`) should let smarter phenotypes survive and lift learning.

# Change
`sim-core/src/metabolism.rs`: sqrt (sub-linear) scaling of the inter+sensory
neuron count and vision-distance cost contributions.

# Result
Best raw learning_slope of the cohort (−0.000238, seed 42 went positive +0.000196)
and rescued seed 2026 (n→4), no explosion. BUT regressed action_effectiveness
(−0.023) and slightly mi_sa (−0.0046) — fails the HOLD gate. Prey collapsed
(inherent).

# Learnings
Confirms the death-spiral lever from a different angle. The mi_sa regression and
AE cost make it inferior to homeostatic-metabolism, BUT it is **mechanistically
orthogonal** (cost-vs-brain coupling vs cost-vs-energy coupling) and gave the best
slope — a candidate to STACK on the champion with HOLD-pillar care.
See [[directions/stack-brain-cost-discount-on-homeostatic]].

# Concerns
action_effectiveness/mi_sa regression. n=3→n=4 confound on the cross-seed mean.

# Reproduce
```
git show 1312d9f ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 1312d9f`
