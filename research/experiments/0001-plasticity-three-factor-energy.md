---
type: Experiment
title: Three-factor (energy-delta) neuromodulated Hebbian update
description: Gate the eligibility→weight term by a bounded self-supervised energy-delta neuromodulator (no value function / TD).
iteration: 1
coordinator: plasticity-genome
agent: three-factor-energy
surface_area: plasticity-genome
base_ref: 70b7700
git_ref: autoresearch/exp-0001-plasticity-three-factor-energy
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0629, prey_consumption_rate: 0.00165, action_effectiveness: 0.4975, mi_sa: 0.1306, learning_slope: -0.000436 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0030, prey_consumption_rate: -0.0201, action_effectiveness: -0.0591, mi_sa: 0.0351, learning_slope: 0.000253 }
tags: [plasticity, three-factor, neuromodulation, rejected]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
A pure covariance Hebbian rule has no success signal, so within-life action
success need not improve. Gating only the `eta·eligibility` learning term by a
bounded neuromodulator derived from the organism's recent net energy change
(consolidate coactivations that preceded a gain, dampen/reverse those before a
loss) should make learned policies track success — WITHOUT a value function or TD
(not a re-introduction of the removed actor-critic).

# Change
`sim-core/src/brain/plasticity.rs` only: `energy_delta_neuromodulator(organism)` =
`clamp(1 + GAIN·clamp(Δenergy,±CLAMP)/CLAMP, [0.2,1.8])` using already-persisted
`energy_at_last_sensing`; multiplies only the learning term (decay term untouched).

# Result
**Authoritative planner re-measurement (per-seed):** 7 = slope −0.000618 / AE
0.5572; 42 = −0.000018 / AE **0.3728**; 123 = −0.000537 / AE 0.5682; 2026 =
−0.000570 / AE 0.4916 (survives). Clean Δ(7/42/123): slope +0.000298 but
**action_effectiveness −0.0572 (seed 42 −0.139)** — fails the HOLD gate. The slope
gain is entirely seed-42-driven and coupled to that seed's AE collapse + pop rise
(2162) → a degenerate/unstable regime, not genuine learning.

# Learnings
The novel three-factor idea is promising in principle but its hyperparameters
(GAIN=1, CLAMP=3, floor 0.2 / cap 1.8) destabilize action quality. Worth re-trying
with a gentler band and a per-tick delta computed against the previous tick (not
last sensing). → [[directions/tune-three-factor-neuromodulation-band]].

# Concerns
Severe AE regression; seed-42 instability. The energy-delta proxy may be too
coarse / too aggressive.

# Reproduce
```
git show f554726 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show f554726`
