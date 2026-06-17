---
type: Direction
title: Tune the three-factor neuromodulation band (gentler, per-tick delta)
description: The energy-delta neuromodulator lifted slope but destabilized action_effectiveness; retry with a gentler band and a cleaner delta.
priority: medium
status: open
surface_area: plasticity-genome
tags: [plasticity, three-factor, neuromodulation]
timestamp: 2026-06-16T00:00:00Z
---

# Direction

[[experiments/0001-plasticity-three-factor-energy]] is the most novel learning
lever (a self-supervised energy-delta neuromodulator, no value function/TD). It
improved learning_slope but cratered action_effectiveness (seed 42 −0.139,
unstable). The mechanism is promising; the hyperparameters are wrong. Retry:

- Gentler band: GAIN ≪ 1 and/or a narrower multiplier range (e.g. [0.8, 1.2])
  so it nudges rather than flips the update.
- Compute Δenergy against the **previous tick** rather than `energy_at_last_sensing`
  (cleaner per-tick credit), or use an EMA of Δenergy.
- Modulate by *sign* only (consolidate on gain, freeze on loss) instead of a
  scaled magnitude.
- Re-baseline on the homeostatic champion (the trade in
  [[findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime]]
  may relax once survival is healthier).
