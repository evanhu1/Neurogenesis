---
type: Finding
title: Learning-slope gains trade against action_effectiveness (except homeostatic metabolism)
description: 11 of 12 iteration-1 experiments that lifted learning_slope regressed action_effectiveness; only homeostatic metabolism broke the trade.
confidence: medium
status: active
supported_by:
  - experiments/0001-metabolism-homeostatic-metabolism
  - experiments/0001-plasticity-three-factor-energy
  - experiments/0001-plasticity-longer-eligibility
  - experiments/0001-plasticity-lower-weight-decay
  - experiments/0001-plasticity-gentler-pruning
  - experiments/0001-food-faster-regrowth
  - experiments/0001-food-lower-food-energy
tags: [learning, action-effectiveness, hold-pillars]
timestamp: 2026-06-16T00:00:00Z
---

# Finding

Across iteration 1, nearly every change that lifted learning_slope **regressed
action_effectiveness** (the HOLD pillar): all four plasticity changes
(−0.040 to −0.057), the flat metabolism cut (−0.048), the grace window (−0.033),
and the food findability/energy levers (−0.03 to −0.08). The plasticity changes
that loosen selectivity (slower decay, gentler pruning, longer eligibility) let
stale/incidental correlations persist → noisier action selection. Food findability
appears to reduce selective pressure on action precision.

**The sole exception is homeostatic metabolism**
([[experiments/0001-metabolism-homeostatic-metabolism]]), which improved
action_effectiveness (+0.0155 seed-for-seed) *while* lifting learning_slope —
because it changes *survival* (more ticks to act) without touching the learning
rule or food selectivity.

Implication: the keystone is best advanced via the **energy/survival economy**,
not by loosening the plasticity rule. Plasticity changes need a mechanism that
lifts slope *without* degrading action precision (e.g. a well-tuned three-factor
neuromodulator — [[directions/tune-three-factor-neuromodulation-band]]). Treat
this as medium-confidence: it may be specific to the current death-pressure
regime; re-test on the new (homeostatic) champion baseline.
