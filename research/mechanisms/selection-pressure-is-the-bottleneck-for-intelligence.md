---
type: Mechanism
title: Selection pressure — not survival ease — is the bottleneck for intelligent brains
description: Every lever that makes life easier (more food, softer metabolism, cheaper brains, free kill-energy) grows/relaxes the population and DEGRADES action_effectiveness & mi_sa. Intelligence needs sustained pressure, not abundance.
confidence: high
supported_by:
  - experiments/0002-metabolism-brain-cost-stack
  - experiments/0002-metabolism-homeostatic-ramp-tune
  - experiments/0002-metabolism-homeostatic-move-cost
  - experiments/0002-predation-kill-reward
  - findings/lower-fertility-threshold-does-not-stack-on-the-homeostatic-base
  - findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime
tags: [intelligence, selection-pressure, action-effectiveness, central-insight]
timestamp: 2026-06-17T00:00:00Z
---

# Mechanism

**The decisive iteration-2 result.** Across six independent levers, anything that
adds *slack* to the resource/survival constraint degrades the intelligence pillars
(`action_effectiveness`, `mi_sa`):

- More food (lower fertility threshold) → population +~50%, **aeff −0.064, mi_sa −0.097**.
- Softer passive metabolism (deeper homeostatic ramp) → **aeff −0.050, mi_sa −0.041**.
- Cheaper movement when low-energy → aeff regressed (seeds 42/123).
- Cheaper brains (sqrt neural cost) → brains balloon but **aeff −0.067 on every seed**.
- Free kill-energy (additive) → population explosion, **aeff 0.565→0.239, mi_sa 0.141→0.008**.

The unifying mechanism: when survival is easy, an organism can persist (and
reproduce) with a *sloppy* action policy — there is no longer selection for
converting sensory state into the *right* action, so evolved policies get noisier
and `action_effectiveness`/`mi_sa` fall. The current learning rule (pure
covariance Hebbian, no reward) does not supply within-life skill pressure, so
selection pressure is the only force maintaining competence — and abundance
removes it.

**Strategic consequence (dead-on for "open-ended evolution of intelligent
brains"):** intelligence cannot be bought with ease. The targets that *raise*
foraging/survival are in partial **tension** with *holding* intelligence. The way
to raise a metric WITHOUT de-pressuring brains is to route the gain through a
**skill that must be earned** — most promising: a **balanced predator–prey arms
race** where energy flows through *competent hunting* (and competent evasion),
which rewards skill rather than relaxing the constraint. That is why
[[directions/redistributive-kill-reward]] is the top lead: it adds a
competence-rewarding niche without minting free energy.

This also predicts the iteration-1
[[findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime]]
was a special case of the same law.

# Citations
Per-seed deltas embedded in the supporting experiments (iteration 2), all
authoritative or planner-gated; champion baseline a90244a.
