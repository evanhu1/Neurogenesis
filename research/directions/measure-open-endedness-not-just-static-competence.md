---
type: Direction
title: Measure open-endedness directly (sustained novelty), not just static competence
description: action_effectiveness is a static snapshot; the goal is OPEN-ENDED evolution (sustained novelty/complexity). Both the baseline and the spike champion converge by ~250k-1M under it. We need a first-class open-endedness signal — a sustained late-run competence slope and/or a behavioral/lineage novelty rate — to steer toward the actual goal and to gate non-stationary niches (the roamer).
priority: high
status: open
surface_area: eval-contract
supported_by: [findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]
tags: [eval-contract, open-endedness, measure-design, novelty]
timestamp: 2026-06-18T00:00:00Z
---

# The gap

The goal is **open-ended** evolution of intelligent brains. `action_effectiveness`
(the new headline) measures *static competence at the end of the run* — it cannot
tell a population that found a good policy and froze from one that keeps
discovering new ones. Both the baseline and the iter12 spike champion **converge**
(2nd-half aeff slope ~0; flat 500k→1M) under it
([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).
So the loop has no signal for the very property the goal names. We are optimizing
*level*, not *open-endedness*.

# Candidate open-endedness signals (read-only, post-hoc over existing worlds)

All computable from `pillars --granular` (50 intervals) and the per-organism reads
— **no engine change** to prototype:

1. **Sustained late competence slope** — regress action_effectiveness over the
   2nd half (250k–500k, or any late window). ~0 = converged; persistently
   positive = still improving = open-ended-ish. (Already used to gate iter12.)
   *Cheap, robust, the default open-endedness number.*
2. **Behavioral novelty rate** — the rate at which the population's
   action/sensory-conditioned policy keeps *changing* late in the run (e.g.
   interval-to-interval drift of the action distribution, or of mi_sa, that does
   NOT settle). Sustained churn = novelty; a flat line = convergence.
3. **Lineage turnover / genealogical novelty** — are new dominant lineages still
   arising late (generation depth still climbing, dominant-species share still
   turning over), or has one lineage fixed? `lineage` + generation distribution.
4. **Complexity growth** — does brain/genome complexity (synapse count, neuron
   count, distinct strategies) keep *increasing*, or saturate? (Caveat: a prior
   law says capability-without-reward DILUTES, so raw complexity is not
   automatically good — pair it with competence.)

# Why now

The roamer (`experiments/0013-ecology-roamer`, a non-stationary hazard) is the
first niche designed to NOT saturate. To know whether it actually delivers
open-endedness — rather than just a different fixed equilibrium — we must gate it
on a novelty signal, not only on the final action_effectiveness level. Signal (1)
is the minimum; (2)/(3) make it robust.

# Risk / caveat

A non-stationary environment can produce a *persistently nonzero* slope that is
just **churn around a fixed mean** (chasing a moving target, not getting better) —
that is open-ended *behavioral* novelty but not open-ended *competence growth*.
Distinguish: (a) sustained positive slope = improving; (b) sustained high
variance, ~0 slope = novelty-without-progress; (c) ~0 slope + low variance =
converged. The goal arguably wants (a), or (b) if "open-ended" means sustained
novelty regardless of a competence ceiling. This is a definitional call worth
making explicit (a human/contract decision like Dir2).

# Next action

Prototype signal (1) (done ad hoc) + (2) as a reusable read-only analysis over the
roamer's eval worlds, and report the open-endedness panel alongside the standard
pillars when gating iter13. If a signal cleanly separates converged vs open-ended
runs, promote it to a first-class gate criterion in `STATE.md`'s contract.
