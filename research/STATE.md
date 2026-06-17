---
type: Research State
title: STATE — autonomous research dashboard
description: The planner's live, compacted working memory. Read this first every session; it is sufficient to resume without rereading the archive.
tags: [autoresearch, state, dashboard]
timestamp: 2026-06-16T00:00:00Z
---

# STATE

**Read this first.** Distilled, compacted thread of knowledge — enough to resume
without rereading every experiment. Dip into `experiments/`, `findings/`, or
`log.md` only for a *specific* fact. The planner **rewrites this file at the end
of every iteration**.

## Goal & targets

Lift the three lagging competence axes to threshold, **cross-seed mean** on the
canonical eval (seeds 7,42,123,2026, 500k ticks). Raw metrics (no [0,1] layer).

| axis | target | baseline (ef6f9bb) | gap |
|---|---|---|---|
| foraging  | `plant_consumption_rate ≥ 0.10`  | 0.0599 | −0.040 |
| predation | `prey_consumption_rate  ≥ 0.025` | 0.0217 | −0.003 (close) |
| learning  | `learning_slope         ≥ +0.0005` | **−0.000689** | the wall (negative) |
| intelligence | hold `action_effectiveness` (0.5566) & `mi_sa` (0.0955) | — | hold |

## Current best program

- **`git_ref`:** `autoresearch/best` @ `ef6f9bb` (== `main`; no experiments
  accepted yet). See `best-program.md`.
- **Metrics:** above, cross-seed mean over **n=3** survivors.

## ⚠ The central obstacle (baseline diagnostic, iteration 0)

- **Seed 2026 collapses to full extinction** (pop 0 @ 500k). The cross-seed mean
  is over 3 seeds; **n=4 (sustain all seeds) is itself a target.**
- Survivors (7/42/123) sit at pop **1316 / 2052 / 1605** — *far below* the world
  cap (62 500 cells). The baseline is in a **scarcity/collapse regime near a
  tipping boundary**, NOT the population-explosion regime. ⇒ there is headroom to
  soften the energy economy (lower metabolism / more findable food) *before*
  hitting the known explosion trap — but overshoot is the trap, so watch run-time
  & population.
- **learning_slope is negative on every survivor** → within-life action success
  declines with age = the **starvation death-spiral** (organisms accumulate
  failed actions in their dying stretch). Reducing starvation is the lever.
- **Predation is corpse-eating** (`prey_consumptions_count` increments on eating
  a `FoodKind::Corpse`). **Starvation deaths leave NO corpse** (energy ≤ 0); only
  age/predation/spike deaths spawn corpses (×0.80 energy retention). ⇒ shifting
  deaths away from starvation should raise *both* learning_slope AND prey rate.
- **Mechanism prior holds:** `plant_rate ≈ metabolism / food_energy` — lower
  food_energy raises foraging but starves (hurts learning); they conflict.
  *Findability* (plant abundance / regrowth speed) is the hoped-for decoupling
  lever: more eating events without per-bite starvation pressure.

## Frontier (per surface area)

| surface area | best lever known | marginal direction | status |
|---|---|---|---|
| food ecology | food_energy (foraging, conflicts w/ learning) | raise *findability* not cut energy | iter1 active |
| metabolism / lifecycle | metabolism (collapse boundary) | soften death-spiral; sustain seed 2026 | iter1 active |
| plasticity genome | pure covariance Hebbian (no reward) | give the rule a success signal w/o re-adding actor-critic | iter1 active |
| corpse / predation mechanics | corpses from non-starvation deaths | attack/corpse tuning | reserved → iter2 |
| brain topology | — | — | open |

## Established mechanisms (durable laws)

Config-level priors carried from prior manual work; re-confirm opportunistically.

- **`plant_consumption_rate ≈ metabolism / food_energy`** (confidence: medium).
- **`food_energy` is the dominant foraging lever and conflicts with learning**
  (confidence: medium).
- **Low metabolism is an eval-time trap** — population explodes, blows time
  budget. *But baseline is on the opposite (collapse) side*, so there is room to
  move before hitting it (confidence: high).
- **Negative learning slope = starvation death-spiral**, not plasticity per se
  (confidence: medium).

## Active directions (untapped alpha)

- **Sustain seed 2026 / move off the collapse boundary** (food-ecology +
  metabolism). Highest leverage: fixes n, learning_slope, and likely prey.
- **Findability decoupling** — raise plant abundance/regrowth so foraging rises
  without per-bite energy cuts (food-ecology).
- **Three-factor / neuromodulated Hebbian** — modulate the covariance update by
  an already-present self-supervised signal (energy/health delta) WITHOUT a value
  function or TD (i.e. not a re-introduction of the removed actor-critic)
  (plasticity).

## Dead ends

- _None recorded yet._ (Very-low metabolism population explosion is a known trap,
  not yet hit in this loop.)

## Bundle census

- experiments: 0 · findings: 0 · directions: 3 (in STATE, not yet filed) ·
  mechanisms: 4 (seeded) · dead-ends: 0
- Last iteration: 0 (baseline recorded; coordinators launching).

## Next actions

1. **Iteration 1** — 3 disjoint code-change coordinators, run sequentially:
   **metabolism-lifecycle**, **plasticity-genome**, **food-ecology**. Each spawns
   ~4 worktree-isolated research agents (fork `ef6f9bb`, one minimal change,
   build, determinism-check, persist `autoresearch/exp-*`, confirm cross-seed via
   the per-seed loop). base_ref = `ef6f9bb`.
2. Gate winners (build ✓ + determinism ✓ + cross-seed no-regression ✓), combine
   non-conflicting, re-gate after each, advance `autoresearch/best`.
3. File OKF concepts, update `best-program.md`, rewrite this file, append `log.md`.
4. Reserve **corpse/predation mechanics** for iteration 2.
