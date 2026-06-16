---
type: Research State
title: STATE — autonomous research dashboard
description: The planner's live, compacted working memory. Read this first every session; it is sufficient to resume without rereading the archive.
tags: [autoresearch, state, dashboard]
timestamp: 2026-06-16T00:00:00Z
---

# STATE

**Read this first.** It is the distilled, compacted thread of knowledge — enough
to resume the research loop without rereading every experiment. Dip into
`experiments/`, `findings/`, or `log.md` only for a *specific* fact. The planner
**rewrites this file at the end of every iteration** (see `../.claude/skills/autoresearch/SKILL.md`).

## Goal & targets

Lift the three lagging competence axes to their thresholds, **cross-seed mean**
on the canonical eval (seeds 7,42,123,2026, 500k ticks). Metrics are raw (the
[0,1] pillar interpretation was removed).

| target | value to reach | current (best-program) | gap |
|---|---|---|---|
| foraging  | `plant_consumption_rate ≥ 0.10`  | _tbd_ | _tbd_ |
| predation | `prey_consumption_rate  ≥ 0.025` | _tbd_ | _tbd_ |
| learning  | `learning_slope         ≥ +0.0005` | _tbd_ | _tbd_ |
| intelligence | `action_effectiveness` & `mi_sa` already healthy — hold, don't regress | _tbd_ | — |

## Current best program

- **`git_ref`:** `autoresearch/best` (not yet created — first iteration forks from `main`).
- **Metrics:** _tbd — run the baseline eval to seed this._
- See `best-program.md` for the exact sha + lineage.

## Frontier (per surface area)

| surface area | best lever known | marginal direction | status |
|---|---|---|---|
| food ecology | — | — | open |
| metabolism / lifecycle | — | — | open |
| corpse / predation mechanics | — | — | open (engine-level; needs code) |
| plasticity genome | — | — | open |
| brain topology | — | — | open |

## Established mechanisms (durable laws)

Carried over from prior manual experimentation; **config-level, so they survive
the engine state on `main`.** Treat as priors, re-confirm opportunistically.

- **`plant_consumption_rate ≈ metabolism / food_energy`.** Foraging rate tracks
  how often an organism must eat. (confidence: medium)
- **`food_energy` is the dominant foraging lever, and it conflicts with
  learning** — low energy raises foraging but starves organisms (negative
  learning slope); high energy buffers learning but depresses foraging. They do
  not co-satisfy at the thresholds with food_energy alone. (confidence: medium)
- **Low metabolism is an eval-time trap** — population explodes toward the world
  cap and runs blow the time budget. Watch run time; avoid very low metab.
  (confidence: high)
- **The negative learning slope is the starvation death-spiral**, not plasticity:
  organisms dying of starvation rack up failed actions in their final stretch →
  within-life success declines with age. Reducing starvation (food always
  findable) is the lever, not reward-shaping. (confidence: medium)

## Active directions (untapped alpha)

_None recorded yet. The planner mines `directions/` to assign coordinator goals._

## Dead ends

_None recorded yet. See `dead-ends/`._

## Bundle census

- experiments: 0 · findings: 0 · directions: 0 · mechanisms: 4 (seeded) · dead-ends: 0
- Last iteration: 0 (not started).

## Next actions

1. Create `autoresearch/best` from `main`; run the baseline canonical eval to
   fill in current metrics above and `best-program.md`.
2. First iteration: assign coordinators to **food ecology** and **plasticity
   genome** (the two with the most config + code headroom on learning/foraging).
3. Keep predation in reserve — it likely needs engine code (corpse mechanics),
   which is a slower code-change loop.
