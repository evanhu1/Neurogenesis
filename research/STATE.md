---
type: Research State
title: STATE — autonomous research dashboard
description: The planner's live, compacted working memory. Read this first every session; it is sufficient to resume without rereading the archive.
tags: [autoresearch, state, dashboard]
timestamp: 2026-06-17T00:00:00Z
---

# STATE

**Read this first.** Distilled working memory over 6 iterations. Dip into
`experiments/`, `findings/`, `log.md` only for a specific fact.

## Goal & targets (cross-seed mean, seeds 7,42,123,2026, 500k; raw metrics)

| axis | target | champion (homeostatic, n=4) | status |
|---|---|---|---|
| foraging  | `plant_consumption_rate ≥ 0.10`  | 0.0690 | stuck (abundance de-pressures brains) |
| predation | `prey_consumption_rate  ≥ 0.025` | 0.0018 | **structurally UNREACHABLE in a stable ecology** |
| learning  | `learning_slope ≥ +0.0005`        | −0.000423 | seed-noise-dominated |
| intelligence | hold action_effectiveness (0.5647) & mi_sa (0.1407) | — | the binding constraint; predation splits them |

## Current best program

- **`autoresearch/best`** = champion code `eb30fff` (homeostatic metabolism, the
  ONLY accepted advance, iter1) + apparatus + 6 iterations of knowledge.
  **Iterations 2–6 produced no further code advance** — but a complete, coherent
  theory of WHAT blocks open-ended evolution of intelligent brains, and a
  validated lead. See `best-program.md`.

## ⚠ THE THEORY (the compounding result of 6 iterations)

**Open-ended evolution of intelligent brains requires (1) selection pressure for
SKILL, and (2) a learning rule that converts that pressure into within-life skill —
and the current eval metrics partly block recognizing it.**

1. **Selection pressure is the bottleneck**
   ([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]], high).
   Every ease-adding lever (more food, softer metabolism, cheaper/bigger brains,
   free kill-energy, free attack-reach) grows/relaxes the population and DEGRADES
   action_effectiveness & mi_sa. Intelligence cannot be bought with ease.
2. **The predator–prey arms race is the skill-demanding niche**
   ([[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]]).
   Energy-conserving kill rewards reliably evolve hunting brains and raise mi_sa.
   But predation is **encounter-limited** ([[mechanisms/predation-is-encounter-limited]]):
   you can't grow it by scarcity (suppresses it via density), lethality, or free
   reach (DE-skills it → intelligence collapses). It must grow through *learned
   skilled pursuit*.
3. **Within-life reward-learning is the missing piece — and it WORKS on the
   predator ecology** ([[experiments/0006-plasticity-three-factor-on-predation]],
   iter6): a gentle three-factor (energy-delta) Hebbian rule on consume-on-kill
   RECOVERED action_effectiveness (0.5088→0.5422) — what iter1's three-factor
   FAILED to do on foraging-only. The full loop (predator niche + within-life
   reward learning) is validated as the mechanism; it narrowly misses the gate
   (vs champion aeff −0.022 / mi_sa −0.007, seed-123-driven).
4. **The metrics partly block the goal:** `prey ≥ 0.025` is structurally
   unreachable ([[findings/prey-consumption-target-is-structurally-unreachable-in-a-stable-ecology]]);
   action_effectiveness penalizes predation (younger death-cohort) while mi_sa
   rewards it. A human call on the eval contract is the deepest unblock
   ([[directions/reconsider-intelligence-metric-under-predation]]).

## Frontier / next levers

| lever | status | note |
|---|---|---|
| **[[directions/reward-sensitive-learning-on-the-predator-ecology]]** | **validated, REFINE** | tune the neuromod band; STACK corpse sensory channel so hunting is perceivable+learnable; fix the seed-123 disruption → aim to clear the gate |
| [[directions/reconsider-intelligence-metric-under-predation]] | human call | prey target unreachable; weight mi_sa? cohort-normalize action_eff? |
| predation mechanics | exhaustively mapped | corpse/lethality/reliability/scarcity/reach all dead-ended |
| metabolism / food | dry / closed | ease de-pressures brains |

## Established mechanisms (durable laws)

- [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] (high) — central.
- [[mechanisms/predation-is-encounter-limited]] (high).
- [[mechanisms/predation-inversely-coupled-to-population-health]] (high).
- [[mechanisms/plant-rate-tracks-metabolism-over-food-energy]] (high).

## Bundle census

- experiments: 24 (12 iter1 + 6 iter2 + 3 iter3 + 2 iter4 + 1 iter5 + 1 iter6) ·
  findings: 7 · mechanisms: 4 · directions: 11 · dead-ends: 2.
- Champion advances: 1 (iter1 homeostatic). Last iteration: **6**.

## Next actions (iteration 7)

1. **Refine the validated lead** ([[directions/reward-sensitive-learning-on-the-predator-ecology]]):
   on the consume-on-kill base, (a) sweep the three-factor band (GAIN/SCALE/bounds)
   and (b) STACK the corpse/prey sensory channel (exp-0003-sensing) so the brain
   can perceive prey and the reward-learning has a clean signal — aim to push
   action_effectiveness ≥ champion AND mi_sa ≥ champion (then it cleanly advances,
   establishing a predator-niche champion with within-life learning).
2. If it still can't clear both HOLD pillars, the binding constraint is confirmed
   to be the **metric contract** — the predator-niche arms race IS open-ended
   evolution but the proxies undervalue it; surface the metric recalibration to the
   user (prey target + mi_sa-vs-action_effectiveness).

## Process / harness (works well now)

- Pre-created per-agent isolated worktrees + single fixed SEMDIR: `main` stays
  clean, no semaphore split, no cross-agent contamination (iter3–6 clean).
- Agents must use UNIQUE artifact names (a cross-agent relative-path `pkill` once
  killed a sibling's run). Planner blanket `pkill` can clip an unpersisted confirm —
  persist branch BEFORE confirm / target by PID.
- An agent's API drop can lose its return JSON — but its persisted worlds + branch
  are recoverable (iter6 recovered from saved 500k worlds; planner committed the branch).
- Planner-controlled clean-worktree gate (sccache + det-check P1/P2) is the
  authoritative backstop. det-check correctly validated the three-factor's new
  energy-delta read (reuses persisted state → deterministic).
