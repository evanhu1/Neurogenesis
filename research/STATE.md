---
type: Research State
title: STATE — autonomous research dashboard
description: The planner's live, compacted working memory. Read this first every session; it is sufficient to resume without rereading the archive.
tags: [autoresearch, state, dashboard]
timestamp: 2026-06-17T00:00:00Z
---

# STATE

**Read this first.** Distilled working memory. Dip into `experiments/`,
`findings/`, `log.md` only for a specific fact. Rewritten each iteration.

## Goal & targets (cross-seed mean, seeds 7,42,123,2026, 500k; raw metrics)

| axis | target | champion (a90244a, n=4) | status |
|---|---|---|---|
| foraging  | `plant_consumption_rate ≥ 0.10`  | 0.0690 | stuck (more-food de-pressures brains) |
| predation | `prey_consumption_rate  ≥ 0.025` | 0.0018 | **top lead = redistributive kill-reward** |
| learning  | `learning_slope ≥ +0.0005`        | −0.000423 | metabolism lever drying up |
| intelligence | hold action_effectiveness (0.5647) & mi_sa (0.1407) | — | the binding constraint |

## Current best program

- **`autoresearch/best` @ `a90244a`** = champion code `eb30fff` (homeostatic
  metabolism) + apparatus + knowledge. **Iteration 2 advanced nothing** (all 6
  experiments dead-ended; carried foraging lever failed the gate). Champion
  metrics unchanged. See `best-program.md`.

## ⚠ THE central insight (iteration 2 — reframes everything)

**Selection pressure, not survival ease, is the bottleneck for intelligent brains**
([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]], high conf).
Every lever that adds slack — more food (lower fertility threshold: aeff −0.064,
mi_sa −0.097), softer metabolism (ramp-tune: aeff −0.050), cheaper movement,
cheaper brains (sqrt cost: aeff −0.067), free kill-energy (explosion: aeff
0.565→0.239) — **grows/relaxes the population and degrades the intelligence
pillars.** With a reward-free covariance learning rule, selection pressure is the
only force maintaining competent action; abundance removes it.

**Consequence:** the foraging/survival/learning targets are in partial TENSION
with *holding* intelligence. You cannot buy intelligence with ease. The way
forward is to route any gain through a **skill that must be earned** — above all a
**balanced predator–prey arms race** (energy via competent hunting), which adds a
competence-rewarding niche without relaxing the constraint.

## Frontier (per surface area)

| surface area | status | next move |
|---|---|---|
| corpse / predation mechanics | **hot** | **[[directions/redistributive-kill-reward]]** — energy-conserving kill reward (TOP) |
| brain topology (sensing) | open | [[directions/corpse-sensory-salience]] — let brains perceive prey/corpses (enabler) |
| metabolism / lifecycle | **drying up** (1 dry iter) | homeostatic ramp near-optimal; further softening trades intelligence |
| food ecology | closed (as abundance) | more standing food de-pressures brains; foraging must come from skill |
| plasticity genome | reserved | a reward-sensitive rule could supply within-life skill pressure (iter1: unstable) |

## Established mechanisms (durable laws)

- **[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]** (high) — NEW, central.
- [[mechanisms/predation-inversely-coupled-to-population-health]] (high).
- [[mechanisms/plant-rate-tracks-metabolism-over-food-energy]] (high).
- Negative learning_slope = starvation death-spiral (medium); homeostatic softened
  but did NOT solve it (starvation still 57–72% of deaths). learning_slope is
  **seed-noise-dominated at n=4** → require sufficient-n + per-seed consistency.

## Active directions (untapped alpha)

- **[[directions/redistributive-kill-reward]]** (HIGH, iter3) — proven lever
  (prey hit 0.036), just needs energy conservation. The arms-race path to BOTH
  prey and intelligence.
- [[directions/corpse-sensory-salience]] (med) — enabler so brains can learn to hunt.
- plasticity: a bounded reward-sensitive rule (three-factor) to supply within-life
  skill pressure — [[directions/tune-three-factor-neuromodulation-band]] (re-test
  on the arms-race ecology, where competence is selected).

## Dead ends (ruled out)

- iter2 predation: richer corpses & more-lethal attacks don't create a scavenger
  niche; additive kill-reward explodes ([[findings/predation-needs-an-energy-conserving-kill-reward-not-richer-corpses]]).
- iter2 metabolism: ramp-tune / brain-cost / move-cost all trade action_effectiveness
  (selection-pressure law).
- foraging-via-abundance: [[findings/lower-fertility-threshold-does-not-stack-on-the-homeostatic-base]]; food_energy cuts ([[mechanisms/plant-rate-tracks-metabolism-over-food-energy]]).
- iter1: [[dead-ends/flat-metabolism-cut-removes-action-selective-pressure]], [[dead-ends/grace-window-causes-per-seed-population-runaway]].

## Bundle census

- experiments: 18 (12 iter1 + 6 iter2; 1 promoted ever) · findings: 5 · mechanisms: 3 ·
  directions: 7 (5 open) · dead-ends: 2 (+ many rejected experiments).
- Last iteration: **2** — dry for champion advance; rich findings + harness rebuild.

## Next actions (iteration 3)

1. **Predation arms race** (TOP): [[directions/redistributive-kill-reward]] —
   energy-conserving kill reward, sweep the fraction, screen population stability
   on seed 7 BEFORE any 500k confirm (additive version OOM'd). Pair with
   [[directions/corpse-sensory-salience]] (disjoint surface area).
2. Watch for the hoped-for signature: prey↑ AND action_effectiveness/mi_sa HOLD or
   RISE (the arms race should *select for* competence, unlike ease-levers).
3. Metabolism is dry — do not re-tune the ramp; pivot learning to plasticity only
   if the arms race establishes a competence-rewarding ecology.

## Process / harness learnings (iteration 2)

- **`isolation:"worktree"` did NOT engage** for planner-spawned background agents —
  they landed in the shared checkout and clobbered each other's source + the shared
  binary; the capable agents detected it and self-isolated (doubling their work).
  **FIXED in SKILL:** agents now self-isolate (own worktree) as step 0; don't trust
  the harness flag.
- **Semaphore split into two pools** (custom `AUTORESEARCH_SEM` override drift) →
  effective cap ~16 not 8. **FIXED:** use sim-run's single fixed default SEMDIR;
  never override per-round.
- Blanket `pkill` of a runaway clipped M3's unpersisted confirm — persist the
  branch BEFORE the confirm / target only the runaway PID.
- Iterative agents are token-heavy (100–360k each) and slow under contention; the
  gate (planner-controlled, clean worktree) is the authoritative backstop and
  worked. sccache + det-check (P1 byte + P2 fingerprint) validated.
