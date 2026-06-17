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

| axis | target | champion (a1d33b7, n=4) | status |
|---|---|---|---|
| foraging  | `plant_consumption_rate ≥ 0.10`  | 0.0690 | stuck (more-food de-pressures brains) |
| predation | `prey_consumption_rate  ≥ 0.025` | 0.0018 | arms race works but metric resists a hunting minority |
| learning  | `learning_slope ≥ +0.0005`        | −0.000423 | seed-noise-dominated; metabolism dry |
| intelligence | hold action_effectiveness (0.5647) & mi_sa (0.1407) | — | **predation splits them: mi_sa↑, action_eff↓** |

## Current best program

- **`autoresearch/best` @ `a1d33b7`** = champion code `eb30fff` (homeostatic
  metabolism) + apparatus + knowledge. **Iterations 2 AND 3 advanced nothing.**
  See `best-program.md`.

## ⚠ Where we are (the strategic crux)

Two compounding insights now define the program:

1. **Selection pressure, not ease, is the bottleneck for intelligent brains**
   ([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]], high).
   Every ease-adding lever degrades the intelligence pillars.
2. **The predator–prey arms race is the goal-aligned mechanism — but the proxy
   metrics block it** ([[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]],
   high). Energy-conserving kill rewards reliably evolve a real predator niche
   with emergent hunting brains (`ContactAhead→Eat`, `Corpse→Eat`) and **rising
   `mi_sa`** where strong — genuine open-ended evolution. But: `prey_consumption_rate`
   (= prey/total_actions) barely moves for a hunting *minority* (tops ~0.004–0.005,
   5–10× short), and `action_effectiveness` regresses (predation kills younger →
   lower death-cohort competence + attack chaos). The two intelligence pillars
   **disagree under predation** (mi_sa↑, action_eff↓). Salience + reward did NOT
   synergize (the combo was worse — compounded dilution).

**Implication:** the most goal-aligned result we can produce (an evolved predator
niche) scores as a dead-end on the current metrics. Either amplify predation until
it dominates the ecology (so prey rises and hunting skill is survival-critical), or
re-weight the intelligence metric toward `mi_sa`
([[directions/reconsider-intelligence-metric-under-predation]] — needs a human call).

## Frontier (per surface area)

| surface area | status | next move |
|---|---|---|
| corpse / predation mechanics | **hot, mapped** | **[[directions/amplify-the-predation-dynamic]]** — make predation dominant (consume-on-kill + scarcer plant → hunt-or-starve) |
| evaluation / metrics | flagged | [[directions/reconsider-intelligence-metric-under-predation]] (mi_sa vs action_eff; human call) |
| brain topology (sensing) | enabler works alone-weak | corpse channel learnable; only useful WITH a strong reward |
| metabolism / lifecycle | dry (2 iters) | do not re-tune |
| food ecology | closed as abundance | foraging must come from skill, not slack |
| plasticity genome | reserved | reward-sensitive rule once an arms-race ecology selects competence |

## Established mechanisms (durable laws)

- [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] (high) — central.
- [[mechanisms/predation-inversely-coupled-to-population-health]] (high).
- [[mechanisms/plant-rate-tracks-metabolism-over-food-energy]] (high).

## Active directions

- **[[directions/amplify-the-predation-dynamic]]** (HIGH, iter4) — predation-dominant
  ecology (consume-on-kill + scarcer plant): hunt-or-starve should raise prey AND
  put hunting *effectiveness* under hard selection (recovering action_eff) AND mi_sa.
- [[directions/reconsider-intelligence-metric-under-predation]] (med, human call).
- [[directions/corpse-sensory-salience]] — enabler, only with a strong reward.
- [[directions/redistributive-kill-reward]] / consume-on-kill — mapped (real niche,
  prey short, action_eff regress); the base for amplification.

## Dead ends

- Predation via richer corpses / more-lethal attacks / additive reward (iter2).
- Salience + reward COMBO (iter3) — worse than components (compounded dilution).
- Metabolism ramp/brain-cost/move-cost; foraging-via-abundance (iter2 findings).

## Bundle census

- experiments: 21 (12 iter1 + 6 iter2 + 3 iter3) · findings: 6 · mechanisms: 3 ·
  directions: 9 · dead-ends: 2.
- Last iteration: **3** — dry for advance; arms race proven inducible; metric
  misalignment surfaced.

## Next actions (iteration 4)

1. **Amplify predation** ([[directions/amplify-the-predation-dynamic]]): base =
   consume-on-kill; deliberately CROSS-FAMILY with reduced plant availability so
   hunting is required to survive — 2 scarcity levels. Watch for the win signature:
   prey↑ toward 0.025 AND action_effectiveness HELD (hunting now survival-critical
   → effective hunting selected) AND mi_sa↑. Screen population/extinction on seed 7
   BEFORE 500k (scarcity risks collapse).
2. If it still can't hold action_effectiveness while raising prey, the proxy-metric
   misalignment is confirmed structural → surface the metric question to the user.

## Process / harness (iteration 3)

- **Self-isolation fix WORKED** — pre-created per-agent worktrees + agent self-check;
  `main` stayed clean (vs iter2 contamination). Single SEMDIR held (no split).
- **New hazard:** an agent's broad `pkill` matched a sibling's relative
  `artifacts/c-42.bin` and killed its run → recipe should mandate UNIQUE artifact
  names per agent + absolute-path process matching. (S1 killed P2's seed-42; P2 recovered.)
- Planner-controlled clean-worktree gate (sccache + det-check P1/P2) remains the
  authoritative backstop and worked. Determinism held across topology + turn-logic combo.
