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

### ⭐ CONTRACT CHANGE (2026-06-18, user decision after Dir1/Dir2)

**The intelligence headline is now `action_effectiveness` (vision-invariant).
`mi_sa` is DEMOTED to a diagnostic — it is partly a vision-range confound (it
rewards myopia; see Dir1).** Promotions are judged on action_effectiveness, not
mi_sa. This re-grounds the whole loop: any prior mi_sa-driven move is re-evaluated.

| axis | target | champion `47a6111` (iter12, n=4) | status under NEW contract |
|---|---|---|---|
| **intelligence (HEADLINE)** | **hold/raise `action_effectiveness`** | **0.5522** | ↑ from iter9's 0.5435 (+0.0088, Dir3 spike fields); now ~0.012 below the all-time homeostatic 0.5647 but EARNED with vision held high (no confound) |
| intelligence (diagnostic) | ~~mi_sa~~ | 0.1089 | demoted; now HONEST/uniform (~0.09–0.13 all seeds) — the seed-7 0.44 confound is GONE |
| foraging  | `plant_consumption_rate ≥ 0.10`  | 0.0761 | within noise of iter9; still short of target |
| predation | `prey_consumption_rate  ≥ 0.025` | 0.00226 | **structurally UNREACHABLE in a stable ecology** |
| learning  | `learning_slope ≥ +0.0005`        | −0.000570 | seed-noise-dominated |

## Current best program

- **`autoresearch/best` @ `47a6111`** (iter12) = the iter9 substrate + **Dir3
  clustered lethal spike fields** ([[experiments/0012-ecology-spike-fields]]).
  Spikes cluster into Perlin fields (coverage `spike_density=0.05`) and a Forward
  INTO a spike scores as a *failed* action → far-field perception becomes
  survival-relevant. **First advance under the action_effectiveness contract AND
  the first to fix the vision-confound at its root:** breaks the seed-7 vision=1
  collapse (seed 7 now vision 9.04; all seeds ~7.5–9.0), de-inflates mi_sa to an
  honest 0.109, and raises aeff +0.0088 (3/4 seeds up). Gate green (build/det
  P1-P2/tests/eval). Lineage 5 deep.
  ([[findings/clustered-lethal-hazards-break-the-vision-myopia-collapse]])
- *(substrate, `1dab610`/iter9:)* homeostatic metabolism (iter1) + consume-on-kill
  (iter3) + three-factor within-life learning (iter6) + GAIN-0.04 tune (iter9) —
  the predator/forager multi-niche ecosystem with within-life learning. Its mi_sa
  headline (0.1952) was later shown to be the seed-7 vision-confound (Dir1).
- *(aeff-optimal alternative base:)* homeostatic-only `eb30fff` (aeff 0.5647) — no
  predator niche / no within-life learning; one revert away if pure-aeff preferred.

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

## ⚑ Dir1 RESULT (2026-06-18) — the seed-7 mi_sa=0.44 outlier is a SHORT-VISION confound

Read-only deep-inspection of the 4 evolved 500k worlds (3 parallel sub-agents:
sensory/policy, wiring/convergence, niche/trajectory). Conclusion (high
confidence, [[findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect]]):

- **Seed 7 is the only seed that converged on `vision_distance = 1`** (mean 1.06,
  94% see 1 hex) vs **~8–9** for all others. Short vision **sharpens the mi_sa
  sensory bins** (food = crisp adjacent/absent) → near-deterministic
  food-direction→action map → high I(S;A). It wires a clean 3-reflex policy
  (`visF→Eat`, `visL→¬Forward`, `visR→TurnRight`; intra-seed cosine 0.975).
- **mi_sa has a degenerate optimum at MYOPIA** — reducing vision range raises it.
  So chasing mi_sa partly optimizes sensory *impoverishment*, the opposite of the
  open-ended goal. The champion's mi_sa headline rests on this one short-vision
  seed. → **the central Dir2 fix:** [[directions/mi_sa-is-confounded-by-vision-range]].
- **But it is also genuinely competent** (seed 7 = highest foraging 0.0925 AND
  predation 0.0034). Short vision here is an *effective, legible* strategy, not
  pure gaming.
- **Enabling niche:** sparse, food-rich (3.3× food/capita), **predation-led not
  starvation-led mortality** (38.5% vs 25–35% predation share). cross-seed
  corr(mi_sa, food/capita)=+0.997. Death-cause mix looks like a skill driver →
  [[directions/predation-led-mortality-selects-for-skill]] (Dir3 lever).
- All breeding pops are single-species monocultures; the "418 founder lineages"
  are inert gen-0 re-seed injections (artifact). Convergence is not the
  discriminator — *which policy* it converged on is.
- *Not measured:* exact H(S) vs H(A|S) split (per-tick records unserialized;
  the sub-agent attempting per-organism sampling leaked memory and was killed).
  Conditional-policy-sharpness rests on the wiring evidence.

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

- experiments: 27 (+0012-ecology-spike-fields) · findings: 11 (+seed-7-outlier,
  +convergence/not-open-ended, +clustered-hazards-break-myopia) · mechanisms: 4 ·
  directions: 14 (mi_sa-vision-confound in-progress) · dead-ends: 2.
- Champion advances: **3** (iter1 homeostatic, iter9 three-factor, **iter12 spike
  fields**). Last iteration: **12**. The Dir1→Dir2→Dir3 arc fixed the
  vision-confound (measure + niche) and produced a clean aeff advance.

## ⛳ STATUS after iter 7–8: mechanism space EXHAUSTED — binding constraint = metric contract

Iters 7–8 tried to complete the loop via PERCEPTION (corpse channel; reward-matched
live-prey channel). **Both regress intelligence** — adding sensory channels dilutes
brain topology more than the hunting signal repays
([[findings/perception-augmentation-dilutes-topology-the-best-arms-race-substrate-is-iter6]]).
The predation mechanism space is now fully mapped (corpse-energy / lethality /
reliability / scarcity / reach / kill-reward / perception). No clean champion advance.

**The best goal-aligned substrate is iter6** (consume-on-kill + three-factor,
branch `autoresearch/exp-0006-plasticity-three-factor-on-predation`, 696def5): a
predator-niche ecosystem with within-life reward-learning, det-check ok, ready to
promote — gated ONLY by a small (seed-123-driven) HOLD-pillar miss (aeff −0.022,
mi_sa −0.007 vs champion).

## ⛳ FRONTIER PLATEAUED (after iter11) — strong champion, architecture mapped

Iters 9–11: iter9 ADVANCED the champion (three-factor GAIN 0.04 → mi_sa 0.1952,
above the original 0.1407 — a real intelligence gain); iter10 (eligibility/decay)
and iter11 (brain substrate) both DRY. **2 consecutive dry iterations = plateau.**
The full architecture is mapped across 11 iterations (metabolism, food, predation
mechanics, plasticity/learning, sensing, topology). Recurring law:
**capability without strong-enough reward dilutes competence** (extra sensing/
synapses/neurons all regress intelligence). The minimal brain + tuned three-factor
+ arms-race ecology is the optimum on this architecture.

## ✅ The Dir1→Dir2→Dir3 arc is COMPLETE (2026-06-18)

- **Dir1 ✅** — seed-7 mi_sa=0.44 = a `vision_distance=1` confound (mi_sa rewards myopia).
- **Dir2 ✅** — measure recalibrated (user): `action_effectiveness` is the headline,
  mi_sa demoted. Skill panel confirmed seed-7's lead was mi_sa-only.
- **Dir3 ✅ PROMOTED** — clustered lethal spike fields
  ([[experiments/0012-ecology-spike-fields]]) make far-field perception
  survival-relevant: break the vision=1 collapse (seed 7 → vision 9) AND raise
  action_effectiveness +0.0088. Champion advanced to `47a6111` (iter12).

## Frontier / next directions (post-Dir3)

1. **Push the spike niche further (open-endedness):** the static field converges
   eventually ([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).
   A **moving / co-evolutionary hazard** (a roaming lethal agent to flee using
   vision) is the design-agent's Candidate 2 — forces far-field sensing AND never
   settles (Red-Queen). Likely the biggest remaining open-endedness lever.
   Also: re-tune coverage/lethality (lethality is still hardcoded 0.10) now that 5%
   coverage is the champion; and a longer horizon (1M) to see if seeds 123/2026
   (still climbing at 500k) pass 0.5647.
2. **An open-endedness metric** — make "sustained 2nd-half competence slope" (or
   behavioral/lineage novelty rate) a first-class signal beside action_effectiveness;
   the goal is *open-ended* evolution, which a static snapshot can't capture.
3. **A better learning rule** (C, unchanged) — prediction-error-gated eligibility
   or a 2nd (health/damage) neuromodulator.
4. Still open: `prey ≥ 0.025` unreachable ([[directions/reconsider-intelligence-metric-under-predation]]).

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
