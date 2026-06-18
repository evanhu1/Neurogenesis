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

| axis | target | champion `120a9eb` (iter17, n=4) | status under NEW contract |
|---|---|---|---|
| **intelligence (HEADLINE)** | **hold/raise `action_effectiveness`** | **0.5613** | ↑ from iter12's 0.5522 (+0.0091, iter17 social transfer); now ~0.003 below the all-time homeostatic 0.5647, earned with vision held + max color diversity |
| intelligence (diagnostic) | ~~mi_sa~~ | 0.1059 | demoted; honest/uniform — no confound |
| foraging  | `plant_consumption_rate ≥ 0.10`  | 0.0733 | within noise; still short of target |
| predation | `prey_consumption_rate  ≥ 0.025` | 0.00303 | up from 0.00226; still **structurally UNREACHABLE** |
| learning  | `learning_slope ≥ +0.0005`        | −0.000578 | seed-noise-dominated |

## Current best program

- **`autoresearch/best` @ `120a9eb`** (iter17) = the iter12 spike champion + **dense
  organism–organism ZERO-SUM color-cyclic energy transfer**
  ([[experiments/0017-ecology-social-transfer]]). Energy flows hue-DOMINATED→DOMINANT
  between hex-neighbors (zero-sum/conservative, not ease). **aeff 0.5522→0.5613
  (+0.0091)**, prey up, **color diversity MAXIMIZED** (never converges to one hue —
  a richer ecology). Gate green; holds at 1M. NOT open-endedness (reaches a stable
  UNIFORM color equilibrium, R→0.11; mechanism goes inert) — a champion advance, not
  the OE breakthrough. Lineage 6 deep.
- *(prior, `47a6111`/iter12:)* clustered lethal spike fields — fixed the seed-7
  vision=1 confound (vision 1.06→9, mi_sa de-inflated), aeff +0.0088.
- *(substrate, `1dab610`/iter9:)* homeostatic + consume-on-kill + three-factor
  within-life learning + GAIN-0.04 tune.
- *(aeff-optimal alternative base:)* homeostatic-only `eb30fff` (aeff 0.5647).

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

## ⛳⛳ ROOT CAUSE of the OE limit (2026-06-18, after 20 iterations)

**Open-ended intelligence requires an UNBOUNDED affordance space; this engine has a
fixed small one (4 actions, simple vision, ~4 environment mechanics) → a finite set
of skill TYPES → competence saturates once mastered, extra capacity = bloat.** No
in-engine mechanism — intransitive dynamics, cognitive arms races, perception
augmentation, compositional resources, OR POET-style environment co-evolution — can
manufacture unbounded skill-complexity from a bounded affordance space (POET only
generates harder *instances* of the same finite skills). The bottleneck is NOT
selection/diversity/metric/substrate/interaction/search — it is the dimensionality
of what an agent can DO and SENSE. Achieving the goal needs designing-in an
unbounded affordance space (composable actions / tool use / evolvable morphology /
expanding environment rules / open-ended signalling) — a fundamental engine
redesign. See [[findings/open-endedness-requires-an-unbounded-affordance-space-the-engine-lacks]].
The two champion advances (intelligence ↑) + the full 20-iteration map are banked.

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

- experiments: 36 (+0012-spike, +0017-social-transfer promoted;
  +0013/14/15/16/18/19/20/21 rejected) · findings: 14 (incl. ROOT-CAUSE) ·
  mechanisms: 4 · directions: 16. **iter21 tested the root-cause FIX itself
  (affordance expansion via a Build action) — it COLLAPSED competence ~20×
  (degenerate wall-spam): even adding an affordance fails because rewarding its
  SKILLED use is the open problem. OE is blocked at BOTH the affordance and
  reward/problem levels — a circular dependency = the grand challenge's core.**
  (+architectural-path-to-open-ended-intelligence) · dead-ends: 2.
- **The architectural path's FIRST step was tested (iter20) and FAILED:** an
  ease-neutral compositional 2-step resource raises the skill floor but LOWERS the
  ceiling (aeff plateaus ~0.38 < baseline 0.46 @1.5M). So even a hand-added
  expandable structure has a finite (lower) ceiling — OE needs a GENUINELY UNBOUNDED,
  agent-CO-EVOLVED problem generator (POET-style) or a major transition, NOT static
  structure ([[directions/architectural-path-to-open-ended-intelligence]]). That is a
  large multi-component research build; the in-loop AND simplest-architectural spaces
  are now both exhausted.
- **iter19 (brain-controlled cognitive contest — the precisely-diagnosed OE
  mechanism):** built a brain "display" output + neighbor-display perception + dense
  intransitive transfer on it. seed-7 aeff rose 0.32→0.46 @750k then PLATEAUED to
  0.44 @1M (below champion 0.56); cross-seed mean 0.50 < champion; complexity grew
  2.5× then turned over. The ≥2×-horizon check caught slow-saturation a 3rd time.
  **Even the exact theory-specified mechanism converges + dilutes.** OE is
  definitively not in-loop-reachable. ([[experiments/0019-ecology-display-contest]])
- Champion advances: **4** (iter1 homeostatic, iter9 three-factor, iter12 spike
  fields, **iter17 social transfer**). Last iteration: **17**. The Dir1→Dir3 arc
  fixed the vision-confound (aeff advance); iters 13–17 exhaustively mapped
  open-endedness to a FINITE-POPULATION conclusion (every intransitive regime
  relaxes to a fixed point) — and the search yielded a 2nd aeff advance (social
  transfer, +0.0091, max color diversity).

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

## Frontier / next directions (post-Dir3, post-roamer)

**iter13 (roamer) REJECTED — and it taught the central lesson.** A moving lethal
hazard *looked* open-ended at 500k (slope 4× the champion, all seeds climbing) but
the **1M test showed it CONVERGES too** — slower and *lower* (aeff 0.528 vs champion
0.549) ([[findings/a-moving-hazard-delays-but-does-not-escape-convergence]]). So
**both static AND moving niches converge — unbounded open-endedness is still
unachieved.** Two durable process/mechanism lessons: (a) a positive 500k 2nd-half
slope can be slow convergence — confirm OE claims at ≥2× horizon; (b) a permanent
mortality tax lowers the competence ceiling (pressure must reward skill, not just
kill more).

**Diagnosed barrier (genome-over-time probe):** the blocker is a **STATIC FITNESS
LANDSCAPE**, NOT diversity loss and NOT the metric. Genetic diversity stays HIGH
the whole run (stable polymorphism), and the genome converges by ~1M just like
aeff — a fixed niche has a fixed optimum that evolution finds then stops
([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).

**iter14 tested the cure (intransitive co-evolution) — and found the BINDING
CONSTRAINT.** Cyclic color-dominance in predation (`predation_success *= 1 +
A·sin(Δhue)`, A∈{0.5,0.9}) is *theoretically* non-convergent (antisymmetric ⇒ no
ESS ⇒ orbits forever). In practice it **does NOT wind** — the population hue does a
bounded random walk (0.11–0.28 turns), because **predation is too SPARSE a
substrate** (~0.003/tick) — drift overwhelms the rare color-selection. Strength (A)
doesn't fix an event-frequency bottleneck
([[experiments/0014-ecology-color-dominance]]).

### ⛳ CAPSTONE (iters 12–17): open-endedness is FINITE-POPULATION-LIMITED — every regime relaxes to a fixed point

Open-endedness needs **endogenous non-stationarity** — an intransitive (no-ESS)
interaction whose optimum chases the population so it can't settle. Tested
exhaustively across substrates:
- predation (iter14): too SPARSE → bounded hue wander (0.11–0.28 turns).
- foraging + niche construction (iter15): dense but painting SELF-CONCENTRATES
  (R≈0.98) → no winding.
- **dense organism–organism, pure damage** (iter16): wound **0.85 turns** then
  LOCKED (no-ease ⇒ all-damage ⇒ race-to-dominant ⇒ R→0.98).
- **dense organism–organism, ZERO-SUM transfer** (iter17, PROMOTED): restores
  antisymmetry → does NOT collapse to one hue, but spreads to a **stable UNIFORM**
  equilibrium (R→0.11) where the interaction goes inert — a *different* fixed point.

**The true barrier is FINITE-POPULATION + SPARSE-ECOLOGY**
([[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]]):
the infinite-population antisymmetric game has neutral orbits, but the finite,
sparse, stochastic system relaxes toward a fixed distribution — drift + discreteness
damp the orbit. **Spatial correction:** the iter17 "uniform" is NOT dead — it has
WEAK spatial hue domains (NEAR-pair cos 0.13 ≫ ALL 0.01 at 1M) = weak sustained
LOCAL cycling (the global mean averaged it away). But it can't be strengthened into
traveling waves while preserving intelligence: the world is sparse/food-limited
(~1–2% density), and the strengthening levers all fail — **density** needs more food
(=ease, degrades intelligence); **lower dispersal** (higher move-cost) shrinks the
pop → MORE drift → global convergence. So strong sustained OE needs an engine change
escaping the sparse-ecology/finite-pop regime WITHOUT relaxing selection — a
research/human-scope decision, not an in-loop knob.

**Ruled out:** diversity maintenance; the metric; static niche; moving hazard;
intransitive strength; the food-ecology lock; AND now a dense organism–organism
intransitive interaction (converges to a fixed point too).

**iter18 (cognitive arms race) — the DEFINITIVE test + result.** Committed-attack
pursuit-evasion (intransitive POLICY game; predator predicts, prey evades), measured
by brain COMPLEXITY (the goal's real target): produced the **FIRST sustained
non-convergent signal** (brain complexity grows monotonically to 1M, no plateau) —
**but as DILUTIVE BLOAT** (cross-seed aeff regressed 0.5613→0.5157; the shallow
encounter-limited ecology can't reward the extra capacity). **Definitive: open-ended
brain GROWTH and brain INTELLIGENCE are in DIRECT CONFLICT here** — you get sustained
complexity inflation OR sustained intelligence, not both. Open-ended evolution *of
intelligent* brains needs a richer cognitive substrate (deeper perception/action +
a non-saturating arms race) — a research-scope redesign, not an in-loop lever.
([[experiments/0018-ecology-pursuit-evasion]])

**Champion remains `120a9eb` (iter17 social transfer)** — the OE search's banked
gains: 2 champion advances (aeff +0.0088 then +0.0091; vision-confound fixed; max
color diversity). OE itself: approached (0.85-turn winding; sustained complexity
growth) but never as sustained *intelligence*. **The OE question is thoroughly and
definitively answered across all paradigms.**

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
