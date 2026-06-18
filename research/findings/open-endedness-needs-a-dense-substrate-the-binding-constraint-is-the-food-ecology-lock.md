---
type: Finding
title: Open-endedness needs a DENSE interaction substrate — the binding constraint is now the food-ecology policy lock
description: Across four mechanisms (static spikes, moving roamer, and intransitive color-predation at two strengths) the system converges or fails to wind. Diversity is maintained, the metric is fine, and the intransitive idea is theoretically non-convergent — but it has no dense substrate to ride on. The only dense organism-level interaction (foraging/food) is policy-locked. So realizing open-endedness requires relaxing the food-ecology invariant — a deliberate human decision. This is the capstone of the open-endedness investigation.
confidence: high
status: active
supported_by: [experiments/0013-ecology-roamer, experiments/0014-ecology-color-dominance, experiments/0015-ecology-forage-color, experiments/0016-ecology-social-color, experiments/0017-ecology-social-transfer]
seeds: [7, 42, 123, 2026]
tags: [open-endedness, substrate, food-ecology, invariant, binding-constraint, human-call, capstone]
timestamp: 2026-06-18T00:00:00Z
---

# The question

Can this engine produce open-ended (unbounded, sustained) evolution within its
invariants (no speciation, determinism, hidden food-ecology policy)? After a
systematic sweep of mechanisms, the answer is sharp.

# What was tried, and what converged

| approach | result | why it fails for open-endedness |
|---|---|---|
| static skill niche (spike fields, iter12 — PROMOTED for competence) | competence plateaus ~250k; flat 500k→1M | fixed niche ⇒ fixed optimum |
| moving hazard (roamer, iter13) | converges too, lower (looked open-ended at 500k mid-climb; 1M corrected it) | fixed-policy threat ⇒ one best evasion, then stop |
| genome-over-time probe | genome means converge by ~1M; **diversity stays HIGH** | rules out diversity-loss and the metric as the barrier |
| intransitive color-predation (iter14, A=0.5 & 0.9) | **no winding** (bounded hue random-walk, 0.11–0.28 turns) | right idea (no ESS) but predation too sparse a substrate |

# The diagnosis (precise)

1. **The barrier is a static fitness landscape**, not diversity loss (diversity is
   maintained — a stable polymorphism) and not the metric. A fixed niche has a
   fixed optimum that evolution finds and then stops at.
2. **The cure is endogenous non-stationarity** — frequency-dependent / intransitive
   interaction whose optimum chases the population, so it can never settle. The
   color-predation mechanism implements this and is *theoretically* non-convergent
   (antisymmetric payoff ⇒ no ESS ⇒ replicator orbits forever).
3. **But it needs a DENSE substrate, and predation is too sparse.** The
   intransitive force rides on predation events (~0.003/organism/tick,
   encounter-limited), so genetic drift overwhelms the rare color-selection signal
   and the population hue diffuses (bounded random walk) instead of orbiting.
   Increasing the per-event strength (A 0.5→0.9) ~doubles the wander rate but does
   not produce winding — it's an *event-frequency* bottleneck, not a strength one.
4. **The only dense organism-level interaction is foraging/food** — which is
   **policy-locked** in `sim-config` (the hidden food-ecology policy, an
   architecture invariant). So within the current invariants the open-endedness
   lever has no dense substrate to ride on.

# FINAL UPDATE (iters 16–17): added the dense organism–organism interaction — it converges too. OE is finite-population-limited.

The capstone said the missing piece was a *dense organism–organism* interaction
(the only inherently-frequency-dependent kind). So the loop **built one**: a dense
color-cyclic adjacency interaction between hex-neighbors.
- **Pure damage** ([[experiments/0016-ecology-social-color]]): wound **0.85 turns**
  (far past everything prior) — then LOCKED (R→0.98). No-ease forced *all-damage*,
  which breaks antisymmetry → race-to-dominant-hue → convergence.
- **Zero-sum energy transfer** ([[experiments/0017-ecology-social-transfer]],
  PROMOTED as champion): restores antisymmetry → the color does NOT collapse to one
  hue. But the 1M test shows it spreads to a **STABLE UNIFORM** distribution
  (R→0.11) where the interaction goes ~inert — a *different fixed point*, not
  sustained rotation/novelty. aeff settles (above champion).

**So every regime converges to SOME fixed point:** a single hue (pure-damage,
foraging) or a uniform distribution (zero-sum). The infinite-population
antisymmetric game has neutral orbits, but the FINITE, spatial, stochastic system
always relaxes to a stable distribution — **drift + discreteness damp the orbit.**

**The true barrier is finite-population / dynamical, not just substrate:** an
intransitive cycle's neutral orbits are not robust to finite-population drift here;
sustaining a *traveling-wave* (winding) limit cycle would need an extra force that
makes the central/uniform fixed point a REPELLER (a stable limit cycle around it),
which none of the ease-safe, determinism-safe, no-speciation mechanisms provide.
Achieving genuine open-endedness likely needs a qualitatively different driver
(e.g. structural/spatial niching, or relaxing an invariant) — a research-scope
question, not a tuning knob.

**Silver lining:** the search produced a real CHAMPION ADVANCE — the zero-sum
social transfer maximizes color diversity and lifts cross-seed action_effectiveness
+0.0091 ([[experiments/0017-ecology-social-transfer]]). Open-ended *novelty* was
approached (0.85-turn winding; maintained spread) but never *sustained*.

# (earlier, iter15) the dense substrate was tried — the barrier is ARCHITECTURAL

The food-layer was NOT a human-call blocker after all (the food-ecology invariant
means "hidden from world-TOML", not "code frozen"), so the loop built it:
intransitive hue-keyed foraging + niche construction on the dense foraging
substrate ([[experiments/0015-ecology-forage-color]], ease-safe, deterministic).
**It also failed to wind — worse than predation** (winding −0.06 turns; hue
concentration R≈0.98). The **niche-construction painting is positive feedback that
self-concentrates** the population+food hue, collapsing the spread the `sin(Δ)`
cycle needs.

**The real barrier is ARCHITECTURAL, not the food-ecology lock:**
- An intransitive cycle needs frequency-dependence (opponent hue tracks the
  population) AND maintained spread. Organism–food frequency-dependence requires
  painting, which concentrates (kills spread). The interaction that is inherently
  frequency-dependent *without* painting is organism–organism — but the only one
  (predation) is too sparse.
- **The engine has no DENSE organism–organism interaction** to host the cycle.
  This is the true binding constraint. Realizing open-endedness would need a
  deeper architectural change: a dense organism–organism interaction, a
  cycle-preserving (not polymorphism-fixing) niche mechanism, or spatial/structural
  niching — beyond config or single-function tuning.

# (superseded) earlier conclusion — the food-ecology policy lock

Realizing open-endedness in this engine most plausibly requires putting an
intransitive / niche-construction dynamic on the **dense food layer** — e.g.
color-keyed digestion where each organism digests food "near" its own hue best,
plus corpse/consumption-driven local shifts of the resource hue distribution
(consumer-resource co-evolution / niche construction). That is a deliberate
**relaxation of the food-ecology invariant**, which the planner will NOT do
autonomously (it is owned by sim-config by design). **This is the next decision and
it is the human's:** *do we open a controlled crack in the food-ecology policy lock
to let endogenous non-stationarity run on the dense substrate?*

The no-speciation and determinism invariants are NOT the blockers and should be
kept. The food-ecology lock is the one to negotiate.

# What is NOT the problem (ruled out — don't re-explore)

- Diversity maintenance (already high; a stable polymorphism).
- The metric (action_effectiveness is fine; it even has a vision-invariant fix).
- Selection strength of the intransitive term (A doesn't help — frequency does).
- A static or moving *external* pressure (both converge).

# Process lesson reinforced

Twice now a positive 500k slope (roamer competence; iter14 hue would-be-winding)
looked open-ended and flattened at 2× horizon. **Confirm every open-endedness claim
at ≥2× horizon** — and prefer a *convergence-immune* signal (an unbounded winding
number) over a slope, because a slope can be slow convergence.

# Citations

[1] [[experiments/0013-ecology-roamer]], [[experiments/0014-ecology-color-dominance]],
[[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]],
[[findings/a-moving-hazard-delays-but-does-not-escape-convergence]],
[[mechanisms/predation-is-encounter-limited]]. Planner-authoritative, 2026-06-18.
