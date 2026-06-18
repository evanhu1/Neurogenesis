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

# DEFINITIVE (iter19): the precisely-diagnosed mechanism ALSO plateaus below champion

iter18 diagnosed the missing piece as a DENSE + BRAIN-CONTROLLED + non-saturating
cognitive contest (so complexity PAYS → intelligence). iter19 built *exactly* that
([[experiments/0019-ecology-display-contest]]: a brain "display" output + perception
of neighbors' displays + a dense zero-sum intransitive transfer on the display).
Result: seed-7 aeff *appeared* to rise (0.32→0.42 @500k, accelerating) — but the 1M
test shows it was slow recovery from the perception-dilution hit: aeff **peaks ~0.46
@750k then turns over to 0.44 @1M, plateauing far below champion 0.56**; complexity
turns over too; cross-seed mean aeff 0.50 < champion. So **even the exact
theory-specified mechanism converges AND stays below champion** (the 2 added
perception channels dilute and the contest never repays it). The ≥2×-horizon check
caught the slow-saturation a THIRD time (roamer, genome, now display). This is
airtight from every angle: no in-loop mechanism yields sustained open-ended
*intelligence*; it needs a fundamentally richer cognitive substrate (non-diluting
perception/action + a non-saturating arms race) — a research-scope redesign.

# DEFINITIVE (iter18): the cognitive arms race gives the FIRST sustained growth — as dilutive bloat. Open-ended growth and intelligence are in DIRECT CONFLICT.

The most goal-aligned test — an intransitive *POLICY* arms race (committed-attack
pursuit-evasion: predator predicts, prey evades), measured by brain COMPLEXITY (the
goal's real target, not passive color) — [[experiments/0018-ecology-pursuit-evasion]]:
- It produced the **FIRST sustained non-convergent signal in the whole program**:
  seed-7 brain complexity rose monotonically through 1M (neurons 12→30, synapses
  13→30, no plateau), and complexity grew on all 4 seeds (mean ~26 neurons / ~30
  synapses vs champion ~18/16).
- **But it is DILUTIVE BLOAT, not intelligence:** cross-seed competence REGRESSED
  (aeff 0.5613→0.5157; seeds 42/123 crashed to 0.38/0.49). The encounter-limited
  ecology makes the arms race too SHALLOW to reward the extra capacity, so the
  growing brain is noise that dilutes competence (the "capacity dilutes" law).

**This is the definitive conclusion, from the sharpest possible angle: open-ended
brain GROWTH and brain INTELLIGENCE are in DIRECT CONFLICT in this engine.** You can
get sustained non-convergent brain evolution (complexity inflation) OR sustained
intelligence — not both. Making the growth PAY (→ open-ended *intelligence*) needs a
DEEP arms race with unbounded strategic depth, which the engine's shallow,
encounter-limited, simple-perception (4-action) ecology structurally cannot provide.
Genuine open-ended evolution of *intelligent* brains is therefore not reachable
in-loop on this engine — it needs a richer cognitive substrate (deeper perception/
action space + an arms race that can't saturate), a research-scope redesign.

# (earlier) SPATIAL CORRECTION (post-iter17): the "uniform" is WEAKLY spatially structured — strengthening levers conflict with the goal

The "stable uniform equilibrium" (R→0.11) claim for iter17 was measured by the
GLOBAL circular-mean, which **cannot distinguish dead-uniform from spatial pattern
formation** (domains/waves) — and spatial structure is the classic thing that
SUSTAINS intransitive cycles (spatial rock-paper-scissors). Re-measured with
spatial autocorrelation (added `q`/`r`/`hue` read fields): at 1M, global R=0.109
but **NEAR-pair (hex-dist ≤4) cos(Δhue)=0.129 ≫ ALL-pair 0.010** — so weak spatial
hue DOMAINS persist (weak sustained local cycling). The earlier "relaxes to a dead
fixed point" was therefore **wrong**: there is weak, persistent spatial structure.

**But it is WEAK, and cannot be strengthened while preserving intelligence:**
- The world is large (250-wide) and **sparse (~1–2% equilibrium density, food-
  limited)** → few neighbors → weak local coupling → diffuse domains.
- **Density lever** (more food → denser → stronger coupling) = "ease", which the
  central law says DEGRADES intelligence — anti-goal.
- **Dispersal lever** (higher `move_action_energy_cost` → less wandering → expect
  lineage/hue clustering): tested at 2.0 — it BACKFIRED. Lower dispersal shrank the
  breeding pop (n≈319) → MORE finite-population drift → faster GLOBAL convergence
  (NEAR 0.79 ≈ ALL 0.72, one hue), the opposite of domains.
- Initial over-seeding (denser start) just crashes back to the food-limited
  carrying capacity.

- **Population-size lever** (larger world, MORE organisms at the SAME density →
  lower drift, ease-free, intelligence-preserving — the cleanest test of the drift
  hypothesis): tested at 400-wide (~2900 organisms, 2.5× N). Winding = 0.97 turns
  over 400k (a slightly faster WANDER, still a bounded random walk, not directed),
  and NEAR-cos ≈ ALL-cos throughout (**no** spatial domains — the bigger world is
  MORE well-mixed). So reducing drift does NOT convert the wander into sustained
  directed winding. Drift-reduction fails too.

- **Density-via-EASE lever** (the trade-off itself — accept ease to force density):
  tested iter17 with 3× food / fast regrowth → pop ~5000 (8% density). It FAILS ON
  EVERY AXIS: NEAR ≈ ALL (still **no** spatial domains — mobility/mixing washes out
  local sorting even at 8% density), winding still a bounded wander (−0.29 turns),
  AND **aeff drops 0.56→0.38** (ease degrades intelligence, exactly as the central
  law predicts). So even accepting the trade-off buys NO open-endedness and loses
  intelligence.

**Final (exhaustive, ~20 experiments):** sustained OE is unreachable here through
ANY tested mechanism or trade-off. Every principled lever — substrate
(predation/foraging/social), interaction form (pure-damage/zero-sum), spatial
structure, dispersal, population size, AND the density/ease trade-off — leaves the
intransitive cycle either CONVERGING (to a hue/uniform) or WANDERING (a bounded
~1-turn random walk); it never sustains directed winding, and strong spatial
domains never form (mobility + finite-population mixing dominate). The conclusion is
not a constraint conflict that a trade-off resolves — the cycle simply does not
sustain in this finite, mobile, spatially-mixing system. Genuine unbounded
open-endedness would require a qualitatively different evolutionary mechanism (the
open-ended-evolution grand challenge), not an in-engine lever. Genuine unbounded
open-endedness of *intelligent* brains needs an engine change that escapes this
sparse-ecology / finite-population regime WITHOUT relaxing selection — a research/
human-scope decision, not an in-loop tuning knob.

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
