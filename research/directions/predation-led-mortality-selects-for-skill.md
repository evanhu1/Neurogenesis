---
type: Direction
title: Predation-led (not starvation-led) mortality is the regime that selects for skill
description: Across the 4 champion seeds, the one whose death mix is predation-dominated (seed 7, 38.5% predation / 48% starvation) evolved 4x the mi_sa of the starvation-dominated seeds (25-29% predation / 54-62% starvation). When culling is by skilled predation rather than density/luck starvation, individual skill becomes decisive. A niche that forces predation-led mortality may broadly induce the seed-7 effect.
priority: high
status: open
surface_area: ecology-niche
supported_by: [findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect]
tags: [ecology-niche, predation, mortality, selection-pressure, open-endedness]
timestamp: 2026-06-18T00:00:00Z
---

# The observation

The four champion seeds split cleanly on **death-cause mix vs intelligence**:

| seed | predation share | starvation share | mi_sa |
|---|---|---|---|
| **7** | **38.5%** | 48% | **0.442** |
| 42 | 35.4% | 54% | 0.121 |
| 123 | 28.8% | 58% | 0.114 |
| 2026 | 25.1% | 62% | 0.103 |

The seed where **predation is the dominant relative cull** evolves dramatically
more skill. Where **starvation dominates** (a density/luck-driven cull, weakly
coupled to individual policy), populations stay at the mi_sa ~0.10 floor.

# The hypothesis

Selection-for-skill needs the cull to *depend on skill*. Starvation in a stable
ecology is largely density-regulated — the marginal organism dies because the
population is too big, not because it is unskilled — so it applies weak pressure
on the sensory→action policy. Predation is the opposite: surviving (and eating)
depends on a *skilled* pursuit/avoidance policy, so it directly rewards a tight
food/threat-direction→action map. This is consistent with the long-standing
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] law and with
[[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]].

# Why it's untapped

Prior predation work tuned the *mechanics* of predation (corpse energy,
lethality, reach, reward) and hit the encounter-limit wall
([[mechanisms/predation-is-encounter-limited]]). It did **not** target the
**death-cause mix** as the control variable. Shifting mortality from
starvation-led to predation-led — without just adding ease — is a fresh lever.

# Candidate experiments (one surface area: ecology-niche)

1. **Reduce starvation's share without adding food ease:** e.g. a gentler
   low-energy metabolism floor *combined with* stronger predation pressure, so
   the net cull shifts toward predation. (Watch: don't relax selection overall —
   ease de-pressures brains. The goal is to *re-route* mortality, not reduce it.)
2. **A skill-gated cull that is not vision-myopia-rewarding:** a niche where
   survival depends on a behavior that *uses* long-range perception (so it does
   not collapse to the vision=1 degenerate optimum — see
   [[directions/mi_sa-is-confounded-by-vision-range]]). E.g. a mobile threat to
   flee, or a spatial hazard to navigate, that rewards far-field sensing.
3. **Measure the death-cause mix as a first-class signal** in the eval (it is
   already in `eco` deaths-by-cause) and check whether predation-share predicts
   mi_sa across many seeds/configs (n>4) — if the +correlation holds, it is a
   Mechanism candidate.

# Caution

Seed 7's skill rode on `vision=1`, which partly games mi_sa. Before celebrating a
predation-led niche, confirm the skill it induces is **not** just more myopia —
evaluate with the vision-controlled measure from
[[directions/mi_sa-is-confounded-by-vision-range]]. The two directions must be
worked together: a better niche (this) + a better measure (that).

# Next action

After the Dir2 measure decision, run an `ecology-niche` coordinator targeting
candidate (1) or (2): shift the death mix toward predation and check mi_sa +
foraging under the vision-controlled metric, cross-seed.
