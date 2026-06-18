---
type: Finding
title: The seed-7 mi_sa=0.44 outlier is a short-vision (crisp-binning) effect, enabled by a predation-led sparse niche
description: Seed 7's 4x mi_sa is caused by it being the only seed to converge on vision_distance=1; short vision sharpens the mi_sa sensory bins so the food-direction→action map is near-deterministic. mi_sa is therefore partly confounded by vision range (a degenerate optimum at myopia), though seed 7's reflex is also genuinely the most competent forager.
confidence: high
status: active
supported_by: []
investigation: dir1-seed7-outlier
base_ref: c542d21
champion_code_ref: 1dab610
seeds: [7, 42, 123, 2026]
tags: [intelligence, mi_sa, metric-confound, vision, predation, open-endedness, investigation]
timestamp: 2026-06-18T00:00:00Z
---

# Question

mi_sa (mutual information I(S;A) between the sensory bin S = *which vision ray
first sees food* ∈ {none, left, center, right} and the action A) is the
champion's single biggest intelligence signal, but it is **seed-7-heavy**: seed 7
= 0.44 while seeds 42/123/2026 sit at ~0.10–0.12. *Why did seed 7 find 4× the
others' mi_sa, and can we induce it broadly?* (Read-only deep-inspection of the
four evolved 500k worlds of champion code `1dab610`; no code changed.)

# Answer (headline)

**Seed 7 is the only seed whose breeding population converged on
`vision_distance = 1`** (mean 1.06; 94% see exactly 1 hex) vs **~8–9 hexes** for
all three low-mi_sa seeds. Short vision **sharpens the mi_sa sensory bins**: with
1-hex vision, food is either in an adjacent hex (a crisp, unambiguous
direction) or not, so the food-direction→action contingency is near-deterministic
→ high I(S;A). Long vision blurs the bins (food visible far away, often in
multiple rays, weakly coupled to the immediate action) → low I(S;A). The high
mi_sa is therefore **partly a vision-range confound** — but seed 7's short-vision
reflex is *also* the most competent forager (highest plant rate), so it is a
genuinely effective strategy, not pure metric-gaming.

# Evidence (measured)

## 1. Vision is the single discriminating gene (active breeders, generation > 50)

| seed | mi_sa | vision mean | vision mode | pop | mean age | age max |
|---|---|---|---|---|---|---|
| **7** | **0.442** | **1.06** | **1 (94%)** | 796 | 572 | 6740 |
| 42 | 0.121 | 7.85 | 9–10 | 1633 | 286 | 4907 |
| 123 | 0.114 | 8.29 | 7–9 | 1467 | 338 | 3991 |
| 2026 | 0.103 | 8.94 | 9–10 | 1243 | 365 | 5968 |

Other genes (neurons, synapses, hebb_eta) do **not** cleanly separate seed 7
from the pack (seed 2026 also has ~39 neurons but mi_sa 0.10). Vision does.

## 2. Seed 7's reflex is the cleanest, most consistent, and most competent

- Dominant-brain wiring (aggregated over 10 top-energy/top-age brains/seed):
  seed 7 carries **three** distinct, consistent food-direction→action reflexes —
  `visForward→Eat (+2.15)`, `visLeft→¬Forward (−1.86)`, `visRight→TurnRight
  (+1.38)` — present in ~10/10 brains, **intra-seed wiring cosine 0.975**
  (highest). Other seeds wire **one** weak/inconsistent coupling (seed 42:
  `visLeft→TurnLeft` only; seed 123 couplings present in only 3/10 brains).
- Seed 7 is *also* the best on the competence pillars: **plant 0.0925 (highest),
  prey 0.0034 (highest)**, action_effectiveness 0.553 (2nd). Short vision is not
  a fitness handicap here — it is a different, more *legible* strategy.

## 3. The niche that enables and sustains it: sparse, food-rich, predation-led

Cross-seed **corr(mi_sa, food-energy-per-capita) = +0.997**.

| seed | pop | food/capita | starvation share | predation share | dom. species |
|---|---|---|---|---|---|
| **7** | 796 (lowest) | **143 (3.3×)** | **48% (lowest)** | **38.5% (highest)** | 100% (1 sp.) |
| 42 | 1633 | 46 | 54% | 35.4% | 100% (1 sp.) |
| 123 | 1467 | 44 | 58% | 28.8% | 100% (1 sp.) |
| 2026 | 1243 | 49 | 62% | 25.1% | 100% (1 sp.) |

Seed 7 is a low-density equilibrium where each organism holds ~3.3× the food
energy, mortality is **predation-led rather than starvation-led**, and organisms
are long-lived. (Every seed's *breeding* population is a single-species
monoculture — the ~418 "founder lineages" `lineage` reports are inert gen-0
re-seed injections: median age 100, 0 consumptions, empty 0-neuron brains. So
"seed 7 is more diverse" was a measurement artifact; convergence is not the
discriminator — *which policy* it converged on is.)

## 4. mi_sa is earned over time, not a flat artifact

Seed 7 starts at the same ~0.02–0.05 floor as everyone, then climbs a **staircase
of phase transitions** (lift-off 20–50k → ~0.25; decisive surge 330–360k → peak
0.50), while seeds 42/123/2026 stay frozen in the ~0.10 basin the entire run.
Within seed 7 over time: corr(mi_sa, pop) = −0.66, corr(mi_sa, plant_rate) =
+0.79. The sparse/food-rich regime stabilizes by ~140–160k, ~190k ticks *before*
the decisive mi_sa surge — so the niche is an **enabling condition**, not an
instantaneous cause; ongoing selection/learning inside the niche does the rest.

# Inferences (mechanistically supported, not directly measured)

1. **Causal chain:** `vision=1` mutation → crisp sensory bins + a clean 3-reflex
   local-forager policy → highest foraging efficiency → well-fed, long-lived
   organisms → sparse equilibrium → predation-led (not starvation-led) mortality
   → which further selects the skilled reflex. mi_sa is high because the bins are
   crisp (vision) *and* each bin maps to a distinct action (the reflex).
2. **mi_sa has a degenerate optimum at myopia.** Because S is *which vision ray
   sees food*, **reducing** vision range raises I(S;A) (crisper bins), all else
   equal. So optimizing mi_sa can push evolution toward sensory *impoverishment* —
   the opposite of the open-ended goal (rich perception used skillfully). This is
   a measure-design defect, directly relevant to the Dir2 contract recalibration.
3. **Predation-led mortality is a candidate intelligence driver.** The seeds
   where culling is starvation-dominated (density/luck) stay at mi_sa ~0.10; the
   one where predation dominates the death mix evolves skill. Death-cause mix is a
   lever Dir3 (new niche) and Dir2 (measure) should both consider.

# Caveat (honesty)

The exact **H(S) vs H(A|S) numeric decomposition** ("is the edge richer sensory
marginal or sharper conditional policy?") was **not** separately measured: the
per-tick `(food_visible, selected_action)` record is not serialized to the world
file, and `query` (single-load batch) rejects `step`, so a paired (S,A) sample
needs many subprocess spawns (the sub-agent that attempted it leaked memory and
was killed). The conditional-policy-sharpness claim rests on the **wiring
evidence** (the clean, consistent 3-reflex), which is the conditional policy
made visible. The vision→bin-crispness link is structural (a direct property of
the mi_sa definition), not a fitted number.

# Implications

- **Dir2 (measure):** mi_sa must be **controlled for vision range** (or replaced
  with a sensory-richness-invariant skill measure), else the research loop is
  partly optimizing myopia. See
  [[directions/mi_sa-is-confounded-by-vision-range]].
- **Dir3 (new niche):** the distinguishing regime is **predation-led mortality +
  sparse/food-rich**. A niche that *forces* predation-led mortality (or otherwise
  makes skill, not luck, the cull) may broadly induce the seed-7 effect.
- **Champion fragility:** the champion's mi_sa headline is one short-vision seed.
  This re-frames [[directions/reconsider-intelligence-metric-under-predation]].

# Reproduce

```
git checkout 1dab610; cargo build -p sim-cli --release
for S in 7 42 123 2026; do
  sim-cli new --seed $S --out artifacts/inv-$S.bin
  sim-cli run-to 500000 --in artifacts/inv-$S.bin
  sim-cli pillars --in artifacts/inv-$S.bin          # mi_sa per seed
  sim-cli find "generation>50" --fields id,vision --limit 4000 --in artifacts/inv-$S.bin  # vision dist
  sim-cli eco --in artifacts/inv-$S.bin; sim-cli food --in artifacts/inv-$S.bin           # niche
  sim-cli top energy --in artifacts/inv-$S.bin; sim-cli brain <id> --view synapses --in artifacts/inv-$S.bin  # wiring
done
```

# Citations

[1] Read-only investigation of the four evolved 500k worlds of champion code
`1dab610` (planner branch `autoresearch/best` @ `c542d21`), 2026-06-18. Three
parallel read-only sub-agents: sensory/policy, wiring/convergence, niche/trajectory.
[2] mi_sa definition: `sim-metrics/src/ledger.rs::sensory_bin` (food_visible rays
−1/0/+1 → bin), `sim-metrics/src/intervals.rs::mi_from_joint` (Miller-Madow I(S;A)).
[3] `sim-types/src/lib.rs`: `VISION_RAY_OFFSETS = [-1,0,1]`, `vision_distance` gene.
