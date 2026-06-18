---
type: Direction
title: mi_sa is confounded by vision range — control for it or replace it
description: Because the mi_sa sensory state S is "which vision ray first sees food", shorter vision yields crisper bins and higher I(S;A). mi_sa thus has a degenerate optimum at sensory impoverishment (myopia), which is anti-correlated with the open-ended-intelligence goal. The measure must control for vision range or be replaced by a sensory-richness-invariant skill metric.
priority: high
status: in-progress
surface_area: eval-contract
supported_by: [findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect]
tags: [eval-contract, mi_sa, metric-confound, open-endedness, measure-design]
timestamp: 2026-06-18T00:00:00Z
---

# The problem

mi_sa = I(S;A) with **S = which vision ray (−1/0/+1) first sees food, else none**.
Holding behavior fixed, **reducing `vision_distance` raises mi_sa** because food
becomes a crisp adjacent/absent signal instead of a blurry far-field one. The
seed-7 outlier (mi_sa 0.44) is exactly this: the only seed to converge on
`vision_distance = 1` (see
[[findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect]]).

So the metric we have been treating as "intelligence" can be increased by
**throwing perception away**. That is the opposite of open-ended cognitive
complexity (rich perception used skillfully), and it means the loop has been
partly optimizing myopia whenever it chased mi_sa.

# Why it matters for the goal ("open-ended evolution of intelligent brains")

A good intelligence measure should *increase* when an organism uses MORE sensory
information more effectively, not when it uses less. mi_sa-as-defined violates
this monotonicity. Any champion selected on mi_sa is therefore suspect until the
confound is removed — including the current champion, whose mi_sa headline rests
on one short-vision seed.

# Candidate fixes (to weigh with the user — this is the Dir2 human call)

1. **Vision-normalize mi_sa.** Report mi_sa per vision-band, or regress out
   vision_distance, so a short-sighted reflex can't win on crispness alone.
2. **Richer sensory state S.** Bin S over the *full* visual field (distance +
   direction + food type), not just "first ray with any food". A long-vision
   organism that navigates skillfully would then score its information use.
   (Risk: higher-dimensional S inflates the Miller-Madow correction / needs more
   samples — verify estimator stability.)
3. **Behavior-level skill instead of channel MI.** Measure *competence under
   counterfactual sensing* (e.g. effective foraging/predation per encounter,
   path efficiency to food) — invariant to how the sensory channel is discretized.
4. **Hold vision fixed in the genome** during intelligence evaluation so the
   metric can't be gamed by evolving the sensor away (changes the experiment, not
   just the metric).

# Open questions

- Is short vision *also genuinely better* here (seed 7 has the highest foraging),
  i.e. is the world too small for long vision to pay off? If so, the niche — not
  just the metric — under-rewards perception. Test by scaling the world / food
  sparsity and seeing whether long vision becomes competitive (ties to Dir3).
- Does action_effectiveness share the confound? (It scores contingent-action
  success, less obviously vision-coupled — check.)

# Prototype result (2026-06-18, read-only over the 4 existing 500k worlds)

Built a vision-invariant skill panel and ranked seed 7 against the other seeds on
each measure. **Seed 7's outsized lead exists ONLY in mi_sa:**

| measure | seed-7 lead over best non-7 seed | seed-7 rank |
|---|---|---|
| mi_sa | **3.65×** | 1/4 |
| action_effectiveness (cleanest vision-invariant) | **0.98× (behind seed 42)** | **2/4** |
| plant_consumption_rate (food-availability-confounded) | 1.16× | 1/4 |
| prey_consumption_rate | 1.52× | 1/4 |
| predations/capita | 1.12× | 1/4 |

**The 3.65× mi_sa lead is a vision-range confound, not a skill gap.** On the
cleanest availability-invariant skill measure (`action_effectiveness`) seed 7 has
**no** advantage. Genuine skill gaps are modest (1.1–1.5×). action_effectiveness
does **not** share the confound (it ranks the long-vision seed 42 first), so it is
the better-behaved intelligence proxy of the two existing pillars.

**Sobering corollary:** the champion's headline iter9 "intelligence gain"
(mi_sa 0.1335→0.1952, promoted, seed-7-heavy —
[[experiments/0009-plasticity-three-factor-tune]]) may be substantially a
vision-confound artifact (seed 7 evolving shorter vision), not a real cognitive
gain. *Unverified hypothesis* — needs a per-seed vision comparison vs the
predecessor champion to confirm.

# Next action

Surface the contract decision to the user (this is the explicit human call). The
panel makes the recommendation concrete: **demote mi_sa as the intelligence
headline in favor of action_effectiveness (vision-invariant), and/or redefine S to
be sensory-richness-invariant.** After the user's call, lock the measure, then
proceed to [[directions/predation-led-mortality-selects-for-skill]] (Dir3) under it.
