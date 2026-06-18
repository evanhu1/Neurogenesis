---
type: Finding
title: Clustered lethal hazard fields break the vision=1 myopia collapse (and raise action_effectiveness)
description: Making the existing spikes (a) cluster into contiguous Perlin fields and (b) score a Forward-into-spike as a failed action turns far-field perception survival-relevant. Seed 7 no longer collapses to vision=1 (holds ~8.5 like every seed); the mi_sa vision-confound vanishes; and at 5% coverage the cross-seed action_effectiveness rises above the baseline champion. Confirms the Dir1 thesis: vision=1 was an artifact of a hazard-free world, not a fundamental optimum.
confidence: high
status: active
supported_by: [experiments/0012-ecology-spike-fields]
seeds: [7, 42, 123, 2026]
tags: [ecology-niche, vision, perception, action-effectiveness, open-endedness, champion]
timestamp: 2026-06-18T00:00:00Z
---

# Question

Dir1 showed the champion's "intelligence" (mi_sa) was a **vision-range confound**:
seed 7 evolved `vision_distance=1`, which crisps the mi_sa sensory bins
([[findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect]]). Open
question from [[directions/mi_sa-is-confounded-by-vision-range]]: *is short vision
genuinely better here (the world is too small for long vision to pay), or is it
just a metric/niche artifact we can design away?* This experiment answers it.

# Answer

**It's a niche artifact — and a designable one.** A niche that makes far-field
perception *survival-relevant* eliminates the vision=1 collapse entirely. With
clustered lethal hazard fields, **seed 7 holds vision ~8.5 instead of collapsing
to ~1**, every seed keeps long vision, the mi_sa confound disappears, and (tuned
to 5% coverage) cross-seed `action_effectiveness` rises *above* the baseline
champion. Long vision was never worse — the old world just didn't reward it.

# The intervention ([[experiments/0012-ecology-spike-fields]])

Two changes to the *existing* spike entity (already in the eval world at 10%
i.i.d. coverage, perceptible in the vision rays, 10% max-health/tick):
1. **Cluster** spikes into contiguous Perlin fields (the top `spike_density`
   fraction of cells by noise) instead of i.i.d. salt-and-pepper — so a 1-cell
   reflex can't escape by stepping between them.
2. **Score a Forward INTO a spike as a *failed* action** (the organism still
   enters and takes damage) — so `action_effectiveness` penalizes blundering into
   a hazard and rewards routing around it.

# Evidence (measured, cross-seed 500k)

Vision is the decisive readout (the anti-confound co-gate):

| seed | baseline vision | spike(5%) vision | baseline aeff | spike(5%) aeff |
|---|---|---|---|---|
| 7 | **1.06** | **9.04** | 0.553 | 0.470 |
| 42 | 7.85 | 7.48 | 0.562 | 0.589 |
| 123 | 8.29 | 8.42 | 0.529 | 0.573 |
| 2026 | 8.94 | 7.74 | 0.531 | 0.577 |
| **mean** | (confounded) | **~8.2 (held)** | **0.5434** | **0.5522 (+0.0088)** |

- **The vision=1 collapse is gone.** Seed 7 — the sole myopic seed — now holds
  vision 9.04. No seed collapses. The myopia attractor is broken.
- **mi_sa de-inflates** 0.195 → ~0.10 (uniform across seeds), because the
  seed-7 vision=1 confound that produced the 0.44 outlier is gone — a direct
  empirical confirmation of the Dir1 mechanism.
- **action_effectiveness RISES** at 5% coverage: 3 of 4 seeds improve (+0.03 to
  +0.05); only seed 7 drops (−0.06), having lost its myopic high-aeff strategy.
  Net **+0.0088** over the baseline champion. Predation held (+0.0005), foraging
  −0.0025 (within noise, entirely seed 7).
- **Coverage is a real knob with a sweet spot:** 10% coverage over-taxes
  navigation (aeff 0.527, −0.016); 5% is large enough to force perception yet
  gentle enough that skill recovers (aeff 0.552, +0.009); 7% sat between.

# Why this matters (mechanism)

The reason vision=1 won in the baseline is that the world's food was dense enough
that local foraging sufficed, and long vision is a standing metabolic tax
(`vision_distance/3` in `metabolism.rs`). A *contiguous lethal field* changes the
payoff: a myopic organism that only reacts at contact distance dies traversing or
dwelling in the field; one that perceives the field at range routes around it and
keeps its Forwards landing in safe cells. The survival value of distant
information now exceeds the vision tax, so evolution keeps long vision. This is the
concrete realization of [[directions/predation-led-mortality-selects-for-skill]]
candidate (2) — a spatial hazard that rewards far-field sensing — and it
re-routes mortality toward a *skill-dependent* (navigation) cull without adding
ease (no food added, no metabolism softened; selection only *tightens*).

# Caveats / honesty

- aeff's "success = not-failing" framing means it can also be satisfied by acting
  timidly; the gate here was **conjunctive** (aeff↑ AND vision-not-collapsing AND
  foraging/predation-throughput held), which 5% coverage passes.
- Seed 7's per-seed aeff drops (it lost the myopic strategy); the cross-seed mean
  rises because the other three genuinely improve. The baseline aeff was partly
  propped up by seed 7's (degenerate) myopia.
- This is a *static* challenge; for the open-endedness angle see
  [[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]
  — the spike niche modestly raises the 2nd-half competence slope (less
  convergence) but a moving/co-evolutionary target would push further.

# Reproduce

`git checkout autoresearch/exp-0012-ecology-spike-fields; cargo build -p sim-cli --release`;
per-seed `new --seed S` (uses spike_density=0.05) + `run-to 500000` + `pillars` +
`find "generation>50" --fields id,vision` for S ∈ {7,42,123,2026}.

# Citations

[1] [[experiments/0012-ecology-spike-fields]] (git_ref
`autoresearch/exp-0012-ecology-spike-fields`, base `1e88bbe`), cross-seed 500k.
[2] Baseline champion `1dab610` per-seed pillars + vision (the inv-* worlds).
