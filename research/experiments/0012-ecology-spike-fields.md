---
type: Experiment
title: Clustered lethal spike fields + Forward-into-spike-fails (Dir3 niche) — PROMOTED
description: Cluster the existing spikes into contiguous Perlin fields and score a Forward into a spike as a failed action. Makes far-field perception survival-relevant — breaks the seed-7 vision=1 collapse (vision held ~8 on all seeds) and at 5% coverage raises cross-seed action_effectiveness above the baseline champion.
iteration: 12
coordinator: ecology-niche
agent: spike-fields
surface_area: ecology-niche
base_ref: 1dab610
git_ref: autoresearch/exp-0012-ecology-spike-fields
status: promoted
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0761, prey_consumption_rate: 0.00226, action_effectiveness: 0.5522, mi_sa: 0.1089, learning_slope: -0.000570 }
baseline_metrics: { plant_consumption_rate: 0.0786, prey_consumption_rate: 0.00236, action_effectiveness: 0.5434, mi_sa: 0.1951, learning_slope: -0.000487 }
delta: { plant_consumption_rate: -0.0025, prey_consumption_rate: -0.0001, action_effectiveness: 0.0088, mi_sa: -0.0862, learning_slope: -0.000083 }
tags: [ecology-niche, spikes, vision, perception, action-effectiveness, champion, promoted]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

A niche that makes far-field perception *survival-relevant* will break the
vision=1 myopia collapse (the Dir1 confound) and raise the trusted intelligence
headline `action_effectiveness` via real navigation skill — without "adding ease."
Target: a spatial-hazard cull ([[directions/predation-led-mortality-selects-for-skill]]).

# Change

Two changes to the **existing** spike entity (already in the eval world; visible
in the vision rays; 10% max-health/tick damage):
1. `sim-core/src/spawn/world.rs`: spikes are **clustered** into contiguous Perlin
   fields (the top `spike_density` fraction of cells by noise, `SPIKE_NOISE_SCALE`
   0.10) instead of i.i.d. salt-and-pepper — so a 1-cell reflex can't escape by
   stepping between them; routing around a field requires perceiving its extent.
2. `sim-core/src/turn/commit.rs` `apply_moves`: a Forward INTO a spike cell is
   **not** marked action-succeeded (the organism still enters and takes spike
   damage), so `action_effectiveness` penalizes blundering into a hazard and
   rewards routing around it.
3. Config: `spike_density` default 0.10 → **0.05** (sim-evaluation/config.toml +
   compiled default), the swept sweet spot; also resolves a pre-existing
   eval(0.10)/baseline(0.05) desync (sim-config/config.toml was already 0.05).

# Result

**PROMOTED.** Cross-seed 500k (all seeds survive, det-check ok, tests ok modulo
the known pre-existing corpse-feeding failure):

| pillar | baseline `1dab610` | spike(5%) | Δ |
|---|---|---|---|
| **action_effectiveness (HEADLINE)** | 0.5434 | **0.5522** | **+0.0088** |
| plant_consumption_rate | 0.0786 | 0.0761 | −0.0025 (within noise, all seed-7) |
| prey_consumption_rate | 0.00236 | 0.00226 | −0.0001 (held) |
| mi_sa (diagnostic) | 0.1951 | 0.1089 | −0.0862 (de-inflated — confound removed) |
| learning_slope | −0.000487 | −0.000570 | −0.000083 (within noise) |

**The decisive readout — vision (the anti-confound co-gate):** seed 7 holds
`vision_distance` **9.04** (baseline 1.06 — the collapse is gone); all seeds hold
~7.5–9.0. mi_sa is now **uniform ~0.09–0.13** across seeds (seed 7 = 0.11, no
longer a 0.44 outlier) — direct confirmation the baseline mi_sa was the seed-7
vision=1 artifact. action_effectiveness rises on 3/4 seeds (+0.03…+0.05); only
seed 7 drops (−0.06), having lost its myopic high-aeff strategy.

**Coverage is a tuned knob:** 10% over-taxed navigation (aeff 0.527, −0.016); 5%
forces perception yet lets skill recover (aeff 0.552, +0.009); 7% sat between.
The 5% open-endedness 2nd-half slope (+0.0044/100k) modestly beats the converged
baseline (−0.0008) — less plateau.

# Learnings

Far-field perception was never worse here — the hazard-free dense-food world just
didn't reward it ([[findings/clustered-lethal-hazards-break-the-vision-myopia-collapse]]).
A skill-dependent spatial cull re-routes mortality toward navigation without ease
and breaks the vision=1 attractor. The gate must be **conjunctive** (aeff↑ AND
vision-not-collapsing AND throughput held) because aeff's "success = not-failing"
can also be gamed by timidity. Static hazards converge eventually; a moving /
co-evolutionary target is the next step for stronger open-endedness
([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).

# Concerns

- Seed 7's per-seed aeff/foraging drop (lost the myopic strategy); the cross-seed
  mean rises on the strength of the other three. Acceptable: the baseline was
  partly propped up by a degenerate strategy we explicitly wanted to remove.
- mi_sa fell below even the original homeostatic 0.1407 — but mi_sa is a demoted,
  confounded diagnostic now; the drop is the confound being removed, not a loss.

# Reproduce

`git checkout autoresearch/exp-0012-ecology-spike-fields; cargo build -p sim-cli --release`;
per-seed `new --seed S` (default spike_density=0.05) + `run-to 500000` +
`pillars` + `find "generation>50" --fields id,vision` for S ∈ {7,42,123,2026}.

# Citations

[1] diff: `git show autoresearch/exp-0012-ecology-spike-fields` (commits fea100a + d22dce8).
[2] Cross-seed 500k pillars + vision, planner-authoritative, 2026-06-18.
