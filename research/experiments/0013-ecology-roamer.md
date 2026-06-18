---
type: Experiment
title: Roaming lethal hazard (moving far-field-flee niche) — delays but does NOT escape convergence; rejected
description: A small number of deterministically-moving lethal agents organisms must flee using far-field vision. Reuses the spike visual channel (no new sensory neuron). At 500k it LOOKED open-ended (all seeds still climbing, slope 4x the champion) — but the 1M test corrected this: it CONVERGES too, just slower and at a LOWER competence level (0.528 vs champion 0.549). A permanent moving-hazard mortality tax caps competence below the champion. Rejected; the 500k 'open-endedness' was mid-climb, not unbounded.
iteration: 13
coordinator: ecology-niche
agent: roamer
surface_area: ecology-niche
base_ref: 47a6111
git_ref: autoresearch/exp-0013-ecology-roamer
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0724, prey_consumption_rate: 0.00237, action_effectiveness: 0.5229, mi_sa: 0.0917, learning_slope: -0.000570 }
baseline_metrics: { plant_consumption_rate: 0.0761, prey_consumption_rate: 0.00226, action_effectiveness: 0.5522, mi_sa: 0.1089, learning_slope: -0.000487 }
delta: { plant_consumption_rate: -0.0037, prey_consumption_rate: 0.00011, action_effectiveness: -0.0293, mi_sa: -0.0172, learning_slope: -0.000083 }
open_endedness: { roamer_aeff_at_500k: 0.5229, roamer_aeff_at_1M: 0.5282, roamer_late_slope_750k_1M: -0.0012, slope_was_500k: 0.0183, champion_aeff_1M: 0.5492 }
tags: [ecology-niche, roamer, moving-hazard, open-endedness, non-stationary, convergence, dead-end]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

A static skill challenge converges
([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).
A **moving** lethal hazard organisms must flee using far-field vision is (a) a
stronger far-field-perception pressure (you must see it *coming*) and (b)
**non-stationary** — it should not saturate, driving sustained open-ended
improvement (the goal).

# Change

New `sim-core/src/roamer.rs`: a config-driven count (`roamer_count`, default 14 ≈
0.3% of the ~4900-cell grid) of lethal agents. Deterministic placement
(`hash_2d(index,0,seed^MIX)`) and motion (each steps 1 hex/tick by
`mix_u64(seed^MIX^turn^index) % 6` — pure hash, serialized state, index-ordered).
A new `Roamers` tick phase between move-resolution and commit. Damage via the
shared spike/kill machinery. **Reuses the existing spike VISUAL channel** (roamers
OR'd into a transient `sensed_spike_map` rebuilt before sensing) so brains sense
them with NO new sensory neuron (dodging the dilution finding). Config plumbed
through both TOMLs. det-check ok (P1+P2), tests ok (only the known pre-existing
failure), clippy clean.

# Result

**REJECTED.** It *looked* open-ended at 500k but the 1M test shows it converges
too — slower and at a LOWER level than the champion.

500k (roamer_count=14) — the tempting-but-misleading snapshot:

| | mean aeff@500k | 2nd-half slope /100k | vision | pop |
|---|---|---|---|---|
| baseline | 0.5434 | −0.0008 (converged) | confounded | — |
| spike champion `47a6111` | 0.5522 | +0.0044 | ~8 | healthy |
| roamer (14) | 0.5229 | **+0.0183 (4×)** | ~8.5 (held) | 1035–1656 (healthy) |

At 500k every seed was still climbing (seed 123 0.33→0.49, seed 2026 0.36→0.46,
ending at their maxima) → it looked like the first sustained open-ended
improvement. **roamer_count tuning didn't recover the 500k level** (count 6→0.479
noisy, 10→0.543, 14→0.523; none ≥0.5522).

**1M extension — the decisive correction:**

| | aeff@500k | aeff@1M | late slope (750k–1M) |
|---|---|---|---|
| spike champion | 0.5522 | 0.5492 | ~0 (converged) |
| roamer (14) | 0.5229 | **0.5282** | **−0.0012 (CONVERGED)** |

By 1M the roamer's late-slope is ~0 — it **converged**, just later than the
champion, and at a **lower** level (0.528 < 0.549). seed 7 even *declined*
(0.582→0.548). The strong 500k slope was **slow convergence on a harder task, not
unbounded open-endedness.**

# Why rejected

It regresses the canonical 500k headline (−0.029) AND, the 1M test confirms,
converges at a lower competence level than the champion (0.528 vs 0.549) — so it
is not a win on either the static OR the longer horizon. The moving-hazard
mortality is a *permanent tax* that caps competence below the champion without
buying unbounded improvement. Champion HELD at `47a6111`.

# Learnings

**A non-stationary hazard DELAYS but does NOT ESCAPE convergence** — and converges
*lower* ([[findings/a-moving-hazard-delays-but-does-not-escape-convergence]]).
Two hard lessons:
1. **The 500k horizon mid-climb MISLEADS** — a positive 2nd-half slope at 500k is
   NOT evidence of open-endedness; it can be slow convergence on a harder task.
   Open-endedness claims require a LONGER horizon (1M here flattened it). Always
   confirm a sustained-slope result at 2× the horizon before believing it.
2. **Adding a permanent mortality tax (harder task) lowers the competence ceiling**
   — consistent with the central law that selection pressure must reward SKILL,
   not just kill more ([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]).
   A moving threat that is *unavoidable noise* (deterministic-but-unpredictable
   motion) taxes without a learnable counter-skill ceiling above the champion's.
Genuine open-endedness likely needs **co-evolution** (the threat itself evolves —
a true Red Queen), not a fixed-policy moving hazard, OR an environment that grows
new *opportunity* (not just new mortality). Both static spikes and this moving
hazard converge.

# Reproduce

`git checkout autoresearch/exp-0013-ecology-roamer; cargo build -p sim-cli --release`;
per-seed `new --seed S` (default roamer_count=14) + `run-to 500000` + `pillars`
(read `granular.intervals[].action_effectiveness` for the slope) + `find
"generation>50" --fields id,vision`.

# Citations

[1] diff: `git show autoresearch/exp-0013-ecology-roamer` (commit 6a2775b).
[2] Cross-seed 500k pillars + granular slopes + roamer_count sweep {6,10,14},
planner-authoritative, 2026-06-18.
