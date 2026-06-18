---
type: Experiment
title: Dense organism–organism ZERO-SUM color-cyclic energy transfer — champion advance (NOT open-endedness; stable uniform equilibrium)
description: Make the dense social color interaction a ZERO-SUM energy transfer (energy flows hue-dominated→dominant) instead of pure damage. Zero-sum (conservative, not ease) restores antisymmetry, so the color does NOT collapse to one hue — but the 1M test shows it spreads to a STABLE UNIFORM distribution (R→0.11), not a rotating cluster, so it's NOT open-endedness. It IS a champion advance: cross-seed action_effectiveness 0.5613 (+0.0091 over champion at the canonical 500k eval), holds at 1M, prey up. Promoted.
iteration: 17
coordinator: ecology-niche
agent: social-transfer
surface_area: ecology-niche
base_ref: 47a6111
git_ref: autoresearch/exp-0017-ecology-social-transfer
status: promoted
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0733, prey_consumption_rate: 0.00303, action_effectiveness: 0.5613, mi_sa: 0.1059, learning_slope: -0.000578 }
baseline_metrics: { plant_consumption_rate: 0.0761, prey_consumption_rate: 0.00226, action_effectiveness: 0.5522, mi_sa: 0.1089, learning_slope: -0.000570 }
delta: { plant_consumption_rate: -0.0028, prey_consumption_rate: 0.00077, action_effectiveness: 0.0091, mi_sa: -0.0030, learning_slope: -0.000008 }
tags: [ecology-niche, intransitive, organism-organism, zero-sum, open-endedness, non-convergence, champion, promoted]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The pure-damage social-color interaction ([[experiments/0016-ecology-social-color]])
wound 0.85 turns then locked (R→0.98) because no-ease forced *all-damage*, breaking
the antisymmetry. Make it **ZERO-SUM**: energy flows from hue-DOMINATED to
hue-DOMINANT adjacent organisms (`net(o) = A·Σ sin(hue_o − hue_n)`). Zero-sum is
conservative — NOT ease (the energy≥0 clamp only ever destroys energy, never
creates; exactly like predation's consume-on-kill redistribution) — and it
restores the antisymmetric structure a sustained cycle needs, so the population
should stay SPREAD (not collapse to one hue).

# Change

`commit.rs apply_social_color_mortality`: full antisymmetric `sin` (not max(0)),
applied as an energy transfer (`organism.energy = (energy + A·Σsin).max(0)`) in a
deterministic snapshot-then-apply pass. `SOCIAL_DAMAGE`=1.0 is now the transfer
rate. A=0 ⇒ baseline byte-identical. det-check ok (P1+P2), tests ok.

# Result

**PROMOTED — champion advance AND the first non-converging mechanism.** Cross-seed
500k:

| pillar | champion `47a6111` | social-transfer | Δ |
|---|---|---|---|
| **action_effectiveness (HEADLINE)** | 0.5522 | **0.5613** | **+0.0091** |
| prey_consumption_rate | 0.00226 | 0.00303 | +0.00077 |
| plant_consumption_rate | 0.0761 | 0.0733 | −0.0028 (within noise) |
| mi_sa (diagnostic) | 0.1089 | 0.1059 | −0.0030 (flat) |
| learning_slope | −0.000570 | −0.000578 | flat |

3/4 seeds rise to **0.585–0.607** aeff (only seed 2026 regresses to 0.453). Pops
healthy (1079–1557). The color does NOT collapse to one hue (R 0.24–0.70 at 500k,
unlike pure-damage's R→0.98).

**1M test — NOT open-endedness (stable uniform equilibrium):** extending seed 7 to
1M, **colorR → 0.11** (the population spreads toward a UNIFORM distribution, not a
rotating cluster) and **aeff settles** 0.585→0.580→0.561 (above champion, not
open-ended-climbing). So the zero-sum transfer reaches a *stable maximal-diversity*
(uniform) equilibrium — at uniform the per-organism net transfer averages to ~0, so
the mechanism goes nearly INERT. It does not converge to one hue, but it does not
sustain novelty either; it's a different *fixed* distribution. **OE remains
unachieved.** (Important: the gain is lasting, not a pure transient — seed-7 aeff
at 1M is 0.561 vs the spike champion's seed-7-at-1M 0.465.)

# Why it's a champion advance but not OE

Zero-sum makes a dominated organism actively LOSE energy to a dominant one, so
there's no "be the leading hue" escape that collapses the population — instead the
population spreads OUT (toward uniform) to minimize everyone's exposure. Uniform is
a stable equilibrium (Σsin over uniform neighbors ≈ 0 → no net transfer → no
torque). So the mechanism drives DIVERSITY to maximum and then rests — it does not
produce the sustained rotation OE needs. The aeff lift is from the transient
color-sorting selection + the maximal-diversity ecology, and it persists.

# Concerns / open

- **NOT open-endedness** — reaches a stable uniform color equilibrium; aeff settles
  (~0.56, above champion). The champion ADVANCE (aeff +0.0091 @ the canonical 500k
  eval, holds at 1M) is solid; the OE claim is explicitly NOT made.
- Seed 2026 regresses (−0.078, the laggard); the cross-seed mean still rises.
- The mechanic couples to the brain only INDIRECTLY (positioning); the aeff gain is
  partly selection/composition, not demonstrably "smarter" brains — promoted on the
  objective gate (cross-seed aeff↑, no regression, det ✓, stable), with this caveat.

# Reproduce

`git checkout autoresearch/exp-0017-ecology-social-transfer; cargo build -p sim-cli --release`;
per-seed `new --seed S` + `run-to 500000` + `pillars` + `find "generation>50" --fields hue`
(circular-mean hue wanders unboundedly; R stays 0.2–0.7 — never converges).

# Citations

[1] diff: `git show autoresearch/exp-0017-ecology-social-transfer` (commit a6214dd).
[2] Cross-seed 500k pillars + per-seed color-spread R, planner-authoritative, 2026-06-18.
