---
type: Finding
title: The champion converges (competence plateaus by ~250k); it is not open-ended
description: Action_effectiveness rises in the first half of a run then flatlines — 2nd-half slopes are ~0 for every seed. The current system reaches a stable competence equilibrium and stops improving, which is the opposite of open-ended evolution. The goal needs a niche that sustains novelty, not a static optimum.
confidence: high
status: active
supported_by: []
investigation: dir3-baseline-open-endedness-probe
champion_code_ref: 1dab610
seeds: [7, 42, 123, 2026]
tags: [open-endedness, convergence, action-effectiveness, plateau, goal]
timestamp: 2026-06-18T00:00:00Z
---

# Question

The goal is *open-ended* evolution of intelligent brains. Open-endedness means
sustained novelty/complexity over time. Does the champion exhibit it — does
competence keep improving, or does it converge to a fixed equilibrium?

# Answer

**It converges.** Under the trusted measure (`action_effectiveness`), every seed
**rises in the first half of the run then plateaus** — the back ~50% of evolution
adds essentially nothing. This is a stable equilibrium, not open-ended evolution.

# Evidence (measured, the 4 evolved 500k worlds of champion `1dab610`)

action_effectiveness regression slope per 100k ticks, first half vs second half,
plus the net change over the second half (250k→500k):

| seed | 1st-half slope /100k | 2nd-half slope /100k | aeff@250k | aeff@500k | 2nd-half net Δ | late std (last 15) |
|---|---|---|---|---|---|---|
| 7 | +0.0083 | **−0.0024** | 0.5336 | 0.5502 | +0.0166 | 0.0331 |
| 42 | +0.0270 | **+0.0008** | 0.5591 | 0.5557 | −0.0033 | 0.0033 |
| 123 | +0.0168 | **−0.0023** | 0.5251 | 0.5252 | +0.0001 | 0.0048 |
| 2026 | +0.1056 | **+0.0007** | 0.5263 | 0.5322 | +0.0058 | 0.0089 |

- **First-half slopes are 3–130× the second-half slopes.** Competence is built
  early (the population discovers a working policy by ~150–250k) and then frozen.
- **Second-half slopes are ~0** (and negative for 2 of 4 seeds). No sustained
  improvement; seeds 42/123/2026 are flat-line plateaus (tiny late std). Only
  seed 7 stays volatile (late std 0.033) — its short-vision niche kept shifting
  (the mi_sa staircase, [[findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect]]) —
  but even it nets only +0.017 over 250k.

# Interpretation

The forager/predator ecology on the current architecture settles into a fixed
attractor: once a population finds a competent policy, selection holds it there
and novelty stops. The 500k horizon is mostly *equilibrium*, not *discovery*.
This is consistent with the plateau the iteration loop hit (iters 10–11 dry) —
the champion is near a local optimum the levers can't escape.

# Implication for the goal

**A static skill challenge will not produce open-endedness** — the population
solves it once and re-plateaus at a higher level. Sustained novelty needs a
*moving target*: a co-evolutionary / Red-Queen dynamic (predator–prey arms race
that never settles), or an environment that keeps changing what "competent"
means. This reframes Dir3: the win condition is not just a higher final
action_effectiveness, but a niche whose pressure *does not saturate* — ideally one
where the second-half slope stays positive. It also argues for a future
**open-endedness metric** (e.g. sustained second-half competence slope, or
behavioral/lineage novelty rate) as a first-class signal alongside the static
action_effectiveness headline.

# Reproduce

```
git checkout 1dab610; cargo build -p sim-cli --release
for S in 7 42 123 2026; do sim-cli new --seed $S --out artifacts/inv-$S.bin; sim-cli run-to 500000 --in artifacts/inv-$S.bin; done
# then read pillars' `granular.intervals[].action_effectiveness` per seed and
# regress the first-half vs second-half slope (50 intervals of 10k ticks each).
```

# Citations

[1] `pillars --granular` per-interval action_effectiveness over the 4 evolved 500k
worlds of champion code `1dab610`, planner branch `autoresearch/best`, 2026-06-18.
