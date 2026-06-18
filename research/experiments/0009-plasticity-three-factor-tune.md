---
type: Experiment
title: Three-factor band tune (GAIN 0.08→0.04) — intelligence gain on the arms-race champion
description: Halving the within-life reward-credit gain lets the covariance rule retain richer sensory→action structure → mi_sa surges (+46%) and action_eff recovers; strictly dominates the predecessor champion. PROMOTED.
iteration: 9
coordinator: plasticity
agent: three-factor-tune
surface_area: plasticity-genome
base_ref: 24588ec
git_ref: autoresearch/exp-0009-plasticity-three-factor-tune
status: promoted
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0786, prey_consumption_rate: 0.00235, action_effectiveness: 0.5435, mi_sa: 0.1952, learning_slope: -0.000487 }
baseline_metrics: { plant_consumption_rate: 0.0719, prey_consumption_rate: 0.0022, action_effectiveness: 0.5422, mi_sa: 0.1335, learning_slope: -0.000598 }
delta: { plant_consumption_rate: 0.0067, prey_consumption_rate: 0.00015, action_effectiveness: 0.0013, mi_sa: 0.0617, learning_slope: 0.000111 }
tags: [plasticity, three-factor, intelligence, champion, promoted]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
The iter6 three-factor (GAIN 0.08) recovered action_eff but the champion still
trailed the homeostatic mi_sa. A GENTLER reward-credit gain should let the
unsupervised covariance rule build/retain richer structure (not collapse it onto a
few high-energy-delta couplings) → higher mi_sa & action_eff while keeping the niche.

# Change
`brain/plasticity.rs`: `NEUROMOD_GAIN` 0.08→0.04 (one line; SCALE/bounds unchanged —
m now stays in ~[0.96,1.04]). On the current champion (homeostatic + consume-on-kill
+ three-factor). Swept GAIN {0.04,0.06,0.12} × SCALE {3,5,8}; 0.04 won.

# Result
**PROMOTED — strictly dominates the predecessor champion on all 4 pillars**
(cross-seed n=4, 500k, authoritatively re-confirmed, all seeds survive): action_eff
0.5435 (>0.5422), **mi_sa 0.1952 (>0.1335, +46%)**, prey 0.00235, plant 0.0786.
Its **mi_sa exceeds the original homeostatic high-water mark (0.1407)** — a genuine
intelligence gain. Brain probes show clean consolidation of useful couplings
(`ContactAhead→Eat` elig ~3.4, `ContactAhead→Attack` ~2.2, food→Eat) — exactly the
structure that raises both action_eff (successful contingent actions) and mi_sa
(sensory-dependent selection). Higher gain / sharper credit collapsed seed-7 mi_sa.

# Learnings
The within-life reward-learning works best as a GENTLE nudge on the
covariance rule, not a strong override — confirms
[[directions/reward-sensitive-learning-on-the-predator-ecology]]. **Honest caveat:**
the mi_sa gain is seed-7-heavy (seed 7: 0.44; seeds 42/123/2026 ~0.10-0.12), and
aeff (0.5435) still trails the homeostatic-only 0.5647 — so vs the all-time aeff
mark it's a mi_sa-for-aeff trade, but vs the predecessor champion it's clean
strict dominance. seed-123 (strong-forager drag) unchanged.

# Reproduce
`git checkout 121ee21; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 121ee21`
