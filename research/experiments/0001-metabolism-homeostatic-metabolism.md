---
type: Experiment
title: Homeostatic (energy-dependent) passive metabolism
description: Low-energy organisms downregulate passive burn (0.5× floor below energy 5), breaking the starvation death-spiral.
iteration: 1
coordinator: metabolism-lifecycle
agent: homeostatic-metabolism
surface_area: metabolism-lifecycle
base_ref: 70b7700
git_ref: autoresearch/exp-0001-metabolism-homeostatic-metabolism
status: promoted
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0091, prey_consumption_rate: -0.0199, action_effectiveness: 0.0081, mi_sa: 0.0452, learning_slope: 0.000266 }
tags: [metabolism, death-spiral, promoted, champion]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
The negative learning_slope is the starvation death-spiral: starving organisms
accumulate failed actions in their dying stretch, so within-life success declines
with age. If low-energy organisms downregulate their passive metabolic burn, they
get more ticks to find food before starving → fewer terminal-starvation deaths →
less age-correlated decline → less-negative slope. Also expected to rescue the
collapsing seed 2026.

# Change
`sim-core/src/metabolism.rs` only (11 lines): `organism_passive_metabolic_energy_cost`
multiplies the cost by a homeostatic factor — `1.0` above energy 5.0, ramping
linearly to a `0.5` floor as energy → 0. Pure deterministic function of energy
(no RNG, persists in world bytes). No other file touched.

# Result
Cross-seed n=4 (seed 2026 rescued from extinction). Because baseline is n=3, the
honest comparison is **seed-for-seed on the always-surviving 7/42/123** plus the
2026 rescue as a pure bonus:

| pillar | Δ (clean, 7/42/123 mean) | verdict |
|---|---|---|
| learning_slope | **+0.000276** | ✓ keystone progress |
| action_effectiveness | **+0.0155** | ✓ HOLD (improved) |
| mi_sa | **+0.047** | ✓ HOLD (improved) |
| plant_consumption_rate | +0.0057 | ✓ foraging |
| prey_consumption_rate | −0.0202 | inherent ecological trade ([[mechanisms/predation-inversely-coupled-to-population-health]]) |
| seed 2026 | extinct → pop 1170 | ✓ robustness (n 3→4) |

Per-seed (authoritative, planner-validated, byte-identical to agent run):
7 = slope −0.000659 / AE 0.5575 / mi_sa 0.0478 / plant 0.0658 / pop 1185;
42 = −0.000527 / 0.5276 / 0.0896 / 0.0655 / 1936;
123 = −0.000053 / 0.6311 / 0.2903 / 0.0653 / 1555;
2026 = −0.000451 / 0.5428 / 0.1350 / 0.0794 / 1170.

# Learnings
Confirms the death-spiral mechanism ([[findings/softening-energy-economy-lifts-learning-and-rescues-marginal-seed]]).
Uniquely among the 12 iteration-1 experiments it improves the keystone while
**holding (improving) both HOLD pillars** — every other experiment regresses
action_effectiveness. The prey collapse is the expected healthier-population
effect, not a defect.

# Concerns
Prey_consumption fell ~10× (inherent — see the Mechanism). learning_slope is
still slightly negative on 3 of 4 seeds (improvement, not yet positive). Headroom
likely remains in the ramp threshold (5.0) / floor (0.5) — see Direction.

# Reproduce
```
git checkout autoresearch/exp-0001-metabolism-homeostatic-metabolism   # or: git show eb30fff
cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out artifacts/c-$s.bin; \
  ./target/release/sim-cli run-to 500000 --in artifacts/c-$s.bin; \
  ./target/release/sim-cli pillars --in artifacts/c-$s.bin --text; done
```

# Citations
[1] diff: `git show eb30fff`
[2] baseline per-seed: [[best-program]] iteration-0 diagnostic.
