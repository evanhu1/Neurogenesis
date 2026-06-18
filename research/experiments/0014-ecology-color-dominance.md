---
type: Experiment
title: Cyclic (intransitive) color-dominance in predation — theoretically non-convergent but too sparse a substrate; rejected
description: Reweight predation success by 1+A*sin(Δhue) of attacker-vs-prey body color. sin is antisymmetric → no body-color is globally best → the game has NO ESS → the population mean hue should orbit forever (endogenous non-stationarity). It does NOT in practice: the hue does a bounded random walk (0.11 turns over 460k at A=0.5; 0.28 over 380k at A=0.9), not a directed orbit, because predation is encounter-limited (~0.003/tick) so the color selection is overwhelmed by drift. Strength doesn't fix it. Rejected. The mechanism is right; the substrate (predation) is too sparse.
iteration: 14
coordinator: ecology-niche
agent: color-dominance
surface_area: ecology-niche
base_ref: 47a6111
git_ref: autoresearch/exp-0014-ecology-color-dominance
status: rejected
determinism: ok
seeds: [7]
metrics: { action_effectiveness_seed7: 0.46, hue_winding_turns_over_460k_A0.5: 0.11, hue_winding_turns_over_380k_A0.9: 0.28 }
baseline_metrics: { action_effectiveness: 0.5522 }
tags: [ecology-niche, intransitive, frequency-dependent, open-endedness, color, dead-end, substrate-sparsity]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The open-endedness barrier is a STATIC fitness landscape
([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).
An **intransitive** (frequency-dependent) interaction has no ESS and so cannot
settle — its optimum endogenously chases the population. Key on the only
continuous, heritable, PERCEIVED trait (body_color → hue), via the only tight
selection interaction (predation), respecting the no-speciation invariant. Design
by a dedicated scoping agent; the math (antisymmetric payoff `sin(Δ)` ⇒ no ESS ⇒
replicator orbits forever) is sound.

# Change

`commit.rs` `resolve_attack_damage`: `predation_success = (pred_size/prey_size) ·
(1 + A·sin(hue_prey − hue_pred))`, clamped. `A` = `COLOR_DOMINANCE_STRENGTH`
(0.5, then 0.9). Added `color_hue` + `predation_color_dominance` to `sim-types`;
exposed `hue`/`cr`/`cg`/`cb` as `sim-cli` read fields to measure the winding
number. Energy on a kill unchanged (no ease); deterministic (pure fn of persisted
genome; reuses the predation hash); no species/types (continuous hue). det-check ok.

# Result

**REJECTED — no winding.** The convergence-immune signal (unwrapped cumulative
rotation of the population circular-mean hue, seed 7, fine 20k intervals):

| A | total unwrapped rotation | over | = turns |
|---|---|---|---|
| 0.5 | +0.666 rad | 460k ticks | **0.11** |
| 0.9 | −1.768 rad | 380k ticks | **0.28** |

The unwrapped hue does a **bounded random walk** (wanders ~[−2.3, +1.4] rad,
mean-reverting near 0), NOT a directed orbit. It does not keep rotating. Stronger
A roughly doubles the wander rate but does not produce winding. Color variance
stays healthy (concentration R = 0.36–0.94 — NOT collapsed), so the failure is not
diversity loss. Competence regresses too (seed-7 aeff ~0.46 vs champion 0.5522,
slightly declining).

# Why it failed (the key insight)

The theory is correct (antisymmetric ⇒ no ESS); the **substrate is too sparse.**
Predation is encounter-limited (`mechanisms/predation-is-encounter-limited`,
prey-rate ~0.003/organism/tick), so the color-dominance term applies to too few
life events to make hue a strong-enough selective force. **Genetic drift dominates
the rare color-selection signal**, so the population hue diffuses (bounded random
walk) instead of orbiting (directed winding). Selection strength (A) cannot fix an
*event-frequency* bottleneck. This is exactly the scoping agent's flagged risk:
"a single intransitive axis on a RARE interaction may produce neither rotation nor
co-moving competence."

# Learnings

Endogenous non-stationarity (the right idea for open-endedness) needs a **DENSE
interaction substrate**, not a rare one. The only dense organism-level interaction
in the engine is **foraging/food**, which is **policy-locked** in sim-config (the
hidden food-ecology policy). So within the current invariants, the intransitive
lever has no dense substrate to ride on → it cannot realize open-endedness. **The
binding constraint is now the food-ecology policy lock**
([[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]]).
Realizing open-endedness likely requires putting an intransitive / niche-construction
dynamic on the food layer — a deliberate relaxation of that invariant (a human call).

# Reproduce

`git checkout autoresearch/exp-0014-ecology-color-dominance; cargo build -p sim-cli --release`;
`new --seed 7` + incremental `run-to` with `find "generation>50" --fields hue`;
compute the circular-mean hue per interval and its UNWRAPPED cumulative rotation
(winding number). It stays bounded.

# Citations

[1] diff: `git show autoresearch/exp-0014-ecology-color-dominance` (commit f484c6a).
[2] Seed-7 fine winding measurements (A=0.5 and A=0.9), planner-authoritative, 2026-06-18.
