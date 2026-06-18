---
type: Experiment
title: Dense organism–organism color-cyclic ADJACENCY MORTALITY (pure damage) — wound 0.85 turns then locked; rejected
description: The missing dense organism–organism intransitive interaction — each tick an organism takes pure damage from hue-dominant adjacent neighbors. It drove the population hue 0.85 of a full rotation (far past predation's 0.11-0.28), confirming the architecture CAN support substantial endogenous non-stationarity. But it then LOCKED (R→0.98): the no-ease constraint forces all-damage, which breaks the antisymmetry → race-to-dominant-hue → convergence. The fix (zero-sum transfer) is exp-0017.
iteration: 16
coordinator: ecology-niche
agent: social-color
surface_area: ecology-niche
base_ref: 20c581d
git_ref: autoresearch/exp-0016-ecology-social-color
status: rejected
determinism: ok
seeds: [7]
metrics: { hue_winding_turns_over_300k: 0.79, hue_concentration_R_final: 0.80, action_effectiveness_seed7: 0.525 }
tags: [ecology-niche, intransitive, organism-organism, open-endedness, near-miss, convergence]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The capstone said open-endedness needs a DENSE *organism–organism* intransitive
interaction (predation is too sparse; organism–food self-concentrates via
painting). Add one: each tick, an organism takes pure damage
`SOCIAL_DAMAGE · Σ max(0, sin(hue_neighbor − hue_self))` from its ≤6 hex-adjacent
organisms whose hue dominates it. Dense (adjacency ≫ predation), strong (health/
death), frequency-dependent (opponents ARE the population — no painting needed, so
spread is preserved). Reuses `color_hue`; deterministic snapshot-then-apply; pure
damage (no energy transfer = no ease). det-check ok; A=0 byte-identical.

# Result

**Best winding yet — then LOCKED.** Seed 7, 20k intervals to 300k: the unwrapped
mean hue accumulated **~0.85 turns of real winding in the first ~120k** (vs
predation 0.11–0.28, foraging 0.06) — the dense antisymmetric kick worked exactly
as predicted early. **But from ~120k it stalled at 0.79–0.86 turns**, mean hue
parked, concentration **R climbing to 0.97–0.98** — the population re-converged
into a tight color cluster and the rotation froze. Bounded, not the sustained
unbounded winding wanted. Pop stable (~1000), aeff ~0.525.

# Why it locked (the key insight → exp-0017)

The **no-ease constraint forced PURE DAMAGE** (health loss, no transfer). Pure
damage is *all-damage*: the universal best response is "be the most-leading hue so
you damage others and take none," so everyone races to the leading edge → the hue
distribution collapses (R→0.98) → `sin(Δ)≈0` for all pairs → the torque dies →
frozen. All-damage **breaks the antisymmetry** a sustained cycle needs. The fix:
make it a **ZERO-SUM energy transfer** (winner gains what loser loses) — conservative
(NOT ease, like predation's consume-on-kill), which restores the antisymmetric
structure → [[experiments/0017-ecology-social-transfer]].

# Reproduce

`git checkout autoresearch/exp-0016-ecology-social-color; cargo build -p sim-cli --release`;
`new --seed 7` + incremental `run-to` + `find "generation>50" --fields hue`; the
unwrapped winding reaches ~0.85 turns by 120k then plateaus as R→0.98.

# Citations

[1] diff: `git show autoresearch/exp-0016-ecology-social-color` (commit c73f3f3).
[2] Seed-7 winding probe to 300k, planner-authoritative, 2026-06-18.
