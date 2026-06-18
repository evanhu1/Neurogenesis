---
type: Experiment
title: Intransitive hue-keyed foraging + niche construction (dense substrate) — niche-construction self-concentrates, no winding; rejected
description: Put the antisymmetric hue dynamic on the DENSE foraging substrate (25x predation) with niche construction (organisms paint plants with their hue). Ease-safe (digestion multiplier ≤1), deterministic. It does NOT wind — worse than predation: the painting is positive feedback that concentrates the population+food hue into a tight cluster (R≈0.98), killing the sin(Δ) gradient the cycle needs. Total winding -0.06 turns over 680k. Rejected. Confirms the architectural barrier: no dense substrate can host the cycle without self-concentrating.
iteration: 15
coordinator: ecology-niche
agent: forage-color
surface_area: ecology-niche
base_ref: 4cf5df6
git_ref: autoresearch/exp-0015-ecology-forage-color
status: rejected
determinism: ok
seeds: [7]
metrics: { action_effectiveness_seed7: 0.56, hue_winding_turns_over_680k: -0.06, hue_concentration_R: 0.98 }
tags: [ecology-niche, intransitive, foraging, niche-construction, open-endedness, dead-end, self-concentration]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The intransitive (no-ESS) cure for the static-fitness-landscape barrier failed on
PREDATION only because predation is too sparse a substrate
([[experiments/0014-ecology-color-dominance]]). Put it on the **dense** foraging
substrate (~25× denser) with **niche construction** to close the endogenous loop:
hue-keyed digestion (`yield = energy·(1+A·sin Δhue)/(1+A)`, ≤1 so no ease) +
organisms paint plants toward their own body color on consumption, so the optimal
forager-hue chases the population-painted food-hue → a frequency-dependent cycle.

# Change

`commit.rs` `consume_food`: plant energy ×= the ≤1 hue-keyed multiplier; record the
consumer's body_color into a serialized `food_painter_color[cell]` buffer.
`spawn/food.rs`: a regrown plant blends toward its cell's painter color. Consts
`FORAGING_COLOR_STRENGTH=0.6`, `FORAGING_PAINT_BLEND=0.5`. A=0 ⇒ baseline
byte-identical. det-check ok (P1+P2), tests ok (only known failure), no-ease proven
(`m ∈ [(1−A)/(1+A), 1]`).

# Result

**REJECTED — no winding; the painting SELF-CONCENTRATES.** Winding number (seed 7,
fine 20k intervals to 700k):

| substrate | winding over ~680k | hue concentration R |
|---|---|---|
| predation (iter14) | 0.11–0.28 turns (bounded wander) | 0.36–0.94 (spread) |
| **foraging + painting (iter15)** | **−0.06 turns (pinned)** | **0.97–0.99 (tight cluster)** |

The population hue locks into a tiny band near 0 and stays there. **Worse than
predation:** the niche-construction painting is *positive feedback* — food painted
toward consumers makes consumers and food co-converge in hue, so the within-
population hue spread collapses (R≈0.98), `sin(Δhue)≈0` for nearly all
interactions, and the cyclic gradient dies. Population stable (~150–500, healthy),
ease-safe (no explosion), aeff ~0.56 (fine) — but zero open-endedness.

# Why it failed — the architectural barrier (the deep conclusion)

There is a **fundamental tension** the engine cannot resolve:
- An intransitive cycle needs the "opponent" hue to **track the population**
  (frequency-dependence) AND the population to **stay spread** (so `sin(Δ)` bites).
- The only way to make organism–food interaction frequency-dependent is **niche
  construction (painting)** — which **concentrates** (positive feedback), killing
  the spread.
- The interaction that is *inherently* frequency-dependent without painting is
  **organism–organism** (the opponent IS the population) — but the only
  organism–organism interaction is **predation**, which is too **sparse**.

So: **the engine has no DENSE organism–organism interaction** to host the
antisymmetric cycle, and the dense organism–food interaction self-concentrates
when made frequency-dependent. Intransitive frequency-dependence — the
theoretically-correct open-endedness cure — is **not realizable in this engine's
interaction structure** without a deeper architectural change (a dense
organism–organism interaction, negative-frequency-dependent niche maintenance that
preserves a cycle rather than a fixed polymorphism, or spatial/structural niching).

# Learnings

Completes the open-endedness map
([[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]]):
static niches converge, moving hazards converge, sparse intransitivity (predation)
is too weak, dense intransitivity with niche construction (foraging) self-
concentrates. The barrier is **architectural**, not a tuning knob.

# Reproduce

`git checkout autoresearch/exp-0015-ecology-forage-color; cargo build -p sim-cli --release`;
`new --seed 7` + incremental `run-to` + `find "generation>50" --fields hue`;
the circular-mean hue stays pinned (winding number ≈ 0, R ≈ 0.98).

# Citations

[1] diff: `git show autoresearch/exp-0015-ecology-forage-color` (commit c9919a7).
[2] Seed-7 winding measurement to 700k, planner-authoritative, 2026-06-18.
