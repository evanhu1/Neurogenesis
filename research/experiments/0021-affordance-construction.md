---
type: Experiment
title: Build action (construction/stigmergy affordance) — the root-cause FIX also fails; degenerates into wall-spam dilution; rejected
description: The root cause is a bounded affordance space; this tests the fix — expand it with a genuinely new skill type (Build: deposit persistent sheltering walls, that compose). Result: competence COLLAPSED ~20x (aeff 0.013 vs baseline 0.30-and-climbing). Agents build heavily (~10.5% of actions, walls over ~31% of the world) but as DEGENERATE SPAM, not purposeful construction — the shelter payoff was too weak/untargeted to make DISCRIMINATING building pay, so selection couldn't distinguish skill from spam, and the build tax + 7th action + world-fragmentation diluted the working policy. Confirms: even adding an affordance fails, because rewarding SKILLED use of it (not spam) is itself the open problem. Rejected.
iteration: 21
coordinator: engine-architecture
agent: affordance-construction
surface_area: engine-architecture
base_ref: a47b951
git_ref: autoresearch/exp-0021-affordance-construction
status: rejected
determinism: ok
seeds: [7]
metrics: { seed7_aeff_1M: 0.013, baseline_aeff_500k: 0.304, build_action_fraction: 0.105, built_walls: 1500, mean_neurons: 0 }
tags: [engine-architecture, affordance-space, construction, stigmergy, open-endedness, dilution, degenerate, dead-end]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The ROOT CAUSE ([[findings/open-endedness-requires-an-unbounded-affordance-space-the-engine-lacks]])
is a bounded affordance space. Test the fix: add a genuinely NEW skill type —
**construction**. A `Build` action deposits a persistent wall (energy cost) that
SHELTERS from the champion's lethal spikes (0.25× damage when wall-adjacent) and
blocks predators, and walls COMPOSE (persist + decay → dynamic structures). If
building PAYS (shelter) and composes, competence should rise via a new skill
(lifting the saturated ceiling) and the affordance space keep expanding.

# Change

`ActionType::Build` (declared last; stable indices), `try_build_wall`/`deposit_wall`/
`decay_built_walls` (`turn/construction.rs`), `is_sheltered_by_wall` in
`apply_spike_hazards`, serialized `built_wall_decay_*` state. `BUILD_ENERGY_COST=15`,
`WALL_LIFETIME_TICKS=4000`, shelter 0.25×. Toggle `CONSTRUCTION_ENABLED` off ⇒
baseline byte-identical (cmp-verified). det-check ok (P1+P2). Ease-safe (build taxed,
walls mint no food, shelter partial + decaying).

# Result

**REJECTED — competence COLLAPSED ~20× (degenerate wall-spam dilution).** Seed 7 to
1M vs apples-to-apples baseline (toggle-off, same CLI/seed/scale):

| | aeff | build frac | built walls | neurons |
|---|---|---|---|---|
| construction @1M | **0.013** | 0.105 | ~1500 (~31% of world) | 0 |
| baseline @500k | **0.304 (still rising)** | — | — | — |

Agents build HEAVILY (~10.5% of actions, ~1500 walls maintained against decay) — but
it is **pure dilution, not OE**: the shelter benefit (0.25×, untargeted) was too weak
to make *discriminating* building pay, so selection couldn't distinguish skilled
construction from SPAM; the build tax + the 7th action neuron + world-fragmentation
into movement-blocking mazes crowded out the working forage/reproduce policy
(descendants ≈ 0, neurons stay 0). No useful structures, no expanding repertoire —
degenerate Build-spamming.

# Learnings (the root-cause fix ALSO fails — the deepest confirmation)

Expanding the affordance space — the fix the root-cause finding identified — **does
not lift the ceiling**; here it COLLAPSED competence. The compounding reason: a new
affordance only helps if selection can REWARD SKILLED USE of it (not spam) — but
making the reward DISCRIMINATING (shelter pays only for purposeful enclosure, not a
single wall) is itself the hard, open problem. Absent it, the affordance degenerates
into a dilutive spam behavior. So the goal is doubly blocked: (1) the affordance set
is bounded (finite skills → saturation), and (2) even adding an affordance fails
because rewarding its skilled use is the open challenge. Mirrors iter20 (compositional
foraging lowered the ceiling) and the perception-dilution law — now confirmed for an
ACTION affordance too. A retry would need a self-limiting, enclosure-gated,
strong-payoff shelter (a tuning/design iteration) — but that, at best, yields a
higher PLATEAU (a finite construction skill), not unbounded open-endedness.

# Reproduce

`git checkout autoresearch/exp-0021-affordance-construction; cargo build -p sim-cli --release`;
seed 7 `new`+`run-to 1000000`+`state`/`pillars`; aeff ~0.013 (vs toggle-off ~0.30),
build_frac ~0.10, ~1500 walls — degenerate spam.

# Citations

[1] diff: `git show autoresearch/exp-0021-affordance-construction` (commit ca740d9).
[2] Seed-7 1M construction-vs-baseline trajectory, planner-authoritative, 2026-06-18.
