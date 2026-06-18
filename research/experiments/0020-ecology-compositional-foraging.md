---
type: Experiment
title: Compositional 2-step primed foraging (expandable problem space, OE Requirement A) — harder structure LOWERS the ceiling; rejected
description: The first experiment of the architectural path: an ease-neutral compositional resource (eat tier-1 → primed → eat tier-2 while primed) to raise the skill ceiling. Result (to 1.5M, 2x+ horizon): it SUPPRESSES competence for most of the run, then transitions late to a sustained-but-LOWER plateau (aeff ~0.38 vs single-tier baseline ~0.46). Organisms DO learn the chain (rate 0.20→0.59, brains grow), but the harder structure lowers the achievable ceiling rather than raising it. Compositional unlocking is not an OE driver here. Rejected.
iteration: 20
coordinator: ecology-niche
agent: compositional-foraging
surface_area: engine-architecture
base_ref: 9b2b25d
git_ref: autoresearch/exp-0020-ecology-compositional-foraging
status: rejected
determinism: ok
seeds: [7]
metrics: { seed7_aeff_1p5M: 0.388, baseline_aeff_1p5M: 0.458, chain_success_rate: 0.55, mean_neurons: 12 }
tags: [engine-architecture, compositional, expandable-problem-space, open-endedness, dead-end]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The architectural path ([[directions/architectural-path-to-open-ended-intelligence]])
says OE needs an EXPANDABLE problem space (no fixed competence ceiling). Test the
simplest version (Requirement A, option 1): a COMPOSITIONAL sequential-dependency
resource. Eating a tier-1 plant "primes" the organism (timer); a tier-2 plant
yields energy ONLY while primed → exploiting it requires the 2-step CHAIN (tier-1 →
primed → tier-2), a deeper skill than single-step foraging. EASE-NEUTRAL: tier-2
REPLACES 1/3 of tier-1 (same total food/energy), so a skilled chainer gets the same
calories and an unskilled one gets less (selection tightens). Hope: the deeper skill
raises the competence ceiling → continued growth.

# Change

`FoodKind::PrimedPlant` (tier-2, distinct visual reusing existing RGB vision),
`OrganismState.primed_ticks`; `consume_food` primes on tier-1 and gates tier-2 yield
on primed; per-tick prime decrement; deterministic per-cell tier-2 marking
(`hash_2d`, replacing 1/3 of tier-1, no new RNG); toggle in the hidden
`FoodEcologyPolicy`. det-check ok; toggle-off ⇒ baseline byte-identical (semantic
fingerprint cmp identical, seeds 7 & 42); ease-neutral verified (t0 plants/energy/
coverage identical to baseline).

# Result

**REJECTED — the harder structure LOWERS the ceiling.** Seed 7 to 1.5M (exp vs
single-tier baseline control, same config):

| horizon | exp aeff | baseline aeff | chain rate | exp neurons |
|---|---|---|---|---|
| 50k–900k | ~0.04 (SUPPRESSED) | transitions by ~300k → ~0.45 | 0.2→0.4 | ~0 |
| 1.0M | 0.293 | 0.452 | 0.55 | 3.8 |
| 1.5M | **0.388** | **0.458 (gently rising)** | 0.51 | 11.9 |

- Compositional foraging **suppresses** the competence transition for ~900k (stuck
  near the early ~0.04), while the single-tier baseline transitions by ~300k and
  sustains ~0.45–0.50.
- Exp-0020 eventually transitions (~900k–1M) to a chain-using regime and SUSTAINS it
  to 1.5M (aeff ~0.35–0.41) — but **below** the baseline. The 2× check confirms this
  is a real sustained-lower plateau, NOT a false-positive transient.
- Organisms genuinely learn the chain (success rate 0.20→0.55–0.59; tier-2 actively
  eaten; brains grow 0→12 neurons to support it) — the mechanism is engaged, not
  bypassed. But mastering the harder task only recovers a population to a *lower*
  competence than the simpler task. The harder structure raises the skill FLOOR,
  not the achievable CEILING.

# Learnings (the architectural path's first step fails)

Making the problem structurally harder (ease-neutral, compositional) **tightens
selection but does not yield open-ended competence growth** — the compositional task
has its OWN competence ceiling, *below* the simpler task's, and the population
reaches it then stops. So "expandable problem space via compositional unlocking"
(the simplest Requirement-A option) is NOT an OE driver on this engine. This is
consistent with and strengthens the capstone
([[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]]):
adding structure shifts the equilibrium (often lower); it does not transcend it.
The remaining Requirement-A options (POET-style co-evolved environment generation;
major-transition aggregation) are far larger multi-component research programs; the
failure of the simplest option is strong evidence they too require capabilities
beyond an in-loop mechanism (a genuinely unbounded, agent-co-evolved problem
generator, not a hand-added structure).

# Caveats

- Single seed (7, run deep to 1.5M — the ceiling test seed). The suppression-then-
  lower-plateau shape is robust across the trajectory; cross-seed would refine but
  it plateaus BELOW baseline, so it's a clear reject, not a promote candidate.
- Breaks `food_ecology_cycles_consumption_and_regrowth_on_fertile_tile` (the test's
  fertile cell now hashes to tier-2 — asserts the old single-tier invariant;
  deliberate food-ecology change, "don't care about backwards compat").

# Reproduce

`git checkout autoresearch/exp-0020-ecology-compositional-foraging; cargo build -p sim-cli --release`;
seed 7 `new`+`run-to 1500000`+`pillars`; aeff plateaus ~0.38 below the single-tier ~0.46.

# Citations

[1] diff: `git show autoresearch/exp-0020-ecology-compositional-foraging` (commit a083f11).
[2] Seed-7 1.5M exp-vs-baseline trajectory, planner-authoritative, 2026-06-18.
