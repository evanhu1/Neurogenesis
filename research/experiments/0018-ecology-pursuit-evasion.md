---
type: Experiment
title: Committed-attack pursuit-evasion (cognitive arms race) — first SUSTAINED brain-complexity growth, but it's dilutive inflation that DEGRADES intelligence; rejected
description: Make predation a 1-tick-committed prediction game (predator anticipates prey's move; prey evades) — an intransitive POLICY arms race. It produces the FIRST sustained non-convergent signal in the program (brain complexity neurons 12→30, synapses 13→30 through 1M, no plateau) — but competence REGRESSES cross-seed (aeff 0.5613→0.5157), because the shallow encounter-limited ecology can't make the extra complexity pay → dilutive bloat. Confirms open-ended brain GROWTH and INTELLIGENCE are in direct conflict here. Rejected.
iteration: 18
coordinator: ecology-niche
agent: pursuit-evasion
surface_area: ecology-niche
base_ref: a41b465
git_ref: autoresearch/exp-0018-ecology-pursuit-evasion
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { action_effectiveness: 0.5157, mean_neurons: 26.3, mean_synapses: 30.5, prey_consumption_rate: 0.00284 }
baseline_metrics: { action_effectiveness: 0.5613, mean_neurons: 18.0, mean_synapses: 16.0 }
tags: [ecology-niche, cognitive-arms-race, pursuit-evasion, open-endedness, complexity-growth, dilution, dead-end]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

The most goal-aligned untested paradigm: an intransitive *POLICY* competition that
targets brain COGNITION (not a passive trait). Make predation a 1-tick-COMMITTED
attack (predator commits to a cell this tick; it resolves next tick) → a
pursuit-evasion PREDICTION game (predator anticipates, prey evades unpredictably,
no fixed best strategy) → brains must keep evolving. Measure open-endedness as
sustained growth in brain COMPLEXITY + competence (the goal's real target).

# Change

`sim-core/src/turn/commit.rs`: an Attack stores a `committed_attack: Option<(q,r)>`
(serialized on OrganismState) instead of resolving inline; a new
`resolve_committed_attacks()` phase at tick start resolves it via the existing
size-gated/`deterministic_predation_sample`/consume-on-kill machinery. `off` ⇒
baseline byte-identical (verified via cmp). det-check ok (P1+P2).

# Result

**REJECTED — sustained complexity growth, but it DILUTES intelligence.** Cross-seed
500k (det-check ok):

| | champion `a41b465` | pursuit-evasion | |
|---|---|---|---|
| **action_effectiveness** | **0.5613** | **0.5157** | **−0.046 (REGRESS)** — seeds 42/123 crash to 0.38/0.49 |
| mean neurons | ~18 | **26.3** | grown |
| mean synapses | ~16 | **30.5** | grown |
| prey_rate | 0.00303 | 0.00284 | ~flat (shallow arms race) |

**The first SUSTAINED non-convergent signal in the whole program:** seed-7 brain
complexity rose monotonically through 1M (neurons 12→30, synapses 13→30, no
plateau) — every prior mechanism converged. The arms race is real (attacks land,
prey evade). **BUT it's dilutive inflation:** the complexity growth comes with a
competence REGRESSION (aeff −0.046 cross-seed, 2 seeds crash), because the
encounter-limited ecology (prey ~0.003, [[mechanisms/predation-is-encounter-limited]])
makes the arms race too SHALLOW to reward the extra capacity — so the growing
neurons/synapses are noise that DILUTES competence
([[experiments/0011-topology-brain-substrate]], "capacity without strong-enough
reward dilutes").

# The definitive conclusion (completes the OE investigation)

This is the most goal-aligned test — a cognitive arms race, measured by brain
complexity — and it confirms the core finding from the sharpest angle:
**open-ended brain GROWTH and brain INTELLIGENCE are in DIRECT CONFLICT in this
engine.** The only way to get sustained, non-convergent brain evolution is
dilutive complexity inflation; making that inflation PAY (→ intelligence) would
require a DEEP arms race (unbounded strategic depth), which the engine's shallow,
encounter-limited, simple-perception ecology cannot provide. So "open-ended
evolution **of intelligent** brains" is not achievable here: you can have
open-ended brain growth (bloat) OR sustained intelligence, not both.
([[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]])

# Caveats

- Breaks `attack_only_interacts_with_organisms_and_kills_on_success` (asserts a
  same-tick kill — no longer true by design; deliberate semantic change, "don't
  care about backwards compat"). The human maintains tests.
- Complexity growth confirmed sustained to 1M; whether it's truly unbounded vs
  very-slow-saturating needs 2M — but it's moot since it dilutes competence.

# Reproduce

`git checkout autoresearch/exp-0018-ecology-pursuit-evasion; cargo build -p sim-cli --release`;
per-seed `new`+`run-to 500000`+`pillars`+`find "generation>50" --fields neurons,synapses`.

# Citations

[1] diff: `git show autoresearch/exp-0018-ecology-pursuit-evasion` (commit 83c65a9).
[2] Cross-seed 500k + seed-7 1M complexity trajectory, planner-authoritative, 2026-06-18.
