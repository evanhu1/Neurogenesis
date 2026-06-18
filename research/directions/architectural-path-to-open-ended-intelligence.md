---
type: Direction
title: The architectural path to open-ended intelligence — an EXPANDABLE problem space + non-diluting perception
description: 19 iterations proved sustained open-ended INTELLIGENCE is not reachable by any in-loop mechanism on the current engine, because (1) a fixed simple environment has a competence ceiling (after which change is drift or bloat) and (2) the small action/perception space caps strategic depth, while adding perception DILUTES. This direction specifies the concrete architectural changes that would lift those two limits — the actionable research-scope path the negative result points to. Each requires an engine redesign (human/research-scope), not an in-loop lever.
priority: high
status: open
surface_area: engine-architecture
supported_by: [findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock, experiments/0018-ecology-pursuit-evasion, experiments/0019-ecology-display-contest]
tags: [open-endedness, architecture, redesign, unbounded-problem-space, perception, research-scope]
timestamp: 2026-06-18T00:00:00Z
---

# Why the current engine cannot do it (the proven blockers)

Across 19 iterations / 34 experiments (every paradigm + the precisely-diagnosed
cognitive contest), every mechanism either CONVERGES (drift to a fixed point) or,
if it sustains change, the change is BLOAT/DILUTION (complexity up, competence
flat-or-down). Two root causes, both architectural:

1. **Fixed, simple problem space ⇒ a competence ceiling.** A static environment
   with a small action/perception space has an optimal policy. Evolution finds it,
   then further change is neutral drift or dilutive inflation — never *more
   intelligence*. Open-ended INTELLIGENCE needs the problem to keep getting harder
   in ways that reward more skill.
2. **Small strategy space + dilutive perception ⇒ shallow, saturating arms races.**
   The only endogenous source of ever-harder problems is co-evolving agents — but
   with 4 actions and minimal sensing, "smarter" saturates fast, and ADDING
   perception/action to deepen it DILUTES competence (the minimal brain is a strong
   attractor; new channels are noise the shallow reward can't repay —
   [[experiments/0019-ecology-display-contest]], [[findings/perception-augmentation-dilutes-topology-the-best-arms-race-substrate-is-iter6]]).

So the goal needs BOTH limits lifted at once: an **expandable problem space** AND
**non-diluting capacity growth**.

# Requirement A — an EXPANDABLE / unbounded problem space

The environment (or the agent-vs-agent problem) must keep opening genuinely new
challenges as agents adapt, so there is no fixed competence ceiling. Concrete,
implementable-but-research-scope options:

1. **Compositional/hierarchical tasks that unlock new ones.** Resources/abilities
   that, once mastered, make NEW resource types reachable (a tech-tree-like
   dependency), so mastery creates new frontier. (Niche construction that
   *expands* the niche set, unlike iter15's hue-painting which *contracted* it.)
2. **Open-ended environment generation co-evolved with agents** (POET-style): the
   distribution of challenges (terrain/hazard/resource layouts) itself evolves to
   stay at the frontier of what the current population can just barely solve —
   maintained by a separate "environment" population graded by agent performance.
3. **Major-transition substrate:** allow multi-agent aggregation (groups/colonies)
   so a new level of organization (and its new problem space) can emerge. (Requires
   relaxing the no-speciation/individual-only assumptions.)

Success signal (vision-confound-safe, slow-saturation-safe): a competence measure
that keeps rising at ≥2× the usual horizon — confirmed at a 3rd horizon, since
500k slopes misled 3×. Track on a vision-invariant behavioral metric.

# Requirement B — non-diluting capacity growth

Adding brain capacity (neurons/synapses) or perception currently degrades the
minimal brain. To let intelligence grow with the expanding problem space:

1. **Modular / protected growth:** new capacity added as a *separate module* with
   near-zero initial influence (gated/zeroed gain), so it can't degrade the working
   policy and is only recruited when it earns reward — avoiding the noise-injection
   that causes dilution.
2. **A learning rule that exploits new channels within-life** (richer than the
   current unsupervised Hebbian) so added perception becomes useful fast, not noise.
   ([[directions/reward-sensitive-learning-on-the-predator-ecology]] is the seed.)
3. **Complexity cost that scales sub-linearly / is offset by the value of the new
   problem** so the cost ceiling rises with the expanding problem space (NOT a flat
   ease discount, which degrades intelligence — [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]).

# The minimal first experiment — DONE, and it FAILED (informative)

Requirement A option (1) was implemented and tested
([[experiments/0020-ecology-compositional-foraging]]): an ease-neutral 2-step
compositional resource (eat tier-1 → primed → eat tier-2). Result at 1.5M: it
SUPPRESSES competence for ~900k then plateaus at aeff ~0.38 — **BELOW** the
single-tier baseline ~0.46. Organisms learn the chain (rate→0.55, brains grow 0→12)
but the harder structure **raises the skill floor while lowering the achievable
ceiling**. So a hand-added compositional dependency is NOT an OE driver — it shifts
the equilibrium (lower), not transcends it.

**Updated implication:** the failure of the SIMPLEST expandable-problem-space option
means the requirement is stronger than "add structure": it needs a problem space
that is GENUINELY UNBOUNDED and **co-evolved with the agents** (option 2, POET-style:
the challenge distribution itself evolves to track the agents' frontier), or a
**major transition** (option 3) that opens a new organizational level. A
hand-authored finite structure (tiers) has a finite ceiling. This is a substantial
multi-component research build, and the compositional failure is evidence it cannot
be shortcut by static structure — the open-endedness must come from an
agent-coupled, never-saturating problem generator.

# Status / why this is here, not done

This is a **research-scope engine redesign**, not an in-loop lever — confirmed by
exhausting the in-loop space. The autonomous loop proved the negative and specified
this path; implementing it is a deliberate human decision (it changes the engine's
core problem structure / invariants). The two champion advances (vision-confound
fix, social transfer) and the full OE map are banked at `autoresearch/best`.
