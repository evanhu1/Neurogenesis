---
type: Finding
title: ROOT CAUSE — open-ended intelligence requires an UNBOUNDED affordance space; this engine's fixed small affordance set caps it (no mechanism, incl. POET, can transcend it)
description: After 20 iterations / 35 experiments, the root cause of every failure is identified and is more fundamental than any mechanism. Open-ended intelligence needs the environment to keep demanding genuinely NEW kinds of skill. This engine has a fixed small affordance set (4 actions, simple vision, ~4 environment mechanics) → a finite set of skill TYPES → intelligence saturates once they're mastered, and extra capacity becomes bloat. No in-engine mechanism — intransitive dynamics, cognitive arms races, compositional resources, or even POET-style environment co-evolution — can manufacture unbounded skill-complexity from a bounded affordance space. Achieving the goal requires designing-in an unbounded affordance space — a fundamental engine redesign.
confidence: high
status: active
supported_by: [findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock, experiments/0018-ecology-pursuit-evasion, experiments/0019-ecology-display-contest, experiments/0020-ecology-compositional-foraging, experiments/0021-affordance-construction]
tags: [open-endedness, root-cause, affordance-space, architecture, capstone, research-scope]
timestamp: 2026-06-18T00:00:00Z
---

# The question, finally answered at the root

Why did ALL 20 iterations / 35 experiments fail to produce open-ended INTELLIGENCE
(sustained competence growth) — across intransitive dynamics, spatial structure,
co-evolution, cognitive arms races, perception augmentation, compositional
resources, AND the precisely-diagnosed dense brain-controlled contest? The reason
is not the mechanism. It is the **affordance space** the mechanisms operate in.

# Root cause: a bounded affordance space ⇒ a competence ceiling

Open-ended intelligence requires the environment to keep posing genuinely **new
kinds** of problems that demand **new kinds** of skill — an *unbounded* skill space.
This engine offers a **fixed, small affordance set**:
- **Actions:** 4 (Idle/Turn/Forward/Eat/Attack/Reproduce — ~4 meaningful).
- **Perception:** food-direction over 3 short rays + a few channels.
- **Environment mechanics:** terrain (Perlin walls), spikes, food (forage), corpses
  (predation). ~4 interaction types.

This yields a **finite set of skill TYPES**: navigate, avoid-hazard, forage, hunt,
and combinations. Once an agent masters them, *harder instances* (more walls,
denser spikes, sequential foraging) do **not** demand new skill types — so:
- competence **saturates** at the ceiling set by the available skills, and
- extra brain capacity becomes **bloat** (the "capacity dilutes" law) — which is
  *exactly* what every experiment showed (converge, plateau, or grow-as-dilution).

Compositional resources (iter20) made this vivid: a 2-step chain raised the skill
FLOOR but the achievable CEILING was LOWER — a harder *instance* of foraging, not a
new skill type, with its own (lower) ceiling.

# Why even POET cannot transcend it (the obvious next idea, ruled out by reason)

POET / open-ended environment co-evolution generates a *curriculum within* an
environment space. Its open-endedness in the literature came from environment
spaces with effectively **unbounded morphological complexity** (e.g. continuous
bipedal-walker terrain). If the environment space has **bounded skill-complexity**
(a finite set of affordance types, as here), the co-evolved curriculum **also
saturates** — it can only generate harder *instances* of the same finite skills.
So a POET build on this engine would, by this same root cause, plateau — at large
cost. The root limit is the affordance space, not the search/curriculum over it.

# The actual requirement (a fundamental engine redesign)

Open-ended intelligence needs an **unbounded affordance space** — a substrate where
mastering skills *opens genuinely new kinds of skill*, without bound. Examples (any
ONE could suffice; all are deep redesigns, not in-loop levers):
- **Open-ended action space:** composable/parameterized actions, tool use,
  construction/niche-building, or a (near-)Turing-complete action grammar — so new
  behaviors keep becoming expressible.
- **Open-ended morphology:** evolvable bodies/sensors/effectors (not a fixed brain
  on a fixed body) — so the agent can grow into new affordances.
- **Open-ended environment mechanics:** an environment whose *rules* (not just
  parameters) keep expanding — agents can create artifacts/signals that become new
  environmental features for others.
- **Open-ended social/communication channel:** an unbounded signalling space agents
  co-construct — language-like, where meaning keeps elaborating.

Each is a research program. The autonomous loop's contribution is to have
**exhaustively localized the bottleneck**: it is not selection, diversity, the
metric, the substrate-density, the interaction form, or the search method — it is
the **dimensionality/unboundedness of what an agent can DO and SENSE**. Designing
that in is the prerequisite to open-ended evolution of intelligent brains; no
mechanism downstream of a bounded affordance space can produce it.

# UPDATE (iter21): even the FIX (adding an affordance) fails — the meta-problem

The fix this finding identified — expand the affordance space — was tested directly:
a `Build`/construction action (deposit sheltering, composable walls)
([[experiments/0021-affordance-construction]]). It did NOT lift the ceiling; it
**collapsed competence ~20×** (aeff 0.013 vs baseline 0.30). Agents built heavily
but as **degenerate spam**, because the shelter payoff couldn't be made to reward
*skilled, discriminating* construction over spam. This reveals the deeper, doubly-
binding structure:
1. The affordance set is bounded ⇒ finite skills ⇒ saturation; AND
2. **even ADDING an affordance fails, because rewarding its SKILLED use (not spam /
   not dilution) is itself the open problem.** A richer affordance only helps if the
   selective environment can discriminate skilled use — which requires a problem that
   *demands and grades* that use, i.e. the unbounded-problem-space requirement again,
   one level up.

So open-endedness here is blocked at BOTH the affordance level and the
reward/problem level — the two are entangled: you can't reward open-ended skill
without an open-ended problem, and you can't pose an open-ended problem without
open-ended affordances. Breaking this circular dependency is the essence of the
open-ended-evolution grand challenge, and it is fundamentally beyond an in-loop
parameter/mechanism change on a fixed engine. The autonomous loop has localized the
blocker to this precise circular core.

# What the loop DID achieve (banked, real)

- **Two champion advances** (intelligence measurably up): vision-confound fix
  (+0.0088 aeff) and the zero-sum social transfer (+0.0091 aeff, max diversity).
- A complete, reproducible map of the open-endedness frontier across 20 iterations,
  with the 2×-horizon discipline that caught three slow-saturation false positives —
  ending at this root-cause requirement, not at "unknown."

# Citations

[1] The full experiment series 0012–0020 + the capstone
[[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]];
[[directions/architectural-path-to-open-ended-intelligence]]. Planner-authoritative,
2026-06-18.
