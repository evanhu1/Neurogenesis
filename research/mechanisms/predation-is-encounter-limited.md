---
type: Mechanism
title: Predation is encounter-limited (density-driven), not hunger-driven
description: A kill fires only when a predator is co-located with prey, so predation rate scales with population DENSITY and predator-prey co-location — not with how hungry/motivated organisms are.
confidence: high
supported_by:
  - experiments/0004-predation-dominant-scarcity
  - experiments/0004-predation-reliable
  - experiments/0003-predation-redistributive-kill-reward
tags: [predation, ecology, density, central]
timestamp: 2026-06-17T00:00:00Z
---

# Mechanism

`resolve_attack_damage` only fires when an organism's moved-to cell holds an
`Occupant::Organism` — i.e. a kill requires a predator to be **physically adjacent
to prey**. So predation rate is governed by **predator–prey co-location
(density)**, not by motivation/hunger. Consequences, all observed:

- **Plant scarcity SUPPRESSES predation** (not amplifies it): scarcity → lower
  carrying capacity → lower density → fewer encounters. Seed-7 predation share fell
  monotonically 17%→4% as plant scarcity rose; predations/tick collapsed 10×
  ([[experiments/0004-predation-dominant-scarcity]]).
- **Reliability tuning grows the niche but hits a ceiling**: making each encounter
  more likely to kill doubled the hunter fraction (<0.2%→~8%) but prey stayed
  ~0.002 because the *encounter supply* is the binding constraint
  ([[experiments/0004-predation-reliable]]).
- The only thing that ever drove predation high was the additive kill-reward's
  **population explosion** (iter2) — i.e. raising density (catastrophically).

**Lever implication:** to grow the predator niche you must amplify *encounters* or
preserve density — candidates: **attack reach** (attack prey within a radius, not
just the adjacent cell), **prey-seeking pursuit** (brains navigate toward perceived
prey — the genuinely *intelligent* hunting loop, combining the corpse/prey sensory
channel with motor pursuit and consume-on-kill), or density-dependent predation.
NOT scarcity, NOT mere lethality. See [[directions/amplify-the-predation-dynamic]].
