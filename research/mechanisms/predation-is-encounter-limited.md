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
preserve density. BUT iteration 5 added the crucial qualifier: a **free encounter
buff DE-SKILLS hunting and collapses intelligence.** Attack-reach (R=2) grew the
niche (prey 0.0053) but let attacks land without aiming/positioning → action_eff
0.51→0.31, mi_sa 0.17→0.03 ([[experiments/0005-predation-attack-reach]]). So the
encounter amplification must come from a **skill that is selected/learned**, not a
free buff: i.e. **prey-seeking pursuit** (brain perceives prey → navigates to it →
attacks adjacent → eats — the genuinely intelligent hunting loop), which requires
within-life reward-learning to acquire ([[directions/reward-sensitive-learning-on-the-predator-ecology]]).
NOT scarcity, NOT lethality, NOT free reach. See [[directions/amplify-the-predation-dynamic]].
