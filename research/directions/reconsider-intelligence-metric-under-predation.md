---
type: Direction
title: Reconsider which metric measures "intelligence" under predation (mi_sa vs action_effectiveness)
description: Predation raises mi_sa (information-rich decisions) but regresses action_effectiveness (younger death-cohort + chaos). The two HOLD pillars disagree; the goal favors mi_sa.
priority: medium
status: open
surface_area: evaluation
tags: [metric, intelligence, predation, evaluation]
timestamp: 2026-06-17T00:00:00Z
---

# Direction

Iteration 3 surfaced a metric conflict (a planner/human decision, not an engine
change). Under an evolved predator niche
([[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]]):

- **`mi_sa` RISES** where predation is strongest (consume-on-kill seeds 123/2026:
  0.18 / 0.36 vs 0.141) — predation selects for information-rich, context-sensitive
  decisions: a strong "intelligent brain" signal.
- **`action_effectiveness` REGRESSES** (−0.056) — but largely as an artifact of
  predation killing organisms younger (lower death-cohort mean competence) plus
  attack-chaos disrupting actions, *not* obviously a loss of competence.

For the north star (open-ended evolution of *intelligent* brains), `mi_sa`
(mutual information between sensory state and action) is arguably the truer
intelligence proxy, and `action_effectiveness` (fraction of actions that "succeed",
averaged over the death cohort) penalizes exactly the predator-prey arms race we
want. **Options for the human/planner:** (a) weight `mi_sa` over
`action_effectiveness` in the HOLD criterion when predation is present; (b)
redefine `action_effectiveness` to be age/cohort-normalized so predation deaths
don't deflate it; (c) keep both but accept an `action_effectiveness` dip if `mi_sa`
and emergent hunting behavior rise. Flag to the user before changing the eval
contract. Until resolved, the disciplined gate still protects both pillars — so
predation experiments won't auto-advance the champion.
