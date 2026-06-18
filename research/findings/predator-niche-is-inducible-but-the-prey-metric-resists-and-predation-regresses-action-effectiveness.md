---
type: Finding
title: A predator niche IS inducible (energy-conserving) and evolves hunting brains — but the prey metric resists the target and predation regresses action_effectiveness
description: Energy-conserving kill rewards reliably grow a real predator niche with emergent hunting brains and (where strong) higher mi_sa; yet prey_consumption_rate stays ~5–10× short of target and action_effectiveness regresses.
confidence: high
status: active
supported_by:
  - experiments/0003-predation-redistributive-kill-reward
  - experiments/0003-predation-consume-on-kill
  - experiments/0003-sensing-corpse-salience
tags: [predation, arms-race, intelligence, metric, central]
timestamp: 2026-06-17T00:00:00Z
---

# Finding

The iteration-2 lead is confirmed and its limits mapped. Two distinct
energy-conserving kill rewards (redistributive; consume-on-kill) BOTH:

- **Induce a genuine predator niche** with NO population explosion — predation
  26–48% of deaths, pure-predator phenotypes (plant=0), and **evolved contextual
  hunting brains** (`ContactAhead→Eat` w≈1.5, positive Attack logit; with the
  corpse sensory channel, `Corpse→Eat` reflexes). This is real open-ended
  behavioral evolution — a new niche that did not exist in the champion.
- Where the niche is strongest, **mi_sa RISES** (consume-on-kill seeds 123/2026:
  0.18 / 0.36 vs 0.141 baseline) — predation selects for information-rich
  decision-making. Foraging (plant) can also rise.

**But neither advances the champion, for two structural reasons:**

1. **The `prey_consumption_rate` metric resists the target.** It is
   `prey_consumptions / total_actions`; a hunting *minority* (~25% of the
   population) barely moves a denominator dominated by foraging actions, so prey
   tops out at ~0.004–0.005 — ~5–10× below the 0.025 target. (consume-on-kill also
   removes a corpse source by suppressing the kill-corpse, hurting low-predation
   seeds.) The mechanism works; the *proxy* undervalues it.
2. **`action_effectiveness` regresses under predation** (−0.056 both variants),
   plausibly because predation kills organisms younger → the death-cohort the
   metric averages over has lower mean competence, and predation adds chaos
   (being attacked disrupts actions). Note this *opposes* mi_sa, which rises —
   the two "intelligence" pillars disagree under predation.

**Strategic implications (toward open-ended evolution of intelligent brains):**
- The predator–prey arms race is the right *mechanism* — it produces emergent
  hunting intelligence and raises mi_sa, unlike every ease-adding lever
  ([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]).
- To make it *register*, predation must become a larger share of the ecology
  (so prey_rate rises) — [[directions/amplify-the-predation-dynamic]] — and/or the
  evaluation should weight `mi_sa` (which predation improves) over
  `action_effectiveness` (which the death-cohort effect penalizes):
  [[directions/reconsider-intelligence-metric-under-predation]].
- Salience alone is a validated-but-weak enabler; its value is in *combination*
  with a reward (planner gate-test).
