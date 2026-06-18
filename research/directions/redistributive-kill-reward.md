---
type: Direction
title: Redistributive kill reward — energy-conserving predation (TOP iteration-3 lead)
description: Make the attack rewarding by moving energy OUT of the corpse into the attacker (conserve total energy), creating a competent-predator niche without the additive-energy explosion.
priority: high
status: open
surface_area: predation-mechanics
tags: [predation, kill-reward, arms-race, intelligence, top-priority]
timestamp: 2026-06-17T00:00:00Z
---

# Direction

The single strongest lead from iteration 2. [[experiments/0002-predation-kill-reward]]
proved that rewarding the *attack* drives prey_consumption above target (0.036) —
but the *additive* reward minted free energy and exploded the population. The fix:

- On a kill, transfer a fraction of the prey's energy to the attacker **out of the
  corpse** (debit the corpse / prey energy by the same amount, or convert the kill
  directly into the attacker's meal and spawn a *smaller* corpse). **Total system
  energy conserved** → no positive-feedback runaway.
- Sweep the fraction (0.25 / 0.5 / 0.75) and confirm population stability + that
  prey_consumption rises toward 0.025 *without* an explosion.

Why it matters for the north star: a balanced predator–prey arms race is the
classic engine of open-ended intelligence, and per
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] it is the one
way to add energy/competence WITHOUT relaxing the survival constraint — hunting
must be done *skillfully* to pay, so it selects FOR competent brains (and prey are
pressured to evade). Watch action_effectiveness/mi_sa: the hoped-for result is
they HOLD or RISE (unlike every ease-adding lever).

Pair with [[directions/corpse-sensory-salience]] so brains can actually perceive
prey/corpses and learn to hunt. Surface area: `sim-core/src/turn/commit.rs`
(`resolve_attack_damage`) + corpse spawn energy. Determinism-critical (turn logic).
Note: explosions OOM at full scale — screen population stability on a short seed-7
run BEFORE any 500k confirm.
