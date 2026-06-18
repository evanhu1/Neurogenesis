---
type: Experiment
title: Direct kill reward (additive) — population explosion
description: Transfer 0.5×prey energy to the attacker on a kill; proves predation lifts prey but additive energy creation explodes the ecology.
iteration: 2
coordinator: predation
agent: kill-reward
surface_area: predation-mechanics
base_ref: a90244a
git_ref: autoresearch/exp-0002-predation-kill-reward
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.00271, prey_consumption_rate: 0.03639, action_effectiveness: 0.2393, mi_sa: 0.0083, learning_slope: -0.000302 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { note: "seed-7 only; seeds 42/123 OOM-killed by population explosion; 2026 not run" }
tags: [predation, kill-reward, dead-end, explosion, breakthrough-lead]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
An attack today rewards the attacker nothing (it only leaves a corpse someone may
eat), so hunting is never selected. Granting the attacker energy on a kill makes
hunting pay → a predator niche → prey_consumption rises.

# Change
In `resolve_attack_damage` (`sim-core/src/turn/commit.rs`), on a kill transfer
`0.5 * prey.energy` to the attacker **additively** (corpse keeps full energy) and
increment `prey_consumptions_count`. Deterministic (reuses the predation hash).

# Result
**Dead-end as built, but the key lead of iteration 2.** The reward IS the right
lever — prey_consumption hit **0.036 (> the 0.025 target)** on seed 7. BUT the
reward is *additive*, so it **mints net-new energy per kill** → predation becomes
an energy *source* → delayed (~tick 300k) positive-feedback **population explosion**
(seed 7: pop 590→30574; predations/tick 1→55). The HOLD pillars collapsed
(action_effectiveness 0.565→0.239, mi_sa 0.141→0.008, plant 0.069→0.003); seeds
42 & 123 grew large enough to **OOM-kill** their runs. prey>target is a *symptom*
of runaway, not a balanced niche.

# Learnings
Proves the predation lever works (reward the attack, not the corpse) AND pinpoints
the fix: make the reward **redistributive** (move energy OUT of the corpse into
the attacker, conserving total system energy) — removes the energy-creation
feedback while still rewarding hunting. This is the top iteration-3 Direction:
[[directions/redistributive-kill-reward]]. Also a strong case of
[[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]] — free energy
removed competence pressure and brains degenerated.

# Reproduce
`git checkout fbc6a5d; cargo build -p sim-cli --release`; seed-7 `new`+`run-to 500000`
reproduces the explosion (other seeds OOM at full scale — expected).

# Citations
[1] diff: `git show fbc6a5d`
