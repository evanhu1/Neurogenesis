---
type: Experiment
title: Full hunting loop — consume-on-kill + three-factor + corpse sensory channel
description: Stacking the corpse channel onto the predator+learning base FAILS — perception/reward MISMATCH (consume-on-kill leaves no corpse and targets live prey, so the corpse channel carries no live-prey signal). Pinpoints the fix: a LIVE-PREY channel.
iteration: 7
coordinator: synthesis
agent: full-hunting-loop
surface_area: predation-x-sensing-x-plasticity
base_ref: 696def5
git_ref: autoresearch/exp-0007-full-hunting-loop
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0713, prey_consumption_rate: 0.00252, action_effectiveness: 0.5264, mi_sa: 0.0925, learning_slope: -0.000515 }
baseline_metrics: { plant_consumption_rate: 0.0690, prey_consumption_rate: 0.0018, action_effectiveness: 0.5647, mi_sa: 0.1407, learning_slope: -0.000423 }
delta: { plant_consumption_rate: 0.0023, prey_consumption_rate: 0.0007, action_effectiveness: -0.0383, mi_sa: -0.0482, learning_slope: -0.000092 }
tags: [predation, sensing, plasticity, perception-reward-mismatch, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
With within-life reward-learning present (three-factor, which recovered aeff in
iter6), adding a distinct corpse/prey sensory channel gives the learning a clean
signal → brains learn see-prey→hunt → action_eff & mi_sa reach champion (clean gate).

# Change
consume-on-kill + three-factor base (696def5) + cherry-pick the corpse
`VisionChannel::Corpse` (exp-0003-sensing). Tested band GAIN 0.08 (kept) vs 0.15 (worse).

# Result
**FAIL — but precisely diagnostic.** The loop was *learnable early*: young (gen 0-1)
brains wired `visF:Corpse→Eat` with high three-factor eligibility (~1.15) — direct
evidence the perception+reward loop CAN form. But the 500k-evolved population did
NOT retain it (only ~2/40 prey-eaters keep any corpse synapse, wired to TurnRight,
zero eligibility). **Root cause — a perception/reward MISMATCH:** consume-on-kill
leaves NO corpse on a predation kill (attacker eats live prey directly) and live
prey reads corpse=0, so the corpse channel fires only on incidental carrion and
carries **no live-prey signal**. The 3 extra neurons diluted topology below both
champion and the g08 base (aeff 0.5264, mi_sa 0.0925). Only prey rate rose (0.0025).

# Learnings
The perception channel must MATCH the reward. consume-on-kill rewards attacking
**live prey** — so the missing primitive is a **live-prey/organism vision channel**
(organisms are currently perceived only as RGB blobs conflated with food), NOT a
corpse channel. → [[directions/live-prey-perception-the-matched-hunting-loop]].
Confirms the iter3 salience-dilution pattern: a sensory channel only helps if it
carries a *rewarded* signal ([[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]).

# Reproduce
`git checkout 651c5c5; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 651c5c5`
