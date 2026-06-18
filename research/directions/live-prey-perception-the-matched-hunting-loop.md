---
type: Direction
title: Live-prey perception — the correctly-matched intelligent-hunting loop
description: consume-on-kill rewards attacking LIVE prey, but brains perceive organisms only as RGB blobs conflated with food. A distinct live-prey vision channel + consume-on-kill + three-factor is the matched full loop.
priority: high
status: open
surface_area: brain-topology
tags: [sensing, predation, plasticity, intelligent-loop, culminating]
timestamp: 2026-06-17T00:00:00Z
---

# Direction

Iteration 7 ([[experiments/0007-full-hunting-loop]]) showed the full loop is
*learnable* (young brains wired perception→eat with reinforced three-factor
eligibility) but failed because of a **perception/reward mismatch**: it used the
**corpse** channel, while consume-on-kill rewards attacking **live prey** and leaves
no corpse — so the channel carried no rewarded signal and evolution dropped it.

The fix: a distinct **live-prey / organism vision channel**. Vision currently has
only Red/Green/Blue/Shape, so a live organism (potential prey) is perceived as a
colored blob indistinguishable from plant food. Add a `VisionChannel::Organism`
(or Prey) that fires for cells holding a live `Occupant::Organism` (ideally at
distance, enabling *pursuit*, not just adjacent contact). On the consume-on-kill +
three-factor base this is the correctly-matched full loop: **perceive live prey →
(learn to) pursue → attack → eat → reward-reinforce**. Hypothesis: the matched
signal is *retained* (unlike the corpse channel) because it directly precedes the
rewarded kill, so the three-factor rule consolidates a real hunting policy →
action_effectiveness & mi_sa reach/exceed champion (a CLEAN advance = a
predator-niche champion with learned hunting — open-ended evolution).

Risks (per iter3/iter7): extra sensory neurons dilute topology; the matched reward
must outweigh it. Watch the seed-123 outlier (strong champion foragers). If this
too fails to clear the gate, the perception/learning machinery is exhausted and the
binding constraint is the **metric contract**
([[directions/reconsider-intelligence-metric-under-predation]],
[[findings/prey-consumption-target-is-structurally-unreachable-in-a-stable-ecology]]) —
a human decision.
