---
type: Experiment
title: Faster plant regrowth (findability)
description: Lower food_regrowth_interval so plants reappear sooner — more foraging encounters without cutting per-bite energy.
iteration: 1
coordinator: food-ecology
agent: faster-regrowth
surface_area: food-ecology
base_ref: 70b7700
git_ref: autoresearch/exp-0001-food-faster-regrowth
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0719, prey_consumption_rate: null, action_effectiveness: 0.4771, mi_sa: 0.1136, learning_slope: -0.000407 }
baseline_metrics: { plant_consumption_rate: 0.0599, prey_consumption_rate: 0.0217, action_effectiveness: 0.5566, mi_sa: 0.0955, learning_slope: -0.000689 }
delta: { plant_consumption_rate: 0.0120, action_effectiveness: -0.0795, mi_sa: 0.0181, learning_slope: 0.000282 }
tags: [food-ecology, findability, rejected]
timestamp: 2026-06-16T00:00:00Z
---

# Hypothesis
Findability lever: faster regrowth → more plant encounters → higher
plant_consumption without per-bite energy cuts (decouples foraging from
starvation).

# Change
`food_regrowth_interval` lowered (config default, both copies).

# Result
Best plant of the food cohort (**0.0719, +0.012**) and rescued seed 2026. BUT
action_effectiveness fell (0.4771; seed 7 anomalously 0.3956) — findability
appears to depress action quality (less selective pressure / dilution). Plant gain
is real but still far from the 0.10 target; fails HOLD. Coordinator
cross-validated its runner byte-identical to the agent.

# Learnings
Findability does raise foraging modestly, confirming the decoupling direction —
but at an action_effectiveness cost, and not nearly enough to reach 0.10 alone.
→ [[directions/food-findability-with-hold-pillar-guard]].

# Concerns
AE regression (partly seed-composition, partly real seed-7 drop); prey not
captured.

# Reproduce
```
git show 7b9e876 ; cargo build -p sim-cli --release
for s in 7 42 123 2026; do ./target/release/sim-cli new --seed $s --out a-$s.bin; ./target/release/sim-cli run-to 500000 --in a-$s.bin; ./target/release/sim-cli pillars --in a-$s.bin --text; done
```

# Citations
[1] diff: `git show 7b9e876`
