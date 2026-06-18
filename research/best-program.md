---
type: BestProgram
title: Current best program
description: The current research champion — a concrete git ref the next iteration forks from.
git_ref: autoresearch/best
iteration: 6
metrics:
  plant_consumption_rate: 0.0690
  prey_consumption_rate: 0.0018
  action_effectiveness: 0.5647
  mi_sa: 0.1407
  learning_slope: -0.000423
lineage:
  - experiments/0001-metabolism-homeostatic-metabolism
tags: [autoresearch, champion]
timestamp: 2026-06-16T00:00:00Z
---

# Current best program

> **Iterations 2 & 3 produced NO champion advance** (champion code unchanged at
> `eb30fff` homeostatic metabolism). Iter2: all dead-ended; central finding
> [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]. **Iter3
> (predator–prey arms race):** energy-conserving kill rewards reliably evolve a
> real predator niche with hunting brains (and rising `mi_sa`) — genuine
> open-ended evolution — but the proxy metrics block it
> ([[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]]):
> `prey_consumption_rate` resists a hunting minority and `action_effectiveness`
> penalizes predation's younger death-cohort. Top lead now
> [[directions/amplify-the-predation-dynamic]]. Metrics below unchanged.
>
> **Iters 4–6 (no advance):** predation is *encounter-limited*
> ([[mechanisms/predation-is-encounter-limited]]) — scarcity suppresses it;
> lethality/reliability hit a structurally-unreachable prey-rate ceiling
> ([[findings/prey-consumption-target-is-structurally-unreachable-in-a-stable-ecology]]);
> free attack-reach *de-skills* hunting (intelligence collapses). **Validated lead
> (iter6):** within-life reward-learning (gentle three-factor) ON the predator
> ecology RECOVERS action_effectiveness — the intelligent-hunting loop, narrowly
> short of a clean gate ([[experiments/0006-plasticity-three-factor-on-predation]]).
> Next: refine the band + stack sensory salience; and/or the metric contract (human).

**Champion = homeostatic metabolism** (iteration 1). Energy-dependent passive
metabolic cost (0.5× floor below energy 5) breaks the starvation death-spiral.

- **Exact commit:** `eb30fffe4cc477c4c6b56993dd358b7e47a61942`
  (`exp metabolism homeostatic-metabolism: energy-dependent passive cost`),
  fast-forwarded onto `autoresearch/best` over the iteration-0 baseline `70b7700`.
- **Change:** `sim-core/src/metabolism.rs` only (11 lines). Determinism ✓
  (byte-identical threads 1 vs 4), build ✓.
- **Reproduce:** `git checkout autoresearch/best` → `cargo build -p sim-cli --release`
  → per-seed `new --seed S` + `run-to 500000` + `pillars` for S ∈ {7,42,123,2026}.

## Metrics (cross-seed mean over all 4 seeds — the new baseline for iteration 2)

| metric | champion (n=4) | iter-0 baseline (n=3) | clean Δ on 7/42/123 | target |
|---|---|---|---|---|
| plant_consumption_rate | 0.0690 | 0.0599 | +0.0057 | ≥ 0.10 (gap remains) |
| prey_consumption_rate  | 0.0018 | 0.0217 | −0.0202 | ≥ 0.025 (inherent trade — see Mechanism) |
| action_effectiveness   | 0.5647 | 0.5566 | +0.0155 | HOLD ✓ |
| mi_sa                  | 0.1407 | 0.0955 | +0.047  | HOLD ✓ |
| learning_slope         | −0.000423 | −0.000689 | +0.000276 | ≥ +0.0005 (closing) |

**All 4 seeds now survive (n 3→4):** seed 2026 rescued from extinction (pop 1170).
Per-seed (authoritative): 7 = plant 0.0658 / AE 0.5575 / mi_sa 0.0478 / slope
−0.000659 / pop 1185; 42 = 0.0655 / 0.5276 / 0.0896 / −0.000527 / 1936;
123 = 0.0653 / 0.6311 / 0.2903 / −0.000053 / 1555; 2026 = 0.0794 / 0.5428 /
0.1350 / −0.000451 / 1170.

The prey collapse is the expected healthier-population effect
([[mechanisms/predation-inversely-coupled-to-population-health]]), not a defect;
predation is reserved for iteration 2
([[directions/predation-needs-energetic-attractiveness]]).

# Lineage

1. [[experiments/0001-metabolism-homeostatic-metabolism]] — energy-dependent
   passive metabolism (eb30fff). The unique iteration-1 experiment that lifted the
   keystone while holding both HOLD pillars and rescuing the marginal seed.

# Citations

- Iteration-0 baseline (n=3) + per-seed diagnostic: recorded in `log.md` iter 0.
- Champion validation: planner authoritative cross-seed eval, byte-identical to
  the agent run (determinism).
