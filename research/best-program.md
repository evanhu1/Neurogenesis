---
type: BestProgram
title: Current best program
description: The current research champion — a concrete git ref the next iteration forks from.
git_ref: autoresearch/best
iteration: 0
metrics:
  plant_consumption_rate: 0.0599
  prey_consumption_rate: 0.0217
  action_effectiveness: 0.5566
  mi_sa: 0.0955
  learning_slope: -0.000689
lineage: []
tags: [autoresearch, champion]
timestamp: 2026-06-16T00:00:00Z
---

# Current best program

**Baseline champion = `main` (no experiments accepted yet).** `autoresearch/best`
sits exactly at `main`; iteration 0 only recorded the baseline canonical metrics.

- **Exact commit:** `ef6f9bb50657cffcdf49e2926d856704ddaaa701`
  (`autoresearch: drop the workflow engine; orchestrate with the Agent tool`).
- **Reproduce:** `git checkout autoresearch/best` →
  `cargo build -p sim-cli --release` →
  `sim-cli sweep --grid food_energy=20 --seeds 7,42,123,2026 --to 500000 --baseline food_energy=20`.

## Metrics (cross-seed mean, raw pillars)

| metric | value | spread (min/max, n) | target | status |
|---|---|---|---|---|
| plant_consumption_rate | 0.0599 | 0.0562 / 0.0624 (n=3) | ≥ 0.10  | ✗ gap −0.040 |
| prey_consumption_rate  | 0.0217 | 0.0201 / 0.0244 (n=3) | ≥ 0.025 | ✗ gap −0.003 (close) |
| action_effectiveness   | 0.5566 | 0.5121 / 0.5920 (n=3) | hold    | ✓ |
| mi_sa                  | 0.0955 | 0.0641 / 0.1409 (n=3) | hold    | ✓ |
| learning_slope         | −0.000689 | −0.000903 / −0.000549 (n=3) | ≥ +0.0005 | ✗ negative — the wall |

**⚠ n=3, not 4 — seed 2026 collapses to full extinction** (population 0 at
turn 500000; pillars NA). The cross-seed mean above is over the 3 survivors
(7/42/123, populations 1316/2052/1605). All three survivors have **negative
learning_slope** (−0.000616 / −0.000903 / −0.000549). Populations are far below
the world cap (62500 cells), so the baseline sits in a **scarcity/collapse
regime near a tipping boundary**, not a population-explosion regime.
**Population robustness (sustaining all 4 seeds, esp. 2026) is a first-class
objective**, not just hitting the rate thresholds — see STATE.

# Lineage

The ordered list of accepted experiments that built this program. Each advance
of `autoresearch/best` appends the merged `experiments/...` concept here, so the
champion's full construction history is traceable.

_(empty — no experiments accepted yet; champion = baseline `main`)_

# Citations

- Baseline sweep: `sweep --grid food_energy=20 --seeds 7,42,123,2026 --to 500000`
  at `ef6f9bb`, 321s wall (4 parallel, threads/run=3). Raw JSON was under
  `artifacts/runs/` (gitignored, transient); the metrics table above is the
  durable record.
