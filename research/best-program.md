---
type: BestProgram
title: Current best program
description: The current research champion — a concrete git ref the next iteration forks from.
git_ref: autoresearch/best
iteration: 0
metrics:
  plant_consumption_rate: null
  prey_consumption_rate: null
  action_effectiveness: null
  mi_sa: null
  learning_slope: null
lineage: []
tags: [autoresearch, champion]
timestamp: 2026-06-16T00:00:00Z
---

# Current best program

**Not yet established.** The first iteration creates branch `autoresearch/best`
from `main` and runs the baseline canonical eval to fill in the metrics above.

- **Exact commit:** _tbd_ (record the sha here each time `autoresearch/best` advances).
- **Reproduce:** `git checkout autoresearch/best` → build → `sim-cli` canonical
  cross-seed sweep (seeds 7,42,123,2026, to 500000).

# Lineage

The ordered list of accepted experiments that built this program. Each advance
of `autoresearch/best` appends the merged `experiments/...` concept here, so the
champion's full construction history is traceable.

_(empty — no experiments accepted yet)_

# Citations

_(sweep result files for the champion's metrics, once evaluated)_
