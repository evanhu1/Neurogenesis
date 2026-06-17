# Log

Append-only iteration history (newest first). One entry per planner iteration:
goals assigned to coordinators, what each handed back, what was merged into
`autoresearch/best`, and what changed in `STATE.md`. The lossless audit trail
that lets `STATE.md` be compacted aggressively.

<!-- new entries go directly below this line -->

## Iteration 1 — metabolism / plasticity / food-ecology (2026-06-16)
* **Coordinators (sequential, Agent-tool):** metabolism-lifecycle, plasticity-genome,
  food-ecology. base_ref `70b7700`. 12 code-change experiments total, each forked +
  built + determinism-checked (threads 1v4 byte-identical) + persisted as
  `autoresearch/exp-0001-*`.
* **Result:** 1 promoted, 11 rejected. Champion advanced `70b7700`→`eb30fff`
  (**homeostatic metabolism**, fast-forward). Authoritatively gate-validated
  (byte-identical to the agent run). Clean seed-for-seed (7/42/123): slope
  +0.000276, action_eff +0.0155, mi_sa +0.047, plant +0.0057, prey −0.0202;
  **seed 2026 rescued (n 3→4)**.
* **Why only homeostatic:** it is the unique experiment that lifts the keystone
  while HOLDING both HOLD pillars; all others regress action_effectiveness
  ([[findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime]]).
* **Big durable law discovered:** predation (corpse-eating) is inversely coupled to
  population health — prey collapsed ~10× under every improving change; the prey
  target needs energetically-attractive predation (engine code), not death-pressure
  reduction ([[mechanisms/predation-inversely-coupled-to-population-health]]).
* **Notable rejects:** longer-eligibility hit slope +0.000592 but AE −0.043;
  three-factor-energy (novel neuromodulation) lifted slope but AE −0.057
  (seed-42 instability) — both kept as Directions. lower-food-energy reached plant
  0.087–0.094 but cratered AE (food_energy conflict confirmed).
* **OKF:** 12 Experiments, 2 Findings, 2 new Mechanisms, 5 Directions, 2 DeadEnds
  written; `best-program.md` + `STATE.md` updated. Process learnings recorded in
  STATE (sequential coordinators, n=3→n=4 confound, subagent-resume limits).
* **Next (iter2):** predation mechanics (top priority) + metabolism combos
  (ramp tuning, stack brain-cost-discount) to push learning_slope positive.

## Iteration 0 — bootstrap (2026-06-16)
* **`autoresearch/best`** reconciled: already existed at `ef6f9bb` (== `main`);
  no divergence, no experiments. Planner operates from worktree
  `/Users/evanhu/code/ng-best`. Orchestration = Agent tool (workflow engine
  dropped upstream).
* **Baseline canonical sweep** (seeds 7,42,123,2026, 500k, 321s wall): cross-seed
  mean (n=3 survivors) plant 0.0599 / prey 0.0217 / act_eff 0.5566 / mi_sa
  0.0955 / learning_slope −0.000689. Recorded to `best-program.md`.
* **Per-seed diagnostic**: seeds 7/42/123 survive at pop 1316/2052/1605 (≪ world
  cap 62 500); **seed 2026 fully extinct (pop 0)** → n=3. All survivors have
  negative learning_slope. Baseline is in a scarcity/collapse regime near a
  tipping boundary, not the explosion regime. Predation = corpse-eating;
  starvation deaths leave no corpse. These findings rewrote `STATE.md`'s central
  obstacle + directions.
* **Next:** iteration 1 — coordinators metabolism-lifecycle, plasticity-genome,
  food-ecology (sequential, code-change experiments, base_ref `ef6f9bb`).

## (no iterations yet)
* **Creation**: bundle scaffolded; `STATE.md` seeded with goal/targets and four
  carried-over config-level mechanisms.
