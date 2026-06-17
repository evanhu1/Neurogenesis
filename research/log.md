# Log

Append-only iteration history (newest first). One entry per planner iteration:
goals assigned to coordinators, what each handed back, what was merged into
`autoresearch/best`, and what changed in `STATE.md`. The lossless audit trail
that lets `STATE.md` be compacted aggressively.

<!-- new entries go directly below this line -->

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
