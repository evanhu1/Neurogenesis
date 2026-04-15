# Sim-Evaluation Performance Log

Goal: cut a 100k-tick evaluation time by 5x. Baseline measured against
`--seed 42 --ticks 10000` so iterations don't blow the budget.

## Benchmark protocol

```
time ./target/release/sim-evaluation --seed 42 --ticks 10000 \
    --report-every 5000 --out /tmp/sim-eval-bench/<label>
```

Single seed, single thread, default config (`sim-evaluation/config.toml`,
world 250x250, 5000 organisms, runtime plasticity on, intent_parallel_threads=8).

## Baseline (commit 8cba448)

| Run | total_time_seconds |
| --- | --- |
| 1 | 2.191 |
| 2 | 2.222 |

~4500 ticks/sec. To get 5x we need ~22500 ticks/sec (i.e. ~0.45 s for 10k).

## Profiling notes

Suspect hot paths added by `instrumentation` feature (vs the server build):

- `update_instrumentation_utilization` runs every tick per organism: two passes
  over `brain.inter` (EMA + count > threshold). For 5k organisms each with
  growing inter populations this is significant.
- `instrument_action_record` constructs an `ActionRecord` per organism per tick;
  `intents.rs` collects `Vec<Option<ActionRecord>>` via `unzip` (extra
  allocation + storage) and the harness sweeps `sim.action_records()` again.
- `food_visible: [bool; VISION_RAY_COUNT]` is captured from rays.

Also evaluation-side per-tick work:

- `interval_population_exposure += sim.organisms().len()` (cheap).
- `sim.action_records().iter().flatten()` then `ledger.update(record)` for every
  living organism every tick — does HashMap lookup, indexes joint table, etc.

## Attempts

### 0. Profiling and measurement methodology

`samply` profiles couldn't resolve symbols even with debug=line-tables-only or
debug=2 (need a dSYM bundle). Switched to using the in-tree `profile_turn_path`
example (cargo run -p sim-core --features 'profiling instrumentation' --example
profile_turn_path -- --config sim-evaluation/config.toml). That gives accurate
per-phase totals.

Measured per-tick wall (337 µs/tick, 8-thread parallelism, 5000 organisms,
post-warmup) breaks down as:

- intents: 53% — almost entirely `evaluate_brain` + pending coactivations.
- spawn: 34% — mostly `apply_post_commit_runtime_weight_updates` (plasticity
  weight updates) plus the periodic-injection cost on every 100th tick.
- commit: 11% — facing/move/attack resolution.
- everything else (lifecycle, move_resolution, reproduction, age,
  metrics_and_delta, consistency_check): <2%.

Within evaluate_brain, the breakdown is:
- scan_ahead 30%, plasticity_sensory_tuning 26%, plasticity_setup 10%,
  inter_accumulation 8%, plasticity_inter_tuning 8%, action_activation 6%,
  remaining inter/action stages 3-4% each, plasticity_prune 0.85%.

Verified the harness's per-tick work (`ledger.update`, etc.) contributes only
~6% to wall time by null-out experiment — the overwhelming majority of the
time is inside `sim.tick()`.

Verified single-thread vs 8-thread on 1 seed:
- 1-thread: 2.535 s for 10k ticks
- 4-thread: 1.737 s
- 8-thread: 2.220 s (oversubscribed)

Multi-seed (8 seeds default) saturates 8 cores already (CPU usage 770-790%).

### 1. Enable `lto = "fat"` and `codegen-units = 1` (committed as f6c2530)

Adds workspace-level [profile.release] section. Re-built `sim-evaluation`.

| Bench | Before | After | Δ |
| --- | --- | --- | --- |
| single seed, 10k ticks | 2.222 s | 2.136 s | -3.9% |
| 8 seeds, 5k ticks | 2.912 s | 2.820 s | -3.2% |

Modest but free, kept.

### Conclusion so far

Most of the wall time is genuinely inside `sim.tick()` and not in the
evaluation harness's instrumentation. The harness cost is small compared to
the simulation kernel. Achieving 5x in this codebase would require either:

- Algorithmic rework of brain evaluation / plasticity (e.g. SIMD batching
  multiple organisms at once, or a fundamentally cheaper representation).
- Skipping a pillar of work, e.g. shipping a "reduced instrumentation" path
  that drops `food_visible`, `utilization`, etc. — but the harness consumes
  all of those today.
- Better cross-seed work scheduling so the 8 cores don't contend on rayon.

These are large refactors well beyond surgical changes; LTO + codegen-units
captures the easy free win without changing semantics or risking determinism.

