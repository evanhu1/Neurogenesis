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

### 2. Hoist hex step deltas out of `scan_ray` (committed as 619f171)

`scan_ray` was calling `hex_neighbor` every step, which match'd on the
`FacingDirection` each iteration. Replaced with a single `facing_delta()`
lookup that's hoisted before the loop, plus inlined the wraparound check.

Per-call cost dropped 0.130 → 0.120 µs (≈8% on `scan_ahead`). Wall-time
impact tiny because `scan_ahead` is only ~28% of brain time.

| Bench | Before | After | Δ |
| --- | --- | --- | --- |
| single seed, 10k ticks | 2.136 s | 2.130 s | -0.3% |
| 8 seeds, 5k ticks | 2.820 s | 2.790 s | -1.1% |

Determinism preserved: `cargo test -p sim-evaluation` and `-p sim-core` pass.

### Conclusion

Combined wins of changes 1 + 2: roughly **4–5% wall-time** on both single-seed
and multi-seed benchmarks. Far short of the requested 5x.

Most of the wall time is genuinely inside `sim.tick()` and not in the
evaluation harness's instrumentation. Reproduced with a `RAW_TICK_ONLY`
experiment that stripped the harness's per-tick work to a single `sim.tick()`
call: throughput barely moved (~6% improvement). The user's premise that
"heavy instrumentation is massively slowing down the simulation" doesn't
match the data — the instrumentation hot path adds only ~6% over the
non-instrumented build (337 vs 319 µs/tick in `profile_turn_path`).

The actual bottleneck breakdown (post-warmup, with instrumentation):

- Intents 53% (mostly `evaluate_brain` + pending coactivations)
- Spawn 34% (mostly `apply_post_commit_runtime_weight_updates`)
- Commit 11%
- Other phases together <2%

Plasticity (pending coactivations + weight updates + prune) is
**~50% of total tick time**. Brain evaluation forward pass is another ~25%.
At 5000 organisms × ~10–30 synapses × per-edge plasticity work, the kernel
is already at SIMD-able simple-arithmetic levels — no obvious structural
slack.

Achieving 5x in this codebase would require either:

- Algorithmic rework of brain evaluation / plasticity (e.g. SIMD batching
  multiple organisms at once, or a packed contiguous synapse representation
  per layer to enable wide SIMD across organisms).
- Dropping or sampling per-organism instrumentation. E.g. compute
  `utilization` only on the report tick rather than every tick, accept the
  EMA-decay drift, and special-case the food_visible array. Estimated 5–10%
  additional gain.
- A fundamentally cheaper rule (e.g. plasticity weight updates only every
  N ticks instead of every tick) — would change experimental semantics and
  require sign-off.

These are large refactors well beyond surgical changes; LTO + codegen-units
+ scan_ray hoist captures the easy free wins without changing semantics or
risking determinism. Determinism verified after each commit.

If 5x is mandatory, the highest-leverage next step is a contiguous-buffer
brain layout: store all organisms' inter activations as one big f32 vec
indexed by `(organism_idx, neuron_idx)` so the inter accumulation, plasticity
sensory tuning, and weight updates all become flat SIMD loops over a single
buffer instead of pointer-chasing per organism. That would touch
`brain/expression.rs`, `brain/evaluation.rs`, `brain/plasticity.rs`, the
genome → brain expression code, and any place reading `brain.inter`
(`turn/intents.rs`, `actor_critic.rs`). Estimated 2–4x kernel speedup if
done well, but a multi-day refactor with high blast radius.

