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

(filled in below as we go)
