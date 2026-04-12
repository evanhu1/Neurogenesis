# Turn Optimization Log

## Baseline

- Date: 2026-04-12
- Commit: `8210bd6`
- Command:
  `cargo run -p sim-core --release --features profiling --example profile_turn_path -- --turns 1000 --warmup-turns 200 --seed 42`
- Result:
  `avg_wall_us_per_turn: 244.992`
- Top turn phases:
  `intents 151.428 ms (62.20%)`
  `spawn 46.621 ms (19.15%)`
  `commit 34.199 ms (14.05%)`
- Top brain stages:
  `scan_ahead 48.152 ms (30.13%)`
  `inter_accumulation 39.567 ms (24.76%)`
  `inter_activation 28.003 ms (17.52%)`
  `action_accumulation 20.408 ms (12.77%)`

## Attempt 1

- Hypothesis:
  The current RGB ray scan regressed because it walks the full ray and composites multiple cells with float math. Restoring first-hit occlusion should reduce `scan_ahead` materially.
- Planned change:
  Keep RGB outputs, but stop the ray when the first non-self organism, food, or wall is encountered. Spikes remain visible only on the hit cell.
- Status:
  Kept.
- Validation:
  `cargo test -p sim-core --release`
- 10000-turn comparison:
  Baseline: `avg_wall_us_per_turn 621.124`, `scan_ahead 0.177 us/call`
  Attempt 1: `avg_wall_us_per_turn 505.993`, `scan_ahead 0.088 us/call`
- Notes:
  This restores the pre-RGB stopping rule while keeping RGB channels, and it materially reduces both total wall time and per-call sensing cost.
