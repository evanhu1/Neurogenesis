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

## Attempt 2

- Hypothesis:
  `inter_accumulation` is now the largest brain stage. The hot loops still decode and bounds-check every target synapse even though runtime topology already guarantees valid inter and action destinations.
- Planned change:
  Replace per-edge `Option`-based indexing in evaluation and plasticity hot loops with direct indexing plus debug assertions.
- Status:
  Reverted.
- 10000-turn checkpoint comparison:
  Checkpoint `caa6223`: `avg_wall_us_per_turn 503.937`
  Attempt 2 candidate: `avg_wall_us_per_turn 502.891`
- Notes:
  The stage-local `inter_accumulation` number improved, but end-to-end wall time moved by only about `0.2%` while adding unsafe hot-path indexing. Not worth keeping.

## Attempt 3

- Hypothesis:
  Brain scratch still does avoidable per-eval work: it copies `prev_inter_states` even though each neuron can read its own prior `state` before updating, and it computes centered action post-signals that are never consumed.
- Planned change:
  Remove the dead centered-action scratch path, drop `prev_inter_states`, and tighten `prepare_inter_buffers` to only stage the data that later phases actually read.
- Status:
  Reverted.
- 10000-turn checkpoint comparison:
  Checkpoint `caa6223`: `avg_wall_us_per_turn 503.937`
  Attempt 3 candidate: `avg_wall_us_per_turn 507.053`
- Notes:
  This lowered `inter_setup` and slightly reduced `evaluate_brain_total`, but the end-to-end run still regressed. Not worth keeping.

## Attempt 4

- Hypothesis:
  Evaluation currently calls `partition_point` on every inter neuron's outgoing synapses twice per brain eval. Caching the inter/action split boundary once per topology refresh should reduce steady-state eval overhead without changing behavior.
- Planned change:
  Add a runtime-only cached split index to `InterNeuronState`, refresh it with parent IDs, and use it in both `inter_accumulation` and `action_accumulation`.
- Status:
  Kept.
- 10000-turn checkpoint comparison:
  Checkpoint `caa6223`: `avg_wall_us_per_turn 503.937`
  Attempt 4: `avg_wall_us_per_turn 500.090`
- Notes:
  End-to-end wall time improved by about `0.8%`. The cached split boundary materially reduced routing overhead inside the brain eval hot path.

## Attempt 5

- Hypothesis:
  The sensing path still pays generic receptor-dispatch overhead even though the sensory layout is fixed and only a single vision ray exists.
- Planned change:
  Specialize `encode_sensory_inputs` to write the fixed sensory slots directly and simplify single-ray handling.
- Status:
  Reverted.
- 10000-turn checkpoint comparison:
  Checkpoint `9f039af`: `avg_wall_us_per_turn 500.090`
  Attempt 5 candidate: `avg_wall_us_per_turn 499.810`
- Notes:
  `scan_ahead` improved noticeably, but end-to-end wall time moved by only about `0.06%`. Not enough to justify extra specialized routing logic.
