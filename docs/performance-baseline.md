# Performance Baseline

Date: 2026-02-07

Command:
```bash
cargo run -p sim-cli -- benchmark --epochs 10 --seed 42
```

Result:
- `elapsed_ms`: 1492
- `avg_ms_per_epoch`: 149.209
- `avg_ms_per_tick`: 7.46045
- `normalized_us_per_unit`: 2.486816666666667

Final metrics snapshot:
- `ticks`: 200
- `epochs`: 10
- `survivors_last_epoch`: 148
- `organisms`: 500
- `synapse_ops_last_tick`: 1572
- `actions_applied_last_tick`: 176

Notes:
- These numbers are pre-optimization reference values for local development.
- Use this file as the comparison point for future hot-path optimization work.
