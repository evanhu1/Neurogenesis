# Repository Guidelines

## Project Structure

Workspace crates:

- `sim-types/`: shared domain types used across all Rust crates.
- `sim-config/`: world + seed-genome config crate; owns the hidden
  food-ecology policy and baseline TOML defaults.
- `sim-core/`: deterministic simulation engine (turn pipeline, brain, genome,
  spawn/world generation).
- `sim-evaluation/`: headless evaluation harness built on a dataset-centric ETL
  pipeline (sim emits raw facts to partitioned Parquet; all metrics/reports are
  derived post-hoc via `analyze`).
- `sim-server/`: Axum HTTP + WebSocket server.
- `web-client/`: React + TailwindCSS + Vite canvas UI.

Config files: `sim-config/config.toml` and `sim-config/seed_genome.toml` are the
baselines; `sim-evaluation/config.toml` and `sim-evaluation/seed_genome.toml` are
the stable evaluation configs. Keep both evaluation files in sync with the
`sim-config` baseline when config-schema or baseline tuning changes affect
evaluation assumptions.

## Build & Test

- `cargo check --workspace`: fast compile check.
- `cargo test --workspace`: run all Rust tests.
- `make fmt`: format Rust code.
- `make lint`: clippy with warnings as errors.
- `cargo run -p sim-evaluation --release --`: run the default 8-seed
  evolution-loop evaluation harness and generate
  `report.html`/`timeseries.csv`/`summary.json`.
- `cargo run -p sim-evaluation --release -- --seed <u64>[,<u64>...]`: override
  the default seed suite. (`make evaluate ARGS="..."` is the Makefile wrapper.)
- `cargo run -p sim-evaluation --release -- analyze <run>`: re-derive
  `summary.json`/`timeseries.csv`/`report.html` from a persisted dataset without
  re-running the sim. `<run>` accepts a path (run root or seed dir), a timestamp
  prefix resolved under `artifacts/evaluation/`, or `latest`.
- `cargo build -p sim-cli --release` then `./target/release/sim-cli <command>`:
  the agent-facing research cockpit — a **stateless one-shot CLI** where a world
  is an explicit file. **You MUST read `docs/sim-cli.md` in full before using
  sim-cli** — it documents the world-as-file model, the metric sidecar, command
  vocabularies, and semantics you will otherwise misuse. Output is JSON by
  default (`--text` to override); invalid args print the valid options. Each call
  reads `--in <world.bin>`, runs one command, and (for mutating commands) writes
  `--out <world.bin>` (defaults to `--in` = advance in place). Typical flow:
  `new [--seed N] [--set k=v]… --out w.bin` → `run-to T --in w.bin` → read
  `pillars` (raw windowed metrics: plant/prey consumption rates,
  action_effectiveness, mi_sa, learning_slope — via the shared `sim-metrics`
  crate, no [0,1] interpretation), `state`, `eco`, `lineage`,
  `genome`, `timeseries`, and per-organism `find`/`top`/`hist`/`inspect`/`brain`/
  `decide`; `bench [N]` for throughput; `query` for batch reads off one load;
  `sweep --grid k=v,v --seeds N,N --to T` for parallel grid×seed experiments
  (writes a result file under `--out-dir`, default `artifacts/runs/`). `pillars`
  includes a `granular` per-interval series behind the windowed scores.
  The metric sidecar (`<world>.metrics`) is minted by `new` and follows the
  world; `pillars`/`eco`-trajectory/`timeseries` need it. **Snapshot/fork a world
  with `cp`; fan out parallel runs by backgrounding invocations.** Put worlds
  under `artifacts/`, not `/tmp`. See `docs/sim-cli.md` (usage),
  `docs/sim-cli-stateless-spec.md` + `SPEC.md` (design).
- `cargo run -p sim-server`: start backend on `127.0.0.1:8080`. Flags:
  - `--champion-pool-path <path>`: override the default `champion_pool.json`
    location used for read-back and persistence of saved champions.
  - `--seed-genome-snapshot <path.bin>`: load a single `OrganismGenome` from a
    bincode-encoded evaluation snapshot
    (`artifacts/evaluation/.../seed_<seed>/genomes/tNNNNNN.bin`) and use it as a
    one-entry in-memory champion pool; every initial organism spawns with this
    genome, and champion-save endpoints no-op (no disk writes) for the session.
- `cd web-client && npm run dev`: run frontend (Vite).
- `cd web-client && npm run build && npm run typecheck`: production build + TS.

## Coding Style

- Rust performance is a top priority: avoid unnecessary allocations/copies,
  prefer zero-cost abstractions.
- Keep Rust and web data models synchronized using this split:
  - `web-client/src/types.ts` `Api*` types must match Rust wire schema
    (`sim-types/src/lib.rs` + `sim-server/src/protocol.rs`) exactly.
  - UI-only types in `web-client/src/types.ts` may extend/derive fields, but
    must be produced by normalization in `web-client/src/protocol.ts`.
  - Do not read raw API payloads directly in feature code; normalize at the
    HTTP/WS boundary first.
  - When changing wire fields, update Rust structs, TS `Api*` types, and
    normalizers in the same change.
- DO NOT CARE ABOUT backwards compatibility.

## Testing

- Do not write new tests. The human author maintains the test suite.
- Run `cargo test --workspace` to verify existing tests still pass after changes.
- For evolution-loop benchmarking and regression checks, run `sim-evaluation`
  (prefer release mode) and compare outputs across seeds/configs.

## Frontend Browser Testing

- For frontend changes, verify behavior with `agent-browser` against the local app.
- Start the backend with `cargo run -p sim-server` and the frontend with
  `cd web-client && npm run dev`.
- If needed, install once with `npm install -g agent-browser && agent-browser install`.
- Typical flow: `agent-browser open http://127.0.0.1:5173`, then use
  `snapshot`, `click`, `fill`, `get text`, and `screenshot` to exercise the UI.
- Stop any background `sim-server` or Vite processes when finished.

## Architecture Invariants

- **Determinism**: fixed config + seed = identical results. All tie-breaking
  uses organism ID ordering; action and predation sampling are deterministic
  hashes of `(seed, turn, organism IDs)`. Any change to turn logic must preserve
  determinism.
- **Canonical tick execution order** lives in
  `sim-core/src/turn/mod.rs::Simulation::tick` — treat that as the source of
  truth for phase ordering.
- The food-ecology policy is hidden/owned by `sim-config`, not configurable in
  the world TOML beyond the documented threshold overrides.
- No species registry / speciation in sim-core.
