# Repository Guidelines

## Project Structure

NeuroGenesis is a neuroevolution artificial-life simulation. The genome is an
**indirect CPPN encoding** (HyperNEAT lineage) that *develops* into the phenotype
brain, and the whole engine is **decoupled from any specific world** behind an
`Environment` trait.

Workspace crates:

- `sim-substrate/`: the environment-agnostic evolutionary meta-learner. Owns the
  CPPN `Genome`, `develop()` (genotype→phenotype, ES-HyperNEAT), the `BrainNet`
  runtime + Hebbian plasticity, the operators (`mutate`/`crossover`/`reproduce`),
  the `Environment` trait boundary, the `PopulationDriver` tick loop, and the
  Quality-Diversity (MAP-Elites) champion archive. Depends only on serde/bincode/
  rand — **no knowledge of any world**.
- `sim-hexworld/`: the hex ecology as an `Environment` impl (grid, terrain, food,
  metabolism, predation, raycast sensing, action resolution) plus `HexSim` — the
  concrete, serializable world the tooling loads/ticks/saves.
- `sim-toyenv/`: a second environment ("Chemotaxis Ribbon", 1-D) that reuses
  `sim-substrate` unchanged — the decoupling proof.
- `sim-types/`: small, generic, environment-neutral value types (colors, visual
  properties, hex facings, food kinds, and helpers over them). **No genome/brain
  types** — those live in `sim-substrate`.
- `sim-evaluation/`: headless multi-seed harness. No fitness function → reports
  Quality-Diversity coverage / QD-score + population/lineage stats per seed.
- `sim-server/`: lean Axum HTTP + JSON backend over one in-memory `HexSim` + a QD
  champion archive.
- `web-client/`: React + Vite canvas UI over the server's REST protocol.

There is **no** `sim-core`, `sim-views`, `sim-metrics`, or `sim-config` crate —
they were removed in the substrate redesign. `CLAUDE.md` is a symlink to this
file.

## Build & Test

- `cargo check --workspace`: fast compile check.
- `cargo test --workspace`: run all Rust tests.
- `make fmt`: format Rust code.
- `make lint`: clippy with warnings as errors.
- **Environment examples** (deterministic proofs; each prints an `OK:` line):
  - `cargo run -p sim-substrate --example headless --release` — substrate ring-life.
  - `cargo run -p sim-hexworld  --example headless --release` — hex proof-of-life.
  - `cargo run -p sim-hexworld  --example simsmoke --release` — `HexSim` save/load
    round-trip + extinction termination.
  - `cargo run -p sim-toyenv    --example headless --release` — decoupling proof.
- `cargo run -p sim-evaluation --release -- [--seeds a,b,c] [--ticks N] [--width W]
  [--founders F] [--out summary.json]`: multi-seed QD evaluation.
- `cargo build -p sim-cli --release` then `./target/release/sim-cli <command>`:
  the agent-facing research CLI — a **stateless one-shot CLI** where a world is an
  explicit `world.bin` file (bincode-serialized `HexSim`). Output is JSON. Each
  call reads `--in <world.bin>`, runs one command, and (for mutating commands)
  writes `--out <world.bin>` (defaults to `--in` = advance in place). See
  `docs/sim-cli.md` for the full command reference. Typical flow:
  `new [--seed N] [--width W] [--founders F] --out w.bin` →
  `run-to <turn> --in w.bin` (stops early on extinction) → read
  `state`/`inspect`/`brain`/`genome`/`decide`/`find`/`lineage`. **There is no
  metric sidecar** and no `pillars`/`sweep`/`query` — those belonged to the old
  ETL/metrics stack. Snapshot/fork a world with `cp`; put worlds under
  `artifacts/`, not `/tmp`.
- `cargo run -p sim-server`: start backend on `127.0.0.1:8080`. Flags:
  - `--seed N` / `--width W` / `--founders F` / `--port P`.
  - `--champion-pool-path <path.json>`: QD champion archive location (default
    `champion_pool.json`, schema version 5).
  - `--seed-genome-snapshot <path.bin>`: seed every founder from one
    bincode-encoded `sim_substrate::Genome`.
  Endpoints (REST + JSON): `GET /api/state`, `GET /api/snapshot`,
  `GET /api/organism/{id}` (CPPN genome + `BrainNet`), `GET /api/champions`,
  `POST /api/champions/{id}` (save to QD archive), `POST /api/control/{play|pause|step}`.
- `cd web-client && npm run dev`: run frontend (Vite). The client polls the REST
  endpoints; set `VITE_API_BASE` to point at a non-default server.
- `cd web-client && npm run build && npm run typecheck`: production build + TS.

## Coding Style

- Rust performance is a top priority: avoid unnecessary allocations/copies,
  prefer zero-cost abstractions.
- **The substrate is world-agnostic.** `sim-substrate` must not depend on
  `sim-types`, `sim-hexworld`, or any world. Physics lives in the environment;
  the environment reads bodies through `BodyView` and requests changes through
  `EffectSink` — it never touches genome/brain/energy internals directly.
- Keep the web client's TS types in sync with the server's JSON. `web-client/src/
  api.ts` declares the wire types (`RenderSnapshot`, `PopulationStats`, `Genome`,
  `BrainNet`, `Champions`); when the server's JSON shape changes, update `api.ts`
  in the same change.
- DO NOT CARE ABOUT backwards compatibility.

## Testing

- Do not write new tests. The human author maintains the test suite.
- Run `cargo test --workspace` to verify existing tests still pass after changes.
- For behavior/regression checks, run the environment examples (they assert
  determinism) and `sim-evaluation` across seeds.

## Frontend Browser Testing

- For frontend changes, verify behavior against the local app: start the backend
  with `cargo run -p sim-server` and the frontend with `cd web-client && npm run
  dev`, then drive `http://127.0.0.1:5173` with a browser tool (canvas render,
  organism inspector, QD champions).
- Stop any background `sim-server` or Vite processes when finished.

## Architecture Invariants

- **Determinism**: fixed config + seed = identical results, tick for tick, and
  byte-identical across save/load. Action selection is a stateless hash of
  `(seed, turn, organism id)`; every cross-organism decision and RNG draw lives
  in deterministic, handle-ordered serial code in the driver, so parallelism
  cannot reorder history. Any change to tick logic must preserve determinism.
- **Canonical tick execution order** lives in
  `sim-substrate/src/driver.rs::PopulationDriver::tick` (metabolism → sensing/
  action → mating → action resolution → world step → gestation/births →
  age/plasticity). Treat it as the source of truth for phase ordering.
- **Indirect encoding**: the genome is a CPPN; `develop()`
  (`sim-substrate/src/develop.rs`) is the only genotype→phenotype path and is
  pure/RNG-free. New heritable dimensions go through the CPPN or the header, not a
  per-synapse gene.
- **Embodied, ecological reproduction**: reproduction is sexual and in-world
  (`Mate` action → crossover + mutate → offspring). Selection is purely
  ecological — no fitness function, no generations, no species registry.
- **No periodic injection**: the driver only seeds founders. A world that goes
  extinct **stays extinct** — `HexSim::tick()` records `extinct_at` and no-ops,
  and every run loop (CLI `run-to`/`step`, server, evaluation) stops on extinction.
- **Physics ownership**: the hex food-ecology and all world rules live in
  `sim-hexworld`, never in `sim-substrate`.
