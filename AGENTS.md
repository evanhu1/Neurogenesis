# Repository Guidelines

## Project Structure & Module Organization

- `sim-core/`: deterministic simulation engine and core logic. Split into
  modules:
  - `lib.rs`: simulation struct, config validation, shared glue.
  - `turn.rs`: turn pipeline and move/consume/starvation resolution.
  - `brain.rs`: network evaluation, mutation, and topology/synapse operators.
  - `spawn.rs`: initial spawn + reproduction/replacement spawn placement.
- `grid.rs`: hex-grid geometry and occupancy helpers.
- `sim-protocol/`: shared API/protocol types used by server and UI.
- `sim-server/`: Axum HTTP + WebSocket server (`src/main.rs`).
- `web-client/`: React + TailwindCSS + Vite canvas UI (`src/`), static entry in
  `index.html`.
- `config/default.toml`: baseline simulation configuration.
- `docs/`: API/behavior docs and protocol examples.

## Build, Test, and Development Commands

- `cargo check --workspace`: fast compile check for all Rust crates.
- `cargo test --workspace`: run Rust unit/integration tests.
- `make fmt`: format all Rust code with `rustfmt`.
- `make lint`: run `clippy` with warnings treated as errors.
- `cargo run -p sim-server`: start backend on `127.0.0.1:8080` by default.
- `cd web-client && npm run dev`: run frontend locally (Vite).
- `cd web-client && npm run build && npm run typecheck`: production build + TS
  checks.

## Coding Style & Naming Conventions

- Rust: use standard `rustfmt` output (4-space indentation), `snake_case` for
  functions/modules, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for
  constants.
- TypeScript/React: strict mode is enabled (`web-client/tsconfig.json`), prefer
  `camelCase` for vars/functions and `PascalCase` for components/types.
- Use Tailwind for front end styling
- Keep protocol field names aligned with backend payloads (both `snake_case` and
  compatibility fields appear in `web-client/src/types.ts`).

## Testing Guidelines

- Primary test runner: `cargo test --workspace`.
- Place integration tests under each crate’s `tests/` directory (example:
  `sim-core/tests/golden.rs` with fixtures in `sim-core/tests/fixtures/`).
- Add deterministic tests for simulation behavior changes (fixed seeds, snapshot
  assertions).
- No enforced coverage threshold currently; include tests for new logic and
  regressions.
