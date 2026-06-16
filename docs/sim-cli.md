# sim-cli — usage reference

`sim-cli` is the **agent-facing research cockpit** for the NeuroGenesis engine:
a **stateless, one-shot CLI** where a simulation world is an explicit file
artifact. Each invocation reads a world from `--in`, runs one command, and (for
mutating commands) writes the advanced world to `--out`. Output is **JSON by
default**, optimized for an agent driving it over the shell.

Design rationale lives in `SPEC.md` and `docs/sim-cli-stateless-spec.md`. The
in-CLI `help` (`sim-cli help`) is the always-current command list.

> **Why a file, not a REPL?** State on disk means: snapshot/fork a world with
> `cp`, fan out parallel runs by backgrounding invocations, and never lose work
> to a crashed long-lived process. The process exiting is the completion signal;
> stdout is the framed result.

## Run

```bash
cargo build -p sim-cli --release        # --release strongly recommended
./target/release/sim-cli <command> [global flags] [command args]
```

## Core model

- **A world is a file** (`world.bin`, CBOR). Built by `new`, advanced by
  `step`/`run-to`/`watch`. Forward-only; to branch, `cp` it.
- **Metrics are a sidecar file** (`<world>.metrics`) holding the recorder
  accumulators. It **follows the world automatically**: `new` mints it, and
  mutating/reading commands pick up the `<world>.metrics` sibling unless you pass
  an explicit `--metrics PATH` or `--no-metrics`. Live pillars are
  **byte-identical** to the offline `sim-evaluation` harness (shared
  `sim-metrics` crate).
- **Determinism:** save→load→advance is byte-identical to advancing in RAM (RNG
  + plasticity state persist exactly). Two `cp` forks advanced identically are
  bit-for-bit equal.

## Global flags

- `--in <world.bin>` — world to read (required by every command except `new`).
- `--out <world.bin>` — where a mutating command writes the advanced world.
  **Defaults to `--in`** (advance in place); pass a different path to fork.
- `--metrics <path>` — explicit sidecar (default: the `<world>.metrics` sibling).
- `--no-metrics` — disable recording for this call (fast scrub path).
- `--out-dir <dir>` — directory for run-mode result files (default
  `artifacts/runs`). Used by `sweep`.
- `--json` / `--text` — output format (JSON is the default).

## Commands

### Mutating (persist the world)
- `new [--config P] [--seed N] [--set k=v]… [--scale W,POP] [--threads K] [--report-every R] --out w.bin [--no-metrics]`
  — construct a world. `--set` patches config fields inline (e.g.
  `--set food_energy=12 --set passive_metabolism_cost_per_unit=0.0035`), no temp
  config files. `--scale W,POP` overrides world_width,num_organisms (marks the
  world non-canonical). Mints the metric sidecar unless `--no-metrics`.
- `step [N] --in w.bin [--out w.bin]` — advance N ticks (default 1).
- `run-to T --in w.bin [--out w.bin]` — advance until turn == T (no backward).
- `watch T [--every E] --in w.bin [--out w.bin]` — advance to T, emitting a
  metrics row every E ticks (JSONL progress log).
- `bench [N] --in w.bin` — time N ticks; the advanced world is discarded
  (no `--out`).

### Reads (stdout only, no `--out`)
- `turn` — current turn.
- `state` — population, energy/health/age/generation summaries, food, last-turn
  ecology (+ a pillars line if a sidecar is present).
- `pillars` — the **raw windowed-mean metrics** (no [0,1] interpretation):
  `plant_consumption_rate` (foraging), `prey_consumption_rate` (predation),
  `action_effectiveness` + `mi_sa` (intelligence), `learning_slope` (learning) —
  **plus a `granular` section** with the full per-interval series behind the
  window (the scoring window is marked). **Needs a sidecar.**
- `eco` — population/food trajectory, deaths-by-cause, rates (trajectory needs a
  sidecar; point-in-time works without).
- `lineage` — generation distribution + founder-lineage composition.
- `genome [--gene G] [--drift]` — per-gene population distributions.
- `timeseries [--cols LIST] [--last K]` — recorded columns as sparklines.
  **Needs a sidecar.** Columns: per-tick `population descendants food births
  deaths consumptions predations reproductions`; per-interval
  `action_effectiveness plant_consumption_rate prey_consumption_rate mi_sa
  learning_slope pop`.
- `food` — plant/corpse counts, energy, coverage.
- `inspect ID` · `top FIELD [N]` · `hist FIELD` · `find EXPR [--fields LIST]
  [--limit N]` · `brain ID [--view summary|synapses|activations|dot]` ·
  `decide ID` — per-organism inspection. Fields/vocabularies: invalid args print
  the valid set.
- `query --in w.bin` — **batch reads**: read read-only commands from stdin (one
  per line) against a single world load. Use for a burst of probes so you pay one
  deserialize, not one per command. Mutating commands are rejected.

### Run modes (own their worlds, write a result file)
- `sweep --grid KEY=v1,v2[,…] [KEY2=…] --seeds N[,N…] --to TICK [--config P]
  [--baseline KEY=v,…] [--report-every R] [--threads K] [--jobs J] [--out-dir D]`
  — run the cartesian product of config overrides × seeds, score each cell's
  pillars (mean/min/max across the seed cohort), and write a JSON result file to
  `--out-dir` (default `artifacts/runs/sweep-<ts>.json`). Prints the file path
  and a compact ranking table (with Δ vs the `--baseline` cell) to stdout.
  `KEY`s are config field names (same vocabulary as `new --set`). Jobs run in
  parallel (bounded to `--jobs`, default = CPU count; each run uses `--threads`
  intent threads, default 1, so parallelism comes from running many worlds at
  once). Example:
  `sweep --grid food_energy=10,12,14 --seeds 7,42,123,2026 --to 500000
  --baseline food_energy=12`.

## Workflows

**Warm once, fork, run only the scored window** (pillars read tick 460k–500k of
500k, so don't re-warm per config):

```bash
sim-cli new --seed 7 --out artifacts/cli/warm.bin
sim-cli run-to 400000 --in artifacts/cli/warm.bin
for e in 10 12 14; do
  cp artifacts/cli/warm.bin     artifacts/cli/e$e.bin
  cp artifacts/cli/warm.metrics artifacts/cli/e$e.metrics    # fork metrics too
  sim-cli run-to 500000 --in artifacts/cli/e$e.bin &         # parallel
done; wait
for e in 10 12 14; do sim-cli pillars --in artifacts/cli/e$e.bin; done
```

**Burst of probes on one state:**
```bash
sim-cli query --in artifacts/cli/warm.bin <<'EOF'
find energy < 5 and age > 400 --limit 10
inspect 8123
decide 8123
brain 8123 --view summary
EOF
```

## Notes

- Put worlds/sidecars under `artifacts/` (not `/tmp`) so they survive a session.
- Determinism: same config + seed + tick count = identical world bytes;
  recording is read-only and draws no RNG, so a `sim-cli` run reproduces the eval
  trajectory exactly.
- A `--config` with `--set` overrides reparses the world TOML; `--set` keys are
  the config field names (section-unique, e.g. `food_energy` under `[food]`).
