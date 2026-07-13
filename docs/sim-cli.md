# sim-cli — usage reference

`sim-cli` is the **agent-facing research cockpit** for the NeuroGenesis engine:
a **stateless, one-shot CLI** where a simulation world is an explicit file
artifact. Each invocation reads a world from `--in`, runs one command, and (for
mutating commands) writes the advanced world to `--out`. Output is **JSON by
default**, optimized for an agent driving it over the shell.

Use `sim-cli help` for the command list and `sim-cli neat --help` for the
current NEAT parameters.

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
- **Determinism:** save→load→advance is byte-identical to advancing in RAM (RNG,
  plasticity state, last-tick decision records, and scaled-run provenance
  persist exactly). Two `cp` forks advanced identically are bit-for-bit equal.
- **Current evolutionary boundary:** a world is a pure clonal evaluator. There
  is no in-world reproduction or genome mutation; generational variation lives
  in `sim-core::evolution` and the `neat` run mode. Legacy lineage/reproduction
  readouts therefore remain zero-depth and are not evolutionary evidence.
- **Current lifetime boundary:** survival is energy-only. The serialized
  `max_organism_age` value is legacy telemetry and is not an enforced age cap.

## Global flags

- `--in <world.bin>` — world to read (required by every command except `new`).
- `--out <world.bin>` — where a mutating command writes the advanced world.
  **Defaults to `--in`** (advance in place); pass a different path to fork.
- `--metrics <path>` — explicit sidecar (default: the `<world>.metrics` sibling).
- `--no-metrics` — disable recording for this call (fast scrub path).
- `--out-dir <dir>` — directory for run-mode result files (default
  `artifacts/runs`). Used by `sweep` and `neat`.
- `--json` / `--text` — output format (JSON is the default).

## Commands

### Mutating (persist the world)
- `new [--config P] [--seed N] [--seed-genome-snapshot P] [--set k=v]… [--scale W,POP] [--threads K] [--report-every R] --out w.bin [--no-metrics]`
  — construct a world. `--set` patches config fields inline (e.g.
  `--set food_energy=12 --set passive_metabolism_cost_per_unit=0.0035`), no temp
  config files. `--scale W,POP` overrides world_width,num_organisms (marks the
  world non-canonical, persistently). `--seed-genome-snapshot` accepts a bincode
  `OrganismGenome` (e.g. an evaluation genome snapshot under
  `artifacts/evaluation/.../genomes/`) and uses it as the exact founder genome.
  (`neat` no longer needs this: it writes a ready-to-run `world.bin` directly —
  see below.) Mints the metric sidecar unless `--no-metrics`.
- `step [N] --in w.bin [--out w.bin]` — advance N ticks (default 1).
- `run-to T --in w.bin [--out w.bin]` — advance until turn == T (no backward).
- `watch T [--every E] --in w.bin [--out w.bin]` — advance to T, emitting a
  metrics row every E ticks (JSONL progress log).
- `bench [N] --in w.bin` — time N ticks; the advanced world is discarded
  (no `--out`).

### Reads (stdout only, no `--out`)
- `turn` — current turn.
- `state` — population, energy/health/age/generation summaries, food, last-turn
  ecology, and the fail-closed per-tick energy ledger (+ a pillars line if a
  sidecar is present). JSON exposes signed organism/food compartment totals,
  every explicit source/sink/transfer, residuals, and the scale-aware tolerance.
- `pillars` — the **raw windowed-mean metrics** (no [0,1] interpretation):
  `plant_consumption_rate` (foraging), `prey_consumption_rate` (predation),
  `action_effectiveness` + `mi_sa` (intelligence), `learning_slope` (learning) —
  **plus a `granular` section** with the full per-interval series behind the
  window (the scoring window is marked). Interval facts are accumulated when
  actions occur, so living survivors contribute to the tail. **Needs a sidecar.**
- `eco` — population/food trajectory, deaths-by-cause, rates (trajectory needs a
  sidecar; point-in-time works without).
- `evolution` — legacy in-world reproduction readout. The current pure-clonal
  engine produces no births; do not use this command as outer-loop evidence.
- `lineage` — founder-lineage composition (generation depth stays zero in the
  current pure-clonal engine).
- `genome [--gene G] [--drift]` — per-gene population distributions.
- `timeseries [--cols LIST] [--last K]` — recorded columns as sparklines.
  **Needs a sidecar.** Columns include per-tick `population food deaths
  consumptions predations` (legacy birth/reproduction columns stay zero) and per-interval
  `action_effectiveness plant_consumption_rate prey_consumption_rate mi_sa
  learning_slope interval_descendants`. The legacy interval label maps to the
  raw `pop` field, which is the total live population in the current engine.
- `food` — plant/corpse counts, energy, coverage.
- `inspect ID` · `top FIELD [N]` · `hist FIELD` · `find EXPR [--fields LIST]
  [--limit N]` · `brain ID [--view summary|synapses|activations|dot]` ·
  `decide ID` — per-organism inspection. Fields/vocabularies: invalid args print
  the valid set.
- `query --in w.bin` — **batch reads**: read read-only commands from stdin (one
  per line) against a single world load. Use for a burst of probes so you pay one
  deserialize, not one per command. Mutating commands are rejected. The batch
  continues to emit all command results but exits nonzero if any line failed.

### Interactive TUI (human-facing, not for agents)

- `tui --in w.bin` — open an existing world in a **split-pane terminal UI**:
  a status bar (`t=… · pop … · plants … · energy …`, `●` marks unsaved changes),
  a scrollback pane, and a command box. `tui --new [--seed N] [--set k=v]…
  [--config P] [--scale W,POP]` starts a fresh world instead (same build flags as
  `new`). Exactly one of `--in`/`--new` is required.
- The world stays **resident in memory**, so reads and advances are instant. Type
  any read or advance command — `state`, `inspect 42`, `top energy 5`, `run-to
  5000` (press **Esc** to cancel a long run), `pillars`, `brain 42` — using the
  same vocabulary as the one-shot CLI. Plus session commands: `save [path]`,
  `help`, `clear`, `quit` (`quit!` discards unsaved changes).
- **Nothing is written unless you `save`.** A bare `save` writes back to the
  opened `--in` file (`--new` worlds require an explicit `save <path>`); the
  metric sidecar follows automatically.
- Keys: `Up`/`Down` scroll the output (wrapped), `PageUp`/`PageDown` by a page,
  `Home`/`End` top/bottom · `Ctrl-P`/`Ctrl-N` command history · `Ctrl-C` quit.
  Long output lines wrap; new output snaps the view back to the bottom.
- Scripting/verification: `tui … --exec "cmd; cmd; …"` runs a `;`-separated
  sequence headlessly (no terminal) and prints the resulting output — handy for
  smoke tests or reproducible sessions.
- This mode is for a human researcher poking at one world. Agents should keep
  using the stateless one-shot commands above.

### Run modes (own their worlds, write a result file)
- `sweep --grid KEY=v1,v2[,…] [KEY2=…] --seeds N[,N…] --to TICK [--config P]
  [--baseline KEY=v,…] [--report-every R] [--threads K] [--jobs J] [--out-dir D]`
  — run the cartesian product of config overrides × seeds, score each cell's
  pillars (mean/min/max across the seed cohort), and
  write a JSON result file to `--out-dir` (default
  `artifacts/runs/sweep-<ts>.json`). The result also retains each seed's raw
  actual `final_turn`, raw pillar and evolution summaries, plus same-seed paired
  deltas against the baseline with sample SD/SE, 95% Student-t confidence intervals, and
  win/tie/loss counts. Prints the file path and compact pillar, evolution, and
  paired-delta tables to stdout.
  Legacy crossover gates target the removed in-world reproduction path and are
  not valid evidence for the current pure-clonal engine.
  `KEY`s are config field names (same vocabulary as `new --set`). Jobs run in
  parallel (bounded to `--jobs`, default = CPU count; each run uses `--threads`
  intent threads, default 1, so parallelism comes from running many worlds at
  once). Example:
  `sweep --grid food_energy=10,12,14 --seeds 7,42,123,2026 --to 500000
  --baseline food_energy=12`.

- `powerplay [--config P] [--seed N] [--depth 1..4] [--population N]
  [--generations N] [--module-width N] [--ticks-per-stage N]
  [--world-width N] [--food-energy F] [--search-seeds N,...]
  [--episode-seeds N,...] [--out-dir D]` — run the bounded sequential-resource
  Causal Ecology PowerPlay pilot. Search and admission each require 16 unique,
  disjoint contexts; accepted residuals must retain every archived prefix while
  every earlier checkpoint and the exact residual knockout fail the new task.
  The result JSON preserves the task x solver matrix, per-episode action traces,
  fixed-escrow closure, and maximum engine-ledger residuals. This command is a
  falsification scaffold, **not open-endedness evidence**: its visible-target
  grammar is finite, depth is capped at four, and payoff does not gate survival.

- `neat [--config P] [--seed N] [--population N] [--generations N]
  [--episode-horizons T[,T...]] [--world-seeds N,N]
  [--audit-seeds N,N|--no-audit]
  [--holdout-seeds N,N|--no-holdout] [--audit-levels N,N] [--audit-every N]
  [--workers K] [--scenarios baseline,scarcity,sparse_search] [--scale W,POP]
  [--no-scale] [--set world_key=value] [--param neat_key=value] [--out-dir D]`
  — run canonical generational NEAT as a separate outer loop. It implements a
  run-owned monotonic innovation registry, matching/disjoint/excess crossover
  with fitter-parent inheritance, compatibility-distance speciation, explicit
  fitness sharing, species stagnation protection, elitism, and complexification
  by add-connection and split-connection/add-node mutations. Run `neat --help`
  for every controllable NEAT parameter.

  Each scenario/seed/horizon case scores candidate founder survival integrated
  over the episode. The default objective averages the lower-tail fraction of
  cases selected by `objective_cvar_fraction`; optional objectives weight later
  survival or contemporaneous relative advantage. Consumption, action, and
  ecology values are diagnostics, not the selection target. Development-audit
  seeds never affect selection, and holdout seeds are untouched until champion
  selection has ended.
  Every candidate is evaluated as an exact clonal founder cohort. Runtime
  plasticity follows the frozen world flag; in-world reproduction and genome
  mutation do not exist. Candidate worlds use one
  intent thread while independent candidates are evaluated in parallel. The
  world config defaults `leaky_neurons_enabled=false`, yielding instantaneous
  `tanh(input)` hidden neurons; when disabled, NEAT also suppresses neutral
  time-constant mutation. The flag can be enabled explicitly for a controlled
  dynamics ablation. `predation_enabled=false` is also the default: the active
  controller interface is five sensors (three food rays, contact, energy) and
  five non-idle actions. Enabling it adds organism rays, health, Attack, and
  predation as one treatment. NEAT does not create or retain genes incident to
  the disabled receptors or Attack output. The default scenario set excludes
  hazards.
  default evaluation scale is deliberately cheap (`25×25`, 30 founders); pass
  `--no-scale` to use the config's canonical size.

  Progress is JSONL on stderr. Stdout returns the durable result JSON path and a
  `<result>.world.bin` (with a minted `.metrics` sidecar) — a clonal colony
  seeded from the champion genome that is directly `run-to`/`pillars`/`inspect`/
  `brain`-able, so there is no separate champion format to re-inject. The result
  JSON contains the complete NEAT
  hyperparameter set, fixed world seeds and horizon, effective world dimensions,
  frozen-contract values, per-generation/species statistics, and champion
  genome. Development and sealed records include exact per-case plant supply,
  actionable supply (excluding final-tick spawns), plant capture, standing-plant
  pressure, time to first plant, spatial coverage, and action distributions.
  Conservation checks fail the run if plant supply does not equal consumed plus
  final standing plants. Evolved-structure contribution is reported both as a
  pure knockout and as a separate ancestral-collapse counterfactual that
  re-enables every initial connection.

  `neat analyze RESULT.json [RESULT2.json ...]` derives compact early/late
  competence and path-connected-complexity slopes, fixed-level foraging trends,
  late innovation supply/zero-origin streaks, adaptive structures that spread,
  and development/sealed knockout deltas from durable results without rerunning
  simulation.

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
