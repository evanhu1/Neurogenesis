# sim-cli — usage reference

`sim-cli` is the **agent-facing research CLI** for the NeuroGenesis engine: a
**stateless, one-shot CLI** where a simulation world is an explicit file artifact.
Each invocation reads a world from `--in`, runs one command, and (for mutating
commands) writes the advanced world to `--out`. Output is **JSON**.

> **Why a file, not a REPL?** State on disk means: snapshot/fork a world with
> `cp`, fan out parallel runs by backgrounding invocations, and never lose work to
> a crashed long-lived process. The process exiting is the completion signal;
> stdout is the framed JSON result.

## Run

```bash
cargo build -p sim-cli --release        # --release strongly recommended
./target/release/sim-cli <command> [args] [--in w.bin] [--out w.bin]
```

## Core model

- **A world is a file** (`world.bin`) — a bincode-serialized `HexSim` (the hex
  world + its `PopulationDriver`). Built by `new`, advanced by `step`/`run-to`.
  Forward-only; to branch, `cp` it.
- **No metric sidecar.** Reads are computed directly from the world; there is no
  `<world>.metrics` file (the old ETL/pillars stack was removed).
- **Determinism**: same seed + tick count = identical world bytes; save→load→
  advance is byte-identical to advancing in RAM.
- **Extinction is terminal**: `run-to`/`step` stop as soon as the world hits zero
  living organisms and report `"extinct": true` with the `extinct_at` turn. There
  is no periodic injection — a dead world stays dead.

## Flags

- `--in <world.bin>` — world to read (required by every command except `new`).
- `--out <world.bin>` — where a mutating command writes the advanced world.
  **Defaults to `--in`** (advance in place); pass a different path to fork.
- `new` also takes `--seed N`, `--width W`, `--founders F`, `--energy E`.

## Commands

### Mutating (persist the world)

- `new [--seed N] [--width W] [--founders F] [--energy E] --out w.bin` —
  construct a fresh world seeded with `F` founders from the primordial seed
  genome on a `W×W` hex grid.
- `step [N] --in w.bin [--out w.bin]` — advance N ticks (default 1); stops early
  on extinction.
- `run-to <turn> --in w.bin [--out w.bin]` — advance until `turn` (or extinction).

### Reads (stdout JSON only, no `--out`)

- `state` — population summary: turn, alive, total ever born, `extinct_at`, and
  mean energy / neurons / edges / generation + max generation.
- `lineage` — living-population generation histogram.
- `find <field> <op> <value>` — filter living organisms. `field` ∈
  `energy | health | age | generation | neurons | edges`; `op` ∈
  `gt | lt | ge | le | eq`. Returns matching ids + the field value.
- `inspect <id>` — one organism: energy/health/age/generation, brain size
  (in/hidden/out + edges), CPPN size (nodes/conns), morphology, and key
  lifecycle/plasticity header genes.
- `genome <id>` — the full CPPN `Genome` as JSON (nodes, connections, header).
- `brain <id>` — the full developed `BrainNet` as JSON (neurons + edges).
- `decide <id>` — run the organism's current sensing + brain once and print the
  softmax action distribution (per actuator + implicit idle) plus the observation
  vector it saw.

## Workflow

```bash
BIN=./target/release/sim-cli
$BIN new --seed 7 --width 32 --founders 200 --out artifacts/w.bin
$BIN run-to 500 --in artifacts/w.bin
$BIN state   --in artifacts/w.bin
$BIN lineage --in artifacts/w.bin

# find a high-energy organism, then interrogate it
ID=$($BIN find energy gt 300 --in artifacts/w.bin | python3 -c 'import json,sys;print(json.load(sys.stdin)["matches"][0]["id"])')
$BIN inspect $ID --in artifacts/w.bin
$BIN decide  $ID --in artifacts/w.bin
$BIN brain   $ID --in artifacts/w.bin
$BIN genome  $ID --in artifacts/w.bin

# fork a world to branch a counterfactual
cp artifacts/w.bin artifacts/fork.bin
$BIN run-to 2000 --in artifacts/fork.bin
```

## Notes

- Put worlds under `artifacts/` (not `/tmp`) so they survive a session.
- For multi-seed / behavioral-coverage evaluation across many worlds, use
  `sim-evaluation` (Quality-Diversity coverage / QD-score summary) rather than the
  CLI.
