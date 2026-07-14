# cli — research interface

`cli` is the sole headless research interface. It is deliberately stateless:
each invocation loads an explicit world file, performs one command, prints a
result, and—only for a mutating command—writes the updated world back out.

Build it first:

```bash
cargo build -p cli --release
CLI=./target/release/cli
```

## World-as-file model

Start a new deterministic world and advance it in place:

```bash
$CLI new --seed 7 --out artifacts/worlds/base.bin
$CLI run-to 5000 --in artifacts/worlds/base.bin
$CLI pillars --in artifacts/worlds/base.bin
```

`new` also creates `<world>.metrics`, a sidecar of raw observation facts.
Copy both files when forking a world:

```bash
cp artifacts/worlds/base.bin artifacts/worlds/arm-a.bin
cp artifacts/worlds/base.metrics artifacts/worlds/arm-a.metrics
```

The world itself is the authoritative simulation state. The metric sidecar is
observational only: it never changes a tick's random choices or outcomes.
`pillars`, `eco`, and `timeseries` need it. Use `--no-metrics` only when a
sidecar is intentionally unnecessary.

The canonical baseline is `config/world.toml` plus
`config/seed_genome.toml`. `new --set key=value` makes an explicit experiment
override; `--scale WIDTH,POPULATION` is a convenience override. Persisted worlds
record whether that shortcut was used.

## Model semantics

- A world is a finite, clonal evaluator. It has no in-world reproduction or
  genome mutation.
- The outer `neat` command owns generations, selection, speciation, crossover,
  and mutation.
- Each tick builds all action intents from a common snapshot, resolves moves,
  resolves eating and attacks, applies optional Hebbian plasticity, then drains
  one energy from each survivor. This order is canonical.
- Plant eating transfers a plant's full energy to the eater. Attacks transfer
  energy directly from victim to attacker. There are no public energy caches or
  hidden evaluator rewards.
- With the default `leaky_neurons_enabled=false`, each brain decision is a
  feed-forward pass. The `EnergyFlowLastTick` sensory receptor is the only
  across-tick signal in the default interface; it is an explicit current input,
  not a retained hidden activation.

`pillars` reports raw, windowed behavior proxies: action effectiveness,
plant/prey consumption rates, a coarse sensory/action mutual information
estimate, and an age/success slope. They are diagnostics, not normalized scores
or fitness by themselves. Inspect organisms with `top`, `find`, `inspect`,
`brain`, and `decide` before interpreting an aggregate trend as a behavior.

## Global flags

- `--in PATH`: input world for all commands except `new`.
- `--out PATH`: output world for mutating commands. Defaults to `--in`.
- `--metrics PATH`: explicit metric sidecar location.
- `--no-metrics`: do not load or persist a sidecar.
- `--out-dir PATH`: durable result directory for `sweep` and `neat`.
- `--text`: human-readable output instead of JSON where supported.

Invalid command arguments print their valid options. JSON is the default so
scripts can consume results without parsing prose.

## Commands

### Create and advance worlds

- `new [--config P] [--seed N] [--seed-genome-snapshot P] [--set k=v]... [--scale W,POP] [--threads K] [--report-every R] --out WORLD`
  creates a new world. The optional genome snapshot makes every founder a clone
  of that stored genome.
- `step [N] --in WORLD [--out WORLD]` advances `N` ticks (default 1).
- `run-to T --in WORLD [--out WORLD]` advances until exactly tick `T`.
- `watch T [--every E] --in WORLD [--out WORLD]` advances to `T` and records
  a metric interval every `E` ticks.
- `bench [N] --in WORLD` runs `N` ticks for throughput only; it does not save
  the simulated result.

### Read a world

All read commands write stdout only and never change the world:

- `turn`, `state`, `pillars`, `eco`, `lineage`, `genome [--gene G]`, `food`
- `timeseries [--cols LIST] [--last K]`
- `inspect ID`, `top FIELD [N]`, `hist FIELD`, `find EXPR`
- `brain ID [--view summary|graph|weights]`, `decide ID`
- `query --in WORLD`, which reads one read-only command per stdin line after a
  single world load.

`decide` exposes the current sensory activations, logits, and chosen action for
one organism. It is the appropriate tool for connecting a claimed strategy to
the controller that produced it.

### Batch world probes

`sweep --grid key=v1,v2... --seeds s1,s2... --to T [--baseline key=value]`
runs a deterministic grid of ordinary worlds in parallel. It is for simulator
or ecology probes, not an alternative evolution algorithm. It writes a durable
JSON result under `--out-dir` (default `artifacts/runs`).

### Generational NEAT

`neat` is the canonical outer evolutionary loop:

```bash
$CLI neat \
  --population 24 --generations 50 --episode-horizons 5000 \
  --world-seeds 7,17,27,37 --out-dir artifacts/research/runs
```

Important arguments include `--population`, `--generations`,
`--episode-horizons`, `--world-seeds`, `--audit-seeds`, `--holdout-seeds`,
`--scale`, and `--param key=value`. Each result persists the complete resolved
NEAT parameter set, so that file is the definitive run contract.

By default, each candidate receives eight contemporary-opponent exposures over
four deterministic world seeds. Two lineages share each evaluator world and are
scored from the same simulation. `--param eval_lineages_per_world=3` creates a
three-lineage mini-ecosystem; `eval_opponents` still means total opponent
exposures, so the number must be compatible with the chosen group size.

Fitness is an episode survival measure, optionally multiplied by relative
survival against the other lineages. Gross energy and behavioral metrics are
persisted as secondary diagnostics. The run emits progress JSONL on stderr and
prints a result JSON path on stdout. That result contains the exact config,
seed suite, every generation, complete population checkpoints, and champion.

Useful follow-ups:

```bash
$CLI neat analyze artifacts/research/runs/RESULT.json
$CLI neat crossplay artifacts/research/runs/RESULT.json
$CLI neat evaluate-panel --focal RUN_A.json --opponents RUN_B.json --world-seeds 101,103,107
```

`crossplay` compares persisted generation champions in the same competitive
setting; it is the right diagnostic for historical-opponent forgetting and
nontransitive cycles. It is not a solo/cannibalism evaluation.

## Reliable research practice

Put worlds and outputs under `artifacts/`, not `/tmp`. Snapshot with `cp`, fan
out independent CLI calls in the shell, and retain result JSONs alongside the
commands/configurations that produced them. A fixed config and seed must yield
identical world bytes and NEAT results on the same build; treat any violation as
a bug.
