# `cli` — task ecology and stateless world interface

`cli` is the sole headless research interface. It has two explicit namespaces:

```text
cli ecology <task> ...
cli world <command> ...
```

Output is JSON by default. Task runs stream progress JSON to stderr and write a
compressed result. World commands are stateless one-shot operations over an
explicit world file.

Build a release binary for experiments:

```bash
cargo build -p cli --release
CLI=./target/release/cli
```

## Task ecology

The built-in environments all use the same brain adapter, evaluation
protocol, asexual mutation, and finite reproduction algorithm:

```bash
$CLI ecology reaction [run|plan] [OPTIONS]
$CLI ecology memory [run|plan] [OPTIONS]
$CLI ecology memory evaluate FROZEN [OPTIONS]
$CLI ecology next-token [run|plan] [OPTIONS]
$CLI ecology continual [run|plan] [OPTIONS]
$CLI ecology renewable [run|plan] [OPTIONS]
$CLI ecology analyze RESULT...
```

`run` is optional. `plan` validates and prints the complete configuration and
maximum task-step budget without evolving. `analyze` reads JSON or `.json.zst`
result artifacts.

Shared options:

- `--seed N`: evolutionary run seed.
- `--population N`: number of genomes and offspring slots.
- `--generations N`: evaluation/reproduction depth.
- `--workers N`: parallel evaluation workers. The default is available
  hardware parallelism.
- `--training-instances N`, `--development-instances N`,
  `--sealed-instances N`: generic panel sizes; these are evaluator settings,
  not task configuration.
- `--training-rollouts N`, `--development-rollouts N`, `--sealed-rollouts N`:
  deterministic rollouts per instance.
- `--seed-config PATH`: founder-genome TOML; defaults to the canonical config.
- `--exact-elites N`: unchanged leading genomes copied between generations.
- `--tournament-size N`: competitors sampled per non-elite offspring slot.
- `--exploration-temperature F`: action-sampling temperature multiplier.
- `--action-selection greedy|sampled`: whether evaluation acts from the
  categorical argmax or deterministic sampling.
- `--learning-rule disabled|immediate_policy|target_prediction_error|temporal_prediction_error`:
  generic plasticity-driving signal. Memory defaults to the calibrated
  immediate-policy rule, next-token prediction uses exact categorical target
  error, and continual tasks use temporal prediction error.
- `--learning-normalization none|nlms`: generic plasticity normalization.
- `--reset-dynamics-at-trial-boundary true|false`: adapter policy at semantic
  trial boundaries. Learned weights are retained.
- `--audit-interval N`: development-audit interval.
- `--param key=value`: override an asexual mutation parameter. Valid keys are
  printed by `cli ecology help` and invalid keys fail explicitly.
- `--out-dir PATH`: result directory; may appear anywhere after `ecology`.

Search does not use scalar fitness, speciation, crossover, target species,
topology rewards, novelty, or in-evaluation births. Each task success event is
one reproductive ticket. After equal evaluation panels finish, one finite
population-sized set of offspring slots is filled by exact elites and fixed-K
tournaments followed by bounded asexual mutation. A generation with no success
events is extinct.

### Reaction

```bash
$CLI ecology reaction plan --population 64 --generations 100
$CLI ecology reaction --seed 17 --population 64 --generations 100 \
  --symbols 17 --out-dir artifacts/research/runs
```

Task option: `--symbols N`, including the terminal `end` observation. Instances
contain shuffled `a`-`d` symbols and a final `end`. A correct matching action is
one success event.

### Memory

```bash
$CLI ecology memory plan --population 256 --generations 500
$CLI ecology memory --seed 101 --population 256 --generations 500 \
  --length 4 --attempts 32 --out-dir artifacts/research/runs
```

Task options: `--length N`, `--attempts N`. The agent receives zero symbolic
input and repeatedly emits a sequence over `a`-`h`. During the default 32
learning attempts, the environment returns balanced immediate reward but no
reproductive success events. Learning is then disabled and a greedy final
probe emits one success event for each correct position; positions are
symmetric, and exact-string accuracy requires all positions to be correct.
Attempt completion is a semantic trial boundary; the adapter owns the
neural-state policy applied there. The default memory preset uses a fixed
100-instance, two-rollout training panel, a 100-instance development panel,
and a 100-instance, two-rollout sealed panel. Larger panels remain available
through the generic panel options but are not the default.

`memory evaluate FROZEN` runs a persisted frozen genome through the same
development/sealed adapter without evolution. It accepts both a bare genome and
the historical frozen-wrapper format.

### Basic next-token prediction

```bash
$CLI ecology next-token plan --population 256 --generations 100
$CLI ecology next-token --seed 101 --population 256 --generations 100 \
  --out-dir artifacts/research/runs
```

The canonical training snippet is `the quick brown fox jumps over the lazy
dog`. Starting from a boundary token, the brain is teacher-forced through the
entire prefix one character at a time and predicts the next character at every
position, including the first character and terminal `end`. Every correct
greedy probe prediction is one success event; exact accuracy requires all 44
targets. The default learner receives four complete supervised passes over the
snippet. Recurrent dynamics reset at each pass boundary while learned weights
persist. After the fourth pass, dynamics reset again and plasticity is frozen
for the scored greedy probe. The common symbol interface contains `a`--`z`,
`space`, and `end`; other tasks expose only their declared subsets.

Task option: `--learning-passes N`. Four is the calibrated default; 16 and 32
passes added compute without improving the discovery run's frozen accuracy.

### Basic continual learning

```bash
$CLI ecology continual plan --population 256 --generations 250
$CLI ecology continual --seed 101 --population 256 --generations 250 \
  --lifetime-ticks 512 --minimum-regime-ticks 32 \
  --maximum-regime-ticks 96 --out-dir artifacts/research/runs
```

The agent receives zero symbolic input during one uninterrupted lifetime. One
hidden action is rewarded at a time, and the target switches to a different
action after a deterministic 32--96 tick regime. Correct actions are atomic
success events. The environment emits no trial boundary, so recurrent dynamics
and learned weights persist across every reversal. The common adapter uses the
generic temporal prediction-error learning rule by default.

### Renewable hidden resource

```bash
$CLI ecology renewable plan --population 256 --generations 250
$CLI ecology renewable --seed 101 --population 256 --generations 250 \
  --lifetime-ticks 512 --resource-stock 64 \
  --out-dir artifacts/research/runs
```

Task options: `--lifetime-ticks N`, `--resource-stock N`. The agent receives
zero symbolic input. Each correct hidden-target action is a success event; after
the stock is consumed the target changes deterministically without a trial
boundary.

### Progress and results

Each generation event reports completed/total generations, progress percent,
leading accuracy and resources, periodic development accuracy, topology size,
elapsed seconds, and ETA. The terminal stdout object reports the result path,
termination, selected generation, development and sealed controls, total work,
and wall time.

Result artifacts contain the complete task, agent, ecology, search, founder,
generation, population, work, development, sealed, and termination contracts.
Development and sealed audits include efference-copy-off and
prediction-error-feedback-off controls. Audit scores retain a historical
representative but never allocate reproduction. The plasticity-off control was
retired after repeated tasks established that the current learner is causal.

## Explicit world simulator

The simulator is a stateless one-shot CLI. A world is always an explicit file.
Every call loads `--in`, performs one command, and exits. Mutating commands write
`--out`; when omitted, `--out` defaults to `--in` and advances in place.

```bash
$CLI world new --seed 7 --out artifacts/worlds/base.bin
$CLI world run-to 500 --in artifacts/worlds/base.bin
$CLI world brain 0 --in artifacts/worlds/base.bin
```

Do not expect process memory to survive between invocations. Snapshot or fork a
world with `cp`, and fan out independent runs by backgrounding invocations.
Keep worlds under `artifacts/`, not `/tmp`.

### World and metric files

- `--in WORLD`: input world, required except for `new`.
- `--out WORLD`: output world for a mutating command; defaults to `--in`.
- `--metrics PATH`: override the metric sidecar location.
- `--no-metrics`: disable sidecar loading and persistence.

`new` mints `<world>.metrics`. The sidecar follows the output world and is
required by `pillars`, `eco` trajectory, and `timeseries`. Copy both files when
forking a measured world. Raw world state remains readable without the sidecar.

### Mutating commands

```text
new [--config P] [--seed N] [--seed-genome-snapshot P]
    [--set k=v]... [--scale W,POP] [--threads K]
    [--report-every R] --out WORLD
step [N] --in WORLD [--out WORLD]
run-to T --in WORLD [--out WORLD]
watch T [--every E] --in WORLD [--out WORLD]
```

`new` reads canonical TOML by default. `--set` overrides a documented world
configuration key. `--seed-genome-snapshot` loads one bincode
`OrganismGenome`, used for every founder. `step` advances by a relative count;
`run-to` advances to an absolute turn; `watch` emits periodic status while
advancing.

### Read-only commands

```text
turn | state | pillars | eco | lineage | genome --in WORLD
timeseries | inspect | top | hist | find | brain | decide --in WORLD
query --in WORLD
```

Read commands never write a world. `query` batches read commands over one load.
`pillars` returns shared raw windowed metrics—plant/prey consumption rates,
action effectiveness, `mi_sa`, and learning slope—with no implied `[0,1]`
interpretation. Its `granular` field contains the per-report-interval series.

### Throughput and sweeps

```text
bench [N] --in WORLD
sweep --grid k=v,v --seeds N,N --to T [--out-dir D]
```

`bench` measures tick throughput without persisting an advanced world. `sweep`
runs the grid by seed in parallel and writes a result under `--out-dir`
(default `artifacts/runs/`).

### Interactive mode

```bash
$CLI world tui --in artifacts/worlds/base.bin
$CLI world tui --new --seed 7
```

The TUI keeps one resident world and reuses the same world-command dispatch.
Changes remain in memory until `save`; `quit` warns about unsaved changes.

## Artifact policy

Generated worlds, datasets, logs, and rendered outputs belong under
`artifacts/research/runs/`. Durable hypotheses, proposals, conclusions, and the
experiment index belong under `research/`.
