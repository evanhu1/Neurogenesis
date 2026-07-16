# `cli` — NEAT research interface

`cli` is the sole headless research interface. Generational NEAT is its default
mode. The lower-level stateless simulator is intentionally namespaced under
`cli world` so an ordinary research command cannot accidentally be mistaken for
a single-world probe.

Build it first:

```bash
cargo build -p cli --release
CLI=./target/release/cli
```

JSON is the default output. Invalid arguments fail with a structured error.

## Core workflow

The main commands are:

- `$CLI [RUN OPTIONS]` or `$CLI run [RUN OPTIONS]`: execute one NEAT run.
- `$CLI plan [RUN OPTIONS]`: validate the same contract without simulating.
- `$CLI batch ... -- [RUN OPTIONS]`: execute several evolutionary seeds as one
  reproducible experiment.
- `$CLI summarize EXPERIMENT`: validate a batch and report contextual
  generation-winner snapshots and structural diagnostics.
- `$CLI analyze RESULT...`: inspect structural history and contextual snapshots;
  it does not infer longitudinal competence.
- `$CLI crossplay RESULT ...`: run a pairwise transfer assay over frozen
  generation winners. This is the only command that compares competence across
  generations.
- `$CLI materialize RESULT --generation N --out WORLD.bin`: create a visual,
  turn-zero clonal world from a persisted generation winner.
- `$CLI world COMMAND ...`: use the ordinary world simulator.

## Planning a run

Always plan a nontrivial run before executing it:

```bash
$CLI plan \
  --population 48 \
  --generations 40 \
  --horizon 500 \
  --world-seeds 11,29,47,61 \
  --founders 96 \
  --workers 14
```

`plan` loads the real world config and calls the same run-option validation used
by execution. It prints:

- simultaneous lineages and founders per lineage;
- contemporary opponent exposure and scored cases per genome;
- evaluator worlds per generation;
- total evaluator worlds and world ticks;
- requested/effective evaluator workers and the machine parallelism;
- the resolved selection-score contract, seeds, horizon, width, and founder count.

No world is simulated and no artifact is written.

### Evaluation-budget terminology

One world configuration produces:

```text
world seeds × horizons
```

scored cases for every lineage represented in that world. Pairwise evaluation
repeats that case bundle for each scheduled opponent; shared-population
evaluation scores all contemporaries together in each case.

Use exactly one budget form:

- `--opponents-per-genome N` directly controls the competitive panel; or
- `--cases-per-genome N` asks the CLI to derive the exact opponent count.

`--cases-per-genome` must divide evenly by the cases per opponent.

### Run options

- `--seed N`: evolutionary seed.
- `--population N`: genomes in each generation.
- `--generations N`: evaluated generations.
- `--population-checkpoint-interval N`: retain the complete evaluated
  population every N generations; defaults to `10`. Every generation winner is
  retained for later crossplay, and the final population is always checkpointed.
- `--horizon N`: ticks in every evaluator world; default `500`.
- `--evaluator pairwise|shared_population`: arrange genomes into sampled
  two-lineage worlds or one world containing the complete contemporary
  population. The default is `shared_population`; use `--evaluator pairwise`
  to opt into paired matchups.
- `--opponents-per-genome N` or `--cases-per-genome N`: evaluation budget.
- `--world-seeds N,N`: deterministic training layouts.
- `--cvar F`: mean only the worst-performing fraction of cases into the
  same-generation selection score;
  `1.0` is the ordinary mean.
- `--workers N`: parallel evaluator groups.
- `--founders N`: organisms divided equally among lineages.
- `--world-width N`: width of the square axial hex world.
- `--set key=value`: explicit world override.
- `--config PATH`: world TOML; defaults to `config/world.toml`.
- `--out-dir PATH`: generated result directory.

Pairwise evaluation requires an even genome population and an even founder
count. Shared-population evaluation derives `population - 1` opponents, rejects
an explicit opponent/case budget, and requires the founder count to be exactly
divisible by the genome population. Every shared world scores every genome
once; a shared case therefore represents all contemporaries simultaneously
rather than one independently attributable opponent.
NEAT parallelizes independent evaluator groups and fixes each world's internal
intent evaluation to one thread; use `--workers`, not
`--set intent_parallel_threads=...`.

`--param key=value` remains available for explicit mutation, speciation,
selection, plasticity, and complexification experiments. Evaluation layout,
contextual-score aggregation, world size, and founder count have first-class
flags and cannot be smuggled through duplicate generic parameters.

## Running one evolutionary seed

Remove `plan` from a validated command and add the evolutionary seed:

```bash
$CLI \
  --seed 17 \
  --population 48 \
  --generations 40 \
  --horizon 500 \
  --world-seeds 11,29,47,61 \
  --founders 96 \
  --workers 14 \
  --out-dir artifacts/research/runs
```

Progress is JSONL on stderr. The completion JSON on stdout points to the result
and materialized final-generation-winner world. The compressed
`result.json.zst` persists the resolved world/NEAT contract, contextual
generation winners, periodic complete population checkpoints, and the final
population.
CLI research commands read the compressed result directly.

## Reproducible multi-seed batches

`batch` owns evolutionary seeds, per-seed worker allocation, output naming,
logs, source identity, and artifact checksums:

```bash
$CLI batch \
  --experiment attack-objective-ablation-control \
  --seeds 7,17,27 \
  --total-workers 14 \
  --out-dir artifacts/research/runs/active \
  -- \
  --population 48 --generations 40 --horizon 500 \
  --world-seeds 11,29,47,61 --founders 96
```

The machine-wide worker budget is divided deterministically in seed order.
`batch` rejects `--seed`, `--workers`, and `--out-dir` after the `--` separator.
Child progress is streamed live to the batch's stderr while the exact same
JSONL is persisted in `seed-N.progress.jsonl`. Every `neat_generation` record
includes `run_seed` and a `progress` object with completed/total generations,
generation time, elapsed time, rolling mean seconds per generation, and ETA.
It writes one stable experiment directory containing `manifest.json`, per-seed
results and final-winner worlds, stdout/progress logs, source revision and dirty
status, and SHA-256 checksums. Existing nonempty experiment directories require
an explicit `--replace`.

Summarize a completed batch:

```bash
$CLI summarize artifacts/research/runs/active/attack-objective-ablation-control
```

Summarization validates result schemas, generation sequences, seeds, and
resolved contracts. It reports each generation winner in its original
contemporary context plus structural and behavioral observations. It does not
compute population-mean competence, all-time champions, score deltas, tail
slopes, or any other cross-generation performance claim. Run `crossplay` for
that question.

## Selection and observation metrics

The selection objective is focal-founder alive-ticks divided by `focal founders
× episode horizon`, aggregated over the configured lower tail of evaluation
cases. This is a contextual, same-generation ranking signal: changing the
contemporary opponent population changes the task. A score from generation N
must not be interpreted as higher or lower competence than a score from
generation M.

The evaluator is a closed-transfer combat arena:

- every founder begins with the same configured energy;
- successful attacks move energy from victim to attacker without creating it;
- passive metabolism and attack-attempt costs dissipate energy from the world.

Thus stealing itself is exactly zero-sum between organisms, while the complete
energy ledger is intentionally dissipative rather than constant-sum.

In shared-population mode, all genomes receive survival and behavior facts from
the same simulation. Results also retain each genome's case-score standard
deviation and extrema across world seeds/scenarios/horizons. Per-opponent score
profiles are intentionally absent because a multi-lineage result cannot be
causally assigned to any single opponent.

Energy-flow and behavior fields are diagnostics, not extra rewards:

- `gross_energy_acquired` is successful attack transfer received. It excludes
  starting energy and attack costs.
- `attack_energy_received` is successful transfer into focal attackers.
- `attack_energy_lost` is successful transfer out of focal victims.
- `attack_attempt_energy_cost` is energy paid for every focal attack action,
  including misses and blocked attempts.
- `net_attack_energy_balance` is `received - lost - attempt cost`.
- `net_energy_profit` is `attack energy received - attack energy lost - attack
  attempt cost`; passive metabolism is deliberately excluded.
- `attack_precision` is successful hits divided by all attack attempts.
- `attack_target_evaded` counts attacks whose snapshot target moved away before
  interaction resolution.
- `distinct_attack_victims` counts unique organisms successfully hit.

The default categorical controller samples one command from implicit idle,
turn-left, turn-right, forward, and attack. The experimental
`compositional_actions_enabled=true` flag keeps those same four neural logits
but samples three independent groups: orientation (none/left/right),
locomotion (none/forward), and interaction (none/attack). Its fixed order is
orientation, simultaneous movement, then interaction; moves may enter cells
vacated in that same movement phase. Consequently a policy can turn, pursue,
and attack in one tick without adding outputs to its genome.

The canonical interaction economy charges `attack_attempt_cost` for every
attack action. A successful adjacent attack then transfers up to
`attack_energy_transfer` from victim to attacker. The transfer is conserved;
the attempt cost is the dissipative component. This makes misses costly and
makes successful hunting capable of paying for unsuccessful attempts.

`pillars` reports raw behavior proxies: action effectiveness, successful attack
rate, and an age/success slope. These values are not fitness scores. Inspect real
organisms before interpreting aggregate movement as behavioral novelty.

## Frozen assays

```bash
$CLI crossplay artifacts/research/runs/RESULT.json.zst \
  --checkpoints 0,5,10,15,20,25,30,35,39 \
  --out artifacts/research/runs/crossplay-heldout.json \
  --horizons 500 --world-seeds 101,131,151,181
```

`crossplay` is the sole longitudinal competence measure. It evaluates distinct
generation-winner genomes in both founder slots and omits clone-versus-clone
comparisons. It is useful for detecting transfer failure and retained strength
against earlier strategies under a common pairwise validation contract. When
training used a shared arena, crossplay is deliberately a simpler transfer
assay rather than a reconstruction of that arena.

## Explicit world-simulator mode

The ordinary simulator remains stateless and world-as-file, but every command
is under `world`:

```bash
$CLI world new --seed 7 --out artifacts/worlds/base.bin
$CLI world run-to 500 --in artifacts/worlds/base.bin
$CLI world pillars --in artifacts/worlds/base.bin
$CLI world top energy 10 --in artifacts/worlds/base.bin
```

`new` creates `<world>.metrics`, an observational sidecar. Copy both files when
forking a world. `pillars`, `eco`, and `timeseries` need the sidecar. The world
state is authoritative; the sidecar never changes decisions or outcomes.

Available simulator commands:

- mutating: `new`, `step`, `run-to`, `watch`;
- performance and grids: `bench`, `sweep`;
- reads: `turn`, `state`, `pillars`, `eco`, `lineage`, `genome`,
  `timeseries`, `inspect`, `top`, `hist`, `find`, `brain`, `decide`, `query`;
- human interactive mode: `tui`.

Use `$CLI world help` for the compact command synopsis. `--in`, `--out`,
`--metrics`, `--no-metrics`, and `--text` are world-mode flags. A mutating
command defaults `--out` to `--in`; read commands never write.

## Model invariants

- A world is a finite clonal evaluator. It contains no reproduction or genome
  mutation.
- The evolution crate owns generations, selection, speciation, crossover, and
  mutation.
- The canonical tick order builds intents from a shared snapshot, applies
  orientation, resolves movement simultaneously, resolves attacks,
  applies optional Hebbian plasticity, and drains one passive energy from each
  survivor.
- With `leaky_neurons_enabled=false`, neuron leak state is disabled, but evolved
  `previous_tick` hidden connections still carry a frozen activation vector
  between decisions. Current-tick connections remain a feed-forward DAG.
  `EnergyFlowLastTick` is an explicit sensor, separate from hidden state.
- Fixed configuration plus seed must reproduce identical results.

Generated worlds and datasets belong under `artifacts/`. Durable hypotheses,
methods, conclusions, and the experiment index belong under `research/`.
