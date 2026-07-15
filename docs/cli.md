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
- `$CLI summarize EXPERIMENT --tail START:END`: validate and aggregate a batch.
- `$CLI analyze RESULT...`: derive per-run trajectory diagnostics.
- `$CLI crossplay RESULT ...`: run a pairwise transfer assay over frozen
  generation champions.
- `$CLI evaluate-panel ...`: evaluate explicit frozen focal/opponent genomes.
- `$CLI world COMMAND ...`: use the ordinary world simulator.

## Planning a run

Always plan a nontrivial run before executing it:

```bash
$CLI plan \
  --population 48 \
  --generations 40 \
  --horizon 500 \
  --lineages-per-world 3 \
  --memberships-per-genome 12 \
  --world-seeds 11,29,47 \
  --scenarios baseline \
  --founders 102 \
  --workers 14
```

`plan` loads the real world config and calls the same run-option validation used
by execution. It prints:

- simultaneous lineages and founders per lineage;
- ecosystem memberships per genome;
- opponent exposures per genome;
- cases per membership and scored cases per genome;
- ecosystem groups and evaluator worlds per generation;
- total evaluator worlds and world ticks;
- requested/effective evaluator workers and the machine parallelism;
- the resolved objective, seeds, scenarios, horizon, width, and founder count.

No world is simulated and no artifact is written.

### Evaluation-budget terminology

One **membership** means that one genome appears in one competitive evaluator
group. A group has `--lineages-per-world` genomes, so each membership exposes a
genome to `lineages_per_world - 1` contemporary opponents.

One membership produces:

```text
world seeds × scenarios × horizons
```

scored cases for that genome. The core interface uses one `--horizon`, so with
three seeds and one baseline scenario each membership produces three cases.

Use exactly one budget form:

- `--memberships-per-genome N` directly controls ecosystem participation; or
- `--cases-per-genome N` asks the CLI to derive the exact membership count.

`--cases-per-genome` must divide evenly by the cases per membership. The old
ambiguous `--opponents` interface is intentionally absent.

### Run options

- `--seed N`: evolutionary seed.
- `--population N`: genomes in each generation.
- `--generations N`: evaluated generations.
- `--horizon N`: ticks in every evaluator world; default `500`.
- `--lineages-per-world 2|3`: genomes sharing each evaluator world.
- `--memberships-per-genome N` or `--cases-per-genome N`: evaluation budget.
- `--world-seeds N,N`: deterministic training layouts.
- `--scenarios baseline[,scarcity,sparse_search]`: environmental treatments.
- `--objective survival|survival_times_relative_advantage`: selection score.
- `--cvar F`: mean only the worst-performing fraction of cases into fitness;
  `1.0` is the ordinary mean.
- `--workers N`: parallel evaluator groups.
- `--founders N`: organisms divided equally among lineages.
- `--world-width N`: width of the square axial hex world.
- `--set key=value`: explicit ecology override.
- `--config PATH`: world TOML; defaults to `config/world.toml`.
- `--out-dir PATH`: generated result directory.

Both the genome population and world founder count must be divisible by the
number of lineages. The CLI rejects invalid schedules with an actionable error.
NEAT parallelizes independent evaluator groups and fixes each world's internal
intent evaluation to one thread; use `--workers`, not
`--set intent_parallel_threads=...`.

`--param key=value` remains available for explicit mutation, speciation,
selection, plasticity, and complexification experiments. Evaluation layout,
objective, world size, and founder count have first-class flags and cannot be
smuggled through duplicate generic parameters.

## Running one evolutionary seed

Remove `plan` from a validated command and add the evolutionary seed:

```bash
$CLI \
  --seed 17 \
  --population 48 \
  --generations 40 \
  --horizon 500 \
  --lineages-per-world 3 \
  --memberships-per-genome 12 \
  --world-seeds 11,29,47 \
  --scenarios baseline \
  --founders 102 \
  --workers 14 \
  --out-dir artifacts/research/runs
```

Progress is JSONL on stderr. The completion JSON on stdout points to the result
and materialized champion world. The result persists the resolved world/NEAT
contract, every generation, complete population checkpoints, and champion.

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
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --scenarios baseline --founders 102 \
  --objective survival_times_relative_advantage
```

The machine-wide worker budget is divided deterministically in seed order.
`batch` rejects `--seed`, `--workers`, and `--out-dir` after the `--` separator.
It writes one stable experiment directory containing `manifest.json`, per-seed
results and champion worlds, stdout/progress logs, source revision and dirty
status, and SHA-256 checksums. Existing nonempty experiment directories require
an explicit `--replace`.

Summarize a completed batch with an explicit inclusive tail window:

```bash
$CLI summarize \
  artifacts/research/runs/active/attack-objective-ablation-control \
  --tail 20:39
```

Summarization validates result schemas, generation sequences, seeds, and
resolved contracts. It parses large result files sequentially and retains only
compact trajectories. The explicit tail prevents an arbitrary hidden
definition of "late progress."

The compact trajectory is deliberately capability-oriented:

- survival fitness, absolute alive-ticks, and end-survival censoring;
- action effectiveness, action allocation, plant/prey success rates, plant
  capture, spatial coverage, and opponent sensitivity;
- attack precision, transfer received/lost, and attempt cost;
- total energy accumulated = plant energy + attack-transfer income;
- net energy profit = plant energy + attack-transfer income - attack-transfer
  losses - attack-attempt cost.

Net energy profit excludes unavoidable per-tick metabolism. Reciprocal attack
transfer therefore cancels while its attempt costs remain negative, making the
measure robust to high-throughput energy cycling. Coarse sensory/action mutual
information and the age-success slope remain raw simulator facts; NEAT summaries
intentionally do not disclose them because neither is a capability or learning
measure under the default non-plastic controller.

## Fitness and observation metrics

The available selection objectives are:

- `survival`: absolute founder alive-ticks divided by the theoretical maximum;
- `survival_times_relative_advantage`: absolute survival multiplied by the
  focal lineage's relative share of combined lineage alive-ticks.

Energy-flow and behavior fields are diagnostics, not extra rewards:

- `gross_energy_acquired` is plant energy plus successful attack transfers
  received by focal attackers. It excludes starting energy and attack costs.
- `plant_energy_acquired` is energy from consumed plants.
- `attack_energy_received` is successful transfer into focal attackers.
- `attack_energy_lost` is successful transfer out of focal victims.
- `attack_attempt_energy_cost` is energy paid for every focal attack action,
  including misses and blocked attempts.
- `net_attack_energy_balance` is `received - lost - attempt cost`.
- `total_energy_accumulated` is `plant energy + attack energy received`.
- `net_energy_profit` is `total accumulated - attack energy lost - attempt
  cost`; passive metabolism is deliberately excluded.
- `attack_precision` is successful hits divided by all attack attempts.
- `distinct_attack_victims` counts unique organisms successfully hit.

The canonical attack economy charges `attack_attempt_cost` for every attack
action. A successful adjacent attack then transfers up to
`attack_energy_transfer` from victim to attacker. The transfer is conserved;
the attempt cost is the dissipative component. This makes misses costly and
makes successful hunting capable of paying for unsuccessful attempts.

`pillars` reports raw behavior proxies such as action effectiveness,
plant/prey-consumption rates, coarse sensory/action mutual information, and an
age/success slope. These values are not normalized fitness scores. Inspect real
organisms before interpreting aggregate movement as behavioral novelty.

## Frozen assays

```bash
$CLI analyze artifacts/research/runs/RESULT.json
$CLI crossplay artifacts/research/runs/RESULT.json \
  --checkpoints 0,5,10,15,20,25,30,35,39 \
  --horizons 500 --world-seeds 101,131,151,181
$CLI evaluate-panel --focal RUN_A.json --opponents RUN_B.json \
  --horizons 500 --world-seeds 101,131,151,181
```

`crossplay` is explicitly a **two-lineage transfer assay**, even if its source
run trained with three lineages. It evaluates distinct checkpoint genomes in
both founder slots and omits clone-versus-clone comparisons. It is useful for
detecting dyadic transfer failure and retained strength against earlier
strategies. It does not measure native three-lineage competence, and its output
must not be described as if it recreated the training ecosystem.

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
- reads: `turn`, `state`, `pillars`, `eco`, `lineage`, `genome`, `food`,
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
- The canonical tick order builds intents from a shared snapshot, resolves
  movement, resolves consumption/attacks, applies optional Hebbian plasticity,
  and drains one passive energy from each survivor.
- With `leaky_neurons_enabled=false`, every brain decision is a feed-forward
  pass. `EnergyFlowLastTick` is an explicit sensor, not hidden activation state.
- Fixed configuration plus seed must reproduce identical results.

Generated worlds and datasets belong under `artifacts/`. Durable hypotheses,
methods, conclusions, and the experiment index belong under `research/`.
