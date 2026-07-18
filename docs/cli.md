# `cli` — modular NEAT interface

`cli` is the sole headless research interface. Its current default route runs
the reference symbol-copy training task through the modular NEAT evaluator. The
world-as-file simulator remains available only under the explicit `cli world`
namespace and is not used for evolutionary fitness.

Build the release binary first:

```bash
cargo build -p cli --release
CLI=./target/release/cli
```

JSON is the default output. Invalid arguments fail with a structured error.

## Task contract

The sensor and action layers have the same fixed alphabet:

```text
a  b  c  d  e  f  g  h  end
```

For each stream position:

1. the current input symbol is one-hot encoded on the nine sensory neurons;
2. the recurrent brain advances exactly once;
3. the action neuron with the largest logit emits its symbol;
4. fitness increases by one when the emitted symbol equals the input symbol.

The score is therefore an integer in `0..=N`, where `N` is the total number of
symbols across the training corpus. Ties between equal logits use alphabet
order. Brain state persists within one stream and resets between streams.

Every genome is evaluated independently against the same inputs. The evaluator
contains no other organisms, match scheduler, arena, or relative scoring.

The default benchmark contains 32 reproducible pseudorandom training streams
and 32 separately seeded holdout streams. Each stream has 16 shuffled body
symbols followed by `end` and is constructed to contain `a` through `h` at
least once. Training fitness has a maximum of 544. The generation winner is
also evaluated on the 544 holdout positions, but holdout performance never
participates in selection or breeding.

## Commands

- `$CLI [OPTIONS]` or `$CLI run [OPTIONS]`: execute one NEAT run.
- `$CLI plan [OPTIONS]`: validate and print the exact compute/task contract.
- `$CLI analyze RESULT...`: read result artifacts and report fitness
  trajectories and final outputs.
- `$CLI hidden-string [run|plan] [OPTIONS]`: run or plan the zero-input,
  reward-driven within-lifetime adaptation task.
- `$CLI hidden-string resume CHECKPOINT [--generations N]`: continue from an
  atomic generation-boundary checkpoint.
- `$CLI hidden-string reevaluate FROZEN [OPTIONS]`: reevaluate an immutable
  champion without evolution.
- `$CLI hidden-string analyze RESULT...`: report training, development, sealed,
  probe-trajectory, and anti-cheating control results for that task.
- `$CLI hidden-string calibrate RESULT --generations G,G,G`: compare sampled
  target/rollout contracts with the full evaluator on saved populations.
- `$CLI hidden-string horizon RESULT`: evaluate a frozen winner at
  8/16/32/64/128 learning attempts.
- `$CLI world COMMAND ...`: use the separate world simulator.

Pairwise evaluation, shared-population evaluation, opponent panels, crossplay,
and world materialization are intentionally absent from the NEAT interface.
The task boundary and extension contract are documented in
[`evaluation-tasks.md`](evaluation-tasks.md).

## Planning

```bash
$CLI plan \
  --population 64 \
  --generations 100
```

`plan` validates the same configuration used by execution and reports the
exact training and holdout streams, symbols per genome/winner, genome
evaluations, training and holdout comparisons, and seed-genome configuration.
It performs no evaluations and writes no artifact.

## Run options

- `--seed N`: evolutionary run seed.
- `--population N`: genomes per generation.
- `--generations N`: evaluated generations.
- `--population-checkpoint-interval N`: persist the complete population every
  N generations. The final population is always persisted.
- `--workers N`: parallel genome evaluations. The explicitly sized local pool
  and worker count are persisted in the NEAT contract.
- `--threshold F`: repeatable normalized-fitness threshold for first-crossing
  telemetry. Hidden-string defaults are `0.2`, `0.5`, `0.8`, and `0.9`.
- `--stream SYMBOLS`: replace the default training corpus with explicit cases;
  repeat to add cases. Accepted forms include `abc`, `a,b,c,end`, and
  `a -> b -> c -> end`. `end` is appended when omitted. Only `a`, `b`, `c`,
  `d`, `e`, `f`, `g`, `h`, and terminal `end` are valid. The holdout corpus is
  unchanged.
- `--leaky-neurons`: enable leaky hidden-neuron state during evaluation.
- `--seed-config PATH`: founder-genome TOML; defaults to
  `config/seed_genome.toml`.
- `--out-dir PATH`: result directory.
- `--param key=value`: explicit NEAT parameter override. Run `--help` for the
  current parameter list.

The default task is the canonical 32-stream training corpus plus its separate
32-stream holdout corpus.

## Running

```bash
$CLI \
  --seed 17 \
  --population 64 \
  --generations 100 \
  --out-dir artifacts/research/runs
```

Progress is emitted as JSONL on stderr. Each generation record includes the
winner's training fitness/accuracy, holdout accuracy, one example from each
corpus, species count, and network size. Completion JSON on stdout points to a
compressed `json.zst` result.

The artifact contains:

- the exact training and holdout corpora and NEAT/seed-genome contract;
- every generation winner's genome, training fitness, holdout result, examples,
  and species summary;
- periodic full evaluated-population checkpoints;
- the complete final evaluated population.

Because fitness has a fixed meaning, generation scores are directly comparable
within and across runs that use the same corpus. The separately seeded holdout
is the generalization assay; it is never an evolutionary objective.

## Analyze

```bash
$CLI analyze artifacts/research/runs/neat-symbol-copy-RESULT.json.zst
```

Analysis reports corpus sizes, final training fitness/accuracy, final holdout
fitness/accuracy, and the generation-by-generation training/holdout trajectory.

## Hidden-string adaptation

```bash
$CLI hidden-string plan --seed 101 --population 64 --generations 1000 --workers 4
$CLI hidden-string --seed 101 --population 64 --generations 500 --workers 4 \
  --out-dir artifacts/research/runs/active/hidden-string-greedy-prefix-v4
$CLI hidden-string analyze \
  artifacts/research/runs/active/hidden-string-greedy-prefix-v4/neat-hidden-string-run-*/result.json.zst
```

This task presents no sensory stream. For each hidden four-symbol target drawn
from `a` through `h`, the brain begins from inherited weights and receives
immediate signed reward for each sampled output. Runtime hidden-to-action
weights persist across 32 attempts. Evolutionary fitness is the final frozen
probe's greedy longest-correct-prefix score. A target receives `0/4`, `1/4`,
`2/4`, `3/4`, or `4/4` according to how many consecutive symbols are correct
from position zero before the first error. Correct symbols after the first
error receive no credit. Fitness is the mean prefix score across target/rollout
cases. Hard exact-string rate remains the normalized competence score used by
threshold telemetry; unordered per-symbol accuracy is diagnostic only.
Training uses the calibrated 1,024-target/two-rollout panel and one frozen final
probe. Development uses 256 disjoint targets, runs primary-only every 25
generations plus the final generation, and reports probes at attempts
0/8/16/32. Sealed evaluation uses 1,024 disjoint targets and two rollouts once
for the final evolutionary winner; shuffled-reward and reset-weights controls
run only their final probe. Panels are hash-shuffled, repeat-pattern matched,
and exactly symbol-balanced at every position.

Population genomes are the parallel unit. A single brain trajectory remains
sequential because recurrent history crosses attempts and hidden nodes form a
small current-step DAG. Fixed seed/configuration results are deterministic
across worker counts.

### Durable hidden-string runs

Each invocation creates a unique directory below `--out-dir`:

```text
neat-hidden-string-run-<unix-ms>-<pid>/
  manifest.json
  result.json.zst
  checkpoints/
    generation-000010.checkpoint.json.zst
    latest.json
  champions/
    historical-generation-000003.frozen.json.zst
    terminal.frozen.json.zst
```

Numbered checkpoints and champions are compressed, atomic, standalone JSON
artifacts. A checkpoint with `next_generation: 10` contains the unevaluated
population for generation 10; generations 0 through 9, breeding for generation
10, species representatives, compatibility state, historical champion,
threshold crossings, and cumulative deterministic work are complete.
Population checkpoints are not embedded in lifecycle-v1 generation summaries.

`SIGINT` and `SIGTERM` set a stop request. The current generation finishes, the
continuation checkpoint is written, and the current winner receives normal
final development and sealed evaluation. The result is marked
`early_stopped` with `signal:SIGINT` or `signal:SIGTERM`; no I/O runs inside the
signal handler.

Resume a numbered checkpoint in the same run directory:

```bash
$CLI hidden-string resume RUN/checkpoints/generation-000500.checkpoint.json.zst
```

The stored generation budget remains the default. The only permitted semantic
override is a terminal generation greater than the checkpoint's
`next_generation`:

```bash
$CLI hidden-string resume RUN/checkpoints/generation-001000.checkpoint.json.zst \
  --generations 1500
```

Task, seed-genome, run-seed, and NEAT settings are restored and incompatible
state is rejected. Resume also rejects a checkpoint older than the run's latest
checkpoint, preventing an accidental overwrite of a later continuation.
Continuation is deterministic; timing and paths are observational metadata.

Reevaluate a frozen champion without modifying it:

```bash
$CLI hidden-string reevaluate RUN/champions/terminal.frozen.json.zst \
  --attempts 16,32,64,128 --panel sealed --rollouts 4 \
  --condition primary --condition plasticity-off \
  --condition shuffled-reward --condition reset-weights
```

Named panels reproduce the artifact's training, development, or sealed panel.
A custom composition-matched sample must declare its provenance:

```bash
$CLI hidden-string reevaluate RUN/champions/terminal.frozen.json.zst \
  --attempts 32,64 --panel custom --panel-seed 9001 --targets 256 \
  --rollout-seed 11 --rollout-seed 29
```

Output records the resolved task contract, panel seed/count, rollout seeds,
conditions, greedy-prefix scores, hard exact-string rates, diagnostic character
accuracy, and brain synapse operations. Results also record
hard-exact competence-threshold crossings,
per-generation population/winner/development work, terminal sealed work, and
per-generation, per-session, and total wall time. Winner work is a diagnostic
subset of population work and is not double-counted in totals. See
[`neat-run-lifecycle.md`](neat-run-lifecycle.md) for the complete contract.

## Explicit world simulator

The optional simulator remains stateless and world-as-file:

```bash
$CLI world new --seed 7 --out artifacts/worlds/base.bin
$CLI world run-to 500 --in artifacts/worlds/base.bin
$CLI world brain 0 --in artifacts/worlds/base.bin
```

It shares the symbolic brain representation for inspection but is not invoked
by `evolution::run_neat` and cannot change evolutionary fitness.

Generated results belong under `artifacts/`. Durable research records belong
under `research/`.
