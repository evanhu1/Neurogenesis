# NEAT run lifecycle and artifact contract

This document specifies durable execution for long-running modular NEAT
experiments. The evolutionary semantics remain in `evolution`; filesystem,
signal, and command orchestration remain in `cli`.

## Run directory

Each new run creates one unique directory under `--out-dir`:

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

All JSON and compressed JSON writes use `views::atomic_write`. Champion and
numbered checkpoint files are immutable. `manifest.json` and `latest.json` are
small atomic discovery pointers. A run never writes outside its own directory.

## Generation boundary

A checkpoint's `next_generation = N` has one exact meaning:

- generations `0..N` have been fully evaluated, speciated, summarized, and
  included in cumulative deterministic counters;
- the stored population is the unevaluated population that will be evaluated
  as generation `N`;
- the stored compatibility threshold is the threshold to use while assigning
  species in generation `N`;
- stored species representatives and `next_species_id` are the state produced
  by generation `N - 1`;
- breeding for generation `N` has already happened using the deterministic
  generation-domain seed.

No runtime neural state or random-number generator cursor is needed. Genome
evaluation is pure for a fixed task contract, and breeding constructs its RNG
from `(run_seed, generation)`. The checkpoint nevertheless stores the complete
semantic contract, population, species state, accumulated summaries, threshold
events, historical champion metadata, and deterministic work counters.

Configured checkpoints are written after every
`population_checkpoint_interval` completed generations and at the final or
early-stop boundary. They are independently readable and are not hidden inside
the terminal result.

## Stop and resume

The CLI registers signal flags for `SIGINT` and `SIGTERM`. A handler only sets
an atomic value. Evolution checks it after completing the current generation;
it never interrupts evaluation, breeding, or an artifact write.

At a requested stop the CLI writes the continuation checkpoint, evaluates the
current training winner with the normal final development and sealed contracts,
writes historical and terminal frozen-genome artifacts, and writes a valid
result whose termination status is `early_stopped` with the signal reason.

`cli hidden-string resume CHECKPOINT [--generations N]` reads the checkpoint,
validates its schema and task identity, reconstructs the task from its stored
configuration, and continues at `next_generation`. `--generations` is the only
semantic override and must be greater than `next_generation`. Resume rejects an
older-than-latest checkpoint in the run directory rather than overwriting a
later continuation. With the same terminal generation, a resumed run has the
same semantic result as an
uninterrupted run; timing and artifact-path metadata are intentionally outside
that guarantee.

## Frozen genomes and reevaluation

Every strict improvement in historical training fitness is written as a
frozen-genome artifact. The terminal winner is written independently even when
it is also the historical champion. A frozen artifact includes task identity
and configuration, run seed, source generation and population index, fitness,
normalized fitness, topology counts, the training evaluation, and the genome.

`cli hidden-string reevaluate FROZEN` is read-only. It supports multiple
attempt horizons, named training/development/sealed panels or a deterministic
custom panel seed and size, explicit rollout count, and primary,
plasticity-off, shuffled-reward, and reset-weights conditions. Output includes
the complete resolved contract and per-condition greedy-prefix, hard
exact-string, and diagnostic character metrics. Sampling never silently changes
evolutionary training.

## Telemetry

Threshold events record the first generation and cumulative population genome
evaluation count at which the winner's normalized hard exact-string rate
reaches a configured threshold. Evolutionary selection fitness is the final
greedy longest-correct-prefix score and is intentionally separate. Default thresholds are
`0.2, 0.5, 0.8, 0.9`; fresh runs may replace them with repeatable `--threshold`
flags.

Deterministic work telemetry includes per-generation population, development,
and sealed genome-evaluation counts and brain synapse operations, plus
cumulative totals. Wall-clock telemetry is separate: every generation records
elapsed seconds and the result records total and per-session elapsed seconds.
These timing values are observational and do not participate in deterministic
resume equivalence.
