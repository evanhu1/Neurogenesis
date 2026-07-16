# NeuroGenesis research workspace

This directory is the git-tracked source of truth for the research program.
It contains the questions, experimental contracts, conclusions, and decision
history. Generated worlds, Parquet datasets, logs, and rendered outputs do not
belong here; they live under `artifacts/research/` and are intentionally ignored
by Git.

## Layout

- [`INDEX.md`](INDEX.md): concise ledger of completed, blocked, aborted, and
  proposed experiments.
- [`BRIEF.md`](BRIEF.md): standing research objective and success criteria.
- `proposed/`: preregistered experiments that have not run yet.
- `archive/experiments/`: completed experiment records and the historical log.
- `archive/reports/`: conceptual designs and adversarial audits that informed
  experiments but are not themselves positive experimental results.
- `archive/atlas.html`: historical visual report.
- `templates/experiment.md`: required experiment record format.

## Artifact layout

Machine outputs use the same stable experiment slug as their tracked record:

```text
artifacts/research/
  runs/
    active/<experiment-slug>/
    completed/<experiment-slug>/
    diagnostics/<diagnostic-slug>/
  visualizations/
```

`active` is for a run in progress. A finished or deliberately aborted run moves
to `completed`; compile checks, determinism checks, and throwaway calibration
runs go to `diagnostics`. An experiment record must not rely on an artifact as
the only copy of its hypothesis, method, or conclusion.

## Experiment lifecycle

The canonical research sequence is:

```text
preregister -> plan -> batch -> summarize -> crossplay -> inspect/analyze -> conclude/archive
```

1. **Preregister.** Copy `templates/experiment.md` into
   `proposed/<YYYY-MM-DD-slug>.md`. State one causal hypothesis, the control and
   treatment, fixed parameters, training seeds, held-out validation seeds,
   crossplay checkpoints and horizons, metrics, compute budget, and decision
   rule before running. Add the proposal to `INDEX.md` with status `proposed`.
2. **Plan.** Run `cli plan` with the exact intended NEAT arguments. Record and
   verify the resolved population, horizon, lineages, opponents, scored cases,
   evaluator worlds, world ticks, worker count, world size, founders, seeds,
   and contextual-score aggregation. A
   nontrivial run must not proceed from an invalid or unreviewed plan.
3. **Batch.** Run the evolutionary-seed suite with `cli batch`, writing to
   `artifacts/research/runs/active/<slug>/`. Never write a research run directly
   to `artifacts/runs/` or the research root. Preserve the exact command,
   resolved config, executable commit and checksum, dirty-source identity,
   result schema, progress logs, timings, and artifact checksums in the batch
   manifest.
4. **Summarize training.** Run `cli summarize <experiment>` before interpreting
   the run. Validate generation completeness and contract equality, then inspect
   each generation winner in its own contemporary context, along with behavior,
   opponent sensitivity, energy flow, population structure, and run timing.
   Training scores rank contemporaries only. Do not compute population-mean
   competence, all-time champions, score slopes, or deltas across generations.
5. **Crossplay validation.** Run `cli crossplay` over preregistered frozen
   checkpoints on held-out world seeds. This is the sole longitudinal
   competence assay. Use it to measure chronological
   progress, historical retention, cycles, strategy forgetting, and transfer to
   unseen layouts. Validation world-seed count must satisfy the CLI contract,
   and validation horizons must extend beyond the training horizon when the
   training summary finds censoring. Crossplay uses one common pairwise
   validation contract; for shared-arena training it is a transfer assay, not a
   reconstruction of the training arena. It writes its own JSON and is
   interpreted alongside, not through, `summary.json`.
6. **Inspect and analyze.** Combine contextual training observations with
   crossplay evidence, then query periodic complete population checkpoints and
   inspect representative organisms behind crossplay gains, regressions,
   cycles, or suspicious metrics. Explicitly check seed robustness, evaluation
   noise, energy accounting, horizon censoring, metric gaming, population
   structure, and whether apparent novelty is adaptive rather than drift.
   `cli analyze` may report structural history and contextual snapshots, but it
   must not substitute training-score trajectories for crossplay.
7. **Conclude and archive.** Apply the preregistered decision rule. Fill in the
   result, interpretation, exact remaining uncertainty, and next causal
   experiment in the same record. Move it from `proposed/` to
   `archive/experiments/`, move outputs from `active/` to `completed/`, and
   update `INDEX.md`. Negative and censored results remain indexed and must not
   be promoted to baselines.

## Naming and status

Use a date plus descriptive kebab-case slug. The slug never changes when the
experiment moves from proposed to archived.

Allowed statuses are `proposed`, `running`, `completed`, `blocked`, and
`aborted`. `Blocked` means the mechanism failed a declared gate and should not
be reopened without a materially new mechanism. `Aborted` means the run did
not produce interpretable evidence.

## Efficiency rules

- Change one causal lever per matched experiment.
- Use the smallest horizon that preserves the behavior being selected; extend
  validation horizons only for a preregistered durability question.
- Match simulator cases or report the compute difference explicitly.
- Use release binaries and at most the efficient worker count of the machine.
- Smoke-test contracts and determinism under `runs/diagnostics/` before paying
  for the full seed suite.
- Persist population checkpoints and per-opponent evaluations once; derive new
  reports post-hoc instead of rerunning simulations.
- Negative results remain indexed. Repeating a rejected mechanism without a
  new causal reason is not a new experiment.

The canonical world configuration is `config/world.toml`; experiment
records list only deliberate overrides from it.

## Evolution energy and motor diagnostics

The result schema separates conserved attack transfer from dissipative attack
cost. Per-case fields and their `Evaluation` aggregates use these definitions:

- Gross energy acquired = attack energy received. It excludes founder starting
  energy.
- Net attack energy balance = attack energy received - attack energy lost -
  attack-attempt energy cost.
- Net energy profit is the same private balance. It excludes passive metabolism;
  reciprocal transfers cancel while both attempt costs remain negative.
- Attack-attempt energy cost is paid for every emitted attack, including misses,
  blocked same-lineage attacks, and insufficient-energy attempts.
- Distinct attack victims are unique organism IDs successfully hit by the focal
  lineage within a case.
- Repeat-hit fraction is the number of successful hits after the first hit by
  the same attacker-victim pair divided by all successful focal hits. It is
  undefined (`null`) when there are no hits; aggregate evaluations pool hits
  across cases before taking the ratio.
- Commands per tick is the mean number of explicit orientation, locomotion, and
  interaction commands emitted per organism tick. It is at most one under the
  categorical controller and can exceed one under compositional control.
- Multi-command tick fraction is the fraction of organism ticks that emitted
  at least two commands.
- Target-evaded attacks had an organism in the intended forward cell at the
  shared snapshot, but no attackable organism in the resolved forward cell
  after simultaneous movement.

These fields are observational only. Use gross attack flow to measure activity
and net attack balance to measure private competitive advantage. Successful
attack transfer is conserved between victim and attacker; attempt cost is the
explicit combat dissipation.
