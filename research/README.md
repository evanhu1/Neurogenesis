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
   verify the resolved population, horizon, lineages, memberships, opponent
   exposures, scored cases, evaluator worlds, world ticks, worker count, world
   size, founders, seeds, scenarios, and objective. A nontrivial run must not
   proceed from an invalid or unreviewed plan.
3. **Batch.** Run the evolutionary-seed suite with `cli batch`, writing to
   `artifacts/research/runs/active/<slug>/`. Never write a research run directly
   to `artifacts/runs/` or the research root. Preserve the exact command,
   resolved config, executable commit and checksum, dirty-source identity,
   result schema, progress logs, timings, and artifact checksums in the batch
   manifest.
4. **Summarize training.** Run `cli summarize <experiment> --tail START:END`
   before interpreting the run. The explicit inclusive tail interval must come
   from the preregistered question rather than a post-hoc favorable window.
   Validate generation completeness and contract equality, then examine
   champion, population mean, and population median competence; tail slopes;
   end-survival censoring; behavior; opponent sensitivity; and energy flow.
   `summarize` describes the evolutionary batch only; it does not consume or
   summarize crossplay output.
5. **Crossplay validation.** Run `cli crossplay` over preregistered frozen
   checkpoints on held-out world seeds. Use it to measure chronological
   progress, historical retention, cycles, strategy forgetting, and transfer to
   unseen layouts. Validation world-seed count must satisfy the CLI contract,
   and validation horizons must extend beyond the training horizon when the
   training summary finds censoring. Crossplay is always a two-lineage transfer
   assay, even when training used triads; report it as pairwise validation, not
   native three-lineage competence. Crossplay writes its own JSON and is
   interpreted alongside, not through, `summary.json`.
6. **Inspect and analyze.** Combine training and crossplay evidence, then query
   complete population checkpoints and inspect representative organisms behind
   important gains, regressions, role changes, or suspicious metrics. Explicitly
   check seed robustness, tail behavior, evaluation noise, energy accounting,
   horizon censoring, metric gaming, population distributions, and whether
   apparent novelty is adaptive rather than drift. Use `cli analyze` or
   `cli evaluate-panel` for targeted diagnostics when the question requires
   them; do not add assays merely because they are available.
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
  only to test a tail claim.
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

## Evolution energy diagnostics

NEAT result schema 22 separates external acquisition, conserved predation
transfer, and dissipative attack-attempt cost. Per-case fields and their
`Evaluation` aggregates use these exact definitions:

- Gross energy acquired = plant energy acquired + attack energy received. It
  excludes founder starting energy and does not count attack-backed prey
  consumption a second time.
- Net attack energy balance = attack energy received - attack energy lost -
  attack-attempt energy cost.
- Total energy accumulated = plant energy acquired + attack energy received.
- Net energy profit = total energy accumulated - attack energy lost -
  attack-attempt energy cost. It excludes passive metabolism so that it measures
  ecological profitability; reciprocal transfers cancel.
- Attack-attempt energy cost is paid for every emitted attack, including misses,
  blocked same-lineage attacks, and insufficient-energy attempts.
- Distinct attack victims are unique organism IDs successfully hit by the focal
  lineage within a case.
- Repeat-hit fraction is the number of successful hits after the first hit by
  the same attacker-victim pair divided by all successful focal hits. It is
  undefined (`null`) when there are no hits; aggregate evaluations pool hits
  across cases before taking the ratio.

These fields are observational only. Use gross attack flow to measure activity,
net attack balance to measure private predatory advantage, and plant energy to
measure new energy entering the organism population. Successful attack transfer
is conserved between victim and attacker; attempt cost is the explicit combat
dissipation.
