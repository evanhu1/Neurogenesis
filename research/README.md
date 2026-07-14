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

1. Copy `templates/experiment.md` into `proposed/<YYYY-MM-DD-slug>.md`.
2. State one causal hypothesis, the control and treatment, fixed parameters,
   seeds, cost, metrics, and decision rule before running.
3. Add the proposal to `INDEX.md` with status `proposed`.
4. Write outputs to `artifacts/research/runs/active/<slug>/`. Never write a
   research run directly to `artifacts/runs/` or the research root.
5. Preserve the exact commands, resolved config, executable commit, result
   schema, and checksums in the artifact directory.
6. Analyze behavior as well as aggregate metrics. Explicitly check seed
   robustness, tail behavior, evaluation noise, energy accounting, and metric
   gaming.
7. Fill in the result and decision in the same record, move it from `proposed/`
   to `archive/experiments/`, move its outputs from `active/` to `completed/`,
   and update `INDEX.md`.

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
