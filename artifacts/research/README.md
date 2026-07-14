# Generated research artifacts

This directory contains untracked machine outputs. The git-tracked research
ledger, proposals, and conclusions live in [`../../research/`](../../research/).

```text
runs/
  active/       experiments currently running
  completed/    completed and deliberately aborted experiment outputs
  diagnostics/  smokes, calibration, determinism, and integration checks
visualizations/ generated screenshots and superseded rendered reports
```

Use the stable experiment slug from `research/INDEX.md` as the run directory.
Each run should contain its exact commands, resolved configuration, Git
revision, result schema, stdout/stderr, and checksums. Raw worlds, metric
sidecars, Parquet data, and JSON results remain ignored by Git.

Other generated-output roots have narrower roles:

- `artifacts/evaluation/`: ordinary `sim-evaluation` datasets and reports;
- `artifacts/runs/`: ad-hoc `sim-cli` output that is not a registered research
  experiment.
