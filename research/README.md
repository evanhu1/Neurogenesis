# NeuroGenesis research workspace

This directory is the tracked source of truth for hypotheses, experimental
contracts, conclusions, and decision history. Generated results and logs live
under `artifacts/research/` and are intentionally ignored by Git.

## Layout

- [`INDEX.md`](INDEX.md): decision ledger and experiment index.
- [`BRIEF.md`](BRIEF.md): current research objective and exact success gate.
- `proposed/`: preregistered experiments that have not run.
- `archive/experiments/`: completed and historical experiment records.
- `archive/reports/`: historical designs and audits.
- `templates/experiment.md`: experiment record template.

Historical ecology, survival, arena, and crossplay records remain archived as
decision history. They do not describe the current evaluator.

## Artifact layout

Use the tracked experiment slug for its generated output:

```text
artifacts/research/runs/
  active/<experiment-slug>/
  completed/<experiment-slug>/
  diagnostics/<diagnostic-slug>/
```

## Current lifecycle

```text
preregister -> plan -> run -> analyze -> conclude/archive
```

1. **Preregister.** State the task contract, search parameters, evolutionary
   seeds, compute budget, and success rule in `proposed/`.
2. **Plan.** Run `cli ecology <task> plan` with the exact arguments. Verify population,
   generations, panels, ticks per evaluation, total genome evaluations, work,
   and seed-genome configuration.
3. **Run.** Execute each evolutionary seed with the release CLI and write
   results to `artifacts/research/runs/active/<slug>/`. Preserve commands,
   source identity, progress logs, timing, and artifacts.
4. **Analyze.** Run `cli ecology analyze` on the result files. Results are
   comparable only when the complete task, agent, ecology, search, and panel
   contracts match. Inspect sealed primary and lesion controls, not only
   training resources.
5. **Conclude and archive.** Apply the preregistered rule, record uncertainty,
   move the record to `archive/experiments/`, move artifacts to `completed/`,
   and update `INDEX.md`.

## Active task-library contract

- Shared input/output alphabet: `a` through `z`, `space`, and `end`; each task
  enables only its declared subset.
- Tasks expose observations, rewards, success events, semantic trial boundaries,
  and termination only. They never inspect or modify brain state.
- The common adapter owns panel sizes, deterministic sampling, learning,
  boundary reset policy, lesion controls, and metrics.
- The common ecology owns finite reproduction and all search mechanics.
- Every genome receives the same deterministic generation panel. Development
  and sealed panels never allocate reproduction.

## Efficiency and integrity

- Use release builds for real runs.
- Smoke-test the exact plan and determinism before a multi-seed suite.
- Change one causal lever per matched experiment.
- Treat any inspected development or sealed cohort as discovery evidence and
  rotate it before a claimed confirmation.
- Persist periodic population checkpoints rather than rerunning evolution for
  structural analysis.
- Keep negative results indexed.
- Use `config/seed_genome.toml` as the canonical founder configuration.

Do not create mirrored baseline TOMLs. Generated data never substitutes for a
tracked hypothesis, method, or conclusion.
