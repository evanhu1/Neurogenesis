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

1. **Preregister.** State the symbol streams, NEAT parameters, evolutionary
   seeds, compute budget, and success rule in `proposed/`.
2. **Plan.** Run `cli plan` with the exact arguments. Verify population,
   generations, training and holdout corpora, symbols per evaluation, total
   genome evaluations, symbol comparisons, and seed-genome configuration.
3. **Run.** Execute each evolutionary seed with the release CLI and write
   results to `artifacts/research/runs/active/<slug>/`. Preserve commands,
   source identity, progress logs, timing, and artifacts.
4. **Analyze.** Run `cli analyze` on the result files. Fitness is an absolute
   count under a fixed task, so trajectories are directly comparable whenever
   corpora match. Inspect unseen holdout emissions, not only training accuracy.
5. **Conclude and archive.** Apply the preregistered rule, record uncertainty,
   move the record to `archive/experiments/`, move artifacts to `completed/`,
   and update `INDEX.md`.

## Current evaluator contract

- Input/output alphabet: `a` through `h`, plus `end`.
- The current hidden-string task has no input stream and uses four-symbol
  targets over `a`–`h`. Training uses the calibrated 1,024-target/two-rollout
  panel; development uses 256 targets every 25 generations; sealed evaluation
  uses 1,024 targets once after final-winner selection.
- Hidden-string training has one frozen final probe. Development reports probes
  at attempts 0/8/16/32 without controls; sealed controls run final-probe only.
- Genome evaluations are parallelized by an explicit worker count. Brain steps
  and recurrent trajectories remain sequential and deterministic.
- One recurrent brain step and one emitted action symbol per input symbol.
- Hidden-string selection fitness is the final mean greedy-prefix score: the
  longest uninterrupted correct prefix divided by four. Hard greedy exact rate
  is the competence/threshold metric; unordered character accuracy is
  diagnostic only.
  The separate symbol-copy reference task retains one-point-per-symbol fitness
  and reports holdout exact matches without using them for selection.
- Independent streams reset brain state; symbols within a stream retain it.
- Every genome is evaluated independently. There are no opponents, pairings,
  arenas, survival scores, or crossplay assays.

## Efficiency and integrity

- Use release builds for real runs.
- Smoke-test the exact plan and determinism before a multi-seed suite.
- Change one causal lever per matched experiment.
- Persist periodic population checkpoints rather than rerunning evolution for
  structural analysis.
- Keep negative results indexed.
- Use `config/seed_genome.toml` as the canonical founder configuration.

Do not create mirrored baseline TOMLs. Generated data never substitutes for a
tracked hypothesis, method, or conclusion.
