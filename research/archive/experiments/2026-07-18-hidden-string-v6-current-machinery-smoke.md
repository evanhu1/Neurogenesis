# Hidden-string v6 current-machinery smoke and determinism audit

Status: completed; confirmation machinery cleared for preregistration

## Question

Does the exact current v6 binary preserve worker-count determinism and produce
complete, internally consistent controls, behavior traces, schema, and source
provenance before any confirmation panel is evaluated?

## Method

Two population-2, one-generation runs used evolutionary seed 929 and explicit
diagnostic contract seed 888888. One used one evaluation worker and the other
used four. The diagnostic contract disables the confirmation panel exclusion
because it derives a completely separate panel and rollout domain; it does not
consume confirmation defaults.

Artifacts:

- `artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v6-final-smoke-workers-1/`
- `artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v6-final-smoke-workers-4/`

## Results

- After removing worker count and observational wall-time fields, the two
  complete result JSON documents were byte-identical after canonical JSON
  sorting. This includes populations, genomes, learning metrics, all controls,
  all traces, and deterministic work.
- Each sealed condition persisted 64 traces: 32 evenly spaced targets under
  both rollout seeds, with 32 attempts per trace.
- The built-in trace audit passed treatment and all six controls: zero reward
  mismatches, zero probability-sum violations, zero final-policy mismatches,
  paired target/rollout keys, and zero nonzero plasticity-off deltas. The
  maximum action-probability sum error was `1.49e-7`.
- Result schema is 6. Both manifests use artifact schema 2 and recorded the
  same source fingerprint and release-executable hash. Contract hashes differ
  only because the worker count is part of the resolved contract.
- The planner reports zero training/development/sealed overlap and zero overlap
  between the default confirmation sealed panel and every retired discovery
  target.
- A separate current-schema smoke successfully resumed under an identical
  source fingerprint and appended a second source session. Resume now rejects
  a changed source fingerprint.

The one-generation behavioral scores are mechanics smoke data, not competence
evidence. Primary sealed exact rate was 1.17%; all six control exact rates were
at or below 0.098%. The dynamics-reset control retained 21.88% character
accuracy while remaining near-zero exact, confirming that a no-position learner
can exploit repeat composition at character level. Exact sequence rate, not a
chance character ceiling, is therefore the decisive lesion metric.

## Decision

Clear the current implementation for a frozen confirmation contract. Keep
confirmation target, rollout, and evolutionary seeds untouched until the five
terminal champions are selected. Treat the hand-authored four-slot temporal
basis narrowly: success would establish reward-only sequence binding atop an
evolvable temporal substrate, not general memory or open-ended cognition.
# Architecture-audit note

This machinery audit concerns a task-installed positional basis that was later
removed. Retain it only as historical diagnostic evidence.
