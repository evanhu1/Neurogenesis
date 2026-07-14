# <experiment id>: <short title>

Status: proposed

## Question

State one causal question.

## Hypothesis

State the expected treatment effect and its proposed mechanism. Also state a
plausible result that would falsify the mechanism.

## Contract

- Code revision:
- Canonical config: `config/world.toml`
- Control:
- Treatment:
- Evolutionary seeds:
- Training world seeds:
- Held-out world seeds:
- Population:
- Generations:
- Episode horizon:
- Opponent exposures / lineages / cases per genome:
- Evaluation workers:
- Estimated simulator worlds and wall time:
- Artifact directory: `artifacts/research/runs/active/<experiment-id>/`

List every deliberate override. Everything else inherits the canonical config.

## Measurements

Name the primary endpoint, secondary competence measures, behavioral measures,
tail interval, uncertainty/seed aggregation, and integrity checks. Define every
derived measure used for a decision.

## Decision rule

Write the rule before running. Include adverse outcomes that reject the
treatment even if its headline metric improves.

## Commands and provenance

Record exact commands in the artifact directory together with resolved config,
Git revision, result-schema version, stdout/stderr, and checksums.

## Result

Not run.

## Interpretation and next decision

Not run.
