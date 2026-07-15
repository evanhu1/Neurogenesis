# 2026-07-14-survival-objective-causal-ablation

Status: proposed

## Question

Does removing the relative-opponent multiplier from NEAT fitness improve late
retention and population competence under the simplified, costly predation
model?

## Hypothesis

The current objective multiplies absolute lineage survival by relative survival
advantage. That can reward a genome for damaging opponents even when its own
attack-energy balance is negative. Selecting on absolute survival alone should
reject privately losing interference while retaining predation that genuinely
extends the focal lineage's lifetime.

The prior retrospective reranking audit found a 0.969 within-generation rank
correlation between these objectives and no improvement in late progression.
This experiment is therefore a causal simplification test, not a high-confidence
claim that the objective alone caused stagnation. The new attack-attempt cost
changes the payoff structure enough that a matched evolutionary run remains
informative.

## Arms

- Control: `survival_times_relative_advantage`.
- Treatment: `survival_fraction`.

No other evaluator, ecology, mutation, crossover, or selection setting differs.

## Contract

- Canonical config: `config/world.toml` after the costly-predation cutover.
- Attack semantics: one successful-hit transfer amount plus an unconditional
  attack-attempt cost, including missed attacks.
- Evolutionary seeds: `7,17,27`.
- Training world seeds: `11,29,47`.
- Population: 48 genomes.
- Generations: 40 (`0..39`).
- Episode horizon: 500 ticks.
- Evaluation: contemporary-only triads, 12 memberships per genome, 24 opponent
  exposures, and 36 scored cases per genome.
- World: canonical 50x50 with 102 founders, 34 per lineage.
- Artifact root:
  `artifacts/research/runs/active/2026-07-14-survival-objective-causal-ablation/`.

The 500-tick horizon is valid only while end-survival remains effectively zero.
If either arm reaches material censoring, stop and rerun both arms at the same
longer horizon.

## Measurements

Primary:

- Population mean and median absolute survival over generations 20..39.
- Final versus historical frozen-checkpoint retention.
- Late chronological strength slope and later-over-earlier win fraction.
- Cross-seed consistency of the treatment-minus-control effect.

Mechanistic:

- Plant energy acquired.
- Attack energy transferred, attack-attempt energy spent, and net private
  attack balance.
- Attack precision, no-target attempts, kills, and hits per kill.
- Continuous plant/prey intake fractions and spatial coverage.
- Per-opponent score dispersion.

Integrity:

- Identical memberships, world seeds, cases, horizons, and worker-independent
  deterministic outputs across arms.
- End-survival/censoring audit.
- Population distributions, not champion-only comparisons.
- Direct inspection of any strategy whose relative and absolute rankings differ.

## Decision rule

Prefer absolute survival as the default objective if it is no worse in final
absolute competence and produces a more positive late retention trajectory in
at least two of three evolutionary seeds, without merely eliminating useful
predation or increasing population collapse.

If the arms are behaviorally and competitively indistinguishable, prefer
absolute survival because it is the simpler objective, but classify the result
as simplification rather than improved evolution. Reject the hypothesis if the
treatment worsens retention or competence, or if any apparent benefit is driven
by a single seed.

## Commands and provenance

To be filled from the resolved `cli batch` manifests after implementation.

## Result

Not run.

## Interpretation and next decision

Not run.
