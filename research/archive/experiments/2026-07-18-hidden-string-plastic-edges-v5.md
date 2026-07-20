# Hidden-string plastic edges v5, 200-generation diagnostic

Status: completed; exact-string competence rejected

## Question

Can NEAT reliably discover a within-lifetime hidden-string learner when reward
is balanced across eight actions, the initial learning rate and per-edge
plasticity evolve, and selection uses continuous marginal target probability?

## Method

Seed 211 ran for 200 generations at population 64 with eight evaluation
workers. Targets were four symbols over `a` through `h`; the task supplied no
sensory input and used 32 immediate-reward attempts. Correct actions received
`+1` and incorrect actions `-1/7`. The final marginal correct-action
probability was selection fitness; hard greedy exact-string rate remained the
competence metric.

Artifact:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v5-200g/neat-hidden-string-run-1784364169531-59413/result.json.zst`

## Results

- Runtime: 86.8 seconds.
- Population work: 12,800 genome evaluations and 72.56 billion synapse
  operations.
- Training target-probability fitness: 0.3710.
- Sealed target-probability fitness: 0.3695.
- Sealed character accuracy: 40.04%, up from 12.5% before learning.
- Sealed hard exact-string rate: 0%.
- Reset-weights control: 17.68% character accuracy and 0.049% exact.
- Shuffled-reward control: 9.34% character accuracy and 0% exact.

Final per-position accuracies were 40.53%, 45.26%, 40.53%, and 33.84%, but
pairwise hidden-state cosine similarities were 0.996 to 1.0. Cross-position
update vectors were negatively aligned in roughly 51% to 55% of comparisons.
Fitness largely plateaued after generation 49 while exact rate remained zero.

## Adversarial finding

The evaluator also keyed every deterministic action draw by the encoded hidden
target. That target-conditioned RNG violated the strict reward-only contract.
It did not rescue exact behavior, but it invalidates v5 as clean evidence of
reward-only adaptation and must not be retained in later contracts.

## Decision

Block this family as a complete route. Marginal target probability admits a
bag-of-symbol solution and output-only plasticity cannot create distinct
temporal keys. More generations or coefficient tuning is not a material new
mechanism. Retain the diagnostics and evolvable coefficients; replace the
autonomous collapsed clock with an explicit recurrent temporal substrate,
restore sequence-level fitness, use target-independent common random draws,
and rotate to untouched panels.
