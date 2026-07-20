# Current objective: general learning in symbolic task ecologies

## Architecture bar

A task may define only its observable environment, legal actions, reward,
atomic success events, semantic trial boundaries, deterministic instances, and
metrics. It may not install the representation or memory strategy needed to
solve itself, inspect genomes or neurons, invoke learning, allocate offspring,
or select and mutate parents.

A brain change is admissible only when the canonical substrate cannot express
the required behavior, evolution demonstrably cannot find it, or the existing
route is cripplingly inefficient for a general reason. Search and learner
interventions must remain general across tasks.

## Clean baseline

All active symbolic environments live in `task-library` and implement one
brain- and optimizer-independent contract. `evolution::TaskEcology` is the sole
adapter: it owns genome expression, agent state, action sampling, learning,
evaluation panels, controls, and conversion of task success events into
reproductive tickets. The asexual ecology owns equal-panel evaluation, a finite
population-sized offspring pool, exact elite retention, fixed-K tournament
selection, bounded reversible mutation, audits, and artifacts. Reproduction
never occurs inside evaluation.

There is no scalar fitness, speciation, crossover, target-species quota,
task-authored representation, topology reward, novelty reward, or implicit
efficiency metric. Competition for finite offspring slots supplies relative
pressure. A zero-resource generation is extinct.

The task library currently contains:

- **Basic reaction:** copy the observed `a`-`d`/`end` symbol; each correct
  reaction is one success event.
- **Basic memory:** infer a hidden `a`-`h` sequence over repeated attempts with
  zero symbolic input, then solve it in a frozen greedy probe. Learning attempts
  emit no reproductive events; each correct final-probe position is one
  symmetric success event. The task supplies neither a clock nor a memory
  representation.
- **Basic next-token prediction:** teacher-force the complete fixed English
  snippet from a boundary token and predict the next character at every prefix
  position. Four complete supervised passes retain learned weights while
  resetting recurrent dynamics at pass boundaries. A final reset begins a
  plasticity-frozen greedy probe; each correct probe prediction is a success
  event.
- **Basic continual learning:** track a hidden rewarded `a`-`h` action through
  deterministic 32--96 tick reversals during one uninterrupted lifetime; every
  correct action is one success event and the task never resets agent state.
- **Renewable resource:** infer a hidden `a`-`h` target in a continuous
  lifetime; correct actions consume renewable stock and emit success events.

## Established evidence

The former basic-reaction benchmark saturated at exact `544/544` training and
unseen holdout copying. The former basic-memory benchmark reached 92.822%
sealed exact accuracy at length four in its calibrated legacy evaluator, but
transfer collapsed at longer lengths; it established bounded learnability, not
a general sequence learner. Those outcomes remain historical evidence. The
old task-specific evaluators and NEAT/speciation/CMA-ES paths have been removed;
reaction and memory now remain only as environments loadable by the common
ecology. As a contract-parity check, the historical solved memory champion
reaches 97.253% sealed character accuracy and 89.258% sealed exact accuracy
through the new common adapter on a fresh deterministic panel. A population-256
run of the clean asexual ecology then reached 92.969% sealed character accuracy
and 73.584% sealed exact accuracy by generation 250, re-establishing the memory
character-accuracy gate without the legacy optimizer. With the 100-target
default, population 1,024 reached 98.0% sealed character accuracy and 92.0%
sealed exact accuracy by generation 250. At a fixed 100 generations, training
exact accuracy increased from 13.5% at population 256 to 73.0% at population
8,192, but sealed exact accuracy peaked at 62.0% for population 4,096 and fell
to 59.5% at 8,192. Breadth therefore scales search on the fixed training panel;
the small fixed panel, not raw search capacity, limits monotonic generalization.

The clean basic-continual-learning environment reached 92.984% sealed action
accuracy at population 256 by generation 100, versus 12.112% with plasticity
disabled. Its selected brain has 10 hidden nodes, 41 enabled edges, and no
orphan nodes. This re-establishes continual reversal competence through the
common task adapter and ecology.

The corrected next-token learner trains on all 44 targets of the fixed pangram
for four passes, then runs a frozen greedy probe. Extending the generic
plastic readout from the complete sensory-plus-hidden representation to every
legal output removed a crippling representation bottleneck. At seed 101,
population 256 reached 95.455% sealed probe accuracy by generation 500, and
population 1,024 reached exact 44/44 by generation 374. The exact winner fell
to 2/44 with plasticity disabled, 43/44 without action efference copy, and
41/44 without prediction-error feedback. This establishes rapid mastery of the
training snippet, not held-out language generalization; evolution repeatedly
sees the same snippet and may specialize its inherited recurrent features.
The complete post-hoc discovery record is [archived here](archive/experiments/2026-07-19-basic-next-token-learning-milestone.md).

The renewable environment previously established that the generic stable-plus-
fast plasticity substrate and general lifetime-learning readout can exceed 93%
sealed action accuracy in population-256, 250-generation runs. The clean
task-library cutover must now preserve reaction and renewable competence while
measuring whether symmetric final-probe resources can drive discovery of a
strong bounded learner.

## Success criterion

The next baseline is established when the same task adapter, search process,
and brain machinery achieve at least 90% sealed primary accuracy on reaction,
memory, basic next-token prediction, basic continual learning, and renewable
environments without task-specific evaluator or learner code, while breadth
scales in expectation with population and training depth scales without
systematic regression with generations.
