# Evaluation Guidance

## 1. The evaluation horizon is a trap

**Short-horizon evaluations systematically penalize changes that require
evolutionary adaptation time.** Any change that adds capacity — new sensors, new
plasticity mechanisms, new degrees of freedom in the genome — looks like
regression at short horizons because evolution hasn't had time to wire it in.
The change's cost (metabolic, informational) shows up immediately; the benefit
requires many generations.

- **Default to ≥500k ticks** for any experiment that adds structural capacity.
  Expect 10–15 minutes of compute per run; it's worth it.
- **Shorter runs only for trivial reversible hyperparameter sweeps** where
  behavior is measurable at convergence within tens of thousands of ticks.
- **A change looking flat or negative at 50k means almost nothing.** Do not
  discard on that evidence alone.

## 2. Optimize for ceiling, not reliability

- **`max` across seeds is the primary signal** for "is this substrate capable of
  reaching a new basin?"
- **`mean` is a secondary signal** useful for sanity-checking that most runs
  aren't collapsing.
- **`min` and `stddev` are weak signals and can actively mislead.** A change
  that raises the ceiling while lowering the floor is almost always a win for
  open-endedness.

A concrete example from this session: action_biases had mean 52.4 (middling) but
max 67.0 (highest of any experiment). Multitask biomes had mean 47.1 and the
_lowest_ stddev (2.58) of any working experiment — i.e., it was the most
"reliable" change. It was also the _least interesting_, because it converged
every seed to similar mediocrity. Low variance is evidence of collapsed
exploration, not of robustness.

## 3. Variance is signal

High per-seed variance means different seeds are finding different basins — the
substrate is exploring. Low variance means all seeds converge to similar
outcomes, which is what happens when evolution can't escape a local optimum.

- **Welcome high stddev** on any experiment whose mean or max improved. It's
  evidence the change opened up new exploration paths.
- **Be suspicious of low-stddev changes**, especially if mean didn't
  meaningfully improve. You may have selected for convergence to a narrow basin.
- **Look at per-seed scores, not just aggregates.** A max-67 seed in a mean-52
  run is more interesting than a uniform-52 run.

## 4. Single-seed evaluations are garbage

- **Always run the full default seed suite** (currently 8 seeds) before keeping
  or discarding.
- If compute is a constraint, **prefer shorter horizon across full suite** over
  longer horizon on one seed.

## 5. Read pillar components, not just pillars

Aggregate pillar scores compress out important detail. The underlying components
tell much more specific stories:

- `adult_mi` and `juvenile_mi` (state–action mutual information) show whether
  the brain is actually conditioning on state.
- `reversal` shows whether lineages can adapt when rules change.
- `entropy` (inside control pillar) is 0 whenever action entropy is above a
  cutoff; don't mistake that for an uninformative reading — it means the policy
  is too uniform to register.
- `p_fwd_food` shows whether foraging is directed or random.
- `anti_idle` is almost always ≥0.95 and rarely discriminates.

When a change moves one component dramatically (e.g., antientropy doubled
`reversal` from 0.05 to 0.40), that's a real finding _even if_ the aggregate
didn't move much.

## 6. Commit hygiene

When experiments run in worktrees, their commits must land on the worktree
branch, not the main autoresearch branch. We had three agents contaminate main
during this session and had to reset. Reviewers of agent work should:

- Check `git branch --show-current` in the worktree before committing.
- After an experiment, confirm `git log autoresearch/apr14` shows only the
  baseline, not the experiment's commit.
- Keep experiment branches in `.claude/worktrees/*` for later reuse — reruns,
  longer horizons, stacking.
