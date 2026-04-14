# Evaluation Guidance

Distilled from ~25 experiments across a single research session. Read before designing or interpreting a run.

## 1. The evaluation horizon is a trap

**Short-horizon evaluations systematically penalize changes that require evolutionary adaptation time.** Any change that adds capacity — new sensors, new plasticity mechanisms, new degrees of freedom in the genome — looks like regression at short horizons because evolution hasn't had time to wire it in. The change's cost (metabolic, informational) shows up immediately; the benefit requires many generations.

We discarded five experiments as failures at 50k ticks. Reran them at 500k ticks. All five beat the baseline, some by >8 points. One of them (more_inputs) turned out to be the best single change we tried.

- **Default to ≥500k ticks** for any experiment that adds structural capacity. Expect 10–15 minutes of compute per run; it's worth it.
- **Shorter runs only for trivial reversible hyperparameter sweeps** where behavior is measurable at convergence within tens of thousands of ticks.
- **A change looking flat or negative at 50k means almost nothing.** Do not discard on that evidence alone.

## 2. Optimize for ceiling, not reliability

Open-ended evolution is a single-trajectory process. Reality ran once; the Cambrian explosion has no confidence interval. The question we're answering is *can this substrate produce something interesting at all*, not *does it do so reliably across independent runs*.

- **`max` across seeds is the primary signal** for "is this substrate capable of reaching a new basin?"
- **`mean` is a secondary signal** useful for sanity-checking that most runs aren't collapsing.
- **`min` and `stddev` are weak signals and can actively mislead.** A change that raises the ceiling while lowering the floor is almost always a win for open-endedness.

A concrete example from this session: action_biases had mean 52.4 (middling) but max 67.0 (highest of any experiment). Multitask biomes had mean 47.1 and the *lowest* stddev (2.58) of any working experiment — i.e., it was the most "reliable" change. It was also the *least interesting*, because it converged every seed to similar mediocrity. Low variance is evidence of collapsed exploration, not of robustness.

## 3. Variance is signal

High per-seed variance means different seeds are finding different basins — the substrate is exploring. Low variance means all seeds converge to similar outcomes, which is what happens when evolution can't escape a local optimum.

- **Welcome high stddev** on any experiment whose mean or max improved. It's evidence the change opened up new exploration paths.
- **Be suspicious of low-stddev changes**, especially if mean didn't meaningfully improve. You may have selected for convergence to a narrow basin.
- **Look at per-seed scores, not just aggregates.** A max-67 seed in a mean-52 run is more interesting than a uniform-52 run.

## 4. Single-seed evaluations are garbage

A change can look like a massive win on seed 42 and collapse across the full suite. This happened to us in the first few experiments — 55.3 on seed 42, 27.6 on the full suite. The single-seed signal is near-random noise.

- **Always run the full default seed suite** (currently 8 seeds) before keeping or discarding.
- If compute is a constraint, **prefer shorter horizon across full suite** over longer horizon on one seed.

## 5. Stacking is not additive

Cherry-picking multiple winning changes into one branch does not compound their wins. We combined the four top single experiments; the combination was *worse* than the best individual. Dropping the weakest member (multitask) recovered most of the gap but still did not beat the best individual's mean.

- **Each stack is its own experiment.** Do not assume that A+B will be A's gain plus B's gain.
- **Environmental changes and architectural changes compound badly.** Biome niches plus 4 new sensors plus recurrence plus action biases was too many simultaneous demands for evolution to satisfy in 500k ticks.
- **Brain-only stacks compose more cleanly** than brain + environment stacks.
- **The combination's `max` may still be higher than any individual's `max`** (we saw this). That's a synergy signal worth chasing even when mean regressed.

## 6. Read pillar components, not just pillars

Aggregate pillar scores compress out important detail. The underlying components tell much more specific stories:
- `adult_mi` and `juvenile_mi` (state–action mutual information) show whether the brain is actually conditioning on state.
- `reversal` shows whether lineages can adapt when rules change.
- `entropy` (inside control pillar) is 0 whenever action entropy is above a cutoff; don't mistake that for an uninformative reading — it means the policy is too uniform to register.
- `p_fwd_food` shows whether foraging is directed or random.
- `anti_idle` is almost always ≥0.95 and rarely discriminates.

When a change moves one component dramatically (e.g., antientropy doubled `reversal` from 0.05 to 0.40), that's a real finding *even if* the aggregate didn't move much.

## 7. "Mean up, min down, variance up" is usually fine

This failure pattern recurred in ~half our experiments and we discarded several for it. In retrospect we were wrong on most of them. For open-ended evolution, this pattern means the change helped some seeds reach higher fitness and destabilized others — which is what exploration looks like.

The pattern only signals a real problem when:
- `mean` regression is large (>3 points), *or*
- The worst seeds collapse to the population-dead floor (~22–28 score), *or*
- A specific pillar component crashes (e.g., `adult_mi` from 0.67 → 0.09)

## 8. Identifying genuinely stuck experiments

Not every failure is a "didn't evolve yet" problem. Some substrates are actually broken. Signs an experiment is truly stuck, not just slow:

- **Identical score at 50k and 500k** — hyperneat gave 27.13 at both. Evolution made zero progress with 10× the time.
- **Extremely tight stddev at low mean** — all seeds converging to the same low basin.
- **`foraging_pillar` near zero** (`p_fwd_food` = 0.0) — organisms aren't directed toward food at all.
- **`adult_mi` near zero** — the brain isn't using sensory input meaningfully.

## 9. Commit hygiene

When experiments run in worktrees, their commits must land on the worktree branch, not the main autoresearch branch. We had three agents contaminate main during this session and had to reset. Reviewers of agent work should:

- Check `git branch --show-current` in the worktree before committing.
- After an experiment, confirm `git log autoresearch/apr14` shows only the baseline, not the experiment's commit.
- Keep experiment branches in `.claude/worktrees/*` for later reuse — reruns, longer horizons, stacking.

## 10. Metrics to add or watch

Current eval reports `aggregate_score`, `aggregate_score_median/min/max/stddev`, and the five pillar components. For future work, add or watch:

- **Per-seed peak-lineage fitness over time** — did any lineage briefly reach a high score before being eclipsed? This is the open-ended signal we actually want.
- **`max_ever`** across the run, not just terminal score.
- **Cross-seed strategy diversity** — are the surviving policies finding the same basin or different ones?
- **Compute-adjusted comparisons** — experiments with denser brains/populations run slower; normalize when comparing across very different substrates.

## Short list to internalize

1. Run ≥500k ticks or don't trust the result.
2. Full seed suite, always.
3. `max` > `mean` > `stddev` in importance for open-ended evolution.
4. High stddev is exploration, not unreliability.
5. Stacks need their own evaluations; don't assume additivity.
6. A change that moves one pillar component dramatically is a real finding.
7. "Mean up, min down" is usually fine; "mean and max down together" is the real discard signal.
