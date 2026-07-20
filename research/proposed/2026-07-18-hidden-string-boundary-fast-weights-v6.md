# Hidden-string boundary fast weights v6: robust confirmation

Status: completed and rejected after the first preregistered confirmation seed
made the all-seed competence gate impossible and failed the boundary lesion

## Question

Does a single symbolic episode-boundary pulse plus an inherited recurrent
temporal basis make reward-only fast-weight learning a reliable exact-string
capability, rather than marginal character adaptation?

## Algorithmic change

- The controller receives `end` only at target position zero. It never senses a
  target symbol, target ID, position index, panel ID, or rollout seed.
- Eight inherited hidden nodes begin as a previous-tick delay line. The boundary
  pulse activates the first node; continuous recurrent evaluation advances the
  activation through internally generated slot states. The founder is
  canonical, but its basis weights, biases, and time constants remain evolvable
  after initialization.
- Every temporal-basis node projects to all eight action symbols. The inherited
  initial learning rate and each readout's nonnegative plasticity coefficient
  evolve. Immediate signed reward updates runtime readout weights.
- Correct actions receive `+1`; incorrect actions receive `-1/7`.
- Selection fitness is the mean product of all four correct-action
  probabilities at the final frozen probe. Marginal target probability is
  diagnostic only; hard greedy exact-string rate is the competence metric.
- Action draws are common random numbers keyed only by rollout seed, attempt,
  and position. Hidden target identity affects learning only through reward.
- The task uses confirmation panel C1 (`0x50414e454c5f4331`). Its training,
  development, and sealed panels are disjoint, position-balanced, and
  repeat-composition matched. The sealed panel is constructed from the explicit
  complement of every target in discovery contract `0x50414e454c5f5636`, and
  the planner must report zero excluded-contract overlap.
- Confirmation uses fresh training/development/sealed action-draw seeds ending
  in `C1`/`C2`; diagnostic and determinism runs must pass `--contract-seed` and
  may not consume these defaults.
- Fresh runs persist the Git commit, tracked patch, untracked source snapshots,
  source/binary/contract hashes, and append-only identity for each execution
  session. Final sealed evaluation persists 32 evenly spaced targets under both
  rollouts for treatment and all controls.

The NEAT loop gains one post-randomization founder-finalization hook. It keeps
the inherited delay-line biases and transition weights canonical while leaving
the readout weights, plasticity coefficients, action biases, and learning rate
randomized and evolvable.

## Diagnostic evidence that justifies confirmation

Seeds 211, 307, and 401 ran for 25 generations at population 32. Sealed hard
exact-string rates were 40.48%, 34.86%, and 33.45%; all crossed 20% training
exact after 320 to 576 population evaluations. Plasticity-off and shuffled
reward were at most 0.098% exact, and reset-weights was at most 0.146%.
Boundary-pulse-off reached 6.54% for seed 211. These artifacts lack complete
source provenance, the per-symbol dynamics lesion, position-permuted reward,
and current behavior traces.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v6-boundary-25g/`

This is a discovery diagnostic, not the confirmation result. Its panel and
evolutionary seeds are excluded from confirmation.

## Preregistered confirmation

Run fresh seeds `509`, `601`, `701`, `809`, and `907` independently with:

- population 64;
- 100 generations;
- eight evaluation workers per seed;
- 1,024 training targets and two rollouts;
- development every 25 generations;
- 1,024 sealed targets and two rollouts, invoked once for the terminal winner;
- probes at attempts 0, 8, 16, and 32;
- checkpoints every 10 generations.

Commands:

```bash
cargo build -p cli --release
./target/release/cli hidden-string plan \
  --seed 509 --population 64 --generations 100 --workers 8
./target/release/cli hidden-string \
  --seed 509 --population 64 --generations 100 --workers 8 \
  --out-dir artifacts/research/runs/active/hidden-string-boundary-fast-weights-v6
```

The other four runs differ only in `--seed`. Execute them sequentially for
machine-specific timing; parallel execution is allowed only when wall time is
reported as load-contaminated and is not used for the efficiency gate.

## Success gate

The route passes only if all of the following hold:

1. At least four of five seeds reach at least 90% sealed hard exact-string rate,
   the median is at least 90%, and no seed is below 75%.
2. Every seed gains at least 74 percentage points over its pre-learning sealed
   exact rate.
3. Plasticity-off, symbol-permuted-reward, position-permuted-reward,
   reset-weights-each-attempt, and dynamics-reset-each-symbol controls each stay
   at or below 0.5% sealed exact. Character accuracy remains diagnostic because
   a position-free learner can exploit the task's repeat composition without
   producing an exact sequence.
4. Boundary-pulse removal also stays at or below 0.5% exact. Otherwise the
   boundary-driven claim fails even if another temporal mechanism survives.
5. For every seed, `max(train, development, sealed) - min(...) <= 0.10` on the
   attempt-32 hard exact rate.
6. Each condition persists 32 evenly spaced sealed targets under both rollouts.
   Trace audit must recompute every signed reward from target and sampled guess,
   verify action probabilities sum to one, verify plasticity-off applies zero
   weight delta, and show fixed greedy probabilities changing after feedback.
   Traces are an integrity assay, not a second small-sample competence gate.
7. A worker-count replay is semantically identical after removing declared
   worker count, paths, and wall-time metadata.
8. Each run completes within 6,400 population evaluations, 6,405 total genome
   evaluations, and 200 billion population synapse operations. Five minutes on
   an otherwise idle current machine is a secondary machine-specific gate.

Passing this gate establishes robust reward-only sequence binding. It unlocks
the delayed-memory task; it does not by itself establish continual learning,
English prediction, multistep reasoning, or open-endedness.

Current-machinery evidence is recorded in the
[smoke and determinism audit](../archive/experiments/2026-07-18-hidden-string-v6-current-machinery-smoke.md).

## Outcome

Seed 509 completed all 100 generations at 49.90% sealed hard exact and 84.80%
character accuracy. Plasticity-off, both reward permutations, weight reset, and
per-symbol dynamics reset stayed at or below 0.098% exact, but boundary-off
reached 5.52%. Because the contract required no seed below 75%, the remaining
four confirmation seeds could not make the route pass and were not run. See the
[confirmation record](../archive/experiments/2026-07-18-hidden-string-boundary-fast-weights-v6-confirmation.md).
# Architecture-audit status

Rejected on 2026-07-18: the proposed temporal basis supplied positional state
that evolution was meant to discover.
