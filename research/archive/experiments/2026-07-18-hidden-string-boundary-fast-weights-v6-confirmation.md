# Hidden-string v6 confirmation: early rejection

Status: completed negative result; the remaining four preregistered seeds were
not run after seed 509 made the all-seed gate impossible

## Question

Can an evolvable recurrent delay line plus reward-modulated output fast weights
learn an unseen four-symbol string reliably enough to pass the preregistered
five-seed confirmation gate?

## Frozen contract

The confirmation used the current `hidden_string_adaptation_v6` contract:

- evolutionary seed 509, population 64, 100 generations, eight workers;
- 1,024 training, 1,024 development, and 1,024 sealed targets with two
  target-independent rollout streams;
- sealed targets drawn from the explicit complement of every target used by the
  earlier discovery contract;
- one `end` pulse at position zero, an evolvable inherited delay line, and
  reward-modulated hidden-to-action runtime weights;
- sequence probability for selection and hard greedy exact-string rate for the
  competence gate;
- six paired causal controls and 64 behavior traces per condition.

The preregistration required at least four of five seeds and the median to
reach 90% sealed exact, no seed below 75%, at least 74 percentage points of
exact adaptation in every seed, and every control at or below 0.5% exact.

Artifact:

`artifacts/research/runs/active/hidden-string-boundary-fast-weights-v6/seed-509/neat-hidden-string-run-1784368844161-170/result.json.zst`

The run captured source fingerprint
`8d03e7d78cf645ed8d514a194f1833ae98355b4ca52e4b47056f4d8f6c78d4e6`,
release executable hash
`73ec0f635863f355a991d3351b72cf2dd40115dd869f180dc14873383dbc1ae1`,
and task-contract hash
`38f4f410f24441ebefebc48632c72f2e8cd9ebaab7479c38a4df1c8df3afe62d`.

## Result

| Cohort | Hard exact | Character accuracy | Sequence probability |
|---|---:|---:|---:|
| Training | 52.39% | 85.14% | 0.3348 |
| Development | 48.83% | 84.67% | 0.3098 |
| Sealed | 49.90% | 84.80% | 0.3037 |

The sealed pre-learning exact rate was 0.098%, so exact adaptation gained 49.80
percentage points. The training/development/sealed spread was only 3.56 points;
the failure is therefore not an ordinary generalization gap. Training crossed
20% exact at generation 7 (512 population evaluations) and 50% at generation
76 (4,928 evaluations). A generation-83 champion reached 61.72% training exact,
but the terminal sequence-probability winner traded some hard exact accuracy
for a slightly higher selection score.

| Sealed condition | Hard exact | Character accuracy |
|---|---:|---:|
| Treatment | 49.90% | 84.80% |
| Plasticity off | 0.098% | 12.50% |
| Symbol-permuted reward | 0.000% | 1.83% |
| Position-permuted reward | 0.049% | 18.60% |
| Reset weights each attempt | 0.098% | 18.86% |
| Dynamics reset each symbol | 0.000% | 42.15% |
| Boundary pulse off | 5.518% | 58.54% |

Every trace audit passed: rewards recomputed from targets and sampled guesses,
probability vectors summed to one, final fixed probes matched aggregate output,
and plasticity-off applied zero runtime weight delta. Each condition stored 32
targets under both rollout seeds. This makes the positive adaptation effect
real, but it also makes the two failed gates real:

1. seed 509 is below the mandatory 75% floor, so the five-seed contract can no
   longer pass regardless of the other four outcomes;
2. boundary-pulse-off is eleven times the 0.5% ceiling, so the claimed
   boundary-driven mechanism is not necessary for the evolved controller.

The run completed 6,400 population evaluations and 6,405 total genome
evaluations in 114.23 seconds. Population evaluation used 123.275 billion
synapse operations and total work used 123.436 billion, inside every efficiency
budget.

## Mechanistic diagnosis

The controller learned useful target-independent temporal states: the four
sealed slot-state cosine similarities ranged from 0.053 to 0.511 rather than
the 0.996--1.0 collapse seen in v5. Reward, persistent runtime weights, and
continuous within-string dynamics were all causally necessary for exact
performance.

However, an output-only fast-weight matrix still superimposes every slot's
updates on shared action logits. Sealed cross-position destructive-interference
rates were 55.0%--59.1%, and 12.29% of requested updates clipped. Removing the
boundary pulse left a much stronger-than-chance autonomous timing policy. The
hand-initialized temporal basis therefore reduced the task to imperfectly
separated contextual bandits; it did not yield a clean, reusable state-update
algorithm.

## Decision

Reject v6. Seeds 601, 701, 809, and 907 were deliberately not launched: once a
preregistered mandatory all-seed gate is impossible, running them cannot turn
the route into a pass and would only consume confirmation data post hoc.

Do not reopen this route for more generations, a larger population, a wider
delay line, or reward-rate tuning. Reopening requires a materially new learning
mechanism that changes how state is written or retrieved, isolates credit
across contexts, and can transfer unchanged to delayed memory, sequence
prediction, and multistep control.
# Architecture-audit note

Invalidated as brain competence on 2026-07-18 because the task installed the
recurrent positional basis. The failed confirmation remains negative evidence.
