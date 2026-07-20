# Renewable symbolic resource ecology

Status: clean v4 baseline implemented and calibrated; competence remains open
Slug: 2026-07-19-renewable-symbolic-resource-ecology
Date: 2026-07-19

## Hypothesis

A symbolic task can serve as an ecology rather than a scalar objective. A
repeatable binary solve event claims finite reproductive resources. Genomes that
solve more quickly or reliably capture more tickets without an optimizer-defined
efficiency metric or evaluator-authored difficulty ladder.

## Physical contract

- The population contains exactly `N` genomes per generation.
- Each genome receives the same fixed, balanced panel of independent physical
  lifetimes (default 64). Brain state is fresh between lifetimes.
- Neural dynamics and runtime synaptic weights persist across every resource
  renewal within that lifetime.
- At least 12 correct actions in a sliding window of 16 decisions solves the
  current resource.
- Reaching the initial hidden target establishes the first reversal but emits
  no ticket. Each subsequent solve emits one reproductive ticket and changes
  the target without resetting the brain.
- Evaluation produces claims only. No birth, death, mating, mutation, or
  replacement occurs until all `N` lifetimes have ended.
- The finite resource is `N` offspring slots. One exact highest-ticket adult is
  retained. Every other slot runs a four-way tournament among ticket-producing
  adults, ranked only by ticket count.
- Each selected parent produces one asexual child. Mutation changes only a
  bounded number of parameters, and matched complexifying/simplifying
  structural events make topology growth reversible.
- The complete offspring population replaces the old population atomically.
  Offspring inherit genes but begin the next lifetime with fresh brain and task
  state.
- Compatibility species remain diagnostic telemetry only. They receive no
  reproductive quota and do not constrain parent sampling or variation.
- If no claim is produced, there is no reproductive distribution: the
  population is extinct and the run terminates.

The task supplies no attempt index, target cue, scheduled reversal, difficulty
ladder, partial credit, efficiency score, novelty reward, topology reward, or
complexity penalty. An easy ecology is allowed to saturate.

## Generic boundary

`ResourceEcologyTask` owns a shared evaluation panel, uninterrupted lifetime
simulation, the binary solve event, and observational audits. It returns task
metrics plus one event for every solve. `run_resource_ecology` owns finite
offspring allocation, tournament selection, mutation, diagnostic speciation,
atomic replacement, extinction, and result persistence.

This is synchronous generational selection, not Moran birth-death. It preserves
the ecological zero-sum pressure while making the causal phases unambiguous:

1. evaluate a fixed population;
2. count solve tickets;
3. allocate finite offspring slots by ticket tournaments;
4. generate all offspring;
5. atomically replace the population.

## Calibration gates

1. Canonical founders must produce a nonzero solve stream under the default
   lifetime and population contract.
2. Fixed seed/configuration must be invariant to worker count.
3. A no-solve generation must terminate as extinction without synthetic
   reproduction.
4. Every child must begin with fresh runtime weights and task state.
5. Every nonterminal generation with claims must produce exactly `N` children.
6. Every solve must emit exactly one ticket, while offspring count remains
   exactly `N`.
7. Development and sealed audits must remain observational and unable to alter
   reproduction.
8. Species count and compatibility threshold changes must not alter reproductive
   allocation except through any explicit mutation-parameter change.
9. A matched experiment must compare this ecology with the retired scalar
   accuracy loop before attributing improved structural discovery to resource
   competition.

## CLI

```bash
cargo build -p cli --release
CLI=./target/release/cli

$CLI continual-reversal plan --seed 101 --population 64 --generations 100
$CLI continual-reversal --seed 101 --population 64 --generations 100 \
  --out-dir artifacts/research/runs/active/renewable-symbolic-resource-v1
```

Scalar selection, survival fraction, crossover, mate choice, and species
offspring allocation are absent. The ecology has one exact elite, fixed-size
ticket tournaments, and reversible bounded mutation. Compatibility species do
not affect reproduction.

## Clean v4 baseline

The initial implementation below was useful diagnostically but had four
optimizer defects: one noisy lifetime per genome, generation/individual-specific
evaluation streams, earliest-claim truncation, and mutation work proportional
to topology size. It also selected the terminal noisy leader and had no exact
retention. V4 replaces those mechanics with a shared 64-lifetime panel, the
12-of-16 post-reversal criterion, one ticket per solve, a fixed `N`-slot
four-way tournament, one exact elite, bounded parameter mutation, reversible
structural mutation, and fixed development/sealed panels.

Matched seed-101 runs establish the intended generation-depth lever. At
population 256, increasing depth from 100 to 250 generations raised sealed
throughput from 4.88 to 7.23 solves per 1,000 ticks and accuracy from 18.23% to
22.57%. The selected topology grew from 5 hidden nodes / 56 enabled edges to
10 / 76. Against the add-only structural control at 250 generations, reversible
mutation reduced topology from 26 / 126 to 10 / 76 while retaining comparable
sealed performance (7.23 versus 8.27 solves per 1,000 ticks).

A three-seed fixed-100-generation sweep tested population breadth. Sealed
throughput, the actual reproductive objective, was:

| Population | Seed 101 | Seed 102 | Seed 103 | Mean |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.87 | 0.00 (extinct) | 4.15 | 2.34 |
| 256 | 4.88 | 4.09 | 5.04 | 4.67 |
| 1,024 | 3.60 | 5.74 | 5.16 | 4.83 |

Population breadth therefore improves the objective in mean and strongly
reduces cold-start extinction, but the marginal gain from 256 to 1,024 is small
and individual seeds are not monotonic. Accuracy is a secondary diagnostic and
was correspondingly noisier. Penultimate-generation telemetry shows that the
four-way tournament retained roughly 43% distinct parents and an effective
offspring-share count near 30% of population at all three sizes, confirming that
selection pressure itself is population-size invariant.

A worker-count determinism probe at population 64 for 10 generations produced
identical semantic artifact hashes with 1 and 14 workers after removing only
worker-count and wall-time fields. Wall time fell from 1.51 seconds to 0.19
seconds. Hardware-derived parallelism is therefore active and
semantics-preserving.

Plasticity remains causal: the 250-generation sealed representative scored
22.57% accuracy and 7.23 solves per 1,000 ticks, while plasticity-off scored
12.47% and zero solves. The clean baseline fixes the search-scaling and
attribution defects; it does not solve continual reversal or approach the 90%
accuracy competence target.

The shared mutator changes were regression-checked against the saturated tasks.
Basic reaction at seed 17, population 64, and 100 generations remained exact at
544/544 on both training and holdout. Basic memory at seed 1, population 64,
and generation 100 improved from the historical 56.8% training / 55.6%
development checkpoint to 77.2% / 75.6%, with 75.3% sealed accuracy.

## Superseded initial calibration

The seed-101, population-256, target-species-8 run completed all 250 generations
with the default 512-tick lifetime and 256 resource tickets. It is stored under
`artifacts/research/runs/diagnostics/resource-generational-pop256-species8-250g/`.

- Total wall time was 1.47 seconds for 64,000 lifetime evaluations and 63,744
  offspring.
- At generation 128 the selected leader had 5 hidden nodes and 63 enabled
  connections; that generation took 3.43 milliseconds.
- The rejected batch-Moran run had reached only generation 128 after 808.10
  seconds, with 438 hidden nodes and 1,236 enabled connections in its selected
  topology.
- The new run's selected topology finished at 17 hidden nodes and 113 enabled
  connections; the selected-topology maxima were 22 and 121.
- Population solve events increased from 87 in generation 0 to 814 in
  generation 249. The 256-ticket pool saturated by generation 20.
- The sealed leader reached 5.34 solves per 1,000 ticks but only 11.27% action
  accuracy, versus 9.92% with plasticity disabled. Chance accuracy is 12.5%.

The algorithmic result is positive: clean phase separation, asexual variation,
and parallel atomic replacement eliminate the crossover-driven topology union
and serial mutation collapse. The task result is negative for continual
learning: competition selected behaviors that obtain correct-action streaks,
apparently dominated by action persistence, without selecting a general
high-accuracy reward learner. The next experiment must treat that behavioral
shortcut as a task-contract question rather than adding optimizer-side shaping.

### 1,000-generation extension

The matched seed-101 run was extended from scratch to 1,000 generations under
the same population-256, target-species-8 contract. It completed in 29.50
seconds, evaluating 256,000 lifetimes and producing 255,744 offspring. The
result is stored under
`artifacts/research/runs/diagnostics/resource-generational-pop256-species8-1000g/`.

Longer search did not recover a general learner. The final sealed primary scored
8.25% accuracy and 4.27 solves per 1,000 ticks; plasticity-off scored 7.90% and
0.67 solves per 1,000 ticks. The best sparse development checkpoint occurred at
generation 524 with 19.78% accuracy and 7.32 solves per 1,000 ticks, but that
behavior was not retained. Population solve events peaked earlier and ended at
705, below generation 249's 814.

Topology growth was gradual rather than explosive: the selected topology ended
at 87 hidden nodes and 358 enabled connections, with selected-topology maxima of
88 and 363. Mean measured evaluation time rose from 4.84 milliseconds per
generation over generations 0-249 to 19.21 milliseconds over generations
750-999. This is ordinary mutation-driven growth, not the former crossover
union, but the absence of behavioral improvement means the additional structure
was not useful. The 1,000-generation result rejects insufficient search duration
as the primary explanation.

### Fixed-evaluation population sweep

A breadth sweep held the organism-lifetime evaluation budget fixed at 256,000,
kept the target-species ratio at one species per 32 individuals, and reduced
generations as population increased. The minimum generation count was 100, so
the maximum matched population was 2,560. Every point used seed 101 and the same
task, resource, mutation, and audit contract.

| Population | Generations | Target/final species | Wall seconds | Synapse ops | Final population mean accuracy | Sealed accuracy | Plasticity-off | Sealed solves/1k | Final hidden/edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1,000 | 8 / 6 | 29.50 | 22.32B | 10.58% | 8.25% | 7.90% | 4.27 | 87 / 358 |
| 512 | 500 | 16 / 13 | 7.98 | 12.40B | 8.23% | 8.62% | 10.80% | 5.68 | 35 / 205 |
| 1,024 | 250 | 32 / 31 | 5.53 | 6.92B | 11.79% | 11.61% | 12.36% | 6.81 | 15 / 110 |
| 2,048 | 125 | 64 / 80 | 5.51 | 4.03B | 13.34% | 13.41% | 10.57% | 6.04 | 10 / 70 |
| 2,560 | 100 | 80 / 106 | 5.12 | 2.98B | 13.84% | 15.10% | 12.25% | 5.86 | 7 / 58 |

The two largest populations are the only points whose sealed primary accuracy
exceeded 12.5% chance and whose primary exceeded plasticity-off by about 2.8
percentage points. Their final population means also exceeded chance, so the
effect is broader than one lucky terminal leader. This is evidence that search
breadth and founder/variation diversity matter more than deep ancestry for this
contract.

The budget was fixed in lifetime evaluations, not actual FLOPs. Synapse work
fell 7.5-fold from the deepest to the broadest point because fewer generations
produced much smaller networks. The high-population improvement therefore did
not come from more neural compute, but population size, evolutionary depth, and
topology age remain confounded. The single-seed maximum is still only 15.10%,
and at population 2,560 disabling prediction-error feedback increased accuracy
to 20.70%. The sweep does not establish the intended temporal mechanism; it
shows that broader search partially mitigates the selection failure.

### Fixed-100-generation population sweep

A second sweep removed evolutionary depth as a confound by holding every point
at 100 generations. Population, target species, lifetime evaluations, and
actual compute increased together. The population-2,560 endpoint is identical
to the endpoint above; the other four points are new matched runs.

| Population | Target/final species | Lifetime evaluations | Wall seconds | Synapse ops | Final population mean accuracy | Sealed accuracy | Plasticity-off | Sealed solves/1k | Final hidden/edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8 / 8 | 25,600 | 0.42 | 0.30B | 14.87% | 15.97% | 12.06% | 5.58 | 10 / 74 |
| 512 | 16 / 13 | 51,200 | 0.68 | 0.59B | 12.54% | 14.83% | 11.65% | 5.77 | 11 / 72 |
| 1,024 | 32 / 45 | 102,400 | 1.37 | 1.26B | 14.02% | 15.52% | 11.68% | 6.77 | 5 / 62 |
| 2,048 | 64 / 105 | 204,800 | 3.79 | 2.72B | 13.82% | 11.24% | 10.65% | 6.01 | 4 / 61 |
| 2,560 | 80 / 106 | 256,000 | 5.12 | 2.98B | 13.84% | 15.10% | 12.25% | 5.86 | 7 / 58 |

At fixed generation depth there is no monotonic population-size effect. Sealed
accuracy remains in a narrow 11.24%-15.97% band, and population-mean accuracy
remains in a 12.54%-14.87% band despite a tenfold increase in population and
compute. The population-2,048 terminal leader was worse than its population
mean, illustrating that one-lifetime ticket production is a noisy way to choose
the final representative.

Comparing matched populations across the two sweeps changes the interpretation
of the equal-evaluation result. Population 256 scored 15.97% at generation 100
but only 8.25% at generation 1,000; population 512 fell from 14.83% at 100 to
8.62% at 500; population 1,024 fell from 15.52% at 100 to 11.61% at 250.
Population 2,048 is the exception, rising from 11.24% at 100 to 13.41% at 125.
The apparent breadth advantage was therefore mostly a depth/degradation
confound: weak learners appear by generation 100, but this selection process
does not reliably retain or improve them.

Prediction-error feedback also remained noncausal or harmful. Removing it
improved the sealed leader at populations 256, 512, 1,024, and 2,560, and was
approximately neutral at 2,048. The principal bottleneck is now retention and
competitive attribution under noisy resource-ticket selection, not insufficient
population breadth.
