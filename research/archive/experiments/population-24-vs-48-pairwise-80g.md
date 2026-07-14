# Population 24 versus 48 under pairwise evaluation, 80 generations

## Bottom line

Doubling the NEAT population did **not** extend progressive evolution. It
accelerated early common-panel improvement on two seeds and supplied more
structural innovations, but all three population-48 runs had negative held-out
strength slopes during generations 40–79. Population 24 remained positive on
all three seeds.

Population 48 also cost twice as many simulator worlds while producing weaker
final competence, worse final retention, and more late regressions. Search was
not limited by the number of genomes or structural mutations available at
population 24.

This does not prove that large populations are intrinsically harmful. Each
population-48 genome still faced only eight of 47 possible opponents, versus
eight of 23 at population 24, and the same 32-case estimate selected the maximum
from twice as many noisy candidates. The result rejects naive horizontal
scaling; it motivates a fitness-rank reliability audit before another expensive
population experiment.

## Contract

The population-24 control is reused without rerunning from the completed exact
pairwise arm in
`artifacts/research/runs/completed/pair-vs-case-triad-80g/`. The treatment
changes only NEAT population size.

- evolutionary seeds: `7,17,27`
- generations: 80 (`0` through `79`)
- horizon: 5,000 ticks
- training world seeds: `11,29,47,61`
- pairwise contemporary-only evaluation
- eight opponent memberships and 32 scored cases per genome
- baseline scenario, predation enabled
- 50x50 world with 102 founders (`51+51`)
- unchanged objective, brain, ecology, mutation, selection, and speciation

| Population | Cases/genome | Worlds/generation | Worlds across three runs |
|---:|---:|---:|---:|
| 24 | 32 | 384 | 92,160 |
| 48 | 32 | 768 | 184,320 |

Every treatment generation emitted the expected counts. Two population-48
preflights were byte-identical.

All checkpoints `0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,79` were
replayed in the same frozen pairwise evaluator on unseen world seeds
`101,131,151,181,211,241,271,311`. Each treatment crossplay contains 17
checkpoints and all 272 distinct-genome ordered cells.

## Aggregate result

| Metric | Population 24 | Population 48 |
|---|---:|---:|
| Early strength slope, generations 0–35 | +0.01062 | **+0.02582** |
| Late strength slope, generations 40–79 | **+0.00453** | **-0.00184** |
| Seeds with positive late slope | **3/3** | 0/3 |
| Early chronological wins | 66/84 | 67/84 |
| Late chronological wins | **71/107** | 51/108 |
| Late adjacent wins | **13/23** | 10/24 |
| Final versus tail predecessors | **15/24** | 7/24 |
| Late regressions | **36** | 57 |
| Runs with a post-40 strength record | **3/3** | 2/3 |
| Mean final best training survival | **0.908** | 0.811 |
| Mean final population survival | **0.456** | 0.415 |
| Mean final population median | **0.500** | 0.421 |
| Final population-mean retention | **86.5%** | 81.3% |
| Final population-median retention | **76.2%** | 69.2% |
| Behaviorally distant and stronger post-40 checkpoints | 3 | 3 |

Population 48's early slope is larger, but early chronological ordering is
essentially tied. It found some large early jumps rather than a more reliable
sequence of improvements. The late result is unambiguous across this seed set:
every paired seed changes from a positive population-24 slope to a lower,
negative population-48 slope.

Average generation-79 held-out common-panel survival falls from `0.618` to
`0.498`; mean pairwise margin falls from `+0.234` to `-0.056`. Seed 7 drives
much of the final-level drop, but all three seeds agree on the direction of the
late slope effect.

## Per-seed result

| Seed | Pop-24 early slope | Pop-48 early slope | Pop-24 late slope | Pop-48 late slope | Pop-24 final tail wins | Pop-48 final tail wins |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | +0.0144 | +0.0077 | +0.0020 | **-0.0034** | 7/8 | 0/8 |
| 17 | -0.0046 | **+0.0317** | +0.0075 | **-0.0001** | 2/8 | 2/8 |
| 27 | +0.0221 | **+0.0380** | +0.0041 | **-0.0020** | 6/8 | 5/8 |

### Seed 7

Population 48 is a clear late failure. No post-40 checkpoint exceeds the best
pre-40 common-panel strength. Generation 79 loses to every saved tail
predecessor, has a common-panel margin of `-0.622`, and survives only `0.176`
on the held-out panel. Its population remains moderately competent in its
contemporary training context, showing that contextual selection can hide the
historical weakness.

### Seed 17

Population 48 produces a much faster early rise, then a flat-to-negative tail.
Its final held-out survival (`0.634`) slightly exceeds population 24 (`0.592`),
but it beats only two of eight tail predecessors and its final training
population mean and median are substantially lower. The final nominal omnivore
averages about 1.94 kills and remains overwhelmingly plant powered.

### Seed 27

Population 48 again accelerates early progress but peaks in common-panel
strength around generation 50. The final held-out survival (`0.685`) is higher
than population 24 (`0.615`), yet the fitted tail is negative and generation 79
retains only five of eight tail matchups. The outcome is a capable plant
forager, not a growing ecological repertoire.

## Behavior and search machinery

The larger population did not create more competent behavioral novelty. Both
arms produced three post-40 checkpoints that passed the same conservative
continuous-feature novelty-and-strength screen. Final champions remain plant
specialists: population-48 seeds 7 and 27 are pure foragers, and seed 17's
minor prey intake is ecologically small.

Population 48 did increase mutation supply. During generations 40–79 it
averaged 50.3 new connection innovations and 13.0 new node innovations per run,
versus 35.7 and 10.3 at population 24. It also produced roughly twice as many
crossovers and structural-mutation successes. More genotypic exploration did
not translate into stronger late behavior.

The speciation controller remained active near its fixed target rather than
fragmenting without bound: population 48 averaged roughly 3.2–3.7 late species,
compared with 2.4–3.4 at population 24. The result is not explained by search
freezing or an absence of structural variation.

## Interpretation

Three mechanisms remain plausible and are not separated by this experiment:

1. **Task exhaustion.** The fixed ecology strongly rewards broadly effective
   plant foraging. More candidates find that regime sooner but do not open new
   adaptive niches.
2. **Sparser opponent coverage.** Eight memberships cover about 35% of the
   other genomes at population 24 but only 17% at population 48. The larger
   population may contain less coherent contemporary selection pressure even
   though each genome has the same number of cases.
3. **Winner's curse from noisy evaluation.** Selecting elites from twice as
   many genomes using the same 32-case estimate increases the opportunity for
   an overestimated candidate to win. The seed-7 training-versus-held-out
   discrepancy is consistent with this, but does not prove it.

The fixed target species count and other NEAT settings were deliberately held
constant to isolate raw population size. Their interaction with population size
could matter, but retuning them now would turn one rejected scaling hypothesis
into an open-ended hyperparameter search.

## Recommendation

Do not adopt population 48 and do not immediately try 96. Population 24 is both
cheaper and more progressive in the measured tail.

Before another evolution run, use the persisted full populations to audit
fitness-rank reliability at selected generations: reevaluate every genome on a
large independent opponent/world panel and compare training rank with external
rank. If population 48 shows poorer rank reliability, the next causal treatment
is broader or more conservative evaluation. If rank reliability is comparable,
the evidence points toward ecology/task exhaustion, and the next lever should
be competent behavioral diversity or ecological niche creation rather than
more brute-force genomes.

This experiment rejects population size as the current primary bottleneck. It
does not establish open-endedness or prove that 24 is globally optimal.

## Artifacts

- `COMMANDS.md`: exact contract
- `run-one.sh`: population-48 evolution launcher
- `run-crossplay.sh`: held-out assay launcher
- `analyze.mjs`: reproducible derivation using the existing population-24 controls
- `analysis.json`: complete aggregate, per-run, checkpoint, behavioral, and trace data
- `pop48-seed-{7,17,27}/`: raw schema-20 result, champion world, metrics,
  progress, wall time, and crossplay
