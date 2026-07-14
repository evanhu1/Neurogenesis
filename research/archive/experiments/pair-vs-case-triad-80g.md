# Pairwise versus case-matched triads, 80 generations

## Bottom line

The 40-generation advantage of case-matched triads **did not replicate in the
predeclared generation-40–79 window**. It became seed-dependent and reversed on
the aggregate retention measures.

- Pairwise had a positive held-out tail-strength slope on all three seeds;
  triads did so on two of three.
- Pairwise won 71/107 late chronological checkpoint comparisons versus 66/108
  for triads.
- Pairwise final checkpoints beat 15/24 saved tail predecessors versus
  13/24 for triads.
- No final checkpoint in either arm beat every available earlier checkpoint.
- Every run found at least one post-40 checkpoint with a better common-panel
  rating than every pre-40 checkpoint, so search did not simply stop at 40.
  However, only two of six final checkpoints directly beat the strongest
  pre-40 checkpoint in their run.
- All six final champions were functionally plant foragers on the held-out
  panel. Late progress was overwhelmingly better plant exploitation, not a
  growing behavioral or trophic repertoire.

This is continued but slowing, non-monotonic optimization with forgetting and
cycling. It is not a solved plateau and is not evidence of open-endedness.

The triad remains computationally efficient: it used one-third fewer training
worlds. It should be regarded as a cheaper alternative evaluator, not a proven
anti-plateau mechanism.

## Contract

Both arms used only contemporary opponents. There is no hall of fame or
historical opponent archive, and opponent renewal was disabled.

- evolutionary seeds: `7,17,27`
- population: 24
- generations: 80 (`0` through `79`)
- horizon: 5,000 ticks
- training world seeds: `11,29,47,61`
- baseline scenario, predation enabled
- 50x50 world, 102 founders
- unchanged objective, NEAT, brain, mutation/speciation, and ecology defaults

| Arm | Lineages/world | Memberships/genome | Exposures/genome | Cases/genome | Worlds/generation |
|---|---:|---:|---:|---:|---:|
| Pair | 2 | 8 | 8 | 32 | 384 |
| Case-matched triad | 3 | 8 | 16 | 32 | 256 |

The two arms match independent world memberships and scored cases, but a triad
necessarily provides two simultaneous opponents and scores three lineages per
world. Total founders and area are fixed: pair worlds contain `51+51` founders;
triads contain `34+34+34`.

Duplicate 50-tick preflights were byte-identical. All 480 emitted generation
records reported the expected case/world counts. Generations 0–38 reproduce the
previous 40-generation trajectories exactly. Generation 39 has the same
genomes and evaluation but different breeding telemetry because it was terminal
in the shorter run and produces generation 40 here.

## Common assay and definitions

Every saved champion was replayed in the same frozen two-lineage evaluator on
eight unseen seeds: `101,131,151,181,211,241,271,311`. Checkpoints were
`0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,79`. Complete all-pairs
crossplay was retained. Byte-identical checkpoint genomes are excluded from
clone-versus-clone cells.

Raw training fitness is contextual and is not compared across pair and triad
as though it were the same measurement.

The main measures are:

- **Common-panel strength:** a checkpoint's mean held-out survival margin
  against every distinct saved checkpoint genome in its own run, with one vote
  per distinct opponent genome.
- **Tail-strength slope:** the linear slope of that common-panel strength over
  checkpoints 40–79. It detects broad directional movement but does not imply
  monotonic progress.
- **Chronological win:** in a direct held-out matchup, the later checkpoint has
  greater mean survival than the earlier checkpoint.
- **Final retention:** generation 79 directly beats an earlier checkpoint.
- **Population retention:** generation-79 training-context population mean or
  median divided by its maximum during generations 40–79. It diagnoses
  population collapse within a run; its absolute level is not a clean
  cross-arm comparison because pair and triad ecologies differ.

## Aggregate result

| Primary metric | Pair | Case triad |
|---|---:|---:|
| Mean common-panel strength slope, generations 0–35 | +0.01062 | **+0.01673** |
| Mean common-panel strength slope, generations 40–79 | **+0.00453** | +0.00396 |
| Seeds with positive tail slope | **3/3** | 2/3 |
| Early chronological wins, checkpoints 0–35 | 66/84 | **72/84** |
| Late chronological wins, checkpoints 40–79 | **71/107** | 66/108 |
| Late per-world wins | **558/856** | 532/864 |
| Late adjacent-checkpoint wins | **13/23** | 11/24 |
| Final versus tail predecessors | **15/24** | 13/24 |
| Final per-world wins against all predecessors | **250/384** | 233/384 |
| Runs where final beats every earlier checkpoint | 0/3 | 0/3 |
| Runs with a post-40 common-panel record | 3/3 | 3/3 |
| Exact checkpoint repeats | 1 | 0 |
| Training worlds across three seeds | 92,160 | **61,440** |

The early-window ordering reproduces the earlier result: triads are better over
the first half. The primary tail window reverses it. Triads do not give a
stable long-horizon advantage.

Both arms also slow substantially. Pairwise mean strength slope falls from
`+0.01062` early to `+0.00453` late; triads fall from `+0.01673` to `+0.00396`.
Positive slopes therefore coexist with a clear reduction in the rate of
progress.

## Per-seed result

| Arm / seed | Tail slope | Tail wins | Final tail wins | Final panel margin | Final population mean retention | Final population median retention |
|---|---:|---:|---:|---:|---:|---:|
| Pair 7 | +0.00197 | 20/35 | 7/8 | +0.242 | 81.4% | 68.5% |
| Pair 17 | +0.00751 | 27/36 | 2/8 | +0.191 | 90.9% | 94.3% |
| Pair 27 | +0.00412 | 24/36 | 6/8 | +0.270 | 87.4% | 65.8% |
| Triad 7 | **-0.00278** | 17/36 | 4/8 | -0.006 | 83.1% | 56.0% |
| Triad 17 | +0.00876 | 27/36 | 7/8 | **+0.412** | 76.8% | **15.8%** |
| Triad 27 | +0.00591 | 22/36 | 2/8 | +0.277 | 91.2% | 100.0% |

### Seed 7

Pairwise improves sharply at generation 45, then cycles. Generation 60 is
exactly repeated at 65. The final checkpoint recovers to a positive margin but
remains below generation 45. It beats the strongest pre-40 checkpoint on five
of eight worlds.

Triadic seed 7 is a clear treatment failure. Its tail slope is negative. It
peaks at generation 65, collapses at 75 to a common-panel margin of `-0.587`,
then partially recovers at 79. The final genome loses to the strongest pre-40
checkpoint on all eight worlds. Its population median retains only 56% of its
tail maximum.

### Seed 17

Pairwise produces broad late plant foragers and has the strongest pairwise tail
slope. Yet the final checkpoint loses to six of eight tail predecessors. Its
generation-75 common-panel leader and generation-79 final genome both lose
directly to generation 10's predatory omnivore; the final loses on seven of
eight held-out worlds. Population competence remains comparatively healthy.

Triadic seed 17 is the strongest evidence in favor of triads. Generations 70–79
establish new common-panel records, and the final genome beats the strongest
pre-40 genome on all eight worlds. But the population does not move with its
elite: the final median survival is `0.098`, only 15.8% of its tail maximum.
The run also abandons predation. Generation 45 averages about 24 kills; by
generation 65 the common-panel champions have zero.

### Seed 27

Pairwise cycles through a low-competence omnivore at generation 60 and recovers
as a plant forager at 70. The final genome is below generation 70 and loses to
the strongest pre-40 predatory checkpoint on all eight worlds.

Triadic seed 27 reaches the experiment's highest common-panel margin (`+0.538`)
at generation 75, then regresses to `+0.277` at 79. The final population is
healthy—the median is at its tail maximum—but the final champion loses to the
strongest pre-40 forager on six of eight worlds. Its all-time contextual
training champion remains generation 17, illustrating why training fitness
alone is a poor progress measure.

## Progress exists, but it is not cumulative

Every run discovers a post-40 checkpoint whose common-panel rating exceeds all
its pre-40 checkpoints. That rejects a literal fixed point at generation 40.

However, a common-panel rating is not transitive dominance. Raw eight-world
traces comparing each run's strongest post-40 checkpoint to its strongest
pre-40 checkpoint show:

| Run | Late versus early | Per-world wins | Mean survival result |
|---|---|---:|---|
| Pair 7 | Gen 45 vs 30 | 7/8 | later wins |
| Pair 17 | Gen 75 vs 10 | 0/8 | later loses |
| Pair 27 | Gen 70 vs 35 | 0/8 | later loses |
| Triad 7 | Gen 65 vs 25 | 5/8 | later slightly loses on mean |
| Triad 17 | Gen 75 vs 35 | 8/8 | later wins |
| Triad 27 | Gen 75 vs 35 | 8/8 | later wins |

Thus later organisms often become broadly better against the changing panel
while remaining exploitable by a particular historical strategy. Generation
79 directly beats the strongest pre-40 checkpoint in only pair seed 7 and
triad seed 17. No final genome beats every prior checkpoint.

The complete per-world case rows—including alive ticks by lineage, plant
supply/capture, consumptions, kills, action fractions, coverage, and energy—are
preserved in `analysis.json` under each run's
`strongestPost40VersusBestPre40Trace` and `finalVersusBestPre40Trace`, as well
as in the raw crossplay files.

## Behavior and ecology

Late improvement does not expand trophic behavior.

- All six generation-79 held-out champions are effectively pure plant
  foragers. Their mean held-out kills are at most `0.017`; these are
  negligible.
- Final training populations contain no pure predators. Across 144 final
  genomes, 130 are categorized as foragers, ten as nominal omnivores, and four
  as nonconsumers. The nominal omnivore champions have negligible prey intake.
- Pair seed 17's strongest pre-40 opponent averages about 74–87 kills depending
  on matchup; pair seed 27's averages about 61–76. Their late foragers achieve
  stronger broad panel ratings but remain vulnerable to those strategies.
- Triad seed 17 carries predation into the early tail, then gains strength while
  eliminating it. Multi-lineage training therefore did not maintain a
  predator-prey arms race.
- The strongest late checkpoints cover 98–100% of the world and obtain nearly
  all energy from plants. Their behavioral differences are mainly movement,
  turning, eating frequency, and plant-capture efficiency.

A standardized continuous-feature screen flags three pairwise and four triadic
post-40 checkpoints as both behaviorally distant from pre-40 checkpoints and
stronger in common-panel rating. Inspection shows these are altered plant
foraging/action mixtures, not new trophic roles or new ecological dynamics.
This is metric novelty, not sufficient evidence of genuinely new behavior.

## Search machinery and population competence

NEAT did not stop producing structure. During generations 40–79, each run
created 5–15 new node innovations and 26–47 new connection innovations. Pair
and triad averages are nearly identical: 10.3 versus 9.7 new nodes and 35.7
versus 36.0 new connections. Continued topology supply therefore does not by
itself prevent behavioral convergence.

Population outcomes are mixed:

- Pairwise retains 86.5% of its maximum tail mean and 76.2% of its maximum tail
  median on average.
- Triads retain 83.7% of their maximum tail mean and 57.3% of their maximum
  tail median.
- Triad seed 17 is the clearest elite/population split: an excellent final
  champion coexists with a deeply degraded median population.

This argues against reading champion progress as population-wide evolutionary
progress.

## Cost

Pairwise simulates 30,720 training worlds per seed; triads simulate 20,480.
Across the three seeds that is 92,160 versus 61,440 worlds.

Concurrent operational wall times averaged approximately:

- pair evolution: 51.8 minutes per run;
- triad evolution: 38.2 minutes per run;
- common crossplay: 4.7 minutes for pair-source runs and 7.1 minutes for
  triad-source runs, with behavior-dependent tick cost.

The runs were concurrent, so summed process wall time is not a throughput
benchmark. Simulator-world count is the cleaner training-cost comparison.

## Conclusion and next causal experiment

Do not sweep four or five lineages next. Three lineages improved the first 40
generations but did not improve the 40–79 tail. Increasing lineage count would
add ecological and evaluation confounds without addressing the observed
failure: plant convergence plus nontransitive forgetting.

The next clean capacity test is **population 24 versus 48 under the pairwise
evaluator**, keeping 32 scored cases per genome and every other setting fixed.
Allow the 48-genome arm to cost twice as much rather than reducing evaluation
precision. Use the same frozen assay and predeclare generations 40–79.

- If population 48 materially improves late strength, final retention, and
  population median across seeds, the remaining limit is partly search
  capacity: more lineages/species preserve more stepping stones.
- If both populations show the same plant convergence and late reversals, the
  fixed ecology/task is the binding limit. The next mechanism should then
  create or preserve distinct competent ecological niches rather than tune
  sensors, mutation rates, or lineage count blindly.

Pairwise should be the scientific baseline for that test because it was more
reliable in the primary tail window. The case triad remains available as a
one-third-cheaper screening evaluator, but it should not be assumed to prevent
plateau.

## Artifacts

- `COMMANDS.md`: exact contract, commands, preflight hashes, and assay
- `run-one.sh`: evolution launcher
- `run-crossplay.sh`: held-out assay launcher
- `analyze.mjs`: reproducible derivation
- `analysis.json`: aggregate, per-seed, checkpoint, continuous behavior,
  structural, and selected raw case traces
- `<arm>-seed-<seed>/`: schema-20 result, full population history, progress
  JSONL, champion world and metric sidecar, crossplay, stderr, and wall times
- `preflight/`: duplicate deterministic smoke artifacts

## Validation

- `make fmt`: passed.
- `make lint`: passed with warnings denied.
- `cargo check --workspace`: passed.
- `cargo test --workspace`: reached only the known failure
  `lethal_attack_spawns_corpse_food_without_feeding_attacker` (`left: 1`,
  `right: 0`).
- `cargo test --workspace -- --skip
  lethal_attack_spawns_corpse_food_without_feeding_attacker`: passed in full.

This report does not claim open-endedness. Eighty generations are enough to
show that the triad's early advantage is not stable in the tail and that neither
evaluator produces cumulative behavioral expansion across these seeds.
