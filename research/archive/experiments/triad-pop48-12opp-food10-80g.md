# Three-lineage population-48 evolution with 10% plant tiles

## Bottom line

Ten-percent plant coverage **solved the narrow plant-only-convergence problem**
but **did not solve the post-30-generation progress problem**.

The late populations are ecologically different from the earlier 40%-plant
runs: averaged across seeds, about 50% of genomes are predators and 44% are
omnivores, and late checkpoint champions obtain about 69% of intake from prey.
Predation is behaviorally substantial (roughly 34 kills per late champion), not
categorical-label noise.

However, frozen held-out strength is almost flat over generations 40-79. Only
57/108 late chronological checkpoint comparisons favor the later genome, late
adjacent checkpoints split 12/24, and final genomes beat only 10/24 tail
predecessors. One seed progresses, one cycles/regresses, and one ends with a
strong final counterstrategy after a non-progressive tail.

The treatment therefore replaces a plant-foraging attractor with a mixed
predator/omnivore regime, but it does not create reliably progressive
coevolution.

## Exact contract

- evolutionary seeds: `7,17,27`
- population: 48 genomes
- generations: 80 (`0` through `79`)
- three contemporary lineages per world
- 12 opponent exposures = 6 triadic memberships per genome
- four training world seeds and 5,000 ticks
- 24 cases/genome and 384 worlds/generation
- 50x50 world, 102 founders (`34+34+34`)
- predation enabled
- `food_tile_fraction=0.10`
- food energy and regrowth unchanged (`20`, `200 +/- 50` ticks)
- no historical opponents or Hall of Fame

All 240 generations emitted the expected schema-20 accounting. Every genome
actually saw 9-12 distinct contemporary opponents at generation 79 (mean about
10.8), despite repeated exposures permitted by the deterministic scheduler.

The held-out assay uses checkpoints
`0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,79`, 5,000 ticks, and unseen
world seeds `101,131,151,181,211,241,271,311`. It is the repository's frozen
two-lineage common panel, so it measures pairwise transferable competence after
triadic training rather than native three-way performance.

## Aggregate progress

| Measure | Result |
|---|---:|
| Early held-out strength slope, generations 0-35 | +0.00246 |
| Late held-out strength slope, generations 40-79 | +0.00009 |
| Materially positive late seeds | 1/3 |
| Early chronological wins | 66/83 (79.5%) |
| Late chronological wins | 57/108 (52.8%) |
| Cross-window wins, late versus early | 180/216 (83.3%) |
| Late adjacent wins | 12/24 (50.0%) |
| Final versus tail predecessors | 10/24 (41.7%) |
| Final versus all earlier checkpoints | 26/48 (54.2%) |
| Runs with a post-40 strength record | 3/3 |
| Final beats every distinct earlier checkpoint | 1/3 |
| Final population mean survival | 0.0977 |
| Final population median survival | 0.1017 |
| Final mean retained from each run's late peak | 84.2% |
| Final median retained from each run's late peak | 86.8% |

The strong 83% cross-window result means the late populations usually retain
the large early adaptive gains. The 53% within-tail result means those gains do
not continue accumulating directionally after generation 40.

## Per-seed result

| Seed | Late strength slope | Late chronological | Final vs tail | Tail interpretation |
|---:|---:|---:|---:|---|
| 7 | approximately 0 | 11/36 | 1/8 | early progress, then cycling and population collapse |
| 17 | +0.00047 | 28/36 | 1/8 | real late rise through generation 75, then final regression |
| 27 | -0.00020 | 18/36 | 8/8 | declining/nonmonotonic tail followed by a strong final counterstrategy |

### Seed 7

Seed 7 becomes strongly predatory: its late populations average 77% predators,
and late champions obtain 97% of intake from prey. A generation-55 pure
predator averages about 50 held-out kills and sets a post-40 strength record.

This does not translate into chronological progress. Only 11/36 late pairs and
2/8 late adjacent pairs favor the later checkpoint. The final population mean
and median retain only 67% and 71% of their late peaks, and generation 79 beats
only one of eight tail predecessors.

### Seed 17

Seed 17 is the only convincingly progressive tail. Its late strength, best
training survival, population mean, and population median all have positive
slopes. Generation 75 is the strongest post-40 checkpoint, with held-out
absolute survival 0.157 and a +0.054 pairwise margin.

Its ecology also shifts back toward plants. The strongest post-40 checkpoint
obtains 92% of intake from plants, and the final held-out genome obtains 97%
from plants. Generation 79 then regresses, beating only one tail predecessor
and ending with a negative pairwise margin. Thus the one progressive seed does
not preserve its best result and partially rediscovers the foraging attractor.

### Seed 27

Seed 27 has a negative fitted late slope and only 18/36 late chronological
wins, yet generation 79 beats every distinct earlier checkpoint, including all
eight tail predecessors. This is a final jump after a weak/nonmonotonic tail,
not steady improvement.

The final genome is a genuine mixed strategy: about 56% of intake is prey,
44% is plant, and it averages 59 kills on the held-out panel. Its population
mean and median retain 85% and 92% of their late peaks.

## Ecology and behavior

The density change clearly alters the evolved energy strategy.

| Held-out checkpoint panel | Plant intake | Prey intake | Mean kills | Plant capture | Standing plant fraction |
|---|---:|---:|---:|---:|---:|
| Early (`0-35`) | 29.1% | 70.9% | 26.0 | 8.0% | 97.9% |
| Late (`40-79`) | 26.7% | 73.3% | 41.3 | 11.9% | 97.2% |
| Final (`79`) | 62.6% | 37.4% | 34.4 | 28.8% | 94.3% |

The final aggregate swings plantward because seed 17 evolves a strong plant
specialist. Seeds 7 and 27 remain prey-majority. The population-level late
training roles are approximately 3% nonconsumers, 3% foragers, 50% predators,
and 44% omnivores.

This is not the previous label artifact: prey intake, attack hits, damage,
energy gain, and kills all move together. But it exposes a new ecological
concern. Even at 10% spatial coverage, 97% of plant sites remain occupied in
the typical late held-out panel. Organisms are not exhausting renewable plant
production; predators are largely harvesting the finite starting energy stored
in other founders. Mean end-survivor fraction is only about 1.1% in the late
panel. The ecology is therefore closer to a finite founder-energy tournament
than a persistent renewable food web.

That is valid for the bounded evaluator, but it explains why reducing plant
density can create predation without automatically creating an escalating
arms race.

## Search and integrity checks

Search did not freeze:

- about 55 new connection and 14 new node innovations per late half/run
- about 1,230 crossover offspring per late half/run
- about 4.4 species on average
- structural mutation success remained high

There were zero cases with zero combined alive-ticks in both training and
held-out crossplay: every case retained a defined alive-tick signal even when
all lineages were dead by the final tick. The evaluator's plant-instance conservation assertion and the engine's
fail-closed energy ledger passed all 92,160 training worlds and all crossplay
worlds. The low final survivor count is an observed ecological outcome, not an
accounting failure.

## Interpretation and next step

This result separates two questions that had been conflated:

1. **Was 40% plant coverage forcing plant-only convergence?** Yes. Moving to
   10% produces sustained, quantitatively important predation and mixed energy
   strategies across all three runs.
2. **Was plant dominance the reason evolution stopped progressing after about
   30 generations?** No, or at least not by itself. The late common-panel trend
   remains essentially flat and seed-dependent.

The next principled step is not immediately to lower food further. At 10%, most
plant sites stand unused and almost all founders die, so lower density would
mainly deepen the finite-energy tournament. The useful follow-up is a narrow
ecology calibration around the transition (for example 10%, 15%, and 20%) using
shorter paired runs, selecting for a regime with all three properties:

- meaningful prey intake and kills,
- plants actively contested rather than 95-98% standing,
- a non-collapsed end-survivor distribution.

Separately, evaluation precision still needs attention. Twelve triadic
opponent exposures cover about 10.8 distinct opponents, but they provide only
six world memberships and 24 cases/genome—less than the 32-case earlier runs.
Before another 80-generation campaign, reevaluate persisted populations on a
larger independent panel to test whether training rank reliably predicts
held-out rank. If rank reliability is poor, increase memberships/cases or use a
more conservative fitness aggregator. If it is good, the remaining plateau is
an ecology/task-structure problem rather than evaluator noise.

## Artifacts

- `COMMANDS.md`: exact run and assay contract
- `run-one.sh`: deterministic evolution launcher
- `run-crossplay.sh`: held-out checkpoint launcher
- `analyze.mjs`: reproducible analysis
- `analysis.json`: full aggregate and per-seed derivation
- `run-seed-{7,17,27}/`: raw schema-20 results, champion worlds, progress,
  wall times, and crossplay matrices
