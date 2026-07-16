# Energy-stealing pairwise versus arena at population 16

Status: completed; arena rejected as sole evaluator

## Hypothesis

Doubling the genome pool and arena ecological capacity may make shared locality
more reliable by placing a broader strategy set in every world. Under the new
cost-1 / transfer-20 attack economy, shared arenas should retain active attack
strategies in more seeds and avoid the passive-survival equilibrium observed at
population 8.

## Matched contract

Both arms use evolutionary seeds `7,17,27`, population 16, generations
`0..39`, a 500-tick horizon, absolute-survival fitness, and mean aggregation.
The energy-stealing task has no plants or Eat action, starting energy 250,
attack-attempt cost 1, and successful attack transfer up to 20.

Training world seeds are `11,29,47,61,73,89,107,127`. Held-out validation
seeds are `101,131,151,181`. World width is 71 and each world has 192 founders.
This approximately doubles the prior world area while preserving founder
density (`192 / 71^2` versus `96 / 50^2`) and preserves 12 founders per lineage
in the shared arena.

The pairwise control exhaustively evaluates all 15 contemporary opponents:
120 cases per genome and 960 worlds per generation. The shared treatment puts
all 16 genomes in each of eight worlds: eight joint scored cases per genome.
Each genome nevertheless has 15 opponent lineages present in all eight worlds,
so both arms expose each genome to 120 opponent-lineage presences. Arena
presences are correlated within eight joint contexts and are not treated as
independent cases.

## Measurements and decision rule

Measure generation-0, generation-20, final, and historical-champion survival;
population mean and median; `20..39` slopes; attack rate, precision, kills,
energy transfer, net profit, and case dispersion. Crossplay checkpoints
`0,10,20,30,39` pairwise on held-out seeds and compare historical champions in
both founder slots.

Promote the arena only if at least two seeds retain positive late population
mean progress, at least two seeds retain materially active and profitable
historical champions, held-out chronological progress is no worse than
pairwise, and direct champion comparison does not systematically favor
pairwise. Record whether either topology converges to the passive score near
`0.498`.

## Commands

```text
cli batch --experiment 2026-07-15-energy-stealing-p16-<arm>-40g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 16 --generations 40 --population-checkpoint-interval 10 \
  --horizon 500 --task energy_stealing \
  --world-seeds 11,29,47,61,73,89,107,127 \
  --world-width 71 --founders 192 --cvar 1 \
  <pairwise: --evaluator pairwise --opponents-per-genome 15> \
  <arena: --evaluator shared_population>
```

## Result

The scale match was exact on opponent-lineage exposure. Both arms exposed each
genome to 15 opponent lineages under each of eight world seeds: 120 opponent
presences per genome. Pairwise realized those as 120 independent cases and 960
worlds per generation; arena realized them as eight correlated joint cases and
eight worlds per generation.

Pairwise took 246--269 seconds per evolutionary seed. Arena took 2.38 seconds,
approximately 107x faster in wall-clock time and 120x cheaper in simulator
worlds.

### Training trajectory

Population-mean fitness slopes over generations `20..39` were:

| Evolution seed | Pairwise | Shared arena |
|---:|---:|---:|
| 7 | -0.00004 | +0.00190 |
| 17 | -0.00203 | +0.00418 |
| 27 | +0.00018 | +0.00212 |

Arena passed the declared late-slope gate in all three seeds; pairwise passed
only in seed 27. All historical champions were active and profitable under
their own training evaluator:

| Arm / seed | Champion generation | Training fitness | Attack fraction | Prey rate | Net attack profit |
|---|---:|---:|---:|---:|---:|
| Pairwise 7 | 2 | 0.5499 | 0.1159 | 0.0303 | +4712.9 |
| Pairwise 17 | 21 | 0.5228 | 0.0583 | 0.0186 | +1175.0 |
| Pairwise 27 | 6 | 0.5192 | 0.0716 | 0.0172 | +1135.0 |
| Arena 7 | 6 | 0.5528 | 0.2606 | 0.0568 | +476.1 |
| Arena 17 | 1 | 0.6484 | 0.2735 | 0.0601 | +1243.6 |
| Arena 27 | 1 | 0.5928 | 0.1761 | 0.0831 | +1030.1 |

The arena's positive late slopes do not mean attack capability kept improving.
From generation 20 to 39, mean gross energy acquired fell in every arena seed
while net profit improved. Final champions attacked much less: arena seeds 17
and 27 ended near low-attack survival policies, and only seed 7 retained a
materially profitable final champion. Late arena progress was primarily reduced
waste and de-escalation after early aggressive champions.

Pairwise did not collapse to zero attack, unlike two seeds in the preceding
cost-10 / transfer-40 experiment. Its seed-27 population evolved low-rate,
profitable attacks and positive late progress. Seeds 7 and 17 became more
aggressive but population attack profit deteriorated, producing flat or
negative late survival.

### Held-out validation

Frozen checkpoints `0,10,20,30,39` were pairwise-crossplayed on held-out world
seeds `101,131,151,181`. Later checkpoints won 25/30 chronological comparisons
for pairwise and 26/30 for arena. Per-seed counts were pairwise `10/10, 8/10,
7/10` and arena `10/10, 7/10, 9/10`.

Final-checkpoint retention was weaker than the aggregate count suggests. The
pairwise final checkpoint lost to one prior checkpoint in seed 17 and two in
seed 27. The arena final checkpoint lost to generations 10, 20, and 30 in seed
17 and to generation 30 in seed 27. Arena seed 7 retained a clean chronological
ordering.

Historical champions from the two arms were then matched directly in both
founder slots. Pairwise won 18 of 24 held-out cases: `5/8` in seed 7, `7/8` in
seed 17, and `6/8` in seed 27. Mean survival across both slot directions favored
pairwise in every seed:

| Seed | Pairwise champion | Arena champion |
|---:|---:|---:|
| 7 | 0.4161 | 0.4102 |
| 17 | 0.4316 | 0.4045 |
| 27 | 0.4343 | 0.4177 |

The arena therefore fails the direct-transfer gate despite its better-looking
training slopes.

## Interpretation

Doubling population and ecological capacity did make the shared arena's
training signal smoother and eliminated the seed-27 zero-attack collapse seen
at population 8 under the older economy. It did not make arena fitness an
unbiased substitute for pairwise competence. Eight joint worlds still provide
only eight correlated outcomes, and a genome's score depends on the other 15
lineages simultaneously. Selection can improve by reducing its own losses or by
exploiting particular mixtures without becoming the strongest frozen opponent.

Cost 1 / transfer 20 changes the task in the intended direction but overshoots
in another dimension: cheap missed attacks preserve exploration, yet population
net attack profit is frequently negative and high attack throughput can lower
survival. The useful result is not simply "more predation." It is that every
seed now discovers a profitable historical attacker, while selection still
fails to retain or consistently refine it.

The arena remains valuable as a roughly 100x-cheaper screening and strategy
generation mechanism. It is not promoted as the sole evaluator. A principled
next design is a two-stage evaluator: use shared arenas broadly, then spend a
small pairwise budget to rerank or select survivors. This directly uses the
arena's throughput while grounding selection in the transfer test it currently
fails.

This is a valid pairwise-versus-arena comparison under the new economy, but it
is not a clean population-scale ablation against the preceding population-8
experiment because attack cost/transfer changed from `10/40` to `1/20` at the
same time.

## Artifacts

- `artifacts/research/runs/completed/2026-07-15-energy-stealing-p16-pairwise-40g/`
- `artifacts/research/runs/completed/2026-07-15-energy-stealing-p16-arena-40g/`

Both directories contain manifests, complete results, champion worlds, live
generation/ETA logs, and held-out checkpoint crossplay matrices.
