# Energy-stealing pairwise versus all-population arena

Status: completed; shared arena rejected as the sole evaluator

## Hypothesis

With plants and Eat removed, a shared eight-genome arena may provide a stronger
locality-based selection signal than it did in the plant ecology because every
fitness-relevant interaction now involves opponent energy. Doubling independent
world/spawn seeds from four to eight should also reduce rank sensitivity to any
one shared context.

## Matched contract

Both arms use:

- evolutionary seeds `7,17,27`;
- population `8`, generations `0..39`, and a 500-tick horizon;
- `energy_stealing`: no plant tiles, no Eat action, equal 250-energy founders,
  conserved 40-energy attack transfer, 10-energy attack-attempt cost, and
  one-energy passive metabolism;
- training world seeds `11,29,47,61,73,89,107,127`;
- held-out validation seeds `101,131,151,181`;
- 50x50 worlds, 96 founders, absolute-survival fitness, and mean aggregation;
- the same NEAT mutation, crossover, speciation, and selection settings.

Pairwise control exhaustively evaluates all seven contemporary opponents in
separate worlds: 56 cases per genome and 224 worlds per generation.

Shared-arena treatment places all eight genomes in each world: eight joint
cases per genome and eight worlds per generation. Every genome is present with
all seven opponents in all eight worlds, but those presences are correlated
within each joint arena and are not counted as independent cases.

## Measurements and decision rule

Training endpoints are champion, population mean, and median survival; full and
`20..39` slopes; case-score dispersion and minima; attack precision, hits,
kills, transfer received/lost, attempt cost, net energy profit, action
effectiveness, and end-survivor censoring.

Frozen checkpoints `0,10,20,30,39` will be crossplayed pairwise on held-out
seeds. Historical champions from the two arms will also be compared directly
in both founder slots.

Shared evaluation is promotable only if all seeds improve over generation 0,
at least two retain nonnegative `20..39` population-mean progress, held-out
chronological progress is not materially worse, and direct champion comparison
does not show a systematic pairwise advantage. Compute savings alone do not
qualify it.

## Commands

```text
cli batch --experiment 2026-07-15-energy-stealing-<arm>-40g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 8 --generations 40 --population-checkpoint-interval 10 \
  --horizon 500 --task energy_stealing \
  --world-seeds 11,29,47,61,73,89,107,127 --founders 96 --cvar 1 \
  <pairwise: --evaluator pairwise --opponents-per-genome 7> \
  <arena: --evaluator shared_population>
```

Each live `neat_generation` progress line contains the evolutionary seed,
completed/total generations, elapsed time, generation time, rolling mean
seconds per generation, and ETA. The identical JSONL is persisted per seed.

## Result

The shared arena was approximately 26x faster in wall-clock time and 28x
cheaper in simulated worlds. Pairwise used 224 worlds per generation and took
45.9--49.5 seconds per evolutionary seed. The arena used eight worlds per
generation and took 1.82--1.86 seconds per seed.

All six runs improved over their random generation-0 populations, but the
preregistered late-progress gate failed. Population-mean fitness over
generations `20..39` had the following linear slopes:

| Evolution seed | Pairwise | Shared arena |
|---:|---:|---:|
| 7 | -0.00110 | -0.00065 |
| 17 | -0.00007 | +0.00465 |
| 27 | +0.00003 | -0.00014 |

Only arena seed 17 retained positive late population progress; the promotion
rule required at least two seeds. Pairwise seeds 17 and 27 converged to the
passive-survival score of approximately `0.498`. Arena seed 27 did the same.
Pairwise seed 7 retained attacks but regressed late. Arena seeds 7 and 17
retained profitable historical champions:

| Arm / seed | Historical champion generation | Training fitness | Attack fraction | Prey rate | Net attack profit |
|---|---:|---:|---:|---:|---:|
| Pairwise 7 | 21 | 0.5082 | 0.00572 | 0.00514 | +291.0 |
| Pairwise 17 | 11 | 0.4980 | 0 | 0 | 0 |
| Pairwise 27 | 4 | 0.4980 | 0 | 0 | 0 |
| Arena 7 | 38 | 0.5594 | 0.01172 | 0.00717 | +373.6 |
| Arena 17 | 30 | 0.5750 | 0.01369 | 0.01051 | +530.3 |
| Arena 27 | 35 | 0.4985 | 0.00280 | 0.00113 | +1.3 |

Pairwise held-out crossplay of frozen checkpoints `0,10,20,30,39` found that
the later checkpoint won 19 of 30 chronological comparisons for the pairwise
arm and 20 of 29 distinct-genome comparisons for the arena arm. Per seed the
counts were pairwise `10/10, 5/10, 4/10` and arena `9/10, 7/10, 4/9`.
Therefore shared training did not produce a noisier historical sequence in
this small test, but neither arm produced seed-robust continuing progress.

Direct held-out historical-champion matchups were evaluated in both founder
slots. Arena champions beat the corresponding pairwise champions on all eight
seed-7 cases and all eight seed-17 cases. Seed 27 was effectively tied and
slot-sensitive: arena won five of eight cases, while the two-slot mean survival
slightly favored pairwise (`0.4881` versus `0.4860`). Thus the arena's active
champions are real, transferable strategies rather than a joint-world scoring
artifact.

## Interpretation

The arena is informationally different, not merely a cheaper approximation of
pairwise evaluation. Higher organism density creates many more attack
opportunities, and a successful attacker can exploit several lineages in one
episode. This produced strong, transferable predators in two seeds. Pairwise
evaluation more often makes abstaining from attack optimal: with no plants, a
non-attacker pays only passive metabolism, reaches `0.498` survival, has zero
case variance, and cannot improve further unless an attack policy earns more
than its attempt costs and exposure to retaliation.

The arena does not remove that equilibrium. Seed 27 discovered passivity and
stopped. Seeds 7 and 17 also show that active coevolution remains contextual:
their historical champions are strong, but final population quality and
historical-champion quality do not move monotonically together. More world
seeds reduced layout sampling noise but did not resolve the game-theoretic
incentive to avoid costly interaction.

The shared arena therefore remains useful as a very cheap screening evaluator
and as a way to generate aggressive strategies, but it is not promoted as the
sole research baseline. The primary bottleneck is now the task payoff: absolute
survival rewards mutual non-aggression with a high, flat score. Increasing
arena cardinality or world-seed count alone cannot guarantee escape from that
attractor.

## Reproduction and artifacts

Training outputs and generation progress logs:

- `artifacts/research/runs/completed/2026-07-15-energy-stealing-pairwise-40g/`
- `artifacts/research/runs/completed/2026-07-15-energy-stealing-arena-40g/`

Each directory contains the batch manifest, three compressed results,
champion worlds, streamed `seed-N.progress.jsonl` logs, and held-out checkpoint
crossplay files. The training commands are recorded above. Crossplay used:

```text
cli crossplay <result> --checkpoints 0,10,20,30,39 \
  --horizons 500 --world-seeds 101,131,151,181 --out <heldout.json>
```

Direct champion comparisons used `cli evaluate-panel` in both focal
directions with the same horizon and held-out world seeds.

## Next causal experiment

Do not scale this treatment directly to a long run. First isolate the passive
equilibrium with a hand-authored or frozen active predator panel: compare the
fitness of a passive controller, a successful arena predator, and small
mutations around each under pairwise and shared evaluation. The next payoff
change should be considered only after measuring whether aggression fails
because encounters are too rare, attack attempts are too costly, retaliation
dominates, or absolute alive-ticks fundamentally favors mutual abstention.
