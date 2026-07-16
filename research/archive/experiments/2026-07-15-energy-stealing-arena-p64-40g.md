# Population-64 shared-arena scale-up

Status: completed; stronger initial search in two seeds, sustained-progress hypothesis rejected

## Hypothesis

Scaling the contemporary strategy pool from 16 to 64 under the shared-arena
default should improve search breadth and delay de-escalation without changing
founder density or per-lineage sampling. The larger pool should produce active,
profitable champions after the early generations and sustain positive late
population progress in at least two evolutionary seeds.

## Contract

- Evolutionary seeds `7,17,27`; generations `0..39`.
- Population 64 under the default `shared_population` evaluator.
- Energy-stealing task, 500 ticks, attack cost 1, transfer 20, no plants/Eat.
- World seeds `11,29,47,61,73,89,107,127`.
- Width 142 and 768 founders: 12 founders per lineage and approximately the
  same organism density as the population-16 width-71/192-founder arena.
- Absolute-survival fitness with mean aggregation.
- Eight joint scored cases per genome. Each case contains all 63 opponents, for
  504 opponent-lineage presences per genome.
- Same NEAT mutation, crossover, selection, and speciation settings as the
  population-16 run.

## Decision rule

Measure generation-0, generation-20, final, and historical champion fitness;
population mean/median; `20..39` slopes; attack rate, precision, kills, transfer,
net profit, case dispersion, species, and topology. Pairwise-crossplay frozen
checkpoints `0,10,20,30,39` on held-out seeds `101,131,151,181`.

Call the scale-up promising only if at least two seeds have positive late mean
slopes, at least two historical champions arise after generation 10 or later
checkpoints systematically defeat earlier ones, and active profitable attack
capability is retained rather than replaced by passive de-escalation. A larger
number of genomes or opponent presences is not itself progress.

## Command

```text
cli batch --experiment 2026-07-15-energy-stealing-arena-p64-40g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 64 --generations 40 --population-checkpoint-interval 10 \
  --horizon 500 --task energy_stealing \
  --world-seeds 11,29,47,61,73,89,107,127 \
  --world-width 142 --founders 768 --cvar 1
```

## Result

All three runs completed in 6.32--6.36 seconds. By generation 39, seeds 7 and
17 retained 7 and 10 species respectively; seed 27 retained 4. Structural
variation remained active, but it did not cause any historical champion.

Population-mean fitness slopes over generations `20..39` were:

| Seed | Mean slope | Generation-20 mean | Generation-39 mean |
|---:|---:|---:|---:|
| 7 | -0.00110 | 0.4859 | 0.4645 |
| 17 | -0.00035 | 0.4786 | 0.4750 |
| 27 | +0.00044 | 0.4765 | 0.4828 |

Only seed 27 retained positive late population progress. The declared gate
required at least two seeds.

Historical champions were highly active and profitable but completely
front-loaded:

| Seed | Champion generation | Fitness | Attack fraction | Prey rate | Net attack profit | Hidden nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 3 | 0.7438 | 0.0935 | 0.0553 | +1915.6 | 0 |
| 17 | 1 | 0.6209 | 0.2084 | 0.0837 | +1230.6 | 0 |
| 27 | 2 | 0.6009 | 0.0959 | 0.0311 | +670.5 | 0 |

Every historical champion used the original 56-connection, zero-hidden-node
topology. A fourfold larger pool therefore improved the chance of discovering
a strong simple policy immediately; it did not provide evidence of sustained
complexification or progressive refinement.

### Held-out validation

Pairwise crossplay of checkpoints `0,10,20,30,39` on held-out seeds produced
21/30 later-over-earlier wins, down from 26/30 for the matched population-16
arena. Per-seed counts were `9/10, 6/10, 6/10`.

The final seed-7 checkpoint beat every earlier checkpoint. Seed 17's final
checkpoint lost to generations 10 and 30; seed 27's final checkpoint lost to
generations 10 and 20. The larger strategy pool therefore did not improve
historical ordering or retention.

Historical population-64 champions were also matched directly against the
corresponding population-16 arena champions in both founder slots:

| Seed | P64 mean survival | P16 mean survival | P64 case wins |
|---:|---:|---:|---:|
| 7 | 0.4302 | 0.3597 | 8/8 |
| 17 | 0.3367 | 0.3908 | 0/8 |
| 27 | 0.4618 | 0.4107 | 8/8 |

Scaling found substantially stronger frozen strategies in two seeds and a
substantially weaker one in the third. This is a genuine but seed-dependent
search-breadth benefit, not sustained evolutionary progress.

## Interpretation

The population-64 arena is computationally practical and maintains much more
genotypic/species diversity than population 16. Its added genomes mainly act as
more draws from the initial simple-controller search distribution. The
historical maxima all occur before structural innovation has time to matter,
and later populations move toward less extreme strategies rather than building
on those maxima.

The run rejects the idea that horizontal population scaling alone delays the
plateau. It strengthens the case for using the arena as a broad proposal
generator, but also strengthens the need for a second-stage retention or
grounding mechanism. The next algorithmic experiment should not increase the
arena to 128. It should compare arena-only selection against arena screening
followed by pairwise reranking of a small survivor set at constant total
compute.

## Artifacts

`artifacts/research/runs/completed/2026-07-15-energy-stealing-arena-p64-40g/`
contains the manifest, results, champion worlds, live progress/ETA logs, and
held-out crossplay matrices.
