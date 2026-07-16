# Population-64 shared arena over 1,000 generations

Status: aborted as an evaluator diagnostic. The run completed, but its
clone-heavy shared arenas and population-order-dependent founder assignment
violate the repaired one-founder-per-genome contract. Its late collapse helped
identify those bugs; its longitudinal scores are not competence evidence.

## Question

Does the population-64 shared arena eventually resume cumulative progress after
the front-loaded 40-generation phase, or does it continue cycling, forgetting,
and replacing early simple-controller champions?

## Contract

- Evolutionary seeds `7,17,27`; generations `0..999`.
- Population 64 under the default shared-population evaluator.
- Eight fixed training world seeds: `11,29,47,61,73,89,107,127`.
- Energy stealing, horizon 500, attack cost 1, transfer 20, no plants/Eat.
- Width 142, 768 founders, 12 founders per lineage.
- Absolute-survival mean fitness; unchanged NEAT mutation, crossover,
  speciation, and selection.
- Complete population checkpoint every 10 generations; compact champion and
  metrics every generation.

## Measurements

Summarize the full run and tails `500..999` and `800..999`. Record historical
champion generation, final champion strength, population mean/median, attack
profit, strategy/activity, species, topology, and whether late improvements
survive frozen pairwise checkpoint crossplay. Materialize generation 999 from
the seed with the strongest final-generation champion into a world for visual
inspection.

This run is promising only if gains occur materially after generation 40 and
late checkpoints systematically improve rather than merely recovering toward
old maxima. A historical champion found early, additional species, or topology
growth without retained competence does not qualify.

## Command

```text
cli batch --experiment 2026-07-15-energy-stealing-arena-p64-1000g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 64 --generations 1000 --population-checkpoint-interval 10 \
  --horizon 500 --task energy_stealing \
  --world-seeds 11,29,47,61,73,89,107,127 \
  --world-width 142 --founders 768 --cvar 1
```
