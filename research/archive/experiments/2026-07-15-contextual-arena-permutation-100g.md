# Contextual arena permutation baseline — 100 generations

Status: completed; the evaluator repair passed, and the run found transferable
adaptation but not reliably cumulative progress.

## Question

After removing food ecology, historical-score assumptions, species stagnation,
and population-order-dependent founder assignments, does contemporary
shared-arena NEAT still produce active predatory strategies over 100
generations?

This is not a monotone-fitness experiment. Same-generation scores rank genomes
only inside the arena population that produced them. Longitudinal behavioral
claims require frozen checkpoint crossplay.

## Treatment contract

- Evaluation task: plant-free energy stealing, the only task.
- Evaluation topology: shared population.
- Population: 64 genomes.
- Generations: 100 (`0..99`).
- Horizon: 500 ticks.
- World seeds: the fixed inclusive range `11..74` (64 cases).
- World width: 42, preserving approximately the former founder density.
- Founders: 64, exactly one organism per genome per arena.
- Attack cost / transfer: 1 / 20.
- Motor interface: compositional orientation, movement, and attack commands;
  pursuit can move and attack in the same tick.
- Selection aggregation: all cases (`cvar = 1`).
- Evolution seeds: `7,17,27`.
- Founder assignment: canonical genome ordering plus one distinct deterministic
  cyclic slot rotation per evaluation case. Across 64 cases every genome must
  occupy every founder-ID slot exactly once, and input population-vector order
  must not change scores.
- No species stagnation and no cross-generation champion comparison.

## Required pre-run audits

1. Reordering an identical frozen population must produce identical per-genome
   evaluation summaries.
2. Movement collision, dependency-cycle, escaping-target, mutual-attack, and
   multiple-attacker tests must pass with exact energy closure.
3. Active configuration and wire schemas must contain no food, plant, Eat, or
   trophic-role fields.
4. `cargo test --workspace`, `make lint`, and `make fmt` must pass.

## Signs-of-life readout

Do not use population mean or a slope of contemporary scores. Report, within
each checkpoint generation:

- top contextual score and rank spread only as selection diagnostics;
- attack attempts, precision, repeat hits, distinct victims, kills, and net
  stolen energy;
- movement, turning, attack, idle, spatial coverage, and end survivors;
- topology/species diversity as search diagnostics.

Then crossplay checkpoints `0,20,40,60,80,99` on held-out seeds
`101,131,151,181,191,211,223,241`. Evidence of life means later checkpoints
introduce at least one reproducible active strategy that wins frozen matchups
through movement and energy theft rather than a favorable founder slot. No
monotone or open-endedness claim is implied by this 100-generation screen.

## Execution

Preflight resolved 64 evaluation worlds and 4,096 scored lineage cases per
generation: one founder for each of 64 genomes, 64 fixed world seeds, and a
distinct deterministic slot rotation per case. Three evolutionary seeds ran in
parallel on 14 workers and finished in 32.1--34.8 seconds each.

```sh
cli batch --out-dir artifacts/research/runs/active \
  --experiment 2026-07-15-contextual-arena-permutation-100g \
  --seeds 7,17,27 --total-workers 14 -- \
  --population 64 --generations 100 \
  --population-checkpoint-interval 20 --horizon 500 --cvar 1

cli crossplay seed-N.result.json.zst \
  --checkpoints 0,20,40,60,80,99 --horizons 500 \
  --world-seeds 101,131,151,181,191,211,223,241 --cvar 1 \
  --out seed-N.crossplay.json
```

The first crossplay attempt exposed and corrected a validation bug: the old
pairwise assay claimed a two-genome pool while spawning 32 clones of each
genome. Schema 5 crossplay now uses one founder per genome on a density-matched
width-7 world. The invalid clone-heavy outputs were replaced.

## Verification

- Population-vector permutation invariance passed end to end: the same frozen
  genomes receive bit-identical scores after reordering.
- Across 64 cases, every genome occupies every founder/ID slot exactly once.
- All 22 deterministic movement/predation tests passed, including swaps,
  dependency cycles, collision ties, escape, move-plus-attack pursuit, mutual
  attacks, lethal cancellation, multiple attackers, and energy-ledger closure.
- `cargo test --workspace`, `make fmt`, `make lint`, web typecheck/build, and
  `git diff --check` passed.
- Active schemas contain no food entity, plant policy, Eat action, trophic
  metric, or persisted champion-ranking path.

## Results

Contemporary winner scores are intentionally not compared across generations.
The frozen held-out crossplay produced this chronological result:

| Evolution seed | Later-checkpoint wins | Broadest checkpoint | Wins vs other checkpoints |
|---:|---:|---:|---:|
| 7 | 12 / 15 | 80 | 5 / 5 |
| 17 | 11 / 15 | 40 | 5 / 5 |
| 27 | 10 / 15 | 40 | 5 / 5 |
| **Total** | **33 / 45 (73.3%)** | -- | -- |

Every final checkpoint beat generation 0. Averaged against all five other
checkpoints, frozen survival changed as follows:

| Seed | Generation 0 | Generation 99 | Change | Final wins |
|---:|---:|---:|---:|---:|
| 7 | 0.240 | 0.408 | +70% | 4 / 5 |
| 17 | 0.205 | 0.481 | +135% | 3 / 5 |
| 27 | 0.265 | 0.418 | +57% | 3 / 5 |

This is real adaptation rather than founder-slot luck. Later checkpoints move,
attack, steal energy, and reproducibly defeat the initial random strategies on
held-out layouts. The direct assay also distinguishes multiple strategies:

- Seed 7's generation 80 is the broadest competitor despite averaging zero
  kills, low attack precision, high spatial coverage, and the least-negative
  energy balance. It is principally an evasion/survival policy. Generation 99
  is more predatory but loses directly to generation 80.
- Seed 17's generation 40 beats every checkpoint. Generations 80 and 99 have
  similar broad mean survival but lose the direct matchup to generation 40.
- Seed 27's generation 40 beats every checkpoint. Generation 99 recovers over
  generation 80 but remains weaker than generations 40 and 60.

The contemporary arenas explain why raw training scores fell from unusually
high generation-0 winners. A random population supplies exploitable prey, so an
early attacker can live beyond the 250-tick founder energy budget. As defense
and movement spread, that prey disappears. By generation 99 the three arena
winners average only 211--233 alive ticks, below the passive 250-tick budget,
and have negative private attack-energy balances. The population has become
harder to exploit, while selection increasingly rewards evasion and low-loss
survival. That is a coevolutionary change, not an absolute regression, but it
also suppresses the predatory gradient.

## Decision

The repaired arena passes the requested signs-of-life screen and is retained as
the efficient contemporary evaluator. The stronger claim is rejected: this is
not reliably progressive past generation 40. The best frozen checkpoint occurs
at generation 40 in two seeds and generation 80 in one, and all three final
checkpoints forget at least one earlier strategy.

The next causal experiment should address the defense-dominant equilibrium,
not add more generations or clones. A principled treatment is to shorten the
initial energy runway while preserving the 1/20 attack economy, so survival
beyond a much earlier boundary requires successful energy theft. It must be
matched against this baseline and pass held-out one-founder crossplay; merely
increasing attack counts does not count.
