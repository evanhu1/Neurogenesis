# 2026-07-14-attack-cost-calibration-1000t-20g

Status: completed; cost 5 advances to a longer matched test

## Question

At a common uncensored 1,000-tick horizon, does lowering attack-attempt cost
from 10 to 5 preserve the successful suppression of indiscriminate attack spam
while allowing economically profitable predation to enter high-fitness parts of
the evolving population?

## Hypothesis

With transfer fixed at 40, lowering attempt cost from 10 to 5 lowers the
best-case precision threshold for covering attempt costs from 25% to 12.5%.
The previous cost-10 run occasionally discovered attack precision in this range
but never produced positive private net attack energy among champions. Cost 5
should therefore make selectively emitted attacks viable without restoring the
high-frequency, strongly negative attack output seen in generation zero.

The mechanism is falsified if cost 5 remains plant-only, restores attack spam,
or improves attack activity without producing positive private attack balance
in competitively strong genomes.

## Contract

- Code revision: captured independently by each `cli batch` manifest.
- Canonical config: `config/world.toml`.
- Control: `attack_attempt_cost=10`, `attack_energy_transfer=40`.
- Treatment: `attack_attempt_cost=5`, `attack_energy_transfer=40`.
- Evolutionary seeds: `7,17,27` in both arms.
- Training world seeds: `11,29,47` in both arms.
- Held-out world seeds: none in this screening run.
- Population: 48 genomes.
- Generations: 20 (`0..19`).
- Episode horizon: 1,000 ticks.
- Evaluation: contemporary-only triads, 12 memberships per genome, 24 opponent
  exposures, and 36 scored cases per genome.
- World: canonical 50x50 with 102 founders, 34 per lineage.
- Objective: `survival_times_relative_advantage`, ordinary mean aggregation.
- Evaluation workers: one shared machine budget of 14; the two arms ran
  concurrently with seven workers each, allocated 3/2/2 across their seeds.
- Simulator work: 11,520 evaluator worlds and 11.52 million world ticks per
  evolutionary seed; 34.56 million ticks per arm and 69.12 million ticks total.
- Artifact directory:
  `artifacts/research/runs/completed/2026-07-14-attack-cost-calibration-1000t-20g/`.

No mutation, selection, topology, sensor, action, food, founder, world-size,
seed, case-budget, or objective setting differed between arms. A structural
diff of the two validated summary contracts contained only the attack-attempt
cost at the final and replay scenario copies.

## Measurements

The explicit tail was generations `10..19` inclusive.

Primary mechanism endpoints:

- Per-genome private net attack balance: attack transfer received minus attack
  transfer lost minus attempt cost.
- Prevalence of positive-net-attack genomes among the top fitness quartile in
  the late population. Population-wide mean net attack balance is not a success
  endpoint because conserved transfers cancel across symmetrically evaluated
  lineages, leaving attempt cost structurally negative.
- Attack precision, action fraction, transfer income share, kills, and distinct
  victims for any profitable high-fitness genome.

Competence and integrity endpoints:

- Champion, population mean, and median absolute survival.
- Champion and population end-survival fractions as the 1,000-tick censoring
  audit.
- Total energy accumulated, net energy profit, plant and prey success rates,
  action effectiveness, plant capture, and spatial coverage.
- Per-seed treatment-minus-control effects and full population distributions.
- Identical resolved contracts apart from attack-attempt cost, schema validity,
  generation completeness, and deterministic artifact provenance.

## Decision rule

Advance cost 5 to the full 40-generation baseline only if all of the following
hold:

1. In at least two of three seeds, a generation-15-to-19 top-fitness-quartile
   genome has positive private net attack balance and nontrivial attack evidence
   rather than a single negligible transfer.
2. Late profitable-predator prevalence is greater than in the matched cost-10
   control.
3. Cost 5 does not restore spam: late population attack fraction remains below
   5%, and attack costs do not drive population net energy profit negative.
4. Treatment late mean absolute survival is no more than 0.05 below control in
   at least two seeds.
5. The 1,000-tick horizon is not materially censored among the compared
   populations.

Reject cost 5 if it only increases attack counts or gross transfer while private
net balance remains negative. If both arms remain plant-only, keep the simpler
cost mechanism and test plant availability as a separate causal lever.

## Commands and provenance

Both plans resolved to 576 evaluator worlds per generation, 36 cases and 24
opponent exposures per genome, 11,520 worlds and 11.52 million ticks per seed.
The arms were launched concurrently as:

```bash
./target/release/cli batch \
  --experiment control-cost10 --seeds 7,17,27 --total-workers 7 \
  --out-dir artifacts/research/runs/active/2026-07-14-attack-cost-calibration-1000t-20g -- \
  --population 48 --generations 20 --horizon 1000 \
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --scenarios baseline --founders 102 \
  --objective survival_times_relative_advantage \
  --set attack_attempt_cost=10 --set attack_energy_transfer=40

./target/release/cli batch \
  --experiment treatment-cost5 --seeds 7,17,27 --total-workers 7 \
  --out-dir artifacts/research/runs/active/2026-07-14-attack-cost-calibration-1000t-20g -- \
  --population 48 --generations 20 --horizon 1000 \
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --scenarios baseline --founders 102 \
  --objective survival_times_relative_advantage \
  --set attack_attempt_cost=5 --set attack_energy_transfer=40
```

Each arm was summarized with `--tail 10:19`. Both manifests validated with
schema-22 results and schema-2 summaries. Wall time was approximately 242
seconds; paired arm timings differed by less than 1.1 seconds at every seed.

## Result

### Cost 5 made predation compatible with high fitness

Across generations 15 through 19, the control had zero positive-net-attack
genomes in its 180 top-quartile genome-generation slots. The treatment had six:
two in seed 7 and four in seed 27. Five had substantial energy share, multiple
victims, or kills; the remaining seed-7 case was a negligible `+1.83` balance.

The clearest treatment organisms were:

| Seed / generation | Fitness rank | Net attack | Attack income share | Attack fraction | Mean kills | Mean distinct victims |
|---|---:|---:|---:|---:|---:|---:|
| 7 / 19 | 8 | +12.2 | 13.5% | 0.21% | 1.75 | 2.17 |
| 27 / 15 | 2 | +52.1 | 32.9% | 0.60% | 4.42 | 6.50 |
| 27 / 17 | 4 | +142.9 | 67.3% | 1.40% | 12.42 | 15.08 |
| 27 / 17 | 6 | +272.7 | 48.0% | 0.82% | 5.67 | 7.50 |
| 27 / 18 | 1 | +106.8 | 40.1% | 0.96% | 7.25 | 9.58 |

The seed-27 generation-18 predator was the run's historical champion, not a
low-fitness curiosity. Its private accounting was `1,741` received, `1,087`
lost, and `547` attempt cost. Its 79% repeat-hit fraction accompanied attacks on
roughly ten distinct victims and seven kills per case, so the aggregate trace is
consistent with multi-hit draining of several prey rather than one cost-free
reciprocal pair loop. A later behavioral replay is still needed to distinguish
pursuit from stationary contact exploitation.

Cost 10 did contain five positive-net-attack genome slots in the complete late
populations, but all fell below the top fitness quartile. Cost 5 therefore did
not create the first possible positive transfer balance; it made that balance
compatible with competitive fitness in two seeds.

### Spam remained suppressed

Late population attack fractions for control versus treatment were:

| Seed | Cost 10 | Cost 5 |
|---:|---:|---:|
| 7 | 0.19% | 0.54% |
| 17 | 0.21% | 0.58% |
| 27 | 0.61% | 1.04% |

All remained far below the 5% gate and far below generation-zero spam. Every
treatment population retained positive net energy profit. The lower cost did
increase attack allocation, but did not restore indiscriminate attack output.

### Competence cost was small at the population level but real among champions

| Seed | Tail mean survival, cost 10 | Tail mean survival, cost 5 | Treatment minus control |
|---:|---:|---:|---:|
| 7 | 0.2774 | 0.2752 | -0.0022 |
| 17 | 0.2875 | 0.2777 | -0.0098 |
| 27 | 0.2745 | 0.2676 | -0.0069 |

Thus all seeds passed the preregistered `-0.05` population-competence margin.
However, cost-10 final best survival exceeded cost 5 by 0.041, 0.043, and 0.054,
and treatment population net energy profit was lower in all three seeds. Cost 5
created a viable predatory niche but diverted some search and energy away from
the stronger plant-foraging solutions found by cost 10.

All control population-survival tail slopes were positive. Treatment seeds 7
and 17 were positive, while seed 27 was approximately flat (`-0.00009` per
generation). The screen therefore does not show that predation improves the
aggregate adaptation rate.

### The 1,000-tick horizon passed

Final champion end-survival was at most 0.74% in the control and 0.41% in the
treatment. Population end-survival was at most 0.022%. Unlike the 500-tick run,
this horizon retains useful ranking resolution at generation 20.

## Interpretation and next decision

Cost 5 passes all preregistered screening gates and advances, but it is not yet
the accepted baseline.

- The intended mechanism worked in two seeds: profitable predation entered the
  high-fitness population, including one historical champion.
- Seed 17 remained a foraging regime, demonstrating strong basin dependence.
- Predation remained sparse and dissipative rather than reverting to spam.
- Absolute population competence remained close to control, but cost-10
  champions and population energy profit were consistently stronger.
- The 1,000-tick horizon is the correct current evaluation horizon.

The next experiment should extend both matched arms to 40 generations rather
than running cost 5 alone. The control remains necessary because the screen
revealed a genuine trophic-diversity versus competence trade-off. The longer run
must ask whether profitable predators persist and spread, whether foragers adapt
against them, and whether the initial survival disadvantage closes or worsens.
Only then should one cost become the canonical baseline. No food-density,
objective, sensor, or action change should be combined with that comparison.
