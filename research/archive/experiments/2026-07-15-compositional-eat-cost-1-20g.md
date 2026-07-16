# 2026-07-15-compositional-eat-cost-1-20g: one-energy Eat attempts

Status: completed; anti-spam mechanism confirmed, canonical-default decision
deferred

## Question

Does charging one energy for every Eat command eliminate the blanket `move +
Eat` policy produced by compositional motor control, while preserving adaptive
plant acquisition?

## Hypothesis

With free Eat attempts, emitting Eat on every moving tick is a costless lottery
ticket. A one-energy cost should create pressure to emit Eat only when the
controller has evidence that a plant is reachable. The mechanism is rejected
if Eat frequency stays near one, Eat precision does not rise across seeds, or
foraging disappears instead of becoming more selective.

## Contract

- Code revision: `9c604c581509f5e7ebc2c06dd95035b752b3d266`
- Tracked source patch SHA-256 at batch launch:
  `86ca2279e260658ac01adcb0d4e72bad1a9248922968b069af8d125f31560106`
- Canonical config: `config/world.toml`
- Control: completed zero-cost compositional arm from
  `2026-07-14-compositional-motor-commands`
- Treatment: identical compositional control with `eat_attempt_cost=1`
- Shared overrides: `attack_attempt_cost=5`,
  `compositional_actions_enabled=true`
- Evolutionary seeds: `7,17,27`
- Training world seeds: `11,29,47`
- Population: 48 genomes
- Generations: 20 (`0..19`)
- Episode horizon: 1,000 ticks
- Evaluation: 12 three-lineage memberships per genome, 24 opponent exposures,
  36 scored cases per genome
- Founders/world: 102 (34 per lineage), 50x50
- Objective: survival times relative advantage, mean aggregation (`cvar=1`)
- Evaluation workers: 14 machine-wide, allocated `5,5,4`
- Per arm: 34,560 evaluator worlds and 34,560,000 world ticks across seeds
- Treatment result schema: 24
- Artifact directory:
  `artifacts/research/runs/completed/2026-07-15-compositional-eat-cost-1-20g/`

The zero-cost control used the same population, seeds, ecology, evaluator,
objective, and compute contract. Cost zero is behaviorally a no-op in the new
implementation; schema-24 adds exact Eat-cost accounting but does not change
the zero-cost turn result.

## Measurements

The declared tail is generations `15..19`, averaged within each evolutionary
seed and then across the three seeds. Eat frequency is the fraction of organism
ticks emitting Eat. Eat precision is plant consumptions divided by Eat
commands. The primary gate requires a seed-consistent reduction in Eat
frequency and increase in Eat precision. Plant-consumption rate and plant
capture test whether the treatment learned targeting instead of merely
suppressing interaction. Survival, end survivors, commands/tick, multi-command
frequency, and net energy profit measure collateral competence costs.

No crossplay was run: this screening question concerns within-policy command
allocation and interaction efficiency, which are observed directly in every
training case. It does not make a historical-retention claim.

## Decision rule

Confirm the mechanism only if all three seeds materially reduce champion Eat
frequency, all three increase Eat precision, and all three retain or improve
plant-consumption rate. Do not promote cost 1 to the canonical default from
this screen alone if it eliminates foraging, causes broad extinction, or
produces a large unexplained competence collapse.

## Commands and provenance

```text
cli plan --population 48 --generations 20 --horizon 1000 \
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --founders 102 --world-width 50 \
  --objective survival_times_relative_advantage --cvar 1 --workers 14 \
  --set attack_attempt_cost=5 --set eat_attempt_cost=1 \
  --set compositional_actions_enabled=true

cli batch --experiment 2026-07-15-compositional-eat-cost-1-20g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 48 --generations 20 --horizon 1000 \
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --founders 102 --world-width 50 \
  --objective survival_times_relative_advantage --cvar 1 \
  --set attack_attempt_cost=5 --set eat_attempt_cost=1 \
  --set compositional_actions_enabled=true

cli summarize \
  artifacts/research/runs/active/2026-07-15-compositional-eat-cost-1-20g \
  --tail 15:19
```

All seeds succeeded in 118–132 seconds. The manifest contains exact commands,
resolved contracts, source identity, logs, result hashes, and artifact hashes.

## Result

The mechanism passed in every seed. Tail champion results were:

| Seed | Eat fraction, cost 0 | Eat fraction, cost 1 | Eat precision, cost 0 | Eat precision, cost 1 | Plant rate, cost 0 | Plant rate, cost 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.816 | 0.152 | 0.056 | 0.358 | 0.046 | 0.054 |
| 17 | 0.943 | 0.242 | 0.050 | 0.280 | 0.047 | 0.068 |
| 27 | 0.792 | 0.251 | 0.064 | 0.262 | 0.051 | 0.064 |

Across seeds, tail champions changed as follows:

| Metric | Cost 0 | Cost 1 | Treatment effect |
|---|---:|---:|---:|
| Eat commands / organism tick | 0.850 | 0.215 | -74.7% |
| Eat precision | 0.057 | 0.300 | 5.3x |
| Plant consumptions / organism tick | 0.048 | 0.062 | +30.2% |
| Actionable plant capture | 0.500 | 0.566 | +13.3% |
| Commands / organism tick | 2.688 | 1.894 | -29.5% |
| Multi-command tick fraction | 0.966 | 0.772 | -0.195 |
| Absolute survival fraction | 0.730 | 0.620 | -15.1% |
| End-survivor fraction | 0.511 | 0.399 | -21.9% |
| Net energy profit | 23,156 | 21,073 | -9.0% |

The population-wide result agreed. Mean Eat frequency fell from 0.813 to 0.231
and mean per-genome Eat precision rose from 0.050 to 0.225. Population survival
fell from 0.563 to 0.425 and population net energy profit from 14,382 to 9,630,
showing that the cost also widened the gap between adapted and poorly adapted
genomes.

The trajectory identifies an evolutionary response rather than an incidental
late checkpoint. By generation 10, every treatment seed had selected a regime
with at most 44% Eat and at least 18% precision; by generations 15–19, seed-7
champions stabilized near 15% Eat with 36–38% precision, and seeds 17/27
converged near 23–26% Eat with 26–28% precision. Multi-command behavior remained
common, so the intervention did not undo compositional control.

## Interpretation and next decision

One energy decisively eliminates blanket Eat spam. The zero-cost champions sat
near the direct break-even point: roughly 5% success times 20 energy per plant
equals one expected energy per Eat attempt. Charging one energy therefore
removes the expected private return of indiscriminate attempts. Evolution
responded exactly as predicted: fewer attempts, roughly fivefold greater
precision, and more plant acquisition per organism tick.

This is not merely action suppression. Treatment champions captured more of the
plant supply and consumed plants faster in all three seeds. The remaining high
multi-command frequency is compatible with useful composition such as turning
while moving and selectively eating.

Cost 1 is nevertheless a strong ecological change. It lowers absolute survival
and especially population-mean profit, while champions pay approximately
3,500–4,900 Eat energy per evaluated case. That may be desirable task
difficulty, but this 20-generation screen cannot distinguish productive
selection from a search landscape that becomes unnecessarily harsh at longer
horizons.

Keep the configurable mechanism and retain the canonical default at zero for
now. The next decision experiment should extend cost 1 beyond generation 30
and use held-out crossplay to test cumulative competence and retention. If the
population gap grows or progress stalls, the materially cleaner calibration is
a fractional/fixed-point Eat cost below one rather than weakening the causal
principle or adding blanket movement costs.

Verification: `make fmt`, `make lint`, the release build, and web build/typecheck
passed. `cargo test --workspace` retained the known stale failure in
`lethal_attack_conserves_energy_without_spawning_food` (expected 59, actual 49
after the canonical 10-energy attack-attempt cost); the other 15 world-sim
tests and every other workspace test passed.
