# 2026-07-14-costly-predation-baseline-40g

Status: completed; rejected as working baseline

## Question

What evolutionary regime does the simplified costly-predation economy produce,
and does the reduced 500-tick, three-world-seed evaluator preserve a useful
survival gradient over 40 generations?

## Hypothesis

Charging every attack attempt 10 energy while transferring up to 40 energy on a
successful hit should make indiscriminate attack output strongly unprofitable,
make reciprocal attack exchange negative, and preserve predation only when
controllers target organisms accurately enough to pay for misses. Evolution
should consequently favor mobile foraging, selective predation, or coherent
mixtures rather than cost-free attack spam.

The treatment is too harsh if predation disappears across all three evolutionary
seeds before accurate attacks can be discovered. It is too weak if populations
retain low-precision attack output with persistently negative net energy profit.

## Contract

- Canonical config: `config/world.toml`.
- Evolutionary seeds: `7,17,27`.
- Training world seeds: `11,29,47`.
- Population: 48 genomes.
- Generations: 40 (`0..39`).
- Episode horizon: 500 ticks.
- Evaluation: contemporary-only triads, 12 memberships per genome, 24 opponent
  exposures, and 36 scored cases per genome.
- Evaluator worlds: 576 per generation, 23,040 per evolutionary seed.
- World: 50x50, 102 founders, 34 per lineage.
- Starting energy: 250.
- Plants: 20% cells, 20 energy, 200-tick regrowth.
- Predation: every attempt costs 10; a hit transfers up to 40 from victim to
  attacker.
- Objective: `survival_times_relative_advantage`.
- Plasticity and leaky hidden state: disabled.
- Machine worker budget: 14, allocated 5/5/4 by `cli batch`.
- Artifact root:
  `artifacts/research/runs/completed/2026-07-14-costly-predation-baseline-40g/`.

## Measurements

Primary:

- Champion, population mean, and population median absolute survival.
- Tail survival slopes and generation-20-to-39 deltas.
- End-survival fraction as the 500-tick censoring gate.
- Total energy accumulated and net energy profit for champion and population.

Predation mechanism:

- Attack fraction and action effectiveness.
- Successful-hit precision over all attack attempts.
- Attempt cost, energy received, energy lost, and net attack balance.
- Kills and distinct victims.
- Continuous plant/prey consumption rates, plant capture, and spatial coverage.

Search health:

- Population dispersion, opponent-score sensitivity, species, and expressed
  topology, treated as diagnostics rather than capability targets.

## Decision rule

Accept this as the new working baseline if:

1. the 500-tick horizon is not materially censored;
2. at least two seeds show positive population-level survival progress;
3. attack costs suppress indiscriminate low-value output without eliminating
   all successful predation; and
4. net energy profit and inspected behavior explain survival rather than gross
   transfer cycling.

If predation disappears in all seeds, retain the attempt-cost mechanism but
reduce the cost from 10 to 5 before changing sensors or rewards. If survival
saturates at 500, rerun the complete contract at one common longer horizon.

## Commands and provenance

The batch was run as:

```bash
./target/release/cli batch \
  --experiment 2026-07-14-costly-predation-baseline-40g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 48 --generations 40 --horizon 500 \
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --scenarios baseline --founders 102 \
  --objective survival_times_relative_advantage
```

The persisted `manifest.json` records the resolved schema-22 contract, source
revision and dirty patch identity, executable checksum, worker schedule, elapsed
times, and artifact checksums. The compact schema-2 report was regenerated with:

```bash
./target/release/cli summarize \
  artifacts/research/runs/active/2026-07-14-costly-predation-baseline-40g \
  --tail 20:39
```

## Result

The run is a strong positive result for basic adaptation and for suppressing
attack spam, but it fails two of the four preregistered baseline gates.

| Seed | Best absolute survival, g0 -> g20 -> g39 | Mean absolute survival, g0 -> g20 -> g39 | End survivors at g39, champion / population | Net energy profit at g39, champion / population |
|---:|---:|---:|---:|---:|
| 7 | 0.505 -> 0.661 -> 0.759 | 0.361 -> 0.558 -> 0.595 | 22.3% / 4.3% | 5,464 / 1,759 |
| 17 | 0.476 -> 0.729 -> 0.861 | 0.276 -> 0.572 -> 0.684 | 40.8% / 13.7% | 7,412 / 3,572 |
| 27 | 0.500 -> 0.799 -> 0.882 | 0.333 -> 0.602 -> 0.631 | 55.9% / 8.1% | 8,773 / 2,519 |

All three generation-20-to-39 population survival slopes were positive:
`+0.00191`, `+0.00678`, and `+0.00194` absolute-survival units per generation.
The corresponding champion slopes were `+0.00521`, `+0.00724`, and `+0.00303`.
This is late within-evaluator progress, although individual generation steps
remain noisy and it is not frozen historical crossplay evidence.

### Immediate attack-spam purge

Generation zero populations spent 7.8%, 17.6%, and 10.1% of actions attacking.
Their mean net energy profits were respectively `-2,364`, `-3,844`, and
`-2,859`. By generation two, attack fractions had fallen to 0.27%, 0.96%, and
1.31%. The largest population-survival jump in every seed occurred at generation
one even though action effectiveness fell slightly. The immediate gain is
therefore mainly deletion of energetically catastrophic attack output, not a
new sensory-motor harvesting strategy.

### Predation was eliminated as a profitable strategy

No generation in any seed had positive champion or population net attack
energy. At a cost of 10 and maximum transfer of 40, an attacker needs at least
25% precision merely to cover attempt costs before accounting for energy lost
while being attacked. Maximum population precisions were only 7.2%, 10.8%, and
18.2%.

Seed 27 briefly produced a champion with 34.3% precision at generation five,
but it attacked on only 0.056% of actions and still lost 24.4 net attack energy
after victim losses and attempt cost. This is sparse opportunism, not an
economically self-supporting predator.

By generation 39, champion attack income contributed only 1.31%, 0.18%, and
1.27% of total acquired energy. Champion prey-success rates were only 0.73%,
0.11%, and 0.65% of their plant-success rates. The three-lineage evaluator had
become predominantly a shared-plant competition, not a predator-prey ecology.

### Later progress was real plant-foraging improvement

After the early attack purge, normalized behavioral measures improved rather
than only lifetime-scaled totals. From generation 20 to 39, champion plant rates
rose in all seeds, ending at 2.18%, 2.54%, and 3.22% of actions. Champion action
effectiveness ended at 8.0%, 31.5%, and 34.1%; spatial coverage ended at 24.0%,
75.2%, and 77.2%. Net profit per alive-tick also rose throughout the sampled
checkpoints. Gross acquisition and net profit agree because reciprocal attack
transfer is now small and its dissipative costs remain visible; energy cycling
does not explain the survival gain.

Seeds 17 and 27 discovered broad, mobile harvesting policies. Seed 7 found a
slower, more local forager. Species counts rose from one to five, five, and six,
and final champions expressed one, two, and three hidden nodes, so search and
complexification remained active. These topology associations are descriptive,
not causal ablations.

### The 500-tick horizon is materially censored

Champion end-survival first exceeded 5% at generations 14, 13, and 14. At the
final generation, 22.3%, 40.8%, and 55.9% of champion founders were still alive
at the cutoff. Population end-survival also reached 13.7% and 8.1% in seeds 17
and 27. Those organisms receive identical credit for any survival beyond tick
500, so the evaluator is already losing resolution precisely among the genomes
selection most needs to distinguish.

## Interpretation and next decision

Do not adopt this contract as the new working baseline.

- Gate 1 fails: 500 ticks is materially censored by the middle of the run.
- Gate 2 passes: all three seeds have positive late population-survival slopes.
- Gate 3 fails: costly attacks suppress indiscriminate output but also eliminate
  profitable predation in every seed.
- Gate 4 passes: survival is explained by plant-derived net profit and normalized
  harvesting behavior, not gross attack-transfer cycling.

Retain the simple attempt-cost/transfer mechanism. The next causal calibration
should evaluate attack cost 10 versus the preregistered fallback cost 5 under a
common 1,000-tick horizon, leaving transfer 40 and the ecology fixed. Using the
same longer horizon in both arms isolates the cost lever while restoring room to
measure survival. If cost 5 also converges to plant-only strategies, the next
separate lever is plant availability; prior 10%-plant evidence makes that more
plausible than further reward inflation. The survival-objective ablation should
wait until this evaluator baseline is accepted, because running it at the now
invalid 500-tick/cost-10 contract would answer the wrong question.

This experiment demonstrates progressive adaptation over 40 generations, not
open-endedness: it has no frozen historical-retention assay, no ecology
perturbation, and a censored tail.
