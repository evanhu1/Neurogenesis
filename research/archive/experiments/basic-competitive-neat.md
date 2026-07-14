# Basic competitive NEAT pilot

> Historical schema-12/13 evidence. Schema 14 supersedes the sampled
> five-genome evaluator with the balanced pairwise contract documented in
> [robust-pairwise-neat.md](robust-pairwise-neat.md).

## Algorithm

This is the minimal generation-level competitive loop. Each NEAT generation is
copied into a read-only genome snapshot. For every candidate and deterministic
evaluation case, the evaluator samples four other genomes from that snapshot
and creates one bounded mixed-founder world. Only the candidate founders'
alive-ticks contribute to that candidate's fitness. NEAT then performs its
ordinary selection, crossover, and mutation to produce the next genome pool.

Generation zero now uses the schema-13 fully connected NEAT default. Every
genome starts with no hidden nodes and the same direct connection from every
active sensor to every active action, while connection weights and action
biases are independently randomized. With predation enabled, that is 9 sensors
× 5 actions = 45 initial connections. Hidden computation is an acyclic graph
evaluated in topological order within the current tick, with plasticity and
leaky state both disabled.

There is no in-world reproduction, opponent renewal, historical archive, or
relative-opponent scalar. Competition happens only because current-generation
genomes share the bounded ecology.

The raw score reported below is the mean candidate-founder alive-ticks per
case. Each case has 12 candidate founders and lasts 2,000 ticks, so the maximum
is `12 * 2,000 = 24,000` alive-ticks. The `35×35`, 60-founder world preserves
approximately the same founder density as the earlier `25×25`, 30-founder
pilot while providing twice as many organisms per bounded match.

## Earlier 20-generation command shape

The following command produced the historical schema-12 table below. The
current schema-13 binary uses the fully connected initialization and therefore
does not reproduce the old ten-connection population from this command alone;
the old artifacts and hash are retained for comparison.

Run once for each outer-loop seed in `7,42,123,2026,9001`, changing `--seed`
and `--out-dir` together:

```bash
./target/release/sim-cli neat \
  --seed 7 \
  --population 24 \
  --generations 20 \
  --episode-horizons 2000 \
  --opponents 4 \
  --world-seeds 11,29 \
  --no-audit \
  --no-holdout \
  --scenarios baseline \
  --workers 4 \
  --scale 35,60 \
  --set predation_enabled=true \
  --param training_seed_rotation_period=0 \
  --param objective_cvar_fraction=1.0 \
  --out-dir artifacts/research/runs/completed/basic-coevolution/larger-seed-7
```

Fixed world seeds and disabled training-seed rotation make scores comparable
between generations. The run seed still independently determines the initial
genome population and every evolutionary random choice.

## Earlier 20-generation sparse-topology result

The following result predates schema 13 and used the former shared random
ten-connection initial topology. It is retained as the baseline that motivated
the fully connected change.

| Outer seed | Generation-0 best | Final best | Peak best | Peak generation | Generation-0 population mean | Final population mean |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 3,987.0 | 17,430.5 | 23,086.5 | 11 | 2,724.8 | 8,486.1 |
| 42 | 21,143.5 | 23,008.0 | 24,000.0 | 7 | 3,522.6 | 10,902.1 |
| 123 | 3,069.5 | 8,648.5 | 11,714.5 | 6 | 2,626.6 | 4,281.3 |
| 2026 | 9,203.0 | 18,858.5 | 20,754.0 | 12 | 3,026.8 | 7,190.4 |
| 9001 | 3,519.0 | 18,597.5 | 21,420.0 | 16 | 2,604.7 | 7,669.3 |

Every independent run improved both its generation-best peak and its final
population-average total survival relative to generation 0. Only one run hit
the 24,000-tick ceiling, so the larger match retains substantially more room for
continued selection than the earlier pilot. This establishes the requested
first result: the basic contemporary-opponent NEAT loop selects progressively
longer-lived organisms over generations on this bounded match format.

It is not evidence of open-ended improvement: the score still has a finite
24,000-tick ceiling. Its purpose is to establish the basic loop and selection
signal at a scale that does not immediately saturate in most seeds.

## Determinism

An exact replay of outer seed 7 produced byte-identical result JSON. Both files
had SHA-256:

```text
c3a479f7f7f100731a652c26e7782ab84678d3deffd4131ae74f0e7e0792989f
```

## Fully connected 100-generation run

The follow-up held the evaluation contract fixed, changed generation-zero
initialization to all 45 sensor-to-action connections, and extended outer seed
7 to 100 generations:

```bash
./target/release/sim-cli neat \
  --seed 7 \
  --population 24 \
  --generations 100 \
  --episode-horizons 2000 \
  --opponents 4 \
  --world-seeds 11,29 \
  --no-audit \
  --no-holdout \
  --scenarios baseline \
  --workers 8 \
  --scale 35,60 \
  --set predation_enabled=true \
  --param training_seed_rotation_period=0 \
  --param objective_cvar_fraction=1.0 \
  --out-dir artifacts/research/runs/completed/basic-coevolution/fully-connected-100g-seed-7
```

| Generation | Best alive-ticks | Population mean | Best hidden nodes | Best expressed connections |
|---:|---:|---:|---:|---:|
| 0 | 17,074.5 | 4,169.0 | 0 | 45 |
| 20 | 14,474.5 | 7,622.3 | 2 | 48 |
| 40 | 10,584.5 | 5,214.1 | 1 | 46 |
| 58 | **22,393.5** | 7,723.2 | 2 | 47 |
| 60 | 17,736.0 | 7,642.4 | 2 | 49 |
| 77 | 20,504.5 | **11,240.4** | 2 | 48 |
| 80 | 20,652.5 | 9,368.2 | 2 | 48 |
| 99 | 10,492.5 | 4,333.5 | 3 | 50 |

The run-wide champion occurred at generation 58, improving absolute survival
by 5,319 alive-ticks over generation 0, but the population did not improve
monotonically and ended near its initial mean. The run generated 109 structural
connection innovations beyond the 45 initial connections and 36 node
innovations; the champion encoded 2 hidden nodes and 50 connections, of which
47 were enabled.

A frozen checkpoint crossplay over generations
`0,20,40,58,77,80,99` did not show a monotonic arms race. In particular, the
generation-20 checkpoint beat the generation-0 checkpoint directly and had
relative survival advantage above 1 against every sampled later checkpoint,
while generation 99 lost directly to generation 0. Thus the 100-generation run
shows real early improvement and structural complexification, followed by
cycling/regression rather than progressively stronger contemporaries.

```bash
./target/release/sim-cli neat crossplay \
  artifacts/research/runs/completed/basic-coevolution/fully-connected-100g-seed-7/neat-1783977539095-58786.json \
  --checkpoints 0,20,40,58,77,80,99 \
  --horizons 2000 \
  --world-seeds 11,29,47,61 \
  --levels 0 \
  --scenarios baseline \
  --allow-same-pool \
  --cvar 1.0
```

Crossplay emits only distinct-genome checkpoint matchups. It omits diagonal
clone-versus-clone cells because those turn a carnivore audit into cannibalism
or remove the distinct prey lineage the strategy depends on.
