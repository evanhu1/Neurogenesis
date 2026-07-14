# Robust pairwise NEAT evaluator

Schema 16 uses a balanced pairwise evaluation to rank the current 24-genome
population and derives behavioral diagnostics from those same competitive
worlds. It does not run a separate clonal champion audit. The result retains a
snapshot of every genome and evaluation in the final population so sparse
opponent/seed schedules can be audited against dense reference panels.

Two independent schema-15 instrumentation smokes were byte-identical with
SHA-256 `c76bee95f46dfaeba57109524dad36929d50871d92696e6d910cabdcf4c3bf2e`.
Their champion was correctly exposed as a predator: zero plant intake, prey
intake fraction `1.0`, mean `2.8125` attack kills per directly instrumented
case, and nonzero prey-consumption and action-effectiveness rates. The complete
generation-0 pool contained 2 nonconsumers, 11 foragers, 6 predators, and 5
omnivores, showing that the role breakdown is not merely a champion label.

## Contract

- Every genome faces eight distinct contemporary opponents per generation.
- Each world contains exactly two genomes and scores both lineages from that
  single deterministic simulation.
- The pairing graph is regular: every genome has the same number of matches.
- Across every block of four world seeds, both genomes occupy both founder/ID
  slots and each receives direct behavioral instrumentation in both slots.
- Training uses four fixed world seeds and a 5,000-tick horizon.
- Fitness is the ordinary mean over all 32 cases per genome (eight opponents ×
  four seeds) of `absolute survival × relative survival advantage`.
- Relative advantage is `2c / (c + o)`, where `c` and `o` are per-founder
  candidate and opponent alive-ticks. Equal survival is 1.0.
- Directly instrumented pairwise cases use the shared `sim-metrics` definitions
  for action effectiveness, plant consumption rate, prey consumption rate,
  sensory/action mutual information, and learning slope.
- Plant and prey intake fractions plus attack kills produce a coarse trophic
  role: `nonconsumer`, `forager`, `scavenger`, `predator`, or `omnivore`.
  Continuous rates and the attack funnel remain the authoritative evidence.
- There is no isolated development or holdout panel. Predators are therefore
  never evaluated through clone-on-clone cannibalism or attack suppression.
- There is no top-contender playoff.

At the default `50×50`, 100-founder scale, a pairwise world contains 50 founders
from each genome. The maximum raw candidate alive-ticks for one 5,000-tick case
is therefore 250,000. Normalized survival remains comparable across scales.

Checkpoint crossplay uses this same exact two-genome evaluator. It scores both
lineages from each shared world and performs the same balanced four-case slot
and direct-instrumentation rotation; it does not duplicate the opponent into a
third founder-pool entry. It compares only distinct genome hashes. Diagonal
clone-versus-clone cells, including identical genomes persisted at different
checkpoints, are excluded because cannibalism or attack suppression would not
fairly measure a carnivore that depends on another lineage as prey.

## Opponent/seed allocation calibration

The `8 opponents × 4 seeds` default is empirical rather than arbitrary. Three
independently initialized 24-genome populations (`701`, `1301`, and `1901`)
were evaluated under several sparse schedules. Each sparse ranking was judged
against the same population's full 23-opponent × 16-seed ranking on a completely
disjoint world-seed suite. The reproductive target was the top five genomes,
because the default survival fraction selects 20% of a 24-genome population.

Normalized regret is the fraction of the dense top-five selection differential
lost by using the sparse schedule. Zero means the sparse schedule chose a set
with the same dense-reference mean as the true top five.

| Cases/genome | Allocation | Mean top-five recall | Worst recall | Mean regret | Exact winner |
|---:|---:|---:|---:|---:|---:|
| 16 | `2×8` | 0.867 | 0.8 | 0.063 | 2/3 |
| 16 | `4×4` | 0.867 | 0.6 | 0.015 | 3/3 |
| 32 | `2×16` | 0.867 | 0.8 | 0.063 | 3/3 |
| 32 | `4×8` | 0.933 | 0.8 | 0.008 | 3/3 |
| 32 | **`8×4`** | **1.000** | **1.0** | **0.000** | **3/3** |
| 64 | `8×8` | 1.000 | 1.0 | 0.000 | 3/3 |
| 64 | `16×4` | 0.933 | 0.8 | 0.011 | 3/3 |

The important comparison is not merely `4×8` versus `8×4`. Increasing two
opponents from eight to sixteen seeds did not improve top-five recovery, while
increasing opponent coverage did. Once eight opponents were represented,
spending the next factor of two on seeds (`8×8`) preserved perfect recovery;
spending it only on opponents (`16×4`) did not. Thus the data support a minimum
four-seed block, opponent expansion until the reproductive set stabilizes, and
then additional seed blocks.

Equal-compute 10-generation runs on seeds `7`, `17`, and `27` checked that the
ranking improvement did not sacrifice evolutionary progress:

| Allocation | Mean generation-9 population survival | Mean late-3 survival | Mean trajectory survival | Mean generation-9 best survival |
|---:|---:|---:|---:|---:|
| `4×8` | 0.234 | 0.214 | 0.154 | 0.568 |
| **`8×4`** | **0.307** | **0.291** | **0.181** | **0.604** |

Late-three population survival was higher under `8×4` in all three paired
runs. The single best genome was worse in seed `17`, so this is evidence for a
better default allocation, not a claim of per-seed dominance.

The general scaling rule is to treat opponent and environment sampling as
different uncertainty sources. For `O` opponents and `B` four-seed blocks,
score uncertainty is approximately

```text
opponent_variance / O
  + seed_block_variance / B
  + interaction_variance / (O * B)
```

At population `P` and `Q` scenario/horizon combinations, the number of
simulated worlds per generation is `2 * P * O * B * Q`, because one world
scores both genomes. Increase the dimension with the larger observed marginal
reduction in held-out top-parent regret. Stop when the reproductive-set gate is
met; use further compute for generations unless a later dense audit shows that
new strategic diversity has increased opponent variance. Opponent count is
therefore not a permanent constant or necessarily a fixed population fraction.

All calibration artifacts and the analysis script are under
`artifacts/research/runs/completed/basic-coevolution/opponent-seed-calibration/`.

## Full-scale smoke run

```bash
./target/release/sim-cli neat \
  --seed 7 \
  --population 24 \
  --generations 10 \
  --episode-horizons 5000 \
  --opponents 8 \
  --world-seeds 11,29,47,61 \
  --scenarios baseline \
  --workers 8 \
  --scale 50,100 \
  --set predation_enabled=true \
  --param training_seed_rotation_period=0 \
  --param objective_cvar_fraction=1.0 \
  --out-dir artifacts/research/runs/completed/basic-coevolution/pairwise-robust-10g-seed-7
```

| Generation | Best fitness | Best absolute survival | Best relative advantage | Population mean survival |
|---:|---:|---:|---:|---:|
| 0 | 0.520 | 0.301 | 1.661 | 0.069 |
| 2 | 0.875 | 0.503 | 1.717 | 0.143 |
| 5 | 0.948 | 0.578 | 1.538 | 0.256 |
| 8 | 1.274 | 0.762 | 1.619 | 0.392 |
| 9 | **1.546** | **0.879** | **1.730** | **0.410** |

The final champion was evaluated over 32 pair/seed cases: eight distinct
opponent genomes, each repeated across four world seeds. Population mean
relative advantage is exactly 1.0 up
to floating-point rounding because every match contributes symmetrically to
both participants; individual relative advantage still determines selection
when multiplied by absolute survival.

The hash below describes the historical schema-14 run that established
pairwise symmetry. Schema 15 intentionally removed its solo-audit columns. The
deterministic two-generation symmetry smoke was replayed byte-for-byte; both
result JSON files had SHA-256:

```text
351ad03b4fce2f485b4f9a873f2fff567be50e65be001be3dc164b99f066321e
```

## Fifty-generation held-out checkpoint audit

Three schema-16 runs at the default `50×50`, 100-founder scale used evolution
seeds `7`, `17`, and `27`. Each generation evaluated 24 genomes against eight
contemporary opponents on training world seeds `11,29,47,61`. Frozen checkpoint
champions from generations `0,10,20,30,40,49` were then crossed against one
another with the exact two-genome evaluator on disjoint world seeds
`101,131,151,181,211,241,271,311`.

The result is evidence for evolutionary improvement, but not yet a
seed-robust monotonic process:

| Run seed | Gen 49 vs gen 0 | Held-out W-L | Mean alive-tick delta | Total energy, gen 0 → 49 | All later-pair wins |
|---:|---|---:|---:|---:|---:|
| 7 | strong loss | 0-8 | -47,587.6 | 102,665.5 → 64,645.0 | 60.0% |
| 17 | strong win | 8-0 | +46,239.6 | 11,706.9 → 132,904.1 | 73.3% |
| 27 | strong win | 8-0 | +228,967.4 | 5,655.0 → 205,395.0 | 73.3% |

Seeds 17 and 27 therefore show robust first-to-last gains in total survival
time and gross energy intake on every held-out world. Seed 7 is the adversarial
counterexample: its final pure forager had higher plant-consumption and
sensor/action-information rates than its initial omnivore, yet lost survival,
relative advantage, and gross energy on every held-out world. Across all runs,
the strongest checkpoint was not necessarily the last one, so the result is
best classified as majority directional progress with substantial
nonmonotonicity, not consistent progressive improvement.

The failure is not explained by a dead evolutionary search. Over generations
33–49, every run retained 2–5 species, crossover occurred in 88–94% of
generations, and 79–86% of attempted structural mutations succeeded. Champions
complexified from the initial 45 expressed connections to 48, 49, and 46
connections respectively; seed 17's final checkpoint expressed two hidden
nodes. The remaining gap is selection reliability across evolutionary seeds,
not an absence of variation.

The complete reproducible report, analyzer, crossplay matrices, and built-in
NEAT analysis are under
`artifacts/research/runs/completed/basic-coevolution/pairwise-50x50-100f-50g-audit/`. The
source runs are the sibling
`pairwise-50x50-100f-50g-seed-{7,17,27}/` directories. The audit intentionally
does not claim open-endedness: it covers a bounded 50-generation interval and
shows a concrete regressing seed.

## Speciation calibration for a 24-genome population

The original generic defaults targeted eight species with a compatibility
threshold that moved by `0.1` per generation. With only 24 genomes, the
controller drove the threshold from `3.0` to its `0.1` floor while the run had
one species. Generation 29 then discontinuously split into 19 species. This was
not harmless bookkeeping: every active species receives at least one offspring
(two while young), so the overshoot consisted mostly of protected singletons
and within-species crossover fell to zero.

The calibrated defaults are:

- initial compatibility threshold `1.0`;
- target species count `4` (six genomes per species on average);
- threshold adjustment `0.05`;
- matching-gene weight-difference coefficient `0.2` instead of `0.4`.

The lower weight coefficient keeps speciation focused on structural separation
instead of ordinary weight drift. A 35-generation staged comparison selected
`0.2` over `0.1`: it maintained roughly four species rather than collapsing
back to two, while still retaining multi-parent species and active crossover.

The directly comparable 50-generation seed-7 confirmation is
`artifacts/research/runs/completed/basic-coevolution/pairwise-robust-speciation-calibrated-50g-seed-7/neat-1783984644776-41136.json`.
It was produced immediately before the schema-15 behavioral-diagnostics
cutover; its competitive/speciation trajectory remains valid, while its legacy
solo-audit fields are not part of the current contract.

| Tail statistic | Original defaults | Calibrated defaults |
|---|---:|---:|
| Mean species, generations 29–48 | 17.65 | 3.75 |
| Species range, generations 29–48 | 13–19 | 3–5 |
| Mean crossovers, generations 29–48 | 0.05 | 13.4 |
| Crossover range, generations 29–48 | 0–1 | 9–19 |
| Mean population survival, generations 40–49 | 0.241 | 0.391 |
| Peak best survival | 0.990 | 0.993 |

The late calibrated species had real breeding cohorts rather than only labels:
generation 49 contained sizes `6,7,1,10`. Structural search also remained
active across the run (24 new node innovations and 60 new connection
innovations, versus 27 and 67 in the original), although only one hidden node
was expressed by a generation champion. The calibration therefore fixes the
observed fragmentation/crossover failure without claiming that topology
complexification itself is solved.

Two independent two-generation replays produced byte-identical result JSON with
SHA-256:

```text
2c23b37561d12f7a6549b1afd4b671f5e44f9aaeee45e6c3ee64ffcd42ae2d37
```
