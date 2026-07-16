# 2026-07-15-shared-population-locality: exhaustive pairs versus one shared world

Status: completed; shared evaluation is a useful cheap screen but rejected as
the sole evolutionary evaluator in the tested form

## Question

Can one world containing all eight contemporary genomes replace separate
pairwise worlds? Random spawn locality would then sample who meets whom while
each simulator tick produces fitness evidence for the complete NEAT population.

This experiment retains plants and the canonical energy economy. It tests only
evaluator topology; removing plants is a separate ecological ablation.

## Implemented semantics

The evolutionary contract now has an explicit evaluator topology:

- `pairwise`: each scheduled genome pair receives its own world;
- `shared_population`: all contemporary genomes coexist in each world and every
  lineage is scored from that same simulation;
- `isolated`: the existing no-opponent mode.

Shared evaluation runs each scenario, world seed, and horizon once. The
lineage-specific survival, action, consumption, energy, attack, and spatial
metrics are all read from one per-world ledger. It does not rerun the world for
each focal genome. Result schema 28 persists the topology and per-genome case
score standard deviation and range. `cli plan`, the batch manifest, and
`summarize` expose the resolved topology and context dispersion.

In a shared world, seven opponent presences are not seven independent cases.
They are correlated parts of one joint context. The treatment therefore has
four statistically distinct cases per genome, not 28, even though every genome
is present with all seven opponents in all four cases.

## Matched contract

Both arms used:

- evolutionary seeds `7,17,27`;
- population `8` and generations `0..39`;
- 500-tick episodes;
- training world seeds `11,29,47,61` and the baseline scenario;
- held-out world seeds `101,131,151,181`;
- 50x50 worlds with 96 founders;
- canonical ecology: 20% plants, 20 plant energy, 250 starting energy,
  10-energy attack attempts, and 40-energy attack transfer;
- categorical actions, feed-forward brains, predation enabled, and plasticity
  disabled;
- absolute normalized survival and ordinary mean aggregation (`cvar=1`);
- population checkpoints at `0,10,20,30,39`.

The pairwise control exhaustively evaluated all seven contemporaries. It used
48 founders per lineage, 28 cases per genome, 112 worlds per generation, and
4,480 worlds per evolutionary seed.

The shared treatment put all eight lineages in each world. It used 12 founders
per lineage, four cases per genome, four worlds per generation, and 160 worlds
per evolutionary seed. This was intentionally not compute matched: the test
asked whether locality itself preserved enough signal to earn reinvestment of
the savings.

Artifact directories:

- `artifacts/research/runs/completed/2026-07-15-pop8-pairwise-all-opponents-40g/`
- `artifacts/research/runs/completed/2026-07-15-pop8-shared-population-40g/`

The source revision was `9c604c581509f5e7ebc2c06dd95035b752b3d266`
with tracked patch SHA-256
`1b3fa1c01543cbbdec02d6943d30d7952bced4740295105f59d6e460cf3ba06b`.
The release executable SHA-256 was
`99258e7779a79a7f26aa69d1f2487ee8f59a59d7deabc537616c5bec912eb803`.

## Decision rule

Shared locality was promotable only if:

1. generation-39 population mean exceeded generation 0 with a positive full
   fitted slope in every seed;
2. the generation `20..39` mean slope was nonnegative in at least two seeds;
3. gains were not driven by one favorable case or endpoint censoring;
4. frozen held-out validation was not materially worse than pairwise; and
5. compute savings were substantial.

## Commands

The batch commands differed only in evaluator arguments:

```text
cli batch --experiment 2026-07-15-pop8-<arm>-40g \
  --seeds 7,17,27 --total-workers 14 \
  --out-dir artifacts/research/runs/active -- \
  --population 8 --generations 40 --population-checkpoint-interval 10 \
  --horizon 500 --world-seeds 11,29,47,61 --scenarios baseline \
  --founders 96 --cvar 1 \
  <pairwise: --evaluator pairwise --opponents-per-genome 7> \
  <shared: --evaluator shared_population>
```

Training was summarized over the preregistered inclusive tail `20..39`.
Checkpoint crossplay used:

```text
cli crossplay seed-N.result.json.zst \
  --checkpoints 0,10,20,30,39 --horizons 500 \
  --world-seeds 101,131,151,181 --out seed-N.heldout-crossplay.json
```

Historical champions were also compared directly in both focal slots with:

```text
cli evaluate-panel --focal PAIRWISE_RESULT --opponents SHARED_RESULT \
  --horizons 500 --world-seeds 101,131,151,181
cli evaluate-panel --focal SHARED_RESULT --opponents PAIRWISE_RESULT \
  --horizons 500 --world-seeds 101,131,151,181
```

## Results

### Training trajectories

Both evaluators produced real early adaptation. Generation-39 population mean
survival exceeded generation 0 in all six runs and every full-run mean slope
was positive. The shared arm nevertheless failed the late-progress gate.

| Arm | Seed | gen-0 mean | gen-39 mean | tail mean delta | tail slope | historical champion |
|---|---:|---:|---:|---:|---:|---:|
| pairwise | 7 | 0.325 | 0.560 | +0.012 | +0.00103 | 0.588 at gen 31 |
| pairwise | 17 | 0.217 | 0.677 | +0.140 | +0.00539 | 0.719 at gen 39 |
| pairwise | 27 | 0.434 | 0.594 | +0.098 | +0.00280 | 0.698 at gen 31 |
| shared | 7 | 0.325 | 0.553 | +0.025 | +0.00146 | 0.597 at gen 38 |
| shared | 17 | 0.232 | 0.525 | -0.001 | -0.00009 | 0.568 at gen 34 |
| shared | 27 | 0.421 | 0.537 | approximately 0 | -0.00048 | 0.583 at gen 28 |

Only shared seed 7 retained positive late population progress. Shared seeds 17
and 27 reached roughly `0.53` by generation 20 and then stopped. Pairwise seeds
17 and 27 instead discovered much stronger high-mobility plant foragers in the
tail. At generation 39 their champion action effectiveness was `0.158` and
`0.097`, versus `0.007` and `0.015` under shared evaluation. Predation was
negligible in both arms, so this difference is not a hidden predator-prey cycle.

Case ranges did not reveal a single catastrophic world seed. Final shared
champion minima were `0.565`, `0.538`, and `0.533`; final pairwise minima were
`0.557`, `0.604`, and `0.566`. However, the four shared cases are joint contexts,
whereas the 28 pairwise cases cross four worlds with seven separately isolated
opponents. Similar raw standard deviations therefore do not imply equal
effective opponent coverage.

### Held-out chronological validation

After collapsing duplicate checkpoint genomes and averaging the two ordered
founder arrangements for each unique chronological pair:

| Arm | Seed | later-checkpoint wins | mean later-minus-earlier survival |
|---|---:|---:|---:|
| pairwise | 7 | 6/6 | +0.0477 |
| pairwise | 17 | 10/10 | +0.0578 |
| pairwise | 27 | 9/10 | +0.0579 |
| shared | 7 | 10/10 | +0.0431 |
| shared | 17 | 5/6 | +0.0239 |
| shared | 27 | 9/10 | +0.0313 |

Shared training therefore did not generate random rankings: 24 of 26 unique
held-out chronological pairs favored the later checkpoint. But its average
progress step was about 40% smaller than pairwise, and shared seed 17 was nearly
flat after generation 10.

The direct historical-champion comparison was more decisive. Across four
held-out seeds and both focal slots, pairwise-trained champions beat
shared-trained champions in 23 of 24 cases:

| Seed | pairwise mean survival | shared mean survival | pairwise wins |
|---|---:|---:|---:|
| 7 | 0.582 | 0.561 | 7/8 |
| 17 | 0.589 | 0.546 | 8/8 |
| 27 | 0.616 | 0.560 | 8/8 |

This rules out the explanation that shared training merely used a compressed
fitness scale. It evolved genuinely weaker frozen champions in two seeds.

### Compute

Pairwise seeds took `25.492`, `25.601`, and `27.871` seconds. Shared seeds took
`1.033`, `1.047`, and `1.048` seconds: a 24.5x to 26.6x speedup, close to the
28x reduction in evaluator worlds.

## Interpretation

Shared locality is a valid but lower-resolution selection signal. It reliably
finds the first broad foraging improvement and orders most frozen checkpoints
chronologically. It is not reliable enough, in this four-context/all-eight
form, to replace pairwise evaluation for sustained improvement.

The mechanistic issue is not simply that the seven opponent exposures are
redundant. They are statistically entangled. Every focal score is a property of
one eight-policy mixture: all lineages consume the same plants, alter local
density, block movement, and change which organisms meet. A mutation controls
only one eighth of that ecology, with 12 clones instead of 48, so its causal
effect on its lineage's outcome is diluted by seven simultaneous policies.
Four terrain/spawn seeds cannot separate those effects. In pairwise worlds the
focal controls half of the organism population and each opponent context is
isolated, producing a stronger and more attributable gradient.

The result does not show that larger shared ecosystems are intrinsically bad.
It shows that nominal opponent count is not sample size and that the compute
savings cannot all be taken as free. Some must be reinvested in independent
contexts, lineage replication, or smaller balanced pods.

## Decision and next experiment

Keep `shared_population` as an explicit experimental evaluator and cheap
screening mode; do not make it the baseline. Retain pairwise evaluation for the
next plant-free test.

The most informative follow-up is not immediately to add more genomes to the
same single world. Isolate why shared selection weakened by comparing:

1. the current four-case shared arm;
2. shared evaluation with substantially more independent world/spawn seeds;
3. balanced four-lineage pods, once supported, so each genome is measured in
   several independently composed groups with more founders per lineage.

That separates insufficient independent contexts from causal dilution by group
size. A compute-matched comparison is unnecessary initially: even 16 shared
worlds per generation would remain seven times cheaper than exhaustive
pairwise evaluation at population eight.

## Verification

`cargo check --workspace`, `make fmt`, and `make lint` passed. A release shared
smoke run reproduced byte-identical result and champion-world hashes. Pairwise
and shared release smokes both produced complete per-lineage instrumentation.
`cargo test --workspace` passed except for the pre-existing stale assertion
`lethal_attack_conserves_energy_without_spawning_food`: the test expects 59
energy while the canonical 10-energy attack attempt cost correctly yields 49.
No tests were added or changed.
