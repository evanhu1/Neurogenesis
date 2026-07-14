# Evolvable plasticity and encoding audit (HEAD f1a2a27)

## Verdict

This route is **active only as a causal-learning substrate repair**, and
**blocked as a complete open-endedness algorithm**. Current runtime plasticity
has causal behavioral effects, but the reward signal is stale, reward timing is
off by one tick, the NEAT outer loop cannot evolve any plasticity parameter, and
the finite direct encoding plus fixed interface offers no reason for novelty to
continue in the tail.

## Exact current path

- Every founder is built in `sim-core/src/spawn/organisms.rs::build_organism`.
  `express_genome` copies each enabled `SynapseGene.weight` into a fresh runtime
  `SynapseEdge`, with eligibility and pending coactivation zeroed.
- Each tick is lifecycle -> intents/brain -> move resolution -> commit -> age ->
  post-commit plasticity. `compute_pending_coactivations` runs during intents;
  `apply_runtime_weight_updates` runs after action outcomes are committed.
- Plasticity changes `organism.brain.{sensory,inter}[].synapses[].weight`, never
  `organism.genome.brain.edges[].weight`.
- There are no in-world births. Every NEAT evaluation creates a new Simulation
  from the heritable genome; champion-pool respawns also clone the stored genome
  and re-express a fresh brain. Learned weights/pruning are therefore Baldwinian
  and are discarded between founders, respawns, and episodes. Saving/copying a
  world preserves learned runtime weights.
- `PlasticityGenes` has five inherited scalars (`hebb_eta_gain`, juvenile scale,
  eligibility retention, per-tick delta cap, prune threshold), but
  `evolution.rs` never mutates them. Crossover clones non-brain genes wholesale
  from the fitter base parent; compatibility distance ignores them. Both world
  baselines disable runtime plasticity.
- Direct NEAT can add/split connections, but has no deletion/duplication/module
  operator or indirect encoding. Runtime IDs cap hidden nodes at 1,000. With
  predation off the direct graph has at most 1,009,020 source-target pairs.

## Defects in the purported reward-learning rule

### A. Missing sensing stash (mandatory regression repair)

`energy_at_last_sensing` is initialized at birth and read after commit, but HEAD
never writes it again. Commit cc9cb4b removed the write while deleting the
EnergyDelta sensor. Thus:

```
implemented delta(t) = energy_after_commit(t) - energy_at_birth
intended delta(t)    = energy_after_commit(t) - energy_at_sensing(t)
```

For seed 303, organism 16 starts at energy 300. On tick 2 it unsuccessfully
turns right and ends at 317.212: current `m=1.04`, while the correct action delta
is -1 and `m=0.992`. On tick 219 it successfully eats and ends at 34.767:
current `m=0.96`, while the correct action delta is +19 and `m=1.04`. The stale
reference therefore rewards failures while the organism remains above birth
energy and damps real rewards while below it.

Smallest repair: restore `organism.energy_at_last_sensing = organism.energy`
inside `encode_sensory_inputs`, after lifecycle and before commit. This adds no
RNG or order dependence.

### B. Current activity is folded after the reward update (separate change)

At tick t, pending covariance is computed before action t, but post-commit code
updates the weight from eligibility(t-1) and only then folds pending(t). Reward
m(t) therefore gates the preceding tick's activity; activity that selected the
rewarded action is first gated by m(t+1).

Fold-before-update is the standard trace ordering:

```
eligibility = retention * eligibility + pending
delta_w = m * eta * eligibility - decay * weight
```

However, B is not safe to interpret as a standalone reward-learning fix. The
modulator is always positive (0.96..1.04) and the eligibility is not cleared on
reward. A single covariance event contributes on every later tick. At the
evaluation retention 0.90 its total asymptotic trace multiplier is 10x; at the
baseline 0.95 it is 20x. Fold-before-update changes the first credited tick but
does not create this persistence; both old and new order repeatedly consolidate
old activity. That is a valid eligibility trace only with a centered reward
prediction error. With the current `1 + small_delta` modulator, it is mostly
unsupervised covariance learning with a weak amplitude bias.

## Deterministic probes

All worlds and four release binaries are in this directory. Variant bits are
`A=stash`, `B=fold-before-update`. The seed-303 t0 worlds from all four binaries
have identical SHA-256
`ddb7a6a90833fdabb78df4e5a306926bfdc9f2b242f6013463f0438bf7269def`.

### Runtime plasticity off/on at tick 500

Same config, founder genomes, seed, and 30-founder scale; only the flag differs.

| Seed | Off pop / consumptions | On pop / consumptions |
|---:|---:|---:|
| 101 | 1 / 66 | 1 / 78 |
| 202 | 0 / 31 | 0 / 24 |
| 303 | 0 / 10 | 1 / 59 |

Plasticity clearly changes behavior, but the direction is not robust. Seed 303
organism 16 changed from 10 encoded/runtime edges at t0 to 9 runtime edges at
t1000; several weights hit +/-1.5. Its learned runtime brain consumed 61 plants
and held energy 126.24. This proves online state change, not adaptive learning.
The observational `learning_slope` was more negative in the dramatic seed-303
treatment, so that pillar is not a causal plasticity assay.

### 2x2 A/B probe at tick 500

| A stash | B fold | Seed 101 pop/cons | Seed 202 pop/cons | Seed 303 pop/cons | Mean consumptions |
|:---:|:---:|---:|---:|---:|---:|
| 0 | 0 | 1/78 | 0/24 | 1/59 | 53.67 |
| 1 | 0 | 0/66 | 0/25 | 1/65 | 52.00 |
| 0 | 1 | 0/54 | 0/29 | 1/56 | 46.33 |
| 1 | 1 | 1/70 | 0/33 | 0/28 | 43.67 |

This n=3 mechanistic probe is not a performance comparison. It shows both
semantic changes alter trajectories substantially and nonlinearly; neither is a
drop-in competence improvement. A is nevertheless mandatory correctness. B must
be evaluated with a centered reward error and reset/freeze counterfactuals.

Reproduction commands:

```bash
# Existing-rule off/on probe
sim-cli new --seed 303 --scale 25,30 \
  --set runtime_plasticity_enabled=true --out seed303-on.world.bin
sim-cli run-to 500 --in seed303-on.world.bin
sim-cli state --in seed303-on.world.bin

# 2x2 binaries (00 base, 10 stash, 01 fold, 11 both)
./sim-cli-11-stash-fold new --seed 303 --scale 25,30 \
  --set runtime_plasticity_enabled=true --out ab-11-seed303.world.bin
./sim-cli-11-stash-fold run-to 500 --in ab-11-seed303.world.bin
./sim-cli-11-stash-fold state --in ab-11-seed303.world.bin
```

## Concrete evolvable-learning mechanism

1. Repair A first and keep B as an independently gated experiment.
2. Replace the positive multiplier with an evolvable, centered three-factor
   rule while retaining Hebbian eligibility:

   `e_t = lambda*e_(t-1) + centered_pre*centered_post`

   `r_t = clamp((action_energy_delta - reward_ema)/reward_scale, -1, 1)`

   `dw = eta_unsup*e_t + eta_reward*r_t*edge_gain*e_t - decay*w`

   Update `reward_ema` after computing r. Start `eta_reward=0` for a strict
   compatibility/control point; evolve `eta_unsup`, `eta_reward`, lambda,
   reward-scale/EMA retention, delta cap, pruning, and nonnegative per-edge
   `edge_gain`. Selective edge gains let evolution protect stable circuitry while
   assigning plasticity to an adaptive subcircuit.
3. Make plasticity evolvable independently of weight mutation. Add bounded/log
   perturbations in a separate `mutate_plasticity` event. Uniformly recombine
   global plasticity scalars; matching `SynapseGene`s carry per-edge gain. Include
   normalized plasticity distance in speciation. Initialize new edges with small
   gain; on split, place the old gain on the outgoing edge and zero on the
   identity incoming edge to preserve the parent's initial function.
4. Select learning on deterministic within-lifetime contingency switches, not
   merely fixed-world survival. Evaluator-only held-out schedules swap left/right
   food-ray semantics (or actuator mapping) at irregular seed-derived boundaries.
   Fitness is a minimum criterion on steady foraging plus post-switch recovery
   area/regret. Use unseen schedules and mappings in development/sealed suites so
   a recurrent tick counter or fixed policy cannot memorize training switches.

Exact diff surface:

- `sim-core/src/brain/sensing.rs`: A stash repair; optional evaluator mapping.
- `sim-types/src/lib.rs`: added global reward genes/runtime reward EMA and
  per-edge plasticity gain in `SynapseGene`/`SynapseEdge`.
- `sim-core/src/brain/expression.rs`: copy edge gain; initialize runtime state.
- `sim-core/src/brain/plasticity.rs`: centered reward error, independently gated
  B ordering, per-edge gain, exposed constants.
- `sim-core/src/evolution.rs`: NeatConfig mutation parameters/validation,
  independent mutation, uniform scalar crossover, compatibility term, switch
  schedule evaluation and phase metrics.
- `sim-cli/src/neat.rs`: parameter parsing and result serialization/analyze.
- `sim-config` and `sim-evaluation` seed/config TOMLs: synchronized defaults.
- `sim-server` plus `web-client` Api types/normalizer if new genes/runtime fields
  remain on the wire.

## Required causal ablations (no new unit tests)

At a phase boundary clone the in-memory `Simulation` four ways; clone already
preserves world/RNG/runtime brain state:

| Arm | Learned weights | Updates after switch | Meaning |
|---|---|---|---|
| carry/continue | carry | on | full adaptive agent |
| carry/freeze | carry | off | value of acquired weights |
| reset/freeze | re-express genome | off | inherited fixed policy |
| reset/continue | re-express genome | on | reacquisition after reset |

Add an evaluator/CLI hook that re-expresses organism brains from their genomes
without touching position, energy, ecology, RNG, or IDs, and a setter that
freezes runtime updates. A full re-expression confounds weights with recurrent
state/eligibility, so also expose a weights-only reset by matching fresh and live
edges on dense pre/post IDs. Compare carry/freeze vs reset/freeze for learned
weight causality; carry/continue vs carry/freeze for ongoing adaptation;
reset/continue vs reset/freeze for reacquisition. Also run reward-neutral
(`eta_reward=0`), sign-inverted reward, runtime-off, topology-fixed, and
plasticity-gene-frozen controls on the same held-out schedules.

## Why this is still likely bounded

- The 5-sensor/4-action foraging interface, finite 1,000-node direct graph, fixed
  finite switch family, and fixed world affordances cap useful strategies.
- Indirect encoding/module duplication improves evolvability and scaling but
  does not create new behavioral demands; earlier paired evidence already showed
  direct complexification can be causally useful without outperforming a fixed
  topology control.
- A switch assay can prove genuine lifetime learning but is a bounded benchmark,
  not sustained open-ended novelty. This route should remain active only if the
  reset/freeze delta persists across held-out switches and late-generation
  adaptation-regret and repertoire metrics keep improving. Otherwise mark it
  blocked; a materially new open-ended driver would need expanding affordances,
  compositional task creation, or endogenous challenge generation.
