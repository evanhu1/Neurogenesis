# QD / minimal-criteria route

## Verdict

`robust_trace_map_elites` is worth implementing as a **selection and audit
scaffold**, but this family is **blocked as a complete open-endedness mechanism
on the present static ecology**. A fixed plant-foraging world has a finite set
of useful affordances; an indefinitely growing archive would eventually be
trace refinement or drift, not new adaptive problems.

Blocked variants: the current averaged-descriptor NSLC; fixed-bin MAP-Elites;
an archive whose radius merely shrinks over time; a horizon-only ratchet; and
layout/seed mutation without new causal affordances.

## Current falsification

Current NSLC uses six mean action fractions, mean absolute spatial coverage,
and mean normalized first-plant time. It admits two highest-novelty descriptors
per generation if they differ by more than `1e-9`, with no competence gate.

Round 1 already shows the failure: archive size reached 74/78 while best novelty
fell from `.339/.358` at generation zero to `.198/.196` at the endpoint. NSLC
lost development and sealed competence to fitness selection on both seeds. The
seed-101 champion was 99.07% Idle, consumed zero plants, and had only 4.13%
coverage on development cases. A direct `sim-cli` replay had all 30 founders
alive with zero consumption at tick 500 and all extinct with zero consumption
at tick 1000. The seed-202 champion was a real forager (532 plants, 26/30 alive
at tick 1000), so this is seed-dependent basin selection, not a reliable QD
result.

The stronger frozen-repertoire control used 20 generations, 20 genomes, a
200-tick baseline episode, seeds 11/29, no crossover, and every mutation
probability zero:

| arm | archive g0/g4/g9/g19 | structural innovations | max new-origin rate |
|---|---:|---:|---:|
| fixed layouts | 0 / 4 / 4 / 4 | 0 | 0 |
| rotating layouts | 0 / 6 / 11 / 21 | 0 | 0 |
| evolution on, fixed layouts | 0 / 8 / 18 / 38 | 25 | .118 |

The rotating frozen arm ended with best and mean novelty exactly zero while its
archive still grew every generation. Archive size therefore confounds policy
novelty with remeasurement on a changing seed panel. The evolution-on arm did
produce a durable forager when replayed to tick 1000 (400 plants, 5/10 alive),
but every 200-tick training fitness was exactly 1.0, so its 38 entries do not
establish competence-conditioned diversity.

## Strongest candidate: fixed-panel robust trace MAP-Elites

1. Keep fitness training seeds free to rotate, but compute archive descriptors
   only from immutable level-zero reference cases: fixed scenarios, horizons,
   `descriptor_world_seeds`, and two deterministic founder-ID salts. Base seeds
   already run in every training epoch, so the first descriptor panel is free.
2. Accumulate an online, ID-free trace profile per case:
   - an opportunity-conditioned `(food cue: none/L/F/R) x action x success`
     histogram;
   - a cue-to-successful-eat latency histogram (1-4, 5-16, >16/no response);
   - 32-tick macro-outcome histograms over net toroidal displacement, newly
     visited cells, successful plant/prey acquisition, and energy-delta sign;
   - early/middle/late versions of those histograms, plus attackable-contact
     response when predation is enabled.
   IDs are only accumulator keys. No absolute `q/r`, raw action count, or
   boundary coordinate enters the profile. Left/right flicker with no move,
   acquisition, or state transition maps to the same inactive macro-outcome.
3. Distance is weighted Jensen-Shannon distance over the normalized
   histograms. For candidate `c` and archive entry `a`, stable distance is
   `min(D_A(c,a), D_B(c,a))` on two disjoint fixed panels. A new niche requires
   stable distance above the predeclared radius on both panels; changing the
   radius during a run is forbidden.
4. Minimal criteria are counterfactual, not hand-picked scalar thresholds.
   Precompute same-case zero-controller and forced-random controls. An entry is
   eligible only if lower-tail late-window plant-derived energy per founder-tick
   exceeds both controls, endpoint survival is no worse than the better control,
   and those statements hold on baseline plus one modest scarcity perturbation.
   For predation, only energy with plant provenance counts; founder reserves or
   evaluator-injected renewal energy never count.
5. Each niche holds a size-2 Pareto set on `(late external-energy rate,
   endpoint survival, opportunity-conditioned success)`. A candidate either
   opens a validated niche or Pareto-improves its nearest niche. Archive parent
   sampling is uniform by niche, then by elite; deterministic curiosity breaks
   ties. NEAT innovation IDs and mutation/crossover remain unchanged.
6. Add three parent modes: `archive`, `uniform_current` (selection-off control),
   and `frozen_generation_zero`. Archive admissions are sorted by stable
   distance, quality vector, then genome fingerprint before mutation RNG is
   drawn, preserving determinism.

## Exact code-diff surface

- `sim-types/src/lib.rs`: extend instrumentation-only `ActionRecord` with
  quantized food-ray activations, contact, energy bin, and source species; no
  wire/UI change.
- `sim-core/src/turn/intents.rs` and `sim-core/src/brain/{mod,evaluation}.rs`:
  populate the sensory snapshot used by the trace accumulator.
- `sim-core/src/evolution.rs`: retain detailed fixed-base cases in each
  `Individual`; accumulate quarter energy/alive counts; replace the 8-scalar
  `BehaviorDescriptor`/quota archive path for the new strategy; add
  `ParentSelectionMode`; leave existing fitness/NSLC modes intact as controls.
- `sim-core/src/evolution/qd.rs` (new): `TraceAccumulator`, `TraceProfile`,
  `QualityVector`, `ArchiveEntry`, stable two-panel admission, Pareto
  replacement, deterministic parent emission, and null-policy baselines.
- `sim-core/src/lib.rs` plus spawn helper: evaluator-only deterministic founder
  ID remapping for the two salt audits, updating occupancy consistently.
- `sim-cli/src/neat.rs`: flags `--descriptor-seeds`,
  `--archive-validation-seeds`, `--descriptor-id-salts`; params
  `selection_strategy=robust_trace_map_elites`,
  `parent_selection_mode=archive|uniform_current|frozen_generation_zero`, and
  fixed archive radius. Persist every entry's genome fingerprint, parent,
  generation, both panel profiles/distances, quality vector, and admission or
  replacement reason. Extend `neat analyze` with tail admissions, validated
  niche count, quality-front improvements, and control deltas.

Do not mutate plasticity genes in the first QD gate: current NEAT mutates only
brain weights/biases/time constants and add-node/add-connection structure.
Evolvable plasticity is a separate causal treatment after selection itself
passes.

## Cost and decisive gate

Online profiles are fixed-size counters and add O(live-organism ticks) integer
work with no per-tick allocation. Fixed panel A reuses base-seed evaluations.
If at most two proposed admissions per generation are checked on two validation
seeds, validation adds about `2*2/(32*3) = 4.2%` simulation work at population
32; a second ID salt makes it about 8.3%. Archive nearest-neighbor search is
linear for the cheap gate; a deterministic sorted vantage tree is needed only
after hundreds of entries.

Run identical outer seeds for three arms: archive selection with normal
mutation; archive selection with crossover and all mutation probabilities zero;
and normal mutation with `parent_selection_mode=uniform_current`. Use population
32, 30 generations, horizons 1000/4000, fixed descriptor seeds 11/29/47,
validation seeds 61/79, scenarios baseline/scarcity, and sealed seeds
131/149/167. The cheap gate passes only if:

- the frozen arm admits no post-generation-zero stable niche;
- the archive arm has more sealed-validated minimal-criterion niches and more
  quality-front improvements than uniform selection on each of at least three
  of four outer seeds;
- the final third still has validated admissions, and their per-organism
  `inspect`/`brain`/`decide` traces show distinct cue-response and macro-outcome
  strategies;
- plasticity-off and evolved-structure knockout audits identify causal
  contribution for claimed learning/complexification.

Archive occupancy, descriptor distance, or a one-time early fill does not pass.

## Exact remaining gap

Even if this gate passes, it demonstrates a better repertoire search, not
open-ended evolution. Round 9 found only finite historical ladders and Round 10
showed that evaluator-renewed opponents either kill the focal or become an
energy farm. The missing mechanism is a conserved endogenous problem generator:
plant-funded, energy-conserving multi-lineage renewal plus heritable
opponent/kin-contingent interaction or persistent niche construction. Mere
reproduction is insufficient if it settles into one fixed forager/predator
equilibrium; it must create new conditional challenges while recycling, never
minting, ecological energy. Only after late interaction energy and historical
coverage continue should this QD archive be used to preserve the resulting
niches.
