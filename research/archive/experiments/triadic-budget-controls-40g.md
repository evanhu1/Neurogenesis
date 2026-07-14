# Triadic evaluation budget controls, 40 generations

## Bottom line

The useful result is not simply “three lineages are better.” It is:

1. **Eight opponent exposures packaged into only four triadic worlds per genome
   are inadequate.** The exposure-matched triad arm regressed badly on two of
   three seeds. Equal opponent count did not compensate for halving the number
   of independent scored cases from 32 to 16.
2. **Triads become competitive once restored to 32 cases per genome.** The
   case-matched triad produced the best overall chronological ordering and
   final retention while using 256 rather than 384 worlds per generation.
3. **The 48-case compute-matched triad had the strongest mean held-out tail
   slope and preserved population competence better than the 32-case triad, but
   its extra evaluation did not improve every endpoint.** Returns beyond 32
   cases were mixed.
4. **The pairwise control itself continued making late gains.** At 40
   generations the earlier claim that ordinary pairwise evolution necessarily
   plateaus near generation 20–30 is not supported across these seeds.
5. **No arm demonstrated reliably progressive evolution in every sense.** All
   adequate-budget arms had positive generation-20–39 held-out strength slopes
   on all three seeds, but every arm also had at least two runs with a late
   reversal, population weakness, or exact checkpoint repetition. This is
   evidence of progress beyond generation 30, not evidence of open-endedness.

For the next longer experiment, the **case-matched triad is the best
cost/quality candidate**: 3 lineages, 16 opponent exposures, 8 world
memberships, 32 cases, and 256 worlds per generation. The compute-matched arm
is the conservative higher-precision option if simulator cost is secondary.

## Experimental contract

All four arms used contemporary opponents only. No historical archive or hall
of fame exists in the evaluator.

- evolutionary seeds: `7,17,27`
- population: 24
- generations: 40 (`0` through `39`)
- horizon: 5,000 ticks
- training world seeds: `11,29,47,61`
- baseline scenario, predation enabled
- 50x50 world, 102 founders
- unchanged NEAT, objective, brain, and ecology settings

| Arm | Lineages/world | Memberships/genome | Exposures/genome | Cases/genome | Worlds/generation |
|---|---:|---:|---:|---:|---:|
| Pair | 2 | 8 | 8 | 32 | 384 |
| Triad exposure-matched | 3 | 4 | 8 | 16 | 128 |
| Triad case-matched | 3 | 8 | 16 | 32 | 256 |
| Triad compute-matched | 3 | 12 | 24 | 48 | 384 |

Every one of the 480 emitted generation records reported the expected case and
world counts. The two new triad configurations were also run twice in a
deterministic smoke test; each pair produced byte-identical result JSON.

Frozen post-evolution crossplay compared checkpoints
`0,5,10,15,20,25,30,35,39` in the same two-lineage evaluator on eight unseen
world seeds: `101,131,151,181,211,241,271,311`. Raw training fitness is not
compared across pair and triad arms because its relative-advantage term has
different population algebra. Cross-arm conclusions use absolute survival and
this common frozen pairwise assay.

## Aggregate results

“Robust tail slope” is the generation-20–39 linear slope of a checkpoint's mean
held-out survival margin against every *distinct* saved checkpoint genome in
that run. Thus every checkpoint faces a common strategy panel, and duplicate
genomes are not allowed to overweight that panel.

| Metric | Pair | Triad exposure | Triad case | Triad compute |
|---|---:|---:|---:|---:|
| Mean final best absolute survival | 0.875 | 0.394 | **0.881** | 0.790 |
| Mean final population survival | 0.392 | 0.207 | 0.361 | **0.396** |
| Mean final population median | 0.378 | 0.209 | 0.263 | **0.416** |
| Robust strength tail slope/gen | +0.0145 | -0.0037 | +0.0185 | **+0.0258** |
| Seeds with positive robust tail slope | 3/3 | 1/3 | 3/3 | 3/3 |
| All later-vs-earlier wins | 81/108 | 69/108 | **92/108** | 89/107 |
| Adjacent checkpoint wins | 16/24 | 16/24 | 17/24 | **17/23** |
| Final-vs-earlier wins | 15/24 | 12/24 | **20/24** | **20/23** |
| Final per-world case wins | 125/192 | 94/192 | 155/192 | **155/184** |
| Tail chronological wins | 19/30 | 15/30 | **25/30** | 23/29 |
| Total evolution worlds, 3 seeds | 46,080 | 15,360 | 30,720 | 46,080 |

The compute arm has reduced denominators because seed 17's generation-39
champion is byte-identical to generation 35. Crossplay correctly omitted their
clone-versus-clone cell.

The cost result is important. The case-matched triad used one third fewer worlds
than pairwise training while improving later-vs-earlier ordering from 75.0% to
85.2% and final retention from 62.5% to 83.3%. The compute-matched triad raised
mean robust tail slope further, but did not improve chronological wins over the
case arm.

## Per-seed tail result

| Arm / seed | Final best | Final pop mean | Robust tail slope | Tail wins | Final wins | Tail interpretation |
|---|---:|---:|---:|---:|---:|---|
| Pair 7 | 0.865 | 0.494 | +0.0104 | 6/10 | 4/8 | Large gain at 30, then regression |
| Pair 17 | 0.959 | 0.369 | +0.0292 | 9/10 | 7/8 | Strong continuing progress |
| Pair 27 | 0.801 | 0.312 | +0.0038 | 4/10 | 4/8 | Strategy cycling, late recovery |
| Exposure 7 | 0.605 | 0.315 | -0.0145 | 0/10 | 1/8 | Severe monotonic tail regression |
| Exposure 17 | 0.172 | 0.115 | -0.0011 | 6/10 | 3/8 | Low-competence plateau/regression |
| Exposure 27 | 0.405 | 0.192 | +0.0045 | 9/10 | 8/8 | Progressive predatory omnivore |
| Case 7 | 0.957 | 0.440 | +0.0065 | 8/10 | 6/8 | Champion improves; population collapses |
| Case 17 | 0.747 | 0.309 | +0.0192 | 8/10 | 7/8 | Large late innovation, then partial reversal |
| Case 27 | 0.937 | 0.334 | +0.0298 | 9/10 | Strong late gains; final below generation 35 |
| Compute 7 | 0.933 | 0.493 | +0.0410 | 8/10 | 6/8 | Large gain after 25; slight final reversal |
| Compute 17 | 0.483 | 0.187 | +0.0046 | 5/9 | 6/7 | Gain through 35, exact repetition at 39 |
| Compute 27 | 0.954 | 0.507 | +0.0318 | 10/10 | Cleanest continuing tail |

Positive fitted slope must not be confused with monotonic progress. For example,
pair seed 7's common-panel margin jumps from `-0.016` at generation 25 to
`+0.420` at 30, but falls to `+0.141` by 39. Case seed 27 rises from `+0.156`
at 20 to `+0.682` at 35, then falls to `+0.619`. Compute seed 17 stops changing
entirely between 35 and 39. Only compute seed 27 wins all ten tail chronological
comparisons.

## What caused the treatment effect?

### Equal exposure is not enough

Pair and exposure-matched triad both expose a genome to eight opponents. The
triad arm sees them two at a time, so it has only four independent world
memberships and 16 scored cases rather than eight memberships and 32 cases.
That arm has by far the worst result: lower competence, negative mean tail
slope, 50% final retention, and substantial population regression.

This falsifies the simple idea that simultaneous multi-lineage context is
automatically more robust. With the current noisy ecological evaluator, the
number of independent scored contexts matters.

### Restoring 32 cases changes the result

Going from exposure-matched to case-matched triads doubles memberships from four
to eight, cases from 16 to 32, opponent exposures from 8 to 16, and worlds from
128 to 256. That produces the largest improvement in the experiment. Because
these quantities move together, the experiment cannot attribute the jump to
one of them individually. It does establish a scalable rule: **do not reduce
below roughly eight independently seeded world memberships / 32 scored cases
per genome merely because a shared world scores three lineages efficiently.**

### More than 32 cases has diminishing, mixed returns

The compute-matched triad increases from 32 to 48 cases and from 16 to 24
opponent exposures. It improves the robust tail slope and population median,
but slightly worsens all-checkpoint chronological ordering and final best
survival. At this scale, 32 cases appear sufficient for selection; 48 cases buy
additional stability rather than a uniformly better champion trajectory.

### What triads plausibly contribute

Pair and case-matched triad both have 32 cases. The triad uses fewer worlds and
packs two simultaneous opponents into every case, yielding twice the opponent
exposure. It improves held-out ordering and final retention, although its
population median is worse. That is consistent with triads providing useful
joint ecological context and opponent-exposure efficiency. It is not a pure
causal isolation of “simultaneity,” because the triad necessarily contains two
opponents per case.

## Behavior and ecology

Triads did not prevent plant convergence.

- Eight of the twelve final champions behave as pure foragers on the held-out
  common panel: all three pair champions; exposure seed 7; case seeds 7 and 27;
  and compute seeds 7 and 27.
- The meaningful exceptions are all seed-dependent. Exposure seed 17 ends as a
  low-coverage predator, exposure seed 27 as an omnivore averaging about 45
  held-out kills, case seed 17 as an omnivore averaging 7.8 kills, and compute
  seed 17 as a stable omnivore averaging about 37 kills.
- Pair seed 27 cycles from a 44-kill omnivore at generation 20, to a pure
  forager at 30, back to a 50-kill omnivore at 35, and finally to a pure forager
  at 39. That is behavioral change and competitive recovery, but not cumulative
  retention of capabilities.
- Case seed 17 shifts from an 18.6-kill omnivore at generation 20 to a broad
  mobile near-forager at 35, then back toward an omnivore at 39 while losing
  some of generation 35's strength. This is a genuine late strategy change, but
  again not monotonic accumulation.
- Case and compute seed 27 obtain their strongest gains as full-coverage pure
  plant foragers. Mini-ecosystems do not by themselves create a predator-prey
  arms race when plants remain the universal profitable resource.

Structural innovation also continued after generation 20 in every arm. Across
the three runs, pair/case/compute generated respectively 28/24/29 new node
innovations in the tail. Continued NEAT complexification therefore was not the
limiting factor, and structural novelty should not be mistaken for behavioral
capability.

As a conservative behavior-discovery diagnostic, `analysis.json` standardizes
held-out action, plant/prey, kill, coverage, information, and action-fraction
features within each run. It calls a late checkpoint “behaviorally novel and
stronger” only when it both beats the best pre-20 common-panel strength and is
farther from every pre-20 behavior than the median pre-20 nearest-neighbor
distance. Counts are pair `5`, exposure `3`, case `4`, compute `4`. This is a
useful observational screen, not proof of qualitative novelty. Crucially, the
pair control is not behaviorally exhausted by generation 20.

## Did any arm avoid plateau?

The narrow answer is **yes, progress often continued beyond generation 30**:
pair, case, and compute all had positive robust tail slopes on every seed, and
case/compute produced strong post-30 held-out gains.

The stronger answer is **not yet reliably**:

- only compute seed 27 was monotonically superior across all tail checkpoint
  pairs;
- case seed 7 and 27 populations lost roughly 22% and 32% of their best tail
  mean survival by generation 39;
- several final champions lost to an earlier tail champion;
- compute seed 17 exactly repeated generation 35 at generation 39;
- plant-specialist convergence remains dominant.

Thus the treatments can move or interrupt a plateau, but 40 generations do not
show sustained cumulative improvement across seeds.

## Recommendation

1. **Adopt the case-matched triad as the next screening evaluator**, not the
   exposure-matched arm. It gives the best competence ordering per simulator
   world: 32 cases, 16 exposures, and 256 worlds/generation.
2. **Keep pairwise frozen crossplay as the common assay.** It revealed both late
   gains and reversals that contextual training fitness obscures.
3. **Run pair versus case-matched triad to 80 generations on the same three
   seeds first.** Predeclare the generation-40–79 common-panel slope, final
   retention, and population mean/median gates. This directly tests whether the
   apparent late progress persists or is merely a later plateau.
4. **If the case triad retains its advantage, test population 24 versus 48 under
   that fixed 32-case-per-genome evaluator.** Keep evaluation precision per
   genome constant and allow the larger-population arm to cost twice as much.
   That is the clean next causal test of whether the remaining plateau is
   search-capacity limited.
5. **Confirm the winning evaluator/capacity treatment on more evolutionary
   seeds before changing brain or ecology.** This three-seed experiment has
   strong seed interactions and cannot justify a representation or ecology
   redesign.

Do not sweep four and five lineages yet. This experiment shows that lowering
independent cases is dangerous and that returns from 32 to 48 cases are already
mixed. A larger-lineage sweep without a fixed membership/case contract would
mostly rediscover evaluation-budget effects.

## Artifacts

- `COMMANDS.md`: exact contract and commands
- `run-one.sh`: evolution launcher
- `run-crossplay.sh`: held-out assay launcher
- `analyze.mjs`: reproducible derivation
- `analysis.json`: complete aggregate, per-run, checkpoint, behavior, and
  comparison data
- `<arm>-seed-<seed>/`: raw schema-20 result, champion world, metric sidecar,
  progress log, wall time, and held-out crossplay

This report does not claim open-endedness. Forty generations can reject an
early fixed plateau and compare evaluators; it cannot establish an unbounded
tail.

## Validation

- `make fmt`: passed.
- `make lint`: passed with warnings denied.
- `cargo check --workspace`: passed.
- `cargo test --workspace`: reached the previously known failure
  `lethal_attack_spawns_corpse_food_without_feeding_attacker` (`left: 1`,
  `right: 0`).
- `cargo test --workspace -- --skip
  lethal_attack_spawns_corpse_food_without_feeding_attacker`: passed in full.
- Every run directory contains exactly one result JSON, champion world, metric
  sidecar, 40 generation progress rows, wall-time record, and held-out crossplay
  result.
