# Two lineages versus three lineages

## Result

Three-lineage mini-ecosystem evaluation improved **final-checkpoint retention**
and mean final absolute survival at the same simulator-world budget, but it did
not produce a uniformly stronger evolutionary process and did not prevent
convergence toward plant exploitation.

This is evidence in favor of retaining the configurable three-lineage evaluator
for further experiments. It is not evidence of open-endedness, and it is not a
reason to replace two-lineage evaluation yet: the experiment is only 25
generations and the treatment's competence effect is strongly seed-dependent.

## What was compared

Both arms are contemporary-only NEAT. There is no historical archive or hall of
fame.

- Two-lineage world: two distinct current-generation genomes coexist, with 51
  clonal founders each.
- Three-lineage world: three distinct current-generation genomes coexist in one
  simulation, with 34 clonal founders each. This is one shared ecosystem, not
  two independent pairwise cases. All three lineages receive a survival score
  from that simulation.
- Shared: population 24, generations 25, horizon 5,000, training seeds
  `11,29,47,61`, run seeds `7,17,27`, baseline scenario, predation enabled,
  50x50 world, 102 total founders.
- Equal simulator work: both arms ran 384 worlds/generation, 9,600 worlds/run,
  and 28,800 worlds/arm. Pairwise used eight world memberships/genome; triadic
  used twelve memberships/genome. Sharing each triadic world across three
  scored lineages yields 48 cases/genome and 96 simultaneous opponent exposures
  versus 32 cases and 32 exposures pairwise, without increasing world count.

The selected objective remains `absolute survival * relative survival
advantage`. For three lineages, relative advantage compares the focal lineage's
mean alive-ticks to the founder-weighted mean of the other two lineages. Neutral
is still 1 and the limit is still 2. Unlike symmetric pairwise play, the
population mean of this nonlinear three-lineage relative term is not guaranteed
to equal 1. Therefore cross-arm claims below use absolute survival and frozen
pairwise crossplay, not raw fitness.

## Training trajectories

| Seed | Pair final best | Triad final best | Pair final population mean | Triad final population mean |
|---:|---:|---:|---:|---:|
| 7 | 0.920 | 0.967 | 0.478 | 0.514 |
| 17 | 0.515 | 0.346 | 0.245 | 0.173 |
| 27 | 0.392 | 0.871 | 0.208 | 0.443 |
| **Mean** | **0.609** | **0.728** | **0.310** | **0.376** |

On average, triadic training ended 0.119 higher in best absolute survival
(+19.5%) and 0.066 higher in population-mean survival (+21.3%). This average is
not uniform robustness: triadic training was much better on seed 27, modestly
better on seed 7, and materially worse on seed 17.

Neither treatment has a clean improving tail. Mean late-generation slopes were:

- best survival: pairwise `+0.00192/generation`, triadic `+0.00012/generation`;
- population-mean survival: pairwise `-0.00131/generation`, triadic
  `-0.00087/generation`.

Both still show population-level late stagnation or mild regression. The mean
final drop from each run's maximum best survival was 0.058 pairwise and 0.067
triadic.

## Held-out frozen-checkpoint crossplay

Each run's generations `0,5,10,15,20,24` were replayed in the exact symmetric
two-lineage evaluator on eight unseen seeds
`101,131,151,181,211,241,271,311`. This deliberately uses pairwise worlds as a
common post-training assay, even for triad-trained genomes.

| Retention test | Pairwise training | Triadic training |
|---|---:|---:|
| Later checkpoint wins, all chronological pairs | 38/45 | 38/45 |
| Later wins, adjacent checkpoints | 13/15 | 13/15 |
| Final checkpoint wins against earlier checkpoints | 11/15 | **14/15** |
| Final checkpoint wins, individual held-out cases | 86/120 (71.7%) | **104/120 (86.7%)** |

The broad chronological ordering is tied. The decisive positive result is at
the final checkpoint: triadic training retained competence against its own
history much more reliably.

Per seed, final-checkpoint matchup wins were:

| Seed | Pairwise | Triadic |
|---:|---:|---:|
| 7 | 2/5 | 4/5 |
| 17 | 4/5 | 5/5 |
| 27 | 5/5 | 5/5 |

Triadic seed 7's only final loss was to generation 0, so the ancestral predator
counterstrategy was still forgotten. Pairwise seed 7 lost to generations 0,
15, and 20. Triadic training reduced historical-context failure; it did not
eliminate it.

## Behavior and trophic outcome

Three lineages did not reliably preserve predators or prevent plant
specialization.

- Seed 7: the triadic final population was 24/24 foragers; its champion had no
  prey intake or kills. Pairwise retained 7 nominal omnivores, although its
  champion was also a pure forager.
- Seed 17: triadic training did preserve a meaningful predatory component. The
  final winner averaged 23.5 kills and prey rate 0.000415, versus 0.19 kills and
  prey rate 0.000002 for the pairwise winner. However, its absolute competence
  was lower.
- Seed 27: pairwise retained a predatory omnivore winner (46.4 kills), whereas
  triadic training produced a pure forager with zero kills. The triadic forager
  nevertheless survived far better.

Across final populations, pairwise contained 3 predator labels and triadic
contained none. Triadic mini-ecosystems can support a predator in one run, but
the treatment alone does not establish stable trophic diversity. Plant energy
remains the universal resource and often dominates selection.

## Cost and integrity

- Both arms report exactly 384 evaluator worlds/generation and 9,600/run.
- Concurrent wall times averaged about 1,348 seconds pairwise and 1,366 seconds
  triadic; that is close enough to call compute matched, but it is not a clean
  benchmark because all six jobs shared the same machine and survival itself
  changes per-tick work.
- Two identical triadic smoke runs produced byte-identical result JSON SHA-256:
  `ed0d74ac3d552b2522d9f786024cda2eac02a604223b1b49576bfa84284e15d1`.
- `make fmt` and `make lint` pass.
- `cargo test --workspace` reaches the known pre-existing failure
  `lethal_attack_spawns_corpse_food_without_feeding_attacker` (`left: 1`,
  `right: 0`). The complete workspace passes when that one test is skipped.

## Interpretation and next experiment

The simple answer is: **three lineages look useful for robustness, especially
final retention, but are not a standalone solution.** They reuse each expensive
simulation more efficiently and expose genomes to joint opponent contexts. In
this experiment that improved final historical crossplay without adding world
compute.

The next paired experiment should keep this evaluator and isolate the remaining
confound:

1. repeat on more run seeds or 50 generations;
2. compare equal-world-budget triads (this treatment) with an
   equal-opponent-exposure triad arm, because the present efficiency gain
   supplies three times as many opponent exposures;
3. combine triads with a separate ecological-role/minimal-criterion mechanism
   if preventing plant monoculture is the goal; merely adding a second
   contemporary opponent did not do that;
4. continue using frozen two-lineage checkpoint crossplay as the common assay,
   since it makes retention comparable across training treatments.

Raw commands are in `COMMANDS.md`; derived data are in `analysis.json`; every
run directory contains its complete schema-20 result, progress log, inspectable
champion world, and held-out crossplay JSON.
