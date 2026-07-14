# Absolute-survival objective audit

## Bottom line

The `absolute survival x relative survival advantage` objective is **not the
primary cause** of predation or late stagnation in the 10%-plant experiment.

Pure absolute survival and the actual objective rank the 48 genomes very
similarly (mean within-generation Spearman correlation `0.969`). They select
the exact same champion in 159/240 generations and 88/120 late generations.
When they disagree, the actual champion is usually the second-best absolute
survivor, not a qualitatively different genome.

Most importantly, retrospective pure-survival winners remain just as
predatory, do not win direct held-out contests reliably, and form a slightly
less progressive late checkpoint sequence. Removing the relative multiplier
may modestly change aggression, but it is unlikely to solve the founder-energy
tournament or the post-generation-30 plateau.

## Scope

This audit uses every persisted member of every generation from the three
completed triadic population-48, 10%-plant runs. For each population it
compares:

- the actual checkpoint champion selected by mean
  `absolute survival x relative survival advantage`;
- the member with maximum mean absolute survival fraction.

It then places both checkpoint families in one held-out crossplay roster using
pure `survival_fraction`, 5,000 ticks, and the same eight unseen world seeds.
Actual checkpoint IDs are `0..79`; absolute-winner IDs are `1000..1079`.

This is retrospective. Descendant populations were produced by the original
objective, so the audit does not show the counterfactual trajectory of an
absolute-only evolution run.

## Training-population reranking

| Measure | All 240 generations | Late 120 generations |
|---|---:|---:|
| Exact same champion | 159 (66.3%) | 88 (73.3%) |
| Mean Spearman rank correlation | 0.969 | 0.975 |
| Actual champion's mean absolute rank | 2.20 | 1.93 |
| Actual champion's median absolute rank | 1 | 1 |
| Actual champion outside absolute top five | 16 | 6 |
| Mean survival gained by choosing absolute winner | 0.00187 | 0.00176 |
| Actual winner relative advantage | 1.290 | 1.309 |
| Absolute winner relative advantage | 1.260 | 1.285 |

The objectives disagree fairly often because several genomes have nearly tied
fitness, but the overall ranking and especially the top of the ranking are
almost the same. The mean training gain from replacing the selected genome is
only 0.19 percentage points of the episode horizon.

The actual objective does behave as designed: its winner has slightly higher
relative advantage. But that pressure is a small perturbation on absolute
survival rather than a wholesale reordering of strategies.

## Predatory behavior

Pure-survival selection does not replace predators with foragers.

| Late training winner behavior | Actual objective | Pure absolute survival |
|---|---:|---:|
| Predator checkpoints | 49/120 | 52/120 |
| Omnivore checkpoints | 71/120 | 68/120 |
| Prey intake fraction | 68.6% | 68.1% |
| Mean kills | 34.2 | 29.9 |
| Mean total energy obtained | 12,543 | 10,974 |
| End-survivor fraction | 1.46% | 1.26% |

Absolute winners are, if anything, nominally predators slightly more often.
Their prey-energy fraction is essentially unchanged. The actual objective does
favor higher-kill, higher-energy genomes within the same broad trophic regime,
but it did not invent that regime.

This revises the causal explanation: predation is adaptive under absolute
lineage survival itself. Stealing energy from another lineage genuinely extends
the focal lineage's integrated alive-ticks. The relative multiplier rewards
opponent harm a second time, but the first and larger incentive already exists
in the absolute objective.

## Held-out audit

Thirty-four checkpoint labels per seed were entered into a shared roster. Exact
genome duplicates were excluded automatically, leaving 20-23 distinct genomes
per seed.

### Direct actual-versus-absolute contests

Only 15/51 saved checkpoint pairs use different genomes; the other 36 are exact
agreements.

| Measure | Result |
|---|---:|
| Pure-survival winner wins | 7/15 |
| Actual winner wins | 8/15 |
| Mean pure-minus-actual survival margin | -0.00946 |
| Late pure-survival wins | 3/6 |
| Late mean margin | -0.00621 |

The training absolute winner therefore does not generalize better in direct
competition. On average the actual objective's winner survives longer.

Among differing checkpoints, actual winners average 31 kills versus 19.6 for
absolute winners, while both obtain roughly 77-78% of intake from prey. The
relative term selects more aggressive variants, but those variants are not
merely reward-hacking their training panel: they remain at least as competitive
on unseen seeds.

### Common-roster strength

Against the intersection of identical third-party opponents:

- 36 checkpoint pairs are exact ties because the selected genome is identical.
- Pure-survival winners are stronger in 11 differing checkpoints.
- Actual winners are stronger in 4 differing checkpoints.
- Mean pure-minus-actual strength is `+0.00289`.

This is the one favorable result for pure survival. It is small and asymmetric:
the positive differences average `+0.0148`, while the four negative differences
average `-0.0037`. Most of the benefit comes from seed 17.

In the late window only six checkpoints differ: pure survival is stronger in
three and the actual objective in three. Mean margin is `+0.00245` for pure
survival.

Thus pure survival occasionally selects a substantially more broadly robust
genome, even though it does not win direct pairings more often. This is a
fitness-precision observation, not an anti-predation result.

## Does pure survival improve tail progress?

No.

| Late chronological ordering | Actual winners | Pure-survival winners |
|---|---:|---:|
| Later checkpoint wins | 57/108 | 51/108 |

Per seed:

| Seed | Actual | Pure survival | Interpretation |
|---:|---:|---:|---|
| 7 | 11/36 | 12/36 | negligible change; both regress |
| 17 | 28/36 | 29/36 | small improvement on the one progressive seed |
| 27 | 18/36 | 10/36 | pure-survival replacements lose the final counterstrategy path |

Common-roster late slopes average `+0.000243` for actual winners and
`+0.000447` for absolute winners, but the difference is driven by seed 17.
Seed 7 remains negative and seed 27 becomes slightly more negative. The
chronological count is the more adversarial result: pure survival does not make
later genomes win more consistently.

## Mechanistic conclusion

The earlier hypothesis was incomplete. The relative multiplier does encourage
opponent suppression, but removing it would not remove the underlying payoff:

1. Every episode resets 102 organisms with a large free energy endowment.
2. A focal lineage can convert opponent energy into additional focal
   alive-ticks.
3. That conversion improves pure absolute survival even if the world later
   collapses.
4. Plants are comparatively spatially difficult and low-value per capture.
5. No within-world reproduction makes prey depletion costless beyond the fixed
   horizon; the next evaluation provides new prey.

Predation is therefore not primarily a mathematical artifact of relative
fitness. It is an adaptive response to the evaluator's energy topology: fresh
founder biomass is the easiest valuable resource.

The relative term is still not obviously necessary. It changes close champion
choices, increases aggression, and may add variance. But an absolute-only run
should be viewed as an objective-simplification experiment, not as the expected
solution to predatory collapse or stagnation.

## Next diagnostic

The next causal audit should track **energy provenance and victim identity**:

- energy originating in founder endowment versus plants;
- prey energy that itself originated in plants versus founder endowment;
- cross-lineage versus same-lineage kills;
- survival gained per unit of stolen founder energy.

That would directly test the founder-liquidation mechanism. A cheaper immediate
replay is to extend representative predator and omnivore checkpoint matchups to
10,000 ticks and measure whether predator advantages disappear after prey
exhaustion. If they do, the necessary intervention is ecological—altering
renewable energy access, predation transfer economics, or prey renewal—not
merely changing the scalar objective.

## Artifacts

- `selection-rows.json`: every member's original and absolute-survival rank
- `combined-seed-{7,17,27}.json`: compact combined checkpoint sources
- `crossplay-seed-{7,17,27}.json`: held-out combined matrices
- `analyze.mjs`: reproducible derivation
- `analysis.json`: full aggregate and per-seed results
- `COMMANDS.md`: exact commands and audit limitations
