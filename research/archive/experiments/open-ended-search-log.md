# NeuroGenesis research log

## 2026-07-10 — fresh baseline after in-world reproduction removal

### Research question

Does the current NEAT outer loop provide evidence for continuing adaptive genome
complexification or a bottomless cognitive learning signal, and which available
selection mechanisms are worth pursuing next?

This round deliberately ignored the deleted historical research corpus and used
the current engine and durable result artifacts as authority.

### Adversarial convergence

Three independent reviews rejected the initial broad claims:

- `eval_opponents > 0` is not yet an arms-race test. It changes founder
  heterogeneity and clone multiplicity, samples different contemporary
  opponents per candidate, and has no frozen or historical cross-play.
- novelty-local competition was uninterpretable in mixed worlds because its
  descriptor was accumulated over the candidate and its sampled opponents.
- the curriculum is a scalar difficulty dial, not endogenous environmental
  complexification. Moving training fitness is not comparable across levels.

The converged screens were therefore:

1. structural mutation ON versus OFF in isolated evaluation;
2. fitness versus candidate-scoped NSLC in isolated evaluation;
3. contemporary mixed evaluation versus isolated evaluation, described only as
   a package effect and compared on common isolated audits/holdouts.

All NEAT screens used two outer-loop seeds (101, 202), population 40, 40
generations, 1,000-tick episodes, fixed training seeds 11/29/47, development
seeds 61/79/97, sealed seeds 131/149/167, fixed audit levels 0/1/2/4, and no
training-seed rotation. This is screening evidence (`n=2`), not proof of
open-endedness.

### Instrumentation repaired before experimentation

The no-reproduction rewrite had retained schema-v2 reproduction semantics.
Before running screens, the evaluator was changed to:

- bump NEAT result schema to v3;
- remove births, mature-offspring, and maturity-followup outputs;
- report explicit world founders, candidate founders, candidate end survivors,
  world final population, and realized founder-pool size;
- scope consumption, first-food time, action fractions, and spatial coverage to
  the candidate lineage, making isolated NSLC and mixed candidate diagnostics
  meaningful;
- cap a requested mixed founder pool to the scenario's actual founder count and
  require every realized pool entry to be represented;
- correct CLI objective text from offspring maturity to candidate
  survival-time fraction.

`cargo check --workspace` passes. `cargo test --workspace` compiles and runs but
one human-maintained predation test is stale: it expects a lethal attack to
produce zero consumptions plus a corpse, while the current engine explicitly
feeds the attacker on kill and suppresses that corpse.

### Round 1A — structural mutation ON versus OFF

Artifacts: `artifacts/research/runs/completed/round-001/structure/`.

| Seed | Arm | Sealed | Development | Adaptive connections/nodes | Structure KO delta dev/sealed |
|---:|:---|---:|---:|---:|---:|
| 101 | ON | 0.820353 | 0.799298 | 6 / 1 | +0.539512 / +0.566732 |
| 101 | OFF | 0.713390 | 0.764098 | 0 / 0 | 0 / 0 |
| 202 | ON | 0.836860 | 0.853948 | 3 / 1 | +0.027853 / +0.026563 |
| 202 | OFF | 0.880200 | 0.884927 | 0 / 0 | 0 / 0 |

Conclusion: NEAT can discover retained, causally useful added structure. Both ON
champions depend positively on evolved topology on development and sealed
audits. But structural mutation did not robustly improve final competence:
matched effects split by seed (mean sealed ON-OFF `+0.031812`, mean development
`+0.002111`). The mechanism is functional; continuing or general adaptive
complexification is not established.

### Round 1B — candidate-scoped novelty-local competition

Artifacts: `artifacts/research/runs/completed/round-001/novelty/`.

| Seed | Fitness sealed/dev | NSLC sealed/dev | NSLC endpoint best/mean novelty | NSLC adaptive C/N |
|---:|---:|---:|---:|---:|
| 101 | 0.820353 / 0.799298 | 0.730768 / 0.726112 | 0.198185 / 0.146649 | 0 / 0 |
| 202 | 0.836860 / 0.853948 | 0.800842 / 0.796898 | 0.195662 / 0.149841 | 4 / 1 |

Conclusion: reject NSLC as the next default. It loses sealed and development
competence in both matched seeds. Endpoint novelty is below generation zero in
both runs, while archive size grows mechanically to 74/78. Structural evidence
is inconsistent and late fixed-audit slopes regress in both NSLC runs.

### Round 1C — mixed-evaluation package

Artifacts: `artifacts/research/runs/completed/round-001/mixed/`.

| Seed | Opponents | Training survival | Isolated dev | Isolated sealed | Expressed H/C | Structure KO dev/sealed |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0 | 0.844053 | 0.799298 | 0.820353 | 2 / 17 | +0.539512 / +0.566732 |
| 101 | 4 | 0.763033 | 0.729383 | 0.738190 | 0 / 12 | -0.005648 / +0.000247 |
| 202 | 0 | 0.899573 | 0.853948 | 0.836860 | 1 / 13 | +0.027853 / +0.026563 |
| 202 | 4 | 1.000000 | 0.848987 | 0.864658 | 1 / 12 | +0.001267 / +0.010157 |

Paired mixed-minus-isolated mean effect is `-0.037438` on development and
`-0.027183` on sealed evaluation. Seed 101's mixed-selected champion consumes
zero plants on the common isolated audit and covers about 4% of the world;
seed 202's mixed-selected champion is a strong forager. Mixed evaluation is not
an obviously beneficial default and does not yet show an arms race.

### Round 2 — frozen-champion durability replay

Artifacts: `artifacts/research/runs/completed/round-002-horizon/`.

This follow-up tested isolated clonal expression of the four selected genomes,
not replay of mixed competition. Worlds were advanced through ticks 250, 500,
750, 1,000, 1,200, 2,000, and 4,000. Phase rates were derived by differencing
cumulative totals because the 10,000-tick metric sidecar makes pre-10k pillars
cumulative. The stale genome `max_organism_age` field is non-operative: current
lifecycle execution kills only on energy depletion.

Seed 101:

- mixed-selected champion: zero consumption at every checkpoint; all 30
  founders starve by approximately tick 767;
- isolated-selected champion: active foraging throughout; 2,156 cumulative
  consumptions and 14/30 survivors at tick 4,000.

Seed 202:

- both champions retain 30/30 founders with zero deaths through tick 4,000;
- late tick-2,000-to-4,000 consumption rates remain stable at 0.838/tick
  (isolated-selected) and 0.871/tick (mixed-selected);
- mean energy rises in both populations through tick 4,000.

Conclusion: finite-horizon passive-runway exploitation is real and severe but
seed-dependent. In seed 101, the mixed-selected fitness `0.763033` nearly
measures how long a zero-consumption policy coasts on founder energy. Seed 202
falsifies the universal version: mixed selection also found a genuinely durable
forager. The bimodality is itself the result—the current objective admits both
sustainable competence and a shallow reserve-burning solution.

### Synthesis

Substantial positive evidence:

- the outer loop can create added topology that is causally required by a
  champion on disjoint and sealed worlds;
- the 1,000-tick horizon is long enough to create a nontrivial gradient;
- at least some selected controllers sustain energy-positive foraging far
  beyond their training horizon.

Substantial negative evidence:

- the bounded survival-area objective is not bottomless and admits passive
  energy-runway solutions;
- contemporary mixed evaluation does not reliably improve transferable
  competence or select added topology;
- current NSLC expands its archive without expanding endpoint behavioral
  novelty or competence;
- no current result measures endogenous environmental complexification or a
  Red Queen against historical opponents.

### Next hypotheses and instrument priorities

1. Replace or augment fixed-horizon survival area with a sustainability signal
   that cannot be earned by burning founder reserves: score phase-local net
   energy balance or harvest per organism-tick after a genome-specific passive
   runway, and evaluate across randomized/log-spaced horizons.
2. Add a first-class `sim-cli neat replay/evaluate` mode for frozen champions on
   arbitrary common seeds, horizons, scenarios, and short reporting intervals.
   The ad-hoc world continuation cannot evaluate a fixed audit suite.
3. Before calling mixed evaluation coevolution, add common frozen opponent
   panels, serialize opponent identities/fingerprints, and generate historical
   champion cross-play matrices. Compare contemporary, frozen, and hall-of-fame
   opponents at matched density.
4. Keep fitness selection for the next iteration. Preserve structural mutation,
   but gate complexification claims on sealed knockout contribution and repeat
   the ON/OFF screen across more outer-loop seeds.
5. Treat scalar curriculum as a robustness tool only. A cognition ladder needs
   new causal affordances or organism-created challenges, not merely less food
   and higher metabolism.

## 2026-07-10 — relative survival and multi-horizon selection

### Round 3 hypothesis: sustainable competition

The proposed treatment preserved absolute survival while adding a bounded
frequency-dependent factor:

`R = 2 * candidate_mean_alive_ticks / (candidate_mean_alive_ticks + opponent_mean_alive_ticks)`

`fitness = absolute_survival_fraction * R`

`R` is in `[0,2]`, is neutral at `1`, and is independent of pool size. The
evaluator fails closed when the treatment cannot instantiate every requested
pool entry or represent entries with equal founder counts. Screens use two
opponents so the three-founder SparseSearch case remains balanced.

Adversarial review rejected relative survival alone and terminal-energy scoring.
Relative survival alone rewards denominator suppression and cannot escape a
universally passive population. Terminal energy selects hoarding/camping and is
pathologically unbounded rather than cognitively open-ended. The product was
approved only as a falsifiable mechanism probe.

### Instrumentation added

NEAT result schema v4 added:

- configurable `fitness_objective` (`survival_fraction` or
  `survival_times_relative_advantage`);
- per-case absolute survival, relative advantage, founder/alive-tick arrays,
  realized pool size, opponent population indices, and zero-denominator state;
- ordinary and lower-tail component summaries, duplicate-opponent accounting,
  and per-generation component trajectories;
- `sim-cli neat evaluate-panel`, which evaluates a persisted focal champion
  against explicit frozen champion opponents on common scenarios, seeds, and
  horizons.

### Frozen-panel gate

Artifacts: `artifacts/research/runs/completed/round-003-objective-gate/`.

Against two different mixed panels, the known seed-101 idler ranked below the
durable seed-101 focal at every horizon at or beyond 1,000 ticks. The margin
persisted through 4,000 ticks, and the idler consumed zero plants in every case.
This passed the positive-control gate.

The adversarial sham panel exposed the limit: idler versus two identical idlers
kept `R ~= 1` at every horizon and consumed nothing. The product collapses back
to absolute survival in an all-passive ecology; it cannot create an escape from
that equilibrium. Two durable seed-202 policies also reversed rank between
1,000 and 2,000 ticks, proving that a single cutoff remained a major confound.

### Multi-horizon evaluator

Schema v5 replaced the single NEAT episode cutoff with explicit strictly
increasing `episode_horizons`. Training and fixed audits pool every
`scenario x seed x horizon` case into the same lower-tail objective, and every
case records its horizon. This makes horizon transfer part of selection rather
than only a post-hoc replay.

### Round 4 matched evolutionary screen

Artifacts: `artifacts/research/runs/completed/round-004-multihorizon/`.

Four paired outer-loop seeds compared absolute survival with the product
treatment. Both arms used population 24, 24 generations, horizons
`[1000,2000,4000]`, two training seeds, two development seeds, two sealed seeds,
audit levels 0/2, and exactly two contemporary opponents. Seed is the
experimental unit; nested horizon/scenario cases are not independent
replicates.

| Seed | Absolute dev/sealed | Product dev/sealed | Product-minus-absolute | Product KO dev/sealed | Passive sealed cases per horizon abs/product |
|---:|---:|---:|---:|---:|---:|
| 101 | .254519 / .250131 | .138517 / .141196 | -.116002 / -.108935 | 0 / 0 | 3 / 12 |
| 202 | .175342 / .183627 | .253819 / .254181 | +.078477 / +.070553 | +.074487 / +.076360 | 0 / 0 |
| 303 | .182063 / .186127 | .180691 / .174294 | -.001372 / -.011833 | +.085198 / +.076919 | 0 / 0 |
| 404 | .142778 / .141852 | .255367 / .255694 | +.112589 / +.113842 | 0 / 0 | 0 / 1 |

Mean paired effects were `+0.018423` development and `+0.015907` sealed, but
signs split 2-2 and the means cancel large wins and losses. Mean sealed
treatment advantage shrank with horizon: `+0.073747` at 1,000,
`+0.045481` at 2,000, and `+0.021713` at 4,000.

The strongest positive case was seed 202: product selection improved isolated
survival at every horizon and found three adaptive connections/two adaptive
nodes with strong positive sealed knockout contribution. Seed 303 also selected
causally necessary structure despite slightly worse survival.

The decisive failure was seed 101: product selection chose a generation-zero,
zero-consumption idler because `R ~= 1.50` outweighed worse absolute survival.
It was passive in every fixed case and transferred about 44-46% worse. Across
all seeds, sealed zero-consumption cases per horizon increased from `3/48` under
absolute selection to `13/48` under the product.

### Round 4 conclusion

Reject `survival_times_relative_advantage` as the default or as the bottomless
signal. Retain it as a measured component and mechanism probe. It changes basin
probabilities and can sometimes select opponent-relevant causal structure, but
it does not reliably remove the passive basin, its advantage decays with
horizon, and complexification evidence appears in only half the seeds.

This is a stronger negative result than the frozen gate: hand-picked frozen
policies were ranked sensibly, but evolution learned to exploit the new scalar.

### Next converged hypotheses

Two non-equivalent directions survived adversarial review:

1. **Dense late-weighted absolute survival with durable anchors.** Use a single
   4,000-tick rollout split into four windows weighted `[1,2,4,8]`, with focal +
   one frozen durable anchor + one contemporary opponent. Every extra alive tick
   remains beneficial, later persistence matters more, and no opponent appears
   in the score denominator. A 2x2 uniform/weighted x contemporary/anchored
   experiment isolates time credit from anti-collapse anchoring.
2. **Direct strategic coupling with historical cross-play.** Enable opponent
   sensing and predation, freeze champions periodically, and measure reciprocal
   checkpoint cross-play. The required evidence is expanding historical
   coverage plus opponent-conditional causal structure, not merely a static
   cycle or more nodes.

The first addresses the demonstrated optimization failure. The second changes
the source of adaptive problems and is closer to the actual open-endedness goal.
Lower founder energy, late-window products, passive-runway subtraction, and
terminal-energy objectives were rejected as standalone fixes because they move
the cutoff, create flat zero plateaus, reward metabolically costly comparators,
or select hoarding.

## 2026-07-10 — Round 5: late-weighted anchors fail the frozen-policy gate

Schema v6 added a dense late-weighted absolute-survival diagnostic and frozen
durable anchors. A 4,000-tick episode was divided into four windows weighted
`[1,2,4,8]`; one fixed anchor could replace one contemporary opponent in each
case. The intended 2x2 evolutionary screen was preregistered behind a cheaper
policy-ranking gate.

Artifacts: `artifacts/research/runs/completed/round-005-weighted-anchor/gate/`.

The gate compared the known durable seed-101 isolated champion against the
known seed-101 mixed-population zero-consumption idler, using two durable frozen
anchors, three disjoint seeds, and curriculum level 0. It failed: the durable
focal mean late-weighted score was `.03010`, below the idler's `.03371`, and the
durable focal won only 3 of 9 cases. Both focal lineages went extinct in every
case while anchors monopolized the ecology. The durable focal paid for movement
and plant capture but often died earlier; the idler spent less and received a
slightly larger survival-area score.

Conclusion: reject late-weighted survival plus dominant durable anchors in this
form. The anchors did not stabilize a learning problem; they crowded out the
focal lineage and recreated selection for metabolic passivity. The planned
16-run 2x2 was stopped before execution.

## 2026-07-10 — Round 6: current predation interface is not a strategic substrate

Schema v7 persisted checkpoint champion genomes at audit/final generations and
added explicit prey-consumption fields to training and fixed-suite evaluations.
Four matched treatments enabled predation while retaining the Round 4 absolute
survival contract: population 24, 24 generations, horizons
`[1000,2000,4000]`, seeds 11/29 for training, 61/79 for development, 131/149
for sealed evaluation, scenarios baseline/scarcity/sparse-search, audit levels
0/2, and two contemporary opponents. Each treatment is compared to the exact
predation-off Round 4 control with the same outer seed.

Artifacts: `artifacts/research/runs/completed/round-006-predation-smoke/`.

| Seed | Training/dev/sealed mean prey | Dev score treatment/control | Sealed score treatment/control | Endpoint/durability and structure |
|---:|---:|---:|---:|---|
| 101 | .667 / 1.583 / 2.000 | .170658 / .254519 | .155079 / .250131 | Prey use transferred, but survival regressed 33-38%; selected champion KO deltas were zero |
| 202 | 0 / 0 / 0 | .218356 / .175342 | .215635 / .183627 | All focal founders dead before 1,000; strong positive KO signal encoded a passive runway policy |
| 303 | .500 / .500 / .333 | stronger CVaR, misleading | stronger CVaR, misleading | 99.1% Idle in training, zero plant consumption, no survivors by 2,000, no evolved-structure signal |
| 404 | 0 / 0 / 0 | .273842 / .142778 | .255995 / .141852 | No predation; Attack below .05%; late survival decayed and sealed KO was negative |

The positive prey counts in development and sealed suites are not reciprocal
opponent victories: fixed audits are isolated clonal colonies, so they measure
clone-on-clone cannibalism. Seed 101 proves only that the motor/perception path
can reach and consume organisms on unseen worlds. Seed 303 demonstrates the
same mechanism weakly while abandoning plant foraging. Seeds 202 and 404 never
realized a prey consumption despite occasionally selecting Attack.

All treatment records contain checkpoint genomes at generations 0, 8, 16, and
23. These are now sufficient for later historical cross-play, but cross-play is
not justified for the present champions: no seed demonstrated productive
between-lineage predation plus durable survival. The apparent treatment gains
in seeds 202-404 are survival-area artifacts: focal populations die early and
the score then falls approximately as `1/T` as the horizon doubles.

Conclusion: reject direct predation as currently exposed. It bundles a costly
sixth action and organism rays with undirected lethal cannibalism, but provides
no kin/lineage distinction or role asymmetry. Do not call this an arms race.
The next mechanism must make ecological interaction opponent-conditional or
role-differentiated, and future evaluation must attribute attacker and victim
founder pools before historical coverage can be interpreted.

### Round 7 preregistration check: morphology rays rejected before implementation

The first proposed repair was to split each organism ray into `smaller` and
`larger-or-equal` target channels. Three adversarial reviews independently found
the same fatal blocker in the current code: body size is derived from lifecycle
`gestation_ticks`, while the NEAT outer loop mutates only brain parameters and
topology. Every evaluated genome therefore retains the same size (300). A
smaller-target channel would be identically zero and the larger/equal channel
would reproduce the existing organism ray.

Making size evolvable was also rejected as the immediate repair. Larger bodies
currently receive more initial energy and health, hit harder, hit smaller prey
more reliably, regenerate more absolute health, and pay sublinear mass
metabolism without a speed or maneuverability penalty. The likely result is a
bounded maximum-size/endowment ratchet, not pursuit-evasion or open-ended roles.

The next diagnostic is therefore evaluator-only cross-pool predation: attacks
may affect organisms from other founder-pool entries but not the attacker's own
entry. This is deliberately magical kin immunity, not proposed architecture.
Its purpose is to falsify identity ambiguity cheaply. If predation remains
passive or non-durable with friendly fire removed, kin blindness is not the
binding constraint. If it succeeds, the next architectural step is an honest
heritable recognition signal plus attacker-to-victim pool attribution.

## 2026-07-10 — Round 7: magical kin immunity rescues foraging, not predation

Schema v8 added the evaluator-only `cross_pool_predation_only` diagnostic.
Founder source is already deterministically recoverable as
`species_id % pool_len`; when the oracle is enabled, an Attack against the same
source-pool entry has no effect. Fixed clonal audits therefore admit no legal
attacks and must record zero prey consumption. This is a falsification tool,
not proposed world architecture.

Artifacts: `artifacts/research/runs/completed/round-007-cross-pool-oracle/`.

Four matched treatments repeated Round 6 with only the oracle enabled.

| Seed | Training prey R7/R6 | Training survival R7/R6 | Dev score R7/R6 | Sealed score R7/R6 | Verdict |
|---:|---:|---:|---:|---:|---|
| 101 | 1.500 / .667 | .415694 / .372365 | .159341 / .170658 | .165529 / .155079 | Cross-pool prey reachable, but rare; durability still `~1/T` |
| 202 | 0 / 0 | .415975 / .414138 | .205274 / .218356 | .207807 / .215635 | Oracle suppresses Attack; passive runway unchanged |
| 303 | 0 / .500 | .6648 / .4275 | durable endpoints restored | durable endpoints restored | Removes collapse: Idle 99.1% -> 49.4%, plants 0 -> 627/case, but no cross-pool kill |
| 404 | 0 / 0 | .806206 / .685011 | .324170 / .273842 | .330966 / .255995 | Stronger durable forager, essentially zero Attack, no predation |

The oracle invariant held: development and sealed prey consumption was exactly
zero in every treatment case. Seed 101 increased training prey at nearly the
same Attack fraction and selected positive structure, but development survival
regressed while sealed survival improved; both remained finite-runway. Seed 303
is the important mechanism split. Removing friendly fire rescued ordinary
plant foraging and long-horizon endpoint survival dramatically, yet all 18
mixed training cases still had zero prey consumption. Seed 404 likewise became
a much stronger durable forager while selecting Attack only about once per
million actions.

Conclusion: kin-blind self-damage is a real cause of passive ecological
collapse, but it is not the binding constraint on a predator niche. Magical
immunity mostly allows evolution to rediscover plant foraging. The only
cross-pool predation signal is seed 101's very shallow 1.5 kills/case; there is
no sustained prey-energy replenishment or historical arms-race evidence.

The next causal hypothesis is kill completion. Equal-size attacks land, but do
only 50% max-health damage; movement resolves before interaction, Idle heals
10% max health, targets can leave adjacency, and the attacker has no target
health or prior-hit signal. Thus a kill requires repeated target acquisition
and action sequencing that the current interface does not expose. Before more
evolution, instrument attempts/hits/nonlethal damage/kills and gate a matched
one-hit-lethal diagnostic behind the cross-pool oracle.

## 2026-07-10 — Round 8: full Attack funnel localizes the bottleneck upstream

Schema v9 adds a sparse internal Attack event and serializes the candidate
funnel into every evaluation case: no-organism target, same-pool block,
eligible attempt, hit, nonlethal hit, same-attacker/same-victim follow-up and
latency, kill, damage, and direct energy transfer. Evaluator-only damage and
Idle-regeneration fractions are now part of the frozen run contract and
`neat evaluate-panel` accepts them directly.

Artifacts: `artifacts/research/runs/completed/round-008-attack-funnel/`.

A frozen 2x2 replay used the Round 7 seed-101 focal against the seed-303 and
seed-404 champions, 16 common seeds, baseline/scarcity/sparse-search at levels
0/2, and horizons 1,000/2,000. Every arm retained cross-pool-only predation.

| Arm | Damage | Idle regen | S at 1k / 2k | Endpoint at 1k / 2k | Kills/case at 1k / 2k |
|---|---:|---:|---:|---:|---:|
| A current | .5 | .10 | .684971 / .488938 | .4125 / .1885 | 11.229 / 11.771 |
| B no regen | .5 | 0 | .700380 / .506496 | .4417 / .2000 | 11.792 / 12.198 |
| C one hit | 1.0 | .10 | .718407 / .519445 | .4594 / .2125 | 12.052 / 12.438 |
| D one hit, no regen | 1.0 | 0 | .718407 / .519445 | .4594 / .2125 | 12.052 / 12.438 |

C and D have exactly identical evaluation arrays. This is the expected sanity
result: lethal hits leave no wounded target on which regeneration can act.
Relative to A, B improved paired case survival in 39/96 cases at 1,000 and
41/96 at 2,000 (22/23 losses); C improved 48/96 and 50/96 (22/20 losses).
Effects are real but modest and heterogeneous.

Arm A proves current combat is not stuck before damage. It averaged 34.75
eligible equal-size hits and 11.23 kills at 1,000, with 20.25 same-pair
follow-ups. By 2,000 it had only 0.54 additional kills. Arm B shows that without
healing, about 86% of nonlethal hits receive a same-pair follow-up at roughly
3.4 ticks and the two-hit mechanism converts almost exactly one kill per two
hits. Reacquisition is therefore functional, not absent.

The dominant loss is earlier in the funnel. In B, across both horizons, 70.7%
of selected Attacks resolved against no organism, 20.5% were blocked against
the focal's own pool, and only 8.8% were eligible cross-pool attempts. Every
eligible equal-size attempt hit. In the one-hit arms only 4-5% were eligible,
because killing on the first hit removes later repeat-hit opportunities.

One-hit predation remains strongly front-loaded: C gained only 37 additional
kills across all 96 cases from 1,000 to 2,000. Baseline and scarcity produced
kills in every seed/level case, but SparseSearch produced any kill in only
6/32 cases and had zero mean endpoint survival by 2,000. Easier finishing
therefore consolidates a finite founder-energy reservoir; it does not create
sustained opponent-generated difficulty or a bottomless signal.

Conclusion: regeneration and multi-hit completion are secondary constraints.
Do not adopt one-hit lethality. The primary predation bottleneck is
opportunity-conditioned action selection: the controller attacks at range or
at protected clones because generic organism rays expose distance but not a
directly usable attackable-contact/identity affordance. The next diagnostic
must separate neural discovery of that conjunction from encounter geometry and
combat economics; success must include late kills/energy, not merely more early
founder consolidation.

## 2026-07-10 — Round 9: checkpoint crossplay finds finite ladders, no Red Queen

The new `neat crossplay` instrument loads persisted checkpoint genomes,
computes stable SHA-256 identities and duplicate provenance, evaluates every
ordered checkpoint pair, and rotates the singleton focal through all three
founder-pool slots. It retains raw cases, absolute/relative survival, endpoints,
and the complete Attack funnel. A smoke matrix exposed up to `.03` relative
survival bias from deterministic founder-slot/ID ordering, validating balanced
slot averaging as mandatory.

Artifacts: `artifacts/research/runs/completed/round-009-crossplay/`.

Four independent Round 7 lineages (outer seeds 101/202/303/404) were assayed at
checkpoints 0/8/16/23. Every lineage had four unique checkpoint hashes. Each
matrix contains 144 aggregate cells and 9,216 raw cases: all 16 ordered pairs,
horizons 1k/2k/4k, three focal slots, 16 disjoint seeds, levels 0/2, and
baseline/scarcity under current `.5` damage and `.1` regeneration.

| Lineage | Historical ordering at 4k | Final coverage | 2k->4k predation energy | Durability | Verdict |
|---:|---|---:|---:|---|---|
| 101 | g16 > g23 > g8 > g0 | 2/3 | exactly 0 in every cell | every focal extinct by 4k | bounded gain then regression |
| 202 | weak g8 > g23 > g16 > g0 relation with two tie-band edges | 2/3 | exactly 0 for final vs every opponent | every final focal extinct by 2k | finite-runway ordering |
| 303 | g16 > g23 > g8 > g0 | 2/3 | no historical increase; mostly 0 | later checkpoints durable | robust foraging, not arms race |
| 404 | g23 ~= g16 > g8 > g0 | 2/3 | exactly 0; all matchups had 0 kills | later checkpoints durable | transitive foraging ladder plateaus |

Seed 101 showed the clearest early progression: coverage counts
`0 -> 1 -> 2 -> 2`, but g23 lost to g16 in every slot and all kills/energy had
already plateaued by tick 1,000. Seed 202's apparent dominance freezes after
extinction; 4k survival area is almost exactly half its 2k value. Seeds 303 and
404 prove that long-horizon survival is achievable, but their dominance comes
from plant foraging and persistence. Seed 303's aggressive g0 can extract far
more prey energy yet loses badly to later foragers; seed 404 records no kills
in any historical pairing.

No lineage contains a decisive replicated three-cycle. Transitivity is not
itself a failure—later-beats-earlier could be a valid ladder—but chronological
coverage stops or regresses at the final checkpoint in every seed. Slot ranges
reach roughly `.04-.07` relative-advantage units for some early diagonals, while
balanced slot means return near-neutral self-play; unrotated matrices would
have produced false edges.

Conclusion: reject historical archive opponents as the next selection change.
The current interaction game produces a short competence transition, then a
fixed ordering. An archive would preserve or overfit that finite benchmark; it
cannot manufacture counterstrategy turnover. The shared causal failure is that
opponent interaction and prey energy end early. Durable policies escape into
plant foraging, while predatory policies exhaust a finite founder reservoir.

The next substrate hypothesis must renew opponent-generated pressure during an
episode. A minimal candidate is evaluator-only opponent replenishment: when a
non-focal source-pool organism dies, respawn the same opponent genome so the
focal still has one life but faces an unending stream of the current strategy.
This preserves the outer-loop NEAT ownership of genetic variation while making
opponent difficulty and prey/competition pressure available at late horizons.
It is an arena diagnostic, not yet a claim about natural reproduction; energy
injection, spawn safety, pool balance, and farming exploits must be controlled
explicitly before adoption.

## 2026-07-11 — Round 10: renewable opponents expose an energy-source trilemma

Schema v10 adds evaluator-owned opponent renewal after each fully scored and
compacted tick. Focals remain one-life; non-focal pools are restored toward
their founder counts after a configurable delay using their exact genomes,
fresh brains, explicit source-pool identity, and deterministic domain-separated
empty-cell placement. The contract records respawns, injected energy, pool
deficit ticks, spawn failures/concentration, respawned-opponent kills and
latency. Kill reward is independently capped.

Artifacts: `artifacts/research/runs/completed/round-010-renewal/`.

The frozen factorial used the Round 7 seed-101 predator against seed-303 and
seed-404 opponents, all three focal slots, 16 disjoint seeds, levels 0/2, all
scenarios, and horizons through 8,000. Renewal used full-energy opponents,
25-tick delay, placement version 1, and current `.5` damage/`.1` regeneration.

| Arm | Renewal | Reward | S at 1k / 8k | Endpoint at 1k / 8k | Kills at 1k / 8k | Interpretation |
|---|---|---|---:|---:|---:|---|
| C0 | no | physical | .6648 / .1502 | .3934 / .0049 | 10.89 / 11.77 | finite prey-energy tail |
| C1 | no | zero | .4189 / .0524 | .0035 / 0 | 9.27 / 9.27 | reward removal is an extinction floor |
| R0 | yes | zero | .3668 / .0459 | 0 / 0 | all focal activity ends by 1k | pressure persists only after focal extinction |
| R1 | yes | physical | .7084 / .4860 | .5167 / .4292 | 36.16 / 231.98 | distributed renewable-energy farm |
| R10 | yes | cap 10 | .3872 / .0484 | 0 / 0 | 16.19 / 16.19 | plant-equivalent cap still floors |

Renewal mechanics themselves passed: corrected R0 maintained 97.64% weighted
opponent uptime, every case exceeded 90%, 14,627 respawns injected 4.388M
energy, no spawn failed, and maximum cumulative same-cell share was 13.3%.
But every focal was extinct by 1,000; corrected focal funnel and consumption
counters were exactly flat thereafter. Renewable background activity without a
living focal is not selection pressure.

R1 is the necessary pathology positive control. At 8,000 it injected 87,334
energy/case, returned 53,397 to the focal through 232 kills, and retained 42.9%
endpoint survival. Late throughput remained 26.8 kills and 6,242 reward per
1,000 ticks. Only 8.2% of renewed kills occurred within 25 ticks, mean kill
latency was 297 ticks, placements were dispersed, and no spawn failed. This is
not a death-cell exploit; it is population-scale harvesting of evaluator-minted
energy. Roughly 61% of injection returns to the focal.

The factorial also caught and repaired an instrumentation defect before final
interpretation. Attack events initially attributed respawned organisms by new
`organism_id % pool_len`, while survival used immutable species/source identity.
Events now carry attacker/victim species IDs directly. Corrected R0 has exactly
zero post-extinction focal actions; corrected R1 changed by only `.009%` of
kills and `.024%` of reward, leaving its farming verdict intact.

### Reward-cap screen

A coarse matched screen used caps 25/50/100/150 on eight seeds, level 0, all
slots/scenarios, and horizons 1k/4k/8k. Advancement required nonzero 4-8k focal
interaction and endpoints while focal reward stayed below 25% of injected
energy.

| Cap | 8k endpoint | 4-8k focal interaction | Reward/injection | Verdict |
|---:|---:|---|---:|---|
| 25 | 0 | exactly 0 | 1.6% cumulative | floor |
| 50 | 0 | exactly 0 | 14.2% while active | transient only; all extinct by 4k |
| 100 | 3.19% overall | 2.91 kills/1k | 16.1% overall, 33.2% baseline | baseline-only localized farming |
| 150 | 28.75% | sustained | 35.7% overall, ~50% baseline | slides back toward subsidy |

Cap 100 lies just above the viability cliff but is not robust: scarcity and
sparse-search have zero 8k survivors, while baseline alone retains 9.6% and
captures a third of injected energy. Cap 150 restores broader persistence by
making subsidy dependence large again. No cap satisfies both durability and
anti-farming gates.

Conclusion: reject evaluator-minted renewable opponents as an evolutionary
substrate. The result is an energy-source trilemma: zero/small reward makes
renewal an immortal hazard that kills all focals; physical/large reward is an
external food fountain; intermediate reward has a sharp scenario-dependent
cliff. The causal positive result is that renewable opponents *can* sustain
late interaction when energy supports the focal, so finite prey exhaustion was
indeed truncating the game. The next mechanism must make renewed opponent energy
endogenous—earned from plants and paid into offspring—so predation recycles
ecological energy rather than evaluator injection. That points to restoring a
minimal energy-conserving in-world reproduction loop (or a strictly equivalent
plant-funded birth ledger) before historical archive selection is revisited.

The 4,000-tick extension strengthens that rejection. Current mechanics reached
11.917 kills/case, `.292507` integrated survival, and `.04375` endpoint
survival; one-hit reached 12.490 kills/case and `.310151` integrated survival
but a slightly *lower* `.03854` endpoint. One-hit buys early survival area, not
durable persistence.

## 2026-07-13 — Round 11: bounded PowerPlay survives evidence repair, not depth 3

The real-Simulation sequential-resource PowerPlay pilot was hardened against
mixed search/admission causal scores, duplicated context aliases, zero-payoff
subnormal escrow, ungated no-op integrity, incomplete generator evidence, and
ambiguous completion traces. Its fixed 10-energy episode escrow closes and two
schema-3 depth-2 executions are byte-identical.

Depth 1 is 16/16 versus old/knockout 0. Depth 2 is 15/16 versus historical
0 and 2 and knockout 2. At depth 3, mutable search finds 16/16 but sealed
admission reaches only 12/16; historical retention is 16/16 and 14/16, so both
retention and causal admission reject. The grammar has only 54 stage candidates
and a depth-4 interpreter cap. This is a sound bounded infrastructure result,
not open-endedness.

## 2026-07-13 — Round 12: causal delayed memory fails every paired ecology gate

The conditional-program v3 evaluator presents four-bit left/right FoodRay cue
sequences, erases body/world residue before every tick, enforces an empty delay,
and requires committed turns under identical response scenes. It binds one-shot
admission panels, exact genomes/configs, full BrainState reset and donor swap,
cue erasure/replay/semantic/random/nuisance controls, grouped mechanism lesions,
fixed escrow, and all sensory/logit/action-sample traces.

Across outer seeds 7, 42, and 123, the candidate scores 16/16, old and exact
knockout score 0/16, all eight complement pairs pass, and all response actions
are unique argmax with minimum logit margin `.6856479`. Nevertheless zero
candidates qualify. The strict four-seed ecology panel finds plant or final
energy regression in every outer run. Seed 7's aggregate plant tie `27 -> 27`
was compensation: one pair falls `8 -> 4`, and two pairs lose final energy.
The causal memory module is real but interferes with ordinary ecology.

Primary artifact:
`artifacts/research/runs/completed/open-ended/conditional-program/final-v3/conditional-program-1783933209955-30868.json`,
SHA-256 `d9f53461c15fa92d491d89ac70c85d4bfb46cfde2c1ee9d7cd39b1ee11cedef1`.

## 2026-07-13 — Round 13: legacy solvers ignore an unfamiliar public preamble

The literal TCPE engine slice was first rejected as internally incompatible:
a meaningful preamble changes recurrent state/turn/action samples, the old T2
trace pays two 5-unit plants rather than one terminal token, and one exact
construction snapshot cannot be several controlled nuisance contexts. Engine
O/F/escrow work was gated behind a cheaper evaluator-owned falsifier.

The executable probe reconstructs actual PowerPlay checkpoints and compares a
fixed 36-tick meaningful task-program cue stream with blank and cue-permuted
arms under identical seeds, turns, FoodId allocation, body resets, task horizon,
and energy. Zero of four evaluable seed x depth pairs passes the predeclared
`meaningful>=14, blank<=2, permuted<=2` zero-shot gate. Full depth-2 success
requires every prefix deadline; a later audit found that the first artifact used
only the last prefix flag, inflating the seed-7 blank count from 8 to 10:

| Source | Depth | Meaningful | Blank | Permuted |
| --- | ---: | ---: | ---: | ---: |
| 7 | 1 | 16 | 7 | 6 |
| 7 | 2 | 13 | 8 | 9 |
| 42 | 1 | 16 | 16 | 16 |
| 123 | 1 | 16 | 16 | 6 |

Seeds 42 and 123 fail to construct depth 2. All 192 matched arms have exact
tick shapes, zero prefix consumption, closed escrow, and maximum core energy
residual zero. The probe never trained on the preamble, so it rejects only
zero-shot reuse of these exact visible-resource checkpoints. It cannot reject a
protected public-aware decoder, and it contains no branch-transfer pipeline.
The corrected schema-v2 artifact is
`artifacts/research/runs/completed/open-ended/public-preamble-probe/final-v2/public-preamble-probe-1783935214211-77625.json`,
SHA-256 `60aa935f570fbf45e13b21bd9aa10cb4c35b4f274a518130c89c082202462bc5`;
an independent schema-v2 replay is byte-identical.

## 2026-07-13 — Round 14: exact formal and empirical boundary

The fixed executable under fixed config/seed and finite RAM/disk is a finite
deterministic transition system. It must halt/fail or revisit a complete state,
after which its observable tail is periodic. This is a proof boundary for the
prompt's literal unbounded reading, not a prediction of an accessible early
plateau. An unbounded-heap idealization can emit infinite syntax and histories,
but fixed task languages can still collapse into universal-interpreter, memory,
time, or size ladders.

One materially new semantic route is persistent, energy-funded, recursively
composable public niche construction: later lineages must causally
use, modify, or depend on artifacts built by earlier lineages, and those
artifacts must become inputs to further construction. It remains unimplemented
and would support only an operational sustained-tail claim through increasing
finite horizons. Required gates include conserved material/energy, public
generic ports, persistence, creator/artifact/consumer knockouts, transfer under
alpha/translation/rebuild, frozen-selection/no-payoff controls, and rejection
of inert artifact growth or a universal constructor. This is a concrete
candidate, not a proof that construction is the unique or smallest route; the
active protected-public-training and construction lanes still need experimental
resolution. Reproduction-only is resolved separately in Round 15.

## 2026-07-13 — Round 15: reproduction-only is historical, not a new driver

The source audit in `research/archive/reports/endogenous-replication-closure.md` closes the active
reproduction-only lane. Before `a5d3c81`, organisms selected a Reproduce action,
paid the complete offspring energy from a live parent at conception, gestated,
and produced a mutated child with exactly that stored starting energy. With the
baseline two-tick gestation, the internal transfer was 300 energy units.

That stronger substrate was already active in the archived 21-experiment
campaign. Multi-seed social transfer approached a uniform fixed point;
pursuit/evasion grew brains while action effectiveness regressed
`.5613 -> .5157`; the dense public display contest peaked and declined by 1M.
The old campaign also injected founders and lacked today's fatal per-tick
ledger, so it is not a clean fully endogenous causal trial. It is sufficient to
show that merely restoring births is not a materially new algorithm. Conserved
reproduction reopens only as part of the active payoff-bearing public
information/artifact route.

## 2026-07-13 — Round 16: protected public training exhausts its declared budget

The zero-shot preamble result did not test a controller trained to use the
public declaration, so a protected residual decoder received the fixed
36-tick program interface while the accepted ecology controller remained
sealed. Search required meaningful and valid-code-swap success at least 14/16,
blank and polarity controls at most 2/16, and ordinary ecology retention at
least 14/16 before touching one sealed admission panel.

Across source seeds 7, 42, and 123, each `64 x 120` search exhausted its
declared budget with zero qualifiers. The best final meaningful/code-swap
counts were `9/9` in all three runs. Seed 7 retained ordinary behavior 16/16
but polarity remained 5/16; seed 42 retained 14/16 with blank/polarity 4/3;
seed 123 retained only 10/16 with blank/polarity 4/2. Sealed admission was never
read and descendant reuse was never attempted.

The exact verdict is limited to this optimizer, source seed, 12-node encoding,
and budget. It does not prove that every public decoder is impossible. It does
reject the current fixed-interface protected decoder as the premise for TCPE
engine work.

Artifacts and SHA-256:

- seed 7: `artifacts/research/runs/completed/open-ended/public-decoder-probe/final-v2/seed-7/`,
  `d1725183391098aedf5e0ca325d49be7b29f2dbaaca9e05efa97b8032b3bd354`;
- seed 42: `artifacts/research/runs/completed/open-ended/public-decoder-probe/final-v2/seed-42/`,
  `3234dfe3e45fb42dbd5a31294bc0a01bf07bd430957d419efaba90df26b6788b`;
- seed 123:
  `artifacts/research/runs/completed/open-ended/public-decoder-probe/final-v2/seed-123/`,
  `ba589804b8fee7db1263293130ed6bb63f2930919a163bb87616e10238568934`.

## 2026-07-13 — Round 17: PPEC mechanics pass, then fail the hostile economics and semantics audit

PPEC Stage 0 adds a persistent public proof-carrying energy cache. Ordinary
plant consumption funds it; public program/challenge responses resolve in a
stable post-commit phase; successful release, loss, and interaction cost enter
the organism/food/artifact ledger. The three-seed, eight-context artifact
passes its 20 mechanism gates: own and foreign evaluator-supplied responses are
48/48, no-payoff accepts but releases zero, code permutation is 9/48,
challenge permutation and artifact knockout are 0/48, and every recorded
ledger closes. An independent run on `main` is byte-identical.

Primary artifact:
`artifacts/research/runs/completed/open-ended/ppec-stage0-final/ppec-mechanism-1783940882587-66432.json`,
SHA-256 `71f7cc440c27e93cebaa90cf829fff0202169e9f176be8814cf0906a4ce744c1`,
result fingerprint
`53b338f61174fdf17d83bc5371e539a72d4c11d8143892f980d9f89be192131b`.

The hostile audit blocks evolutionary Stage 1 on the shipped representation:

- At cache energy 7 and cost `.25`, random responses earn `+72` net over 48
  trials and constants earn `+23/+93/+86/+86`. The "strongly reduced" gates
  are acceptance gates, not negative-payoff gates.
- Cost `2.5` makes pooled random/constants negative and exact responses `+216`,
  but a fixed response still earns `+2` on seeds 42 and 123. More importantly,
  NAND programs can be constant, making "fixed loses" require `cost > reward`
  while "exact wins" requires `cost < reward`.
- A deterministic 10,000-program screen found constant functions in
  `394/10000`, `85/10000`, and `23/10000` programs at arities 2, 3, and 4;
  roughly 29-32% had a majority output above one half.
- Only 18/38 and 18/54 opcode gates in the two shipped programs influence the
  output. Flipping a dead opcode preserves the exhaustive truth table while
  changing the public fingerprint. The arity-3 program ignores its third
  input. Syntactic drift therefore passes the advertised protocol-variation
  gate.
- Organisms have no artifact receptor, response head, Interact, Construct, or
  Deposit action. Every successful answer is evaluator-computed. Caches stack
  without expiry/capacity/maintenance, requests are externally unbounded, and
  reset erases all caches between evolutionary episodes.

The full derivation and reproduction commands are in
`artifacts/research/runs/completed/open-ended/ppec-adversarial-audit.md`. Retain Stage 0 as a
narrow deterministic mechanism checkpoint, but block the NAND-byte proof-cache
family as an algorithm. A bounded canonical reversible-program decoder could
test organism-owned access, but its 24 two-bit functions would still be bounded
novelty. The remaining construction route needs canonical balanced semantics,
autonomous neural use, conserved material/compute/storage/lifecycle, and new
downstream ecological relations rather than longer public programs.

The current-main eight-seed regression at tick 2,500 remains healthy with PPEC
disabled: all five metric axes have 8/8 coverage, mean plant rate `.117952`,
action effectiveness `.455734`, mutual information `.073679`, and zero
predation as configured. Artifact:
`artifacts/evaluation/open-ended-current-main-8seed-t2500/`.

## 2026-07-13 — Round 18: procedural ecology passes mechanics and fails as an open-ended driver

The last incompatible route tested a translation-equivariant local ecology
whose plant-release carrier could remain stationary, move as a front, or react
to whether the prior release was consumed. A fixed 200-energy evaluator escrow
funded 20 releases of 10 energy in each independent counterfactual world.

Across seeds 7, 42, and 123, the evaluator produced 24 cases, 48,000 wrapper
ledger rows, and 480 releases. All 19 narrowly scoped mechanism gates pass;
duplicate artifacts are byte-identical, every recorded residual is exactly
zero, translated traces agree, clamping the consumption input changes the
responsive trace, and a repaired physical fingerprint collapses a stationary
versus clamped-responsive bookkeeping alias.

This is a negative result. The artifact serializes
`stage_1_authorized=false` and `open_endedness_demonstrated=false`. The escrow
is evaluator-endowed rather than endogenously sourced; same-boundary
reclaim/release lacks intermediate physical checkpoints; occupied target cells
abort; consumption is read from private world membership; the disabled hook is
an empty no-op; and canonical/procedural plants lack provenance. The fixed
eater also makes the three seeds replay checks rather than behavioral samples.

Current brains observe every tested ecology only through generic food rays and
act through the fixed six-action channel. A recurrent search-and-track policy
is therefore a serious alternative explanation for the three release
trajectories, but no tracker experiment ran. The finite interface does not by
itself bound temporal strategy complexity. The current implementation is
blocked; the broader procedural-ecology family remains an untested hypothesis
requiring a dual-population regret experiment with tracker, transfer, knockout,
and increasing-horizon controls.

Primary artifact:
`artifacts/research/runs/completed/open-ended/procedural-ecology-stage0-final-v2/procedural-ecology-stage0-1783943744919-31422.json`,
SHA-256 `42faf58ce86eaf43dfeaa96b09c8ea9636a2414485c631ad0c0a6676af1a0d76`,
result fingerprint
`2c515fa5755af20a529f7800a8359ded7063699e368a9e2750b3515693facdfd`.
The full audit is `research/archive/reports/procedural-ecology-adversarial-audit.md`.
