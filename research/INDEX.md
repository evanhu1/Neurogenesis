# Experiment index

This is the concise decision ledger. Detailed evidence is linked from each
entry; bulk outputs live under `artifacts/research/runs/` and are not tracked.

## Proposed

| ID | Hypothesis | Method | Decision rule | Record |
|---|---|---|---|---|
| 2026-07-14-survival-objective-causal-ablation | Removing relative-opponent advantage should reject privately losing interference under costly predation. | Matched three-seed, 40-generation causal evolution run comparing relative-weighted fitness with absolute survival at a 500-tick horizon. | Prefer absolute survival if it preserves competence and improves late retention across at least two seeds; otherwise treat it only as a possible simplification. | [proposal](proposed/2026-07-14-survival-objective-causal-ablation.md) |
| 2026-07-14-unified-consume-action | Merging `Eat` and `Attack` removes an artificial action-coordination burden and improves early evolutionary progress without collapsing trophic diversity. | Matched 20-generation, three-seed NEAT comparison of the current split actions against one target-dispatched `Consume` action. | Prefer the unified action only if held-out progress or competence improves without worse seed consistency, extinction, or behavioral diversity. | [proposal](proposed/2026-07-14-unified-consume-action.md) |

## Recent competitive-NEAT experiments

| Experiment | Hypothesis | Method | Result | Status / record |
|---|---|---|---|---|
| Cost 5 / transfer 40 over 80 generations | The cost-5 economy should preserve profitable predation while competence continues improving. | Three seeds, population 48, 1,000-tick contemporary triads, 36 cases per genome, 80 generations. | Plant-foraging competence improved through the tail, but the horizon became censored and profitable predation vanished from every top fitness quartile by generation 70. | completed; rejected as long-run baseline — [results](archive/experiments/2026-07-14-cost5-transfer40-80g.md) |
| Attack-cost calibration at 1,000 ticks | Lowering attempt cost from 10 to 5 should make selective predation privately profitable without restoring attack spam. | Matched cost-10/cost-5 triadic arms, three seeds, population 48, 20 generations, 36 cases per genome. | Cost 5 put profitable predators in the late top fitness quartile in seeds 7 and 27, including one historical champion. Spam stayed suppressed and the horizon was uncensored, but cost-10 champions remained stronger. | completed; cost 5 advances to matched 40-generation test — [results](archive/experiments/2026-07-14-attack-cost-calibration-1000t-20g.md) |
| Costly-predation baseline | A 10-energy attack cost with 40-energy transfer should remove attack spam while retaining selective predation, and 500 ticks should retain evaluation resolution. | Population 48, contemporary triads, 36 cases per genome, three world seeds, three evolutionary seeds, 40 generations. | Survival and plant-foraging competence improved in all seeds, but no generation had profitable net predation and 22–56% of champion founders hit the horizon alive. | completed; baseline rejected — [results](archive/experiments/2026-07-14-costly-predation-baseline-40g.md) |
| 50x50 competitive baseline | Contemporary competition should produce directional improvement over 50 generations. | Population 24, 100 founders, 5,000 ticks; checkpoint crossplay on seeds 7, 17, 27. | Large adaptation occurred, but one seed cycled and later populations forgot earlier strategies; progress was not reliably cumulative. | completed — [audit](archive/experiments/pairwise-50x50-100f-50g-audit.md) |
| Historical opponent pressure | Replacing some contemporary opponents with archived champions should improve retention. | Matched 8/0, 6/2, and 4/4 contemporary/historical opponent panels at constant 32 cases per genome. | Both archive treatments underperformed the contemporary-only control on chronological and final retention. | completed; rejected — [results](archive/experiments/historical-opponent-pressure.md) |
| Two versus three lineages | A three-lineage mini-ecosystem should produce more robust strategies than pairwise worlds. | Equal simulator-world budget, population 24, 25 generations, three evolutionary seeds. | Triads improved some final retention but were seed-dependent and still converged on plants. | completed — [results](archive/experiments/two-vs-three-lineages.md) |
| Triadic budget controls | Earlier triad results may reflect cases and exposure count rather than ecosystem cardinality. | Four matched pair/triad arms for 40 generations with exposure-, case-, and compute-matched budgets. | Sixteen triadic cases were inadequate; 32 cases were competitive and economical, but no arm eliminated reversals. | completed — [results](archive/experiments/triadic-budget-controls-40g.md) |
| Pairwise versus case-matched triads | The 40-generation triad advantage should persist through generations 40–79. | Pairwise and 32-case triad arms, 80 generations, common held-out checkpoint panel. | The triad advantage did not replicate; pairwise had slightly stronger aggregate retention. Both became plant specialists. | completed; hypothesis rejected — [results](archive/experiments/pair-vs-case-triad-80g.md) |
| Population 24 versus 48 | A larger NEAT pool should delay the late plateau by searching more structures. | Pairwise 80-generation comparison with equal 32 cases per genome. | Population 48 improved earlier on some seeds but regressed in every late tail, cost twice as much, and did not extend progress. | completed; naive scaling rejected — [results](archive/experiments/population-24-vs-48-pairwise-80g.md) |
| Reduced plant density pre-run | Moving from an assumed 40% baseline to 20% plants should increase predation. | Planned population-48 triadic comparison. | Stopped when the supposed 40% baseline was recognized as invalid and ecologically oversaturated; partial data are not evidence. | aborted — [record](archive/experiments/triad-pop48-12opp-food-density-80g-aborted.md) |
| Ten-percent plants | Scarcer plants should prevent plant-only convergence and create meaningful predation. | Population 48, triads, 12 opponent exposures, 80 generations, 10% plant tiles. | Predation became substantial and trophic roles diversified, but held-out progress after generation 40 remained nearly flat and non-monotonic. | completed — [results](archive/experiments/triad-pop48-12opp-food10-80g.md) |
| Absolute-survival reranking | Relative competitive fitness may be causing aggression and stagnation. | Retrospective reranking of every persisted population plus common held-out crossplay. | Absolute survival chose almost the same genomes and produced no more progressive checkpoint sequence. | completed; explanation rejected — [audit](archive/experiments/absolute-survival-objective-audit.md) |

## Earlier open-endedness mechanism search

The complete numerical record for Rounds 1–18 is preserved in the
[historical log](archive/experiments/open-ended-search-log.md). The table below
is the decision-level summary.

| Round | Hypothesis / method | Result | Status |
|---|---|---|---|
| 1A — structural mutation | Compare topology mutation on/off under isolated evaluation. | Added structure was sometimes causally useful, but competence effects split by seed. | completed; mechanism retained |
| 1B — novelty-local competition | Candidate-scoped novelty should sustain behavioral exploration. | Competence and late novelty regressed while archive size grew mechanically. | blocked in tested form |
| 1C — mixed evaluation | Contemporary opponents should improve transferable competence. | Mean transfer worsened and outcomes were strongly seed-dependent. | completed; rejected as evidence of arms race |
| 2 — durability replay | Extend frozen champions beyond their training horizon. | Exposed both zero-consumption runway policies and genuinely durable foragers. | completed; evaluator confound found |
| 3 — relative survival gate | Multiply survival by relative advantage against opponents. | Ranked a hand-picked idler correctly but collapsed in all-passive panels. | completed; insufficient alone |
| 4 — multi-horizon evolution | Multiple horizons plus relative survival should remove cutoff exploitation. | Effects split by seed; evolution found new scalar exploits and more passive cases. | completed; rejected as default |
| 5 — late weighting and anchors | Weight late survival and add durable frozen opponents. | Anchors crowded out focal lineages and favored metabolic passivity. | blocked at preregistered gate |
| 6 — direct predation | Opponent sensing and attack should create reciprocal strategic pressure. | The interface mostly produced cannibalism, passivity, or no attacks; no arms race. | blocked in old interface |
| 7 — cross-pool predation oracle | Prevent same-pool attack to test whether target identity was the bottleneck. | Rescued foraging more than predation and depended on evaluator-provided identity. | diagnostic only |
| 8 — attack funnel | Mechanically strengthen attacks to localize the predation bottleneck. | Easier killing improved early survival but not durable persistence. | completed; upstream bottleneck |
| 9 — checkpoint crossplay | Historical matrices should reveal a Red Queen if one exists. | Found finite ladders and cycles, not expanding historical coverage. | completed; negative |
| 10 — renewable opponents | Renew opponents to prevent finite prey exhaustion. | Low reward caused extinction; high reward became an external energy farm; intermediate rewards were brittle. | blocked |
| 11 — bounded PowerPlay | A task-solver ladder should preserve and extend solutions. | Passed depths 1–2 but failed sealed admission and retention at depth 3; grammar was finite. | blocked in tested encoding |
| 12 — delayed conditional memory | Causal memory should transfer without harming ecology. | Memory behavior was real, but every candidate failed paired ecological retention. | blocked |
| 13 — public preamble | Existing solvers should interpret a novel public task declaration. | No source/depth pair passed the zero-shot semantic gate. | blocked for zero-shot route |
| 14 — finiteness audit | Establish what literal unboundedness can mean on fixed hardware. | Proved the fixed executable is finite-state; operational tail growth remains testable. | completed — [audit](archive/reports/open-endedness-finiteness-audit.md) |
| 15 — reproduction-only | Conserved reproduction alone may provide the missing ecological driver. | Historical substrate already had a stronger reproduction loop and still plateaued. | blocked alone — [closure](archive/reports/endogenous-replication-closure.md) |
| 16 — protected public decoder | Train a protected residual decoder for public task programs. | All three searches exhausted budget with zero qualifiers. | blocked in tested representation |
| 17 — public energy cache | A persistent proof-carrying cache may create cumulative public niches. | Mechanics and ledger passed, but payoffs were hackable and organisms did not own the interface. | blocked — [route audit](archive/reports/tcpe-feasibility-audit.md) |
| 18 — procedural ecology | Consumption-responsive plant release rules may create solver-dependent niches. | Deterministic mechanics passed; funding and semantics remained evaluator-owned and no adaptive comparison ran. | blocked implementation — [audit](archive/reports/procedural-ecology-adversarial-audit.md) |

## Archived designs and reports

- [Quality-diversity / minimal-criteria route](archive/experiments/qd-minimal.md):
  fixed-panel trace MAP-Elites was specified, but the tested novelty archive
  grew while stable novelty and competence regressed; blocked as a complete
  driver on a static ecology.
- [Plasticity and encoding audit](archive/experiments/plasticity-encoding.md):
  found causal runtime effects and defects in the historical learning rule, but
  no evolvable or open-ended learning result. This describes a superseded
  sensor/energy substrate.
- [Open-endedness derivation](archive/reports/open-endedness-derivation.md)
- [Approach registry](archive/reports/open-ended-approach-registry.md)
- [PPEC hostile audit](archive/reports/ppec-adversarial-audit.md)
- [Conditional-foraging adversarial audit](archive/reports/conditional-foraging-adversarial-audit.md)
- [Solver-dependent public ecology / TCPE route](archive/reports/solver-dependent-ecology-route.md)
- [Original redesign](archive/redesign.md)
- [Historical visual atlas](archive/atlas.html)

These documents are retained as decision history. Their references to removed
engine mechanisms describe the historical substrate, not current defaults.
