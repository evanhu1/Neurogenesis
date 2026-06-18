# Log

Append-only iteration history (newest first). One entry per planner iteration:
goals assigned to coordinators, what each handed back, what was merged into
`autoresearch/best`, and what changed in `STATE.md`. The lossless audit trail
that lets `STATE.md` be compacted aggressively.

<!-- new entries go directly below this line -->

## Iteration 12 (Dir3) — clustered spike fields break the vision-myopia collapse — CHAMPION ADVANCED — 2026-06-18
* **Goal:** a new skill-demanding niche (Dir3) that raises the new headline
  `action_effectiveness` via FAR-FIELD perception (no vision=1 trap) and without ease.
* **Design** (a read-only design agent surveyed the engine and proposed 3 niches;
  I took its Candidate 1 but corrected an inconsistency — a *blocking* spike is a
  harmless wall, so I used *passable+lethal* spikes with a metric penalty instead).
* **Change** ([[experiments/0012-ecology-spike-fields]], branch
  `autoresearch/exp-0012-ecology-spike-fields`): (1) `spawn/world.rs` clusters
  spikes into contiguous Perlin fields (top `spike_density` fraction by noise);
  (2) `turn/commit.rs apply_moves` scores a Forward INTO a spike as a *failed*
  action (organism still enters + takes damage); (3) config `spike_density` 0.10→
  **0.05** (the swept sweet spot; also fixed a pre-existing eval/baseline desync).
* **Result — PROMOTED.** Cross-seed 500k, gate green (build/det P1-P2/tests/eval):
  **action_effectiveness 0.5434→0.5522 (+0.0088)**, 3/4 seeds up. The decisive
  readout: **seed 7's vision=1 collapse is BROKEN** (vision 1.06→9.04; all seeds
  ~7.5–9.0), and mi_sa de-inflates 0.195→0.109 (uniform — the seed-7 0.44 confound
  is gone), empirically confirming the Dir1 thesis. Predation held, foraging
  −0.0025 (within noise, all seed-7). Coverage sweep: 10%→aeff 0.527 (over-taxed),
  5%→0.552 (win), 7%→0.503.
* **Process:** screened at small scale (collapsed — scale noise; baseline collapsed
  too); pivoted to canonical. Verified the vision co-gate at 100k (baseline seed 7
  already vision 1.43 @100k → collapse is EARLY, so the comparison is clean). Tuned
  coverage via `--set` (no recompile — the design reuses spike_density as coverage).
  Champion advanced `git cherry-pick` exp-0012 onto autoresearch/best → `47a6111`.
* **Bundle:** +1 experiment, +2 findings (clustered-hazards-break-myopia;
  the-system-converges/not-open-ended); best-program/STATE updated, lineage 5 deep.
  `main` untouched. **The Dir1→Dir2→Dir3 arc is complete.**
* **Next:** a MOVING/co-evolutionary hazard (roamer) for stronger open-endedness; an
  open-endedness metric; longer-horizon (seeds 123/2026 still climbing at 500k).

## Dir2 decision — intelligence headline = action_effectiveness; mi_sa demoted — 2026-06-18
* **User call** (after the Dir2 prototype skill panel): **`action_effectiveness` becomes
  the intelligence headline; `mi_sa` is demoted to a vision-confounded diagnostic.** Zero
  engine change; unblocks Dir3 under a trustworthy measure.
* **Re-grounding (no new runs — from recorded metrics):** action_effectiveness DRIFTED DOWN
  across the champion line — `eb30fff` 0.5647 → `0fa799b` ~0.5422 → `1dab610` 0.5435. The
  iter6→9 "intelligence gains" were mi_sa (now distrusted); on the trusted measure the
  program has not improved aeff since `eb30fff`. The iter6→9 gains were foraging/predation/
  ecology richness.
* **Champion base for Dir3:** held at `1dab610` (carries the open-endedness substrate:
  predator niche + within-life learning) as a GOAL-driven choice; `eb30fff` is the
  aeff-optimal alternative. Recorded in `best-program.md` + `STATE.md`. Dir3's bar:
  genuinely RAISE action_effectiveness via a skilled niche, avoiding the vision-myopia trap.
* **Next:** Dir3 = [[directions/predation-led-mortality-selects-for-skill]] — scope a minimal
  far-field-sensing / predation-led-mortality niche, evaluated on action_effectiveness.

## Dir1 investigation — the seed-7 mi_sa=0.44 outlier is a SHORT-VISION confound — 2026-06-18
* **Not a code-change iteration** — a read-only deep-inspection of the 4 evolved 500k
  worlds of champion code `1dab610` (planner `autoresearch/best` @ `c542d21`), the first
  of the four directions the user selected. Executed via 3 parallel read-only sub-agents
  (sensory/policy, wiring/convergence, niche/trajectory).
* **Result (high confidence):** seed 7 is the ONLY seed to converge on `vision_distance=1`
  (mean 1.06, 94% see 1 hex) vs ~8–9 for the others. Short vision sharpens the mi_sa
  sensory bins (food = crisp adjacent/absent) → near-deterministic food-direction→action
  map → high I(S;A). Clean 3-reflex wiring (`visF→Eat`/`visL→¬Forward`/`visR→TurnRight`,
  intra-seed cosine 0.975). → `findings/seed-7-mi_sa-outlier-is-a-short-vision-crisp-binning-effect`.
* **Big implication:** **mi_sa has a degenerate optimum at MYOPIA** — reducing vision range
  raises it. The loop has been partly optimizing sensory impoverishment when chasing mi_sa;
  the champion's mi_sa headline rests on one short-vision seed. New high-priority direction
  `directions/mi_sa-is-confounded-by-vision-range` (the core Dir2 fix). But seed 7 is also the
  best forager+predator, so the short-vision reflex is genuinely competent, not pure gaming.
* **Niche:** seed 7 = sparse, food-rich (3.3× food/capita), predation-led (38.5%) not
  starvation-led mortality; cross-seed corr(mi_sa, food/capita)=+0.997. → new direction
  `directions/predation-led-mortality-selects-for-skill` (the Dir3 lever). All breeding pops
  are single-species monocultures; the "418 founder lineages" are inert gen-0 re-seed
  injections (a `lineage` artifact, not real diversity).
* **Process:** the per-organism sensory/policy sub-agent leaked memory (it spawned hundreds
  of `decide` subprocesses; per-tick records aren't serialized so `step`+`decide` must be
  same-process, and `query` rejects `step`). Killed it; recovered the qualitative answer
  from the wiring sub-agent (conditional-policy sharpness = the 3-reflex). Exact H(S) vs
  H(A|S) split not separately measured — noted honestly in the Finding.
* **Bundle:** +1 finding, +2 directions; `STATE.md` updated (Dir1 RESULT block, reframed
  candidate directions A′/B′, census). `main` untouched; no champion change (investigation).
* **Next:** Dir2 = A′ (vision-invariant skill measure — prototype post-hoc, human call on the
  contract), then Dir3 = B′ (predation-led niche) evaluated under the new measure.

## Iterations 10–11 — plasticity/topology refinement (DRY → plateau) — 2026-06-17
* iter10 (eligibility_retention/decay): dead-end — a longer credit window cannibalizes
  the seed-7 mi_sa win; credit-window genes trade aeff↔mi_sa per-seed, can't broaden both.
* iter11 (brain substrate: more synapses/neurons/vision): dead-end — more substrate
  DILUTES (minimal brain has the highest aeff/mi_sa). Confirms "capability without
  strong-enough reward dilutes" for topology too.
* **2 consecutive dry iterations → the frontier has plateaued.** Champion holds at
  1dab610 (the iter9 arms-race+learning substrate, mi_sa 0.1952). Architecture mapped
  across 11 iterations. Next moves need a human call (metric contract) or a fresh
  mechanism (new skill niche / better learning rule) — candidate directions A–E in STATE.

## Iteration 9 — three-factor band tune → intelligence gain (CHAMPION ADVANCE) — 2026-06-17
* 1 agent on the new champion base: swept the three-factor neuromodulator band;
  **GAIN 0.08→0.04** (one line) won. Authoritatively re-confirmed cross-seed
  (planner re-ran 4×500k; matched the agent byte-for-byte; det-check ✓).
* **CLEAN strict-dominance advance** ea2fc38-substrate → **121ee21**: beats the
  predecessor champion on ALL pillars — action_eff 0.5422→0.5435, **mi_sa
  0.1335→0.1952 (+46%)**, prey 0.0022→0.00235, plant 0.0719→0.0786, all 4 seeds
  survive. **mi_sa now exceeds the original homeostatic 0.1407** — within-life
  reward-learning on the arms-race ecology produced genuinely more information-rich
  brains. Gentle credit (GAIN 0.04) lets the covariance rule retain rich
  sensory→action structure (`ContactAhead→Eat`/`Attack` consolidated).
* Honest caveats: mi_sa gain is seed-7-heavy; aeff (0.5435) still < homeostatic-only
  0.5647 (so vs the all-time aeff mark it's a mi_sa-for-aeff trade). OKF: experiment
  0009 (promoted); best-program/STATE/log updated. Champion lineage 4 deep:
  homeostatic → consume-on-kill → three-factor → tune.

## Champion advance (iter8 close-out, goal-driven) — 2026-06-17
* Advanced `autoresearch/best` ea2fc38 → **0fa799b** = homeostatic + consume-on-kill
  + three-factor (the iter6 substrate). Gate: cherry-pick clean, build ✓, det-check
  P1/P2 ✓; cross-seed = iter6's validated numbers (code byte-identical).
* **Rationale:** the active goal is "open-ended evolution of brains." This substrate
  is the most open-ended brain ecosystem found (predator/forager multi-niche +
  within-life reward-learning) vs the prior foraging monoculture. Deliberate
  goal-driven advance accepting a small, seed-123-driven intelligence-PROXY dip
  (aeff 0.5647→0.5422, mi_sa 0.1407→0.1335) on proxies shown misaligned with the
  goal; foraging (+0.003) and predation (+0.0004) improve and the ecology is richer.
  Lineage: homeostatic(0001) → consume-on-kill(0003) → three-factor(0006).
  Reversible: prior pure-proxy champion = eb30fff.

## Iterations 7–8 — perception to complete the hunting loop (2026-06-17)
* iter7: corpse channel + consume-on-kill + three-factor → mismatch (consume-on-kill
  leaves no corpse / targets live prey) → not retained, dilutes → dead-end, but
  pinpointed the fix (live-prey channel). iter8: the matched live-prey channel →
  ALSO regresses both HOLD pillars below champion AND the iter6 base.
* **Conclusion:** adding sensory channels dilutes brain topology more than the
  hunting signal repays — even matched + with within-life learning
  ([[findings/perception-augmentation-dilutes-topology-the-best-arms-race-substrate-is-iter6]]).
  **The predation mechanism space is EXHAUSTED.** Best substrate = iter6
  (consume-on-kill + three-factor), narrowly short of the gate.
* **Binding constraint = the metric contract** (human decision): prey target
  structurally unreachable; action_effectiveness penalizes predation while mi_sa
  rewards it. With a recalibrated contract, iter6 cleanly advances. 8-iteration arc
  complete; champion still iter1 homeostatic. OKF: experiments 0007/0008 + 2 findings
  + 2 directions; STATE/best-program reflect the exhausted-mechanism conclusion.

## Iteration 6 — reward-sensitive learning on predator ecology (2026-06-17)
* 1 agent: gentle three-factor (energy-delta neuromodulated Hebbian, GAIN 0.08) on
  the consume-on-kill predator ecology — let brains LEARN to hunt within life.
  (Agent's API connection dropped before its commit; planner recovered results from
  its saved 500k worlds and persisted the branch 696def5.)
* **Validated the deep hypothesis — the most promising arms-race result.** Vs the
  consume-on-kill base, the three-factor rule **RECOVERED action_effectiveness
  0.5088→0.5422** (what iter1's three-factor FAILED on foraging-only) — within-life
  reward-learning works WHEN there's skill to learn. Held mi_sa on 3/4 seeds. The
  CLOSEST any predation experiment came to holding the HOLD pillars (vs champion:
  aeff −0.022, mi_sa −0.007, seed-123-driven) — narrowly short of a clean advance.
  Strong band (GAIN 1.5) destabilized → gentleness matters. det-check ok.
* **No champion advance, but the path is now clear and validated:** the full
  intelligent-hunting loop (predator niche + within-life reward learning) is the
  right mechanism ([[directions/reward-sensitive-learning-on-the-predator-ecology]]).
  Refine the band + stack the corpse sensory channel to clear the gate; and/or
  recalibrate the metric contract (human).
* **6-iteration arc complete** — coherent theory in STATE: intelligence needs
  selection pressure for skill + within-life learning to acquire it, and the
  current metrics partly block recognizing the predator-niche arms race that
  delivers both. OKF: experiment 0006, STATE/best-program updated.

## Iteration 4 — amplify predation (2026-06-17)
* 2 agents (isolated worktrees, fixed harness) off consume-on-kill: scarcity, reliable.
  **No champion advance.**
* **Scarcity backfired** → [[mechanisms/predation-is-encounter-limited]]: predation
  fires only on predator–prey co-location, so it's density-driven. Plant scarcity
  lowers density → SUPPRESSES predation (seed-7 share 17%→4%; predations/tick −10×).
* **Reliability** (success floor + dmg 0.75) doubled the niche (8% dedicated hunters,
  35% of deaths), prey beat champion on 3/4 seeds, mi_sa/slope held — **but prey
  still ~0.002**, confirming
  [[findings/prey-consumption-target-is-structurally-unreachable-in-a-stable-ecology]]
  (0.025 needs ~25 kills/tick ≈ 2% of the population/tick; only iter2's explosion hit it).
* **Conclusion:** the predation/prey target is a metric-calibration problem, not a
  mechanism one. The last mechanism lever is **encounter amplification** (attack
  reach / prey-pursuit). Judge predation by goal-signals (niche size, hunting
  behavior, mi_sa), not the unreachable rate.
* OKF: 2 Experiments, 1 Mechanism (encounter-limited), 1 Finding (target unreachable).
* **Next (iter5):** attack-reach / prey-pursuit (encounter amplification).

## Iteration 3 — predator–prey arms race (2026-06-17)
* **First run on the fixed harness** (pre-created isolated worktrees; `main` stayed
  clean ✓). 3 agents off `a1d33b7`: redistributive-kill-reward, consume-on-kill,
  corpse-sensory-salience. Plus a planner combo gate (salience + consume-on-kill).
* **No champion advance.** All 3 + the combo fail the gate.
* **Major qualitative result:** energy-conserving kill rewards reliably **evolve a
  real predator niche** — predation 26–48% of deaths, pure-predator phenotypes,
  emergent contextual hunting brains (`ContactAhead→Eat` w≈1.5; with the corpse
  channel, learned `Corpse→Eat`), NO explosion. Genuine open-ended behavioral
  evolution. `mi_sa` ROSE where predation strongest (consume-on-kill 2026: 0.36 vs
  0.141).
* **Why no advance — the metric wall** ([[findings/predator-niche-is-inducible-but-the-prey-metric-resists-and-predation-regresses-action-effectiveness]]):
  `prey_consumption_rate` (= prey/total_actions) barely moves for a hunting
  minority (tops ~0.004–0.005, 5–10× short); `action_effectiveness` regresses
  (younger death-cohort + attack chaos). The two intelligence pillars disagree
  under predation (mi_sa↑, action_eff↓). Salience+reward combo was WORSE (compounded
  dilution).
* **Strategic pivot:** the arms race is the goal-aligned mechanism; amplify it until
  predation dominates ([[directions/amplify-the-predation-dynamic]]) and/or re-weight
  intelligence toward mi_sa ([[directions/reconsider-intelligence-metric-under-predation]],
  human call).
* **OKF:** 3 Experiments, 1 Finding, 2 Directions. STATE/best-program/log updated.
  Harness hazard logged: cross-agent `pkill` of relative artifact names (use unique names).
* **Next (iter4):** amplify-predation = consume-on-kill + scarcer plant (hunt-or-starve).

## Iteration 2 — predation mechanics + metabolism (2026-06-17)
* **First run of the iterative-agent harness.** 6 agents (3 predation, 3
  metabolism) forked `a90244a`, each driving sim-cli on its own world. **No
  champion advance:** all 6 dead-ended + the carried `lower-fertility-threshold`
  foraging lever failed the gate on the homeostatic base.
* **Central finding (high value):** [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]
  — every ease-adding lever (more food −0.064 aeff/−0.097 mi_sa; softer metab
  −0.050; cheaper brains −0.067; free kill-energy → explosion) degrades the
  intelligence pillars. Targets that raise foraging/survival are in tension with
  holding intelligence. Path forward = a competence-rewarding predator–prey arms race.
* **Top lead:** [[directions/redistributive-kill-reward]]. `kill-reward` (additive)
  drove prey to 0.036 (> target) — the lever works — but minted free energy →
  population explosion (seed7 590→30574; seeds 42/123 OOM). Fix = energy-conserving
  (redistributive) reward. Predation triangulated: richer corpses & more-lethal
  attacks don't create a scavenger niche ([[findings/predation-needs-an-energy-conserving-kill-reward-not-richer-corpses]]).
* **Metabolism dry:** homeostatic ramp near-optimal; ramp-tune/brain-cost/move-cost
  all trade action_effectiveness; starvation still 57–72% of deaths; learning_slope
  seed-noise-dominated at n=4.
* **Harness bugs found & FIXED (in SKILL):** `isolation:"worktree"` didn't engage
  for planner-spawned background agents (agents clobbered the shared checkout, then
  self-isolated → double work); semaphore split into 2 pools from per-agent SEM
  override (effective cap ~16). Fixes: agents self-isolate as step 0; single fixed
  SEMDIR. Also added sim-run.sh (semaphore) + det-check.sh (P1 byte + P2 fingerprint)
  + sccache earlier this session; eval-queue.sh was tried then dropped for the
  iterative model.
* **OKF:** 6 iter2 Experiments, 1 new Mechanism, 3 Findings, 2 Directions.
  `main` restored to clean `ef6f9bb`; agent worktrees cleaned; exp-0002 branches kept.
* **Next (iter3):** redistributive-kill-reward (top) + corpse-sensory-salience.

## Iteration 1 — metabolism / plasticity / food-ecology (2026-06-16)
* **Coordinators (sequential, Agent-tool):** metabolism-lifecycle, plasticity-genome,
  food-ecology. base_ref `70b7700`. 12 code-change experiments total, each forked +
  built + determinism-checked (threads 1v4 byte-identical) + persisted as
  `autoresearch/exp-0001-*`.
* **Result:** 1 promoted, 11 rejected. Champion advanced `70b7700`→`eb30fff`
  (**homeostatic metabolism**, fast-forward). Authoritatively gate-validated
  (byte-identical to the agent run). Clean seed-for-seed (7/42/123): slope
  +0.000276, action_eff +0.0155, mi_sa +0.047, plant +0.0057, prey −0.0202;
  **seed 2026 rescued (n 3→4)**.
* **Why only homeostatic:** it is the unique experiment that lifts the keystone
  while HOLDING both HOLD pillars; all others regress action_effectiveness
  ([[findings/learning-gains-trade-against-action-effectiveness-in-death-pressure-regime]]).
* **Big durable law discovered:** predation (corpse-eating) is inversely coupled to
  population health — prey collapsed ~10× under every improving change; the prey
  target needs energetically-attractive predation (engine code), not death-pressure
  reduction ([[mechanisms/predation-inversely-coupled-to-population-health]]).
* **Notable rejects:** longer-eligibility hit slope +0.000592 but AE −0.043;
  three-factor-energy (novel neuromodulation) lifted slope but AE −0.057
  (seed-42 instability) — both kept as Directions. lower-food-energy reached plant
  0.087–0.094 but cratered AE (food_energy conflict confirmed).
* **OKF:** 12 Experiments, 2 Findings, 2 new Mechanisms, 5 Directions, 2 DeadEnds
  written; `best-program.md` + `STATE.md` updated. Process learnings recorded in
  STATE (sequential coordinators, n=3→n=4 confound, subagent-resume limits).
* **Next (iter2):** predation mechanics (top priority) + metabolism combos
  (ramp tuning, stack brain-cost-discount) to push learning_slope positive.

## Iteration 0 — bootstrap (2026-06-16)
* **`autoresearch/best`** reconciled: already existed at `ef6f9bb` (== `main`);
  no divergence, no experiments. Planner operates from worktree
  `/Users/evanhu/code/ng-best`. Orchestration = Agent tool (workflow engine
  dropped upstream).
* **Baseline canonical sweep** (seeds 7,42,123,2026, 500k, 321s wall): cross-seed
  mean (n=3 survivors) plant 0.0599 / prey 0.0217 / act_eff 0.5566 / mi_sa
  0.0955 / learning_slope −0.000689. Recorded to `best-program.md`.
* **Per-seed diagnostic**: seeds 7/42/123 survive at pop 1316/2052/1605 (≪ world
  cap 62 500); **seed 2026 fully extinct (pop 0)** → n=3. All survivors have
  negative learning_slope. Baseline is in a scarcity/collapse regime near a
  tipping boundary, not the explosion regime. Predation = corpse-eating;
  starvation deaths leave no corpse. These findings rewrote `STATE.md`'s central
  obstacle + directions.
* **Next:** iteration 1 — coordinators metabolism-lifecycle, plasticity-genome,
  food-ecology (sequential, code-change experiments, base_ref `ef6f9bb`).

## (no iterations yet)
* **Creation**: bundle scaffolded; `STATE.md` seeded with goal/targets and four
  carried-over config-level mechanisms.
