# Log

Append-only iteration history (newest first). One entry per planner iteration:
goals assigned to coordinators, what each handed back, what was merged into
`autoresearch/best`, and what changed in `STATE.md`. The lossless audit trail
that lets `STATE.md` be compacted aggressively.

<!-- new entries go directly below this line -->

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
