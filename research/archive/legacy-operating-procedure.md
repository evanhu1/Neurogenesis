# Autonomous research operating procedure (NeuroGenesis)

> Superseded historical procedure. Its paths and substrate parameters are
> intentionally preserved as they were. Use `research/README.md` for the
> current workflow.

Distilled from AsterLab's "Scaling autonomous research to thousands of agents"
(planner→worker→subagent hierarchy, Cluster-Elites archive, surface-area
decomposition, warm/frozen baselines, handoff loops) and mapped onto **this**
project's stack: `sim-cli`, Claude Code subagents (the `Agent` tool), and
`/loop` goal-looping.

## 0. The one idea to internalize

Two **different** kinds of parallelism, and we must not confuse them:

- **Compute parallelism (huge):** many simulation worlds running at once. This
  lives in **`sim-cli sweep`** (parallel grid×seed) and `cp`-forking — hundreds
  of runs, ~free to scale, no LLM cost.
- **Cognitive parallelism (modest):** LLM agents proposing/analyzing. This is a
  **small fleet of subagents** (~5–12), one per *surface area*. Each commands a
  *large* sweep.

AsterLab's "1000 concurrent calls" is mostly their training jobs. Our analog of
"1000 agents" is **1000 sim runs driven by ~10 worker agents.** Do **not** spawn
hundreds of LLM subagents; spawn a handful and let each fan out over compute.

## 1. Three-tier architecture → our tools

| AsterLab role | Our implementation | Owns |
|---|---|---|
| **Planner** | the **main Claude Code session** (you) | the research goal, the candidate **archive**, synthesis, the outer `/loop` |
| **Worker** | a **subagent** (`Agent` tool), one per *surface area* | one lever-family; proposes hypotheses, runs sweeps, returns a handoff |
| **Subagent / experiment** | a **`sim-cli sweep` job** (one grid cell × seed) | a single world run → raw pillar metrics |

The round (fan-out workers → aggregate handoffs) is plain **`Agent`-tool**
orchestration: launch the worker subagents in parallel (multiple Agent calls in
one message and/or `run_in_background`), each worktree-isolated via
`isolation: "worktree"`; resume is free because each experiment persists an
`autoresearch/exp-*` branch + OKF concept. (No workflow engine — for our small,
durable-artifact fan-out it adds complexity without payoff.)

## 2. Surface-area decomposition (the anti-confusion rule)

AsterLab's hardest-won lesson: varying hyperparameters + architecture + data at
once "collapsed because it could not compare experiments across different sources
of change." **One worker = one lever-family.** For us the lever-families are:

- **food ecology** — `food_energy`, `food_regrowth_interval`,
  `food_tile_fraction`
- **metabolism / lifecycle** — `passive_metabolism_cost_per_unit`,
  `move_action_energy_cost`, `action_temperature`
- **corpse / predation mechanics** — (engine levers; code-level, slower loop)
- **plasticity genome** — `hebb_eta_gain`, `eligibility_retention`,
  `synapse_prune_threshold`, `juvenile_eta_scale` (via seed genome)
- **brain topology** — neuron/synapse counts, vision distance (seed genome)
- **world scale / density** — `world_width`, `num_organisms`

A worker varies **only its family** (`sweep --grid` over those keys), holding
everything else at the current baseline. Cross-family interactions are the
**planner's** job to test, deliberately, after single-family effects are known.

## 3. The control loop (`/goal` loop)

Run the planner as a self-paced loop (`/loop` with no interval → you decide
cadence via `ScheduleWakeup`; persist state to disk so it survives a session).
Each **round**:

1. **Plan.** State the round goal and the current **baseline config** (the best
   known point). Pick which surface areas to probe this round (default: all
   under-explored ones).
2. **Fan out workers.** One subagent per surface area. Give each: the goal, the
   baseline, its lever-family + allowed ranges, and the **handoff contract** (§6).
   Each worker runs a `sweep` over its family and returns its handoff.
3. **Collect handoffs.** Read every worker's structured result (best cell, delta
   vs baseline, learnings, concerns). Do **not** dump their sweep files into
   context — keep the conclusions.
4. **Synthesize.** Update the **archive** (§4). Form the next baseline by
   grafting the best **independent** single-lever wins, then **test the combo**
   explicitly (a small planner-run cross-family sweep) — combined single-lever
   wins rarely add linearly.
5. **Gate** the new baseline through §5 before it becomes "current best."
6. **Persist & decide.** Append the round to `artifacts/research/log.md`, write
   the archive, and either schedule the next round or stop (§7).

## 4. The candidate archive (Cluster-Elites analog)

Single-best tracking gets trapped in local minima. Keep a **diverse archive**,
not one champion. Persist it (survives sessions, like AsterLab's population
memory):

- File: `artifacts/research/archive.jsonl` — one line per kept config:
  `{config_overrides, seed_set, metrics:{plant_rate, prey_rate, action_eff,
  mi_sa, learning_slope}, round, provenance}`.
- **Keep for diversity, not just score.** Our metric space is the 5 raw pillars
  (now that the [0,1] interpretive layer is gone — raw values are harder to
  game/misread). Bin loosely by the axis a config is *best* at; keep the top few
  per bin so foraging-specialists, predation-specialists, learning-specialists
  all survive. This is the cheap stand-in for Cluster-Elites: re-bin over the
  whole history each round so no niche goes empty.
- **Branch from several archive members**, not only the global best — especially
  into under-filled bins (assign a worker to "improve learning_slope from the
  best learning-leaning config", etc.).

## 5. Verification gates (the anti-overfit ladder)

AsterLab's failure: optimizers "were not able to validate overfit… at larger
scales equivalent to AdamW." Our analog of overfitting is **single-seed /
scaled-world luck.** Promote a result only as far up this ladder as it survives:

1. **Screen (cheap, directional only):** `--scale` small world or short `--to`.
   *Never trust the value or even the sign of `learning_slope` from scaled runs.*
   Use only to rank candidates and kill obvious losers.
2. **Confirm (canonical):** seed 7, full 500k, via **warm-once-fork** (§ below).
   This is the real per-config number.
3. **Validate (cross-seed):** the eval seed set `7,42,123,2026`. A win must hold
   as **mean ± spread**, not on one seed. `sweep --seeds 7,42,123,2026` does this.
4. **Adversarially review:** a final subagent whose job is to *refute* the
   finding — "is this a measurement artifact (survivor draining, partial window,
   stale sidecar)? does it reproduce? is the delta within seed noise?" Only
   findings that survive refutation update the baseline.

**Warm-once-fork** (the "frozen baseline" trick — biggest compute saver): pillars
read only the last 10% window, so warm to 400k **once**, then fork per
experiment and pay only the final 100k:
```bash
sim-cli new --seed 7 --set <baseline> --out artifacts/research/warm.bin
sim-cli run-to 400000 --in artifacts/research/warm.bin
for cfg in …; do cp warm.bin armX.bin; cp warm.metrics armX.metrics
  sim-cli run-to 500000 --in armX.bin & ; done; wait
```
Caveat: a `--set` that only takes effect at fork time measures a *transient*;
confirm keepers from a cold canonical run.

## 6. Worker handoff contract (structured, lossy-on-purpose)

Every worker subagent returns **only** this (not its raw sweep dumps):
```
{
  surface_area, baseline_used,
  swept: {keys, ranges, seeds, to},
  best_cell: {overrides, metrics_mean, metrics_spread, delta_vs_baseline},
  ranked_table: [top 3-5 cells],
  learnings: "what moved which metric and the likely mechanism",
  concerns: "confounds, instability, eval-time cost, suspected artifacts",
  recommend: "promote | screen-further | dead-end"
}
```
The planner keeps the handoff, discards the sweep files (they live in
`artifacts/runs/` if needed). This is AsterLab's "what was done, learnings, and
concerns" handoff.

## 7. Budget & stopping criteria

- **Per round:** cap concurrent **worker subagents** (~5–12) and cap each sweep's
  `--jobs` so total sim processes ≈ cores. Compute scales in `sweep`, not in
  agent count.
- **Targets** (raw metrics, post-interpretive-layer): set explicit thresholds,
  e.g. `plant_consumption_rate ≥ 0.10`, `prey_consumption_rate ≥ 0.025`,
  `learning_slope ≥ +0.0005`, each as **cross-seed mean**.
- **Stop a thread** when a surface area yields no archive-improving cell for **K=2
  consecutive rounds** (loop-until-dry), or eval-time per round blows the budget
  (low-metabolism population explosions are a known dead end — watch run time).
- **Stop the loop** when all targets are met cross-seed, or the global archive
  hasn't improved for K rounds, or the token/wall-clock budget is spent. Use
  `ScheduleWakeup` long fallbacks; don't poll sims (the harness re-invokes you
  when a backgrounded run finishes).

## 8. Failure-mode mitigations (AsterLab → us)

| Failure | AsterLab fix | Our mechanism |
|---|---|---|
| Local minima | branch from many candidates | diverse `archive.jsonl`, fork several members |
| Attribution collapse | one worker per surface area | one lever-family per worker; `sweep` isolates |
| Empty archive niches | re-cluster all history | re-bin archive each round; assign workers to weak axes |
| Overfit / no generalization | scaling-law validation | cross-seed gate + canonical confirm + adversarial refute |
| Hallucinated findings | leaderboard verification | raw metrics (no gameable score) + refutation subagent + `validate_state` on every world |
| Wasted compute | frozen model | warm-once-fork; screen-then-confirm ladder; loop-until-dry |

## 9. Durable memory

- **Archive + round log** in `artifacts/research/` (per-run, like population
  memory).
- **Cross-session learnings** in the project auto-memory (`MEMORY.md` +
  `memory/`): keepers, dead ends, and *why* (e.g. "low metabolism wrecks
  eval-time", "food_energy is the dominant foraging lever and conflicts with
  learning"). Write these as you would any durable project fact.

## 10. Minimal worked round (commands)

```bash
# Planner sets baseline; warm a canonical checkpoint once.
sim-cli new --seed 7 --set food_energy=12 --set food_regrowth_interval=60 \
  --out artifacts/research/warm.bin
sim-cli run-to 400000 --in artifacts/research/warm.bin

# Worker "food ecology" (a subagent) screens its family, then confirms cross-seed:
sim-cli sweep --grid food_energy=8,12,16 food_regrowth_interval=40,60 \
  --seeds 7,42,123,2026 --to 500000 --baseline food_energy=12,food_regrowth_interval=60 \
  --out-dir artifacts/research/runs
# → returns handoff (best cell, deltas vs baseline, learnings, concerns)

# Planner: graft best independent wins → test the combo → adversarial-review →
# update archive.jsonl + log.md → next round.
```

---

**TL;DR:** *You* are the planner. Spawn ~one subagent per **lever-family**, each
commanding a **`sweep`** (that's where the thousands of runs live). Keep a
**diverse archive**, promote only through **screen→confirm→cross-seed→refute**,
warm-once-fork to keep it cheap, persist everything under `artifacts/research/`,
and run the whole thing as a self-paced `/loop` that stops on target or dry-up.
