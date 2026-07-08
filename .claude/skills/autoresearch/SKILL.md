---
name: autoresearch
description: Start or resume autonomous research on the NeuroGenesis simulation. Use when the user runs /autoresearch, or asks to "start autonomous research", "run the research loop", "improve the competence pillars/metrics", or "keep researching". You become the PLANNER — load research/STATE.md, run a loop of iterative research agents that drive sim-cli on their own worlds, advance the autoresearch/best branch behind hard gates, and rewrite STATE.md each iteration against a versioned OKF knowledge bundle (research/).
---

# autoresearch — you are the PLANNER

> **⚠️ NEEDS REWORK after the substrate redesign.** This skill drives the removed
> `sim-cli pillars` / `sim-metrics` "competence axes" and the direct-encoded
> genome. Those commands and metrics no longer exist. Before running this loop,
> its measurement layer must be re-pointed at the current stack — `sim-evaluation`
> Quality-Diversity coverage / QD-score and the lean `sim-cli` reads
> (`state`/`inspect`/`brain`/`genome`/`find`/`lineage`). Treat everything below as
> methodology, not runnable commands.

You run autonomous research to improve the NeuroGenesis simulation's competence
metrics. The model is **planner + a fleet of iterative research agents**:

> **planner** (you, the main session) → **research agents** (worktree-isolated
> subagents — each an *iterative experimentalist* that makes ONE code change and
> then **drives `sim-cli` on its own worlds**: fast-forward, inspect, fork,
> refine, cross-seed confirm) → **database** (the OKF bundle in `research/`).

Agents are **researchers, not build-bots.** An experiment is not a one-shot
measurement — it is an investigation: form a hypothesis, change the code, then
*interrogate the simulation* (incremental `run-to` while watching the metrics,
`inspect`/`brain`/`decide` on organisms, `eco` deaths-by-cause, `cp`-fork to
compare variants, fix the code if it's wrong) until you understand *what the
change does and why*. That mechanistic understanding is the highest-value output
— it feeds the OKF `Finding`/`Mechanism` concepts and the next iteration's
hypotheses. `sim-cli` is built for exactly this (world-as-file, `cp`-fork,
incremental advance, `query` batch reads, per-organism inspection).

You (planner) assign experiments, **gate** winners, advance `autoresearch/best`,
and curate the database. You do not micromanage agents' investigations.

## 0. Load context first (every session)

Read, in order:
1. **`research/STATE.md`** — your distilled working memory. Sufficient to resume.
2. `research/index.md` and `research/CONVENTIONS.md` — database layout, concept
   types, **provenance rules**.
3. `research/best-program.md` — the current champion (git ref + lineage + metrics).
4. `docs/sim-cli.md` — the simulator (`new`/`run-to`/`pillars`/`eco`/`find`/
   `inspect`/`brain`/`decide`/`query`, world-as-file, `cp`-fork, warm-once-fork).
5. `AGENTS.md` — build/test commands and the **determinism invariant** you must
   never break.
6. The apparatus: `.claude/skills/autoresearch/sim-run.sh` (the global-cap wrapper
   for heavy runs) and `det-check.sh` (the determinism check). `docs/research-operating-procedure.md`
   for background methodology.

Dip into `research/experiments/`, `findings/`, `log.md` only for a *specific*
fact — never reread them wholesale.

## Objective & targets

Reach these **cross-seed means** (seeds `7,42,123,2026`, 500k ticks). Raw metrics
(no [0,1] interpretation). **Compare honestly seed-for-seed** (n=3→n=4 rule below):

| axis | target | note |
|---|---|---|
| foraging | `plant_consumption_rate ≥ 0.10` | findability + soft survival, not food_energy cuts |
| predation | `prey_consumption_rate ≥ 0.025` | inversely coupled to pop health → needs engine predation rewards |
| learning | `learning_slope ≥ +0.0005` | the keystone; starvation death-spiral is the wall |
| intelligence | hold `action_effectiveness` & `mi_sa`, don't regress | the decisive HOLD pillars |

## Branch & write model

- You **operate on `autoresearch/best`** via a worktree (e.g. `/Users/evanhu/code/ng-best`),
  so the user's `main` checkout is undisturbed. **All commits land on
  `autoresearch/best`:** accepted experiment code + the `research/` bundle +
  apparatus changes. So `autoresearch/best` = `main` + accepted code + knowledge.
- `main` stays pristine; it changes only via a human PR.
- `autoresearch/exp-*` branches hold individual experiment changes (forked
  detached from `autoresearch/best`). Keep them — durable provenance.

## Hard rules (never violate)

- **`main` is human-PR-only.** NEVER commit/merge/push to `main` autonomously.
- **Worktrees fork from `autoresearch/best`**, never from `main`.
- **Every research agent persists its change as `autoresearch/exp-*` before
  returning** — worktrees are ephemeral; an unpersisted change is lost on resume.
- **Every heavy world-advancing run goes through `sim-run`** (the global
  semaphore) — `run-to`/`step`/`watch`/`bench`. Never bypass it with a fan-out of
  raw `run-to`s; that is what OOM'd iteration 1. (Reads — `pillars`/`state`/`find`
  /… — are ungated; routing them through `sim-run` is harmless, it passes them
  through.)
- **Advance `autoresearch/best` only through the gate:** build ✓ + determinism ✓
  (`det-check.sh` → `ok`) + cross-seed eval shows the **HOLD pillars held** and the
  targeted axis improved, on the **honest seed-for-seed comparison**. Conflicts
  that don't auto-resolve → surfaced, never forced.
- **One surface area per experiment.**
- **Preserve determinism.** A change that breaks it (`det-check.sh` ≠ `ok`) is
  rejected outright.
- **Trust determinism — never re-run someone else's result "to be sure."** A given
  (code, seed, ticks) is byte-reproducible. Agents run their own experiments; the
  planner re-validates only the 1–3 promote candidates, once.

## Concurrency model (how the fleet shares one host)

Agents run **in parallel and at their own pace** — implementation times differ,
so they reach their heavy `run-to` phases staggered. To keep that staggering from
ever aligning into an OOM, **all heavy runs pass through `sim-run.sh`**, a global
counting semaphore (mkdir-based; macOS-native):

- `sim-run` uses ONE fixed shared semaphore dir by default (`/tmp/autoresearch-sem`).
  **Do NOT override `AUTORESEARCH_SEM` per-agent or per-round** — that splits the
  cap into independent pools (the iteration-2 bug: a custom pool + the default pool
  gave an effective cap of ~16, not 8). Keep the dir fixed for everyone (agents +
  the planner's gate); only `AUTORESEARCH_SEM_SLOTS` (default 8) is tuned to the host.
- Total concurrent heavy sims ≤ slots across ALL agents; excess blocks briefly,
  nobody is serialized. An agent's investigation is naturally light (≈1 advancing
  run at a time); the spike is each agent's 4-seed confirm — the semaphore spreads
  those. Reads cost nothing. (`det-check.sh`'s sims are tiny scaled 4k-tick runs and
  run ungated — they must NOT consume heavy slots.)

## The n=3→n=4 composition rule (read this — it bit us in iteration 1)

When a change rescues a previously-extinct seed (e.g. 2026), the cross-seed
**mean** mixes cohorts: the rescued seed enters at lower action_effectiveness and
drags the n=4 mean below the n=3 baseline, masking a real HOLD. **Judge on the
seed-for-seed delta over COMMON survivors** (seeds where both base and the
experiment survive); treat a rescued seed as a pure robustness bonus. **Guard the
n:** with only 2–3 common survivors a ±0.0003 mean is within noise (same order as
the `learning_slope` target) — require ≥2–3 common survivors AND per-seed
consistency before advancing on a small delta.

## The iteration loop

### Bootstrap (only if `autoresearch/best` doesn't exist)
`git branch autoresearch/best main`; add a worktree on it; build sim-cli; run a
baseline cross-seed eval (per-seed `new`+`sim-run run-to 500000`+`pillars`).
Record sha + per-seed metrics in `best-program.md` + `STATE.md`. If it exists,
reuse it (ff onto `main` if it advanced cleanly, else note divergence). Ignore
stale `autoresearch/*` branches (`high`, `mar26*`, …).

### Each iteration
1. **Plan from `STATE.md`.** Pick **1–3 disjoint surface areas**. `base_ref` =
   current `autoresearch/best` sha. Set `AUTORESEARCH_SEM`/`_SLOTS` for the round.
2. **Design experiments** — 3–5 independent, single-surface-area,
   mechanistically-distinct code changes per area (yourself, or via one cheap
   *designer* agent per area that just returns `{id,hypothesis,change}` specs).
   Skip any that already have an `exp-*` branch (free resume).
3. **Spawn research agents** (Agent tool, `isolation: "worktree"`,
   `run_in_background: true`), one per experiment, with the **Research-agent
   recipe** below + the shared `AUTORESEARCH_SEM`/`_SLOTS`. They investigate
   iteratively and return numbers + mechanism + their `exp-*` branch.
4. **Collect reports.** Keep the conclusions + mechanistic understanding.
5. **Gate & advance.** Pick winners (see **Gate** below); re-validate in a gate
   worktree; advance `autoresearch/best`. Combine disjoint winners in sequence,
   re-gate after each. Conflicts → `DeadEnd`, surface, don't force.
6. **Update the OKF** — one `Experiment` per experiment (full provenance: git_ref,
   base_ref, embedded metrics, `# Reproduce`; capture the agent's mechanism in
   `# Learnings`). Promote → `Finding`/`Direction`/`DeadEnd`/`Mechanism`. Update
   `best-program.md` iff the champion advanced.
7. **Rewrite `STATE.md` (compact)** + append `log.md`. The compounding step.
8. **Decide & loop.** Self-pace with `/loop`/`ScheduleWakeup`; don't poll sims
   (the harness re-invokes you when a backgrounded agent finishes).

## Research-agent recipe (an iterative experimentalist)

Spawn with `isolation: "worktree"`, `subagent_type: "general-purpose"`. Prompt:

> You are a RESEARCH AGENT — an iterative experimentalist for NeuroGenesis. Read
> `docs/sim-cli.md` (world-as-file, `cp`-fork, incremental `run-to`, `query`,
> warm-once-fork, per-organism `inspect`/`brain`/`decide`) and `AGENTS.md`
> (determinism invariant) first.
> **Setup — ISOLATE YOURSELF FIRST (do NOT trust that you were given a private
> worktree; background-agent isolation can silently fail, dropping you in the
> shared repo where sibling agents clobber your source edits and the shared
> binary):** run `git rev-parse --show-toplevel`; if it is the shared repo
> (`/Users/evanhu/code/NeuroGenesis`) or any path a sibling could share, do
> `git worktree add --detach /Users/evanhu/code/ng-exp-<iter4>-<coord>-<id> <base_ref>`
> and `cd` into it. Do ALL work in this private worktree (its own `target/` and
> `artifacts/`). Do NOT set `AUTORESEARCH_SEM` — use `sim-run`'s fixed default so
> the global cap stays ONE pool. Use `.claude/skills/autoresearch/sim-run.sh` for
> EVERY world-advancing run (`run-to`/`step`/`watch`/`bench`); call
> `./target/release/sim-cli` directly for reads.
> Experiment **<id>** (surface area **<lever-family>**): <hypothesis> — <change>.
> 1. Confirm you are detached on `<base_ref>` in your private worktree. Validate the
>    fork is the champion: `new --seed 7` + a short `sim-run run-to` should show the
>    champion's seed-7 population — if not, stop and fix before spending compute.
> 2. Implement the change — minimal, strictly this surface area; if you change a
>    config/genome default edit BOTH `sim-evaluation/<file>` and `sim-config/<file>`.
> 3. Build: `RUSTC_WRAPPER=sccache cargo build -p sim-cli --release`. Compile
>    fails → return `{"id","status":"build-failed","git_ref":null}` and STOP.
> 4. **INVESTIGATE ITERATIVELY — this is the experiment, not a one-shot:**
>    - *Sanity:* `new --seed 7 --out artifacts/w.bin`; `sim-run run-to 50000 --in
>      artifacts/w.bin`; read `state`/`eco` — alive? sane? not exploding toward
>      the cap? If the change is broken, FIX THE CODE and rebuild (iterate).
>    - *Fast-forward while watching:* advance the same world (or forks) toward
>      500k in steps, reading `pillars`/`timeseries`/`eco` to see HOW the targeted
>      metric evolves and WHY (e.g. deaths-by-cause shifting, age structure).
>    - *Understand the mechanism:* `find`/`inspect`/`brain`/`decide` on organisms;
>      `lineage`; `cp`-fork the world to A/B a sub-parameter or compare variants;
>      `query` for batch reads off one load. (warm-once-fork — warm to 400k once,
>      `cp`, advance the last 100k — is fine for probing the scored window across
>      your OWN variants.)
>    Build a real mechanistic story of what your change does. Keep every advancing
>    run going through `sim-run` (don't launch many raw parallel `run-to`s).
> 5. **Determinism:** run `<repo>/.claude/skills/autoresearch/det-check.sh` (from
>    the worktree) → must print `ok`. Else return
>    `{"id","status":"determinism-broken","git_ref":<branch after step 7>}`.
> 6. **Cross-seed confirm:** for S in 7,42,123,2026: `new --seed S --out
>    artifacts/c-$S.bin`; `sim-run run-to 500000 --in artifacts/c-$S.bin`;
>    read `pillars` + `state` (population). Record per-seed pillars + which seeds
>    survive (pop 0 / NA = extinct).
> 7. **Persist NOW** (worktree is ephemeral): `git checkout -b
>    autoresearch/exp-<iter4>-<coord>-<id>`; `git add -A` (artifacts/ gitignored →
>    code only); `git -c user.name=autoresearch -c user.email=autoresearch@local
>    commit -m "exp <coord> <id>: <one-line>"`.
> 8. Return ONLY JSON: `{ "id", "git_ref":"autoresearch/exp-…", "status":"ready",
>    "determinism":"ok", "per_seed":{"7":{plant,prey,aeff,misa,slope,pop},"42":…,
>    "123":…,"2026":… or "extinct"}, "n_surviving", "mechanism":"<what your change
>    does & WHY, from your investigation>", "concerns", "recommend":"promote|
>    screen-further|dead-end" }`. Report EXACT numbers — never fabricate; a clean
>    null result is valuable. Do NOT write to `research/`.

## Gate (planner advances `autoresearch/best`)

From the agent reports, a candidate is a **winner** iff `determinism==ok`, its
**targeted** axis improves on the seed-for-seed clean delta over common survivors
(7/42/123 + any rescue bonus), with ≥2–3 common survivors and per-seed
consistency, and the HOLD pillars (action_effectiveness, mi_sa) don't materially
regress. Then **re-validate authoritatively** (the agent's numbers are advisory):

```bash
git worktree add --detach /tmp/gate autoresearch/best   # or a path under artifacts/
cd /tmp/gate && git cherry-pick <exp-branch>            # or merge; ff if linear
RUSTC_WRAPPER=sccache cargo build -p sim-cli --release || REJECT "build"
.claude/skills/autoresearch/det-check.sh                 # must print ok
# cross-seed confirm of the candidate AND base, same environment, for a clean
# seed-for-seed delta — heavy runs via sim-run so the cap holds:
for S in 7 42 123 2026; do sim-cli new --seed $S --out artifacts/g-$S.bin; \
  <repo>/.claude/skills/autoresearch/sim-run.sh run-to 500000 --in artifacts/g-$S.bin; \
  sim-cli pillars --in artifacts/g-$S.bin --text; done
```

Advance `autoresearch/best` only if det `ok` and the clean delta holds (target up,
HOLD pillars held, no worse seed survival). **Combine** disjoint winners in
sequence, **re-gate after each** (combined single-lever wins rarely add linearly).
Conflicts that don't auto-resolve → `DeadEnd`, surface, don't force. Clean up the
gate worktree.

## Provenance discipline

Per `research/CONVENTIONS.md`: every `Experiment` records `git_ref` + `base_ref` +
embedded `metrics` + a `# Reproduce` command + the agent's mechanism in
`# Learnings`; every `Finding`/`Mechanism` links its supporting experiments;
`best-program.md` lineage is the ordered chain of accepted experiments. `log.md`
append-only; `STATE.md` compacted each iteration.

## Budget & stopping

- ≤ 3 surface areas/iteration; ~3–8 research agents at once. Global heavy-sim
  concurrency is bounded by `AUTORESEARCH_SEM_SLOTS` (default 8) — tune to the host
  (8 single-thread sims ≈ safe on 14 cores/36 GB; OOM hit ~16–24). Never spawn a
  fan-out that bypasses `sim-run`.
- Agents may **screen cheaply first** in their own investigation (short `run-to` /
  single seed) to kill obvious losers before the 4-seed confirm — but never trust
  the sign/value of `learning_slope` from a short run; the 500k confirm is the
  number.
- Stop a surface area after **2 dry iterations** (no archive-improving result).
- Stop the loop when targets are met cross-seed, the frontier plateaus for K
  iterations, or budget is spent. Surface a milestone and offer a human PR from
  `autoresearch/best` → `main` (never auto-PR to main).

## End-of-iteration checklist

- [ ] each experiment → an `Experiment` concept (git_ref/base_ref/metrics/Reproduce/mechanism)
- [ ] `best-program.md` updated (sha, lineage, metrics) iff the champion advanced
- [ ] `STATE.md` rewritten + compacted; `log.md` appended
- [ ] `autoresearch/best` advanced only via the gate (det ok + clean-delta basis)
- [ ] zero writes to `main`
- [ ] gate worktree + the round's `AUTORESEARCH_SEM` dir cleaned up
