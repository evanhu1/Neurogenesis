---
name: autoresearch
description: Start or resume autonomous research on the NeuroGenesis simulation. Use when the user runs /autoresearch, or asks to "start autonomous research", "run the research loop", "improve the competence pillars/metrics", or "keep researching". You become the PLANNER — load research/STATE.md, run the planner→research-agent→eval-queue loop against a versioned OKF knowledge bundle (research/), advance the autoresearch/best branch behind hard gates, and rewrite STATE.md each iteration.
---

# autoresearch — you are the PLANNER

You run autonomous research to improve the NeuroGenesis simulation's competence
metrics. The architecture is **2-tier** (revised after iteration 1, which spent
~6 h at ~8–10× overhead because every agent ran its own uncoordinated sims):

> **planner** (you, the main session) → **research agents** (worktree-isolated
> subagents that make ONE code change each and **build + determinism-check +
> persist a branch — NO sims**) → **eval-queue** (`eval-queue.sh`, a single
> planner-owned evaluator with a HARD concurrency cap that screens→confirms ALL
> branches' sims through one pipe) → **database** (the OKF bundle in `research/`).

**The split that matters:** *cognitive* work (designing experiments, interpreting
results) is distributed and cheap; *compute* work (running `sim-cli`) is a shared,
capacity-limited resource and is scheduled **centrally** by the eval-queue. Never
let N agents each fire their own cross-seed sweeps — that was the iteration-1
oversubscription/OOM disaster.

The compounding thread of knowledge lives in `research/STATE.md` (your working
memory) backed by the append-only OKF database. You assign experiments, run the
eval-queue, gate winners, and curate the database.

## 0. Load context first (every session)

Read, in order:
1. **`research/STATE.md`** — your distilled working memory. Sufficient to resume.
2. `research/index.md` and `research/CONVENTIONS.md` — database layout, concept
   types, **provenance rules**.
3. `research/best-program.md` — the current champion (git ref + lineage + metrics).
4. `docs/sim-cli.md` — the simulator CLI (`new`/`run-to`/`pillars`, raw metrics,
   world-as-file, `cp`-fork).
5. `AGENTS.md` — build/test commands and the **determinism invariant** (fixed
   config+seed ⇒ identical results) you must never break.
6. `.claude/skills/autoresearch/eval-queue.sh` — the evaluator you run (skim its
   header). `docs/research-operating-procedure.md` — background methodology
   (surface-area decomposition, the verification ladder).

Dip into `research/experiments/`, `findings/`, `log.md` only for a *specific*
fact — never reread them wholesale.

## Objective & targets

Reach these **cross-seed means** (seeds `7,42,123,2026`, 500k ticks). Raw metrics
(no [0,1] interpretation). **Compare honestly seed-for-seed** (see the n=3→n=4
rule below):

| axis | target | note |
|---|---|---|
| foraging | `plant_consumption_rate ≥ 0.10` | findability + soft survival, not food_energy cuts |
| predation | `prey_consumption_rate ≥ 0.025` | inversely coupled to pop health → needs engine predation rewards |
| learning | `learning_slope ≥ +0.0005` | the keystone; starvation death-spiral is the wall |
| intelligence | hold `action_effectiveness` & `mi_sa`, don't regress | the decisive HOLD pillars |

## Branch & write model

- You **operate on `autoresearch/best`** via a worktree (e.g. `/Users/evanhu/code/ng-best`),
  so the user's `main` checkout is undisturbed. **All commits land on
  `autoresearch/best`:** accepted experiment code *and* the `research/` bundle
  *and* apparatus changes (this skill, `eval-queue.sh`). So `autoresearch/best` =
  `main` + accepted code + accumulated knowledge + improved apparatus.
- `main` stays pristine; it changes only via a human PR.
- `autoresearch/exp-*` branches hold individual experiment code changes (forked
  detached from `autoresearch/best`). Keep them — they are the durable provenance.

## Hard rules (never violate)

- **`main` is human-PR-only.** NEVER commit/merge/push to `main` autonomously.
- **Worktrees fork from `autoresearch/best`**, never from `main`.
- **Every research agent persists its change as `autoresearch/exp-*` before
  returning** — worktrees are ephemeral; an unpersisted change is lost on resume.
- **Research agents do NOT run cross-seed sims.** They build + determinism-check +
  persist. ALL evaluation goes through the **single planner-owned eval-queue** so
  global sim concurrency is capped. (This is the #1 fix from iteration 1.)
- **Advance `autoresearch/best` only through the gate:** build ✓ + determinism ✓ +
  cross-seed eval shows the **HOLD pillars held** and the targeted axis improved,
  on the **honest seed-for-seed comparison**. Conflicts that don't auto-resolve →
  surfaced, never forced.
- **One surface area per experiment** — never vary multiple lever-families in one
  change (no attribution otherwise).
- **Preserve determinism.** A change that breaks it is rejected outright.
- **Trust determinism — never re-run an experiment "to be sure."** A given
  (code, seed, ticks) is byte-reproducible; the eval-queue runs each branch once.

## The n=3→n=4 composition rule (read this — it bit us in iteration 1)

When a change rescues a previously-extinct seed (e.g. seed 2026), the cross-seed
**mean** mixes cohorts: the rescued seed enters at lower action_effectiveness and
drags the n=4 mean below the n=3 baseline, masking a real HOLD. **Always judge on
the seed-for-seed delta over COMMON survivors** (seeds where both base and the
experiment survive), and treat a rescued seed as a pure robustness bonus.
`eval-queue.sh` computes exactly this (`clean_delta_vs_base_common_survivors`) —
gate on it, not on the raw n=4 mean. **But guard the n:** with only 2–3 common
survivors a clean-delta mean of ±0.0003 is within noise and the same order as the
`learning_slope` target — the queue flags `sufficient_n` (≥ `--min-common`) and
emits `per_seed_delta_vs_base`; require both a sufficient n and per-seed
consistency before advancing on a small delta.

## The iteration loop

### Bootstrap (only if `autoresearch/best` doesn't exist)
`git branch autoresearch/best main`; add a worktree on it; build sim-cli;
run a baseline cross-seed eval (`eval-queue.sh --base-ref <best-sha>
--branches "" --no-screen` evaluates just the base, or run per-seed
`new`+`run-to 500000`+`pillars`). Record sha + per-seed metrics in
`best-program.md` + `STATE.md`. (If `autoresearch/best` exists, reuse it;
fast-forward onto `main` if it advanced cleanly, else note divergence in STATE.)
Ignore stale `autoresearch/*` branches (`high`, `mar26*`, …) — not part of this loop.

### Each iteration
1. **Plan from `STATE.md`.** Pick **1–3 disjoint surface areas** from the frontier
   + open `directions/`. `base_ref` = current `autoresearch/best` sha.
2. **Design experiments.** For each surface area, design **3–5 independent,
   single-surface-area, mechanistically-distinct code changes** (structural edits
   or principled config/genome-default changes). Either design them yourself, or
   spawn one cheap **designer** agent per surface area (Agent tool) that ONLY
   returns a JSON list of `{id, hypothesis, change}` (it spawns nothing, runs no
   sims, returns fast). Skip any experiment that already has an `exp-*` branch
   (free resume).
3. **Spawn research agents** (Agent tool, `isolation: "worktree"`,
   `run_in_background: true`), one per experiment — give each the **Research-agent
   recipe** below. They build + determinism-check + persist `autoresearch/exp-*`
   and return fast (~2–5 min; no sims). Up to ~8 at once is fine (they're cheap).
4. **Run the eval-queue** over all persisted branches (one call, capped) — see
   **Evaluation** below. It screens→confirms and emits per-branch
   `clean_delta_vs_base_common_survivors` + per-seed + rescue/lost seeds.
5. **Gate & advance `autoresearch/best`.** From the queue's summary, a branch is a
   **winner** iff: built ✓, determinism ok ✓, its **targeted** axis improves on
   the clean delta, and the **HOLD pillars (action_effectiveness, mi_sa) do not
   materially regress** on the clean delta (rescuing a dead seed is a bonus, never
   a regression). Advance: fast-forward if linear, else cherry-pick onto a gate
   worktree. **Combine winners** from disjoint surface areas in sequence and
   **re-run the eval-queue on the combined branch** (re-gate) — combined
   single-lever wins rarely add linearly. Conflicts that don't auto-resolve →
   record a `DeadEnd`, surface to the user, don't force.
6. **Update the OKF database.** One `Experiment` concept per experiment with full
   provenance (`git_ref`, `base_ref`, embedded `metrics`, `# Reproduce`). Promote
   → `Finding` / `Direction` / `DeadEnd` / `Mechanism`. Update `best-program.md`
   (sha, lineage, metrics) iff the champion advanced.
7. **Rewrite `STATE.md` (compact)** + append one entry to `log.md`. The
   cross-session compounding step — do it every iteration.
8. **Decide & loop.** Targets met? Frontier improving? Budget left? Self-pace with
   `/loop` / `ScheduleWakeup`; don't poll sims (the harness re-invokes you when a
   backgrounded run finishes).

## Research-agent recipe (build + persist ONLY — no sims)

Spawn with `isolation: "worktree"`, `subagent_type: "general-purpose"`. Prompt:

> You are a RESEARCH AGENT in an ISOLATED git worktree for NeuroGenesis. Do ALL
> work here. Read `docs/sim-cli.md` + `AGENTS.md` (determinism invariant) first.
> Experiment **<id>** (surface area **<lever-family>**): <hypothesis> — <change>.
> 1. `git checkout --detach <base_ref>` (fork the champion).
> 2. Implement the change. Stay strictly within this surface area; keep the diff
>    minimal. If you change a config/genome default, edit BOTH
>    `sim-evaluation/<file>` and `sim-config/<file>` (keep them in sync).
> 3. Build: `RUSTC_WRAPPER=sccache cargo build -p sim-cli --release` (sccache is a
>    shared compile cache — fast). Compile fails → return
>    `{"id","status":"build-failed","git_ref":null}` and STOP.
> 4. **Persist the code NOW** (worktree is ephemeral): `git checkout -b
>    autoresearch/exp-<iter4>-<coord>-<id>`; `git add -A` (artifacts/ is
>    gitignored → only code staged); `git -c user.name=autoresearch
>    -c user.email=autoresearch@local commit -m "exp <coord> <id>: <one-line>"`.
> 5. Return ONLY JSON: `{ "id", "git_ref":"autoresearch/exp-…", "status":"ready",
>    "surface_area", "change_summary":"<2 lines: what moved + the mechanism>" }`.
>    **Do NOT run any sims** (no determinism check, no eval) — the planner's
>    eval-queue rebuilds each branch and runs the authoritative determinism check
>    (P1 save/load byte-cmp + P2 cross-thread semantic fingerprint) and all
>    cross-seed evaluation. Do NOT write to `research/`.

## Evaluation (planner-owned eval-queue — the gate)

Run ONE capped evaluator over all persisted branches. **Background it** and poll
`summary.json` (don't block the planner — draft the next iteration's `directions/`
while it runs; do NOT poll the sims):

```bash
.claude/skills/autoresearch/eval-queue.sh \
  --base-ref <autoresearch/best sha> \
  --branches "autoresearch/exp-A autoresearch/exp-B …" \
  --cap 10 --build-cap 2     # cap = HARD global sim concurrency (memory-safe; OOM hit ~16-24).
                             # build-cap = concurrent release builds (builds are the heavy tenant).
  # defaults: --seeds 7,42,123,2026 --screen-seed 7 --screen-to 200000
  #           --confirm-to 500000 --explode-pop 40000 --min-common 2
# → run with run_in_background:true; then read <scratch>/summary.json when done.
```

It (1) **builds** each branch (base first, to warm sccache) in its own worktree
and runs the authoritative **determinism check** — P1 (two threads=1 processes →
byte-identical: catches HashMap/RNG/uninit) and P2 (threads=1 vs threads=4 →
identical semantic fingerprint: cross-thread invariant). `det≠ok` ⇒ the branch is
rejected and listed in `dropped` (never silently gone). (2) **Screens** each at
seed 7 → 200k as a **coarse DISASTER FILTER ONLY** — drops collapse (pop 0) /
explosion (pop > `explode-pop`, the eval-time trap; a per-sim watchdog also kills
runaways). **No action_effectiveness screen** — short-horizon aeff is noisy and a
seed-7 dip can be a seed-2026 rescue; the 500k confirm is the real HOLD gate.
(3) **Confirms** survivors (+ base) on all 4 seeds → 500k, all pooled at `--cap`.
(4) Emits `<scratch>/summary.json`:
- `branches[]` — per branch: `determinism`, `n_surviving`, `rescued_seeds`,
  `lost_seeds`, `common_survivors`, `sufficient_n`,
  **`clean_delta_vs_base_common_survivors`** (GATE ON THIS),
  `per_seed_delta_vs_base` (eyeball consistency), `metrics_mean_n4_DO_NOT_GATE`,
  `per_seed`.
- `dropped[]` — every build-failed / determinism-broken / screened-out branch with
  its stage + reason, so you can always tell "loser" from "never ran" (this is
  what prevents the iteration-1 distrust-and-re-run loop).

**Gate (read the JSON):** a branch is a winner iff `determinism==ok`,
`sufficient_n` is true (≥ `--min-common` common survivors — else the clean delta
is within-noise; do NOT trust it, especially for the `learning_slope`-scale
target), its targeted axis improves on the **clean delta**, and the HOLD pillars
(action_effectiveness, mi_sa) don't materially regress on the clean delta. Check
`per_seed_delta_vs_base` for consistency (not a one-seed fluke). A rescued seed is
a bonus, never a regression. The queue IS the gate's build+determinism+cross-seed
checks — advance the branch (fast-forward / cherry-pick) and re-run the queue on
any combined branch. The queue cleans up its scratch worktrees on exit.

(`sim-cli sweep` still exists for ad-hoc config grids, but the disciplined in-loop
path for code-change experiments is the eval-queue.)

## Provenance discipline

Per `research/CONVENTIONS.md`: every `Experiment` records `git_ref` + `base_ref` +
embedded `metrics` + a `# Reproduce` command (durable evidence; transient run
files are not relied upon); every `Finding`/`Mechanism` links its supporting
experiments; `best-program.md` lineage is the ordered chain of accepted
experiments. `log.md` append-only; `STATE.md` compacted each iteration.

## Budget & stopping

- ≤ 3 surface areas/iteration; ~8 research agents at once (cheap — build only).
  ALL sim compute flows through the eval-queue at `--cap 10 --build-cap 2`; never
  spawn agents that each run their own cross-seed sims.
- **Screen** is a coarse disaster filter (cheap, seed 7, 200k); **confirm** (4
  seeds, 500k) is the authoritative number. Never trust the sign/value of
  `learning_slope` from the short screen.
- Stop a surface area after **2 dry iterations** (no archive-improving result).
- Stop the loop when targets are met cross-seed, the frontier plateaus for K
  iterations, or budget is spent. Surface a milestone and offer a human PR from
  `autoresearch/best` → `main` (you never auto-PR to main).

## End-of-iteration checklist

- [ ] each experiment → an `Experiment` concept with `git_ref`/`base_ref`/metrics/Reproduce
- [ ] `best-program.md` updated (sha, lineage, metrics) iff the champion advanced
- [ ] `STATE.md` rewritten + compacted; `log.md` appended
- [ ] `autoresearch/best` advanced only via the eval-queue gate (clean-delta basis)
- [ ] zero writes to `main`
- [ ] eval-queue scratch worktrees cleaned up (the script does this unless `--keep`)
