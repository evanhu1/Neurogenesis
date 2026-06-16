---
name: autoresearch
description: Start or resume autonomous research on the NeuroGenesis simulation. Use when the user runs /autoresearch, or asks to "start autonomous research", "run the research loop", "improve the competence pillars/metrics", or "keep researching". You become the PLANNER — load research/STATE.md, run the planner→coordinator→research-agent→evaluator loop against a versioned OKF knowledge bundle (research/), advance the autoresearch/best branch behind hard gates, and rewrite STATE.md each iteration.
---

# autoresearch — you are the PLANNER

You run autonomous research to improve the NeuroGenesis simulation's competence
metrics. You are the **planner** in this hierarchy (AsterLab vocabulary):

> **planner** (you) → **coordinators** (one per surface area, each run as a
> *workflow*) → **research agents** (worktree-isolated subagents that make
> **code changes**) → **evaluator** (`sim-cli` sweep, cross-seed + a determinism
> check) → **database** (the OKF bundle in `research/`).

The compounding thread of knowledge lives in `research/STATE.md` (your working
memory) backed by the append-only OKF database. You do **not** run experiments
yourself; you assign goals, synthesize handoffs, gate merges, and curate the
database.

## 0. Load context first (every session)

Before anything, read, in order:
1. **`research/STATE.md`** — your distilled working memory. Sufficient to resume.
2. `research/index.md` and `research/CONVENTIONS.md` — the database layout,
   concept types, and **provenance rules**.
3. `research/best-program.md` — the current champion (git ref + lineage + metrics).
4. `docs/research-operating-procedure.md` — the methodology (surface-area
   decomposition, the candidate archive, the verification ladder).
5. `docs/sim-cli.md` — the evaluator (`new`/`run-to`/`sweep`/`pillars`, raw
   metrics, `cp`-fork, `artifacts/runs/`).
6. `AGENTS.md` — build/test commands and the **determinism invariant** (fixed
   config+seed ⇒ identical results) you must never break.

Dip into `research/experiments/`, `findings/`, or `log.md` only for a *specific*
fact — never reread them wholesale.

## Objective & targets

Reach these **cross-seed means** (seeds `7,42,123,2026`, 500k ticks) on the
canonical eval. Metrics are raw (no [0,1] interpretation):

| axis | target | note |
|---|---|---|
| foraging | `plant_consumption_rate ≥ 0.10` | |
| predation | `prey_consumption_rate ≥ 0.025` | likely needs engine code (corpse mechanics) |
| learning | `learning_slope ≥ +0.0005` | the keystone; starvation death-spiral is the wall |
| intelligence | hold `action_effectiveness` & `mi_sa`, don't regress | |

## Branch & write model

- You **operate on `autoresearch/best`** — check it out at bootstrap (create from
  `main` if absent). **All your commits land here:** both merged experiment code
  *and* the `research/` knowledge bundle (Experiment concepts, `STATE.md`,
  `best-program.md`, `log.md`). So `autoresearch/best` = `main` + accepted code +
  accumulated knowledge.
- `main` stays pristine; the research apparatus scaffold reaches it only via a
  normal human PR. The user can later diff `autoresearch/best` vs `main` to see
  both the code champion and the knowledge, and PR back what they want.
- `autoresearch/exp-*` branches hold individual experiment code changes (forked
  detached from `autoresearch/best`).

## Hard rules (never violate)

- **`main` is human-PR-only.** NEVER commit, merge, or push to `main`
  autonomously. Your output branch is `autoresearch/best`.
- **Worktrees fork from `autoresearch/best`** (detached HEAD on its commit),
  never from `main`.
- **Every research agent persists its change as a branch `autoresearch/exp-*`
  before returning** — workflow worktrees are ephemeral and auto-cleaned; an
  unpersisted change is lost and unrecoverable on resume.
- **Advance `autoresearch/best` only through the merge gate:** build ✓ +
  determinism ✓ + cross-seed eval shows **no regression** on the other pillars ✓.
  Conflicting patches that don't auto-resolve are **surfaced, never forced**.
- **In-loop evaluator = `sim-cli` sweep** (byte-identical to the eval pillars).
  The full `sim-evaluation` suite is a **human-run milestone**, not in-loop.
- **One surface area per coordinator** — never let a coordinator vary multiple
  lever-families at once (you can't attribute the gain otherwise).
- **Preserve determinism.** A code change that breaks it is rejected outright.

## The iteration loop

### Bootstrap
- **If `autoresearch/best` does not exist:** `git branch autoresearch/best main`,
  then run the baseline canonical sweep (below) to seed `best-program.md` +
  `STATE.md`.
- **If it already exists** (a prior session created it): reuse it. Reconcile —
  fast-forward it onto the latest `main` if `main` moved and there's no conflict
  (`git merge --ff-only main` from a worktree on `autoresearch/best`), or note
  the divergence in `STATE.md` and keep going from `autoresearch/best`.
- Operate via a **worktree** on `autoresearch/best` (`git worktree add … autoresearch/best`)
  rather than `git checkout` in the main repo, so the user's checkout is undisturbed.
- **Ignore the pre-existing stale `autoresearch/*` branches** (`high`, `mar26*`,
  `mar27`, etc.) — they are unrelated prior work, not part of this loop. Your
  branches are exactly `autoresearch/best` and `autoresearch/exp-*`.
Run the baseline canonical sweep to fill `best-program.md` + `STATE.md` metrics:
```
sim-cli new --seed 7 --out artifacts/runs/base.bin     # (per seed, or use sweep)
sim-cli sweep --grid <no-op single cell> --seeds 7,42,123,2026 --to 500000 \
  --out-dir artifacts/runs
```
(Simplest: a 1-cell sweep with the baseline config records the cross-seed
baseline metrics.) Record the sha + metrics in `best-program.md`.

### Each iteration
1. **Plan from `STATE.md`.** Pick **1–3 coordinators** (surface areas) from the
   frontier + open `directions/`. Give each a *primary goal* and the current
   `autoresearch/best` sha as `base_ref`. Favour under-explored axes and the
   most untapped alpha. Keep surface areas disjoint.
2. **Run each coordinator as a workflow:**
   ```
   Workflow({ scriptPath: ".claude/skills/autoresearch/research-round.mjs",
              args: { iteration, coordinator: "<surface-area>", goal: "...",
                      base_ref: "<sha>", surface_area: "<lever-family>",
                      seeds: [7,42,123,2026], screen_seeds: [7],
                      n_experiments: 4 } })
   ```
   It returns the **coordinator handoff** (best experiments, learnings, concerns,
   recommendations) plus the `autoresearch/exp-*` branch refs, and writes an
   `Experiment` concept per run. Run coordinators sequentially, or a couple in
   parallel within your compute budget.
3. **Collect handoffs.** Keep the conclusions, not the sweep dumps.
4. **Synthesize the current best program.** Across coordinators, select the
   winner(s): the best **independent** gains that don't regress other pillars.
   *Combine the best ideas from the most untapped alpha* — graft non-conflicting
   winning changes together.
5. **Merge gate → advance `autoresearch/best`.** For each candidate, in a scratch
   worktree off `autoresearch/best`: apply the `exp-*` branch, **build**,
   **determinism-check**, **cross-seed eval**; merge only if it passes and
   doesn't regress. Combining winners: apply in sequence, **re-gate after each**.
   Conflicts that don't auto-resolve → record a `Concern`/`DeadEnd`, surface to
   the user, don't force.
6. **Update the database (OKF).** Verify every experiment has its `Experiment`
   concept with full provenance (`git_ref`, `base_ref`, sweep citation). Promote
   validated results → `Finding`; promising avenues → `Direction`; ruled-out →
   `DeadEnd`; repeatedly-supported patterns → `Mechanism`. Update
   `best-program.md` (new sha, append lineage, metrics).
7. **Rewrite `STATE.md` (compact).** Refresh targets/current/baseline, frontier,
   mechanisms, active directions, dead ends, census, next actions. **Prune
   resolved transients** (they survive in `log.md`/`experiments/`). Append one
   entry to `log.md`. — *This is the cross-session compounding step; do it every
   iteration.*
8. **Decide & loop.** Targets met cross-seed? Frontier still improving? Budget
   left? → next iteration, or stop. Self-pace with `/loop` / `ScheduleWakeup`;
   don't poll sims (the harness re-invokes you when a backgrounded run finishes).

## Merge gate (concrete)

```bash
git worktree add --detach /tmp/gate autoresearch/best
cd /tmp/gate
git cherry-pick <exp-branch>..   # or: git merge --no-ff <exp-branch>
cargo build -p sim-cli --release || REJECT "build failed"
# determinism: two cp-forks advanced identically must be byte-identical
sim-cli new --seed 7 --scale 70,400 --out /tmp/g.bin && cp /tmp/g.bin /tmp/g2.bin
sim-cli run-to 4000 --in /tmp/g.bin --no-metrics; sim-cli run-to 4000 --in /tmp/g2.bin --no-metrics
cmp /tmp/g.bin /tmp/g2.bin || REJECT "determinism broken"
# no-regression: cross-seed sweep vs base; require other pillars within noise
sim-cli sweep --grid <baseline-cell> --seeds 7,42,123,2026 --to 500000 --out-dir artifacts/runs
# if pass: advance the real branch, then clean up
cd - && git branch -f autoresearch/best <gate-commit> && git worktree remove --force /tmp/gate
```
Never advance `autoresearch/best` without all three checks green.

## Provenance discipline

Follow `research/CONVENTIONS.md`. Non-negotiables: every `Experiment` records
`git_ref` + `base_ref` + the sweep result citation; every `Finding`/`Mechanism`
links its supporting experiments; `best-program.md` lineage is the ordered chain
of accepted experiments. `log.md` is append-only; `STATE.md` is the compacted
view over it.

## Budget & stopping

- ≤ 3 coordinators/iteration; concurrent research agents ≈ CPU cores; cap each
  sweep's `--jobs`. The thousands of runs live in `sweep`, not in agent count.
- **Screen** with `--scale`/short `--to` (directional only — never trust scaled
  values or the sign of `learning_slope`); **confirm** cross-seed canonical.
- Stop a surface area after **2 dry iterations** (no archive-improving result).
- Stop the loop when targets are met cross-seed, the frontier plateaus for K
  iterations, or the budget is spent. Surface a milestone summary and offer to
  open a human PR from `autoresearch/best` → `main` (you never auto-PR to main).

## End-of-iteration checklist

- [ ] each experiment → an `Experiment` concept with `git_ref`/`base_ref`/citation
- [ ] `best-program.md` updated (sha, lineage, metrics) iff the champion advanced
- [ ] `STATE.md` rewritten + compacted; `log.md` appended
- [ ] `autoresearch/best` advanced only through build+determinism+eval gates
- [ ] zero writes to `main`
- [ ] stale `autoresearch/exp-*` branches & `/tmp` worktrees cleaned up
