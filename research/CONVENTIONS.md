---
type: Bundle Conventions
title: Research bundle conventions (OKF)
description: Concept types, frontmatter schema, and provenance rules for the autonomous-research knowledge bundle.
tags: [okf, conventions, autoresearch]
---

# Research knowledge bundle (OKF)

This directory is an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
bundle: the **database** of the autonomous-research system. Every `.md` file is a
**Concept** (YAML frontmatter + markdown body). Concept **ID = file path minus
`.md`** (e.g. `experiments/0007-food-ecology-a` → id `experiments/0007-food-ecology-a`).
Relationships are plain markdown links. Evidence goes under `# Citations`.

It is **version-controlled** (committed) — the durable thread of knowledge. Raw,
transient evidence (sweep result JSON) lives in the gitignored `artifacts/runs/`
and is *cited*, not stored here.

Reserved files: `index.md` (navigational index), `STATE.md` (the planner's
live dashboard — read first every session), `log.md` (append-only iteration
history). `best-program.md` is the single current champion.

## Concept types & frontmatter

All concepts share OKF's recommended keys (`type`, `title`, `description`,
`tags`, `timestamp`) plus the extensions below. **Provenance fields are
mandatory** for `Experiment` and `BestProgram` — they are how a discovery is
traced back to the code + run that produced it.

### `type: Experiment` — `experiments/<iter4>-<coordinator>-<id>.md`
One research-agent run = one code change + its evaluation. The atomic unit of
provenance.
```yaml
type: Experiment
title: <short>
description: <one line>
iteration: <int>            # planner iteration that spawned it
coordinator: <surface-area> # e.g. plasticity-genome
agent: <id>                 # research agent id within the coordinator
surface_area: <lever-family>
base_ref: <sha>             # commit it forked from (autoresearch/best at spawn)
git_ref: autoresearch/exp-<iter4>-<coordinator>-<id>   # branch holding the change (durable!)
status: built | build-failed | evaluated | promoted | rejected
determinism: ok | broken | not-checked
seeds: [7, 42, 123, 2026]
metrics:                    # cross-seed mean of the RAW pillars (no [0,1])
  plant_consumption_rate: <f>
  prey_consumption_rate: <f>
  action_effectiveness: <f>
  mi_sa: <f>
  learning_slope: <f>
baseline_metrics: { ... }   # same shape, the base_ref's metrics
delta: { ... }              # metrics - baseline_metrics
timestamp: <ISO 8601>
tags: [...]
```
Body: `# Hypothesis`, `# Change` (what code moved + why), `# Result` (table,
cross-seed mean±spread), `# Learnings`, `# Concerns`. `# Citations` link the
sweep result file(s) under `artifacts/runs/` and the diff.

### `type: Finding` — `findings/<slug>.md`
A validated conclusion (survived the verification gate). Links to its supporting
Experiments.
```yaml
type: Finding
confidence: high | medium | low
status: active | superseded
supported_by: [experiments/..., ...]   # also linked in body prose
```

### `type: Direction` — `directions/<slug>.md`
A promising, **not-yet-exhausted** avenue ("untapped alpha"). The planner mines
these to assign coordinator goals.
```yaml
type: Direction
priority: high | medium | low
status: open | in-progress | exhausted
surface_area: <lever-family>
```

### `type: Mechanism` — `mechanisms/<slug>.md`
A durable "law" of the system (e.g. *plant_rate ≈ metabolism / food_energy*).
Slow-growing, highest-value. Promoted from repeatedly-supported Findings.
```yaml
type: Mechanism
confidence: high | medium | low
supported_by: [experiments/..., findings/...]
```

### `type: DeadEnd` — `dead-ends/<slug>.md`
An avenue ruled out, with the reason, so it is never re-explored.
```yaml
type: DeadEnd
reason: <one line>
ruled_out_by: [experiments/...]
```

### `type: BestProgram` — `best-program.md` (singleton)
The current champion = a concrete git ref the next iteration forks from.
```yaml
type: BestProgram
git_ref: autoresearch/best          # branch; body records the exact sha + iteration
iteration: <int>
metrics: { ...cross-seed raw pillars... }
lineage: [experiments/..., ...]     # the accepted experiments that built it, in order
```

## Provenance rules (get this right)

1. **Every experiment persists its code as a git branch** `autoresearch/exp-*`
   *before* the agent returns (worktrees are ephemeral). The `Experiment`
   concept records `git_ref` + `base_ref`.
2. **Durable evidence = the concept itself, not external files.** The raw sweep
   JSON lives in gitignored `artifacts/runs/` *inside an ephemeral worktree*, so
   it does **not** survive — never rely on a citation to it. An experiment's
   durable, reproducible evidence is:
   - the **`metrics` frontmatter** (the numbers, embedded), and
   - the **`git_ref`** (the diff — `git show <git_ref>`), and
   - a **`# Reproduce`** line: the exact `sim-cli` command + seeds + base_ref, so
     anyone can regenerate the numbers from the committed code.
   Cite the diff, not the transient JSON. (If you truly want to keep a result
   file, copy it into the committed `references/` — but the embedded metrics are
   normally enough.)
3. **Every claim links to evidence.** Findings/Directions/Mechanisms link the
   Experiments that support them.
4. **The champion's lineage is a link chain.** `best-program.md` lists every
   accepted experiment in order — you can replay how the best program was built.
5. **`log.md` is append-only** (one entry per iteration). `STATE.md` is
   rewritten/compacted each iteration but never loses anything that isn't already
   in the append-only layer.
6. **IDs are stable paths.** Prefer absolute links (`/experiments/0007-...md`).
   Experiment files are named `experiments/<iter4>-<coordinator>-<id>.md`
   (e.g. `experiments/0007-metabolism-lower-floor.md`), matching the
   `autoresearch/exp-<iter4>-<coordinator>-<id>` branch.
