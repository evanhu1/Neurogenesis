# sim-cli redesign — stateless, file-state, agent-native

> **⚠️ Historical / superseded.** This proposal predates the substrate redesign.
> The stateless world-as-file *shape* it argued for did land, but the concrete
> commands, the CBOR world format, the metric sidecar, and the `sim-metrics`
> pillars it references are gone. See `docs/sim-cli.md` for the current CLI.


**Status:** proposal (spec). **Audience:** implementer (human or agent).
**Supersedes:** the interactive stdin REPL shape described in `docs/sim-cli.md`.

## 1. Motivation

`sim-cli` exists for exactly one consumer: an LLM agent driving the NeuroGenesis
engine to do research (scrub a world forward, interrogate it, branch
counterfactuals, sweep configs). There are **no human users** to preserve a
TTY/REPL experience for.

The current shape is a long-lived stdin REPL holding one `Simulation` in RAM.
That is a *human-at-a-terminal* model, and it fights the agent's actual
interface — isolated request/response shell calls. Driving the REPL from an
agent harness requires a named-pipe + keep-open-holder + poll-the-output-file
contraption with **no command-completion signal and no output framing**, and the
long-lived process is a crash/`exit 144` liability that loses all in-RAM state
(snapshots included).

The agent's native interface is the shell: **a one-shot command that reads
explicit inputs, writes explicit outputs, prints JSON to stdout, and exits.**
The process exiting *is* the completion signal; stdout *is* the framed result;
the Bash tool already delivers it straight into context. So the optimal shape is:

> **A stateless one-shot CLI where the world is an explicit file artifact.**

State as a file (not hidden in a process) is strictly better for this consumer:

| Need | REPL/server | Stateless file CLI |
|---|---|---|
| Result framing | hand-rolled, fragile | free (stdout + exit code) |
| Persistent state | in-RAM, dies with process | a file on disk |
| Snapshot / fork | bespoke in-RAM clone map | `cp world.bin variant.bin` |
| Parallel sweep | orchestrate N processes + IPC | background N invocations |
| Robustness | crash loses everything | every call names its inputs |
| Build cost | MCP/daemon + session lifecycle | arg-parse + load/save |

The GO/NO-GO spike (§7) confirms the only thing that could have killed this —
serialization cost — is negligible (~30–100 ms vs a multi-minute `run-to`).

## 2. Core model

- **A world is a file.** `world.bin` holds a fully serialized `Simulation`
  (turn, seed, rng, counters, organisms, foods, grids — everything needed to
  deterministically resume).
- **Metrics history is a separate sidecar file.** `world.metrics` holds the
  recorder accumulators (ledger, tick summaries, lifetime rows, eco samples,
  `report_every`, `started_turn`). It is optional and only touched by commands
  that produce or consume history.
- **Mutating commands** (`new`, `step`, `run-to`, `watch`) read `--in`, write
  `--out`. `--out` **defaults to `--in`** (advance in place); pass an explicit
  `--out` (or `cp` first) to fork. They also accept `--metrics PATH` to
  accumulate history.
- **Read commands** (`state`, `pillars`, `inspect`, …) read `--in` (+ optional
  `--metrics` for the history-backed ones) and write nothing but stdout.
- **Output is JSON by default** (flip of today's text default), with `--text`
  to override. Errors print `{"error": "..."}` and exit non-zero.
- **Artifacts persist under `artifacts/`, never `/tmp`.** World blobs, metric
  sidecars, and sweep results default under `artifacts/cli/` (configurable) so
  they survive an agent session boundary. (A 5-hour research session lost all
  in-flight worlds + results to a `/tmp` wipe — see §12.)
- **Inline config overrides:** `new --set food_energy=12 --set
  passive_metabolism_cost_per_unit=0.0035` patches the loaded `WorldConfig`
  before world generation — no hand-built temp config dirs. Keys are the TOML
  config field paths; invalid keys list the valid vocabulary.
- **Determinism guarantee:** `save → load → step^n` is byte-identical to
  `step^n` in RAM. Enforced by a golden test (§9). RNG, seed, `turn`, and all
  `next_*_id` counters serialize exactly; transient/derived fields are rebuilt
  on load and never trusted for behavior.

## 3. Command surface

`load` becomes `new` (a constructor). `record`/`format` disappear (replaced by
the `--metrics` flag and the global `--json`/`--text` flag). Everything else
keeps its name and semantics.

| Command | Form | Mutates | Metrics |
|---|---|---|---|
| `new` | `new --seed N [--config P] [--set k=v]… [--scale W,POP] [--threads K] [--report-every R] --out w.bin [--metrics m.bin \| --no-metrics]` | creates world | mints sidecar (default on) |
| `step` | `step [N] --in w.bin [--out w.bin] [--metrics m.bin]` | advance N | updates sidecar if given |
| `run-to` | `run-to T --in w.bin [--out w.bin] [--metrics m.bin]` | advance to T | updates sidecar if given |
| `watch` | `watch T [--every E] --in w.bin [--out w.bin] [--metrics m.bin]` | advance to T, stream JSONL rows | pillars cols need sidecar |
| `bench` | `bench [N] --in w.bin` (no `--out`; discards) | advances, timed | reports `--metrics` flag |
| `turn` | `turn --in w.bin` | read | — |
| `state` | `state --in w.bin [--metrics m.bin]` | read | pillars line only if sidecar |
| `pillars` | `pillars --in w.bin --metrics m.bin` | read | **requires** sidecar |
| `eco` | `eco --in w.bin [--metrics m.bin]` | read | trajectory block needs sidecar |
| `timeseries` | `timeseries --in w.bin --metrics m.bin [--cols …] [--last K]` | read | **requires** sidecar |
| `lineage` | `lineage --in w.bin` | read | point-in-time |
| `genome` | `genome --in w.bin [--gene G] [--drift]` | read | point-in-time |
| `food` | `food --in w.bin` | read | — |
| `inspect` | `inspect ID --in w.bin` | read | — |
| `top` | `top FIELD [N] --in w.bin` | read | — |
| `hist` | `hist FIELD --in w.bin` | read | — |
| `find` | `find EXPR --in w.bin [--fields …] [--limit N]` | read | — |
| `brain` | `brain ID [--view …] --in w.bin` | read | — |
| `decide` | `decide ID --in w.bin` | read | — |
| `query` | `query --in w.bin [--metrics m.bin]` reads read-only commands from stdin | read (batch) | as needed |

Gaps to close while reworking: **add `--json` to `inspect`, `top`, `hist`**
(text-only today) so the surface is uniformly machine-parseable.

### 3.1 Read-burst batching (`query`) — the one efficiency valve

A pure read does ~0 ms of work but a world deserialize is ~50–100 ms. So 50
reads that each reload the world waste 2.5–5 s of pure I/O. Fix: **load once,
run many reads.** `query --in w.bin` reads read-only commands from stdin (one per
line, the existing mini-grammar) and emits a JSON array of results, then exits.

```bash
sim-cli query --in w.bin <<'EOF'
find energy < 5 and age > 400 --limit 10
inspect 8123
decide 8123
brain 8123 --view summary
EOF
```

This recovers the REPL's "one load, many probes" efficiency without any
long-lived state or framing problem — it's still one invocation, one exit, one
framed result. Mutating commands are intentionally **not** batchable here
(observe-between-mutations is rare, and each mutating op's cost dwarfs the
reload anyway — §7).

## 4. Implementation — world serialization (sim-core)

Per the serde-gap analysis, this is small. On `Simulation` (sim-core/src/lib.rs):

```rust
#[derive(Debug, Serialize, Deserialize)]   // add serde to the existing derive
pub struct Simulation {
    // ... all data fields serialize as-is (see table below) ...
    #[serde(skip)] cached_thread_pool: OnceLock<Arc<rayon::ThreadPool>>, // rebuilt lazily
    #[serde(skip)] turn_scratch: turn::TurnScratch,                       // Default on load
    #[cfg(feature = "instrumentation")]
    #[serde(skip)] action_records: Vec<Option<ActionRecord>>,            // instrumentation only
}
```

- **No Cargo edits.** `rand_chacha = { features = ["serde"] }` is already enabled
  (workspace Cargo.toml). `ChaCha8Rng` serializes its full internal state →
  exact RNG round-trip.
- **No new derives on shared types.** `WorldConfig`, `OrganismState`,
  `OrganismGenome`, `FoodState`, `Occupant`, `VisualProperties`,
  `PendingActionState`, `TerrainCell` all already derive serde. The only field
  type lacking it is `ActionRecord`, which we `#[serde(skip)]`.
- **Determinism-critical state** (`rng`, `seed`, `turn`, `next_organism_id`,
  `next_food_id`, `pending_actions`) all serialize exactly.
- **Derived caches** (`metrics`, `visual_map*`, `terrain_cells`): serialize
  as-is for v1 (simplest, can't diverge — they don't feed RNG). On load, call
  the existing `validate_state()` as the integrity gate. *(Optional later size
  win: `#[serde(skip)]` the three `visual_map*` grids + `terrain_cells` and
  rebuild via `build_visual_map_base()` on load — cuts the blob ~11→7 MiB. Defer;
  it adds a divergence-risk rebuild path for marginal benefit.)*

### 4.1 Serialization format — **decision needed (D1)**

The world blob carries runtime brain state, and `SensoryNeuronState.receptor`
uses `#[serde(flatten)]`, which **bincode cannot encode** (needs a known-size-
ahead format). Two options:

- **(Recommended) CBOR via `ciborium` for the world blob.** Self-describing,
  handles `flatten`, **touches no shared wire type and no web client.** Measured
  ~11 MiB / ~100 ms round-trip — already negligible. Keep `bincode` for the
  existing genome snapshots; the world blob is an isolated new call site.
- **bincode, by removing the `#[serde(flatten)]`** from `SensoryNeuronState`
  (replace with an explicit `receptor` field). ~3–5 MiB / ~30 ms. But this is a
  **wire-schema change**: per `AGENTS.md`, the TS `Api*` types + `protocol.ts`
  normalizers must change in lockstep. More blast radius for a perf win we don't
  need.

→ Recommend **ciborium** for v1; revisit bincode only if blob size ever matters.

## 5. Implementation — metrics sidecar (sim-cli + sim-metrics)

- Make `sim_metrics::Ledger` (+ `OrganismEntry`, `LearningAccumulator`) and the
  CLI's `EcoSample` derive `Serialize, Deserialize` (all plain ints / `Vec<u64>`
  / id-keyed maps — mechanical; watch map iteration order, but these feed only
  metrics, not behavior, so order can't break sim determinism).
- Sidecar payload: `{ ledger, tick_summary, lifetimes, samples, started_turn,
  report_every }`, serialized with ciborium (or bincode — no flatten here).
- **Recording = sidecar presence.** Mutating commands with `--metrics out`
  load (if it exists), ingest each tick, write back. Without `--metrics`, they
  take the existing allocation-free `advance(recorder = None)` fast path.
- **Default on:** `new` mints the sidecar next to the world unless `--no-metrics`
  is passed, because "create → accumulate → query pillars" is the dominant
  workflow and threading the flag everywhere is friction. Throughput-sensitive
  scrubs opt out.
- `report_every` (today set at `load`, used by `derive_interval_metrics`) lives
  in the sidecar.
- History-backed reads (`pillars`, `timeseries`, `eco` trajectory, the optional
  `state` pillars line) take `--metrics in` and error/degrade exactly as the
  current point-in-time-vs-recorded contract does when it's absent.

## 6. Agent ergonomics (the point of the redesign)

- **JSON by default** (sentinel `{"error":…}` + non-zero exit on failure).
- **`--out` defaults to `--in`** → the common "advance my world" case is just
  `sim-cli run-to 500000 --in w.bin`. Forking is an explicit `--out` or `cp`.
- **Invalid args still print the valid vocabulary** (keep today's good behavior).
- **`--scale`/non-canonical marker persists in the blob** so downstream
  `pillars`/`state` keep tagging `[scaled: non-canonical]`.
- Research workflows this unlocks directly:
  ```bash
  # Branch a counterfactual from a common state — no re-sim:
  sim-cli new --seed 7 --out base.bin
  sim-cli run-to 100000 --in base.bin            # advance in place
  cp base.bin armA.bin; cp base.bin armB.bin     # fork = copy
  sim-cli run-to 500000 --in armA.bin --metrics armA.metrics &   # parallel
  sim-cli run-to 500000 --in armB.bin --metrics armB.metrics &
  wait
  sim-cli pillars --in armA.bin --metrics armA.metrics   # compare JSON
  sim-cli pillars --in armB.bin --metrics armB.metrics
  ```
  (This is exactly the A/B config comparison done by hand earlier in the
  learning investigation — now a few cheap shell lines.)

## 7. GO/NO-GO evidence (measured)

Canonical seed-7 world advanced to turn 175,000 → **1,406 organisms**:

| component | size | serialize | deserialize |
|---|---|---|---|
| organism population (CBOR) | 6.2 MiB | 11.4 ms | 44.2 ms |
| grids (occupancy/visual/terrain/food schedule) | ~5.0 MiB | ~POD memcpy | — |
| **total world blob** | **~11 MiB** | **~30–100 ms round-trip** | |

- vs a `run-to 500000` (~180,000 ms of sim): **~0.03–0.06% overhead.** Even a
  tiny `run-to +1000` (~360 ms) dwarfs the I/O. **GO.**
- Caveat that shaped §3.1 and §4.1: reads are free, so per-read reload is the
  one place I/O isn't negligible → batch reads (`query`); and bincode is blocked
  by `flatten` → use ciborium.
- Metrics history at 500k ticks ≈ 28 MB → sidecar, not inlined (§5).

## 8. What's explicitly dropped

- The long-lived stdin REPL and its in-RAM session. (`query` covers read bursts.)
- `record on/off/status` → `--metrics` presence. `format` → global `--json`.
- `watch` and `bench` are kept but are the two "odd" commands: `watch` emits
  N JSONL rows from one invocation (a run+progress-log); `bench` advances a
  throwaway world for timing and takes no `--out`.

## 8a. Research-infra additions (folded in from a 5-hour CLI session)

A real multi-hour learning investigation surfaced friction the bare
read/transform surface doesn't address. Ranked by the cost they imposed:

### A. Warm-once, fork, run-window — the headline speedup
The pillars only read the **last 10% window** (tick 460k–500k of 500k), yet every
config re-paid 460k ticks of identical warmup. With world-as-file this collapses
to: warm **once**, fork with `cp`, pay only the window per intervention.

```bash
sim-cli new --seed 7 --out artifacts/cli/warm.bin --metrics artifacts/cli/warm.metrics
sim-cli run-to 400000 --in artifacts/cli/warm.bin --metrics artifacts/cli/warm.metrics
# fan out N late-window interventions, each paying only 100k ticks:
for e in 10 12 14; do
  cp artifacts/cli/warm.bin     artifacts/cli/e$e.bin
  cp artifacts/cli/warm.metrics artifacts/cli/e$e.metrics
  sim-cli run-to 500000 --in artifacts/cli/e$e.bin --metrics artifacts/cli/e$e.metrics &
done; wait
```
≈5× fewer ticks across a sweep. *Caveat:* an intervention that only takes effect
at fork time (a config `--set` mid-run) measures a **transient**, not the
evolved equilibrium — fine for directional screening, confirm keepers from a
cold canonical run. (Config is fixed at `new`; a future `set` mutating command
could patch a warmed world's `WorldConfig` in place for this pattern.)

### B. Granular data folded into `pillars` (✅ shipped — no separate command)
A pillar number says *what* but not *why*. Rather than a separate `--explain`
command, **`pillars` now always emits a `granular` section**: the full
per-interval metric series behind the windowed scores (each interval's
action_effectiveness, plant/prey consumption rates, mi_sa, learning_slope, pop),
with the scoring window marked. The agent reads each pillar's windowed mean
against its underlying trajectory in one call — e.g. seeing the learning_slope
go more negative interval-by-interval as a cohort spirals toward starvation.

Future deeper decompositions (learning slope by death-cause × age-bucket;
per-action success-vs-age curve; food-desert metric) need new `sim-metrics`
aggregations and would extend `OrganismLifetimeRow` (which also feeds the eval
Parquet schema) — deferred to avoid that blast radius. The per-interval series
covers the temporal "why" without it.

### C. `sweep` — built-in cartesian experiment harness (✅ shipped)
Hand-driven sequential heredocs were the orchestration bottleneck.

```
sim-cli sweep --grid food_energy=10,12,14 passive_metabolism_cost_per_unit=0.003,0.005 \
              --seeds 7,42,123,2026 --to 500000 \
              [--baseline food_energy=12,passive_metabolism_cost_per_unit=0.005] \
              [--threads K] [--jobs J] [--out-dir D]
```
- Runs the cartesian product × seeds, **parallel** (bounded to `--jobs`, default
  CPU count; each run uses `--threads` intent threads, default 1 — parallelism
  comes from running many independent worlds at once).
- Aggregates per cell: **mean/min/max across the seed cohort** (matches how the
  eval scores — never trust a single seed; §12 #5).
- **Writes a JSON result file** to `--out-dir` (default `artifacts/runs/`, so it
  survives the session) and prints the path + a ranked table with **Δ vs the
  `--baseline` cell** to stdout. `KEY`s are config field names (same vocabulary
  as `new --set`).
- Not built (deferred, were in the original sketch): auto-append to
  `EXPERIMENT_LOG.md` (replaced by the durable result file), the
  `(config-hash,seed,to)` result cache, and `--warm-from` checkpoint reuse. The
  warm-once-fork pattern (§8a-A) covers checkpoint reuse via `cp` today.

This also maps onto the agent's `Workflow` tool (fan-out → aggregate); the
built-in `sweep` is the single-invocation path. Prefer one over hand-sequencing.

### D. Cohort scoring (`--seeds`, `--aggregate`)
Single-seed-by-hand risks overfitting (the eval uses 4 seeds). `sweep` covers the
batch path; for a single config, `pillars --aggregate` over a small seed cohort
(several world files) reports mean ± spread in one call. In the stateless model a
"cohort" is just N world files sharing a config; the aggregator globs them.

### E. Validated fast proxy (methodology, low build priority)
`--scale` proved unreliable for pillar *values* and even learning *direction*.
Rather than rediscovering that per-experiment, calibrate a cheap signal (shorter
horizon or smaller world) against canonical pillars **once**, record where the
correlation holds, and only use the proxy there. A later `sim-cli calibrate`
could automate the correlation check; for now it's a documented discipline.

## 9. Validation

- **Determinism golden test:** build a world, `step` N in RAM; separately
  save→load→`step` N; assert byte-identical `Simulation` (or a state hash).
  Run across a couple of seeds and a mid-run save point (non-zero turn, with
  pending gestation/actions in flight) to catch transient-field bugs.
- Save→load→save must be a fixed point (idempotent blob).
- `validate_state()` invoked on every load as the integrity gate.
- Existing `cargo test --workspace` must stay green (engine behavior unchanged;
  this is purely additive serialization + a new CLI front end).

## 10. Phasing

1. **Engine serde** (§4) + determinism check (§9). Lands independently; useful on
   its own (world save/load, repro states). ✅ *done*
2. **Ledger/EcoSample serde + sidecar I/O** (§5).
3. **New CLI front end** (§3): arg surface, `--in/--out/--metrics`, JSON-default,
   `--set` overrides (§8a-cohort), `artifacts/` defaults, `query` batch mode,
   `--json` on inspect/top/hist. Retire the REPL.
4. **Research infra** (§8a): `pillars --explain` first cut (death-cause × age
   slope + per-action success-vs-age), then `sweep` (grid × seeds, parallel,
   delta table, `EXPERIMENT_LOG.md` append, `artifacts/` result cache).
5. Update `docs/sim-cli.md` + `AGENTS.md` to the new surface; full
   `cargo test --workspace` green; A/B + warm-fork smoke test.

Order rationale: 1–3 are the foundation (a usable stateless CLI). 4 is the
research-throughput multiplier and depends on the sidecar (2) + surface (3).
The warm-once-fork pattern (§8a-A) needs nothing beyond 1–3 (it's just `cp`).

## 11. Decisions (resolved)

- **D1 — world blob format:** ✅ **ciborium (CBOR).** No wire-schema change, no
  web-client sync. §4.1.
- **D2 — metrics:** ✅ **default-on** (`new` mints the sidecar; `--no-metrics`
  opts out).
- **D3 — grids:** ✅ **serialized as-is for v1** (simplest, can't diverge).
- **D4 — REPL:** ✅ **retire** — agent-only + `query` covers read bursts.

## 12. Source: operational learnings (5-hour CLI research session)

Friction observed, ranked by cost, with the §8a item that addresses each:
1. **Wall-clock per data point** (5–8 min canonical; serial; ~1 exp / 6–15 min) →
   §8a-A warm-once-fork (only pay the scored window) + §8a-C parallel `sweep`.
2. **No config-param override on load** (~12 hand-built temp config dirs) →
   `new --set k=v` (§2, §8a-C).
3. **No checkpoint/fork of a warmed sim** (re-ran 460k warmup every config) →
   world-as-file + `cp` (§2) and §8a-A.
4. **No causal decomposition** (had to burn a whole run to prove plasticity
   wasn't the cause; inferred the food-desert story from death-cause aggregates)
   → §8a-B `pillars --explain`.
5. **Single-seed by hand** (overfitting risk; eval uses 4 seeds) → §8a-C/D
   cohort scoring.
6. **`/tmp` outputs ephemeral** (session boundary wiped results + killed an
   in-flight run) → `artifacts/` defaults (§2).

Harness-level (outside sim-cli): long `until`-loop waits auto-backgrounded; no
parallel-seed execution. Addressed by `sweep`'s internal parallelism and by
driving fan-out through the `Workflow` tool rather than sequential heredocs.
