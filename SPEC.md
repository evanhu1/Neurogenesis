# sim-cli v2 — Specification

> **⚠️ Historical / superseded.** This spec predates the substrate redesign and
> describes the previous stateful, `Simulation`-in-memory `sim-cli` built on the
> `sim-metrics` ETL / `pillars` stack (all removed). The current CLI is a lean,
> **stateless world-as-file** tool — see `docs/sim-cli.md` for the live command
> reference. Kept for design-history context only.


## Overview

`sim-cli` is a headless, stateful command client for the NeuroGenesis engine,
built to be **the primary research instrument for a coding agent piloting the
simulation**. It holds one `Simulation` in memory, scrubs it forward, and
exposes signal-dense, token-efficient observability into every layer of the
artificial-life run: evolution progress, organism health, ecological dynamics,
world state, individual brains, and lineage structure.

The overarching project goal is *evolving intelligent brains through artificial
life*. The CLI's job is to let an agent **understand and reason about whether,
how, and why** that is happening — both by deeply interrogating a single run and
by running hypothesis-testing experiments across seeds/configs.

Today's `sim-cli` (single 646-line `main.rs`) is a narrow foraging-debug REPL
that hand-reimplements one eval metric. v2 generalizes it into a full research
cockpit and removes the metric-drift trap by sharing computation with the eval
harness.

## Goals & Non-Goals

### Goals
- **Agent-first ergonomics**: every command emits compact, deterministic,
  parse-friendly output by default, with `--json` for exact extraction.
- **Single source of metric truth**: the eval pillars and their raw sub-signals
  are computed by a *shared library* used by both `sim-evaluation` and `sim-cli`,
  so live CLI numbers are byte-identical to eval numbers (no hand-rolled drift).
- **Full observability**: aggregate dashboards (pillars, ecology, lineage,
  genome drift, population timeseries) **and** per-organism drill-down (brain
  topology, activations, decision explanation, plasticity state).
- **Speed without limits**: max-throughput scrubbing (`record off`), parallelism
  control, optional scale-down for fast iteration, a `bench` command.
- **Two research loops**: deep single-run understanding *and* agent-driven
  hypothesis testing / sweeps (sweep *primitives*; a built-in runner is deferred).

### Non-Goals (v2)
- **No state mutation / interventions** (inject, kill, lesion, edit-config
  mid-run). Deferred to v3 — but the command/verb space is reserved so they can
  be added without rework. v2 is strictly read-only.
- **No backwards time-travel / checkpointing.** Scrubbing is forward-only
  (`step`, `run`, `goto`). Going "back" = `reset` and re-run.
- **No built-in multi-process sweep runner.** v2 ships the primitives (fast
  `load`/`reset`, `--json`, scriptable stdin); the agent orchestrates sweeps.
- **No Parquet/ETL.** Persistence stays in `sim-evaluation`.

## Key Decisions & Tradeoffs

These were settled during the design interview; rationale recorded so they are
not relitigated.

| Decision | Choice | Rationale |
|---|---|---|
| **Eval coupling** | Extract a shared **`sim-metrics`** lib crate (ledger + intervals + pillars). Both `sim-evaluation` and `sim-cli` depend on it. | The pillars are the project's working definition of "progress." Making them a shared definition (not eval's private property) means the agent can reproduce/debug an eval score live, and CLI↔eval can never drift. Worth the upfront refactor of eval. |
| **Primary readout** | **Pillars + decomposition side by side.** | Agents reason on raw numbers; the clamped [0,1] composite alone hides information. Always show the composite *and* the raw sub-signals that feed it. |
| **Research mode** | Deep single-run **and** hypothesis testing/sweeps. Forward-only scrub by chunks + `goto T`. | The two real loops. Backwards adds checkpoint cost for little value (`reset` covers it). |
| **Must-have capabilities** | Brain/neural inspection · Lineage & genome drift · Ecological dynamics. | These are the layers eval doesn't surface but an agent needs to *understand* a run. |
| **Interventions** | **Read-only in v2**, verb space reserved for v3. | Keeps every session exactly eval-reproducible; ship observability first. |
| **Recording** | **Explicit `record on/off`** (single switch). | Predictable cost. Fast scrubs pay nothing; the agent enables recording only for the window it wants to measure. Point-in-time queries always work without it. |
| **Output** | **Compact text default + `--json`** (and global `format json`). | Dense/skimmable for reasoning, structured when exact parsing is needed. |

## Technical Architecture

### Crate topology

```
sim-types ── sim-config ── sim-core ─┐
     │                                │
     └──────────── sim-metrics  ◄─────┘   (NEW lib crate)
                    ▲       ▲
        sim-evaluation     sim-cli
        (ETL→Parquet)      (live, interactive)
```

#### `sim-metrics` (new library crate) — the keystone

Extract from `sim-evaluation` the metric computation that is currently
binary-only, and make it consumable both from persisted Parquet rows (eval) and
from a live in-memory ledger (CLI). Moves out of `sim-evaluation` and into
`sim-metrics`:

> NOTE: the actual pillars/signals were verified against current source during
> Milestone 1 (the earlier review summaries that seeded this spec were stale).
> The real axes are **foraging / predation / intelligence / learning** with the
> signals below — not the foraging/intelligence/competition set drafted earlier.

- **Schema row types & enums** (`sim-metrics/src/schema.rs`): `TickSummaryRow`
  `{ tick, descendant_population }`, `OrganismLifetimeRow`
  `{ id, origin, death_tick, total_actions, contingent_actions, failed_actions,
  plant_consumptions, prey_consumptions, joint_sensory_action[SENSORY_BIN×ACTION],
  learning_slope }`, `OrganismOrigin`, and the histogram-shape constants
  (`ACTION_COUNT`, `SENSORY_BIN_COUNT`, `JOINT_LEN`, `DESCENDANT_CODE`).
  (`GenomeSnapshotIndexRow` stays in `sim-evaluation` — it's persistence-only.)
- **Interval layer** (`sim-metrics/src/intervals.rs`): `IntervalMetrics` +
  `derive_interval_metrics(&[TickSummaryRow], &[OrganismLifetimeRow],
  report_every, total_ticks) -> Vec<IntervalMetrics>`. Pools descendant lifetime
  rows by death-tick into the raw windowed signals: `action_effectiveness`
  (successful contingent / total actions), `plant_consumption_rate`,
  `prey_consumption_rate`, `mi_sa` (Miller-Madow I(S;A)), `learning_slope`
  (mean within-life success-vs-age slope), plus `pop` context.
- **Pillar layer** (`sim-metrics/src/pillars.rs`): `PillarScores` with four axes
  over the last-10%-of-intervals window — `foraging` = clamp01(plant_rate / 0.20),
  `predation` = clamp01(prey_rate / 0.05), `intelligence` = softened weighted
  geomean of `action_effectiveness` and `mi_sa/0.16` (0.5/0.5), `learning` =
  clamp01(learning_slope / 0.001).
- **Ledger + live ingest** (`sim-metrics/src/ledger.rs`, `ingest.rs`): the same
  `Ledger` that accumulates per-organism rows, plus storage-agnostic free
  functions that drive it from live tick data — identical sequence to the eval
  orchestration, so both observe the same rows:
  ```rust
  pub fn register_founders(ledger: &mut Ledger, organisms: &[OrganismState]);
  // Per tick after Simulation::tick(); returns the lifetime rows of organisms
  // that died this tick (caller persists them or collects them in memory).
  pub fn ingest_tick(ledger: &mut Ledger, tick: u64, delta: &TickDelta,
                     action_records: &[Option<ActionRecord>]) -> Vec<OrganismLifetimeRow>;
  ```
  The CLI's `Recorder` holds the `Ledger`, a `Vec<TickSummaryRow>`, and a growing
  `Vec<OrganismLifetimeRow>` (deaths + end-of-run survivors); it derives metrics
  on demand with `derive_interval_metrics(&ticks, &lifetimes, …)` /
  `compute_pillar_scores(&intervals)`.

**Status (Milestone 1, done):** `sim-metrics` extracted with `schema`, `ledger`,
`ingest`, `intervals` (refactored to take row slices, not a `DatasetReader`),
`pillars`, and `stats`. `sim-evaluation` now consumes it: the moved symbols are
re-exported under their historical `crate::…` paths (minimal churn) and the
orchestration loop calls `register_founders`/`ingest_tick`. `sim-metrics` enables
`sim-types/instrumentation` unconditionally (the ledger needs `ActionRecord`).
**Acceptance gate met:** on seed 1 (20k ticks), the re-analyzed `timeseries.csv`
is byte-identical, and a fresh run reproduces identical timeseries, pillars, and
`state_hash` (`56b0d6eee6418f9e`). 31 workspace tests pass; clippy `-D warnings`
clean.

### `sim-cli` v2 internals

```rust
struct App {
    sim: Option<Simulation>,
    recorder: Option<Recorder>,   // Some only while `record on`
    format: Format,               // Text | Json (global default)
    report_every: u64,            // default 10_000 (eval parity)
    session: SessionFlags,        // scaled, champions, (future) diverged
    load_args: LoadArgs,          // for `reset`
}

struct Recorder {
    ledger: sim_metrics::Ledger,           // per-organism lifetime accumulation
    ticks: Vec<sim_metrics::TickSummaryRow>,        // per-tick descendant pop
    lifetimes: Vec<sim_metrics::OrganismLifetimeRow>, // deaths + end-of-run survivors
    started_turn: u64,                     // for "partial"-window labeling
}
```

- **Stepping is built on `tick()`** (not `advance_n`, which discards the
  `TickDelta`) so the recorder can see `delta.removed_positions` (deaths),
  `reproduction_events`, etc. When `recorder` is `None`, the loop just calls
  `tick()` and drops the delta → max scrub speed. When `Some`, each delta + the
  current `organisms()` + `action_records()` is fed to the recorder.
- **`Simulation` is built with `features = ["instrumentation"]`** (already the
  case) so `action_records()`, `ActionRecord.food_visible/utilization`, and
  `OrganismInstrumentationState.inter_ema` are available.
- **Read-only invariant**: the CLI never mutates `Simulation` except via
  `tick()`/`reset()`. Recording draws no RNG → determinism preserved (a CLI run
  at seed N reproduces the eval trajectory exactly).

## Command Surface

Global: `--json` on any command (or `format json` to flip the default) emits
minified JSON with stable keys. Lines starting `#` are comments; blank lines
ignored; `quit`/`exit`/EOF exits; errors print `error: …` and continue the REPL.

### Session & control

| command | effect |
|---|---|
| `load [--config P] [--seed N] [--threads K] [--report-every R] [--scale W,POP] [--champions P]` | Build `Simulation::new`. Defaults: config `sim-evaluation/config.toml`, seed 0, `report-every` 10_000. `--threads`→`config.intent_parallel_threads`. `--scale` overrides `world_width,num_organisms` for fast iteration (marks session **scaled** → not eval-canonical). `--champions` loads a pool (marks session non-canonical). |
| `reset [--seed N]` | Rebuild from the loaded config (new seed optional). Clears recorder. This is the "go back to start" / next-sweep-cell primitive. |
| `config [get KEY]` | Dump effective `WorldConfig` knobs (read-only in v2): world size, food ecology thresholds, metabolism/move costs, mutation modifier, `meta_mutation_enabled`, `runtime_plasticity_enabled`, etc. |
| `seed-genome` | Dump `SeedGenomeConfig` (the founder blueprint). |
| `record on\|off\|status` | Toggle the `Recorder`. `on` starts accumulating from the current turn (earlier-born organisms' lifetime counters are partial — labeled). `status` shows recorded span + whether full-window pillars are available. |
| `format text\|json` | Set global default output format. |
| `help` / `quit` | — |

### Time / stepping

| command | effect |
|---|---|
| `step [N]` | Advance N ticks (default 1). Fast path when not recording. |
| `run +DT` / `run-to T` / `goto T` | Advance to absolute/relative turn. `goto`==`run-to`. Error if target < current (no backwards → suggests `reset`). |
| `watch T [--every E] [--cols LIST]` | Advance to T, emitting one compact metrics row every E ticks (default E=`report_every`). One-shot population/evolution graph. Derived columns require `record on`; the light summary (pop/desc/food/max_gen/action-mix) is always available. |
| `turn` | Print current turn. |
| `bench [N]` | Time N ticks (default 100k), report ticks/sec and ns/tick (with/without recording). |

### Aggregate dashboards

| command | effect | needs record |
|---|---|---|
| `state` | The cockpit (one screen): turn; pop total/descendant/founder; generation max/p50/mean; energy·health·age stats (min/p50/mean/p90/max); food (plants/corpses/energy/coverage); last-turn ecology (consum/pred/repro/starv/age-death); **pillars+decomposition line if recording**. | partial |
| `pillars [--window W]` | The four `sim-metrics` pillars **with sub-signals side by side** over the last W intervals (default = eval's last-10% window): foraging←plant_consumption_rate; predation←prey_consumption_rate; intelligence←action_effectiveness + mi_sa; learning←learning_slope. Identical to the eval card. | yes |
| `actions [--window W] [--origin desc\|all]` | Population action histogram + live-derived extras the pillars don't score: idle fraction, failed-action rate, attack attempt/success, mean `ActionRecord.utilization`. Computed directly from `action_records()`, clearly separated from pillar signals. | yes |
| `eco [--last K]` | Ecological dynamics: population & food trajectory sparklines, birth/death rates, **deaths by cause** (starve/age/prey split), predation & consumption rates, carrying-capacity estimate. | trajectories yes; point-in-time no |
| `lineage` | Generation distribution (max/p50/mean + histogram); species (founder-lineage) composition — top lineages by population share; distinct-lineage count (diversity); mean parent-age-at-reproduction (generation time, derived live from `reproduction_events` while recording). | gen dist live; gen-time yes |
| `genome [--gene G] [--group-by species\|generation] [--drift]` | Population genome distribution per gene (min/p50/mean/max): topology (num_neurons, synapses, vision_distance), lifecycle, plasticity, and the 16 mutation-rate genes. `--drift` shows change since last call / over recorded window — *what evolution is selecting for*. Flags hot/cold mutation operators. | live (drift uses ring) |
| `food` | Food summary: plants/corpses, total energy, grid coverage %. | no |
| `timeseries [--cols LIST] [--last K]` | Dump recorded ring-buffer columns as sparklines + tail values (population graphs, pillar trajectories). | yes |

### Per-organism / cohort

| command | effect |
|---|---|
| `top FIELD [N] [--gen >=G] [--species S] [--origin desc\|founder\|all]` | Top-N by `energy\|health\|age\|generation\|consumptions\|prey\|plant\|reproductions\|neurons\|synapses\|util\|vision\|hebb_eta`, with cohort filters. |
| `hist FIELD [--bins B]` | Text/JSON histogram of any scalar field above. |
| `find EXPR [--fields LIST] [--limit N]` | Filter organisms by a predicate over scalar + genome fields (e.g. `energy>500 and generation>=3 and prey_consumptions>0`), print selected fields. Token-efficient targeted query. |
| `inspect ID` | Compact one-organism dump: pos/facing, energy/health/age/gen/species/gestating, consumption/repro counts, last action, genome summary, action logits, this-tick instrumentation. |
| `brain ID [--view summary\|synapses\|activations\|dot]` | Neural inspection. **summary**: neuron counts per layer, synapse_count, top-K synapses by \|weight\| (pre→post w/elig), inter time-constant range, plasticity genes + effective learning rate (juvenile-scaled), utilization. **synapses**: full sorted edge list (weight/eligibility/pending). **activations**: current sensory by receptor name, inter, action logits. **dot**: graphviz adjacency. |
| `decide ID` | Explain *this tick's* decision: sensory inputs by receptor (vision L/F/R × R/G/B/Shape, contact-ahead, energy, health, energy-delta, last-action flags), inter-activation summary, action logits → softmax **probabilities** (with temperature + idle bias), `food_visible` rays, and the selected action. |

### Sweep primitive (deferred runner)

No built-in runner in v2. The agent drives sweeps by scripting stdin or
launching processes, e.g.:

```bash
cargo run -p sim-cli --release <<'EOF'
load --seed 0 --report-every 10000
record on
run-to 500000
pillars --json
reset --seed 1
record on
run-to 500000
pillars --json
EOF
```

`reset` is the per-cell primitive; `--json` makes results machine-collectable. A
first-class `sweep` command is a v3 candidate.

## Output Format (illustrative)

`state` (text default):
```
turn=450000  pop=4180 (desc=3902 founder=278)  gen: max=312 p50=180 mean=176
energy:  min=0.4 p50=612 mean=590 p90=980 max=1840
health:  min=0.0 p50=1.00 mean=0.97 p90=1.00 max=1.00
age:     min=0 p50=210 mean=243 p90=540 max=1000
food: plants=480 corpses=32 energy=51200 coverage=8.19%
last-turn: consum=388 pred=21 repro=64 starv=51 age=12
pillars: forage .38←plant .076 | pred .09←prey .0047 | intel .69←eff .59 mi .12 | learn .00←slope -7e-4
```

`pillars` (side-by-side decomposition):
```
window: intervals 46–50 (ticks 450001–500000), 5 intervals, FULL
foraging      0.384   plant_consumption_rate 0.0768  (saturation 0.20)
predation     0.094   prey_consumption_rate  0.0047  (saturation 0.05)
intelligence  0.690   action_effectiveness 0.589  mi_sa 0.123b/0.16
learning      0.000   learning_slope -7.5e-4  (saturation 0.001; negative→0)
```

`timeseries --cols pop,food,forage --last 50`:
```
pop    ▁▂▃▅▆▇▇▆▅▅▆▇█▇▆ … 4180
food   █▇▅▃▂▁▁▂▃▄▅▅▄▃▃ … 512
intel  ▁▁▂▃▄▅▆▆▇▇▇▆▆▇▇ … 0.69
```

`decide 4213`:
```
sensory: vision L[r0 g.2 b0 s.1] F[r0 g.8 b0 s.4] R[r0 g0 b0 s0]
         contact=0 energy=.71 health=1.0 dEnergy=+.03 lastFwd=1 lastEat=0
inter(12): mean=.21 |max|=.88  active=8/12
logits: Fwd 2.11 Eat 0.34 TurnL -0.2 TurnR 0.1 Atk -1.4 Repro -2.0  (T=… idle_bias=-0.01)
softmax: Fwd .71 Eat .11 ... Idle .04   →  selected=Forward  food_visible=[F]
```

JSON mode mirrors each with minified, stably-keyed objects.

## Data Model (reachable surface)

All from a held `Simulation` (instrumentation on). Key sources verified during
review (file:line):

- **World/aggregate**: `Simulation::{turn, config, metrics, organisms, foods,
  action_records, snapshot}` (`sim-core/src/lib.rs`); `MetricsSnapshot`
  (last-turn consum/pred/repro/starv/age-death, synapse_ops, totals)
  (`sim-types/src/lib.rs:713`).
- **Per-tick events**: `TickDelta { moves, facing_updates, removed_positions,
  spawned, reproduction_events, food_spawned, metrics }`
  (`sim-types/src/lib.rs:774`) — deaths via `removed_positions`
  (`EntityId::Organism`), births via `reproduction_events`/`spawned`.
- **Per-organism**: `OrganismState` (id, species_id, q/r, generation, age_turns,
  facing, energy, health/max_health, damage_taken_last_turn, is_gestating,
  consumptions/plant/prey/reproductions counts, last_action_taken, brain, genome)
  (`sim-types/src/lib.rs:605`).
- **Brain**: `BrainState { sensory[18], inter[N], action[6], synapse_count,
  *_mean_activation }`; `SynapseEdge { weight, eligibility, pending_coactivation }`
  (`sim-types/src/lib.rs:580,504`). Receptor taxonomy: 3 vision rays
  (`VISION_RAY_OFFSETS=[-1,0,1]`) × {R,G,B,Shape} + {ContactAhead, Energy,
  Health, EnergyDelta, LastActionForward, LastActionEat}.
- **Genome**: `OrganismGenome { topology, lifecycle, plasticity, mutation_rates
  (16 genes), brain (inter_biases, inter_log_time_constants, action_biases,
  edges) }` (`sim-types/src/lib.rs:449`).
- **Instrumentation**: `ActionRecord { selected_action, action_failed,
  food_visible[3], age_turns, utilization, consumptions_count }`
  (`sim-types/src/lib.rs:83`); `OrganismInstrumentationState.inter_ema`.
- **Lineage semantics**: `generation` = hops from founder (0 for
  founders/injections, parent+1 for offspring). `species_id` is a *founder-lineage
  marker* (founder's own id, inherited by descendants) — **not** a speciation
  cluster; aggregate "lineage depth" = generation distribution, "diversity" =
  count of distinct surviving `species_id`.

## Edge Cases & Error Handling

- **`record on` mid-run**: lifetime counters for organisms born before the start
  turn are partial; pillars over a window fully inside the recorded span are
  exact, otherwise labeled `PARTIAL`. `record status` reports the recorded span.
- **`goto T` with T < turn**: error with hint to `reset` (no backwards).
- **`inspect`/`brain`/`decide` on dead/unknown id**: `error: no live organism …`.
- **`decide` with no action record this tick** (e.g. just after `load` before any
  `tick`): explain that a step is required.
- **Empty population**: dashboards degrade gracefully (`(no organisms)`), pillars
  report `N/A`.
- **Non-canonical session** (`--scale`, `--champions`, future interventions):
  `state`/`pillars` print a `[non-canonical: scaled]` tag so the agent never
  mistakes the numbers for eval-comparable.
- **Pillar denominator zero** (e.g. no food-ahead deaths in an interval):
  interval skipped, surfaced explicitly (matches eval semantics).
- **Determinism**: recording/instrumentation must not draw RNG; a regression
  check asserts CLI seed-N trajectory == eval seed-N.

## Concerns & Constraints

- **Performance**: release build mandatory. `record off` is the max-speed path
  (no ledger, delta dropped). `--threads` exposes `intent_parallel_threads`.
  `--scale` enables fast iteration at the cost of eval parity. Recommended
  pattern for measuring a late window: scrub fast with `record off`, then
  `record on` and run only the measurement window. Recorder uses parallel-vec /
  reused-buffer accumulation (no per-tick HashMaps), consistent with the engine's
  hot-path allocation norms.
- **Memory**: the timeseries ring buffer is bounded (capacity ≈ total_ticks /
  report_every plus an optional fine `--every` sampling); the lifetime ledger
  holds one compact row per organism over the recorded span (same footprint as
  eval's in-memory accumulation).
- **Maintenance**: pillar/interval math lives only in `sim-metrics`; the CLI
  must use `sim_types::ActionType::{index, ALL}` and shared `ACTION_COUNT`
  constants rather than hand-rolled 7-element arrays / local `action_index`
  (a current drift source to delete).

## Milestones / Build Order

1. **Extract `sim-metrics`** ✅ *done* — moved schema rows + ledger + ingest +
   intervals (slice-based) + pillars + stats out of `sim-evaluation`; eval
   consumes it via re-exports and the shared `register_founders`/`ingest_tick`.
   Golden gate met on seed 1 (timeseries + pillars + state_hash identical).
   Remaining: confirm on the full 8-seed suite before declaring closed.
2. **CLI recorder** ✅ *done* — `Recorder` over `sim_metrics::Ledger` + per-tick
   `TickSummaryRow`/`OrganismLifetimeRow` vecs; `record on/off/status`;
   `step`/`run-to` route through a recorder-aware `advance` (fast path preserved
   when off); `load --report-every`; mid-run `record on` back-registers the live
   population (partial windows labelled). `pillars` + a `state` pillars line read
   the four axes via `derive_interval_metrics`/`compute_pillar_scores`. Verified:
   live pillars at seed 1 / 20k == eval baseline (forage .384, pred .094,
   intel .690, learn 0). Added `Ledger::register_existing` / `ingest::register_existing`.
3. **Output layer** ✅ *done* — `output` module: `Format` text/json switch,
   `Stats` (min/p50/mean/p90/max), `sparkline`; `format` command + per-command
   `--json`/`--text`; `state`/`pillars`/`food` json; stale hand-rolled `forage`
   + pillar constants deleted. `EcoSample` per-tick recorder stream added for
   trajectories.
4. **Dashboards** ✅ *done* (`sim-cli/src/dashboards.rs`) — `eco` (pop/food
   sparklines, deaths-by-cause incl. `other`, rates, carrying-capacity),
   `lineage` (generation dist + founder-lineage composition), `genome`
   (per-gene Stats, mutation hot/cold, `--gene`, `--drift`), `timeseries`
   (`--cols`/`--last` sparklines), `watch T --every E`. Text + json.
5. **Per-organism** ✅ *done* (`sim-cli/src/inspect_ext.rs`) — `find EXPR`
   (predicate, `--fields`/`--limit`; and/or left-to-right), `brain ID --view
   summary|synapses|activations|dot`, `decide ID` (sensory→logits→softmax probs,
   exact reproduction of `evaluation.rs`). Text + json. (`top`/`hist`/`inspect`
   retained from v1; expanded fields/filters deferred.)
6. **Perf & ergonomics** ✅ *done* — `load --report-every/--threads/--scale`
   (`--scale W,POP` marks the session `[scaled: non-canonical]`, surfaced in
   `state`/`pillars`/json); `bench [N]` reports ticks/sec + ns/tick (notes
   debug builds). `--threads` maps to `config.intent_parallel_threads`.
7. **Determinism/parity check** ✅ live `pillars` match eval `summary.json` at
   seed 1 (20k). Full 8-seed / 500k golden run still optional.

All of M1–M5 reviewed by four parallel correctness/quality agents: no
blockers/majors; the `sim-metrics` extraction is byte-identical and `decide`'s
softmax exactly reproduces the engine. Minor review fixes applied (`find`
`==`/`!=` tolerance, `decide` post-tick note, `eco` deaths-by-cause `other`
bucket, `timeseries --last 0` guard, help precedence note).

## Open Questions (deferred)

- **v3 interventions**: exact verb set (`inject`, `kill`, `lesion`, `set
  config.…`, `spawn-food`) and how a `[DIVERGED]` session tag propagates through
  every metric readout.
- **First-class `sweep`** command vs. keeping orchestration agent-side.
- **`learning ID`**: a windowed view of eligibility/weight drift to watch
  plasticity happen within one lifetime — valuable, possibly fold into `brain`.
- **`find` grammar scope**: how rich the predicate language needs to be (scalar
  comparisons + boolean ops are the v2 floor).
