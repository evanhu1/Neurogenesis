# sim-cli — headless stateful simulation client (design)

## Goal

A headless command-driven client for the engine: hold one `Simulation` in
memory, scrub it forward by N turns or to turn T, and dump full state at any
point — so a human (or agent) can debug emergent behavior without the web
client. First use case: explain why eval **seed 7** scored `foraging = 0.000`.

## Why a REPL over flags

Statefulness is the whole point: building the world and scrubbing to turn 450k is
expensive, and we want to inspect repeatedly without redoing it. So `sim-cli`
reads **commands from stdin, one per line**, holding the `Simulation` across
commands. This is trivially agent-drivable from Bash:

```bash
cargo run -p sim-cli --release <<'EOF'
load --config sim-evaluation/config.toml --seed 7
run-to 450000
state
forage 50000
EOF
```

Single process, single build of the world, many observations. Lines starting
with `#` are comments; blank lines ignored; `quit`/EOF exits.

## Crate

New workspace member `sim-cli` (binary). Dependencies:
- `sim-config` — `load_world_config_from_path` (auto-loads sibling
  `seed_genome.toml`), so config matches the eval byte-for-byte.
- `sim-core` **with `features = ["instrumentation"]`** — `Simulation`, plus
  `action_records()` (per-tick `ActionRecord { selected_action, food_visible:
  [bool;3], age_turns, ... }`), which is exactly what the eval uses to derive
  `p_fwd_food`.
- `sim-types` with `instrumentation`.
- `anyhow`. (Hand-rolled line parser; no clap needed for a line REPL, but clap is
  acceptable for tokenized arg parsing per command.)

No dependency on `sim-evaluation` (binary-only, no lib). We re-derive the
foraging metric locally from instrumentation.

## Reproducing the eval exactly

The eval builds each seed as `Simulation::new(load_world_config_from_path(
"sim-evaluation/config.toml"), seed)` (no champion pool; fresh founders), then
runs 500_000 ticks. Scoring uses the **last 10%** of the timeseries (ticks
450k–500k), **descendant-only** (organisms born in-run, i.e. `generation > 0`;
founders/injections excluded). `load --seed 7` defaults `--config` to
`sim-evaluation/config.toml`, giving an identical world.

## The foraging metric (verified against eval source)

**Exact eval pipeline** (`sim-evaluation`): per organism, lifetime counters
`food_ahead_ticks` / `fwd_when_food_ahead` accumulate every tick; on death (and
at end-of-run) the organism's lifetime row is bucketed by **`death_tick`** into
`report_every`=10k-tick intervals; each interval computes its own ratio
`Σfwd/Σfood_ahead` over **descendant** deaths in it; the pillar is the
**unweighted mean of per-interval ratios over the last 5 intervals** (last 10% of
50), then `clamp01((mean − p_base)/(0.55 − p_base))` with
`p_base = 1/ACTION_COUNT = 1/7 ≈ 0.143` (`ACTION_COUNT = ActionType::ALL.len()+1`,
i.e. 6 actions + Idle). Verified: `ledger.rs:249-256,321-325`,
`analysis/intervals.rs:75-85,259-263,410-412`, `analysis/pillars.rs:12-44`,
`dataset/schema.rs:11`.

Field rules (verified):
- "food ahead" = food anywhere along the **center ray** within `vision_distance`
  = `food_visible[offset 0]`. `VISION_RAY_OFFSETS = [-1,0,1]` so offset 0 = index
  1; the eval looks it up via `food_visible_at_offset(0)` (`ledger.rs:321`,
  `sim-types/src/lib.rs:96,246`). Not gated on any action.
- "moved forward" = `selected_action == ActionType::Forward` (`ledger.rs:324`).
- descendant: eval classifies by `parent_id.is_some()` (`ledger.rs:249-256`).
  **`generation > 0` is an exactly-equivalent proxy** — founders and periodic
  injections both spawn at `generation 0` (`spawn/organisms.rs:52,151`), only
  reproduced organisms reach ≥1. Injections are *enabled*
  (`sim-evaluation/config.toml`), but gen>0 excludes them anyway.

**`sim-cli` provides two readouts, clearly labeled:**
1. **Live behavioral probe** (primary diagnostic) — over a window, aggregate every
   descendant organism-tick from `action_records()` (index-aligned to
   `organisms()`; read `generation` from `organisms()`): `food_ahead` count, `fwd`
   count, `p_fwd_food = fwd/food_ahead`, the **full action histogram when food is
   ahead**, and the fraction of descendant-ticks with food ahead. This discriminates
   the hypotheses regardless of exact-pillar parity.
2. **Eval-style pillar (window-traced)** — finalize per-organism lifetime counters
   on death (via `tick()`'s `delta.removed_positions`), bucket by death-tick, take
   the mean-of-interval-ratios over the window's intervals, apply the pillar
   formula. Counters start at the trace `FROM`, so organisms born earlier are
   approximate; labeled as such. Confirms direction of the `0.000`.

`0.000` ⇒ over the last 5 death-tick intervals, descendant deaths either had no
food-ahead ticks (denominator 0 → interval skipped) or went forward at ≤ chance.

## Commands (v1)

| command | effect |
|---|---|
| `load [--config PATH] [--seed N]` | build `Simulation::new`; default config `sim-evaluation/config.toml`, seed 0. (No `--champions` in v1 — the file is a private versioned `ChampionPoolFile` envelope and the eval uses no pool, so a pool would *diverge* from the eval.) |
| `step [N]` | advance N ticks (default 1) via `tick()`, no foraging accumulation (fast scrub) |
| `run-to T` | advance via `tick()` until `turn == T` (no-op if already ≥ T) |
| `turn` | print current turn |
| `state` | turn; population; #descendants (gen>0); energy/health/age min·mean·median·max; generation histogram; food (plants/corpses, total energy); last-turn metrics from `metrics()` (consumptions/predations/reproductions/starvations/age_deaths) |
| `forage FROM TO` | scrub to FROM if needed, then step FROM→TO via `tick()` **accumulating** per-descendant-tick stats + death-bucketed lifetime rows; print the **live probe** (p_fwd_food, food-ahead fraction, action histogram when food ahead) **and** the eval-style window pillar |
| `food` | food summary + how many organisms sit on / are adjacent to food |
| `inspect ID` | one organism: pos, energy/health/age/gen/species, last_action, genome summary (num_neurons, synapse_count, vision_distance, plasticity genes), **action logits** (`brain.action[].logit`) + chosen action, and `food_visible` (instrumentation) |
| `top FIELD [N]` | top-N organisms by `energy`/`age`/`generation`/`consumptions`/`reproductions` |
| `hist FIELD` | text histogram of a scalar field across the population |
| `help` / `quit` | — |

Stepping is built **directly on `tick()`** (not `advance_n`, which discards the
`TickDelta`), so `forage` can read `delta.removed_positions` to detect deaths.
Build/run with `-p sim-cli` to avoid feature-unification turning instrumentation
on for `sim-server` in a shared build.

Output is human-readable text (clearly labeled), append `--json` later if needed.
Errors print `error: ...` and continue the REPL (don't abort the session).

## Investigation plan for seed 7

1. `load --seed 7`
2. checkpoint `state` at 50k, 100k, 250k, 450k, 500k — watch population,
   descendant share, food density, action mix evolve.
3. `forage 50000` over 450k–500k to reproduce `p_fwd_food ≈ 0` and read the
   action distribution **when food is ahead**: distinguishes the hypotheses —
   (a) food never ahead (food-saturated world → denominator 0), (b) descendants
   sit and `Eat`/`Turn` instead of `Forward`, (c) a genuinely non-foraging but
   high-MI lineage. `inspect` a few `top energy` descendants to see their wiring.

## Open questions for review

1. **Ray index for "ahead"**: is `food_visible[1]` the straight-ahead ray? Verify
   against `sensing.rs` ray ordering and how the eval maps `food_visible` →
   `food_ahead_ticks`.
2. **Descendant definition**: is `generation > 0` exactly the eval's
   `OrganismOrigin::Descendant`, or do periodic injections also get `generation
   0` (they would, as fresh founders) — confirm injections are config-disabled in
   `sim-evaluation/config.toml` so it doesn't matter.
3. **Window semantics**: eval window is the last 10% of *reporting intervals*,
   not raw ticks — does aggregating raw ticks 450k–500k match closely enough for
   debugging? (We only need to reproduce the *direction* of the 0, not the exact
   float.)
4. **Cost/throughput**: 500k ticks single-threaded at world 250 / 5000 orgs —
   acceptable for an interactive debug session, or should `load` allow a smaller
   `world_width`/`num_organisms` override for faster iteration (at the cost of
   not reproducing seed 7 exactly)?
5. **Determinism**: confirm a `sim-cli` run with seed 7 reproduces the eval's
   trajectory (same `Simulation::new(config, 7)`, no extra RNG draws from
   instrumentation).
