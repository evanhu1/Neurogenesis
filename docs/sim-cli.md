# sim-cli — usage reference

`sim-cli` is the **agent-facing research cockpit** for the NeuroGenesis engine:
a headless, stateful stdin REPL that holds one `Simulation` in memory so a world
can be scrubbed forward and interrogated repeatedly without rebuilding. It is
optimized for signal-dense, token-efficient, machine-parseable output.

The in-REPL `help` command is the always-current source of truth; this file is
the fuller reference (vocabularies, semantics). Design rationale lives in
`SPEC.md`.

## Run

```bash
cargo run -p sim-cli --release            # --release strongly recommended
```

Reads commands from stdin, one per line. `#`-prefixed and blank lines are
ignored; `quit`/`exit`/EOF exits. Errors print `error: …` and continue. Drive it
from a heredoc:

```bash
cargo run -p sim-cli --release <<'EOF'
load --seed 7 --report-every 10000
record on
run-to 500000
pillars
state
EOF
```

## Core model

- **Stateful**: one `Simulation`, advanced only forward (`step`, `run-to`). To
  restart, `load` again (there is no backward scrub).
- **Recording**: metrics that need history (`pillars`, `eco` trajectories,
  `timeseries`, `lineage` generation-time) require `record on` first. The
  recorder is built on the shared `sim-metrics` crate, so live `pillars` are
  **byte-identical** to the offline `sim-evaluation` harness. Point-in-time
  queries (`state`, `food`, `inspect`, `top`, `hist`, `find`, `brain`, `decide`,
  `genome`, `lineage` distribution) work without recording.
  - `record on` at turn 0 reproduces the eval exactly; mid-run it back-registers
    the live population with partial lifetime counters and labels windows
    `[PARTIAL]`.
- **Output format**: compact text by default. `format json` flips the global
  default; any command also takes `--json` / `--text` to override per call.
- **Canonical vs scaled**: a default `load` reproduces the eval world. `--scale`
  (or `--champions`, future interventions) marks the session
  `[scaled: non-canonical]` in `state`/`pillars`/json so metrics aren't mistaken
  for eval-comparable.

## Commands

### Session & control
- `load [--config PATH] [--seed N] [--report-every R] [--threads K] [--scale W,POP]`
  — build `Simulation::new`. Defaults: config `sim-evaluation/config.toml`,
  seed 0, report-every 10000. `--threads` → `config.intent_parallel_threads`.
  `--scale W,POP` overrides `world_width,num_organisms` for fast iteration
  (non-canonical). Clears any recorder.
- `record on|off|status` — toggle the metric recorder.
- `format json|text` — set the default output format.
- `help`, `quit`/`exit`.

### Time
- `step [N]` — advance N ticks (default 1).
- `run-to T` — advance until turn == T (no-op if already ≥ T; no backward).
- `watch T [--every E]` — advance to T, emitting a compact metrics row every E
  ticks (default E = report_every). One-shot evolution graph.
- `turn` — print current turn.
- `bench [N]` — time N ticks (default 100000) through the current advance path
  (respects recording); reports ticks/sec + ns/tick. Use `--release` for real
  numbers.

### Aggregate readouts
- `pillars` — the four competence axes with sub-signals: **foraging**
  ←plant_consumption_rate, **predation** ←prey_consumption_rate,
  **intelligence** ←action_effectiveness + mi_sa, **learning** ←learning_slope.
  Needs recording. (Saturation anchors / formulas: `sim-metrics/src/pillars.rs`.)
- `state` — turn; population (total/descendant/founder); energy·health·age·
  generation five-number summaries; food; last-turn ecology; + a pillars line if
  recording.
- `eco` — population & food sparklines, birth/death rates, deaths-by-cause
  (starvation/age/predation/other), consumption/predation rates, carrying-
  capacity estimate. Trajectories need recording.
- `lineage` — generation distribution + founder-lineage (`species_id`)
  composition (top lineages by share, distinct-lineage diversity).
- `genome [--gene NAME] [--drift]` — population per-gene five-number summaries +
  mutation-rate hot/cold. Genes: `num_neurons num_synapses vision_distance
  age_of_maturity gestation_ticks max_organism_age hebb_eta_gain
  juvenile_eta_scale eligibility_retention max_weight_delta_per_tick
  synapse_prune_threshold` (plus the 16 mutation-rate genes shown in the
  mutation-rate section).
- `timeseries [--cols LIST] [--last K]` — recorded columns as sparklines. Needs
  recording. Columns: per-tick `population descendants food births deaths
  consumptions predations reproductions`; per-interval `action_effectiveness
  plant_consumption_rate prey_consumption_rate mi_sa learning_slope pop`.
- `food` — plant/corpse counts, energy, grid coverage.

### Per-organism
- `top FIELD [N]` — top-N by `energy|health|age|generation|consumptions|reproductions`.
- `hist FIELD` — histogram of `energy|health|age|generation`.
- `find EXPR [--fields LIST] [--limit N]` — filter by a predicate. Comparisons
  `field OP value` (OP ∈ `< <= > >= == !=`) joined by `and`/`or`, evaluated
  **strictly left-to-right (no precedence, no parentheses)**. Fields:
  `id energy health max_health age generation species consumptions plant prey
  reproductions neurons synapses vision hebb_eta gestating`. `--fields` selects
  output columns (same names; default id,energy,age,generation,consumptions,
  reproductions); `--limit` caps rows (default 20).
- `inspect ID` — one-organism dump (pos, energy/health/age/gen/species, counts,
  genome summary, action logits, this-tick instrumentation).
- `brain ID [--view summary|synapses|activations|dot]` — neural inspection.
  summary: layer counts, top synapses by |weight|, plasticity genes + effective
  learning rate, utilization. synapses: full edge list. activations: current
  sensory/inter/action. dot: graphviz adjacency.
- `decide ID` — explain the organism's current decision: sensory inputs → action
  logits → softmax probabilities (exact reproduction of the engine) → selected
  action. Note: logits/probs reflect the post-tick brain state, while
  `selected`/`food_visible` are from the decision tick just executed.

## Sweeps (agent-orchestrated)

No built-in sweep runner. Drive multiple seeds/configs by scripting stdin
(`load … / record on / run-to … / pillars --json / load --seed …`) or launching
multiple processes, and collect the `--json` output.

## Notes

- Determinism: same config + seed = identical results; recording is read-only and
  draws no RNG, so a `sim-cli` run reproduces the eval trajectory exactly.
- Architecture: metric computation is shared via the `sim-metrics` crate
  (`ledger`/`ingest`/`intervals`/`pillars`); the CLI's `Recorder` feeds it live.
