# autoresearch

Autonomous experimentation loop for NeuroGenesis — a neuroevolution simulation
where organisms with neural-network brains evolve on a hex grid to forage,
reproduce, and survive.

## Setup

To set up a new experiment:

1. **Set a run tag**: Create a tag based on today's date (e.g. `mar26`). The
   branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current
   main.
3. **Read AGENTS.md**: Read `AGENTS.md` for repository context and architecture.
4. **Verify build**: Run `cargo check --workspace` and `cargo test --workspace`
   to ensure a clean starting state.
5. **Create the archive directory**: Put all autoresearch scratch outputs under
   `.autoresearch_archive/`. Do not scatter run logs, TSVs, or experiment
   artifacts in the repo root.
6. **Initialize the results ledger**: Create
   `.autoresearch_archive/results.tsv` with just the header row. The baseline
   will be recorded after the first run.

Once done, kick off the experimentation.

## Experimentation

Each experiment runs the evaluation harness for a fixed tick budget (default
50,000 ticks). You launch it as:

```
cargo run -p sim-evaluation --release -- --seed 42 --ticks 50000 --out .autoresearch_archive/artifacts/evaluation/exp_name > .autoresearch_archive/run.log 2>&1
```

Always use `--release` mode. Always redirect output — do NOT use tee or let
output flood your context. All logs and evaluation outputs must live under
`.autoresearch_archive/`.

**What you CAN modify:**

- `sim-core/src/brain.rs` — neural network architecture, activation functions,
  neuron dynamics, sensory processing, action selection.
- `sim-core/src/plasticity.rs` — runtime learning rules, eligibility traces,
  reward signals, synapse pruning.
- `sim-core/src/genome.rs` — mutation operators, genome structure, mutation
  rates logic.
- `sim-config/config.toml` + `sim-config/seed_genome.toml` — world parameters,
  seed genome config, and mutation rates.
- `sim-evaluation/config.toml` + `sim-evaluation/seed_genome.toml` —
  evaluation config (keep in sync with sim-config when needed).

**What you CANNOT modify:**

- `sim-core/src/turn.rs` — turn pipeline, intent resolution, energy costs,
  consumption/predation mechanics.
- `sim-core/src/spawn.rs` — reproduction mechanics, offspring placement.
- `sim-core/src/grid.rs` — grid helpers (be careful with determinism).
- `sim-core/src/lib.rs` — simulation initialization and config wiring.
- `sim-types/src/lib.rs` — shared types (when needed to support sim-core
  changes).
- `sim-evaluation/src/main.rs` — the evaluation harness and aggregate scoring
  function are the ground truth.
- `sim-evaluation/src/metrics.rs` — the metric definitions (p_fwd_food, MI,
  entropy) are fixed.
- `sim-evaluation/src/ledger.rs` — lifetime tracking is fixed.
- `sim-evaluation/src/report.rs` — report generation is fixed.

**What you MUST preserve:**

- **Determinism**: Fixed config + seed = identical results. All tie-breaking
  uses organism ID ordering. Any change to turn logic must preserve determinism.
- **Compilation**: `cargo check --workspace` must pass.
  `cargo clippy --workspace --all-targets -- -D warnings` should be clean.

**The goal is simple: get the highest `aggregate_score` by improving the
open-ended evolution of cognition. We want to improve the evolutionary search
efficiency, the ability of brains to learn, or tune other aspects of the
artificial life / neuroevolution simulation, while avoiding trivial, degenerate
strategies.**

Higher score = organisms that forage better, respond more intelligently to
stimuli, and maintain diverse behavioral repertoires.

**Everything in the brain, organism, and genome is fair game**: The only
constraints are that the code compiles, determinism is preserved, and the
evaluation harness/metrics are untouched.

**Task-preservation constraint**: Do not improve `aggregate_score` by making the
world easier or by collapsing behaviors the agents are supposed to learn.
Changes that remove cognitive difficulty must be discarded even if they increase
score.

Examples of forbidden task-simplifying changes:

- making consumption, reproduction, or other explicit actions automatic
- removing distinctions between actions that organisms are meant to learn to
  choose between
- tuning ecology/config primarily to reduce decision pressure rather than
  improve evolution of cognition

**We don't want to make progress by converging to unintelligent behavior, such
as only moving forward**. Don't change how the world works to make the score
better, for example by making consumption of food or birth automatic.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Removing something and
getting equal or better results is a great outcome. When evaluating whether to
keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the
baseline — run the evaluation harness on unmodified code.

## Output format

The evaluation harness prints a final summary:

```
aggregate_score: 42.31
total_time_seconds: 187.432
```

You can extract key metrics from the log:

```
grep "^aggregate_score:" .autoresearch_archive/run.log
```

And from the generated `summary.json` for detailed breakdown:

```
cat .autoresearch_archive/artifacts/evaluation/*/summary.json | python3 -c "import sys,json; d=json.load(sys.stdin); s=d['aggregate_score']; print(f'score={s[\"score\"]:.2f} p={s.get(\"mean_p_fwd_food\",0):.4f} mi={s.get(\"mean_mi_sa\",0):.4f} h={s.get(\"mean_h_action\",0):.4f} pred={s.get(\"mean_predation_rate\",0):.6f}')"
```

## Logging results

When an experiment is done, log it to `.autoresearch_archive/results.tsv`
(tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	score	time_s	status	description
```

1. git commit hash (short, 7 chars)
2. aggregate_score achieved (e.g. 42.31) — use 0.00 for crashes
3. total_time_seconds, round to .1f (e.g. 187.4) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	score	time_s	status	description
a1b2c3d	42.31	187.4	keep	baseline
b2c3d4e	45.67	192.1	keep	double vision distance to 20
c3d4e5f	38.90	185.3	discard	remove sensory ray offsets
d4e5f6g	0.00	0.0	crash	add recurrent connections (type error)
```

## Multi-seed evaluation

The default validation command already runs the fixed benchmark seed suite:

```
cargo run -p sim-evaluation --release -- --ticks 50000 --out .autoresearch_archive/artifacts/evaluation/full_suite > .autoresearch_archive/run_full_suite.log 2>&1
```

A change that improves the mean but regresses the median or has a much worse
minimum seed is likely overfitting to initial conditions. Prefer changes that
hold up across the full suite.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar26`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Formulate a hypothesis — what change do you expect to improve adaptation?
3. Implement the change in the in-scope files.
4. git commit the change.
5. Run the experiment:
   `cargo run -p sim-evaluation --release -- --seed 42 --ticks 50000 --out .autoresearch_archive/artifacts/evaluation/exp_name > .autoresearch_archive/run.log 2>&1`
   (redirect everything — do NOT use tee or let output flood your context).
6. Read out the results:
   `grep "^aggregate_score:\|^aggregate_score_median:\|^aggregate_score_stddev:\|^aggregate_score_min:\|^aggregate_score_max:\|^total_time_seconds:" .autoresearch_archive/run.log`
7. If the grep output is empty, the run crashed. Run
   `tail -n 50 .autoresearch_archive/run.log` to read the error and attempt a
   fix. If you can't get things to work after more than a few attempts, give up
   on this idea.
8. Record the results in `.autoresearch_archive/results.tsv` (NOTE: do not
   commit it, leave it untracked by git).
9. If the aggregate result improved without introducing an obvious median/min
   regression, you "advance" the branch, keeping the git commit.
10. If aggregate_score is equal or worse, you `git reset --hard` back to where
    you started.

**Timeout**: Each experiment should take a few minutes. If a run exceeds 15
minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: if it's a typo or missing
import, fix and re-run. If the idea itself is fundamentally broken, log "crash",
revert, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the
human if you should continue. Do NOT ask "should I keep going?" or "is this a
good stopping point?". The human might be away and expects you to continue
working _indefinitely_ until manually stopped. You are autonomous. If you run
out of ideas, think harder — re-read the architecture docs, read research papers
related to artificial life or neuroevolution for novel ideas, look at the
scoring formula for overlooked components, try combining previous near-misses,
try more radical changes. The loop runs until the human interrupts you, period.
