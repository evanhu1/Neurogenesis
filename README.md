# NeuroGenesis

NeuroGenesis evolves recurrent neural networks with a task-agnostic
generational NEAT loop. Its current reference training task is deterministic
single-agent symbol copying.

The task is deliberately exact. A representative stream is:

```text
input:   a -> b -> c -> end
output:  a -> b -> c -> end
fitness: one point per exact symbol
```

The complete sensor and action alphabet is `a` through `h`, plus `end`. Each
input symbol activates one one-hot sensor, the brain advances once, and the
largest action logit emits one symbol from the same alphabet. Fitness is the
raw number of positions emitted correctly. There are no opponents, shared
arenas, pairwise matchups, relative scores, survival objectives, or crossplay.

The canonical benchmark uses 32 deterministic pseudorandom training streams
and 32 separately seeded holdout streams. Each stream contains 16 shuffled
body symbols followed by `end`, with all eight body symbols guaranteed
to appear. Selection uses only the 544 training positions; holdout accuracy is
reported but never affects breeding.

## Structure

- `types/`: the nine-symbol interface and shared genome/runtime types.
- `brain/`: genome expression, current-step DAG evaluation, previous-step
  recurrence, and optional Hebbian plasticity used by the separate simulator.
- `evolution/`: task-agnostic generational NEAT plus modular evaluation tasks.
  `tasks/symbol_copy.rs` contains the reference training task.
- `config/seed_genome.toml`: canonical founder-genome configuration.
- `cli/`: the headless NEAT research interface.
- `world-sim/`, `metrics/`, `views/`, `sim-server/`, `web-client/`: an optional
  deterministic visualization sandbox. It is not an evolutionary evaluator
  and does not contribute to fitness.

## Run the task

```bash
cargo build -p cli --release

./target/release/cli plan \
  --population 64 --generations 100

./target/release/cli \
  --seed 17 --population 64 --generations 100 \
  --out-dir artifacts/research/runs
```

Passing `--stream` replaces the canonical training corpus with explicit cases;
the holdout corpus remains unseen. `end` is appended when omitted, so
`--stream abc` and `--stream a,b,c,end` are equivalent. Recurrent brain state
is reset between streams and retained between symbols inside a stream.

The result artifact records every generation winner, periodic full population
checkpoints, the final population, the exact training and holdout corpora, the
canonical seed-genome contract, and the full NEAT configuration. A fixed
configuration and seed are deterministic.

Read [docs/cli.md](docs/cli.md) before using the CLI.
See [docs/evaluation-tasks.md](docs/evaluation-tasks.md) to add a new task
without changing the NEAT loop.

## Development

```bash
cargo check --workspace
cargo test --workspace
make fmt
make lint
cd web-client && npm run typecheck && npm run build
```

Generated run artifacts belong under `artifacts/`. Durable hypotheses,
methods, and conclusions belong under `research/`.
