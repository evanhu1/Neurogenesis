# NeuroGenesis

NeuroGenesis evolves recurrent, plastic neural controllers in finite task
ecologies. Symbolic environments are independent of both the brain and the
optimizer: they expose observations, actions, rewards, success events, trial
boundaries, and termination. A generic adapter executes any task with the
canonical brain, and a generic asexual search turns success events into a
finite pool of reproductive opportunities.

## Structure

- `task-library/`: brain- and optimizer-independent symbolic environments.
- `brain/`: genome encoding/expression, recurrent evaluation, and plasticity.
- `evolution/`: generic task adapter and asexual task-ecology search.
- `types/`: shared symbolic and genome/runtime types.
- `config/`: canonical world and founder-genome configuration.
- `cli/`: sole headless research interface.
- `world-sim/`, `metrics/`, `views/`, `sim-server/`, `web-client/`: the
  deterministic ecological simulator and visualization stack.

The included symbolic tasks are basic reaction, basic memory, basic next-token
prediction, basic continual learning, and renewable hidden resource. They all
load through the same task-ecology path.

## Run task ecology

```bash
cargo build -p cli --release

./target/release/cli ecology reaction plan \
  --population 64 --generations 100

./target/release/cli ecology memory \
  --seed 17 --population 256 --generations 500 \
  --out-dir artifacts/research/runs

./target/release/cli ecology next-token \
  --seed 101 --population 256 --generations 100 \
  --out-dir artifacts/research/runs

./target/release/cli ecology continual \
  --seed 101 --population 256 --generations 100 \
  --out-dir artifacts/research/runs

./target/release/cli ecology renewable \
  --seed 101 --population 256 --generations 250 \
  --out-dir artifacts/research/runs
```

Fixed configuration plus seed is deterministic. Runs stream generation,
accuracy, resource, topology, elapsed-time, and ETA events to stderr and write
one compressed result artifact under `artifacts/research/runs/`.

Read [docs/cli.md](docs/cli.md) before using the CLI. See
[docs/evaluation-tasks.md](docs/evaluation-tasks.md) for the task boundary.

## Development

```bash
cargo check --workspace
cargo test --workspace
make fmt
make lint
cd web-client && npm run typecheck && npm run build
```

Generated outputs belong under `artifacts/`. Durable hypotheses, methods, and
conclusions belong under `research/`.
