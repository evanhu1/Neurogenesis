# NeuroGenesis

NeuroGenesis is a deterministic artificial-life research substrate. A hex-grid
world evaluates organisms driven by evolvable neural networks; a separate
generational NEAT loop mutates, recombines, speciates, and selects their genomes.
Given a configuration and seed, every world and evolutionary run is reproducible.

The project currently studies whether competitive neuroevolution can keep
discovering stronger, behaviorally distinct strategies rather than merely
optimizing one fixed behavior.

## What runs where

```
config/       canonical world.toml and seed_genome.toml
types/        shared wire/domain types
brain/        genomes, expression, recurrent evaluation, plasticity
world-sim/    deterministic world, sensing, metabolism, tick pipeline
evolution/    NEAT population management and competitive evaluation
metrics/      raw observation facts and derived behavioral metrics
views/        shared inspection and output rendering
cli/          headless research interface
sim-server/ + web-client/   optional interactive viewer
research/     tracked experiment record and proposed work
```

`config/world.toml` and `config/seed_genome.toml` are the only baseline
configuration files. Generated worlds and experiment outputs belong under
`artifacts/`; the human-readable research record belongs under `research/`.

## Model

An organism observes six egocentric proximity rays, its own normalized energy,
and the predation energy gained or lost on the previous tick. Its controller is a
recurrent neural network: current-tick edges form a feed-forward DAG, while
evolvable previous-tick hidden edges carry explicit temporal state.
The optional leaky-neuron and Hebbian-plasticity flags add explicitly configured
within-lifetime state and learning.

The world uses a deliberately small competitive energy economy:

- Every organism loses one energy per tick and dies at zero.
- Every attack action pays `attack_attempt_cost`, including misses. A successful
  adjacent attack then conserves up to `attack_energy_transfer` from victim to
  attacker.
- No external energy enters an episode. Every founder starts with the same
  configured energy.

There is no reproduction inside a world. A finite simulation episode is an
evaluator. The `evolution` crate owns all genetic change between generations.

Successful attacks transfer existing energy; passive metabolism and attack
attempts dissipate it. There are no plants, food tiles, or Eat action in either
the evaluator or the ordinary simulator.

## Research workflow

Build the CLI and preflight the exact evaluation budget before starting a run:

```bash
cargo build -p cli --release

./target/release/cli plan \
  --population 48 --generations 40 --horizon 500 \
  --world-seeds 11,29,47,61 --founders 96
```

NEAT is the default mode, so the same options without `plan` execute the run:

```bash
cargo run -p cli --release -- \
  --seed 17 --population 48 --generations 40 --horizon 500 \
  --world-seeds 11,29,47,61 --founders 96 \
  --out-dir artifacts/research/runs
```

The result JSON contains each generation's contextual winner, population
checkpoints, species statistics, behavioral diagnostics, and the complete
frozen world/NEAT contract. Training scores rank contemporaries only; they are
not a monotone learning curve and are never compared across generations.
`cli crossplay RESULT.json.zst` is the sole longitudinal competence assay: it
re-evaluates frozen generation winners against one another under a common
pairwise contract.

The ordinary simulator remains available as an explicit special mode:

```bash
./target/release/cli world new --seed 7 --out artifacts/worlds/example.bin
./target/release/cli world run-to 500 --in artifacts/worlds/example.bin
./target/release/cli world pillars --in artifacts/worlds/example.bin
```

Read [docs/cli.md](docs/cli.md) before using the CLI: it explains the
world-as-file model, sidecar metrics, commands, and their semantics.

## Development

```bash
cargo check --workspace
cargo test --workspace
make fmt
make lint
cd web-client && npm run typecheck && npm run build
```

The canonical tick order is in
`world-sim/src/turn/mod.rs::Simulation::tick`. Preserve determinism whenever
changing that code: tie-breaking and action/predation sampling must remain
independent of thread scheduling.

For the interactive viewer:

```bash
cargo run -p sim-server
cd web-client && npm run dev
```

## Research record

[research/README.md](research/README.md) explains the tracked experiment atlas,
how results are archived, and how new proposals are registered. The standing
research question and current evidence are in [research/BRIEF.md](research/BRIEF.md).
