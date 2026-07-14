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
brain/        genomes, expression, feed-forward evaluation, plasticity
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

An organism observes six egocentric rays. Each ray supplies a proximity signal
and a signed energy-affordance signal; it also sees its own normalized energy
and the energy gained or lost on the previous tick. Its controller is a
feed-forward neural network by default: activations do not carry through time.
The optional leaky-neuron and Hebbian-plasticity flags add explicitly configured
within-lifetime state and learning.

The world uses a simple conserved-energy economy:

- Every organism loses one energy per tick and dies at zero.
- Eating a plant transfers the plant's full energy to the eater.
- A successful adjacent attack transfers up to `attack_energy_transfer` from
  victim to attacker.
- Food tiles regrow deterministically after their configured interval.

There is no reproduction inside a world. A finite simulation episode is an
evaluator. The `evolution` crate owns all genetic change between generations.

## Research workflow

Build the CLI, create a world, advance it, then inspect its raw behavior.

```bash
cargo build -p cli --release

./target/release/cli new --seed 7 --out artifacts/worlds/example.bin
./target/release/cli run-to 5000 --in artifacts/worlds/example.bin
./target/release/cli pillars --in artifacts/worlds/example.bin
./target/release/cli top energy 10 --in artifacts/worlds/example.bin
```

Run canonical generational NEAT through the same CLI:

```bash
cargo run -p cli --release -- neat \
  --population 24 --generations 50 --episode-horizons 5000 \
  --world-seeds 7,17,27,37 --out-dir artifacts/research/runs
```

The result JSON contains each generation's population evaluations, champion,
species statistics, behavioral diagnostics, and the complete frozen world/NEAT
contract. `neat analyze RESULT.json` and `neat crossplay RESULT.json` derive
trajectory and historical-competition diagnostics without silently changing the
evolutionary run.

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
