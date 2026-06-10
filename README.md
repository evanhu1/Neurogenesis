<div align="center">

# Neurogenesis

### Watch brains evolve from nothing.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/evanhu1/Neurogenesis)

2,000 digital creatures spawn into a world with the simplest possible brains:
**zero hidden neurons and ten synapses.** No training data. No reward function.
No gradient descent. Just energy, food, predators, and death.

Then evolution takes the wheel.

![A live Neurogenesis run: each triangle is an organism driven by its own neural network. The right panel shows the live brain of a selected organism.](docs/demo.gif)

*A live run. Every triangle is an organism steering itself with its own neural
network. The panel on the right is the selected organism's brain — sensory
neurons in orange, evolved inter neurons in blue, action neurons in green —
updating every tick until the moment it dies.*

</div>

## What you're looking at

| | |
| :--- | :--- |
| **Triangles** | Organisms. The color is a heritable gene; the point is the direction they're facing. Everything they do — turn, move, eat, attack, reproduce — is decided by their own neural circuit reading their own senses. |
| **Bright green dots** | Plants. The renewable energy source that powers everything. |
| **Glowing organisms** | Pregnant. Reproduction costs real energy and takes real time. |
| **The right panel** | A live MRI. Click any organism to watch its neurons fire, inspect its genome, and see its mutation rates — which are themselves evolvable genes. |

Every organism pays a metabolic tax proportional to the size of its brain and
the sharpness of its eyes. Thinking is not free. Bigger brains have to *earn*
their keep — exactly the constraint real nervous systems evolved under.

## Brains literally grow here

The seed genome has **0 inter neurons**. There is no hidden layer at the start
— sensors wire straight to actions. Over generations, NEAT-style structural
mutations add neurons and synapses, runtime plasticity tunes the weights during
each organism's lifetime, and selection decides what was worth the metabolic
cost.

After ~1,300 generations, champion genomes carry brains that grew to **9 inter
neurons and 25+ synapses** — circuitry that evolution invented, paid for, and
kept.

| The world up close | A brain, mid-life |
| :---: | :---: |
| ![Close-up of the hex world: organisms, plants, and terrain](docs/world.png) | ![Organism inspector showing live stats and the evolved neural network](docs/inspector.png) |

## Why

The brain is the only existence proof of general intelligence, and it was
produced by evolution. It took ~600 million years from the first neuron — but
nature had to solve problems we don't: physical embodiment, scarce energy,
generation times measured in years, and 2 billion years just to assemble the
molecular machinery of reproduction itself.

Evolution *in silico* rewrites those constraints. Generations take seconds.
Populations are observable down to the synapse. Every run is perfectly
reproducible. **Can a well-designed evolutionary search over brain-like systems
be a path to AGI?** This project is an attempt to find out.

## Run it in 60 seconds

Prerequisites: Rust toolchain + Node.js.

```bash
# 1. Backend
cargo run -p sim-server

# 2. Frontend (new shell; npm install on first run)
cd web-client && npm install && npm run dev

# 3. Open http://127.0.0.1:5173 and press Run
```

![The Neurogenesis lab: world view, population telemetry, simulation controls, and the organism inspector](docs/hero.png)

Things to try:

- **Fast mode** — rendering pauses and the engine runs flat out (hundreds of
  thousands of ticks in minutes). Pop back to live view to see what evolution
  built while you weren't looking.
- **Click an organism** — live brain activity, genome, and per-operator
  mutation rates.
- **Save champions** — the server keeps a persistent champion-genome pool and
  bootstraps new worlds from it, so progress compounds across sessions.

## The rules of life

- **The world** — a 250×250 toroidal hex grid: Perlin-noise terrain walls, a
  hidden fertility map, and event-driven plant regrowth. One entity per cell.
- **The body** — organisms sense through RGB vision rays with evolvable range,
  plus internal state like energy flux, and act by turning, moving, eating,
  attacking, or reproducing. Action selection is softmax sampling over the
  brain's output neurons.
- **The brain** — a three-layer circuit of evolvable topology. Inter neurons are leaky
  integrators with evolvable time constants. During life, dopamine derived from
  energy swings gates Hebbian plasticity: eat well and recent coactivations
  strengthen, starve and they weaken. Useless synapses get pruned.
- **Evolution** — asexual reproduction with structural mutation (add synapse,
  remove synapse, split an edge into a new neuron), scalar mutation, spatially
  biased wiring priors, and meta-mutation: the mutation rates themselves
  evolve. Periodic injections of fresh seed genomes keep diversity flowing.
- **Death matters** — corpses return 80% of stored energy, plants only 20%.
  Predation is real, and every kill leaves a corpse worth eating. Ecology, not
  a fitness function, decides who reproduces.

No species registry, no speciation bookkeeping, no hand-written fitness target.
Selection pressure is emergent.

## Built like a physics engine, not a screensaver

- **Deterministic to the bit.** Fixed config + seed ⇒ identical history, every
  time. Tie-breaks are ordered, and all sampling is a deterministic hash of
  `(seed, turn, organism IDs)`. Evolution experiments are reproducible science,
  not anecdotes.
- **Fast.** The Rust engine runs whole-world simulation at thousands of ticks
  per second on a laptop — a 30-second Fast-mode burst covered 230,000 ticks
  (~7,700 t/s) — with a CI perf budget guarding the hot path.
- **Workspace** — `sim-types` (shared domain types), `sim-config` (world +
  seed-genome TOML baselines), `sim-core` (the deterministic engine),
  `sim-server` (Axum HTTP + WebSocket), `web-client` (React + Tailwind + Vite
  canvas UI), `sim-evaluation` (headless science harness).

The canonical tick order lives in `sim-core/src/turn/mod.rs::Simulation::tick`
— treat it as the source of truth for phase ordering.

## Measuring whether intelligence is actually emerging

Vibes don't count. `sim-evaluation` runs multi-seed, hundreds-of-thousands-of-
ticks benchmarks where the sim emits raw facts to partitioned Parquet and every
metric is derived post-hoc: foraging skill, action-information metrics like
MI(S;A), competition stats, population dynamics.

![Evaluation report with pillar scores for foraging, intelligence, and competition](docs/evaluation.png)

```bash
# Default 8-seed evolution-loop benchmark → report.html / timeseries.csv / summary.json
cargo run -p sim-evaluation --release --

# Custom seeds
cargo run -p sim-evaluation --release -- --seed 42,123,7

# Random-action control baseline
cargo run -p sim-evaluation --release -- --baseline

# Quick smoke run
cargo run -p sim-evaluation --release -- --ticks 1000 --report-every 250

# Re-derive reports from a persisted dataset without re-running the sim
cargo run -p sim-evaluation --release -- analyze latest
cargo run -p sim-evaluation --release -- analyze 20260416T002137Z   # timestamp prefix
cargo run -p sim-evaluation --release -- analyze <path>             # run root or seed dir
```

Artifacts land under `artifacts/evaluation/...`.

## Development

```bash
cargo check --workspace   # fast compile check
cargo test --workspace    # run all tests
make fmt                  # format
make lint                 # clippy, warnings as errors
```

Performance benchmarking:

```bash
cargo bench -p sim-core --bench turn_throughput

# Perf regression guard (ignored by default)
cargo test -p sim-core --release performance_regression -- --ignored --nocapture

# Optional CI budget override
SIM_CORE_TICK_BUDGET_NS_PER_TURN=130000 make perf-test
```

Server flags:

- `--champion-pool-path <path>` — override the default `champion_pool.json`
  used for champion persistence.
- `--seed-genome-snapshot <path.bin>` — boot every organism from a single
  evaluation-snapshot genome (`.../genomes/tNNNNNN.bin`); champion saves no-op
  for the session.

Config baselines live in `sim-config/config.toml` and
`sim-config/seed_genome.toml` (the evaluation copies in `sim-evaluation/` are
kept in sync with them).
