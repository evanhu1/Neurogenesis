<div align="center">

# Neurogenesis

### Watch brains evolve from nothing

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/evanhu1/Neurogenesis)

This is a neuroevolution artificial-life simulation built in Rust. Digital
creatures spawn into a world carrying a compact **generative genome** — a small
CPPN program that *grows* a brain. Then ecology (energy, food, predators, and
death) and evolution (mutation, sexual crossover, and selection) "train"
intelligent brains over thousands of generations. No fitness function, no
generations counter, no target — just survival and reproduction.

![A live Neurogenesis run: each organism is driven by its own neural network. The right panel shows the live brain of a selected organism.](docs/demo.gif)

</div>

## How brains evolve

The genome is **not** a list of synapses. It is an indirectly-encoding
**CPPN** (Compositional Pattern Producing Network, the HyperNEAT lineage) that is
queried across a spatial substrate to *develop* the phenotype: which senses and
actuators grow, where hidden neurons sit, how they wire, and how each synapse
learns. Over generations, NEAT-style structural mutations complexify the CPPN,
crossover recombines two parents, within-life Hebbian plasticity tunes the
developed weights, and selection decides what was worth the metabolic cost.

Because the encoding is generative, there are **no blind spots**: the sensory /
actuator interface, brain size, morphology, plasticity distribution, and the
mutation rates themselves are all heritable and all evolvable.

|                              The world up close                              |                                      A brain, mid-life                                      |
| :--------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Close-up of the hex world: organisms, plants, and terrain](docs/world.png) | ![Organism inspector showing live stats and the developed neural network](docs/inspector.png) |

## Why

The brain is the only existence proof of general intelligence, and it was
produced by evolution. Evolution _in silico_ rewrites nature's constraints:
generations take seconds, populations are observable down to the synapse, and any
run can be replayed exactly from its seed. Can a well-designed evolutionary search
over brain-like systems be a path to AGI? This project is an attempt to find out.

## Core ideas

**The substrate is environment-agnostic.** The genome, the developmental map, the
brain runtime, the operators, and the population driver live in `sim-substrate`
and know nothing about any world. A world is an `Environment` implementation that
supplies *physics only*. The hex ecology (`sim-hexworld`) is the first
environment; a hex-free 1-D "Chemotaxis Ribbon" (`sim-toyenv`) runs on the
**unchanged** substrate as a decoupling proof.

**The genome (indirect / generative encoding).**

- A CPPN program plus a small header of direct scalars. `develop()` expands it
  into the phenotype via ES-HyperNEAT: interface selection (which sensors/
  actuators express), an adaptive quadtree that discovers hidden neurons, and a
  link-expression gate (LEO) that decides connectivity separately from weight.
- **Plasticity is painted per connection** by the CPPN (adaptive-HyperNEAT), not
  a single global knob.
- Node/connection identity is a 64-bit structural hash, so crossover aligns
  homologous genes deterministically under parallel, continuous-birth reproduction.

**The brain.**

- Lifetime learning is **unsupervised Hebbian plasticity** — no reward, no value
  head. A centered-covariance rule accumulates pre/post coactivation into a
  per-synapse eligibility trace, gently modulated by within-tick energy change,
  with an evolvable juvenile critical period and maturity-gated pruning. Learned
  weights are **discarded at reproduction** (non-Lamarckian).
- Leaky-integrator neurons (evolvable time constants), a Padé `fast_tanh`, and
  softmax action selection with an implicit Idle option.
- Thinking costs energy: metabolism scales with neuron count, vision range, and
  Kleiber mass^0.75 body scaling, so every neuron a lineage keeps must pay for
  itself.

**Evolution.**

- **Sexual, embodied reproduction**: mating is an in-world action — two adjacent
  organisms mate, producing a crossover-then-mutate offspring born nearby.
- NEAT-style structural mutation of the CPPN (add connection, add node by
  splitting an edge, perturb weights/biases/activations).
- Meta-mutation: per-operator mutation rates are themselves genes, evolved in
  logit space with a zero-absorbing floor.
- A **Quality-Diversity (MAP-Elites)** champion archive keeps the best genome per
  behavioral niche; coverage / QD-score is the open-ended progress signal that
  replaces a fitness curve.

**The ecology.**

- A toroidal hex grid with value-noise terrain walls, spikes, a hidden fertility
  map, and event-driven plant regrowth. One entity per cell.
- Energy enters only as plants and drains through metabolism; predation leaves a
  corpse worth eating, so death feeds the food web. A zero-sum social-color field
  couples neighbors by hue.
- **No population cap and no periodic injection** — thermodynamics regulates the
  population, and a world that goes extinct **stays extinct** (the run ends there).
- No species registry, no hand-written fitness target. Selection pressure comes
  from the ecology itself.

## Run it in 60 seconds

Prerequisites: Rust toolchain + Node.js.

```bash
# 1. Backend
cargo run -p sim-server

# 2. Frontend (new shell; npm install on first run)
cd web-client && npm install && npm run dev

# 3. Open http://127.0.0.1:5173
```

![The Neurogenesis lab: world view, population telemetry, simulation controls, and the organism inspector](docs/hero.png)

Things to try:

- **Play / Pause / Step** — the server auto-ticks in the background; pause to
  study a moment, or step one tick at a time.
- **Click an organism** — its live stats, CPPN genome summary, and the developed
  brain network (excitatory/inhibitory synapses).
- **Save champions** — the server keeps a Quality-Diversity champion archive
  keyed by behavior; saved genomes can seed future worlds via
  `--seed-genome-snapshot`.

## Prove it works, headlessly

Every environment ships a deterministic example that asserts a live, reproducing
population and byte-identical results across identical-seed runs:

```bash
cargo run -p sim-substrate --example headless --release   # substrate ring-life
cargo run -p sim-hexworld  --example headless --release   # hex proof-of-life
cargo run -p sim-hexworld  --example simsmoke --release   # save/load + extinction
cargo run -p sim-toyenv    --example headless --release   # decoupling proof
```

## Measuring whether intelligence is emerging

With no fitness function, progress is measured by **behavioral coverage**.
`sim-evaluation` runs multi-seed headless experiments, builds a Quality-Diversity
(MAP-Elites) archive over a behavior descriptor, and reports coverage / QD-score
alongside population and lineage depth per seed. A seed that goes extinct records
its extinction turn.

```bash
# Default 8-seed evaluation → JSON summary on stdout
cargo run -p sim-evaluation --release --

# Custom seeds / horizon / world
cargo run -p sim-evaluation --release -- --seeds 42,123,7 --ticks 5000 --width 32

# Write the summary to a file
cargo run -p sim-evaluation --release -- --out summary.json
```

## The research CLI

`sim-cli` is a stateless, world-as-file research tool: a world is an explicit
`world.bin`, each call runs one command, output is JSON. Full reference in
[`docs/sim-cli.md`](docs/sim-cli.md).

```bash
cargo build -p sim-cli --release
BIN=./target/release/sim-cli
$BIN new --seed 7 --out artifacts/w.bin
$BIN run-to 500 --in artifacts/w.bin      # stops early on extinction
$BIN state   --in artifacts/w.bin
$BIN lineage --in artifacts/w.bin
$BIN inspect <id> --in artifacts/w.bin    # brain / genome / decide / find too
```

## Development

```bash
cargo check --workspace   # fast compile check
cargo test --workspace    # run all tests
make fmt                  # format
make lint                 # clippy, warnings as errors
```

Workspace layout: `sim-substrate` (environment-agnostic genome / develop / brain
/ operators / driver / QD archive), `sim-hexworld` (hex ecology + the
serializable `HexSim` world), `sim-toyenv` (second environment), `sim-types`
(generic value types), `sim-server` (Axum REST backend), `web-client` (React +
Vite canvas UI), `sim-evaluation` (headless QD harness).

The canonical tick order lives in
`sim-substrate/src/driver.rs::PopulationDriver::tick`; treat it as the source of
truth for phase ordering. See [`docs/SYSTEMS.md`](docs/SYSTEMS.md) for a full
tour of the brain, evolution, and ecology mechanisms.

Server flags:

- `--seed N` / `--width W` / `--founders F` / `--port P`.
- `--champion-pool-path <path.json>` — Quality-Diversity champion archive
  (default `champion_pool.json`).
- `--seed-genome-snapshot <path.bin>` — seed every founder from one bincode
  `sim_substrate::Genome` (e.g. a saved champion).
