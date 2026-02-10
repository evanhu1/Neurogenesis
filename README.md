# NeuroGenesis

Neuromorphic brains grown from scratch with simulated evolution in Rust.

## Layout

- `sim-core`: deterministic simulation engine
- `sim-protocol`: shared protocol types
- `sim-server`: HTTP + WS simulation server
- `web-client`: HTML5 canvas renderer

## Quickstart

1. `cargo check --workspace`
2. `cargo test --workspace`
3. Start server: `cargo run -p sim-server`
4. In another shell: `cd web-client && npm install && npm run dev`
5. Open `http://127.0.0.1:5173`

## Performance

`cargo bench -p sim-core --bench turn_throughput`

## Config

A default config is available at `config/default.toml`.

## To Do:

- [ ] Cull useless neurons and synapses, neuronal pruning (also simulating
      exuberant synaptogenesis)
- [ ] Use Hebbian/SDTP to guide synaptogenesis (neurons that fire together wire
      together, neurons that fire out of sync lose their link). Synaptogenesis
      should not be random
- [ ] Implement temporal credit assignment
- [ ] Experiment with local gradient descent in the brain.
- [ ] Create multiple concentric evolution loops. Innermost is the organism
      loop. Evolve worlds, with a world DNA substrate that sets the "laws of
      physics" and is mutated. The fitness of a world is the max fitness
      achieved by life in that world.
