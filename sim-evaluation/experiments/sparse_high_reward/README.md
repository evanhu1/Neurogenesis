# Sparse High-Reward Food Experiment

Treatment config for testing whether a much sparser plant landscape with much higher per-plant energy pushes evolution toward better navigation/foraging or more predation.

Design:
- Control: [`sim-evaluation/config.toml`](/Users/evanhu/code/NeuroGenesis/sim-evaluation/config.toml)
- Treatment: `food_tile_fraction = 0.4`, `food_energy = 400`
- Unchanged: regrowth timing, terrain, seed genome, seeds, tick budget

Rationale:
- Food tiles are now an uncorrelated deterministic random subset, so this legacy
  treatment varies reward size only. Use `food_tile_fraction` directly for
  future density experiments.
