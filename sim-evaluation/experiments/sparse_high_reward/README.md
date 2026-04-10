# Sparse High-Reward Food Experiment

Treatment config for testing whether a much sparser plant landscape with much higher per-plant energy pushes evolution toward better navigation/foraging or more predation.

Design:
- Control: [`sim-evaluation/config.toml`](/Users/evanhu/code/NeuroGenesis/sim-evaluation/config.toml)
- Treatment: `food_fertility_threshold = 0.6`, `food_fertility_jitter_strength = 4.0`, `food_energy = 400`
- Unchanged: regrowth timing, terrain, seed genome, seeds, tick budget

Rationale:
- `food_fertility_jitter_strength = 4.0` punches many more holes inside the same broad fertile zones, instead of shrinking those zones by raising the threshold.
- `food_energy = 400` offsets the reduced plant count so total standing plant energy stays close to baseline, isolating patchiness/reward size better than simply starving the world.
