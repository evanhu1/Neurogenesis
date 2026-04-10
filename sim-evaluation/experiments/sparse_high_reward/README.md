# Sparse High-Reward Food Experiment

Treatment config for testing whether a much sparser plant landscape with much higher per-plant energy pushes evolution toward better navigation/foraging or more predation.

Design:
- Control: [`sim-evaluation/config.toml`](/Users/evanhu/code/NeuroGenesis/sim-evaluation/config.toml)
- Treatment: `food_fertility_threshold = 0.85`, `food_energy = 400`
- Unchanged: regrowth timing, terrain, seed genome, seeds, tick budget

Rationale:
- `food_fertility_threshold = 0.85` reduces standing plants sharply relative to the baseline `0.6`.
- `food_energy = 400` offsets the reduced plant count so total standing plant energy stays close to baseline, isolating patchiness/reward size better than simply starving the world.
