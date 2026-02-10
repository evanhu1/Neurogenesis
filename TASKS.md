# TASKS

## Goal

Split config DNA into:

- `WorldConfig`: session/world-level parameters.
- `SpeciesConfig`: per-species evolutionary parameters.
- `WorldConfig.seed_species_config`: the seed DNA used only at world
  initialization/reset time.

Runtime model target:

- `Simulation` stores all current species as a stable integer ID ->
  `SpeciesConfig` mapping.
- Runtime species registry is exposed in protocol snapshots.
- Every organism stores a `species_id` reference.
- This change supports multiple species structurally, but this iteration only
  initializes and uses one species (no speciation/new-species creation logic).
- `SpeciesId` is `u32`; seeded species starts at ID `0`.

Proposed field split:

- `WorldConfig`: `world_width`, `steps_per_second`, `num_organisms`,
  `center_spawn_min_fraction`, `center_spawn_max_fraction`.
- `SpeciesConfig`: `num_neurons`, `max_num_neurons`, `num_synapses`,
  `turns_to_starve`, `mutation_chance`, `mutation_magnitude`,
  `mutation_operations`.

## Sequence

1. Define `SpeciesConfig` in `sim-protocol/src/lib.rs` and move species fields
   out of `WorldConfig`.
2. Rename the nested seed field to `seed_species_config` in `WorldConfig` and
   update serde/default behavior.
3. Update `config/default.toml` to the new shape (world keys + nested
   `seed_species_config` keys) while preserving default behavior.
4. Add a stable integer species ID type and wire it through protocol/core
   models where needed (`SpeciesId`, `species_id` on organism state, etc.).
5. Add runtime species registry to `Simulation` (species ID -> `SpeciesConfig`)
   plus deterministic ID allocation for future species additions.
6. Initialize/reset simulation species registry from
   `WorldConfig.seed_species_config` with exactly one species for now (for
   species ID `0`).
7. Refactor sim-core execution paths to resolve per-organism species config
   through `species_id` instead of directly from world config:
   `sim-core/src/brain.rs`, `sim-core/src/spawn.rs`, `sim-core/src/turn.rs`,
   `sim-core/src/grid.rs`, and `sim-core/src/lib.rs`.
8. Ensure all spawn paths assign and preserve organism `species_id` correctly:
   initial population, starvation replacement, and reproduction.
   Reproduction inherits parent `species_id`; starvation replacement picks a
   random species from the current registry.
9. Split validation into world-level and species-level validation, including
   validation of `seed_species_config`.
10. Update protocol payloads and snapshots for new schema/fields, including
    config nesting, exposed species registry, and organism species linkage;
    bump `PROTOCOL_VERSION` because wire JSON changes.
11. Update `sim-server/src/main.rs` and tests to compile and run with the new
    config and organism/species model.
12. Update web client types and parsers (`web-client/src/types.ts`) for
    `seed_species_config`, `SpeciesConfig`, and organism `species_id`; adjust
    consuming code as needed.
13. Update docs in `specs/spec.md` to document:
    world DNA (`WorldConfig`),
    seed species DNA (`seed_species_config`),
    runtime species registry semantics,
    and current single-species limitation.
14. Update tests in `sim-core`, `sim-protocol`, `sim-server`, and regenerate
    `sim-core/tests/fixtures/golden_seed42_turn30.json` for the new snapshot
    schema.
15. Verify end-to-end with:
    `cargo test --workspace`,
    `cargo check --workspace`,
    `cd web-client && npm run typecheck && npm run build`.
