# TASKS

## Coordination Rules

- Keep frontend data types/models in sync with backend data types/models as
  changes are made.
- Update `specs/spec.md` continuously so it stays fully aligned with the
  implemented system behavior.
- Keep turn-runner behavior updates explicitly documented in `specs/spec.md`
  because `specs/TURN_RUNNER_SPEC.md` is not present in this repo.

## Sequential Implementation Tasks

1. Lock behavioral semantics for energy-based lifecycle and reproduction in the
   turn runner before coding:
   - Define exact per-turn order for: movement, move-energy spend, reproduce
     action validation, reproduce-energy spend, baseline energy decay, and
     starvation removal.
   - Define whether move-energy cost is charged on move attempt or successful
     committed move.
   - Define whether consumption changes energy (and how).
2. Update protocol/config schemas (`sim-protocol`) and defaults:
   - Add world-level energy knobs to `WorldConfig` and `config/default.toml`:
     `starting_energy`, `reproduction_energy_cost`, `move_action_energy_cost`.
   - Add `ActionType::Reproduce` and extend `ActionType::ALL`.
   - Replace `OrganismState.turns_since_last_consumption` with
     `OrganismState.energy: f32`.
   - Add/confirm per-organism successful reproduction metric field.
   - Add world metric `reproductions_last_turn` and remove
     `births_last_turn`.
   - Remove species starvation dependency (`turns_to_starve`) entirely.
3. Refactor `sim-core` turn-runner state structures:
   - Expand brain evaluation action slots from 3 to 4.
   - Add reproduce intent capture in `build_intents`.
   - Remove hunger snapshot state and use energy snapshot state.
4. Rework `commit_phase` + lifecycle handling in `sim-core/src/turn.rs`:
   - Remove consume-triggered reproduction queueing.
   - Apply move energy cost using the chosen semantics.
   - Resolve reproduce action attempts using post-commit occupancy checks:
     insufficient energy or blocked spawn cell => no-op; otherwise enqueue spawn
     and deduct reproduction energy.
   - Increment organism + world reproduction success metrics on successful
     reproduce actions.
   - Apply per-turn baseline energy decay and starvation at `energy <= 0.0`.
5. Update spawn logic in `sim-core/src/spawn.rs`:
   - Initialize new organisms with `starting_energy`.
   - Keep reproduction placement opposite parent facing; skip when blocked/OOB.
   - Ensure starvation replacements also start with configured energy.
6. Remove hunger-based fields and logic across core:
   - Delete all `turns_since_last_consumption` reads/writes and associated
     validation.
   - Remove now-obsolete `turns_to_starve` validation/use across the project.
7. Update frontend protocol mirror/types and UI strings (`web-client`):
   - Parse new world config energy fields.
   - Replace focused-organism stats line from hunger counter to energy.
   - Add reproduction success metric display if introduced in protocol.
8. Update and expand tests:
   - Update existing helpers for 4 action neurons (include `Reproduce`).
   - Add deterministic tests for reproduce success/failure cases:
     insufficient energy, blocked spawn, enough energy + free spawn.
   - Add tests for move-energy spending and starvation at zero energy.
   - Update/rebaseline golden fixture(s) affected by action-set and lifecycle
     changes.
9. Update docs/spec:
   - Revise config, action set, turn pipeline, lifecycle, and metrics sections
     in `specs/spec.md` to match the new energy budget model.
10. Validate end-to-end:
   - Run `cargo check --workspace`, `cargo test --workspace`, and
     `cd web-client && npm run typecheck` to confirm protocol/core/client
     consistency.
