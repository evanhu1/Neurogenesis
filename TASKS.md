# TASKS

## Coordination Rules

- Keep frontend data types/models in sync with backend data types/models as
  changes are made.
- Update `specs/spec.md` continuously so it stays fully aligned with the
  implemented system behavior.
- Treat `specs/TURN_RUNNER_SPEC.md` as the source of truth for the centralized
  runner design and resolution semantics.

## Sequential Implementation Tasks

1. Add centralized turn-runner data structures in `sim-core`:
   - `TurnSnapshot`
   - `OrganismIntent`
   - `MoveCandidate` / `MoveResolution`
   - `SpawnRequest`
   - any helper indexes needed for O(1) lookups.
2. Implement Snapshot Phase to build immutable start-of-turn state:
   - occupancy snapshot
   - stable organism ordering by `OrganismId`
   - pose/facing/hunger state
   - per-organism move confidence value derived from the hidden-state signal
     required by the spec.
3. Implement Intent Phase:
   - evaluate every organism brain against the same frozen snapshot
   - compute `facing_after_turn`, `wants_move`, `move_target` (if in bounds)
   - attach deterministic move confidence for conflict resolution.
4. Implement global Move Resolution Phase:
   - resolve all intents simultaneously
   - use highest confidence as primary winner rule
   - use deterministic tie-break fallback (`OrganismId`) for equal confidence
   - support empty-target moves, occupied-target contesting, and vacated-target
     entry in the same turn.
5. Extend Move Resolution Phase for graph cases:
   - support two-way swaps and longer move cycles
   - ensure cycle handling remains deterministic and conflict-safe.
6. Implement Commit Phase (single atomic commit):
   - apply all facing updates
   - apply resolved moves
   - apply eat-and-replace kills
   - update eater lifecycle fields (`turns_since_last_meal`, `meals_eaten`)
   - enqueue reproduction spawn requests.
7. Implement Lifecycle Phase (starvation) inside the same runner:
   - increment hunger for non-eaters
   - apply starvation deaths
   - enqueue starvation replacement spawn requests.
8. Implement Spawn Resolution Phase:
   - process queued spawns in deterministic order
   - if full, skip spawn
   - spawn on empty hexes sampled from a center-weighted Gaussian distribution
     per spec.
9. Integrate Metrics + Delta Phase:
   - finalize `turn` increment and all per-turn metrics
   - produce accurate movement deltas from committed results.
10. Remove/retire old per-organism immediate action path so only the centralized
    runner mutates movement/eat/starve/spawn outcomes.
11. Update API/client-facing integration points as needed:
    - ensure server event flow and web-client state application match centralized
      commit semantics.
12. Sync `specs/spec.md` with the final implemented behavior after code changes
    are complete.

## Final Testing Phase (Move Resolution Focus)

13. Add a concentrated move-resolution test suite in `sim-core` covering:
    - move into cell vacated in same turn
    - two-organism swap
    - multi-attacker single target (confidence winner + deterministic tie break)
    - attacker vs escaping prey
    - multi-node cycle resolution
    - contested occupied target where occupant remains (eat path).
14. Add lifecycle + resolution interaction tests:
    - starvation/reproduction interactions in the same centralized turn
    - spawn-queue processing order under limited free space
    - no-overlap invariant after mixed move/eat/starve/spawn turns.
15. Add determinism regression tests for centralized resolution:
    - same seed/config => identical snapshots
    - targeted scenario snapshots for complex resolution turns.
16. Run full verification only after the above tests are in place:
    - `cargo test --workspace`
    - `cargo check --workspace`
    - `cd web-client && npm run typecheck`
    - `cd web-client && npm run build`.
