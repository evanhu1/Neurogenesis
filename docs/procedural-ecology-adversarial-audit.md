# Procedural-ecology Stage-0 adversarial audit

Status: deterministic mechanics pass; evolutionary Stage 1 and open-endedness
claim rejected.

Date: 2026-07-13.

## Verdict

The evaluator-owned procedural-ecology wrapper is a reproducible mechanism
probe, not an open-ended neuroevolution algorithm. It establishes that a local
ecology rule can react causally to consumption while a fixed release budget is
tracked through a separate escrow. It does not establish endogenous funding,
organism learnability, coevolution, seed robustness, a valid novelty
descriptor, or an expanding behavioral affordance space.

The serialized result therefore sets both:

```text
stage_1_authorized = false
open_endedness_demonstrated = false
```

This implementation is blocked as a complete solution. It has three
hand-authored policies, one carrier, one-bit feedback, a fixed release schedule,
and a fixed eater. It contains no evolved ecology population, organism-learning
result, regret selection, transfer, immutable behavior archive, or tail test.
The broader family of translation-equivariant procedural ecologies is not
disproved by this bounded probe.

## Reproduction

Build the release CLI, then execute two independent runs:

```sh
cargo build -p sim-cli --release

./target/release/sim-cli procedural-ecology-stage0 \
  --run-seeds 7,42,123 \
  --horizon 2000 \
  --release-interval 100 \
  --plant-energy 10 \
  --out-dir artifacts/research/open-ended/procedural-ecology-stage0-final-v2

./target/release/sim-cli procedural-ecology-stage0 \
  --run-seeds 7,42,123 \
  --horizon 2000 \
  --release-interval 100 \
  --plant-energy 10 \
  --out-dir artifacts/research/open-ended/procedural-ecology-stage0-final-v2-replay
```

Primary artifact:
`artifacts/research/open-ended/procedural-ecology-stage0-final-v2/procedural-ecology-stage0-1783943744919-31422.json`.

Replay artifact:
`artifacts/research/open-ended/procedural-ecology-stage0-final-v2-replay/procedural-ecology-stage0-1783943744923-31416.json`.

The files are byte-identical, SHA-256
`42faf58ce86eaf43dfeaa96b09c8ea9636a2414485c631ad0c0a6676af1a0d76`.
The embedded result fingerprint is
`2c515fa5755af20a529f7800a8359ded7063699e368a9e2750b3515693facdfd`.

## What passed

The three run seeds produce 24 cases: stationary, moving-front, and
consumption-responsive policies; translated copies; a consumption-input clamp;
and duplicate replay. The artifact contains 48,000 wrapper ledger rows and 480
scheduled releases.

All 19 scoped mechanics gates pass:

- release times, counts, and energy are fixed across arms;
- the preloaded per-case escrow covers every scheduled release;
- every boundary flow agrees with its release-event row;
- the wrapper observes no unaccounted canonical plant spawn;
- translated traces agree after coordinate normalization;
- clamping the behavior input changes the responsive placement trace;
- physically identical stationary and clamped-responsive cases have the same
  physical fingerprint even though their diagnostic fingerprints differ;
- duplicate execution is exact; and
- every recorded organism, food, ecology, transfer, engine, and total residual
  is exactly zero.

The seeds deliberately do not vary the forced eater's behavior. This is replay
coverage, not multi-seed evolutionary robustness.

## Hostile findings

### 1. The initial ecology energy is an evaluator endowment

Each case initializes `release_count * plant_energy`, or 200 energy in the
reported configuration, without debiting a canonical organism, food, artifact,
or world compartment (`sim-core/src/procedural_ecology.rs`). The subsequent
reclaim/release transfers close within that case, but the 24 counterfactual
worlds are independently endowed and their energy cannot be aggregated into an
endogenous conservation claim.

### 2. Gross boundary transfers can cancel in one row

On 162 rows, an unconsumed plant is reclaimed and a new plant is released in
the same wrapper boundary. When both carry 10 energy, food and escrow each have
equal before/after totals while all four gross transfer fields are nonzero.
Static code inspection confirms both operations happen, but a fail-closed
Stage-1 ledger would need intermediate post-reclaim and post-release physical
checkpoints.

### 3. Evolved movement can abort the evaluator

Release currently fails if the carrier cell is occupied. The fixed eater avoids
that branch; an evolved mover need not. Stage 1 would require deterministic,
translation-equivariant deferral or relocation semantics whose energy and
schedule effects are explicit.

### 4. Consumption is read through private world membership

The wrapper infers consumption from the released `FoodId` disappearing from
`Simulation::foods`. A public removal event exists in `TickDelta`, so a valid
coevolutionary mechanism should consume only that public fact and reject an
unexplained disappearance rather than treating it as behavior.

### 5. The disabled-path comparison is a no-op regression only

`disabled_ecology_hook` is literally empty. Its exact replay shows that an
empty call changes nothing; it does not exercise the enabled controller's real
state allocation, before/after-tick entry points, save/load behavior, or
disabled branch. The gate is named accordingly in the result.

### 6. Canonical and procedural plants lack provenance

The fixture disables the hidden canonical food ecology. Procedural releases
are nevertheless ordinary `FoodKind::Plant` values. In a combined world, their
consumption could schedule canonical tile regrowth, and the current rows cannot
distinguish procedural from canonical energy. Coexistence requires explicit
resource provenance and separate flows.

### 7. Diagnostic labels are not behavior

The initial descriptor included raw/applied input bookkeeping. Seed 7's
stationary case and clamped responsive case have identical release positions,
2,000-row physical ledgers, escrow, and final world fingerprint, yet their
diagnostic hashes differ. The implementation now publishes a separate physical
hash; those two cases share
`abba2775a83c7360048c13329cc11d55b9d366d48f402dfcbdecc497667ab485`.
No Stage-1 behavior descriptor has been validated.

## Why Stage 0 cannot authorize the broader route

The organism does not observe ecology state, a released food's identity,
consumption events, program opcodes, or task semantics. It observes generic
food-distance rays, contact, energy, and optionally generic organism/health
rays (`sim-core/src/brain/sensing.rs`, `sim-types/src/lib.rs`). Its physical
outputs remain turn, move, eat, attack, or idle (`sim-types/src/lib.rs`).

Because the three policies differ mainly in the trajectory of identical plant
releases, a recurrent search-and-track controller is a plausible saturation
strategy. This has not been demonstrated. A sensor-matched tracker experiment
is required before claiming that ecology-specific inference is unnecessary or
that this family collapses to generic tracking.

A fixed two-state ecology can still be a useful causal-learning falsifier:
select teacher rules by best-response regret, evolve protected residual
solvers, replay every historical ecology, and compare closed-loop, open-loop,
frozen, input-clamped, fixed-tracker, oracle-tracker, and knockout arms. Kill
that bounded route if the fixed tracker reaches admission or if new teacher
syntax continues after solver traces converge to tracking.

A finite observation/action alphabet does not itself bound temporal strategy
complexity when recurrent controller state, ecology-program state, or
persistent spatial structure can grow. The unresolved empirical question is
whether later distinctions remain observable and learnable through the current
sensors, and whether admitted organism behavior continues growing after strong
generic-tracker controls. Even a bounded Stage-1 pass would demonstrate only
adaptive exploitation through its tested horizon.

The strongest concrete substrate candidate remains a public, endogenously
funded, recursively composable affordance whose causal effects are available to
organism brains and can become inputs to later construction. That open gap and
the fixed-resource finiteness boundary are derived in
`docs/open-endedness-finiteness-audit.md`.
