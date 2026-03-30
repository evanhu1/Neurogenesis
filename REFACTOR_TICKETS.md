# Refactor Tickets

Sequenced for behavior preservation first: extract helpers and invariants, then
split orchestration-heavy files, then consolidate policy/config ownership, and
only after that delete thin wrappers or reduce public config surface.

- [x] T01: Establish the sequenced refactor backlog in-repo so work can be
  tracked incrementally.
- [x] T02: Extract shared topology helpers and invariants into a neutral module
  used by brain/genome/plasticity.
- [x] T03: Remove the unused RNG parameter from `express_genome` and update all
  callers.
- [x] T04: Centralize repeated organism/food construction helpers in
  `spawn.rs`.
- [x] T05: Centralize organism side-array compaction in `turn.rs` so
  `organisms`, `pending_actions`, and `reward_ledgers` stay index-aligned
  through one path.
- [x] T06: Split `turn.rs` into phase-focused modules while keeping `tick`
  orchestration short and behavior-identical.
- [x] T07: Introduce explicit turn/intention context structs to reduce config
  sprawl through `build_intents` and `build_intent_for_organism`.
- [x] T08: Extract the reproduction state machine from `turn.rs`
  (`reproduction_phase`, completion queueing, and pending-action transitions).
- [x] T09: Refactor `commit_phase` into smaller helpers with one canonical death
  path and one canonical food-consumption path.
- [x] T10: Split `brain.rs` into `expression`, `sensing`, `evaluation`, and
  `topology`, with scratch-buffer lifecycle encapsulated behind helper methods.
- [x] T11: Replace manual sensory-neuron construction and action metadata with
  canonical descriptor tables derived from source-of-truth enums/constants.
- [x] T12: Split `genome.rs` into seed, mutation-rate accessors, scalar
  mutation, topology mutation, spatial prior, and sanitization modules.
- [x] T13: Introduce reusable genome vector alignment and mutate-many helpers to
  remove repeated control-flow patterns.
- [x] T14: Extract plasticity policy derivation and unify repeated sensory/inter
  synapse passes behind shared abstractions.
- [x] T15: Split `spawn.rs` into organism spawning, world generation, and food
  ecology/regrowth modules.
- [x] T16: Split `sim-validation/src/main.rs` into CLI, orchestration,
  aggregation, comparison, and output/reporting modules.
- [x] T17: Consolidate hidden policy constants versus TOML-exposed config into
  explicit owned policy/defaults objects by domain.
- [x] T18: Reorganize config documentation by domain and generate docs from the
  config source of truth where practical.
