# TCPE Slice A feasibility and adversarial audit

Status: implementation-feasibility audit and go/no-go contract, not a TCPE
result and not evidence of open-endedness.

Audited against `main` at `fe1527a` (`Add delayed conditional capacity pilot`).

This document audits the first vertical slice proposed in
`docs/solver-dependent-ecology-route.md`. It cross-references that design; it
does not replace or restate the full route. The route remains a hypothesis until
the public-semantics and transfer gate in this document passes.

## Verdict

The engine can support a public, serialized task runtime and a third physical
energy compartment without violating deterministic tick order or the hidden
food-ecology boundary. The literal Slice A contract cannot be implemented as
written, however. Three requirements are mutually incompatible:

1. a meaningful public preamble cannot leave the admitted PowerPlay controllers
   behaviorally unchanged;
2. the admitted depth-2 task's two 5-unit rewards cannot be converted into one
   terminal 10-unit TCPE token without changing its physical trace; and
3. one exact construction-world snapshot cannot also be several controlled
   evaluation contexts with different nuisance values.

Those contradictions invalidate literal Slice A, not the entire TCPE route. A
corrected slice is feasible if it treats the old PowerPlay run as provenance and
a separate compatibility oracle, uses a creator-energy certificate to back
controlled evaluation templates, and trains public-program-aware solvers under
a new behavioral contract.

The current decision is therefore:

> **NO-GO on engine O/F/E implementation until an evaluator-owned,
> explicitly non-evidentiary public-semantics/transfer probe passes the complete
> gate in this document.**

Escrow accounting is substantial but mechanically well specified after the
corrections below. The uncertain premise is whether this fixed sensor/action
substrate can learn a causally necessary public program and gain a real
cross-branch transfer advantage. Test that premise before paying the integration
cost.

## Evidence base and current substrate

The bounded pilot's sealed evidence is recorded in
`artifacts/research/open-ended/powerplay-hardened/powerplay-1783929960884-83609.json`
and summarized in `docs/solver-dependent-ecology-route.md:39-78`. Its admitted
matrix is:

| Solver checkpoint | depth-1 task | depth-2 task |
| --- | ---: | ---: |
| U0 / depth 0 | 0/16 | 0/16 |
| U1 / depth 1 | 16/16 | 2/16 |
| U2 / depth 2 | 16/16 | 15/16 |

The current engine surfaces relevant to feasibility are:

- `Simulation` is the whole-world serialization boundary
  (`sim-core/src/lib.rs:79-136`). Its manual `Clone`, constructor/reset,
  snapshot, save/load, and validation paths are at lines 139-173, 177-220,
  340-380, 307-332, 395-407, and 468 onward. A behavior-affecting task runtime
  must participate in all of those except the public render snapshot.
- canonical phase order is `Simulation::tick`
  (`sim-core/src/turn/mod.rs:175-286`): capture O/F energy, lifecycle, sensing and
  intents, move resolution, commit, age, plasticity, consistency, ledger, then
  metrics/delta.
- `CommitResult` has energy aggregates but no always-on typed consumption or
  task event (`sim-core/src/turn/mod.rs:117-136`). `consume_food` performs the
  normal F-to-O transfer and classifies only Plant/Corpse
  (`sim-core/src/turn/commit.rs:157-228`).
- ordinary food construction rejects zero energy and jitters Plant visuals
  through the world RNG (`sim-core/src/spawn/food.rs:146-174`). The food-tile
  selection policy is owned by `sim-config::food_ecology_policy`
  (`sim-config/src/config.rs:104-108`).
- `FoodRay` sensing observes only the nearest `Occupant::Food` and distance,
  never food kind, energy, ID, task hash, or escrow
  (`sim-core/src/brain/sensing.rs:128-167`). This is an adequate public cue
  channel in a controlled empty arena and an ambiguous channel in ordinary
  plant ecology.
- `FoodKind` has only Plant/Corpse and `EnergyLedgerRow` has only O/F
  compartments (`sim-types/src/lib.rs:176-180`, `813-843`). Both are already
  server-visible through `FoodState` and `MetricsSnapshot`.
- `ProtectedResidual`, all-history retention, and extension knockout are
  reusable, but task obligations are still keyed by numeric `task_id` and
  `exact_fingerprint` hashes ordered CBOR without machine minimization
  (`sim-core/src/progressive.rs:23-181`, `210-220`, `231-408`).
- the new conditional pilot has a useful controlled task configuration and
  intervention artifact shapes, but it explicitly records that task runtime and
  escrow remain evaluator-owned and that exact strong/stutter canonicalization
  is absent (`sim-core/src/conditional.rs:544-549`).

The conditional pilot's `prepare_phase` must not become the production task
runtime. It directly clears foods, fertile cells, and regrowth state, writes
zero-energy objects as `FoodKind::Plant`, and transfers escrow directly to the
organism between ticks (`sim-core/src/conditional.rs:2139-2276`). That is useful
for a disposable falsification probe, but it is precisely the evaluator
injection TCPE is intended to remove.

## Fatal contradictions in literal Slice A

### F1: meaningful preamble versus unchanged legacy behavior

The route requires a self-describing FoodRay preamble and says any behavioral
change from the admitted depth-1/depth-2 matrix blocks Slice A
(`docs/solver-dependent-ecology-route.md:138-154`, `451-459`). A meaningful
preamble necessarily executes additional neural evaluations before the old
task:

- recurrent state and action activations advance during every sensing pass;
- age increments after each commit (`sim-core/src/turn/snapshot.rs:42-45`);
- Hebbian state can change when enabled;
- `Simulation::turn` advances; and
- the next action sample changes because it hashes `(seed, turn, organism ID)`.

The exact implementation is at `sim-core/src/turn/mod.rs:521-530`:

```text
mixed = sim_seed
      XOR wrapping_mul(turn, 0x9E3779B97F4A7C15)
      XOR wrapping_mul(organism_id, 0xBF58476D1CE4E5B9)
sample = high_24_bits(mix_u64(mixed)) / (2^24 - 1)
```

For `seed = 7`, `organism_id = 0`, using the current Rust f32 conversion, the
samples are:

| Turn | high-24 numerator | sample |
| ---: | ---: | ---: |
| 0 | 1,224,240 | 0.072970398 |
| 1 | 16,211,716 | 0.966293633 |
| 8 | 5,504,214 | 0.328076750 |
| 16 | 9,198,736 | 0.548287451 |
| 32 | 910,000 | 0.054240230 |
| 64 | 6,737,189 | 0.401567787 |

Thus even a 16-tick preamble changes the action variate from `0.072970398` to
`0.548287451` before considering neural state. Resetting or reconstructing the
world after the preamble can recover the old trace only by erasing the neural
information the preamble was meant to convey. Such a preamble would fail the
blank-preamble and opcode-permutation causality controls.

**Disposition:** fatal to literal Slice A. Preserve the old matrix in an
independent no-preamble compatibility arm. Public TCPE solvers receive a new
matrix and must pass public-semantics interventions; byte-identical old behavior
is not a valid requirement.

### F2: one terminal token versus two legacy depth-2 transfers

TCPE specifies exactly one payoff token `E* = 10`, staged only at terminal
(`docs/solver-dependent-ecology-route.md:392-418`). PowerPlay divides its fixed
10-unit episode budget by program depth:

```text
stage_energy = config.food_energy / program.stages.len()
```

That code is at `sim-core/src/powerplay.rs:1064`, and each completed stage spawns
the next resource at lines 1093-1100. Depth 1 therefore transfers one 10-unit
Plant. Depth 2 transfers two 5-unit Plants. In the admitted seed-7 depth-2 trace,
the organism's energy changes `300 -> 305 -> 310`, with stage completions at
turns 3 and 9.

Replacing the first 5-unit Plant with a zero-energy cue and releasing one 10-unit
terminal reward changes the Energy receptor after the first consumption and the
entire physical/brain trace. Retaining two 5-unit `TaskReward`s violates the
single-token contract.

**Disposition:** fatal to literal trace equivalence. T1/T2 may seed causal
provenance and a compatibility oracle; their TCPE recompilations are new public
tasks with new traces.

### F3: exact construction snapshot versus crossed controlled contexts

The route requires an actual ordinary plant capture and O-to-E deposit, then
says evaluation contexts fork that exact serialized snapshot
(`docs/solver-dependent-ecology-route.md:394-412`). Its audits also require
different seeds, poses, alpha-numberings, and legal serializations in a
controlled arena with no ambient food or regrowth.

Those cannot all be the same world bytes:

- changing a world seed or serialized task context changes `Simulation`;
- a construction world with an ordinary food tile retains hidden ecology and a
  regrowth schedule after capture;
- clearing that state after the certificate is an additional world mutation;
- the strict task arena intentionally sets `food_tile_fraction = 0`, whereas the
  construction trace must contain an ordinary plant source.

**Disposition:** fatal to literal exact-snapshot wording. The construction world
must emit a `CreatorEnergyCertificate` that binds source-world hashes, task
hashes, creator, actual Plant consumption, deposit, build cost, and closing
ledger row. A deterministic controlled evaluation template imports the certified
10 units as initial escrow. Each nuisance context forks a template. No energy is
added after a template is instantiated; counterfactual forks replicate the same
certified initial condition in the same sense that they replicate founder
energy.

## Repairable requirements that must be frozen before production work

| Gap | Required decision |
| --- | --- |
| Public protocol | Version a concrete prefix-free binary format, opcode table, length/framing rules, and FoodRay rendering. The design currently defines symbols but not bytes. |
| Event semantics | Define one fixed compound post-commit event per tick, including selected physical action plus actual move/consumption outcome. Transitions must never score a logit alone. |
| Construction trigger | The current action enum has no Build/Deposit. The smallest slice lets the outer creator algorithm queue an install; the canonical tick performs the real debit after actual Plant capture. If organism-authored construction is required, add a physical action and reopen scope. |
| Construction price | Freeze exact `c_state` and `c_edge` values in the protocol version. They should be exactly representable energy values and cannot depend on task age, score, or archive index. |
| Source traces | Regenerate T1/T2 traces. `BehaviorStep` records action, pose, food positions, energy, and completion only (`sim-core/src/powerplay.rs:161-175`); it lacks sensory vectors, typed commit facts, transition/effect rows, and per-tick energy ledgers. |
| Counterfactual source | Add an explicit diagnostic action-override path that forks a world and drives one alternate action through ordinary resolution/commit. A different genome is not the same-state one-action counterfactual. |
| Canonical bytes | Use a constrained, versioned, array-only integer representation or another proven canonical CBOR encoder. Ordered `ciborium` serialization alone is not a bisimulation or alpha-canonicalization certificate. |
| Token accounting | Replace cumulative ambiguous `staged` wording with the token state machine and per-tick conservation equations below. |

## Corrected smallest honest vertical slice

The smallest honest slice has three consolidated task identities, two solver
branches, one trace-derived cross-branch task, and one claimed transfer. It does
not contain three simultaneous task runtimes and does not claim an open-ended
tail.

### 1. Separate legacy compatibility from TCPE evidence

Keep a frozen compatibility arm that executes the existing PowerPlay code with
no preamble and reproduces the sealed U0/U1/U2 matrix. This detects accidental
changes to the imported genomes and old task evaluator. It is not a public-task
result.

Regenerate complete successful traces for U1 on depth 1 and U2 on depth 2. Store
normalized public facts and non-observable provenance separately. Compile those
facts into new canonical T1 and T2. Do not require their action/energy traces to
equal the compatibility arm.

### 2. Establish two independent branches

Branch A owns `(T1, resident A)` and enumerates trace-split candidates from
successful and exact one-action-counterfactual T1 slices. Branch B owns
`(T2, resident B)` and enumerates causal-lift/trace-splice candidates from T2.
Search the branches independently with equal budgets and disjoint mutable
contexts. Persist every candidate and rejection, including canonical duplicates.

The branches are archive/evolution objects, not multiple hidden task truths in
one `Simulation`.

### 3. Derive, do not hand-author, T3

Enumerate splice interfaces in fixed source-hash/range order. A legal interface
must have the same normalized public scene, task-token state, escrow balance,
and organism-energy baseline. Canonicalize every result and reject unreachable,
bisimilar, stutter-only, old-hash, or unexecuted changes.

T3 is the lexicographically first feasible semantic hash whose successful trace
actually reaches provenance-backed transitions from both A and B. If no such T3
exists, the route is blocked; do not substitute a hand-authored task.

### 4. Test transfer and consolidation

For frozen T3, run direct evaluation and exactly equal-budget protected
adaptation from resident A, resident B, and U2. Persist the complete solver x
task direct/adapted matrix. A qualifying result still needs:

- foreign-branch adapted advantage at least 6/16 over both resident and
  universal seeds;
- at least 4/16 loss when the foreign source module is knocked out;
- one consolidated solver at least 14/16 on T1, T2, and T3;
- every earlier checkpoint and the exact new residual knockout at most 2/16 on
  T3; and
- complete alpha/order, held-out recombination, blank-preamble, opcode,
  energy, replay, and fixed-ecology controls.

The strongest allowed claim after this slice is one solver-dependent,
energy-grounded, canonically new public task with one causally necessary
cross-pair stepping stone and complete historical retention.

## Mandatory pre-engine falsification probe

The existing conditional pilot provides a cheap way to attack the scientific
premise before changing `Simulation`, `FoodKind`, the ledger, and the wire
schema. This probe is intentionally evaluator-owned and therefore can never be
reported as Slice A evidence.

### Probe construction

1. Reuse the controlled one-organism, no-wall, no-food, zero-cost configuration
   in `sim-core/src/conditional.rs:2112-2127`.
2. Implement a disposable reference Mealy scheduler using the current
   evaluator-owned cue/pose machinery. Keep its escrow and scoring explicitly
   out of all result claims.
3. Present fixed-length versioned public preambles for T1/T2 and a
   deterministically trace-spliced T3.
4. Run two independent protected-residual solver branches with equal proposal
   and mutation budgets.
5. Persist exact sensory/action/commit traces and all candidate rejections; an
   aggregate pass count is insufficient.

### Probe pass gate

The probe is GO only if all of these hold on three independent outer seeds:

- the enabled consolidated controller reaches at least 14/16 on each of T1,
  T2, and T3;
- every earlier controller and exact new residual knockout is at most 2/16 on
  T3;
- a foreign branch exceeds both the resident and universal seeds by at least
  6/16 after the same adaptation budget;
- knocking out the foreign-source module removes at least 4/16 of that
  advantage;
- state renumbering, transition reordering, unseen legal serialization, and
  held-out cross-lineage recombination remain at least 14/16;
- blanking the public program or permuting one physical opcode lowers success to
  at most 2/16;
- the response follows donor neural state in brain-swap controls rather than
  hidden host truth; and
- a frozen solver/task repertoire stops admitting semantic tasks after its
  reachable counterexamples are exhausted.

Any failed condition is NO-GO for TCPE on the current affordance substrate. Do
not respond by increasing state count, preamble length, population, or search
budget. The next mechanism must change ecological affordances or construction,
not enlarge this task language.

Only a complete GO result authorizes implementation of the engine O/F/E slice.

## Production task runtime if the probe is GO

### Canonical object model

The production module should use versioned values equivalent to:

```text
CanonicalTaskProgram {
  protocol_version,
  contexts,
  roots,
  states,
  fixed_event_alphabet,
}

TaskRuntime {
  execution_hash,
  semantic_hash,
  current_state,
  phase,
  preamble_cursor,
  participant_id,
  canonical_pose,
  escrow_locked,
  token_state,
  actual_trace,
  creator_energy_certificate,
}

TaskTokenState = Locked | Staged { food_id, energy } | Captured
```

Program canonicalization must validate total transitions and flows, remove
unreachable states, compute the coarsest strong bisimulation, quotient,
alpha-canonicalize exact colored roles, produce a separate weak/stutter
quotient, encode canonical bytes, and store original-to-quotient certificates.
Semantic hash, never archive index or numeric task ID, keys obligations.

Normalized creator input contains only public observation, abstract emission,
actual committed event, task transition/effect, and escrow flow. IDs, absolute
coordinates, raw RNG values, hashes, seeds, and unobserved state remain in the
non-observable provenance record.

### Canonical tick integration

The production phase order must be:

```text
clear transient state
capture organism O, food F, and escrow E totals
lifecycle
task pre-sense render / construction activation
normal sensing and intent construction
normal move resolution
normal commit
task actual-event observation / creator deposit
age
normal post-commit plasticity
task normalization, reward recovery, and settlement
consistency
extended O/F/E ledger
metrics and delta
```

The post-commit observer must run before intent/resolution scratch buffers are
returned. Settlement runs after plasticity so real TaskReward capture remains a
normal within-tick energy consequence for Hebbian learning. Task cues and
standing unconsumed rewards are removed atomically while occupancy is repaired.

### Hidden ecology firewall

- task code must not import `sim_config::food_ecology_policy` or mutate
  `food_tiles`, `food_regrowth_due_turn`, or `food_regrowth_schedule`;
- controlled evaluation worlds use the existing public config overrides
  `food_tile_fraction = 0` and `terrain_threshold = 1`, then assert the realized
  empty arena;
- task spawn uses a dedicated deterministic allocator, permits zero energy only
  for `TaskCue`, consumes no `Simulation::rng`, and never schedules regrowth;
- `TaskReward` is positive, finite, escrow-backed, and never counted as Plant,
  prey, or ordinary consumption; and
- the brain receives only existing FoodRay/contact/energy inputs. Task hash,
  archive position, escrow, token role, and creator metadata never enter
  `Simulation::seed`, action sampling, or neural inputs.

### Creator-funded construction

Because no Build/Deposit action exists, the smallest slice uses a host-queued
`PendingTaskInstall`. The queue is an outer-algorithm proposal, not a direct
energy mutation. A canonical post-commit hook activates it only if:

- the named creator still exists;
- its recorded ordinary Plant capture since construction start covers `E* + B`;
- its current energy can fund the deposit and build cost;
- task hashes and canonicalization certificate validate; and
- the resulting O-to-E transfer and O-to-sink build cost close in that tick's
  ledger.

The certificate binds this construction fact to the controlled evaluation
template. If an autonomous organism decision to construct is required, the
current action encoding cannot express it; that is a larger morphology/action
change, not part of this slice.

## Exact O/F/E energy contract

Define the following nonnegative finite flows for one tick:

- `X_F`, `X_O`: ordinary Plant/Corpse food debit and organism credit;
- `D_O`, `D_E`: creator deposit organism debit and escrow credit;
- `S_E`, `S_F`: task staging escrow debit and food credit;
- `C_F`, `C_O`: TaskReward capture food debit and organism credit;
- `R_F`, `R_E`: unconsumed reward recovery food debit and escrow credit;
- `B`: canonical construction cost dissipated from organism energy;
- `M`, `A`: existing metabolism and action sinks; and
- existing predation, corpse, unrecycled-removal, and signed removal-adjustment
  terms retain their current meanings.

Per-compartment expectations are:

```text
O_expected =
  O_before - M - A - D_O - B
  + X_O + C_O + predation_energy_credit
  - unrecycled_energy_removed
  - predation_prey_energy_removed
  - corpse_source_energy_removed

F_expected =
  F_before + plant_spawn_energy + corpse_spawn_energy
  - X_F - C_F + S_F - R_F

E_expected =
  E_before + D_E - S_E + R_E
```

The total expectation is the existing total-compartment equation extended by
`E_before` and `-B`:

```text
Total_expected =
  O_before + F_before + E_before
  + plant_spawn_energy
  - M - A - B
  - predation_retention_loss
  - corpse_retention_loss
  + removal_adjustment
```

Deposit, staging, capture, and recovery cancel in O+F+E, but each transfer must
close independently so errors cannot cancel:

```text
deposit_residual  = D_E - D_O
staging_residual  = S_F - S_E
capture_residual  = C_O - C_F
recovery_residual = R_E - R_F
```

Assert the O, F, E, total, and all four transfer residuals against the existing
scale-aware tolerance. Include all new flows in the tolerance scale.

The unambiguous per-tick token-supply equation is:

```text
standing_before + S_F = C_F + R_F + standing_after
```

For a constructed task with fixed `E* = 10`, the episode invariant is:

```text
E* = escrow_locked_now + standing_reward_now + cumulative_captured
```

Recovery moves the same token from Staged back to Locked; restaging it does not
create a second cumulative supply. At most one TaskReward food ID may be live.
Negative/nonfinite escrow, cue energy, top-up, duplicate token, Plant
classification, regrowth, or any residual above tolerance aborts the run.

## Exact implementation surfaces after GO

| Surface | Required production change |
| --- | --- |
| `sim-core/src/task.rs` (new) | Machine validation/canonicalization, public renderer, serialized runtime, trace normalization/provenance, event observer, certificate verification, settlement. |
| `sim-core/src/coevolution.rs` (new) | Pair archive, deterministic split/splice/lift enumeration, branch search, transfer sweep, rejected-candidate table, consolidation. |
| `sim-core/src/lib.rs:79-173` | Optional serialized `TaskRuntime` and pending install; manual Clone. |
| `sim-core/src/lib.rs:307-407`, `468+` | Do not expose runtime through normal snapshot; persist it through save/load and validate hashes, escrow, token/food ownership, participant, and canonical state. Reset clears it unless using the dedicated task constructor. |
| `sim-core/src/turn/mod.rs:117-136`, `175-286`, `354-505` | Typed task commit facts, three task hooks, E snapshot, extended ledger/residual assertions. |
| `sim-core/src/turn/commit.rs:157-228`, `428-464` | Classify TaskCue/TaskReward consumption, no ecology counters/regrowth, preserve actual transfer facts through compaction/finalization. |
| `sim-core/src/spawn/food.rs:39-105`, `146-181` | Dedicated no-RNG task spawn/remove helpers; ordinary replenish remains Plant-only. |
| `sim-types/src/lib.rs:176-180`, `772-843`, `919-935` | Task food kinds/visuals and escrow ledger fields. Keep full program/runtime in sim-core unless deliberately exposed. |
| `sim-core/src/progressive.rs:210-408` | Semantic-hash obligation keys and one-to-one numeric metadata assertion; retain protected residual and history/knockout logic. |
| `sim-cli/src/coevolution.rs` (new) | One-shot probe/production commands, canonical task inspection, matrices, and single-context trace artifact output. |
| `sim-views/src/lib.rs:539-649`, `994-1005` | Report E/task flows and make FoodKind summaries exhaustive without polluting Plant counts. |
| `sim-server/src/protocol.rs:66-119` | Review the existing passthrough schema; do not add task truth to normal world frames. |
| `web-client/src/types.ts:30`, `239-290` | Mirror new FoodKind variants and every ledger field exactly. |
| `web-client/src/protocol.ts:78-80`, `113-171` | Normalize/forward the updated food and ledger schema explicitly. |

No task protocol field belongs in `WorldConfig`, so baseline TOMLs require no
schema change. If a later design makes task behavior configurable through
`WorldConfig`, both sim-config and sim-evaluation baselines must change together.

## Verification and artifact contract

No new tests should be added. After implementation, run the maintained suite and
repository checks:

```text
cargo check --workspace
cargo test --workspace
make fmt
make lint
cd web-client && npm run build && npm run typecheck
```

Any render-visible frontend change also requires local browser verification.

The production slice is not complete without persisted, inspectable facts:

- canonical execution/semantic CBOR and hashes, quotient maps, and certificates;
- creator construction world hash, ordinary Plant capture, deposit/build ledger,
  and controlled-template certificate;
- every per-tick O/F/E ledger and token reconciliation;
- normalized source slices plus exact non-observable provenance DAG;
- complete candidate/rejection, solver x task, adaptation, transfer, and
  knockout matrices;
- per-tick preamble, sensory vector, brain hash, action sample/logits/action,
  actual commit event, transition, pose normalization, and energy flow;
- all alpha/order/blank/opcode/replay/brain-swap/fixed-ecology controls; and
- directly loadable world files at decisive task boundaries so `sim-cli`
  `inspect`, `brain`, and `decide` can independently check behavior.

Save/load/fork determinism must be demonstrated by byte-identical world files
for identical commands, including task runtime, escrow, token state, trace, and
ID allocation. A scalar score, aggregate matrix without traces, or closed custom
episode sum without per-tick engine rows does not satisfy this audit.

## Final decision rule

1. Literal Slice A is **NO-GO** because F1-F3 are contradictions, not missing
   code.
2. The corrected TCPE route is **conditionally feasible** on the current engine.
3. Run the evaluator-owned public-semantics/transfer probe first. It is a
   falsifier only and cannot count as implementation evidence.
4. If every probe gate passes on three outer seeds, the result is **GO** for the
   production TaskRuntime and O/F/E vertical slice described here.
5. If any gate fails, the result is **NO-GO** for TCPE on the fixed current
   sensor/action affordance substrate. Do not implement escrow infrastructure or
   reopen the route with only more scale or preamble complexity.
