# Delayed conditional foraging: adversarial design audit

Status: pre-implementation audit, grounded in repository commit `d75b23c`.
This audit was performed independently of the conditional-program
implementation route.

## Verdict

A naive delayed conditional-foraging task is not a valid memory or open-endedness
test on this substrate. The organism can carry the cue in position, facing,
energy, runtime plasticity state, or evaluator/world-seed correlations; a task
food injected between ticks can also bypass the current energy ledger. Merely
lengthening the delay grows an exact timed automaton but leaves the required
policy at one bit. That is a bounded, static ladder.

A two-gate task can be made into a valid causal capacity audit if all of the
contracts below are implemented. The decisive test is not aggregate reward. It
is a paired intervention: for the same delay, action-sampling stream, organism
ID, turn schedule, food IDs, and canonical pose, the two cue values require
mutually exclusive gate-consumption vectors. External state is erased after
every tick. Preserving the full neural state must solve the pair; re-expressing
the genome after the cue must collapse to chance; swapping the full neural state
between the paired worlds must swap the response.

Even the stronger growing delayed-copy family described below is only a
capacity/minimal-criterion instrument. Its abstract memory rank is unbounded,
but its challenges are supplied by a predeclared generator and are therefore
implicit in that generator. Passing it does not, by itself, demonstrate
open-ended behavioral or ecological novelty.

## 1. Current substrate facts that constrain the task

All claims in this section refer to `d75b23c`.

### Tick and action semantics

The canonical tick captures organism and food energy, runs lifecycle, builds
intents (and therefore senses and evaluates the brain), resolves moves, commits
actions, increments age, applies runtime plasticity, checks consistency, and
only then builds the ledger (`sim-core/src/turn/mod.rs:175-266`). Therefore:

- A cue or gate must be staged before `build_intents`; staging it after sensing
  makes it actionable only on a later tick.
- The selected action is not the outcome. Facing and action cost are committed
  first, then moves, then Eat/Attack interactions
  (`sim-core/src/turn/commit.rs:52-56`, `60-90`, `115-154`). The evaluator must
  score the consumption recorded by commit.
- An Eat targets exactly the forward neighbor
  (`sim-core/src/turn/intents.rs:183-229`). A cue placed on a side ray is visible
  but cannot be eaten on its cue tick.
- Food consumed in commit is removed before regrowth, while food replenishment
  occurs during commit finalization (`sim-core/src/turn/commit.rs:157-228`,
  `428-450`). Task food must not use the ecological regrowth path.

### What the organism can observe

With predation disabled, the active receptors are the three relative FoodRay
inputs, ContactAhead, and Energy (`sim-types/src/lib.rs:337-370`). Ray casting
returns only the nearest occupant type and distance; every `FoodKind` currently
looks alike to the brain (`sim-core/src/brain/sensing.rs:77-97`, `128-163`).
ContactAhead is occupancy of the forward neighbor, and Energy is a monotone
function of organism energy (`sim-core/src/brain/sensing.rs:106-121`).

This makes a left-versus-right food marker a usable cue, but it also means that
ambient plants, corpses, task cues, and task rewards must be separated by the
evaluator even though the brain sees the same generic food signal. The only
existing food kinds are Plant and Corpse (`sim-types/src/lib.rs:175-180`).

### Neural state is larger than hidden activation

Brain evaluation reads previous inter-neuron activations before updating the
current hidden state (`sim-core/src/brain/evaluation.rs:55-99`). A fresh
expression initializes inter state and activation, action logits, activation
means, runtime weights, eligibility, and pending coactivation
(`sim-core/src/brain/expression.rs:3-44`, `153-177`). Runtime plasticity runs
after commit and uses `energy - energy_at_last_sensing` as its modulator
(`sim-core/src/brain/plasticity.rs:18-31`, `321-345`). The persisted neural
memory surface therefore includes:

- inter-neuron `state` and activation;
- runtime synaptic weight;
- eligibility and pending coactivation;
- sensory/inter/action activation means and `means_initialized`;
- current sensory activations and action logits.

The corresponding serialized fields are explicit in
`sim-types/src/lib.rs:565-680`. A purported neural reset that only zeros hidden
activation is not a reset.

World reset does re-seed the RNG, reset turn and entity IDs, clear entities,
rebuild terrain/organisms/food, and freshly express each founder genome
(`sim-core/src/lib.rs:338-377`, `sim-core/src/spawn/organisms.rs:101-143`). A
task case should use this fresh-construction path rather than reuse an advanced
organism.

### Randomness and seed coupling

Action sampling is a deterministic hash of `(simulation seed, turn, organism
ID)` (`sim-core/src/turn/mod.rs:521-530`) and is passed into brain evaluation
before softmax sampling (`sim-core/src/turn/intents.rs:119-160`,
`sim-core/src/brain/evaluation.rs:121-147`). This is valuable: cue-paired cases
can use exactly the same random variate at every action. It is also a leak if
cue truth is derived from the simulation seed.

Normal world construction is unsuitable for the audit without a controlled
constructor. Terrain depends on the simulation seed
(`sim-core/src/spawn/world.rs:4-20`), initial positions are shuffled and facing
is random (`sim-core/src/spawn/organisms.rs:55-98`), food-tile layout depends on
the seed, and plant color/regrowth consumes the shared world RNG
(`sim-core/src/spawn/food.rs:13-35`, `120-127`, `142-173`). The task renderer
must not draw from `Simulation::rng`.

### Existing evaluator and ledger boundaries

The current evolutionary evaluator creates a normal
`Simulation::new_with_champion_pool(scenario.world, world_seed, pool)` and then
calls `sim.tick()` for the episode (`sim-core/src/evolution.rs:2270-2352`,
`2444-2447`). Its `CaseEvaluation` is a survival/ecology case, not a task case
(`sim-core/src/evolution.rs:2742-2765`). Conditional tasks need a separate
evaluator contract, not hidden mutations around this loop.

The current fail-closed ledger has only organism and food compartments
(`sim-types/src/lib.rs:804-843`). It treats plant spawns as exogenous energy,
food consumption as a debit/credit transfer, and predation/corpse retention as
explicit losses (`sim-core/src/turn/mod.rs:354-451`). Injecting or deleting task
food outside `Simulation::tick` would not be represented by those flows.

## 2. Canonical base task

### Controlled world

Every task context starts from a fresh simulation with this enforced profile:

- world width 32;
- exactly one founder, always organism ID 0;
- no walls (`terrain_threshold = 1`), no ambient food
  (`food_tile_fraction = 0`), no other organisms;
- predation and force-random-actions disabled;
- passive metabolism, body-mass metabolism, and action cost all zero;
- fixed initial organism energy and health;
- no ecological food seeding or regrowth;
- task renderer uses a domain-separated deterministic task stream and consumes
  no `Simulation::rng` draws.

These values are legal in the current config schema
(`sim-config/src/config.rs:130-159`, `386-421`). The controlled constructor must
assert the realized world, not merely trust the manifest.

### Episode schedule

Let `R > 0` be the energy of each gate token and let `d` be the blank delay.
The base episode is:

| Tick | Scene before sensing | Required event |
| --- | --- | --- |
| 0 | one zero-energy cue marker at relative FoodRay -1 for `FIRST`, or +1 for `SECOND`; forward cell empty | no task consumption |
| 1 through `d` | empty | no task consumption |
| `d + 1` | one `TaskReward(R)` directly ahead | `FIRST`: consume; `SECOND`: withhold |
| `d + 2` through `d + 4` | empty | no task consumption |
| `d + 5` | an identical `TaskReward(R)` directly ahead | `FIRST`: withhold; `SECOND`: consume |
| after `d + 5` | terminal | exactly one correct task reward consumed |

Both cue scenes contain exactly one marker at distance one and use the same food
ID. Both gate scenes contain exactly one positive-energy token with the same
kind, energy, relative position, visual, and ID schedule. Cue marker, gate token,
and ambient ecology must be distinct internal kinds/roles even though all
produce the same FoodRay signal. The zero-energy cue requires a dedicated task
spawn path because the current food helper rejects nonpositive energy
(`sim-core/src/spawn/food.rs:146-173`).

Success is the exact post-commit event vector:

- `FIRST = [ConsumedTaskReward, NoTaskConsumption]`;
- `SECOND = [NoTaskConsumption, ConsumedTaskReward]`.

Selected Eat logits/actions are evidence but are not the score. Consuming a cue,
ambient plant, corpse, wrong gate, duplicate token, or evaluator-created token
is an immediate case failure. No partial scalar reward is an acceptance result.

### External-state erasure

After every tick, including the cue tick and every blank tick, the task phase
must atomically:

1. remove/recover all task scene entities;
2. restore the organism to the context's canonical `(q, r, facing)` and repair
   occupancy consistently;
3. settle task energy back to the fixed baseline through explicit ledger flows;
4. normalize all non-neural action aftermath that could differ by cue (damage,
   last-action marker, task-related consumption counters, and the
   `energy_at_last_sensing` stash after plasticity has consumed the current
   tick's value);
5. preserve the entire `BrainState` in the treatment arm.

Age, organism ID, species, generation, turn, config, and action-sampling stream
are identical within each cue pair. Age increments after commit
(`sim-core/src/turn/snapshot.rs:42-45`) but is not a current receptor; it may
serve as a cue-independent clock. The two delays and explicit gate scene are
needed to prevent a policy tuned to one fixed tick from masquerading as a
conditional response.

The normalizer must operate through a simulation-owned method. Directly editing
`q/r/facing` without updating occupancy violates the consistency invariants
checked in `sim-core/src/grid.rs:86-125`.

## 3. Exact finite-state task representation

Represent each finite task as a deterministic, total, multi-rooted Mealy
machine:

`T = (Q, C, root, A, emit, delta, effect, terminal)`

where:

- `Q` is a finite set of explicit one-tick states;
- `C` is the finite set of start contexts/cue values and `root: C -> Q`;
- `A` is a fixed, ordered event alphabet derived from commit facts, including
  `NoTaskConsumption`, `ConsumedTaskReward(role)`, and `InvalidTaskEvent`;
- `emit(q)` is an abstract relative scene: task entity role/kind, ray or relative
  coordinate, energy-token role, and visual role; it contains no absolute pose
  or allocated entity ID;
- `delta(q, a)` is total and deterministic;
- `effect(q, a)` is the exact score, escrow transfer, and invalidity vector;
- `terminal(q)` is either nonterminal or an exact terminal verdict.

Durations are expanded to explicit one-tick states. The renderer is a separate
pure function of `(canonical task, state, context pose, deterministic ID
schedule)`; rendering choices cannot alter task identity. `TaskRuntime` stores
the canonical task hash, current quotient-state ID, initial context, escrow,
token ledger, and accumulated actual events inside the serialized world. There
must be no evaluator-only mutable truth that can drift from the world.

### Validation and exact canonicalization

Canonicalization is fail-closed and exact:

1. Reject an empty context/root set, missing transition, nonfinite or negative
   token, invalid relative scene, context-dependent physical action alphabet,
   dangling state, terminal state with effects, or task flow that can overdraft
   escrow.
2. Remove states unreachable from every context root.
3. Compute the coarsest strong bisimulation by partition refinement. Initial
   colors include terminal verdict and abstract emission. Refinement signatures
   include the exact `effect(q,a)` and the successor partition for every event
   in fixed order.
4. Quotient the machine by that bisimulation.
5. Alpha-normalize context, cue, token-role, and response-role symbols. Treat
   fixed physical meanings (Eat, no consumption, energy amount, relative
   geometry) as colored constants, not renameable symbols. Use exact canonical
   colored-graph labeling; equivalently, enumerate admissible symbol
   permutations, run rooted BFS in fixed event order, and select the
   lexicographically least encoding. An optimization may use
   individualization/refinement, but it must fall back to an exact result rather
   than a heuristic hash.
6. Encode the quotient as canonical CBOR and hash it with SHA-256.

Two hashes are required:

- `execution_hash`: strong bisimulation with every tick preserved, used for
  deterministic replay and artifact identity;
- `semantic_hash`: weak/stutter bisimulation additionally hides a blank state
  only when it emits nothing, has zero effect for every event, and cannot
  observe or alter task tokens. This prevents delay padding and chains of no-op
  states from receiving behavioral-novelty credit.

The report must retain the original/reachable/strong-quotient/stutter-quotient
state counts and a machine-checkable old-to-canonical state map. A new state
count or execution hash alone is never novelty evidence.

### Semantic capability rank

For each decision boundary, compute the Myhill-Nerode equivalence classes of
cue histories under their required future event vectors. Report:

- number of pairwise distinguishable histories `H`;
- memory lower bound `ceil(log2(H))` bits;
- required output-vector length;
- retention horizon separately (never add it to semantic rank);
- concrete distinguishing suffix/witness for every claimed split.

An accepted capability increment requires a higher memory/output dependency
rank, causal intervention success, and retention of all ancestor tasks. It
cannot be obtained by alpha-renaming, unreachable states, bisimilar copies, or
blank-delay padding.

## 4. Fixed-escrow energy contract

The base task begins with exactly `E0 = 2R` in a task-escrow compartment. It
never tops up. Each gate stages one `R` token from escrow to task food. An
unconsumed token is recovered from task food to escrow after commit. A consumed
token follows the ordinary food-to-organism debit/credit path and is then
reclaimed from organism to escrow before the next sensing pass, so Energy cannot
carry which gate was eaten.

Let the physical compartments before/after a tick be organism `O_b/O_a`, food
`F_b/F_a`, and escrow `E_b/E_a`. Add four explicit internal transfers:

- `x_EF`: escrow to staged task food;
- `x_FE`: unconsumed task food recovered to escrow;
- `x_EO`: optional direct escrow grant to organism (zero in this task);
- `x_OE`: organism energy reclaimed to escrow after task consumption.

Extending the current ledger equations gives:

```text
O_expected = O_b - passive - action_cost
               + food_credit + predation_credit
               - unrecycled_removed - predation_prey_removed
               - corpse_source_removed
               + x_EO - x_OE

F_expected = F_b - food_debit
               + plant_spawn + corpse_spawn
               + x_EF - x_FE

E_expected = E_b - x_EF + x_FE - x_EO + x_OE
```

The existing removal adjustment and predation/corpse retention terms remain in
the total equation. All four task transfers cancel from `O + F + E`; task
staging must not appear as `plant_spawn_energy`. Transfer columns use measured
compartment deltas (promoted to `f64`), not nominal `R`, and participate in the
ledger tolerance scale.

Per-token conservation must also close exactly within tolerance:

```text
staged_task_food = consumed_task_food + recovered_task_food + standing_task_food
```

For the base episode, terminal standing task food is zero, terminal escrow is
`2R`, and organism energy is the initial baseline. Any negative/nonfinite
escrow, top-up, unknown task token, residual outside tolerance, or token left in
occupancy is a hard failure. This accounting is necessary even though the audit
world sets ordinary energy costs to zero.

## 5. The 16-context intervention panel

The minimum audit panel is the full factorial:

| Factor | Values |
| --- | --- |
| cue | `FIRST`, `SECOND` |
| blank delay | 7, 19 ticks |
| action-sampling seed | 11, 29 |
| canonical pose | `P0 = ((16,16), East)`, `P1 = ((7,23), SouthWest)` |

This yields 16 contexts and eight cue-paired nuisance tuples. Within each tuple,
the two cases have identical config, genome, seed, organism ID 0, turn count,
action random variates, task entity count, entity-ID allocation schedule, token
energy, gate scenes, and pose. Only the tick-0 side-ray cue differs. `P1` is a
translation and rotation of `P0`; all task scenes are rendered relative to
facing.

The panel is a minimum causal audit unit, not sufficient multi-seed evidence for
the overall research claim. Evolution uses independently domain-separated
rotating training panels; development and sealed audit panels use disjoint
action-stream keys and poses. The 16-context audit is rerun on every claimed
champion and on every modest ecology/world perturbation panel.

### Required arms and interventions

1. **Neural treatment.** Preserve the complete `BrainState`; erase external
   state after every tick as specified above.
2. **Post-cue full-neural reset.** Immediately after tick 0 normalization,
   replace `organism.brain` with a fresh `express_genome(organism.genome)`.
   This resets hidden state/activation, sensory activation, logits, runtime
   weights, eligibility, pending coactivation, means, and initialization flag.
   The two cue worlds must then have identical agent-reachable state and future
   action samples. Also restore `energy_at_last_sensing` to the canonical energy
   baseline; it is outside `BrainState` but feeds the plasticity modulator.
3. **Forced replay.** Record the treatment action sequence for one cue and feed
   that exact sequence through normal intent/commit semantics in its paired cue
   world. Do not score an offline logit. This directly measures an
   open-loop/replay policy.
4. **Brain swap.** Immediately before gate 1, swap the entire serialized
   `BrainState` between the cue-paired treatment worlds. Body, world, turn, and
   future action samples are already paired. The two-gate response vector must
   follow the donor cue, not the host's hidden task truth.
5. **External-only reset audit.** Reapply pose/energy/entity normalization at
   every boundary while leaving neural state intact. This is the normal
   treatment, but its pre-sensing fingerprints are retained to prove that the
   solver cannot use position, facing, food remnants, or energy.
6. **Mechanism lesions (diagnostic).** One lesion clears recurrent
   state/activation while preserving runtime weights; the reciprocal lesion
   restores birth weights/eligibility/means while preserving recurrent state.
   These classify recurrent versus plastic memory. They are not substitutes for
   the full reset.

### Acceptance inequalities

For the base 16-context panel:

- treatment: at least 14/16 correct;
- at least 7/8 for each cue value, delay, action seed, and pose marginal;
- both members correct in at least 6/8 cue-paired nuisance tuples;
- full-neural-reset arm: at most 8/16 and zero tuples with both members correct;
- forced replay: at most 8/16 and zero tuples with both members correct;
- treatment minus full reset: at least 6 cases;
- brain-swap: donor-cue response in both directions for at least 6/8 nuisance
  tuples, with no host-truth advantage.

The reset/replay upper bound is not empirical. After cue erasure, a cue pair has
identical agent state, scene sequence, and deterministic action samples, so it
must produce one response vector. The required vectors `[Eat, Withhold]` and
`[Withhold, Eat]` are mutually exclusive; at most one member of each of eight
pairs can pass. A reset or replay score above 8, or any pair with both correct,
proves an evaluator leak or broken pairing and invalidates the run.

## 6. Cheat matrix and required blockers

### Position and facing

**Cheat:** turn or move on the cue tick, then read the side effect through later
FoodRay/ContactAhead geometry. **Blocker:** canonicalize pose and occupancy after
every tick, render every scene relative to each of two transformed poses, and
assert paired pre-sensing external fingerprints. Canonicalizing only once before
gate 1 is insufficient because the organism can rewrite body state during the
delay.

### Energy and other organism scalars

**Cheat:** non-idle action cost, cue consumption, or evaluator reward changes
Energy; plasticity also reads within-tick energy change. **Blocker:** cue is
side-ray and zero-energy, ordinary costs/metabolism are zero, task reward comes
from fixed escrow, and all reward energy is reclaimed before the next sensing
pass. Normalize damage, health, last action, and task counters. Retain the
plasticity update as neural state, but never retain an organism-energy cue.

### Age, turn, and open-loop timing

**Cheat:** a recurrent oscillator or action-sampling sequence emits a memorized
action at one fixed gate tick without storing the cue. **Blocker:** explicit
identical gate scenes act as queries, delays 7 and 19 vary timing, and each cue
is fully crossed with the same turn/action stream. A clock may locate a gate but
cannot choose between the paired contradictory responses.

### Food count, order, IDs, visuals, and regrowth

**Cheat:** cue variants allocate a different count/order/ID, consume plant-color
RNG differently, or leave different regrowth schedule entries. **Blocker:** no
ambient ecology; dedicated cue/reward spawn path with no RNG or regrowth; exact
equal entity counts and ID schedule; same energy/visual for both gates; task
roles classified separately in commit. As a diagnostic, permute otherwise
unobservable task IDs and require unchanged results.

### Deterministic RNG

**Cheat:** derive cue from world seed, task seed, genome index, or an RNG draw
whose consumption differs between cues. **Blocker:** cue is an explicit crossed
factor, every action seed occurs with every cue, task generation uses a
domain-separated counter/hash, and rendering consumes no simulation RNG. Store
the per-tick action sample in the trace and assert equality within each pair.

### Action timing

**Cheat:** inject a gate after sensing, score a selected Eat that never
consumed, or leave a gate for an extra tick. **Blocker:** render after energy
capture but before lifecycle/sensing; observe the actual dedicated task-food
consumption in commit; settle/remove the scene before the ledger; assert each
state emits for exactly one tick.

### Stale neural state

**Cheat:** reuse the same advanced organism between contexts, or call a partial
reset that leaves learned weights/eligibility/means. **Blocker:** fresh
Simulation per context and fresh expression for the reset arm. Hash every field
listed in `BrainState` before cue, after cue, before each gate, and after reset.

### Genome/task-seed correlation

**Cheat:** evolution learns `genome hash -> cue` or the task generator uses the
same RNG stream as mutation/world generation. **Blocker:** full cue cross
product for every nuisance tuple, domain-separated task IDs, rotating training
keys, and disjoint sealed audit keys. No genome hash, population index,
generation, or scenario ID enters cue selection or the observable scene.

### Evaluator injection and energy laundering

**Cheat:** mutate foods, organism energy, pose, or task truth between `tick()`
calls, outside consistency and ledger accounting; classify a task reward as a
plant spawn. **Blocker:** serialized `TaskRuntime` and simulation-owned
pre-sense/observe/settle phases inside the canonical tick; dedicated task food
kinds; escrow columns in every energy row; fail on any unaccounted mutation.

### ID-order and population artifacts

**Cheat:** use organism ordering/tie resolution or another organism as external
memory. **Blocker:** exactly one organism with ID 0, no births/opponents, and a
fixed task entity schedule. This also removes the normal shuffled founder layout
and all movement contention.

## 7. Implementation hooks

No task code was added by this audit. A sound implementation needs these narrow
hooks:

1. Add `sim-core/src/task.rs` containing canonical task manifests,
   canonicalization, `TaskRuntime`, controlled renderer, event observer,
   settlement, and task evaluator. Keep this separate from survival
   `CaseEvaluation`.
2. Add an optional serialized task runtime/escrow field to `Simulation` plus a
   `new_task_case` constructor. The constructor realizes one founder and asserts
   the empty controlled arena; it must freshly express the supplied genome.
3. Split task participation in `Simulation::tick` into explicit phases:
   capture `O/F/E`; task pre-sense render; normal lifecycle/intents/resolution;
   normal commit; task observe-commit; age/plasticity; task settle and external
   normalization; consistency; extended ledger. The task transition uses the
   commit event, never a selected action.
4. Extend `CommitResult`/`consume_food` in
   `sim-core/src/turn/commit.rs` to classify dedicated `TaskCue` and
   `TaskReward` events. They do not schedule regrowth and do not increment plant
   or prey metrics.
5. Add a dedicated spawn/remove path in `sim-core/src/spawn/food.rs` that allows
   zero-energy cue markers, uses deterministic task IDs/visuals, and never
   consumes `Simulation::rng`.
6. Extend `EnergyLedgerRow` in `sim-types/src/lib.rs` with escrow before/after,
   the four task transfers, escrow residual, token-supply residual, and revised
   total residual. Update every wire `Api*` type and normalizer if the row or task
   trace is exposed through server/web protocol.
7. Add a separate task evaluation bundle in `sim-core/src/evolution.rs` with
   train/development/sealed panels, full context matrix, ancestor retention, and
   lesion results. Do not overload survival objective fields or reuse an
   advanced world across contexts.
8. Extend `sim-cli neat` with canonicalize, audit, and single-context trace
   operations. Every operation writes the canonical task, context manifest,
   genome, trace, energy rows, and lesion matrix under `artifacts/`; stdout may
   summarize but is not the evidence store.
9. Make `sim-evaluation` ingest task result datasets post-hoc, preserving the
   repository's raw-facts-first analysis model. Selection code may read the
   fail-closed pass/fail result, but reports are derived from persisted facts.

## 8. Required artifacts and reproducible interface

The implementation should make these commands real (names may be adjusted once,
but the capabilities and files are mandatory):

```text
sim-cli neat task canonicalize --task tasks/delayed-two-gate.json \
  --out artifacts/tasks/<execution_hash>/canonical.cbor

sim-cli neat task audit --task tasks/delayed-two-gate.json \
  --genome artifacts/genomes/candidate.bin --panel conditional-16 \
  --arms treatment,brain-reset,replay,brain-swap,recurrent-lesion,plastic-lesion \
  --out-dir artifacts/task-audits/<run-id>

sim-cli neat task trace --audit artifacts/task-audits/<run-id> \
  --context cue=FIRST,delay=19,seed=29,pose=P1
```

Each audit directory must contain:

- source commit/executable hash, full config, genome bytes/hash, and canonical
  task CBOR plus both hashes;
- context matrix with cue, delay, action seed, pose, expected event vector,
  entity-ID schedule, and domain-separation keys;
- per-tick raw trace: canonical task state, rendered scene, exact sensory vector,
  action sample, action logits/probabilities/action, actual commit event,
  pose/energy before and after normalization, brain hash, and escrow transfers;
- extended energy-ledger rows and per-token supply reconciliation;
- all treatment/reset/replay/swap/lesion outcomes, paired differences, marginals,
  and acceptance inequalities;
- canonicalization certificate: state maps/counts and distinguishability
  witnesses;
- a directly loadable world snapshot at cue, pre-gate 1, and pre-gate 2 for at
  least one passing and one failing context, enabling `inspect`, `brain`, and
  `decide` verification.

Aggregate JSON without the per-context and per-tick facts is not acceptable
evidence.

## 9. Fail-closed acceptance checklist

Reject a candidate, task, or run if any item fails:

- task validation, exact alpha normalization, strong quotient, stutter quotient,
  or canonical hashing does not complete;
- an alpha-renamed, unreachable-state-padded, bisimilar-state-padded, or
  blank-delay-padded task is admitted as semantic novelty;
- controlled world contains a wall, ambient food, second organism, regrowth
  entry, or unexpected RNG draw;
- cue-paired context differs in ID schedule, action sample, turn, non-neural
  pre-sensing state, gate scene, or token energy;
- treatment does not meet every total/pair/marginal threshold;
- neural reset or replay exceeds the proven 8/16 ceiling or gets both members of
  any pair correct;
- brain-swap response follows host task truth instead of donor neural state;
- a task event score disagrees with actual post-commit consumption;
- task food is counted as plant/prey ecology, regrows, persists an extra tick, or
  can be consumed twice;
- escrow is topped up, negative, nonfinite, or fails its compartment residual;
- `staged = consumed + recovered + standing` fails for any token/tick/run;
- organism Energy, pose, facing, task counters, or food remnants differ within a
  pair before any post-cue sensing pass;
- a reset arm retains runtime weights, eligibility, pending coactivation,
  activation means, state, activation, logits, or cue-dependent scalar state;
- a task seed/cue depends on genome hash, population index, mutation RNG, world
  RNG, or simulation seed rather than the explicit crossed context;
- a solver passes a new capability task but loses any previously accepted
  ancestor task;
- the predecessor/knockout genome also passes the claimed new capability, or the
  candidate's advantage disappears on sealed action streams/poses;
- evidence consists only of reward, population, aggregate pillars, a single
  seed, or a visualization without inspectable action/consumption traces.

## 10. Is growing cue-response depth open-ended?

### Delay alone: no

For `FIRST` versus `SECOND`, increasing `d` only asks the controller to retain
one bit longer. The exact execution machine gains countdown states, but its
cue-history equivalence at the first decision remains two classes and its memory
lower bound remains one bit. A leaky unit, recurrent latch, or plastic weight can
solve every delay within its retention range without learning a new
sensory-motor strategy. This is precisely why delay is reported separately and
blank stutters are removed from the semantic hash.

### A genuinely rank-growing family

A stronger finite instance presents an `n`-bit cue sequence (left/right marker
per bit), waits, then presents `n` identical one-tick gate tokens. The required
Eat/Withhold vector is the delayed copy of all `n` bits, or a declared
noncompressible permutation of them. Across all histories there are `2^n`
pairwise distinguishable futures, so any deterministic finite controller needs
at least `n` bits at the decision boundary. Parity, majority, or a repeated
single cue do not qualify: those can have constant or logarithmically smaller
state despite a long input.

Each concrete `n` instance is finite and can use the canonical machine above.
The parametric family has no semantic-rank ceiling in its abstract
specification. The current genome representation can complexify to an enormous
but still finite `u32`-indexed hidden-node bound
(`sim-core/src/genome/mod.rs:5-8`), and every actual machine, world, float state,
and run is physically finite. Consequently the audit can establish observed
rank growth over a tail; it cannot prove literal infinite capacity.

For `n <= 4`, all `2^n` cue histories can be enumerated, but nuisance factors
still multiply that set. For larger `n`, use a deterministic balanced history
suite plus explicit Myhill-Nerode distinguishing pairs; do not mistake a
16-case sample for exhaustive history coverage.

### Why even rank growth is not the repository's open-endedness result

The delayed-copy generator, its cue alphabet, and its response rule all exist
from the beginning. Increasing `n` is an externally programmed curriculum. It
can demonstrate that selection retains and expands conditional memory rather
than diluting new capacity, but it does not demonstrate new ecology, new niches,
new interactive dynamics, or behavioral objectives not implicit in the early
task generator. It is therefore suitable as:

- a fail-closed capacity gate for topology/plasticity changes;
- a minimal criterion in a broader quality-diversity or coevolutionary loop;
- a causal audit that a novelty claim depends on neural memory.

It is not suitable as the headline proof of sustained open-ended
neuroevolution. That proof still requires tail-of-run behavioral/ecological
novelty and competence across seeds, with this task family serving only as one
anti-degeneration instrument.
