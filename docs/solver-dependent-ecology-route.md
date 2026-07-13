# Solver-dependent public ecology route

Status: design and first-falsification contract, not an open-endedness result.
Grounded in `main` at `f420a3e` and the current bounded PowerPlay artifacts.

## Verdict

The next materially distinct mechanism should not be a deeper conditional-copy
task or a larger list of resource motions. It should be a hybrid of PowerPlay
and POET:

- multiple probationary environment-solver pairs generate different stepping
  stones and exchange solvers;
- a task is generated from the causal trace of an actually executed prior
  environment-solver pair, not from generation number or a fixed ordered
  curriculum;
- task semantics are public, canonical finite-state programs funded by one
  fixed endogenous energy escrow;
- only a consolidated solver that passes the new task and every archived task
  becomes a qualifying checkpoint.

I call the route **trace-compiled public ecology (TCPE)**. It is implementable
on the present substrate and is structurally different from the bounded pilot.
It is not yet a complete answer to open-endedness. A fixed public program
language has two serious terminal modes: coevolved password lookup, or a
universal interpreter after which later tasks are only a memory/time size
ladder. The first vertical slice must try to falsify those modes before any long
run is justified.

The design borrows only the retention principle from PowerPlay—new solver,
previous solver failure, and retention of old tasks—and the paired branches and
cross-environment transfer principle from POET. The repository's admission
rules, energy accounting, and causal interventions remain stricter than either
generic framework.

Background: [PowerPlay](https://arxiv.org/abs/1112.5309) and
[POET](https://arxiv.org/abs/1901.01753).

## 1. Evidence floor and the exact current blocker

### What the bounded pilot established

The current `sim-core/src/powerplay.rs` already provides several reusable
pieces:

- disjoint 16-context search and admission suites and fixed 14/16 versus 2/16
  gates (`sim-core/src/powerplay.rs:37-127`);
- every-checkpoint failure on the proposed task, protected residual solver
  search, every-task replay, and exact module knockout
  (`sim-core/src/powerplay.rs:342-515`);
- a constant 10-energy episode budget divided among stages, custom episode
  closure, and independent engine-ledger residual capture
  (`sim-core/src/powerplay.rs:1037-1197`);
- per-tick action, pose, food, energy, and completion traces.

The hardened artifact at
`artifacts/research/open-ended/powerplay-hardened/powerplay-1783929960884-83609.json`
contains this sealed matrix:

| Solver | depth-1 task | depth-2 task |
| --- | ---: | ---: |
| depth 0 | 0/16 | 0/16 |
| depth 1 | 16/16 | 2/16 |
| depth 2 | 16/16 | 15/16 |

At depth 3, all archived solvers were at most 2/16 and search reached 16/16,
but disjoint admission was 12/16; the candidate was correctly rejected. Across
the admitted path, the largest custom resource residual was `2.3841858e-7`, the
largest organism transfer residual was `3.0517578e-5`, and the largest engine
ledger residual was `1.0251998901367188e-5`, below the recorded engine
tolerance.

That is useful sequential evidence, but the task creator is explicitly finite:
`grammar()` is exactly 3 motions x 3 distances x 6 relative directions = 54
stage choices (`sim-core/src/powerplay.rs:652-670`), and config rejects depth
above four (`sim-core/src/powerplay.rs:78-84`). Stage selection merely finds
which member of that menu the checkpoint solvers fail
(`sim-core/src/powerplay.rs:672-733`).

### What the conditional audit established

The delayed conditional audit shows why increasing delay or copying more cue
bits is not the next mechanism. A binary delayed choice requires paired cue
interventions, complete external-state erasure, full neural reset/replay, and
brain swap before it is even valid memory evidence
(`docs/conditional-foraging-adversarial-audit.md:337-408`). Longer delay remains
a one-bit task, and even the rank-growing delayed-copy family is an externally
programmed curriculum rather than an ecological novelty source
(`docs/conditional-foraging-adversarial-audit.md:611-658`).

TCPE retains that audit as a control: public task descriptions are cue streams,
so their meaning must survive alpha-renaming and counterfactual presentation,
and the response must follow neural state rather than body/world residue.

### Current hooks versus missing hooks

`ProtectedResidual` already seals a controller, projects mutations into a
residual-only space, and restores the exact predecessor on knockout
(`sim-core/src/progressive.rs:23-119`). `enforce_retention` already checks a
complete archive once per task, and `verify_extension_effect` already requires
unique contexts, a strict pass gap, and different behavior traces
(`sim-core/src/progressive.rs:227-408`). These are directly reusable after task
identity changes from numeric `task_id` to canonical semantic hash.

The missing surfaces are material:

- no canonical public task machine or serialized task runtime exists;
- task food is still ordinary `FoodKind::Plant`, staged between `tick()` calls,
  and covered by a custom after-the-fact closure rather than an escrow
  compartment in every engine row (`sim-core/src/powerplay.rs:1061-1095`,
  `1259-1291`);
- `EnergyLedgerRow` still has only organism and food compartments
  (`sim-types/src/lib.rs:804-843`);
- task/environment genomes and paired archives do not exist in the ordinary
  evolutionary loop;
- the organism has only the fixed action and receptor enums and no task ID or
  generic program channel (`sim-types/src/lib.rs:175-205`, `337-370`).

## 2. The TCPE object model

### Public task

Each task is a deterministic, total multi-rooted Mealy machine of the exact form
defined by the conditional audit:

```text
T = (Q, C, root, public_event_alphabet, emit, delta, effect, terminal)
```

The physical alphabet is deliberately small:

- emissions: blank, public bit 0, public bit 1, separator, and one escrow-backed
  reward token at a relative location;
- events: selected physical action, move success/failure, task-token
  consumption/non-consumption, and invalid event;
- effects: task-state transition, fixed escrow transfer, and terminal verdict.

Before execution, the task streams a self-delimiting encoding of its minimized
public machine through relative FoodRay cues: left is bit 0, right is bit 1, and
the center ray is a separator. Pose, energy, task entities, and all other
external aftermath are normalized after every description tick. The task hash,
archive index, creator lineage, genome hash, absolute coordinates, and seed are
never rendered or passed to the brain.

The executable task then emits its public scenes and observes normal commit
events. Only actual escrow-backed token consumption is competence. The program
preamble makes the rule inspectable in principle; without it, a paired
creator/solver could encode a private action password that no foreign solver
could infer.

Every context presents an independently alpha-permuted state numbering and a
different legal serialization order. Thus a solver must follow the graph and
physical symbols, not memorize raw bytes. Fixed physical meanings—left/right
cue bit, Eat, movement, and energy transfer—are not alpha-renamed.

### Environment-solver pair

An active pair is:

```text
Pair {
  task_semantic_hash,
  task_execution_hash,
  resident_solver,
  source_pair_hashes,
  source_trace_slices,
  creator_energy_certificate,
  obligation_hashes,
  status: probationary | consolidated,
}
```

Probationary pairs may specialize, branch, and exchange solvers. They are search
stepping stones, not discoveries. A pair becomes consolidated only when one
protected solver passes its task and every previously consolidated task on the
sealed panels. This preserves POET-style branching without weakening the
PowerPlay all-history condition.

There is one monotone `UniversalCheckpoint U_k` after `k` qualifying tasks. A
new pair can be explored by any resident solver, but task `T_(k+1)` is admitted
only together with `U_(k+1)` satisfying:

```text
passes(U_(k+1), T_i) >= 14/16  for every i <= k+1
passes(U_j,       T_(k+1)) <= 2/16  for every archived j <= k
knockout(U_(k+1)) == U_k exactly
passes(knockout,  T_(k+1)) <= 2/16
```

This is intentionally more expensive than lineage-local retention. Sampling
old tasks may guide search, but cannot decide admission.

## 3. Solver-dependent task creation

### Source material

A creator may use only normalized causal slices from actual archived executions:

```text
(public observation, task emission, organism action, commit outcome,
 task transition, escrow flow)
```

IDs, absolute coordinates, raw RNG values, task hashes, and unobserved task
state are removed. Each slice retains the exact source task, solver, context,
tick interval, and energy-flow certificate as non-observable provenance.

Counterfactual source slices are legal only when produced by replaying one
different physical action through the same normal Simulation commit path. The
creator cannot fabricate a response or payoff edge that has never occurred in
an actual or one-action-counterfactual trace.

### Three recursive creator operators

The first implementation needs only three operators:

1. **Trace split.** At a visited transition, split on a public observation or
   commit event witnessed in another context. One branch retains the parent
   suffix; the other uses an actually witnessed counterfactual suffix.
2. **Trace splice.** Join a prefix from one accepted pair to a suffix from
   another at an interface with the same normalized scene, escrow balance, and
   organism-energy baseline.
3. **Causal lift.** Move a public distinction earlier in a trace into the
   preamble and require its witnessed successful response after an
   observationally identical interval. This turns an embodied dependency into
   a neural conditional without choosing a depth from generation number.

The operators recurse over their own accepted outputs. No operator receives
generation, target depth, desired state count, or an ordered task-list index.
Specific offspring therefore depend on the evolving archive's behavior and
cannot be replayed from a solver-independent curriculum schedule.

After every operation, exact validation/minimization occurs before evaluation.
If the changed transition is unreachable, bisimilar, stutter-only, or absent
from every successful/counterfactual causal slice, the candidate is discarded
without running a solver.

### Creator search objective

Task search uses only mutable search contexts. For a candidate `T`, define:

- `old_full(T)`: maximum full pass count among every consolidated solver;
- `old_prefix(T)`: maximum normalized causal-prefix completion among those
  solvers;
- `learn_B(T,S)`: pass-count improvement after exactly `B` solver-search
  generations from seed solver `S`;
- `transfer_B(T)`: best `learn_B` from a solver belonging to a different pair
  lineage minus the best result from the resident and universal seeds;
- `size(T)`: number of states/transitions after strong and stutter quotient.

A candidate is creator-feasible only if:

- canonicalization and all escrow proofs close;
- it has a valid trace provenance DAG;
- no consolidated solver exceeds 4/16 full success on search contexts;
- `old_prefix(T)` is in `[0.15, 0.85]`, providing a causal gradient rather than
  an impossible blank wall;
- at least one equal-budget probe has `learn_B >= 6/16`;
- semantic hash and minimized causal slice are absent from the archive.

Among feasible tasks, use deterministic lexicographic ordering:

1. larger `max(transfer_B, learn_B)`;
2. smaller `old_full`;
3. larger minimum semantic distance in the all-solver outcome vector;
4. smaller `size(T)`;
5. lexical semantic hash as the final tie-break.

The size term is a simplicity pressure, never a novelty reward. The task is
frozen before full solver optimization. Disjoint admission contexts are not
read during task generation, probe adaptation, or solver selection.

### Why this is solver-dependent rather than a hidden curriculum

The decisive control is a frozen solver archive. With the same creator
operators, trace sources, layouts, and seeds, disable every solver mutation and
transfer. Once all counterexamples to that finite solver set have been admitted
or rejected, qualifying tasks must stop. If task admission continues through
alpha variants, longer descriptions, or changing seeds, the generator is
measuring its own syntax/context drift.

## 4. Solver optimization and cross-pair transfer

### Solver objective

For a fixed candidate task, every solver seed receives the same mutation RNG
budget, episode contexts, and protected-residual capacity. Rank candidates
lexicographically by:

1. minimum pass count over every consolidated task;
2. candidate-task pass count;
3. candidate causal-prefix completion;
4. net escrow-backed task energy captured;
5. fixed-ecology survival and plant-capture floor;
6. fewer added active nodes/connections.

A scalar weighted sum is forbidden because it lets a large new-task reward buy
forgetting. Search may stop early only when every hard threshold is met.

### Transfer sweep

For each probationary task `T_i`, evaluate in fixed canonical order:

- its resident solver;
- the current universal checkpoint;
- every active/consolidated solver from another task lineage.

Run both direct transfer and exactly `B` generations of protected adaptation.
The complete source-solver x target-task matrix is persisted, including failed
transfers. The best solver may replace the resident only if it does not lose any
obligation in the union of source and target histories.

A transfer is a qualifying stepping stone only if:

```text
post_B(foreign solver, T_i)
  - max(post_B(resident, T_i), post_B(universal, T_i)) >= 6/16
```

and the foreign solver originates in a different task branch. Repeating the
same comparison after knocking out the module added on the foreign source task
must remove at least 4/16 of that advantage. Otherwise the transfer edge is
mere lucky initialization, not reuse of an evolved stepping stone.

The solver that ultimately consolidates a task still must replay the complete
global archive. Pair-local success and transfer alone never qualify as a
capability discovery.

## 5. Password, task-ID, and padding rejection

Old-solver failure is dangerously easy to manufacture. Every proposed task and
solver must pass all of these before admission.

### No identifier channel

- No task ID, hash, archive index, creator/solver lineage, genome fingerprint,
  generation, or absolute pose enters task emissions, Simulation seed, action
  sample, or neural inputs.
- Changing all metadata IDs while preserving public semantics must leave world
  bytes after construction and every response trace identical apart from the
  metadata artifact.
- Replacing the task program while preserving a fake metadata ID must change
  behavior according to the public program, proving the ID is inert.

Numeric IDs can remain in host-side archive maps, but retention APIs should key
their obligations by semantic hash and assert a one-to-one metadata mapping.

### Exact semantic canonicalization

Reuse the conditional audit's two-level contract:

- validate, remove unreachable states, compute the coarsest strong
  bisimulation, quotient, alpha-canonicalize colored symbols, canonical-CBOR
  encode, SHA-256 hash;
- separately weak/stutter quotient pure blank zero-effect states for semantic
  novelty.

Store the original-to-quotient proof. An unreachable-state, duplicate-state,
renumbering, no-op transition, longer blank, or alternate serialization mutant
must have the same semantic hash and cannot enter creator evaluation.

### Public-semantics interventions

For every sealed context suite:

1. independently renumber states and reorder serialized transitions;
2. swap alpha-renamable cue symbols consistently in description and execution;
3. present an unseen legal serialization of the same canonical machine;
4. splice the same public causal operators into a held-out task assembled from a
   different creator lineage;
5. blank the public preamble while keeping execution timing and energy fixed;
6. permute one physical opcode in the preamble while keeping nuisance values
   fixed.

The solver must remain at least 14/16 on 1-4 and fall to at most 2/16 on 5-6.
Passing exact archived programs but failing 1-4 is password lookup. Passing 5-6
means the public program is decorative and another channel controls behavior.

### Transfer/generalization gate

A co-mutated creator and solver can memorize even a public circuit. Therefore a
new solver must additionally do one of:

- solve at least two held-out tasks from distinct creator lineages that use the
  newly claimed causal operator, with neither task seen during its mutation
  search; or
- provide the causal cross-pair transfer advantage defined above on one held-out
  target and retain all archived tasks.

Self-solution alone never admits a task.

## 6. Fixed endogenous escrow

Every task has exactly one payoff token `E* = 10` simulation-energy units,
independent of machine size, age, depth, or creator score. The token must be
funded by an actual source trace:

1. in a construction episode, the creator consumes ordinary plant energy;
2. a simulation-owned commit phase transfers exactly `E*` from creator organism
   energy into task escrow;
3. any canonical-machine construction cost is separately dissipated from the
   creator, never deducted from or added to `E*`;
4. the resulting serialized task snapshot and ledger row form the creator
   energy certificate;
5. evaluation contexts fork that exact snapshot, as ordinary deterministic
   counterfactual evaluations fork any world. No energy is added within a fork.

If the creator cannot fund `E*` plus construction cost, the task cannot be
proposed. The construction cost is
`c_state * reachable_states + c_edge * reachable_transitions` after quotient,
so padding cannot impose cost or claim complexity. The cost comes from ordinary
plant capture over ecological time; it is not an evaluator grant.

Task cues and descriptions carry zero energy. The terminal reward is staged
from escrow to dedicated `TaskReward` food. Consumption transfers food to the
organism through normal commit. An unconsumed token returns to escrow. No
intermediate payoff is released, so Energy cannot reveal hidden task progress
before the terminal decision.

Extend the physical ledger from organism `O` and food `F` to escrow `E` with
explicit transfers:

```text
creator deposit: O -> E
task staging:    E -> F
task capture:    F -> O
task recovery:   F -> E
build cost:      O -> sink
```

All transfers cancel in `O + F + E`; only ordinary plant spawn remains an
external source and build cost is an explicit sink. Every tick and terminal
certificate must satisfy:

```text
initial escrow = staged + locked
staged = consumed + recovered + standing
```

Negative/nonfinite escrow, top-up, token duplication, task reward counted as a
plant spawn, or residual above the engine tolerance aborts the run. The current
PowerPlay practice of writing a Plant between ticks and checking a custom final
sum is not sufficient for this route.

## 7. Minimal implementable vertical slice

Do not begin with a persistent cache ecology or a 400-generation run. The
smallest decisive slice is three canonical tasks, two task branches, and one
causal transfer.

### Slice A: task runtime and energy

1. Implement the canonical task machine, public bit preamble, dedicated
   TaskCue/TaskReward kinds, serialized runtime, and third ledger compartment.
2. Import the current admitted depth-1 and depth-2 PowerPlay traces as provenance
   only; compile their causal slices into canonical `T1` and `T2`.
3. Reproduce their existing pass matrix and fixed escrow under the new runtime.
   Any behavioral change, non-closing row, or dependence on old numeric task IDs
   blocks the route.

### Slice B: two probationary branches

1. From a successful `T1` trace, produce branch `A` with one trace split.
2. From a successful `T2` trace, produce branch `B` with one trace splice/lift.
3. Canonicalize every generated candidate and retain the complete rejected
   candidate table.
4. Optimize resident solvers independently on mutable contexts only.

### Slice C: transfer and consolidation

1. Generate frozen task `T3` by splicing one causal slice from A and one from B.
2. Run equal-budget protected adaptation from A's resident, B's resident, and
   the universal depth-2 checkpoint.
3. Require a foreign branch solver to exceed both other seeds by at least 6/16
   after the same budget, and require the foreign-source module knockout to
   remove at least 4/16.
4. Consolidate the winning solver and replay `T1`, `T2`, and `T3` on sealed
   contexts. Require `[>=14, >=14, >=14]`; every earlier universal checkpoint and
   the exact new residual knockout must be `<=2` on `T3`.
5. Run the complete identifier, alpha, preamble, held-out recombination, replay,
   fixed-ecology, and energy controls before accepting the single discovery.

This slice is large enough to expose the proposed mechanism—behavior-derived
task creation, parallel stepping stones, transfer, consolidation, and public
semantics—without pretending that three tasks are open-ended.

### Exact implementation surfaces

- `sim-core/src/task.rs`: canonical machine, exact minimizers/labeling, renderer,
  runtime, source-trace provenance, task event observer, escrow settlement.
- `sim-core/src/coevolution.rs`: pair archive, three creator operators, creator
  objective, probationary branches, transfer sweep, consolidation.
- `sim-core/src/progressive.rs`: semantic-hash obligation keys, union-history
  transfer evidence, foreign-module transfer knockout.
- `sim-core/src/turn/mod.rs`: task pre-sense, post-commit observation,
  post-plasticity normalization/settlement, and extended ledger inside tick.
- `sim-core/src/turn/commit.rs` and `spawn/food.rs`: dedicated zero-energy cue and
  escrow-backed reward paths with no plant metrics/regrowth/shared RNG.
- `sim-types/src/lib.rs`: task food kinds/events and escrow ledger columns. Only
  add wire-visible task types if server/web types and normalizers change in the
  same commit.
- `sim-cli/src/coevolution.rs`: one-shot `coevolve-public` run, canonical task
  inspection, pair matrix, transfer matrix, and single-context trace.
- `sim-evaluation`: post-hoc ingestion only after the slice passes; do not build
  a second task/ledger implementation.

## 8. First falsification gate

The first gate is **public transfer, not task count**. Kill or redesign TCPE
before scaling if any condition fails on three independent outer run seeds:

1. `T3` is not a new stutter-quotiented semantic hash with a reached new causal
   transition and exact provenance from both branches.
2. A foreign branch solver does not achieve the prespecified 6/16 equal-budget
   adaptation advantage, or its source-module knockout does not remove 4/16.
3. The consolidated solver does not pass all three tasks at least 14/16 while
   every prior checkpoint and exact residual knockout remain at most 2/16 on
   `T3`.
4. Performance falls below 14/16 on state renumbering, transition reordering,
   unseen legal serialization, or held-out cross-lineage recombination.
5. Performance remains above 2/16 when the public program is blanked or a
   physical opcode meaning is permuted.
6. Fixed ecology survival/plant capture falls below its paired noninferiority
   floor, or any task/creator escrow row fails.
7. A frozen-repertoire creator continues admitting semantic tasks after its
   reachable counterexamples are exhausted.

Condition 4 is the password test. Condition 2 is the stepping-stone test. If a
pair can self-solve but cannot generalize or transfer, the route is merely an
arbitrary code arms race and is not reopened by more creator mutations,
population, states, or generations.

Mandatory control arms for this first gate:

- no cross-pair transfer;
- fixed task set with the same number of solver evaluations;
- frozen solver/task repertoires with rotating contexts;
- zero payoff with identical cues/task objects;
- fixed controller capacity and exact residual knockout;
- task-ID/hash sham channel injected only into a diagnostic copy (it must make
  memorization easy, demonstrating that the real treatment removed the channel).

## 9. Artifacts required from the slice

Persist, rather than summarize:

- canonical CBOR, execution/semantic hashes, original-to-quotient maps, and
  stutter/bisimulation certificates for every proposed task;
- complete creator candidate table with source trace ranges, provenance DAG,
  feasibility vector, rejection reason, and no hidden admission-panel values;
- construction snapshot, creator plant-capture/deposit trace, fixed escrow
  certificate, every per-tick ledger row, and terminal token reconciliation;
- every solver x task direct score and equal-budget adapted score, including
  failures, RNG keys, and mutation count;
- resident/universal/foreign transfer comparison and causal module knockouts;
- full all-history checkpoint matrix and exact behavior fingerprints;
- per-tick public preamble, sensory vector, task state, action sample/logits,
  action/commit event, energy flow, pose-normalization, and brain hash;
- alpha/serialization/held-out-recombination/blank-preamble/opcode-permutation
  intervention results;
- fixed-ecology paired results and the no-transfer/fixed-task/frozen/no-payoff
  controls.

Suggested durable interface:

```text
sim-cli coevolve-public --seed 7 --pairs 2 --discoveries 1 \
  --search-contexts <16> --admission-contexts <16> \
  --out-dir artifacts/research/open-ended/tcpe/seed-7

sim-cli task canonicalize --in candidate-task.cbor --certificate
sim-cli coevolve-public matrix --in result.json
sim-cli coevolve-public trace --in result.json --task T3 --context 11
```

The spelling is provisional; the stored facts are not.

## 10. Is this task source genuinely open, or a disguised curriculum?

It is not a finite menu in the sense that blocks the current pilot. Trace split
and splice recurse over an archive that itself grows; the set of finite machines
has no configured state/depth ceiling, and candidate identity depends on actual
solver/environment traces. There is no generation-to-difficulty mapping or
prewritten ordered list. Operationally, this is a solver-dependent task source.

That does not prove open-endedness. Three exact gaps remain:

1. **Any fixed computable grammar implicitly defines a task language.** TCPE
   removes a finite menu and external schedule, but cannot make a formal claim
   that future tasks were absent from the universal grammar.
2. **A universal interpreter can saturate the language.** Once a controller can
   parse and execute every public machine, further state growth may be only a
   memory/time ladder. Old-solver failure then measures capacity, not a new
   ecological behavior. The semantic trace and transfer gates detect some but
   not all versions of this collapse.
3. **The physical affordance alphabet remains fixed.** Food rays, movement,
   Eat, and a single energy transfer cannot guarantee indefinitely new niches or
   multi-lineage ecological dynamics. A successful TCPE slice would justify a
   persistent public-cache ecology; it would not eliminate the eventual need
   for endogenous new observables/actions, recursive construction, morphology,
   or major transitions.

Therefore the rigorous claim available after the vertical slice is only:

> solver-dependent, energy-grounded, canonically novel task creation with one
> causally necessary cross-pair stepping stone and complete historical
> retention.

If the slice passes, run geometric tail blocks and controls. If it fails the
public-transfer gate, this route is exhausted at a precise boundary: a fixed
sensor/action substrate plus public finite programs produced passwords or a
static interpreter ladder, and the next materially new mechanism must change
the ecological affordance substrate itself rather than enlarge the task
generator.
