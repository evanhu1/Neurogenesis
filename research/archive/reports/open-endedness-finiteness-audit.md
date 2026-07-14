# Open-endedness finiteness audit

Status: tracked computational sanity audit, not an open-endedness result.
Audited against `main` at `8ae902c` (`Document TCPE feasibility gate`).

## Verdict

The current compiled substrate cannot literally produce an infinite stream of
pairwise nonrepeating behaviors under a fixed resource bound. Once the
executable, config, seed, RAM, and persistent-storage budget are fixed, the
complete deterministic evolutionary process is a finite-state machine. It must
eventually halt/fail or revisit a complete state; after a revisit its observable
tail is periodic. This is a formal obstruction to the strongest reading of
"unbounded" in the standing prompt, not evidence that a practical plateau must
occur at any accessible horizon.

The opposite conclusion is possible only after changing the model being
discussed. An abstract algorithm with unbounded integers and unbounded heap/disk
can have an infinite state space, even with a fixed finite action alphabet. That
idealization permits infinitely many temporal strategies and finite programs,
but it does not by itself provide indefinitely new *semantic affordances*.
Topology growth, larger worlds, longer tasks, and public program interpreters
can all reduce to a size/memory/time ladder after one general algorithm appears.

Finite multi-seed tail experiments remain scientifically useful. They can reject
stagnation over an observed range, distinguish adaptive novelty from drift, and
show increasingly capable behavior at successively later checkpoints. They
cannot demonstrate a literal infinite tail. The defensible empirical claim is
therefore "sustained discovery through the tested horizon under a stated
resource envelope," with a geometric sequence of later horizons and adversarial
controls, not a proof of unboundedness.

One materially new substrate mechanism that directly targets the semantic gap
is **persistent, energy-funded, recursively composable niche construction**:
organisms must be able to build public artifacts that persist into later
evolutionary episodes, expose generic observable ports, and become inputs to
further construction. That creates endogenous ecological inheritance and makes
new organism-artifact and lineage-lineage relations possible. It still cannot
defeat physical finiteness, but unlike another archive, controller, task, or
world-size ladder, it can expand the operational affordance space whose tail is
being measured. This audit does not prove that construction is the unique or
smallest possible mechanism: an unbounded temporal strategy over fixed
primitives, a morphology with new public interactions, or a synthesis of
protected learning and endogenous ecology could also satisfy an operational
tail contract and must be tested rather than excluded by assertion.

## 1. Three claims that must not be conflated

| Claim | Resource model | What follows | What does not follow |
| --- | --- | --- | --- |
| Formal implementation claim | Fixed executable and finite RAM/disk/output budget | The complete deterministic process has finitely many states; it halts/fails or is ultimately periodic. | It does not give a useful upper bound on when a repeat occurs. |
| Mathematical algorithm claim | Unbounded integers plus unbounded heap/tape and an infinite run loop | A process can store ever more history and emit infinitely many distinct finite objects or action histories. | Syntactic growth does not establish new ecological meaning, capability, or non-implicit novelty. |
| Empirical research claim | Finite compute, increasingly late checkpoints | A study can show novelty and competence continue through every tested tail block and survive controls. | No finite set of runs proves an infinite nonstagnating tail. |

The first claim applies to the repository as compiled. The second applies to a
mathematical extension, not to the current Rust types. The third is the only
claim that `sim-cli` and `sim-evaluation` can directly support.

### "Not implicit" also needs an operational definition

For fixed code, config, and seed, every future state is a deterministic
consequence of the initial state. In the strict computability sense, the full
run is implicit at time zero. Pseudorandomness does not change this: the world
stores a seeded `ChaCha8Rng`, and the action and predation samples are hashes of
finite seed/turn/ID tuples (`sim-core/src/lib.rs:78-90`,
`sim-core/src/turn/mod.rs:521-547`). Adding an external true-random source would
break the repository's determinism contract and would supply variation, not
adaptive capability.

The prompt's phrase "not implicit at early times" can still be made testable:

- the behavior is absent from the canonicalized early trace archive;
- frozen early organisms cannot produce it, including after extra evaluation
  time and across the new contexts;
- it depends causally on a later evolutionary or ecological innovation;
- its success transfers across seeds and modest ecology perturbations; and
- its novelty survives quotienting IDs, absolute coordinates, clock time, and
  stuttering or repeated no-op cycles.

That is discovery relative to the realized early repertoire, not a claim that
the deterministic program did not entail its own future.

## 2. Finite-state derivation for the actual executable

Let `X_t` be the complete behavior-affecting state at evolutionary step `t`:

```text
X_t = (
  every live Simulation,
  population and genomes,
  evolutionary RNG and innovation/species state,
  novelty/task/environment archives,
  persistent output needed by future selection,
  finite resource-manager state
)
```

Fix the compiled executable, config, seed, and finite resource envelope. Every
scalar stored by the process then has a finite bit representation, and every
vector/map/file has a finite maximum length. Consequently the set `S` of
possible `X_t` is finite. Determinism gives a partial transition function
`F: S -> S`: it is partial because the process can finish, panic, or run out of
resources.

If the run never terminates, the pigeonhole principle gives indices `i < j`
with `X_i = X_j`. Determinism then gives `X_(i+k) = X_(j+k)` for every later
`k`. Any trace, behavior descriptor, or rendered world is a function of those
states and is therefore ultimately periodic too. If storage grows until it is
exhausted instead, the stream is finite. Neither case is an infinite stream of
pairwise nonrepeating behavior.

This proof requires the **complete** process state. Looking only at a world
snapshot while allowing an external archive to grow would be an invalid finite
state argument. Conversely, declaring the archive to be "unbounded" while
running it on a fixed machine silently switches from the executable to the
mathematical idealization.

The state count is fantastically large, so this is a claim boundary, not a
short-horizon forecast. It does not establish that the current behavior
repertoire is small, that evolution will soon cycle, or that late-tail evidence
has no value.

## 3. Exact finite surfaces in the current substrate

### 3.1 World and simultaneous ecology

`WorldConfig` stores `world_width` and founder count as `u32`; resource,
metabolism, terrain, and policy values are `f32`, `u32`, or booleans
(`sim-config/src/config.rs:110-163`). A world has exactly `width * width`
occupancy slots (`sim-core/src/grid.rs:5-7`). The canonical width is 50, hence
2,500 slots, and it requests 100 founders (`sim-evaluation/config.toml:1-6`).

Founders are truncated to the number of open positions and assigned unique
cells (`sim-core/src/spawn/organisms.rs:55-63`). Each slot contains at most an
organism, food, or wall, and the consistency check equates the counts of those
entities with occupied slots (`sim-types/src/lib.rs:856-862`,
`sim-core/src/grid.rs:86-103`). Therefore, for fixed width `W`:

```text
live_organisms + live_foods + walls <= W^2
```

The world is a torus with six facing directions
(`sim-types/src/lib.rs:276-288`, `sim-core/src/grid.rs:45-70`). The entity
vocabulary is also fixed:

- two food kinds: `Plant` and `Corpse` (`sim-types/src/lib.rs:175-180`);
- one terrain kind: `Mountain` (`sim-types/src/lib.rs:325-328`);
- six selected actions including `Idle`: turn left/right, move, eat, attack,
  or idle (`sim-types/src/lib.rs:182-205`); and
- five baseline receptors or nine with predation: three food rays,
  contact-ahead, energy, three organism rays, and health
  (`sim-types/src/lib.rs:337-370`).

Vision distance is constrained to 1 through 10
(`sim-config/src/config.rs:428-433`). A longer action history can encode a rich
temporal strategy, but every immediate observation and physical operation still
has one of these fixed meanings.

Terrain and persistent plant tiles are deterministic functions of finite width,
seed, thresholds, and hidden seed mixes. The hidden policies themselves contain
only fixed `u64` seed mixes (`sim-config/src/config.rs:52-60`, `98-108`);
terrain is a fixed noise mask (`sim-core/src/spawn/world.rs:4-20`), and food
tiles are a fixed ranking of non-wall cells (`sim-core/src/spawn/food.rs:183-210`).
Plants regrow and corpses can be created, but no new resource type, terrain
operation, sensor, or action semantics evolves.

The evaluator also has no in-world births. It seeds a founder cohort once; the
turn's former spawn phase now only applies plasticity
(`sim-core/src/turn/mod.rs:232-238`). Evolution is explicitly the outer loop and
`Simulation` is a pure clonal evaluator (`sim-core/src/evolution.rs:1-7`). Thus
`Simulation` does not carry constructed niches or world/ecological state from
one evaluator episode into the next; inherited genomes and research archives
live only in the outer loop.

### 3.2 Machine numbers, IDs, turn, and RNG

Continuous-looking state is not continuous in the implementation. Each `f32`
has only `2^32` bit patterns (including exceptional encodings), and config and
genome sanitizers reject or repair nonfinite values where relevant. Organism
position is two `i32`s; energy, health, neural activations, weights, traces, and
visuals are finite-width floats. Organism, food, and species IDs are `u64`,
runtime neuron IDs are `u32`, and stable gene/innovation identities are `u64`
(`sim-types/src/lib.rs:5-30`, `607-719`, `772-780`).

The world turn is `u64` and advances with `saturating_add(1)`, so it stops
changing at `u64::MAX` (`sim-core/src/lib.rs:78-90`,
`sim-core/src/turn/mod.rs:244-256`). Organism ages also saturate
(`sim-core/src/turn/snapshot.rs:42-45`). Organism and food allocators use `u64`
and ordinary `+= 1` (`sim-core/src/spawn/organisms.rs:150-153`,
`sim-core/src/spawn/food.rs:176-179`); at exhaustion they panic under overflow
checks or wrap and collide without them. Neither behavior supplies unbounded
fresh identity. Raw IDs or an ever-changing displayed counter must therefore
never count as behavioral novelty.

The seeded `ChaCha8Rng`, ID counters, turn, organisms, foods, occupancy, regrowth
schedule, and metrics are fields of `Simulation`
(`sim-core/src/lib.rs:78-136`). The complete behavior-affecting world and RNG
state is CBOR-serialized, with byte-identical deterministic continuation as the
documented contract (`sim-core/src/lib.rs:388-407`). Serialization makes the
world explicit; it does not enlarge its semantic state space.

### 3.3 Genome, runtime brain, and plasticity

Genome node and connection identity is finite. A `GeneNodeId` and
`InnovationId` is a `u64`; node roles consume two high domain bits and split
nodes use a deterministic 62-bit hash payload
(`sim-types/src/lib.rs:19-30`, `89-133`). Eventually a hash/identity namespace
must collide or saturate; it cannot name infinitely many distinct structures.

The old 1,000-hidden-node boundary is no longer the limit. Runtime hidden IDs
skip the five fixed action-ID slots, yielding the exact compiled upper bound:

```text
MAX_INTER_NEURONS
  = u32::MAX - 1000 + 1 - 5
  = 4,294,966,291
```

This is enormous but finite (`sim-core/src/genome/mod.rs:1-8`). Genome
sanitization deduplicates hidden IDs, truncates at that bound, fixes the action
bias vector to the five non-idle action neurons, drops invalid/nonfinite edges,
deduplicates endpoint pairs, and rejects colliding innovation groups
(`sim-core/src/genome/sanitization.rs:6-47`, `92-150`). For a finite node set,
the set of permitted directed endpoint pairs is finite, so the edge vector is
finite even though no useful small edge-count constant is configured.

A genome is expressed once at birth into dense sensory/inter/action arrays;
runtime evaluation does not add nodes (`sim-core/src/brain/expression.rs:3-45`,
`138-150`). Hebbian plasticity changes finite `f32` synapse state and activation
means, not the external action/receptor vocabulary or the runtime topology
(`sim-types/src/lib.rs:553-579`, `652-680`). Plasticity can produce genuine
lifetime learning within this state space; it does not make that space
unbounded.

`Vec` needs special care in this audit. Several vectors have semantic limits,
such as hidden nodes and action biases; others, such as an outer archive, have
no small configured limit. In the compiled executable a vector length is a
finite `usize` and allocation is bounded by the fixed resource envelope. In an
abstract unbounded-heap model, a sequence of finite-valued elements can itself
create infinitely many states. The finite-state conclusion therefore does not
come from seeing `Vec` in a Rust struct; it comes from the stated implementation
resource model.

### 3.4 The current evolutionary loop is configured to finish

The canonical NEAT config stores a `u32 generations` and `usize
population_size`; a run iterates exactly `0..config.generations`
(`sim-core/src/evolution.rs:34-38`, `997-1036`). The novelty archive is a growing
`Vec`, but additions occur only inside that finite generation loop
(`sim-core/src/evolution.rs:1031-1036`, `1134-1136`). The run-owned innovation
registry uses a `u64 next` counter and finite maps; the counter saturates
(`sim-core/src/evolution.rs:845-869`). Species IDs are `u64` and likewise
saturate (`sim-core/src/evolution.rs:3160-3208`).

The optional curriculum does not create an infinite executable loop. Its level
is `u32`, and sparse-search world width is still `u32` with saturating arithmetic
(`sim-core/src/evolution.rs:1143-1158`, `1782-1829`). More importantly, it
changes quantitative scarcity and size according to an external curriculum
level. It does not add a new organism-created affordance.

Accordingly, today's `run_neat` can only return a finite collection of
generation summaries. Calling it repeatedly from an external driver moves the
finiteness question to that driver's complete state and resource budget; it does
not remove the boundary.

## 4. What changes under an unbounded-memory idealization

An idealization must change more than the machine's RAM label. To describe a
genuinely infinite algorithmic run, replace bounded `u32`/`u64`/`usize` counters
and IDs with mathematical integers, allow unbounded sequences/maps, and replace
the configured `for generation in 0..generations` with a nonterminating driver.
Under those assumptions the full state space need not be finite.

Several important consequences follow:

1. A finite action alphabet can generate countably infinitely many finite action
   strings. Fixed actions therefore do not, by themselves, prove that every
   temporally extended policy is one of a finite set in the unbounded
   idealization.
2. An archive or controller with unbounded memory can distinguish arbitrarily
   long histories. It may therefore emit a nonperiodic tail.
3. An unbounded task or genome grammar can generate infinitely many distinct
   syntax trees or finite-state machines.

None establishes open-ended semantic novelty. A single universal interpreter
may solve every later member of a fixed task language, after which increasing
program state or delay measures capacity rather than a new sensory-motor or
ecological strategy. The current TCPE design already records this exact failure:
its fixed computable grammar can become a universal-interpreter ladder, while
the physical affordance alphabet remains FoodRay/movement/Eat/energy transfer
(`research/archive/reports/solver-dependent-ecology-route.md:578-613`).

Likewise, an indirect encoding can emit ever larger networks while expressing
the same policy; a world can grow while requiring the same search rule for
longer; and a counter can produce a never-repeated trace with no adaptive
change. Unbounded description length is necessary for some formal forms of
growth, but it is not sufficient for the standing prompt's behavioral and
ecological claim.

## 5. Mechanism audit: expansion versus ladders

| Mechanism | What it can add | Exact terminal risk on this substrate | Expands semantic affordances? |
| --- | --- | --- | --- |
| Novelty search / MAP-Elites / QD | Pressure to retain different descriptors and competent niches | A fixed descriptor and fixed archive budget fill; an unbounded archive can drift through traces or IDs. | No; it selects among behaviors expressed through existing sensors/actions. |
| NEAT complexification | Memory, nonlinear computation, and longer temporal policies | Finite `u32` runtime namespace in the executable; under idealization, a controller-size ladder or behaviorally silent structure. | No external expansion by itself. |
| Indirect/generative encoding | Compact regular networks and scalable morphology *descriptions* | More generated structure can implement the same policy; a fixed decoded body/interface remains a size ladder. | Only if decoded morphology creates genuinely new public interactions. |
| Evolvable Hebbian plasticity / intrinsic information drives | Faster lifetime adaptation, exploration, memory use | Finite `f32` brain state; surprise/empowerment can reward noise, flicker, or predictable self-motion. | No; it changes use of existing signals. |
| Coevolution and predator-prey arms races | Frequency-dependent strategies and interactive lineage dynamics | Fixed rays, attack, health, resource kinds, and episode reset can cycle among a finite strategic repertoire. | It recombines current affordances; it is strong empirical pressure but not an expanding substrate. |
| Procedural tasks / TCPE / PowerPlay | Solver-dependent challenges and historical retention | Passwords, a universal interpreter, or memory/time/state ladders over one fixed public language. | Not while task effects compile to the same fixed physical alphabet. |
| World scaling / quantitative curriculum | Longer search, scarcer food, different layouts | External schedule, `u32` width, and the same strategy repeated over a larger instance. | No. |
| Persistent composable niche construction | New public structures, ecological inheritance, tools/signals, and lineage dependencies | Can still collapse to inert junk, one universal constructor, or artifact-count gaming; remains finite on real hardware. | Yes operationally, if effects are compositional, public, energy-grounded, and causally used. |

This classification is deliberately stricter than "increases the number of
possible controller states." Controller and task complexity are useful only
when they produce a new causal organization of behavior, resources, or
interactions. Otherwise they are universal-interpreter/size ladders.

## 6. One materially new substrate mechanism

The fixed resource/action/receptor affordance set is finite in the compiled
implementation, and its *primitive physical vocabulary* remains fixed even in
the ordinary unbounded-memory idealization. A generic, recursively reusable
construction channel is a concrete change that directly addresses this rather
than adding a long enum of hand-authored new objects. It is a falsifiable
candidate, not a minimality theorem.

### Persistent public constructor ecology

Add a persistent `Artifact` ecology with these minimum semantics:

1. **Generic manipulation.** Add one ordinary physical `Interact/Construct`
   operation. It spends organism energy/material to append or alter a module in
   a local artifact; success and failure are normal committed world events.
2. **Public generic observation.** Expose nearby artifact ports/state through a
   generic local receptor. No task ID, creator ID, archive index, hidden target,
   or evaluator truth is visible to the brain.
3. **Recursive composition.** A constructed module can store state, route a
   bounded public signal or resource, alter local traversal, or expose a port
   that later construction consumes. Artifacts must be composable graphs, not a
   fixed list of recipes selected by an integer tag.
4. **Ecological inheritance.** Admitted artifacts persist into later
   evolutionary episodes or seed descendant environments. A later lineage can
   exploit, modify, defend, or depend on a structure made by an earlier lineage.
5. **Closed accounting.** Construction debits an explicit material/energy
   compartment, artifact storage/transformation is ledgered, and no module can
   mint energy. Artifact effects must be reproducible from public serialized
   state.

Recursive composition is the material part. Merely adding `BuildWall`, a finite
recipe menu, colored tokens, or a bigger task bytecode creates another bounded
catalog. A generic constructor allows an admitted ecological object to change
the causal opportunities available to subsequent constructors and organisms.
That supports operationally new niches such as trails, gates, caches, signals,
tools, parasitism on constructed infrastructure, symbiosis, and division of
labor without pre-enumerating each relation.

This mechanism still does not make a fixed machine literally infinite. Under
the mathematical unbounded-storage idealization, however, recursive public
construction can grow the realized causal interaction graph rather than only
the length of a solver's private program. Under finite compute it gives a
meaningful target for sustained-tail evidence.

### Minimum causal admission gate

An artifact lineage should count as a semantic affordance expansion only if all
of the following pass on disjoint seeds:

- a later organism/lineage gains capability in an ecology containing the
  artifact;
- deleting the artifact or replacing it with a canonical inert structure
  removes that capability;
- freezing the creator lineage before construction prevents the same niche from
  appearing;
- the consumer transfers to alpha-renamed, translated, and independently
  rebuilt instances, ruling out IDs and absolute coordinates;
- the artifact's exact energy/material ledger closes;
- the causal interaction graph contains a new canonical relation, not merely a
  larger module count or longer wait; and
- old frozen organisms and a universal-constructor baseline fail the new
  relation while the admitted descendant retains the historical repertoire.

Required controls include no-persistence, no-construction-payoff,
artifact-shuffle, fixed-recipe, frozen-ecology, random-action, and
selection-frozen arms. If construction produces activity without downstream
adaptive use, it is niche-construction drift, not open-endedness.

## 7. What finite experiments can honestly demonstrate

The repository should reserve `open_endedness_demonstrated` for an operational
contract, because a finite experiment cannot make the literal claim. The
existing conditional pilot correctly fails this field closed and states that a
finite stage-budgeted run cannot establish an unbounded tail
(`sim-core/src/conditional.rs:542-560`).

For a defensible sustained-tail result, run multiple seeds to geometrically
increasing discovery or generation horizons, for example blocks ending at
`G, 2G, 4G, 8G`. Predeclare and report for every block:

- count of newly admitted canonical causal behavior relations, with raw
  per-organism action/pose/resource/artifact traces;
- retained competence on all historical relations and fixed audit ecologies;
- transfer to sealed seeds and modest food/terrain perturbations;
- selection-frozen and payoff-removed novelty controls to separate adaptation
  from drift;
- ID/time/position/stutter canonicalization and targeted brain/artifact
  knockouts to reject observer gaming;
- population and every physical energy/material compartment, guarding against
  activity created by an accounting change; and
- discovery rate and waiting-time trend in the *latest* blocks, not only a
  cumulative count dominated by early discoveries.

The strongest valid statement after such a run is:

> Across the tested seed suite and through horizon `H`, the process continued to
> admit causally new, retained, transferable organism-ecology relations with
> competence above frozen-selection and no-payoff controls, without a detected
> decline to zero discovery rate in the measured tail.

Extending `H` can strengthen or falsify that statement. It cannot turn the
finite observation into a theorem about all future time.

## 8. Consequence for the standing task

Under a fixed finite RAM/disk/output envelope, the prompt's literal
conjunction—an **unbounded** run, indefinitely nonrepeating behavior, and
discoveries not implicit at early times—has a proof-level obstruction. That is
the resource model of any one actual executable invocation, but it is an extra
assumption relative to an abstract unbounded algorithm. It cannot be used to
declare the algorithmic search exhausted. No finite experiment can prove an
all-time tail; it can establish only an operational sustained-tail result.

The exact open gap is therefore not "more tuning" or merely "more compute." It
is:

1. define open-endedness operationally as reproducible sustained-tail discovery
   under increasing finite horizons, rather than literal infinity; and
2. implement and test endogenous problem/affordance generation coupled to
   protected solver learning and historical retention. Persistent energy-funded
   recursive public construction is one concrete candidate, not a proven unique
   minimum. Any survivor must demonstrate that later capability depends on new
   causal ecological relations rather than on drift, IDs, a universal
   interpreter, or a size ladder.

Until both are done and survive the standing adversarial audits, the complete
open-endedness claim is not established.
