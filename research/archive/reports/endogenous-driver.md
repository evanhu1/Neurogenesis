# Endogenous-driver audit: public proof-carrying energy caches

Status: **UNPROVEN REOPEN CONDITION**, not a result. No tracked files were
changed and no open-endedness claim is made.

## Verdict

Every endogenous driver that leaves the current affordance space fixed is
blocked:

- One-step empowerment is bounded by the action channel:
  `I(A_t; S_{t+1} | S_t) <= H(A_t) <= log2(6)` even with predation enabled.
  Longer-horizon empowerment in this finite, fixed world can delay convergence,
  but cannot create new affordance types. It also rewards easy controllability
  (turning/moving in empty space) rather than ecological competence.
- Prediction error/surprise has an immediate degenerate optimum: hash-driven
  predation outcomes, self-induced oscillation, or other agents' stochasticity.
  It does not require useful energy capture.
- Fixed-width signaling was tested more strongly than an information bonus:
  experiment 0019's dense, brain-controlled, zero-sum display contest was active
  and load-bearing, yet competence peaked and declined by 1M while complexity
  turned over.
- Predator-prey Red Queen dynamics were tested in experiment 0018. Complexity
  continued, but as competence-reducing bloat.
- Energy-conserving in-world reproduction was present throughout the archived
  21-experiment campaign. Restoring reproduction alone merely restores the
  already-falsified substrate; current main intentionally has no births.
- Fixed compositional resources (0020) learned a two-step chain but converged to
  a lower plateau. Persistent construction (0021) became wall spam and collapsed
  competence.

The smallest materially new reopen condition therefore needs all four of:

1. an affordance grammar whose realized programs can grow structurally;
2. an endogenous adversary that keeps those programs relevant;
3. an energy-conserving payoff that makes successful use competence, not drift;
4. protected controller growth plus historical audits, so complexity cannot buy
   novelty by diluting or forgetting the working policy.

The concrete candidate below is the smallest mechanism I found satisfying those
requirements. Removing any one component reduces it to one of the archived
failure families.

## Candidate algorithm: public proof-carrying energy caches (PPEC)

### Ecological object

Add a **nonblocking artifact overlay** (not an `Occupant`, so it cannot fragment
the world like 0021's walls). Every successful plant consumption splits its
existing energy exactly:

```text
consumer receives       (1 - rho) * plant_energy
new/extended cache gets rho * plant_energy
```

`rho=0` is byte-identical baseline. A cache is owned by the producer genome's
public protocol fingerprint, persists on the consumed cell, and can be opened by
any organism. Thus the dense food path, rather than sparse predation, creates the
interaction. The producer's clonal cohort benefits by later recovering its
escrow; rivals benefit by cracking and stealing it. Nothing is created by the
mechanism.

An organism adjacent to a cache enters a protocol interaction through a separate
`Interact` response head. The artifact publishes both its complete protocol and
a deterministic challenge stream. Correct responses advance the public protocol;
acceptance transfers the stored energy. Wrong responses reset progress and pay
the ordinary action cost.

### Public, structurally growing protocol

Add to `OrganismGenome` a public straight-line NAND circuit:

```rust
ProtocolGenome {
    input_arity: u32,                 // can grow with structural mutation
    gates: Vec<ProtocolGateGene>, // stable innovation IDs
    output_a: ProtocolWire,
    output_b: ProtocolWire,
}

ProtocolGateGene {
    innovation: InnovationId,
    left: ProtocolWire,
    right: ProtocolWire,
    enabled: bool,
}
```

NAND is a deliberately tiny universal grammar. `input_arity` is not capped:
add-input mutations extend the public challenge, so the union of expressible
functions is not the finite Boolean-function set of one fixed input width. The
challenge inputs are a public bit stream derived deterministically from `(world
seed, artifact id, attempt ordinal, local occupancy sketch)`. The cache serializes
its enabled circuit and challenge bits through the protocol receptors. The
correct two-bit response is the circuit output. There is no secret key or
lineage-ID oracle: a rival can in principle solve every cache from public
observations.

NEAT mutates the circuit by add-input, add-gate, rewire, output-repoint, and
parameter-free enable/disable operations using the same run-owned innovation
registry used for brain structure. Crossover aligns protocol gates by
innovation. Program length is not itself rewarded.

Add-input/add-gate mutations start unreachable and therefore leave the expressed
protocol unchanged. They may only become expressed through a later output/rewire
mutation after compatible controller capacity exists. This is the protocol-side
analogue of protected complexification; a one-step mutation that simultaneously
breaks the owner's solver would otherwise create an impassable fitness valley.

At cache creation, `protocol_build_cost * reachable_gate_count` is dissipated
from the producer's cache allocation. Each proof attempt debits
`protocol_op_cost * reachable_gate_count` from the attempting organism, never
from the cache; otherwise a rival could win by denial-of-service draining. Before
archive admission, the circuit is canonicalized and dead gates removed.
Therefore appending unreachable/no-op gates cannot improve defense, cannot count
as novelty, and reachable gates cost both producer and solver. A longer protocol
must actually protect or recover more escrow to survive selection.

### Non-diluting neural interface

Do not append raw cache channels to the ordinary locomotor softmax. Add a tagged
protocol module with serial-bit/index/challenge receptors and two response heads.
New protocol nodes begin with zero output gain and cannot affect locomotion. The
module is evaluated only while adjacent to a cache, but its neurons still pay the
normal metabolic cost. A heritable gate can recruit it once useful. This is the
minimum protection against the perception/action dilution seen in 0018/0019 and
must be ablated.

Runtime plasticity receives explicit response-success and energy-transfer credit
only after the current plasticity timing contract is repaired. Failed/correct
response facts must remain separate from the reward signal so the observer
cannot manufacture learning.

### Evolutionary loop

Evaluate every genome symmetrically in mixed-founder worlds against:

- contemporaries;
- a deterministic sample of historic protocol creators;
- a deterministic sample of historic solver genomes.

No evaluator-funded opponent renewal is allowed. If renewal is later required,
all respawn energy must come from a finite arena escrow debited at turn 0.

Selection is feasibility-first, then Pareto/QD within the feasible set. A genome
is feasible only if all of the following hold on the fixed audit panel:

1. ordinary plant capture and absolute survival exceed frozen baseline floors;
2. its cohort recovers nonzero own-cache energy;
3. it captures nonzero foreign-cache energy;
4. its public protocol is self-solvable on multiple contexts;
5. the energy ledger closes;
6. it has no worse lower-tail fixed-panel competence than its parent/archive
   incumbent beyond the prespecified tolerance.

The archive stores **paired genomes and causal traces**, not endpoint action
fractions. An entry contains the public protocol, solver, actual cache event
sequence, and energy-flow graph. A new entry is admitted only if it either:

- solves a competence-qualified protocol that the current solver archive could
  not solve, while retaining required historical competence; or
- supplies a self-solvable public protocol that defeats current/historic solvers
  and causes an actual protected-energy advantage.

Syntactic circuit novelty, cache count, protocol length, and action entropy never
qualify on their own. Protocol novelty is distance between causal
challenge-response functions on a fixed development challenge distribution, not
between program hashes. Never-touched sealed challenges are used only for the
final generalization audit and cannot influence archive admission.

## Exact energy contract

Record, every tick, raw floating-point facts for:

```text
T = sum(organism energy) + sum(food energy) + sum(cache energy)

T(t+1) - T(t)
  = plant energy spawned
- passive metabolism
- ordinary action costs
- protocol construction costs
- protocol execution costs
  - cache decay loss
  - corpse-retention loss
  - starvation/removal overshoot loss
```

Plant spawning remains the sole external source. The following operations are
zero-sum transfers before their explicit loss term:

- plant -> organism + cache;
- cache -> opener;
- prey -> predator at the existing corpse-retention fraction;
- any future parent -> child transfer.

The automatic plant split uses only the just-consumed plant energy; any later
manual cache extension checks available post-action organism energy before
applying. Protocol
construction cost is deducted from the cache allocation before it becomes
available; proof execution cost is deducted from the attempting organism. Cache
release is
`organism += amount; cache -= amount` in the same commit. Cache decay destroys
energy; it never feeds plant regrowth. Resolution uses stable organism/artifact ID
order and a snapshot-then-apply delta buffer. A per-tick closure failure is fatal,
not a debug assertion. Report both absolute and relative residuals.

The current action-cost order can take an organism negative after lifecycle and
then let a same-tick Eat rescue it. PPEC must not silently change that baseline;
either repair it as a separately controlled engine-contract change, or include
the exact negative overshoot in the ledger.

## Exact code surfaces

- `sim-types/src/lib.rs`: `ArtifactId`, cache state/events, protocol genes,
  protocol receptors/response heads, world snapshot fields. If wire-visible,
  update Rust protocol structs and `web-client/src/types.ts`/normalizer together.
- `sim-config/src/config.rs`, `sim-config/config.toml`,
  `sim-evaluation/config.toml`: hidden policy switch, `rho`, decay, per-op cost;
  keep baselines synchronized.
- `sim-core/src/lib.rs`: serialized artifact overlay, ID allocator, deterministic
  fingerprint/canonical circuit storage.
- `sim-core/src/turn/mod.rs`: preserve canonical order. Sense a start-of-tick
  cache snapshot; resolve response/deposit/release after movement in Commit and
  before metrics; apply decay once after transfers. Never read partially updated
  caches.
- `sim-core/src/turn/intents.rs` and `commit.rs`: protocol intents, stable-ID
  resolution, plant split, transfers, event facts.
- `sim-core/src/brain/{sensing,topology,expression,evaluation,plasticity}.rs`:
  protected module and response credit.
- `sim-core/src/evolution.rs`: protocol innovation/mutation/crossover,
  compatibility term, symmetric historic crossplay, feasibility vector, paired
  archive, checkpoint artifacts. The current selection objective is survival and
  current novelty is only six action fractions plus coverage/time-to-food; neither
  is adequate.
- `sim-metrics/src/{schema,ingest,ledger,intervals}.rs`: per-tick energy sources,
  sinks, transfers, cache/protocol event facts, right-censor-safe summaries.
- `sim-views` and `sim-cli`: `energy-ledger`, `caches`, `protocol-trace`, and
  per-organism cache energy won/lost/recovered.
- `sim-evaluation`: Parquet schema/report for the same raw facts. Do not derive a
  second metric implementation.

Current anchors motivating these surfaces:

- canonical tick/no births: `sim-core/src/turn/mod.rs:153-202`;
- lifecycle energy removal: `sim-core/src/turn/lifecycle.rs:18-50`;
- action debit and food transfer: `sim-core/src/turn/commit.rs:60-75,143-180`;
- predation transfer: `sim-core/src/turn/commit.rs:235-323`;
- only fixed actions/sensors/genome today: `sim-types/src/lib.rs:183-213,
  339-365,446-482`;
- current novelty descriptor: `sim-core/src/evolution.rs:775-800`;
- archive admits any descriptor difference over `1e-9`:
  `sim-core/src/evolution.rs:2108-2134`;
- current mutation has only brain weights/nodes/connections:
  `sim-core/src/evolution.rs:3673-3737`.

## Required causal trace

For each archived behavior, persist an ordered trace of:

```text
(tick, organism, cache, public_program_hash, challenge_index,
 observed_bit/index, response_bits, accepted/reset,
 cache_energy_before/after, organism_energy_before/after,
 local_food/organism context, selected locomotor action)
```

Derive a deterministic causal behavior graph with nodes for observations,
responses, transfers, movement, foraging, and counterpart protocol. Novelty is
graph distance after removing IDs and absolute coordinates, evaluated by
canonical replay on fixed development contexts. It only counts when the graph
includes a successful energy-flow path and the fixed competence floors hold.

Mandatory inspections for an early and late champion:

- `inspect`, `brain`, and `decide` at the cache approach, first response,
  correction after failure, successful release, and subsequent foraging;
- protocol-state trace proving responses depend on published challenge bits;
- code-bit permutation intervention: performance must collapse when semantics
  are permuted while occupancy, energy, and timing are preserved;
- extra-gate knockout: late capability must fall when post-early protocol gates
  are removed;
- locomotor-module-only and protocol-module-only counterfactuals.

## Adversarial controls

### Novelty versus drift

1. Freeze the exact founder repertoire: all mutation, crossover, structural
   protocol mutation, and plasticity off. Rotate training layouts. After every
   reachable fixed trace is observed, competent archive additions must be
   exactly zero. If it keeps growing, the observer is measuring context drift.
2. Preserve the same cache split, objects, sensors, and action costs, but set
   cache release efficiency to zero so a successful opening dissipates rather
   than credits energy. This sham must not show the treatment's qualified
   discovery or capability trend.
3. Shuffle selection scores within generation deterministically. Diversity may
   remain, but qualified cache energy capture and sealed generalization must not
   rise.
4. Force random actions. High response entropy or cache contact is not skill.

### Capability and metric gaming

5. Fixed four-gate protocol cap versus structural add-gate treatment. If both
   plateau together, structural growth is not doing work.
6. Public-code permutation preserving bit marginals/timing. If performance holds,
   agents are exploiting ID/order/context rather than interpreting protocols.
7. Replace every public circuit with an equal-length constant-output circuit. If
   the headline holds, circuit complexity is decorative.
8. Hold energy payoff constant while replaying traces. Longer handshakes must not
   count as more capable solely because they contain more events.
9. Track plant capture, ordinary survival, main-policy action effectiveness, and
   cache capture jointly. No cache metric can compensate for collapse elsewhere.

### Tail, seeds, and ecology

10. Checkpoints at generations 25/50/100/200/400; extend any positive 200 result
    to 400, then 800. Require nonzero qualified discoveries in every late window,
    not just cumulative archive size.
11. At each checkpoint, sealed crossplay against all earlier protocol/solver
    checkpoints. Require the lower envelope of capability across history to rise;
    current-opponent Elo alone is cyclic and insufficient.
12. Eight explicit world seeds, four development seeds, four never-touched sealed
    seeds. Report paired treatment-control intervals and per-seed failures.
13. Re-run sealed champions at food energy 7.2/8.0/8.8, regrowth interval
    180/200/220, and two world scales at constant density. Cache success cannot be
    tied to one threshold set.
14. Energy closure and population/plant supply trajectories are hard gates before
    interpreting novelty.

## Reproducible staged commands after implementation

The proposed flags/read commands do not exist yet; their spelling below is the
implementation contract.

### Stage 0: determinism and energy closure

```bash
cargo build -p sim-cli --release
./target/release/sim-cli new --seed 7 --scale 25,30 \
  --set protocol_cache_enabled=true --set cache_energy_fraction=0.35 \
  --set protocol_max_gates=4 \
  --out artifacts/research/runs/completed/open-ended/ppec/smoke-a.bin
cp artifacts/research/runs/completed/open-ended/ppec/smoke-a.bin \
   artifacts/research/runs/completed/open-ended/ppec/smoke-b.bin
cp artifacts/research/runs/completed/open-ended/ppec/smoke-a.metrics \
   artifacts/research/runs/completed/open-ended/ppec/smoke-b.metrics
./target/release/sim-cli run-to 5000 \
  --in artifacts/research/runs/completed/open-ended/ppec/smoke-a.bin
./target/release/sim-cli run-to 5000 \
  --in artifacts/research/runs/completed/open-ended/ppec/smoke-b.bin
cmp artifacts/research/runs/completed/open-ended/ppec/smoke-a.bin \
    artifacts/research/runs/completed/open-ended/ppec/smoke-b.bin
./target/release/sim-cli energy-ledger \
  --in artifacts/research/runs/completed/open-ended/ppec/smoke-a.bin
./target/release/sim-cli protocol-trace --last 200 \
  --in artifacts/research/runs/completed/open-ended/ppec/smoke-a.bin
```

Also compare intent thread counts 1 and 8 and require identical worlds, derived
metrics, cache states, and traces.

### Stage 1: bounded mechanism engagement

```bash
./target/release/sim-cli neat --seed 2718 --population 96 --generations 50 \
  --episode-horizons 500,1000,2000 \
  --world-seeds 7,42,123,2026,2718,31415,65537,99991 \
  --audit-seeds 61,79,97,113 --holdout-seeds 131,149,167,181 \
  --scenarios baseline,scarcity,sparse_search --workers 8 \
  --set protocol_cache_enabled=true --set cache_energy_fraction=0.35 \
  --param protocol_max_gates=4 --param protocol_add_gate_probability=0 \
  --param selection_strategy=feasible_protocol_qd \
  --out-dir artifacts/research/runs/completed/open-ended/ppec/fixed-cap
```

Require actual own-cache recovery and foreign-cache theft on every training seed,
sealed code-permutation sensitivity, and competence floors before structural
growth is enabled.

### Stage 2: structural treatment and shams

Run the same seed/config with:

```text
treatment: protocol_add_gate_probability=0.03
fixed-cap:  protocol_add_gate_probability=0, protocol_max_gates=4
no-payoff:  same cache split/objects/costs, cache_release_efficiency=0
frozen:     crossover_probability=0, every mutation probability=0
random:     force_random_actions=true
```

All conditions retain identical world seeds and audit/holdout partitions.

### Stage 3: tail and historical crossplay

```bash
./target/release/sim-cli neat --seed 2718 --population 128 --generations 400 \
  --episode-horizons 500,1000,2000 \
  --world-seeds 7,42,123,2026,2718,31415,65537,99991 \
  --audit-seeds 61,79,97,113 --holdout-seeds 131,149,167,181 \
  --audit-every 25 --scenarios baseline,scarcity,sparse_search --workers 8 \
  --set protocol_cache_enabled=true --set cache_energy_fraction=0.35 \
  --param protocol_add_gate_probability=0.03 \
  --param selection_strategy=feasible_protocol_qd \
  --out-dir artifacts/research/runs/completed/open-ended/ppec/tail

./target/release/sim-cli neat crossplay \
  artifacts/research/runs/completed/open-ended/ppec/tail/neat-RESULT.json \
  --checkpoints all --horizons 1000,2000,4000 \
  --world-seeds 131,149,167,181
```

Extend to 800 generations only if the 200-400 tail survives all controls. Then
run the food/regrowth/world-size perturbations and inspect late organisms/traces.

## Kill criterion / exact remaining gap

PPEC is **not** evidence until implemented. Kill it immediately if any of these
occur:

- fixed/frozen/no-payoff archives grow comparably;
- public-code permutation does not reduce energy capture;
- extra-gate knockout does not reduce late capability;
- protocol complexity rises while fixed-panel foraging/survival falls;
- qualified discoveries go to zero in the 200-400 or 400-800 tail;
- energy closure fails or cache splitting changes total plant supply;
- gains disappear under modest ecology perturbation;
- late controllers memorize creator IDs/program hashes and fail on independently
  evolved sealed public programs.

If PPEC fails, the endogenous family is exhausted at a precise boundary: this
substrate lacks a demonstrated expandable, payoff-bearing affordance grammar
whose interpretation generalizes without diluting the existing policy. A
materially new reopen would then need recursive morphology/major transitions or
another universal ecological construction substrate, not another scalar
intrinsic bonus, fixed signal channel, predation tweak, or cache parameter tune.
