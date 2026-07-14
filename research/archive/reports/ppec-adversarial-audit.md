# PPEC Stage-0 adversarial audit

Date: 2026-07-13

Status: **mechanism engagement established; evolutionary Stage 1 is NO-GO on
the current representation and interface.** This is not open-endedness
evidence.

## Reproducible positive result

Current `main` at `f4aef93` deterministically splits consumed plant energy
between the eater and a serialized nonblocking public cache, resolves queued
responses after ordinary commit, and closes organism/food/artifact transfers in
the fail-closed tick ledger. The three-seed, eight-context Stage-0 artifact and
an independent execution are byte-identical:

```text
artifacts/research/runs/completed/open-ended/ppec-stage0-final/
  ppec-mechanism-1783940882587-66432.json
SHA-256 71f7cc440c27e93cebaa90cf829fff0202169e9f176be8814cf0906a4ce744c1
result fingerprint 53b338f61174fdf17d83bc5371e539a72d4c11d8143892f980d9f89be192131b
```

```bash
target/release/sim-cli ppec-mechanism \
  --run-seeds 7,42,123 --contexts 8 --persistence-ticks 3 \
  --cache-fraction 0.35 --interaction-cost 0.25 \
  --out-dir artifacts/research/runs/completed/open-ended/ppec-stage0-main-replay
```

The artifact reports own and foreign evaluator-supplied answers at `48/48`,
code permutation `9/48`, challenge permutation `0/48`, artifact knockout
`0/48`, and random response `12/48`. All recorded tick ledgers close. The
artifact correctly sets `open_endedness_demonstrated` to `false`.

## P0 failure 1: the passing controls make energy

For one attempt with cache energy `E`, release efficiency `eta`, success
probability `p`, and interaction cost `c`, actor payoff is:

```text
EV = p * eta * E - c
```

At the shipped probe values `E=7`, `eta=1`, and `c=.25`, direct aggregation of
the artifact's ledger rows gives:

| Arm | Gross release | Cost | Actor net |
| --- | ---: | ---: | ---: |
| own protocol | 336 | 12 | +324 |
| foreign protocol | 336 | 12 | +324 |
| no payoff | 0 | 12 | -12 |
| code permutation | 63 | 12 | +51 |
| challenge permutation | 0 | 12 | -12 |
| constant 0 | 35 | 12 | +23 |
| constant 1 | 105 | 12 | +93 |
| constant 2 | 98 | 12 | +86 |
| constant 3 | 98 | 12 | +86 |
| random response | 84 | 12 | +72 |
| artifact knockout | 0 | 0 | 0 |

Thus every constant policy and the random policy are profitable. The current
gate calls a control "strongly reduced" at acceptance `<=24/48`; at exactly
that bound a blind policy would still earn `+3.25` per attempt.

A matched probe at cost `2.5` fixes only the pooled statistic:

```bash
target/release/sim-cli ppec-mechanism \
  --run-seeds 7,42,123 --contexts 8 --persistence-ticks 3 \
  --cache-fraction 0.35 --interaction-cost 2.5 \
  --out-dir artifacts/research/runs/completed/open-ended/ppec-negative-blind-payoff
```

Artifact SHA-256:
`bc92ab07735caf5a8f381d3de222936b5511029f8933e6b1a94da728feb9bb97`.
Pooled random payoff becomes `-36`, every pooled constant becomes negative,
and exact own/foreign payoff remains `+216`. But seed 42 constant response 1
and seed 123 constant response 3 each still earn `+2` over their 16-trial
panels. Per-seed fixed-response safety for these sampled programs requires
`2.625 < c < 7`.

Pricing cannot solve the evolved-program case. The NAND grammar admits a
constant function with `p=1`; making that fixed answer lose requires `c>E`,
while making its identical "exact" answer win requires `c<E`. The requirements
are contradictory. A deterministic 10,000-program screen found constant
functions in `394/10000`, `85/10000`, and `23/10000` random programs at arities
2, 3, and 4. About 29--32% had one response above 50% of the exhaustive truth
table. Balanced challenge dependence must therefore be guaranteed by the
representation, not inferred from a sampled panel.

## P0 failure 2: fingerprints count neutral bytes

`public_protocol_for_genome` serializes raw lifecycle fields, node and
innovation identities, float bits, disabled edges, and action biases. The
fingerprint hashes every byte, although only the reverse dependency cone of the
final two NAND gates affects the answer.

Exhaustive reverse-cone analysis of the two shipped programs found:

| Program | Arity | Opcode bytes | Causal gates | Dead bytes |
| --- | ---: | ---: | ---: | ---: |
| `5c995e...` | 2 | 38 | 18 | 20 |
| `312208...` | 3 | 54 | 18 | 36 |

Flipping dead byte 6 in the first program preserves exhaustive outputs
`[2,3,2,1]` while changing the fingerprint from `5c995e...` to `ebeea2...`.
Flipping dead byte 7 in the second preserves
`[0,3,1,2,0,3,1,2]` while changing `312208...` to `cc5c9e...`. The second
program also ignores its third declared input. Consequently the current
"variable public protocols" gate accepts syntactic drift and behaviorally
silent bloat.

## P0 failure 3: no organism uses the channel

The organism action vocabulary remains Idle/turn/move/Eat/Attack, and its
receptors remain food/contact/energy plus optional organism/health signals.
There is no artifact receptor, response head, Interact, Construct, or Deposit
intent. Every successful response in Stage 0 is computed by the evaluator and
queued through `Simulation::queue_artifact_interaction`. Standard world
snapshots and evaluation datasets also omit the raw artifact facts needed for a
tail claim.

"Own" and "foreign" therefore mean only that the evaluator supplied the right
answer to a manually positioned organism. They do not describe evolved neural
behavior.

## P0 failure 4: persistence and computation are unbounded for the wrong reason

Every plant consumption automatically creates a cache. Caches can stack, never
expire, have no capacity or maintenance cost, clone the entire program, and
are cleared on `Simulation::reset`; they are neither organism-authored nor
ecologically inherited across evolutionary episodes. External callers can
queue unbounded requests, including several requests from one organism in one
tick. Sequential artifact-ID resolution lets an early release finance later
attempts in the same phase. Low organism IDs can repeatedly buy denial with a
wrong response. Saturating ID/ordinal arithmetic eventually aliases state.

This admits a fixed bank-and-harvest policy and memory/CPU denial, not a growing
behavioral repertoire. The tick ledger is real, but the experiment's plant
injection, teleports, energy rewrites, cache knockout, and answer computation
occur outside it and are acceptable only as explicitly labeled Stage-0
counterfactual setup.

## Stage-1 admission contract

No neural/evolutionary run is admissible until all of these hold together:

1. Dedicated canonical protocol genes, not raw organism-genome bytes.
2. Exact removal/quotienting of unreachable, alpha-renamed, duplicate, and
   behaviorally equivalent structure for the bounded representation.
3. Balanced challenge dependence guaranteed structurally; random, constants,
   lookup, and replay lose energy on every seed/config while exact use wins.
4. An organism-owned public receptor/response/construct path with at most one
   paid intent per organism per tick and no evaluator-computed answers.
5. Conserved material collateral for bytes/modules, storage, computation,
   maintenance, expiry, garbage collection, and a bounded queue/capacity.
6. Simultaneous snapshot-then-apply contention with ID rotations and no
   same-phase release subsidy.
7. Exactly one resource-preserving successor ecology, or paid reconstruction
   from immutable blueprints; audit forks may not feed duplicated energy back
   into selection.
8. Raw artifact/material/event evidence in sim-cli, the evaluation dataset,
   and per-organism traces.

Even a pass would establish only organism-owned use of a bounded public
language. A fixed interpreter would terminate that difficulty source. The
open-ended route still needs downstream, recursively composable ecological
relations whose causal use—not byte count, program size, or task depth—keeps
changing in late geometric blocks.

## Verdict

Retain PPEC Stage 0 as a deterministic conserved mechanism checkpoint. Block
the current NAND-byte proof-cache family as an evolutionary algorithm. Reopen
only after a materially different canonical, balanced, organism-owned, and
resource-bounded construction substrate exists. No tail block was reached;
`open_endedness_demonstrated=false` remains the only supported result.
