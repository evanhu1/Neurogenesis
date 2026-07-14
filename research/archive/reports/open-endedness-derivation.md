# Open-ended neuroevolution derivation

Date: 2026-07-13

This is a live, evidence-first derivation. It distinguishes substrate repairs,
bounded mechanism screens, and any result that could support an operational
open-endedness claim. No archive count, topology count, or scalar fitness trend
is accepted as behavioral novelty by itself.

## Adversarial acceptance contract

A task/solver pair is a qualifying discovery only if, on 16 deterministic
admission contexts:

- every earlier checkpoint solver succeeds on the new task at most 2/16;
- the new solver succeeds on the new task at least 14/16;
- the new solver succeeds on every historical task at least 14/16;
- alpha-renamed/isomorphic contexts remain at least 14/16;
- random/replayed actions and the new-module knockout are at most 2/16;
- a causal-semantics permutation is at most 2/16;
- the minimized causal slice sets a strict necessary-depth or policy-state
  record and transition knockouts reduce success;
- fixed-ecology survival and plant capture remain non-inferior;
- a fixed energy escrow closes under a per-tick compartment ledger.

The tail gate uses outer seeds `7,42,123,2026,2718,31415,65537,99991` and
blocks `[100,200)`, `[200,400)`, and `[400,800)`. At least seven seeds must
contain a qualifying rank record in every block, retain all historical skills,
and sustain at least half the preceding discovery rate. Treatment must beat
frozen-repertoire, uniform-selection, no-payoff, and fixed-task/fixed-capacity
controls in at least 7/8 paired seeds.

## Falsified families

| Family | Decisive evidence | Failure mode |
|---|---|---|
| Fixed scalar survival | Rounds 2-5 | passive runway/crowding; finite cutoff |
| Existing NSLC | archive 74/78 while endpoint novelty and competence fell | metric gaming and seed-dependent collapse |
| Frozen rotating-panel archive | 0/6/11/21 entries with zero mutation/crossover | context drift counted as novelty |
| NEAT complexification alone | causal structure, but fixed topology matched competence | complexity without growing demand |
| Contemporary/historical predation | short ladders then zero late predation energy | bounded opponent pressure |
| Evaluator-funded renewal | extinction, energy farm, or brittle threshold | energy-source trilemma |
| Archived indirect-encoding redesign | QD coverage 31.5 at 5k, 21.5 at 20k | terminal structural descriptor, no behavioral selection |
| Fixed intrinsic games | signaling/pursuit/composition/construction plateaued or gamed | bounded affordances |

See `approach-registry.md`, `qd-minimal/report.md`,
`plasticity-encoding/findings.md`, and `endogenous-driver.md` for the full
artifacts and reproduction commands.

## Mandatory substrate repairs

Seven commits establish an honest evidence floor:

- `f392a3f`: action-time intervals include live survivors; sidecar gaps cannot
  fabricate activity; continuous and save/resume runs have byte-identical world
  and metric hashes; tail windows are duration-weighted; aggregate summaries
  expose contributing seed counts.
- `0480b76`: within-tick plasticity reward uses the current sensing baseline;
  NEAT refuses forced-random behavior; hidden-node overflow cannot count in the
  genotype while disappearing from the phenotype; historical innovation IDs
  survive materialization.
- `003642f`: baseline and evaluation configs are schema-strict, byte-identical,
  and retuned away from universal early extinction; the default suite is eight
  explicit seeds.
- `117b8b0`: accepted controllers can be sealed as immutable recurrent prefixes;
  later candidates are projected into a residual-only search space, all-history
  task/controller/seed/behavior replays are SHA-256 fingerprinted, and runtime
  hidden IDs skip the stable action-ID island instead of truncating at 1,000.
- `d75b23c`: every simulation tick now closes signed organism and food energy
  compartments, the food-consumption transfer, and their total under a
  scale-dependent tolerance; non-finite values and unexplained sources or sinks
  panic instead of becoming research evidence.
- `7fa757d`: residual verification now rejects a mere behavior-fingerprint
  change; admission requires explicit enabled and knockout capability bounds and
  a strict enabled advantage.
- `20e2030`: progressive task, replay, and knockout panels reject duplicate
  trial seeds and malformed task/controller fingerprints, preventing a repeated
  context from masquerading as 14/16 evidence.

Post-repair probes:

- report interval 100, horizon 100: tail `(90,100]`, action effectiveness
  `0.082`; horizon 101: tail `(90,101]`, effectiveness `0.083636`. The one-tick
  partial no longer replaces the prior interval.
- a 1,001-hidden-node external genome materializes as exactly 1,000 nodes and
  one valid edge, while retained innovation `42` remains `42`.
- `force_random_actions=true` fails an ordinary NEAT run before evaluation.
- eight-seed tick-2,500 evaluation has metric coverage 8/8 for every axis:

| Seed | final pop | plant/action | action effectiveness |
|---:|---:|---:|---:|
| 7 | 4 | 0.1430 | 0.5650 |
| 42 | 14 | 0.1271 | 0.5147 |
| 123 | 7 | 0.1443 | 0.5897 |
| 2026 | 5 | 0.1026 | 0.4638 |
| 2718 | 11 | 0.1022 | 0.4271 |
| 31415 | 8 | 0.0845 | 0.3045 |
| 65537 | 11 | 0.1313 | 0.4433 |
| 99991 | 8 | 0.0973 | 0.3710 |

Artifact: `measurement-repair/eval-8seed-viable`.

The release `progressive_capacity_probe` materialized 1,501/1,501 encoded and
runtime hidden nodes with 3/3 edges; the high hidden runtime ID `2505` and action
ID `2000` were correctly ordered into inter/action partitions. Empty extension
worlds were byte-identical initially and after 64 ticks, illegal writes into a
sealed node were removed, and the stage-2 knockout exactly fingerprinted to the
stage-1 controller. The tightened deterministic action criterion passed 2/2
with the residual and 0/2 under exact knockout. This establishes non-diluting,
causally necessary capacity on a deliberately small criterion, not a stream of
adaptive gains or open-endedness.

The fail-closed energy ledger enforces, for each tick,

```text
O1 = O0 - M - A + Cfood + Cpred - U - Rpred - Rcorpse
F1 = F0 - Dfood + Pplant + Scorpse
```

and separately requires `Cfood - Dfood = 0`. Retention losses and signed removal
adjustments close the combined compartment equation. The tolerance is
`32 * f32::EPSILON * max(1, |O0 + F0| + sum(|flows|))`; organism, food,
transfer, and total residuals all hard-fail above it. Reproducible probes covered
foraging (`debit=credit=40`), plant regrowth (`source=60`), starvation with a
negative-energy removal adjustment, and lossy predation
(`removed=599.9000`, `credited=479.9200`, `loss=119.97998`), with zero reported
residuals. Duplicate 100-tick worlds hashed identically. A non-finite action
cost panicked on turn 1. Reset-time initialization, evaluator-only organism
injection, and the not-yet-implemented task escrow remain outside this tick
ledger and must be accounted explicitly by any expanding-task mechanism.

The workspace compiles, formats, and passes strict Clippy. The sole existing
test failure is the already-documented contradiction between the old
`lethal_attack_spawns_corpse_food_without_feeding_attacker` test and the current
consume-on-kill predation contract; this campaign did not change that ecology.

## Strongest mechanism under test

The remaining materially new family is a monotone ecology PowerPlay:

1. freeze a newly proposed, payoff-bearing task before solver search;
2. require all historical solvers to fail it;
3. add only protected residual controller capacity;
4. accept only after new-task mastery, all-task retention, causal knockout,
   semantic intervention, fixed-escrow closure, and fixed-ecology competence;
5. retain the full task x solver x context outcome matrix, not an averaged
   descriptor archive.

A sequential-resource pilot and the protected-capacity/energy-ledger components
were screened independently. The pilot (`234e4c0`, hardened through `b070d2a`)
uses ordinary Simulation ticks, a fixed 10-energy episode escrow, disjoint and
unique 16-seed search/admission panels, every-checkpoint novelty, every-task
retention, exact residual knockout, all 54 task-generator candidates, persisted
candidate genomes, and unambiguous before/after behavior traces. It admitted
depth 1 at 16/16 versus old and knockout 0/16, then depth 2 at 15/16 versus
prior solvers 0/16 and 2/16 and knockout 2/16. At depth 3, search found 16/16
but sealed admission was only 12/16; retained tasks were 16/16 and 14/16, so
both retention and same-suite causal gates reject it. Across 528 recorded
episodes, custom escrow residual was at most `2.38e-7`, organism closure error
at most `3.05e-5`, and the engine-ledger residual at most `1.03e-5` under its
tolerance. The current schema-3 artifact is
`powerplay-schema3/powerplay-1783930277214-28309.json`, SHA-256
`bb3554bdfcc9ce10174d71d8aeca72a9984a8414ea29b84654c210dc5c3188e0`.
Two independently executed schema-3 depth-2 runs were byte-identical, SHA-256
`2a821a881186f5439be84f66a29e9db9ceb9a8408e66890bbf377bd6e5bd4722`.

The audit found real false-evidence paths before accepting the bounded result:
mixed search/admission causal scores, repeated seed aliases, a positive
subnormal escrow that admitted two zero-payoff stages, ungated no-op integrity,
and mislabeled controller-coupled target drift. All now fail closed or are
named accurately. `food_energy=1e-45` rejects during validation and `1e-6`
rejects after a nominal capture produces zero observed organism-energy gain.

This is a falsification, not open-endedness evidence. The stage interpreter is
hard-capped at four and draws from a finite visible-target vocabulary; payoff
does not gate survival. More search or a larger cap would remain bounded novelty.
The active, materially distinct route is a delayed conditional program in which
cue information is removed from the body/world before an observationally
identical response phase, forcing recurrent policy state under a fixed escrow.

The independent audit in `research/archive/reports/conditional-foraging-adversarial-audit.md`
narrows what that route can prove. A valid memory claim requires complete
external-state erasure after every tick, full neural reset/replay and donor
brain-swap interventions, a third escrow compartment, and exact alpha plus
strong/stutter-bisimulation task hashes. Delay padding has constant one-bit
semantic rank. An `n`-bit delayed-copy family has a genuine `2^n`
Myhill-Nerode history rank, but it is still a predeclared curriculum: even a
causal pass is progressive-capacity evidence, not open-ended ecological novelty.

## Conditional-capacity result

The final delayed-copy capacity audit is
`conditional-program/final-v3/conditional-program-1783933209955-30868.json`,
SHA-256
`d9f53461c15fa92d491d89ac70c85d4bfb46cfde2c1ee9d7cd39b1ee11cedef1`.
An independent execution is byte-identical. The artifact embeds the exact
task/random/ecology configs, every proposed and knockout genome, all sensory
activations and action logits, deterministic action samples, core and task
energy rows, and all paired controls.

For each outer seed 7, 42, and 123, the rank-4 residual solves all 16 task
contexts; the preceding controller and exact residual knockout solve zero;
all eight complement pairs are jointly correct; donor-brain, reset, replay,
semantic-permutation, random, nuisance, and grouped-lesion gates pass. Every
one of the 64 scored response ticks per seed selects the unique raw-logit
argmax with minimum margin `0.6856479` at temperature `.01`, while deterministic
samples span `.009263218..98312557`. The task behavior is therefore causal
four-bit delayed memory, not a seed-assisted marginal softmax decision.

No candidate is admitted. The per-seed ecology gate requires survival ticks,
plant capture, and final energy to be noninferior in every one of four matched
ecology seeds. Seed 7's aggregate 27-to-27 plant tie concealed one 8-to-4 plant
regression and final-energy regressions in two pairs; seeds 42 and 123 regress
more broadly. Search and admission exhaust the same 16 rank-4 semantic
histories, so admission is correctly labeled a disjoint world/RNG/pose
replication, not unseen-history generalization. `open_endedness_demonstrated`
is false and the summary reports zero qualifying discoveries.

This blocks the delayed-copy route as an algorithm. It establishes isolated,
causal recurrent capacity but shows that the added residual is not safely
arbitrated away from ordinary ecology. Its formal `2^n` grammar also remains a
predeclared curriculum, and a 16-history panel cannot empirically certify more
than four distinguishable bits.

## Public-ecology zero-shot interface result

`research/archive/reports/tcpe-feasibility-audit.md` finds three contradictions in literal TCPE
Slice A: a meaningful preamble necessarily changes recurrent state/age/turn
and the action hash; the old two-stage trace pays two 5-energy plants rather
than one terminal 10-energy token; and one exact construction snapshot cannot
also be several nuisance-varied controlled templates. The engine O/F/escrow
work was therefore gated behind an evaluator-owned public-semantics probe.

The corrected probe artifact is
`public-preamble-probe/final-v2/public-preamble-probe-1783935214211-77625.json`,
SHA-256
`60aa935f570fbf45e13b21bd9aa10cb4c35b4f274a518130c89c082202462bc5`.
An independent schema-v2 replay is byte-identical. It reconstructs actual
PowerPlay checkpoints, freezes 16 new contexts, and gives meaningful, blank,
and cue-permuted arms the same 36 prefix ticks, action-hash schedule, body
resets, FoodId allocation, task horizon, and escrow path.

Zero of four evaluable seed x depth pairs passes the prespecified
meaningful>=14, blank<=2, permuted<=2 gate. Seed 7 scores `16/7/6` at depth 1
and `13/8/9` at depth 2 when full success requires every prefix deadline. Seed
42 scores `16/16/16` at depth 1; seed 123 scores
`16/16/6`. Seeds 42 and 123 never construct a depth-2 source checkpoint, so
only four of six requested pairs are evaluable. All 192 arms have the exact
54- or 72-tick shape, zero prefix consumptions, closed custom escrow, and zero
maximum core-ledger residual. The initial artifact's last-prefix-only success
scalar inflated seed-7 depth-2 blank from 8 to 10; this does not change the
zero-shot rejection but must not be used as the all-prefix statistic. There is
no executable branch-transfer pipeline in this probe.

This rejects zero-shot import of the current visible-resource controllers into
this unfamiliar public program interface. It does not falsify protected
public-aware training, a different encoding, or the solver-dependent ecology
family. A valid next gate must train only a protected decoder on a code-dependent
declaration, retain ordinary ecology, use blank/opcode/wrong-valid-program
controls on disjoint panels, and only then test causal foreign-branch transfer.

## Protected public-training result

The protected decoder gate was implemented and run after the zero-shot result.
It froze the accepted source controller, added a 12-node residual, and selected
on 16 mutable declaration contexts with simultaneous meaningful, valid-code-swap,
blank, polarity, and ordinary-retention requirements. Only a complete search
qualifier could touch one sealed admission panel; descendant reuse followed only
after that verdict.

Seeds 7, 42, and 123 each exhausted a population of 64 for 120 generations with
zero qualifiers. All three final best candidates reached only `9/16` meaningful
and `9/16` valid-code-swap success, below the required `14/16`. Seed 7 retained
ordinary behavior 16/16 but polarity was 5/16; seed 42 retained 14/16 with
blank/polarity 4/3; seed 123 retained 10/16 with blank/polarity 4/2. The sealed
panel remained untouched and descendant reuse was not attempted.

Artifacts and SHA-256 are:

- seed 7: `public-decoder-probe/final-v2/seed-7/`,
  `d1725183391098aedf5e0ca325d49be7b29f2dbaaca9e05efa97b8032b3bd354`;
- seed 42: `public-decoder-probe/final-v2/seed-42/`,
  `3234dfe3e45fb42dbd5a31294bc0a01bf07bd430957d419efaba90df26b6788b`;
- seed 123: `public-decoder-probe/final-v2/seed-123/`,
  `ba589804b8fee7db1263293130ed6bb63f2930919a163bb87616e10238568934`.

This blocks the current protected preamble route for the tested optimizer,
source seeds, encoding, and budget. It does not prove decoder-family
impossibility, but more budget on the same interface is not a materially new
mechanism and cannot authorize TCPE engine work.

## PPEC Stage-0 mechanism and hostile-audit result

Commit `f4aef93` implements a deterministic persistent public proof-carrying
energy cache. Ordinary plant consumption splits physical energy between the
eater and a nonblocking serialized artifact; queued responses resolve after
ordinary commit; cache release, loss, and interaction cost close in the
organism/food/artifact tick ledger. The three-seed, eight-context Stage-0
artifact has own/foreign evaluator-supplied success `48/48`, no-payoff
acceptance with zero release, code permutation `9/48`, challenge permutation
and artifact knockout `0/48`, and closed recorded ledgers. A current-main replay
is byte-identical.

Artifact:
`ppec-stage0-final/ppec-mechanism-1783940882587-66432.json`, SHA-256
`71f7cc440c27e93cebaa90cf829fff0202169e9f176be8814cf0906a4ce744c1`,
result fingerprint
`53b338f61174fdf17d83bc5371e539a72d4c11d8143892f980d9f89be192131b`.

The mechanism does not survive the required hostile audit:

- With cache energy 7 and cost `.25`, random response has actor net `+72` over
  48 trials; constant responses net `+23/+93/+86/+86`. The current reduction
  gate therefore rewards every supposedly degenerate control.
- Cost `2.5` makes pooled controls negative and exact use `+216`, but fixed
  policies remain positive on individual seed panels. No price can fix an
  evolved constant-output program: fixed-negative requires `cost > reward`,
  while exact-positive requires `cost < reward`.
- A deterministic 10,000-program screen finds constant functions in
  `394/10000`, `85/10000`, and `23/10000` random NAND programs at arities 2,
  3, and 4; roughly 29-32% have a majority response above one half.
- Reverse-cone analysis finds only 18/38 and 18/54 shipped opcode gates causal.
  Flipping a dead opcode changes the public fingerprint while preserving the
  exhaustive truth table; the arity-3 program ignores its third input.
- Organisms have no artifact receptor, neural response, Interact, Construct, or
  Deposit action. The evaluator computes every answer. Caches have no expiry,
  capacity, maintenance, GC, or cross-evaluation inheritance and allow
  unbounded externally queued work.

The complete audit is `ppec-adversarial-audit.md`. PPEC Stage 0 is retained as
a narrow positive mechanism checkpoint, but the NAND-byte proof-cache family is
blocked as an evolutionary algorithm. A canonical balanced reversible
two-bit program could support only 24 functions and would establish bounded
organism-owned channel access, not open-endedness.

## Procedural-ecology Stage-0 mechanism and grammar result

The final incompatible lane wraps ordinary simulation ticks with an
evaluator-owned, translation-equivariant ecology carrier. Stationary,
moving-front, and consumption-responsive policies receive the same 20-release,
200-energy per-case endowment. Translated, input-clamped, and duplicate arms
produce 24 cases across seeds 7, 42, and 123.

All 19 scoped mechanics gates pass over 48,000 ledger rows and 480 releases.
Duplicate artifacts are byte-identical; all recorded residuals are exactly
zero; clamping the public-equivalent consumption bit changes the responsive
placement trace; and physically identical stationary and clamped-responsive
cases share one physical hash while retaining distinct diagnostic hashes.

Artifact:
`procedural-ecology-stage0-final-v2/procedural-ecology-stage0-1783943744919-31422.json`,
SHA-256 `42faf58ce86eaf43dfeaa96b09c8ea9636a2414485c631ad0c0a6676af1a0d76`,
result fingerprint
`2c515fa5755af20a529f7800a8359ded7063699e368a9e2750b3515693facdfd`.

The artifact explicitly sets Stage 1 and open-endedness false. Its escrow is an
independent evaluator endowment in each arm; same-boundary reclaim/release
lacks intermediate checkpoints; occupied targets abort; consumption is read
through private world membership; the disabled hook is empty; and ordinary
Plant values have no canonical/procedural provenance. The fixed eater makes
the seed suite a replay check rather than a behavioral sample.

The three hand-authored release trajectories are compatible with a recurrent
search-and-track explanation, but no tracker experiment ran. The fixed
observation/action alphabet does not itself bound temporal strategy complexity
when controller, ecology, or persistent spatial state can grow. The current
implementation is blocked; the broader procedural-ecology family remains an
untested hypothesis requiring a dual-population regret experiment with
tracker, transfer, knockout, and increasing-horizon controls. The full audit
is `research/archive/reports/procedural-ecology-adversarial-audit.md`.

## Exact open gap

No implemented run passes the operational tail contract. Moreover,
`research/archive/reports/open-endedness-finiteness-audit.md` proves that a fixed executable,
config, seed, RAM, and persistent-storage envelope is a finite deterministic
state machine: it must halt/fail or eventually repeat, so literal infinite
pairwise novelty is impossible on the actual executable. An unbounded-heap
idealization removes that formal bound but does not stop fixed task languages
from collapsing into universal-interpreter, memory, time, or size ladders.

The exact remaining gap is an implemented endogenous problem/affordance
generator coupled to protected solver learning and historical retention.
Persistent, energy-funded, recursively composable public niche construction is
one concrete candidate: organisms create serialized artifacts with generic
public ports that survive into later evolutionary episodes and can themselves
become inputs to further construction. Admission must require a new canonical
causal organism-artifact or lineage-lineage relation, downstream adaptive use,
artifact/creator/consumer knockouts, transfer under alpha/translation/rebuild,
closed energy/material accounting, and rejection of inert growth or a universal
constructor. It is not proven to be the unique or smallest route.

Even that mechanism can only support an operational claim: causally new,
retained, transferable ecological relations continue through geometrically
increasing tested horizons across seeds. The standing literal unbounded claim
requires either changing that success definition or moving to an explicitly
unbounded mathematical resource model. Protected public training and the
current PPEC proof-cache encoding are blocked. The current procedural Stage-0
implementation is also blocked: its 19 scoped gates do not repair evaluator
funding, public resource provenance, or fail-closed boundary semantics, and it
contains no evolved ecology or organism learning. The broader
procedural-ecology family is untested, not disproved. A dual-population regret
experiment must reject generic tracking and demonstrate continued observable,
learnable behavior growth through increasing horizons; hidden ecology-program
growth does not count. Reproduction-only is closed by
`research/archive/reports/endogenous-replication-closure.md`: the historical engine already had an
exact parent-to-child energy transfer, brain-selected births, birth mutation,
and multi-generation lineages, yet its multi-seed information games converged
or regressed. Its old periodic founder injection prevents a clean endogenous
energy claim, so reproduction reopens only coupled to the active public
artifact/problem generator rather than as a standalone lever.
