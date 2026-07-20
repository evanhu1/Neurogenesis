# Hidden-string v7: boundary-contrastive signed-evidence fast memory

Status: frozen-champion and three-seed fixed-length discovery gates passed;
variable-length transfer and fresh confirmation remain unrun

## Question

Can one target-independent online associative-memory rule remove both v6
failure modes--shared-readout interference and the autonomous-clock shortcut--
without receiving a target symbol or position index?

## Algorithm

The inherited recurrent controller remains a slow context generator. At each
symbol step it is evaluated twice from matched dynamics:

- the live arm receives the symbolic `end` boundary at the episode start;
- the null arm receives the all-zero symbolic stream.

Let their hidden vectors be `h+` and `h0`. The candidate context key is the
boundary contrast `d = h+ - h0`, scaled as
`q = d / max(norm(d), 0.05)`. The fixed 0.05 floor prevents an arbitrarily
small boundary watermark from being amplified into a unit key. No target,
target ID, position, panel, rollout seed, reward, or sampled action
participates in key construction.

A separate runtime matrix `F[hidden, action]` begins at zero for every target
and persists across its attempts. It contributes `F^T q` to the inherited
action logits. After an action is sampled and scalar reward arrives, only that
action column is updated:

```text
F[:, sampled] <- clip(F[:, sampled]
                       + alpha * sign(reward) * q,
                       -fast_weight_bound,
                       +fast_weight_bound)
```

The brain's inherited learning-rate gene supplies `alpha`. Learning uses no RNG
and applies updates in hidden-index order. Target start clears `F`. Attempt
start clears matched live/null dynamics while preserving `F`. Probes disable
writes and operate on reset dynamics while reading the learned `F`.

This differs materially from v6: it does not mutate inherited hidden-to-action
weights and its writes are stored in a separate lifetime matrix. Subtracting
`h0` removes the bias-driven autonomous trajectory; boundary-off must therefore
produce a zero key and no fast-memory capability. The causal interference
measure is the actual logit effect `abs(q_read^T delta F[:, sampled])` after
clipping, not merely a cosine between nominal keys.

## Why this might be general--and why it might not be

The favorable interpretation is a generic online key-value memory: a recurrent
controller produces context, boundary contrast isolates stimulus-caused state,
and signed scalar feedback writes values. The same primitive could be frozen
and tested on delayed association, next-token prediction, and multistep
control.

An earlier QR variant was discarded before implementation because an ordered
Gram--Schmidt basis would allocate one evaluator-owned address per decision and
act like a supplied program counter. Direct contrast leaves address generation
inside the recurrent substrate. Hidden-string success is still insufficient:
variable lengths, boundary lead-ins, event-order permutations, delayed recall,
and later task transfer are mandatory before calling the mechanism reusable
memory.

The current contract is event-driven: one recurrent transition corresponds to
one emitted symbol. A pre-boundary zero-input lead-in is valid, but inserting
uncommitted recurrent ticks between symbols changes the address. Environments
with such ticks require an explicit symbolic commit/write-enable event; v7 does
not claim arbitrary wall-clock invariance.

## Pre-implementation replay

An independent exact replay used the persisted terminal seed-509 genome and all
64 sealed treatment traces. The inherited policy achieved 48.4% exact on those
traces. Direct boundary-contrast keys had raw norms
`[1.085, 1.449, 1.235, 0.0486]` and pairwise cosines zero at the reported
precision. With the 0.05 floor and update scale 0.5 or 1.0, selected-action
signed evidence reached 100% exact and 100% per-position accuracy. Boundary-off
made the contrast zero and remained at 0% exact / 12.5% characters.

The floor is decisive anti-gaming machinery. Without it, reducing the boundary
amplitude to 0.001 still left 96.88% exact because normalization amplified a
vanishing watermark. With the floor, half amplitude fell to 65.63%--81.25%,
quarter amplitude to 15.63%, and near-zero amplitude to zero exact. The route
therefore depends on a material boundary-caused state, not an epsilon-sized
tag.

This replay selected the mechanism family. It is contaminated diagnostic
evidence, not an implementation result or confirmation.

## Cheap go/no-go diagnostic

First evaluate the persisted seed-509 terminal champion on 128 balanced custom
targets under two fresh rollout seeds for 32 attempts. Do not evolve.

The implementation advances only if:

1. primary exact is at least 90%;
2. boundary-off, plasticity-off, reset-fast-each-attempt, symbol-permuted reward,
   and position-permuted reward are each at most 0.5% exact;
3. actual off-diagonal cross-key fast-logit effects are reported after clipping
   and remain small enough to explain the competence result;
4. no key is non-finite; raw and scaled key norms are persisted; half- and
   quarter-boundary amplitude lesions are reported;
5. one-worker and four-worker outputs are semantically identical.

Also run normalized raw-live-state keys. It is a diagnostic comparator, not a
causal ceiling; the record must report whether matched contrast contributes.

Kill the route before evolution if the primary misses 90%, any mandatory
control exceeds 0.5%, or boundary lead-in/event-order probes show that the
controller is only exploiting an external evaluation counter.

## Promotion gate

Passing the cheap diagnostic only permits a three-seed evolutionary discovery
run under variable target lengths 2--6 and target-independent timing gaps.
Before any fresh confirmation, move the fast matrix into serialized
brain runtime state; add evolvable learning scale, fast-weight bound, retention,
and decay genes; synchronize Rust/TypeScript wire types and views; and freeze
the same rule for every downstream task.

Hidden-string confirmation retains the v6 five-seed gate: at least four of five
fresh seeds and the median at 90% sealed exact, none below 75%, all mandatory
controls low, deterministic replay, complete traces, and a fixed work budget.
Passing it establishes reward-only binding only. Delayed memory, at least 90%
held-out toy-English next-token accuracy, multistep pointer chasing, continual
retention, and sustained-tail open-endedness remain separate gates.

## Fixed-length evolutionary discovery screen

Before implementing the variable-length task, run evolutionary seeds `947`,
`953`, and `967` on diagnostic contract seed `7777777` with population 32, 25
generations, and eight workers. This panel and its rollouts are discovery data
and may not be reused for confirmation.

Advance only if:

- the median sealed exact rate is at least 90% and no seed is below 75%;
- plasticity-off, both reward permutations, reset-fast-each-attempt,
  boundary-off, and per-symbol dynamics reset are each at most 0.5% exact;
- every trace audit passes, all raw/effective key norms are finite, and actual
  off-diagonal fast-logit effects are reported;
- worker determinism remains established by the preceding matched smoke;
- each seed uses at most 800 population evaluations.

Failure kills fixed four-symbol v7 at this budget. Passing only unlocks the
variable-length, boundary-lead-in, and delayed-association implementation; it
is not confirmation.

## Diagnostic outcome

The Rust implementation reached 100% exact on 256 fresh custom-panel cases.
Plasticity-off, both reward permutations, boundary-off, and per-symbol dynamics
reset reached 0% exact; reset-fast-each-attempt reached 0.391%. Raw-live-state
keys reached only 52.73%. Half and quarter boundary amplitudes fell to 56.25%
and 12.50%, verifying that the 0.05 floor blocks vanishing-watermark gain. See
the [frozen-champion record](../archive/experiments/2026-07-18-hidden-string-v7-frozen-champion-diagnostic.md).

The fixed-length evolutionary discovery also passed: seeds 947/953/967 reached
100.00%, 97.17%, and 97.75% sealed exact inside 800 population evaluations,
with every mandatory control at or below 0.293%. See the
[discovery record](../archive/experiments/2026-07-18-hidden-string-v7-fixed-length-discovery.md).
# Architecture-audit status

Rejected on 2026-07-18: the evaluator-owned key transform and side matrix fail
the canonical-brain bar.
