# Delayed association v2 serialized latched memory

Status: completed positive bounded-memory result

## Question

Can the same evolved fast-memory substrate retain a symbol-dependent binding
through long, irrelevant recurrent activity and retrieve it only at a later
query?

## Mechanism and contract

One cue evaluation captures a normalized hidden-state key into serialized
`BrainState`. Distractors run through ordinary recurrent evaluation with key
capture disabled, so they cannot overwrite the latch. One later `end` query
reads the already latched key in a single forward evaluation; reward is supplied
only after that query. No target, target index, null-brain trajectory, or
task-owned answer is injected into the controller.

Seeds `1423`, `1429`, and `1427` used population 64, 100 generations, four
workers, 16 study rounds, study delays 1/3/5, sealed recall delays 2/4/6/8,
128 sealed cases, and two sealed rollout seeds. Training, development, and
sealed case panels were disjoint.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-delayed-association-v2-latched-discovery/`

## Result

| Seed | Query accuracy | Whole association set exact | Robust lifetime exact |
|---:|---:|---:|---:|
| 1423 | 99.768% | 99.072% | 97.656% |
| 1429 | 99.292% | 97.168% | 93.359% |
| 1427 | 97.156% | 88.623% | 86.328% |

Accuracy was flat across delays 2, 4, 6, and 8. Plasticity-off and fast-reset
controls were 12.34%--12.44% query accuracy with zero association sets exact.
Resetting memory after every study trial reached only 15.47%--16.20%, again
with zero sets exact. Deleting the cue produced 20.14%--20.89% query accuracy,
but zero sets exact; this is a value-frequency baseline, not eight-way chance.
Permuted reward and swapped-key controls followed the wrong mapping at
96.37%--99.79% while matching the intended mapping at no more than 1.09%,
showing that the stored association, rather than an inherited answer, controls
retrieval. Resetting recurrent dynamics after cue capture preserved
97.42%--99.68% query accuracy, localizing persistence to the serialized key
rather than recurrent working state.

## Horizon falsifier

Frozen winners were reevaluated without evolution at delays
8/16/32/64/128/256:

`artifacts/research/runs/diagnostics/2026-07-18-delayed-association-v2-horizon-256/`

At delay 256, query accuracy / whole-set exact was 99.756% / 99.023% for seed
1423, 99.414% / 97.656% for seed 1429, and 96.924% / 87.695% for seed 1427.
Plasticity-off and reset-fast controls remained near 12.5% with zero sets
exact. Performance therefore does not depend on a recurrent trajectory
surviving for the delay duration.

Result SHA-256 hashes for seeds 1423, 1429, and 1427 respectively are
`69702c4ea6794835f56b1574767b9491aa31e06a5de9e4b3d13f8d93d1b99a65`,
`3316228cbc28bb551bc4da2cb44edcbcac47eeb2ec6d0edf5be53fa35cd0798e`,
and `e98bec409fc1bc5be3b68f6973e986a3ce6ca8c68046860d32d4e7dcb08c4fee`.

## Decision

The memory goal passes in the precise form demonstrated: a learned binding is
stored in an explicit organism-owned key/readout register and survives at least
256 distractor steps. This is not evidence that evolution discovered generic
autonomous memory gating, an unbounded store, or a recurrent attractor memory.
# Architecture-audit note

Invalidated and removed on 2026-07-18. The evaluator preserved a cue-derived
key outside recurrent dynamics, so distractors never tested delayed memory.
The measurements below describe the removed engineered latch only.
