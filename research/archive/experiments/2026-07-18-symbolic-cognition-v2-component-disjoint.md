# Symbolic cognition v2 component-disjoint lexical and pointer tasks

Status: completed positive bounded-task result; narrow language and reasoning scope

## Question

Can one evolved genome use the same scalar-reward fast memory for a toy lexical
next-token task and for closed-loop multistep symbolic control on mappings not
seen in evolution?

## Contract

Each case alpha-renames three fixed topic-to-verb pairs (`cats run`,
`dogs sleep`, `birds fly`) into the eight-symbol alphabet. Twenty-four study
rounds supply scalar reward for the topic-to-verb binding. The balanced,
shuffled BOS-to-topic transition is uninformative, unrewarded, and unscored.

A separate component learns a fresh deranged eight-symbol successor mapping
over 32 reward study rounds. At probe time, every emitted output becomes the
next input. No step is teacher-forced. Depths 2 through 7 are selected jointly
with the lexical component by the geometric mean of lexical correct-action
probability and mean full-path probability.

V1 reused the same pointer mappings between training and sealed panels and is
invalid as component-generalization evidence. V2 generates the lexical and
pointer panels independently. Each component has 64 training, 32 development,
and 128 sealed cases, and all six within-component train/development/sealed
overlaps are zero.

Seeds `1523`, `1511`, and `1531` used population 64, 100 generations, four
workers, and two sealed rollouts.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-symbolic-cognition-v2-component-disjoint/`

## Result

| Seed | Lexical top-1 | Lexical probability | Pointer full-path exact | Depth-7 exact | Hop accuracy |
|---:|---:|---:|---:|---:|---:|
| 1523 | 99.740% | 76.676% | 99.585% | 99.414% | 99.707% |
| 1511 | 99.349% | 74.305% | 99.862% | 99.854% | 99.902% |
| 1531 | 99.740% | 77.745% | 100.000% | 100.000% | 100.000% |

Plasticity-off/fast-reset lexical accuracy was 13.28%--14.32%, and pointer
full-path exact was 0.146%--0.960%. Reward permutation reduced lexical accuracy
to 0.13%--0.26% and pointer exact to zero. Blank-cue lexical accuracy was
24.22%--30.21%, a corpus value-frequency baseline rather than chance; pointer
exact remained at or below 0.024%.

Result SHA-256 hashes for seeds 1523, 1511, and 1531 respectively are
`e316ec6ecefa2d348ef3a68eb846acc033565a1793c1767799603d82d6a62622`,
`7c41c564a8ebb91dc294658493e99b0ae463b440a9acb9a7078f7688bbd37902`,
and `110e3935f2c40946bbad3d0c1ba2b41ab89d8cf9735ba4d273846ecdb739a415`.

## Horizon falsifier

Frozen winners were reevaluated without evolution at depths
8/16/32/64/128/256:

`artifacts/research/runs/diagnostics/2026-07-18-symbolic-cognition-v2-horizon-256/`

For every tested depth, full-path exact stayed fixed at 99.365%, 99.854%, and
100% for seeds 1523, 1511, and 1531. That apparent tail is finite cycle closure:
the controller repeatedly applies the same learned eight-entry permutation.
The corresponding full-path probability at depth 256 fell to
`1.24e-22`, `7.55e-22`, and `7.67e-22`.

## Decision and claim boundary

The task-specific gates pass: all three genomes exceed 90% on the scored toy
lexical transition and on a closed-loop multistep pointer task with disjoint
mappings. The honest language claim is one predictable lexical transition in a
three-sentence corpus, not general English, semantics, syntax, BOS prediction,
or calibrated language modeling. The honest problem-solving claim is repeated
application of an online-learned finite successor rule, not planning,
compositional reasoning, growing task depth, or open-ended cognition.
# Architecture-audit note

Invalidated and removed on 2026-07-18. The task installed the one-hot encoder
and complete associative table required by both component tasks. The results
below do not establish evolved symbolic cognition.
