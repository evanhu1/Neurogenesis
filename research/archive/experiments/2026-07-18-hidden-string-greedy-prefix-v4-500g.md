# Hidden-string greedy-prefix v4, 500-generation replication

Status: completed; robustness rejected

## Question

Does ordered longest-prefix fitness turn the encouraging 100-generation v4
trajectory into reliable exact four-symbol adaptation with more search?

## Method

Seeds 211, 307, and 401 each ran population 64 for 500 generations with eight
evaluation workers. Training used 1,024 targets and two rollouts; development
used 256 targets; the once-per-run sealed cohort used 1,024 targets and two
rollouts. Selection rewarded the longest greedy correct prefix from `0/4` to
`4/4`; hard exact-string rate remained the competence gate.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-greedy-prefix-v4-500g-run/`

## Results

| Seed | Wall time | Sealed character accuracy | Sealed exact strings |
|---:|---:|---:|---:|
| 211 | 260.1 s | 62.04% | 6.59% |
| 307 | 290.4 s | 55.82% | 4.10% |
| 401 | 218.0 s | 56.98% | 3.12% |

Every run used 32,000 population genome evaluations plus 21 scheduled assays.
No seed crossed even the 20% hard-exact threshold. Shuffled-reward exact rate
was zero for all three seeds; reset-weights exact rate was zero, 0.049%, and
0.049%.

## Decision

Reject greedy-prefix selection as a robust route. Ten times the diagnostic
budget improved character-level and partial-prefix behavior but did not yield
exact sequence binding. More generations would be scale without a material new
mechanism. V5 therefore tested evolvable readout plasticity, and v6 replaces
the collapsed autonomous clock with a boundary-driven recurrent temporal
basis.
