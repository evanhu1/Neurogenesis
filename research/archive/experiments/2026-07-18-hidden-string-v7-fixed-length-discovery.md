# Hidden-string v7 fixed-length evolutionary discovery

Status: completed positive discovery result; not confirmation

## Contract

Three fresh evolutionary seeds (`947`, `953`, `967`) ran independently on
diagnostic contract seed `7777777` with population 32, 25 generations, eight
workers, and the v7 boundary-contrastive signed-evidence fast readout. Each run
used 800 population evaluations and one terminal development/sealed evaluation.

The fixed-length screen was preregistered in the v7 proposal before execution.
It required median sealed exact at least 90%, no seed below 75%, every mandatory
control at or below 0.5%, passing traces, finite key telemetry, and no more than
800 population evaluations per seed.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v7-fixed-discovery/`

All three manifests captured source fingerprint
`c1c867dcda65ed0c524a856288af3849dd3092d18f523f300aff0246358c5118`
and executable hash
`7334438087026d9ffd4d950e5d2b4efee02fe9b64381248394af99ba341ec792`.

## Result

| Seed | Train exact | Development exact | Sealed exact | Sealed characters | Wall time |
|---:|---:|---:|---:|---:|---:|
| 947 | 100.00% | 100.00% | 100.00% | 100.00% | 17.45 s |
| 953 | 96.09% | 96.88% | 97.17% | 99.29% | 17.97 s |
| 967 | 98.29% | 99.61% | 97.75% | 99.44% | 17.04 s |

The median sealed exact rate was 97.75%; every seed exceeded 97%. Training,
development, and sealed cohorts agreed, so this was not a panel-specific gain.
All runs completed at generation 25 with 800 population evaluations and 802
total genome evaluations. Total synapse-operation counts were 49.62--50.81
billion.

| Maximum sealed exact across seeds | Rate |
|---|---:|
| Plasticity off | 0.000% |
| Symbol-permuted reward | 0.000% |
| Position-permuted reward | 0.000% |
| Reset fast weights each attempt | 0.146% |
| Boundary pulse off | 0.098% |
| Dynamics reset each symbol | 0.293% |

Every primary and control trace audit passed. Each stored 32 targets under two
rollouts per condition. The effects therefore require signed reward, persistent
fast state, the correct position/action relation, a material boundary-caused
state, and continuous event-to-event recurrent dynamics.

The selected genomes used 13--14 hidden nodes, 81 enabled connections, and
learning rates 0.660--0.853. Raw contrast norms stayed above 0.091, 0.310, and
0.435 for the three sealed winners, safely above the fixed 0.05 anti-watermark
floor.

The fast memories were not perfectly orthogonal in every evolved winner.
Seed 947 had zero measured off-diagonal effects. Seeds 953 and 967 had mean
off-diagonal actual logit effects up to 0.173 and 0.135 respectively, while
retaining 97%--98% exact competence. Component-update clip fractions remained
high at 44.7%--60.2%. V7 removes v6's catastrophic interference but has not
shown a well-conditioned general memory across longer sequences.

Result hashes:

- seed 947:
  `42b79de8b01f59a66b9058343c476e48a2c099d41104b680d72c9f6012868663`;
- seed 953:
  `71c6c3c5098dd9f3fd28aac991f595520f5504cd137ce19b134fcc151c4422ae`;
- seed 967:
  `1c212c52226f1d3381484656794ef6171c2bb7c4187f1b52f396dc7306cba3d5`.

## Decision

The fixed-length discovery gate passes. Advance the same learning rule to
variable pre-boundary timing, variable sequence lengths, serialized runtime
ownership, and a delayed-association task. Do not run fresh confirmation yet:
the current four-event delay-line contract remains too close to a fixed program
counter, and no downstream task has used the mechanism unchanged.
# Architecture-audit note

Invalidated as brain competence on 2026-07-18. The matched-null key transform
and separate lifetime matrix supplied the representation and memory operation.
