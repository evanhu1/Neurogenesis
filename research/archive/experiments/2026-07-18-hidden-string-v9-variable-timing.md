# Hidden-string v9 event-gated variable-timing discovery

Status: completed positive bounded-task result; not open-endedness evidence

## Question

Can one inherited symbolic controller learn an unseen four-symbol string from
signed scalar reward alone when target and position are never sensory inputs,
and when variable silent timing prevents a wall-clock program counter?

## Mechanism and contract

V9 adds three evolvable fast-memory genes (learning rate, retention, and bound),
per-edge plasticity, per-hidden-node fast-key gain, and a per-hidden-node symbol
event gate. `BrainState` owns and serializes both the fast readout matrix and a
latched key. Every scored event supplies the same target-independent `end`
sensor. Zero through three deterministic all-zero ticks precede each event;
event gates can freeze recurrent state during those ticks. A sampled action is
followed by balanced scalar reward (`+1` or `-1/7`), which writes signed evidence
into plastic hidden-to-action edges. The target and output position are never
passed to the brain or to action sampling.

Three independent evolutionary seeds (`1213`, `1217`, `1223`) used population
32, 25 generations, four evaluation workers, 1,024 training targets under two
rollouts, 256 development targets, and 1,024 sealed targets under two disjoint
rollouts. Each target lifetime contained 32 learning attempts. Training,
development, sealed, and control schedules used disjoint deterministic rollout
seeds.

Artifacts:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v9-variable-timing-discovery/`

All three manifests captured source fingerprint
`a0b2fb9f62f7857019866a05b0012b1d00f00ddd99d1c4792a1fa7034c3b8e95`
and executable hash
`ceefbb64aac5ac3812cb9966d9fe94657e22ca032cf2352c72629d7710caa89d`.

## Result

| Seed | Sealed exact | Character accuracy | Commit-off character | Commit-off worst position | Causal gate |
|---:|---:|---:|---:|---:|---:|
| 1213 | 98.975% | 99.744% | 12.500% | 12.500% | 1.0 |
| 1217 | 97.217% | 99.304% | 12.500% | 12.500% | 1.0 |
| 1223 | 95.459% | 98.865% | 12.817% | 13.037% | 1.0 |

The mean and median sealed exact rate were both 97.217%; the worst seed reached
95.459%. Commit-off exact rate was zero for every seed. Plasticity-off,
symbol-permuted reward, position-permuted reward, and dynamics-reset controls
were at most 0.098% exact. Resetting fast weights every attempt was the largest
remaining control at 0.488% exact. All are below the 0.5% ceiling.

V8 is not evidence for this result. Its exact-only causal gate allowed a
commit-off controller to retain 38.51% character accuracy and 75.05% in one
position. V9 always evaluates a commit-off control and gates on exact,
aggregate-character, and worst-position ceilings. The three V9 controls above
are at eight-way chance at every position.

Raw result SHA-256 hashes:

- seed 1213: `77e1f2d82a2f0f6c298220138fef3a30a468b53482d45791d247bb2beb9cb173`;
- seed 1217: `383856496619184e2e2a3431fe61de2e11d5e29d305b0a67ebbd2d8ca3cdf405`;
- seed 1223: `6b3107bf8767f688c800731485f5bfc6207c81254e78b961185086fa95cdf028`.

## Determinism

Seed 1213 was rerun with eight workers under the same semantic contract:

`artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v9-worker-determinism/`

The four- and eight-worker artifacts have identical canonical hashes for final
population (`d3d351f690370c33e21da27936b115283dfa57c3462a0d46ca8f172ec1ed0192`),
development evaluation (`c412bbdea7696b501bfe38da607cdd2550bed7d1409ac2f743c8debea36bccff`),
sealed evaluation (`ed01050e7223cd0ba21c69c2a33c02291263b3c8a1f1236b617dfbe0c8810350`),
threshold events (`271ef7e546f709334507a95f63d5322a35954e7cd5a7853244386465ebe1ae64`),
deterministic work (`46fbc15c1dc1baff284673a78a4f7151a324a1a5485a20560cf004ab47ef6782`),
and generation summaries after removing wall time
(`e161e37183704723b024a1a87eb9e02556d5fd3ec411f2d891ac48fbd3126b1c`).

## Decision

The reward-only hidden-string goal passes as a multi-seed bounded benchmark.
The causal mechanism requires event input, reward, plastic edges, and persistent
organism-owned fast state; variable silent timing no longer exposes a wall-clock
position counter. This result does not show an unbounded sequence length,
general continual learning, or open-ended novelty.
# Architecture-audit note

Invalidated as brain competence on 2026-07-18. The evaluator installed a fixed
event-driven shift register and used a separate dense fast readout. The results
below remain historical evidence about that engineered mechanism only.
