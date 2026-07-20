# Basic next-token supervised-learning milestone

Date: 2026-07-19  
Status: completed post-hoc discovery milestone; independent confirmation and
held-out-language generalization remain open.  
Source base: `e4a50989b2f925325f25cab4e2a4a490e073d1bf` plus the uncommitted clean
task-ecology cutover described below.

## Result

The common symbolic brain, learner, and asexual task ecology learned the entire
canonical English pangram after four supervised passes and emitted the exact
44-token target in a reset, plasticity-frozen greedy probe.

At evolutionary seed 101:

| Population | Generations | Selected generation | Frozen correct | Accuracy | Exact sequence | Plasticity off |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 100 | 74 | 37/44 | 84.091% | no | 1/44 |
| 256 | 100 | 99 | 38/44 | 86.364% | no | 0/44 |
| 1,024 | 100 | 99 | 36/44 | 81.818% | no | 1/44 |
| 256 | 500 | 349 | 42/44 | 95.455% | no | 1/44 |
| 1,024 | 500 | 374 | 44/44 | 100.000% | yes | 2/44 |

Population width was not monotonic at the shallow 100-generation budget, but
both population 256 and 1,024 crossed the 90% gate with greater generation
depth. The exact population-1,024 winner remained at 44/44 through generation
499 after first appearing in the periodic audit at generation 374.

This experiment was produced by iterative diagnosis rather than preregistered
confirmation. The numbers establish a milestone and choose a baseline; they do
not supply an unbiased estimate of replication probability.

## Task and evaluation contract

The canonical snippet is:

```text
the quick brown fox jumps over the lazy dog
```

The alphabet is `a` through `z`, `space`, and `end`. The snippet contains 43
characters. Appending the terminal `end` target creates 44 next-token examples.
At position zero the observed input is the boundary symbol `end`; thereafter
the environment supplies the preceding ground-truth character. The recurrent
brain therefore receives the prefix sequentially rather than receiving a
task-authored position or elapsed-time feature.

One lifetime consists of four complete teacher-forced passes over all 44
targets. The actual next character is supplied only as the categorical
supervised label after the prediction. At each pass boundary, recurrent and
action dynamics reset while learned synaptic weights persist. After pass four,
dynamics reset again and plasticity is disabled. A greedy 44-step probe then
allocates one reproductive ticket per correct next-token prediction. Exact
sequence success requires all 44 probe positions.

Training predictions allocate no tickets. Development and sealed audits use
the same fixed snippet and therefore measure frozen training-snippet mastery,
not held-out textual generalization.

## Learner

The generic target-prediction rule applies the exact softmax cross-entropy
output error:

```text
delta_w = eta * plasticity_coefficient * presynaptic_activation
          * (one_hot(target) - probability)
```

Each requested update is bounded by the genome's maximum per-tick delta. The
plasticity coefficient remains evolvable per synapse. Fast-weight retention
controls displacement decay toward the inherited weight.

The critical general intervention was to make the plastic categorical readout
complete: every task-enabled sensory feature and every hidden feature has an
evolvable-plastic connection to every legal action. The target-prediction rule
updates both sensory-to-action and hidden-to-action synapses. Previously it
updated only hidden-to-action edges while guaranteeing a readout from only one
hidden neuron. Evolution therefore had to discover a large input-copy and
readout structure before it could express an ordinary supervised learner.

This head supplies no target identity, position, prefix code, or pangram
transition. It is the generic dense trainable output layer normally required
to learn a categorical mapping from a representation. Evolution still owns
hidden representation, recurrence, inherited weights, per-edge plasticity,
learning rate, retention, and update cap.

## Search configuration

The successful runs used:

- evolutionary seed `101`;
- canonical founder capacity: one hidden neuron and 24 initial synapses;
- one training, development, and sealed instance with one rollout each;
- greedy action selection;
- target-prediction-error learning;
- no NLMS normalization;
- dynamics reset at semantic trial/pass boundaries;
- one exact elite and tournament size four;
- independent random founder topology and parameters for every population
  member;
- task-interface filtering of inactive edges and learning genes;
- functional-class-balanced add-connection mutation;
- approximately behavior-preserving node splitting without an automatic
  random recurrent self-loop;
- multiscale weight mutation, favoring local 1--2-coordinate proposals while
  retaining progressively rarer broader proposals;
- three evaluation workers in the recorded matched runs.

The default mutation probabilities were weight perturbation `0.75`, weight
replacement conditional probability `0.05`, bias mutation `0.20`, time
constant mutation `0.10`, learning-rate-family mutation `0.20`, plasticity
coefficient mutation `0.15`, add/delete connection `0.09/0.09`, and add/delete
node `0.05/0.05`. Weight and bias perturbation standard deviations were `0.10`.

The exact winner evolved:

- initial learning rate `0.76908946`;
- fast-weight retention `1.0`;
- maximum weight delta per tick `0.32869267`;
- action-temperature scale `0.49382883` (irrelevant to greedy choice);
- mean applied absolute update `0.00865184`;
- 1,202 clipped updates among 88,704 reported edge updates.

## Exact winner structure and causal controls

The selected population-1,024 genome has 17 hidden neurons and 1,310 enabled
connections:

| Connection class | Count |
|---|---:|
| sensory to action, current tick | 784 |
| hidden to action, current tick | 476 |
| sensory to hidden, current tick | 21 |
| hidden to hidden, current tick | 8 |
| hidden to hidden, previous tick | 13 |
| previous action to hidden | 8 |

The 1,260 readout edges form the generic plastic head. The remaining 50 edges
form the inherited recurrent feature generator. Readout plasticity
coefficients span `0.00218` to `2.0`, with mean `1.00596`.

Sealed causal lesions on the exact winner were:

| Condition | Frozen correct | Accuracy | Interpretation |
|---|---:|---:|---|
| Primary | 44/44 | 100.000% | Exact learned sequence |
| Plasticity disabled | 2/44 | 4.545% | Rules out inherited output memorization as the solution |
| Previous-action feedback disabled | 43/44 | 97.727% | Efference copy contributes one disambiguation |
| Prediction-error feedback disabled | 41/44 | 93.182% | Error-modulated hidden dynamics contribute three positions |

The primary online training accuracy was only 34.091% because it averages
predictions made while weights are still being acquired across all four
passes. The final frozen probe is the endpoint competence measure. Its mean
target probability was `0.578394`; all 44 target logits nevertheless won the
greedy argmax.

## Pass-count diagnosis

A population-64, 50-generation matched sweep established four passes as the
efficient knee after the complete readout correction:

| Learning passes | Frozen correct | Accuracy | Plasticity off |
|---:|---:|---:|---:|
| 1 | 18/44 | 40.909% | 1/44 |
| 4 | 34/44 | 77.273% | 0/44 |
| 16 | 34/44 | 77.273% | 0/44 |
| 32 | 34/44 | 77.273% | 1/44 |

The selected genomes drove fast-weight retention to exactly `1.0`. More than
four passes multiplied evaluation work without improving the frozen result, so
four became the canonical default.

## Evolutionary progression

Periodic audits of the exact population-1,024 run show cumulative improvement:

| Generation | Frozen correct | Accuracy | Hidden nodes | Enabled connections |
|---:|---:|---:|---:|---:|
| 24 | 32/44 | 72.727% | 2 | 844 |
| 49 | 34/44 | 77.273% | 2 | 846 |
| 74 | 35/44 | 79.545% | 2 | 848 |
| 99 | 36/44 | 81.818% | 3 | 878 |
| 149 | 38/44 | 86.364% | 7 | 1,001 |
| 174 | 41/44 | 93.182% | 12 | 1,150 |
| 199 | 43/44 | 97.727% | 13 | 1,181 |
| 374 | 44/44 | 100.000% | 17 | 1,310 |

This is consistent with evolution first improving the direct supervised
mapping, then adding recurrent contextual features needed to distinguish
repeated-character contexts.

## Reproduction commands

```bash
cargo build -p cli --release

./target/release/cli ecology next-token \
  --seed 101 --population 256 --generations 500 \
  --learning-passes 4 --workers 3 \
  --out-dir artifacts/research/runs/diagnostics/basic-next-token-full-readout-p256-g500

./target/release/cli ecology next-token \
  --seed 101 --population 1024 --generations 500 \
  --learning-passes 4 --workers 3 \
  --out-dir artifacts/research/runs/diagnostics/basic-next-token-full-readout-p1024-g500
```

Primary result artifacts:

- `artifacts/research/runs/diagnostics/basic-next-token-full-readout-p256-g500/task-ecology-basic_next_token_prediction-1784503984781-56364.json.zst`
- `artifacts/research/runs/diagnostics/basic-next-token-full-readout-p1024-g500/task-ecology-basic_next_token_prediction-1784504076916-56372.json.zst`

The artifacts persist the complete task, agent, ecology, search, founder,
generation, selected-genome, lesion, work, and timing contracts.

## Claim boundary and next confirmation

Established:

- the current brain can express an exact next-token learner for a 44-target
  full-alphabet sequence;
- the common supervised plasticity rule can acquire the sequence in four
  passes;
- competence depends overwhelmingly on lifetime plasticity;
- population 256 and 1,024 exceed the 90% gate with 500 generations;
- recurrent feature growth explains performance beyond the one-character
  transition ceiling.

Not established:

- replication across evolutionary seeds;
- monotonic population scaling at fixed shallow depth;
- generalization to unseen snippets, longer text, or natural-language
  distributions;
- whether the inherited recurrent feature generator is pangram-specific;
- comparable efficiency to gradient descent.

The next honest experiment should freeze the evolved learner machinery and
measure acquisition of unseen full-alphabet snippets, followed by a
preregistered multi-seed replication of the fixed training contract.
